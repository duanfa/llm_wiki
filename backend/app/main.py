from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .extraction import ExtractionError, TEXT_EXTENSIONS, extract_text
from .ingest import ingest_sources
from .vector import vector_search

APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"
MAX_FILE_CONTENT_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_FILES = 2000
HARD_MAX_FILES = 10000

DATA_ROOT = Path(os.getenv("LLM_WIKI_DATA_ROOT", "~/llm-wiki-data")).expanduser().resolve()
ALLOW_ABSOLUTE_PATHS = os.getenv("LLM_WIKI_ALLOW_ABSOLUTE_PATHS", "0") == "1"
API_TOKEN = os.getenv("LLM_WIKI_API_TOKEN", "").strip()
WEB_DIST = Path(os.getenv("LLM_WIKI_WEB_DIST", "")).expanduser() if os.getenv("LLM_WIKI_WEB_DIST") else None
REGISTRY_PATH = DATA_ROOT / ".server" / "projects.json"
MODEL_CONFIG_PATH = Path(
    os.getenv(
        "LLM_WIKI_MODEL_CONFIG",
        Path(__file__).resolve().parents[2] / "config" / "model_config.yaml",
    )
).expanduser().resolve()

app = FastAPI(title="LLM Wiki Web API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("LLM_WIKI_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FileNode(BaseModel):
    name: str
    path: str
    is_dir: bool
    children: list["FileNode"] | None = None


class WikiProject(BaseModel):
    id: str
    name: str
    path: str


class ProjectEntry(WikiProject):
    current: bool = False


class PathRequest(BaseModel):
    path: str


class WriteFileRequest(PathRequest):
    contents: str


class CopyFileRequest(BaseModel):
    source: str
    destination: str


class ExtractImagesRequest(BaseModel):
    sourcePath: str
    destDir: str
    relTo: str


class ProjectRequest(BaseModel):
    name: str | None = None
    path: str | None = None
    projectId: str | None = None


class SearchRequest(BaseModel):
    query: str
    topK: int = Field(default=20, ge=1, le=50)
    includeContent: bool = False
    queryEmbedding: list[float] | None = None
    embeddingConfig: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    llmConfig: dict[str, Any] | None = None
    embeddingConfig: dict[str, Any] | None = None


class ModelDefaults(BaseModel):
    llmConfig: dict[str, Any] | None = None
    multimodalConfig: dict[str, Any] | None = None
    embeddingConfig: dict[str, Any] | None = None


class ActivitySnapshotRequest(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    event: str = "snapshot"


ACTIVITY_SUBSCRIBERS: dict[str, set[asyncio.Queue[str]]] = {}


def dump_model(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[attr-defined]
    return model.dict(**kwargs)


def read_model_defaults() -> ModelDefaults:
    if not MODEL_CONFIG_PATH.exists():
        return ModelDefaults()
    try:
        import yaml
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PyYAML is required to read model_config.yaml") from exc
    try:
        raw = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read model config: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=500, detail="model_config.yaml must contain a mapping")
    return ModelDefaults(
        llmConfig=raw.get("llmConfig"),
        multimodalConfig=raw.get("multimodalConfig"),
        embeddingConfig=raw.get("embeddingConfig"),
    )


def require_auth(authorization: str | None = Header(None), x_llm_wiki_token: str | None = Header(None)) -> None:
    if not API_TOKEN:
        return
    bearer = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if bearer == API_TOKEN or x_llm_wiki_token == API_TOKEN:
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def ensure_root() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_path(raw_path: str) -> Path:
    if not raw_path:
        raise HTTPException(status_code=400, detail="path is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    resolved = path.resolve()
    if ALLOW_ABSOLUTE_PATHS:
        return resolved
    try:
        resolved.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path is outside LLM_WIKI_DATA_ROOT") from exc
    return resolved


def project_meta_path(project_path: Path) -> Path:
    return project_path / ".llm-wiki" / "project.json"


def read_registry() -> list[ProjectEntry]:
    ensure_root()
    if not REGISTRY_PATH.exists():
        return []
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return [ProjectEntry(**item) for item in raw if isinstance(item, dict)]
    except Exception:
        return []


def write_registry(projects: list[ProjectEntry]) -> None:
    ensure_root()
    REGISTRY_PATH.write_text(
        json.dumps([dump_model(project) for project in projects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_registered_project_by_path(project_path: Path) -> ProjectEntry | None:
    resolved = project_path.resolve().as_posix()
    for project in read_registry():
        try:
            if Path(project.path).expanduser().resolve().as_posix() == resolved:
                return project
        except Exception:
            if project.path == resolved:
                return project
    return None


def activity_path(project_path: Path) -> Path:
    return project_path / ".llm-wiki" / "activity.json"


def read_activity(project_path: Path) -> list[dict[str, Any]]:
    path = activity_path(project_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    except Exception:
        return []
    return []


def write_activity(project_path: Path, items: list[dict[str, Any]]) -> None:
    path = activity_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items[:100], ensure_ascii=False, indent=2), encoding="utf-8")


def sse_payload(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def publish_activity(project_id: str, payload: dict[str, Any]) -> None:
    message = sse_payload("activity", payload)
    for queue in list(ACTIVITY_SUBSCRIBERS.get(project_id, set())):
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


def validate_project_id(project_id: str) -> str:
    candidate = project_id.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="projectId is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate):
        raise HTTPException(
            status_code=400,
            detail="projectId must use only letters, numbers, dot, underscore, or hyphen, and start with a letter or number",
        )
    if candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise HTTPException(status_code=400, detail="projectId must be a single path segment")
    return candidate


def server_project_path(project_id: str) -> Path:
    safe_id = validate_project_id(project_id)
    path = (DATA_ROOT / safe_id).resolve()
    try:
        path.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="project path is outside LLM_WIKI_DATA_ROOT") from exc
    return path


def ensure_project_identity(project_path: Path, name: str | None = None, project_id: str | None = None) -> WikiProject:
    meta_path = project_meta_path(project_path)
    registered_project = find_registered_project_by_path(project_path)
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_project_id = str(raw.get("id") or (registered_project.id if registered_project else uuid.uuid4()))
            project_name = str(name or raw.get("name") or (registered_project.name if registered_project else project_path.name))
        except Exception:
            existing_project_id = registered_project.id if registered_project else str(uuid.uuid4())
            project_name = name or (registered_project.name if registered_project else project_path.name)
    else:
        existing_project_id = registered_project.id if registered_project else str(uuid.uuid4())
        project_name = name or (registered_project.name if registered_project else project_path.name)

    resolved_project_id = validate_project_id(project_id) if project_id else existing_project_id
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"id": resolved_project_id, "name": project_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return WikiProject(id=resolved_project_id, name=project_name, path=project_path.as_posix())


def upsert_project(project: WikiProject, current: bool = True) -> WikiProject:
    projects = read_registry()
    next_projects: list[ProjectEntry] = []
    inserted = False
    for existing in projects:
        if existing.id == project.id or existing.path == project.path:
            next_projects.append(ProjectEntry(**dump_model(project), current=current))
            inserted = True
        else:
            next_projects.append(ProjectEntry(**dump_model(existing, exclude={"current"}), current=False if current else existing.current))
    if not inserted:
        next_projects.insert(0, ProjectEntry(**dump_model(project), current=current))
    write_registry(next_projects[:50])
    return project


def create_project_structure(project_path: Path) -> None:
    for directory in [
        project_path / "raw" / "sources",
        project_path / "raw" / "assets",
        project_path / "wiki" / "sources",
        project_path / "wiki" / "entities",
        project_path / "wiki" / "concepts",
        project_path / "wiki" / "queries",
        project_path / "wiki" / "synthesis",
        project_path / "wiki" / "comparisons",
        project_path / ".llm-wiki",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    defaults = {
        "purpose.md": "# Purpose\n\nDescribe what this wiki is for.\n",
        "schema.md": "# Schema\n\nDescribe the wiki structure and rules.\n",
        "wiki/index.md": "# Index\n\n",
        "wiki/log.md": "# Log\n\n",
        "wiki/overview.md": "# Overview\n\n",
    }
    for relative, contents in defaults.items():
        target = project_path / relative
        if not target.exists():
            target.write_text(contents, encoding="utf-8")


def list_directory_tree(path: Path) -> list[FileNode]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {path}")
    if not path.is_dir():
        return [FileNode(name=path.name, path=path.as_posix(), is_dir=False)]

    nodes: list[FileNode] = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name in {".DS_Store", "node_modules", "target"}:
            continue
        if child.is_dir():
            nodes.append(
                FileNode(
                    name=child.name,
                    path=child.as_posix(),
                    is_dir=True,
                    children=list_directory_tree(child),
                )
            )
        else:
            nodes.append(FileNode(name=child.name, path=child.as_posix(), is_dir=False))
    return nodes


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {path}")
    ext = path.suffix.lower()
    if path.stat().st_size > MAX_FILE_CONTENT_BYTES and ext in TEXT_EXTENSIONS:
        raise HTTPException(status_code=413, detail="File is too large to read through the Web API")
    if ext not in TEXT_EXTENSIONS:
        return extract_text(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return extract_text(path)


def preprocess_source_file(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {path}")
    try:
        return extract_text(path)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {exc}") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel_path_for(path: Path, rel_to: Path) -> str:
    try:
        return path.relative_to(rel_to).as_posix()
    except ValueError:
        return path.as_posix()


def image_mime_from_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    if ext in {"png", "jpeg", "gif", "webp", "bmp"}:
        return f"image/{ext}"
    if ext in {"tif", "tiff"}:
        return "image/tiff"
    if ext == "svg":
        return "image/svg+xml"
    return "application/octet-stream"


def image_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return image.width, image.height
    except Exception:
        return 0, 0


def save_extracted_image(
    data: bytes,
    dest_dir: Path,
    rel_to: Path,
    file_name: str,
    index: int,
    mime_type: str,
    page: int | None,
    kind: str = "embedded",
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / file_name
    target.write_bytes(data)
    return {
        "index": index,
        "mimeType": mime_type,
        "kind": kind,
        "page": page,
        "width": width,
        "height": height,
        "relPath": rel_path_for(target, rel_to),
        "absPath": target.as_posix(),
        "sha256": sha256_hex(data),
    }


def extract_pdf_images_for_web(source: Path, dest_dir: Path, rel_to: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PDF image extraction requires PyMuPDF") from exc

    images: list[dict[str, Any]] = []
    max_images = 500
    min_width = 100
    min_height = 100
    screenshot_threshold = 3
    with fitz.open(source) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_saved = 0
            seen_xrefs: set[int] = set()
            for item in page.get_images(full=True):
                xref = int(item[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                except Exception:
                    continue
                data = extracted.get("image")
                if not isinstance(data, bytes):
                    continue
                width = int(extracted.get("width") or 0)
                height = int(extracted.get("height") or 0)
                if width < min_width or height < min_height:
                    continue
                ext = str(extracted.get("ext") or "png").lower()
                mime_type = image_mime_from_ext(ext)
                index = len(images) + 1
                images.append(
                    save_extracted_image(
                        data,
                        dest_dir,
                        rel_to,
                        f"img-{index}.{ext}",
                        index,
                        mime_type,
                        page_index,
                        width=width,
                        height=height,
                    )
                )
                page_saved += 1
                if len(images) >= max_images:
                    return images

            if page_saved >= screenshot_threshold and len(images) < max_images:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                data = pix.tobytes("png")
                index = len(images) + 1
                images.append(
                    save_extracted_image(
                        data,
                        dest_dir,
                        rel_to,
                        f"page-{page_index}.png",
                        index,
                        "image/png",
                        page_index,
                        kind="pageScreenshot",
                        width=pix.width,
                        height=pix.height,
                    )
                )
    return images


def pptx_media_slide_map(archive: zipfile.ZipFile) -> dict[str, int]:
    mapping: dict[str, int] = {}
    slide_names = sorted(
        name
        for name in archive.namelist()
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
    )
    for slide_index, slide_name in enumerate(slide_names, start=1):
        rels_name = f"ppt/slides/_rels/{Path(slide_name).name}.rels"
        if rels_name not in archive.namelist():
            continue
        rels = archive.read(rels_name).decode("utf-8", errors="replace")
        rel_by_id = {
            match.group(1): match.group(2)
            for match in re.finditer(r'Id="([^"]+)".+?Target="([^"]+)"', rels)
        }
        slide_xml = archive.read(slide_name).decode("utf-8", errors="replace")
        for rid in re.findall(r'r:embed="([^"]+)"', slide_xml):
            target = rel_by_id.get(rid)
            if not target:
                continue
            media_path = (Path("ppt/slides") / target).as_posix()
            while "/../" in media_path:
                media_path = re.sub(r"[^/]+/\.\./", "", media_path, count=1)
            if media_path.startswith("../"):
                media_path = f"ppt/{media_path.removeprefix('../')}"
            mapping[media_path] = slide_index
    return mapping


def extract_office_images_for_web(source: Path, dest_dir: Path, rel_to: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    max_images = 500
    min_width = 100
    min_height = 100
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        slide_map = pptx_media_slide_map(archive) if any(name.startswith("ppt/slides/") for name in names) else {}
        media_names = [
            name for name in names
            if (
                name.startswith("word/media/")
                or name.startswith("ppt/media/")
                or name.startswith("xl/media/")
            )
        ]
        for media_name in media_names:
            ext = Path(media_name).suffix.lower().lstrip(".")
            mime_type = image_mime_from_ext(ext)
            if not mime_type.startswith("image/"):
                continue
            data = archive.read(media_name)
            width, height = image_size(data)
            if width and height and (width < min_width or height < min_height):
                continue
            index = len(images) + 1
            images.append(
                save_extracted_image(
                    data,
                    dest_dir,
                    rel_to,
                    f"img-{index}.{ext or 'bin'}",
                    index,
                    mime_type,
                    slide_map.get(media_name),
                    width=width,
                    height=height,
                )
            )
            if len(images) >= max_images:
                break
    return images


def write_text_file(path: Path, contents: str, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        path.write_text(contents, encoding="utf-8")
        return
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def find_project(project_id: str) -> ProjectEntry:
    for project in read_registry():
        if project.id == project_id or project.path == project_id or project.name == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")


def iter_markdown_files(root: Path, max_files: int = DEFAULT_MAX_FILES) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        files.append(path)
        if len(files) >= min(max_files, HARD_MAX_FILES):
            break
    return files


def title_for_markdown(path: Path, content: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else path.stem


def search_project_files(project_path: Path, request: SearchRequest) -> dict[str, Any]:
    tokens = [token for token in re.split(r"\s+", request.query.lower().strip()) if token]
    if not tokens:
        return {"mode": "keyword", "results": [], "tokenHits": 0, "vectorHits": 0}
    results: list[dict[str, Any]] = []
    wiki_root = project_path / "wiki"
    for file_path in iter_markdown_files(wiki_root):
        try:
            content = read_text_file(file_path)
        except HTTPException:
            continue
        lower = content.lower()
        title = title_for_markdown(file_path, content)
        title_match = any(token in title.lower() for token in tokens)
        hits = sum(lower.count(token) for token in tokens)
        if not hits and not title_match:
            continue
        first_hit = min([lower.find(token) for token in tokens if lower.find(token) >= 0] or [0])
        start = max(0, first_hit - 80)
        snippet = content[start : start + 220].replace("\n", " ").strip()
        score = hits + (5 if title_match else 0)
        results.append(
            {
                "path": file_path.relative_to(project_path).as_posix(),
                "title": title,
                "snippet": snippet,
                "titleMatch": title_match,
                "score": score,
                "images": [],
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    token_results = results[: request.topK]
    vector_hits = vector_search(project_path, request.query, request.embeddingConfig, request.topK)
    by_path = {item["path"]: item for item in token_results}
    for hit in vector_hits:
        rel_path = hit["path"]
        page_path = project_path / rel_path
        try:
            content = read_text_file(page_path)
        except HTTPException:
            content = hit.get("text", "")
        existing = by_path.get(rel_path)
        vector_score = float(hit["score"])
        if existing:
            existing["vectorScore"] = vector_score
            existing["score"] = float(existing["score"]) + vector_score * 10
        else:
            by_path[rel_path] = {
                "path": rel_path,
                "title": title_for_markdown(page_path, content),
                "snippet": str(hit.get("text") or content[:220]).replace("\n", " ").strip(),
                "titleMatch": False,
                "score": vector_score * 10,
                "vectorScore": vector_score,
                "images": [],
            }
    merged = sorted(by_path.values(), key=lambda item: item["score"], reverse=True)[: request.topK]
    return {
        "mode": "hybrid" if vector_hits else "keyword",
        "results": merged,
        "tokenHits": len(results),
        "vectorHits": len(vector_hits),
    }


def graph_for_project(project_path: Path) -> dict[str, Any]:
    wiki_root = project_path / "wiki"
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    known: dict[str, str] = {}

    files = iter_markdown_files(wiki_root, HARD_MAX_FILES)
    for file_path in files:
        relative = file_path.relative_to(wiki_root).as_posix()
        page_id = relative.removesuffix(".md")
        content = read_text_file(file_path)
        nodes[page_id] = {"id": page_id, "label": title_for_markdown(file_path, content), "type": page_id.split("/", 1)[0]}
        known[page_id.lower()] = page_id
        known[Path(page_id).name.lower()] = page_id

    for file_path in files:
        source = file_path.relative_to(wiki_root).as_posix().removesuffix(".md")
        content = read_text_file(file_path)
        for raw_target in re.findall(r"\[\[([^\]|#]+)", content):
            target = known.get(raw_target.strip().lower())
            if target and target != source:
                edges.append({"source": source, "target": target, "weight": 1.0})
    return {"ok": True, "nodes": list(nodes.values()), "edges": edges}


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "running",
        "version": APP_VERSION,
        "authRequired": bool(API_TOKEN),
        "enabled": True,
        "backend": "fastapi",
        "dataRoot": DATA_ROOT.as_posix(),
    }


@app.get(f"{API_PREFIX}/config/model", dependencies=[Depends(require_auth)])
def model_config() -> dict[str, Any]:
    defaults = read_model_defaults()
    return {
        "ok": True,
        "path": MODEL_CONFIG_PATH.as_posix(),
        **dump_model(defaults, exclude_none=True),
    }


@app.get(f"{API_PREFIX}/projects", dependencies=[Depends(require_auth)])
def projects() -> dict[str, Any]:
    entries = read_registry()
    return {"ok": True, "projects": [dump_model(entry) for entry in entries], "currentProject": next((dump_model(p) for p in entries if p.current), None)}


@app.post(f"{API_PREFIX}/projects/create", dependencies=[Depends(require_auth)])
def create_project(request: ProjectRequest) -> WikiProject:
    project_id = request.projectId or str(uuid.uuid4())
    project_path = server_project_path(project_id)
    if project_path.exists():
        if not project_path.is_dir():
            raise HTTPException(status_code=409, detail=f"Project path exists and is not a directory: {project_path}")
        create_project_structure(project_path)
    else:
        create_project_structure(project_path)
    return upsert_project(ensure_project_identity(project_path, request.name or project_id, project_id), current=True)


@app.post(f"{API_PREFIX}/projects/open", dependencies=[Depends(require_auth)])
def open_project(request: ProjectRequest) -> WikiProject:
    if not request.path:
        raise HTTPException(status_code=400, detail="path is required")
    project_path = normalize_path(request.path)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project path does not exist")
    registered_project = find_registered_project_by_path(project_path)
    project_id = request.projectId or (registered_project.id if registered_project else None)
    return upsert_project(ensure_project_identity(project_path, request.name, project_id), current=True)


@app.get(f"{API_PREFIX}/projects/{{project_id}}/files", dependencies=[Depends(require_auth)])
def project_files(project_id: str, root: str = "wiki") -> dict[str, Any]:
    project = find_project(project_id)
    base = normalize_path(project.path) / root
    return {"ok": True, "root": base.as_posix(), "files": list_directory_tree(base)}


@app.get(f"{API_PREFIX}/projects/{{project_id}}/activity", dependencies=[Depends(require_auth)])
def get_activity(project_id: str) -> dict[str, Any]:
    project = find_project(project_id)
    return {
        "ok": True,
        "projectId": project_id,
        "items": read_activity(normalize_path(project.path)),
    }


@app.post(f"{API_PREFIX}/projects/{{project_id}}/activity", dependencies=[Depends(require_auth)])
async def update_activity(project_id: str, request: ActivitySnapshotRequest) -> dict[str, Any]:
    project = find_project(project_id)
    items = request.items[:100]
    write_activity(normalize_path(project.path), items)
    await publish_activity(project_id, {
        "projectId": project_id,
        "event": request.event,
        "items": items,
        "updatedAt": int(time.time() * 1000),
    })
    return {"ok": True, "projectId": project_id, "count": len(items)}


@app.get(f"{API_PREFIX}/projects/{{project_id}}/activity/stream", dependencies=[Depends(require_auth)])
async def stream_activity(project_id: str, request: Request) -> StreamingResponse:
    project = find_project(project_id)
    project_path = normalize_path(project.path)
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    subscribers = ACTIVITY_SUBSCRIBERS.setdefault(project_id, set())
    subscribers.add(queue)

    async def events():
        try:
            yield sse_payload("activity", {
                "projectId": project_id,
                "event": "snapshot",
                "items": read_activity(project_path),
                "updatedAt": int(time.time() * 1000),
            })
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield message
                except asyncio.TimeoutError:
                    yield sse_payload("ping", {"ts": int(time.time() * 1000)})
        finally:
            subscribers.discard(queue)
            if not subscribers:
                ACTIVITY_SUBSCRIBERS.pop(project_id, None)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get(f"{API_PREFIX}/projects/{{project_id}}/files/content", dependencies=[Depends(require_auth)])
def project_file_content(project_id: str, path: str = Query(...)) -> dict[str, Any]:
    project = find_project(project_id)
    requested = normalize_path(path)
    normalize_path(project.path)
    return {"ok": True, "path": requested.as_posix(), "content": read_text_file(requested)}


@app.post(f"{API_PREFIX}/projects/{{project_id}}/search", dependencies=[Depends(require_auth)])
def search(project_id: str, request: SearchRequest) -> dict[str, Any]:
    project = find_project(project_id)
    return search_project_files(normalize_path(project.path), request)


@app.get(f"{API_PREFIX}/projects/{{project_id}}/graph", dependencies=[Depends(require_auth)])
def graph(project_id: str) -> dict[str, Any]:
    project = find_project(project_id)
    return graph_for_project(normalize_path(project.path))


@app.post(f"{API_PREFIX}/projects/{{project_id}}/sources/upload", dependencies=[Depends(require_auth)])
async def upload_sources(project_id: str, files: list[UploadFile]) -> dict[str, Any]:
    project = find_project(project_id)
    sources_root = normalize_path(project.path) / "raw" / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        safe_name = re.sub(r'[<>:"|?*\x00-\x1f/\\\\]+', "_", upload.filename or "upload")
        destination = sources_root / safe_name
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        while destination.exists():
            destination = sources_root / f"{stem}-{counter}{suffix}"
            counter += 1
        with destination.open("wb") as fh:
            while chunk := await upload.read(1024 * 1024):
                fh.write(chunk)
        saved.append(destination.as_posix())
    return {"ok": True, "paths": saved}


@app.post(f"{API_PREFIX}/projects/{{project_id}}/ingest", dependencies=[Depends(require_auth)])
async def ingest(project_id: str, request: IngestRequest) -> dict[str, Any]:
    project = find_project(project_id)
    project_path = normalize_path(project.path)
    source_root = project_path / "raw" / "sources"
    defaults = read_model_defaults()
    llm_config = request.llmConfig or defaults.llmConfig
    embedding_config = request.embeddingConfig or defaults.embeddingConfig
    paths: list[Path] = []
    for raw_path in request.paths:
        candidate = normalize_path(raw_path)
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="ingest paths must live under raw/sources") from exc
        if candidate.is_file():
            paths.append(candidate)
    first_title = paths[0].name if paths else "Ingest"
    activity_id = f"activity-{int(time.time() * 1000)}"
    activity_item = {
        "id": activity_id,
        "type": "ingest",
        "title": first_title,
        "status": "running",
        "detail": "Reading source...",
        "filesWritten": [],
        "createdAt": int(time.time() * 1000),
    }
    loop = asyncio.get_running_loop()

    def update_progress(status: str, detail: str, files_written: list[str] | None) -> None:
        activity_item.update({
            "status": status,
            "detail": detail,
        })
        if files_written is not None:
            activity_item["filesWritten"] = files_written
        items = [activity_item]
        write_activity(project_path, items)
        payload = {
                "projectId": project.id,
                "event": "update",
                "items": items,
                "updatedAt": int(time.time() * 1000),
            }
        loop.call_soon_threadsafe(lambda: asyncio.create_task(publish_activity(project.id, payload)))
        if project_id != project.id:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(publish_activity(project_id, payload)))

    update_progress("running", "Reading source...", [])
    results = await asyncio.to_thread(
        ingest_sources,
        project_path,
        paths,
        llm_config,
        embedding_config,
        update_progress,
    )
    return {
        "ok": True,
        "llmConfigUsed": {
            "provider": (llm_config or {}).get("provider"),
            "model": (llm_config or {}).get("model"),
            "customEndpoint": (llm_config or {}).get("customEndpoint"),
            "source": "request" if request.llmConfig else "model_config.yaml",
        },
        "results": [
            {
                "sourcePath": result.source_path,
                "wikiPath": result.wiki_path,
                "status": result.status,
                "title": result.title,
                "sha256": result.digest,
                "error": result.error,
                "generatedFiles": result.generated_files or [],
                "cacheHit": result.cache_hit,
                "reviewItems": result.review_items or [],
            }
            for result in results
        ],
    }


@app.post(f"{API_PREFIX}/projects/{{project_id}}/sources/rescan", dependencies=[Depends(require_auth)])
def rescan(project_id: str) -> dict[str, Any]:
    project = find_project(project_id)
    source_root = normalize_path(project.path) / "raw" / "sources"
    return {"ok": True, "projectId": project_id, "sources": list_directory_tree(source_root) if source_root.exists() else []}


@app.post(f"{API_PREFIX}/fs/read", dependencies=[Depends(require_auth)])
def fs_read(request: PathRequest) -> dict[str, str]:
    path = normalize_path(request.path)
    return {"content": read_text_file(path)}


@app.post(f"{API_PREFIX}/fs/write", dependencies=[Depends(require_auth)])
def fs_write(request: WriteFileRequest) -> dict[str, bool]:
    write_text_file(normalize_path(request.path), request.contents)
    return {"ok": True}


@app.post(f"{API_PREFIX}/fs/write-atomic", dependencies=[Depends(require_auth)])
def fs_write_atomic(request: WriteFileRequest) -> dict[str, bool]:
    write_text_file(normalize_path(request.path), request.contents, atomic=True)
    return {"ok": True}


@app.post(f"{API_PREFIX}/fs/list", dependencies=[Depends(require_auth)])
def fs_list(request: PathRequest) -> list[FileNode]:
    return list_directory_tree(normalize_path(request.path))


@app.post(f"{API_PREFIX}/fs/copy-file", dependencies=[Depends(require_auth)])
def fs_copy_file(request: CopyFileRequest) -> dict[str, bool]:
    source = normalize_path(request.source)
    destination = normalize_path(request.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"ok": True}


@app.post(f"{API_PREFIX}/fs/copy-directory", dependencies=[Depends(require_auth)])
def fs_copy_directory(request: CopyFileRequest) -> list[str]:
    source = normalize_path(request.source)
    destination = normalize_path(request.destination)
    copied: list[str] = []
    for src in source.rglob("*"):
        if src.is_file():
            dst = destination / src.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst.as_posix())
    return copied


@app.post(f"{API_PREFIX}/fs/delete", dependencies=[Depends(require_auth)])
def fs_delete(request: PathRequest) -> dict[str, bool]:
    path = normalize_path(request.path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return {"ok": True}


@app.post(f"{API_PREFIX}/fs/mkdir", dependencies=[Depends(require_auth)])
def fs_mkdir(request: PathRequest) -> dict[str, bool]:
    normalize_path(request.path).mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@app.post(f"{API_PREFIX}/fs/exists", dependencies=[Depends(require_auth)])
def fs_exists(request: PathRequest) -> dict[str, bool]:
    return {"exists": normalize_path(request.path).exists()}


@app.post(f"{API_PREFIX}/fs/stat", dependencies=[Depends(require_auth)])
def fs_stat(request: PathRequest) -> dict[str, int]:
    path = normalize_path(request.path)
    stat = path.stat()
    return {"modifiedTime": int(stat.st_mtime * 1000), "size": stat.st_size}


@app.post(f"{API_PREFIX}/fs/md5", dependencies=[Depends(require_auth)])
def fs_md5(request: PathRequest) -> dict[str, str]:
    digest = hashlib.md5()
    with normalize_path(request.path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"md5": digest.hexdigest()}


@app.post(f"{API_PREFIX}/fs/base64", dependencies=[Depends(require_auth)])
def fs_base64(request: PathRequest) -> dict[str, str]:
    path = normalize_path(request.path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {"base64": base64.b64encode(path.read_bytes()).decode("ascii"), "mimeType": mime_type}


@app.post(f"{API_PREFIX}/fs/preprocess", dependencies=[Depends(require_auth)])
def fs_preprocess(request: PathRequest) -> dict[str, str]:
    return {"content": preprocess_source_file(normalize_path(request.path))}


@app.post(f"{API_PREFIX}/fs/extract-images", dependencies=[Depends(require_auth)])
def fs_extract_images(request: ExtractImagesRequest) -> dict[str, Any]:
    source = normalize_path(request.sourcePath)
    dest_dir = normalize_path(request.destDir)
    rel_to = normalize_path(request.relTo)
    ext = source.suffix.lower()
    if ext == ".pdf":
        images = extract_pdf_images_for_web(source, dest_dir, rel_to)
    elif ext in {".docx", ".pptx"}:
        images = extract_office_images_for_web(source, dest_dir, rel_to)
    else:
        images = []
    return {"ok": True, "images": images}


@app.get(f"{API_PREFIX}/assets/file", dependencies=[Depends(require_auth)])
def asset_file(path: str) -> FileResponse:
    return FileResponse(normalize_path(path))


@app.middleware("http")
async def spa_fallback(request: Request, call_next):
    response = await call_next(request)
    if response.status_code != 404 or request.url.path.startswith(API_PREFIX):
        return response
    if WEB_DIST and WEB_DIST.exists() and (WEB_DIST / "index.html").exists():
        return FileResponse(WEB_DIST / "index.html")
    return response


if WEB_DIST and WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
