from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .extraction import ExtractionError, extract_text
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


class ProjectRequest(BaseModel):
    name: str | None = None
    path: str


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


def dump_model(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[attr-defined]
    return model.dict(**kwargs)


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


def ensure_project_identity(project_path: Path, name: str | None = None) -> WikiProject:
    meta_path = project_meta_path(project_path)
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            project_id = str(raw.get("id") or uuid.uuid4())
            project_name = str(raw.get("name") or name or project_path.name)
        except Exception:
            project_id = str(uuid.uuid4())
            project_name = name or project_path.name
    else:
        project_id = str(uuid.uuid4())
        project_name = name or project_path.name

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"id": project_id, "name": project_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return WikiProject(id=project_id, name=project_name, path=project_path.as_posix())


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
    if path.stat().st_size > MAX_FILE_CONTENT_BYTES:
        raise HTTPException(status_code=413, detail="File is too large to read through the Web API")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="File is not UTF-8 text") from exc


def preprocess_source_file(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {path}")
    try:
        return extract_text(path)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {exc}") from exc


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
        if project.id == project_id or project.path == project_id:
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
    }


@app.get(f"{API_PREFIX}/projects", dependencies=[Depends(require_auth)])
def projects() -> dict[str, Any]:
    entries = read_registry()
    return {"ok": True, "projects": [dump_model(entry) for entry in entries], "currentProject": next((dump_model(p) for p in entries if p.current), None)}


@app.post(f"{API_PREFIX}/projects/create", dependencies=[Depends(require_auth)])
def create_project(request: ProjectRequest) -> WikiProject:
    project_path = normalize_path(request.path)
    create_project_structure(project_path)
    return upsert_project(ensure_project_identity(project_path, request.name), current=True)


@app.post(f"{API_PREFIX}/projects/open", dependencies=[Depends(require_auth)])
def open_project(request: ProjectRequest) -> WikiProject:
    project_path = normalize_path(request.path)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project path does not exist")
    return upsert_project(ensure_project_identity(project_path, request.name), current=True)


@app.get(f"{API_PREFIX}/projects/{{project_id}}/files", dependencies=[Depends(require_auth)])
def project_files(project_id: str, root: str = "wiki") -> dict[str, Any]:
    project = find_project(project_id)
    base = normalize_path(project.path) / root
    return {"ok": True, "root": base.as_posix(), "files": list_directory_tree(base)}


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
def ingest(project_id: str, request: IngestRequest) -> dict[str, Any]:
    project = find_project(project_id)
    project_path = normalize_path(project.path)
    source_root = project_path / "raw" / "sources"
    paths: list[Path] = []
    for raw_path in request.paths:
        candidate = normalize_path(raw_path)
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="ingest paths must live under raw/sources") from exc
        if candidate.is_file():
            paths.append(candidate)
    results = ingest_sources(project_path, paths, request.llmConfig, request.embeddingConfig)
    return {
        "ok": True,
        "results": [
            {
                "sourcePath": result.source_path,
                "wikiPath": result.wiki_path,
                "status": result.status,
                "title": result.title,
                "sha256": result.digest,
                "error": result.error,
                "generatedFiles": result.generated_files or [],
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
