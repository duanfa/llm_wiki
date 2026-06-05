from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .extraction import ExtractionError, compact_json, extract_text, slugify
from .llm import has_llm_config, two_stage_generate
from .vector import rebuild_vector_index


@dataclass
class IngestResult:
    source_path: str
    wiki_path: str
    status: str
    title: str
    digest: str
    error: str | None = None
    generated_files: list[str] | None = None
    cache_hit: bool = False
    review_items: list[dict[str, Any]] | None = None


def ingest_sources(
    project_path: Path,
    source_paths: list[Path],
    llm_config: dict[str, Any] | None = None,
    embedding_config: dict[str, Any] | None = None,
    progress: Callable[[str, str, list[str] | None], None] | None = None,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for source_path in source_paths:
        results.append(ingest_source(project_path, source_path, llm_config, progress))
    update_index(project_path)
    update_overview(project_path)
    append_log(project_path, results)
    rebuild_vector_index(project_path, embedding_config)
    return results


def ingest_source(
    project_path: Path,
    source_path: Path,
    llm_config: dict[str, Any] | None = None,
    progress: Callable[[str, str, list[str] | None], None] | None = None,
) -> IngestResult:
    digest = file_sha256(source_path)
    source_identity = source_identity_for_path(project_path, source_path)
    slug = source_summary_slug_from_identity(source_identity)
    wiki_path = project_path / "wiki" / "sources" / f"{slug}.md"
    progress_emit(progress, "running", "Reading source...", [])

    try:
        extracted = extract_text(source_path)
        status = "done"
        error = None
    except ExtractionError as exc:
        extracted = f"[Extraction failed: {exc}]"
        status = "failed"
        error = str(exc)
    except Exception as exc:
        extracted = f"[Extraction failed: {exc}]"
        status = "failed"
        error = str(exc)

    title = source_path.stem
    cached_files = check_ingest_cache(project_path, source_identity, extracted)
    if cached_files is not None:
        detail = f"Skipped (unchanged) — {len(cached_files)} files from previous ingest"
        progress_emit(progress, "done", detail, cached_files)
        return IngestResult(
            source_path=source_path.as_posix(),
            wiki_path=wiki_path.relative_to(project_path).as_posix(),
            status=status,
            title=title,
            digest=digest,
            error=error,
            generated_files=cached_files,
            cache_hit=True,
            review_items=[],
        )

    content = build_source_summary(project_path, source_path, source_identity, title, digest, extracted, status, error)
    wiki_path.parent.mkdir(parents=True, exist_ok=True)
    wiki_path.write_text(content, encoding="utf-8")
    generated_files = [wiki_path.relative_to(project_path).as_posix()]
    generated, review_items = run_two_stage_generation(
            project_path,
            source_path,
            source_identity,
            wiki_path.relative_to(project_path).as_posix(),
            title,
            extracted,
            llm_config,
            progress,
        )
    generated_files.extend(generated)
    generated_files = dedupe_preserve_order(generated_files)
    save_ingest_cache(project_path, source_identity, extracted, generated_files)
    detail = f"{len(generated_files)} files written"
    if review_items:
        detail += f", {len(review_items)} review item(s)"
    progress_emit(progress, "done", detail, generated_files)
    return IngestResult(
        source_path=source_path.as_posix(),
        wiki_path=wiki_path.relative_to(project_path).as_posix(),
        status=status,
        title=title,
        digest=digest,
        error=error,
        generated_files=generated_files,
        review_items=review_items,
    )


def run_two_stage_generation(
    project_path: Path,
    source_path: Path,
    source_identity: str,
    source_summary_path: str,
    title: str,
    extracted: str,
    llm_config: dict[str, Any] | None,
    progress: Callable[[str, str, list[str] | None], None] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not has_llm_config(llm_config):
        append_backend_note(project_path, f"LLM two-stage generation skipped for {source_path.name}: llmConfig is missing endpoint or model")
        return [], []
    try:
        progress_emit(progress, "running", "Step 1/2: Analyzing source...", None)
        result = two_stage_generate(
            title,
            source_identity,
            extracted,
            llm_config,
            purpose=read_optional_text(project_path / "purpose.md"),
            schema=read_optional_text(project_path / "schema.md"),
            index=read_optional_text(project_path / "wiki" / "index.md"),
            overview=read_optional_text(project_path / "wiki" / "overview.md"),
            source_summary_path=source_summary_path,
        )
    except Exception as exc:
        append_backend_note(project_path, f"LLM two-stage generation failed for {source_path.name}: {exc}")
        return [], []
    if result is None:
        append_backend_note(project_path, f"LLM two-stage generation skipped for {source_path.name}: llmConfig is not configured")
        return [], []

    progress_emit(progress, "running", "Step 2/2: Generating wiki pages...", None)
    source_summary_slug = source_summary_slug_from_identity(source_identity)
    analysis_path = project_path / ".llm-wiki" / "analysis" / f"{source_summary_slug}.md"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(result.analysis + "\n", encoding="utf-8")
    generation_path = project_path / ".llm-wiki" / "generation" / f"{source_summary_slug}.md"
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_path.write_text(result.generation + "\n", encoding="utf-8")

    written: list[str] = []
    progress_emit(progress, "running", "Writing files...", None)
    for rel_path, content in result.files:
        target = project_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == "wiki/log.md" and target.exists():
            content = target.read_text(encoding="utf-8", errors="replace").rstrip() + "\n\n" + content.lstrip()
        elif rel_path not in {"wiki/index.md", "wiki/overview.md"} and target.exists():
            content = merge_generated_page(target.read_text(encoding="utf-8", errors="replace"), content)
        target.write_text(content, encoding="utf-8")
        if rel_path not in written:
            written.append(rel_path)
    if not written:
        append_backend_note(
            project_path,
            f"LLM two-stage generation produced no FILE blocks for {source_path.name}; raw output: {generation_path.as_posix()}",
        )
    review_items = parse_review_blocks(result.generation, source_path.as_posix())
    if review_items:
        save_review_items(project_path, review_items)
    return written, review_items


def merge_generated_page(existing: str, generated: str) -> str:
    if generated.strip() in existing:
        return existing
    return existing.rstrip() + "\n\n---\n\n" + generated.lstrip()


def append_backend_note(project_path: Path, message: str) -> None:
    note_path = project_path / ".llm-wiki" / "backend-notes.log"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with note_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{int(time.time() * 1000)} {message}\n")


def progress_emit(
    progress: Callable[[str, str, list[str] | None], None] | None,
    status: str,
    detail: str,
    files_written: list[str] | None,
) -> None:
    if progress:
        progress(status, detail, files_written)


def ingest_cache_path(project_path: Path) -> Path:
    return project_path / ".llm-wiki" / "ingest-cache.json"


def load_ingest_cache(project_path: Path) -> dict[str, Any]:
    path = ingest_cache_path(project_path)
    if not path.exists():
        return {"entries": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {"entries": {}}
    except Exception:
        return {"entries": {}}


def save_ingest_cache_file(project_path: Path, cache: dict[str, Any]) -> None:
    path = ingest_cache_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_cache_hash(source_content: str) -> str:
    version = "ingest-cache-v2-image-context"
    return hashlib.sha256(f"{version}\n{source_content}".encode("utf-8")).hexdigest()


def check_ingest_cache(project_path: Path, source_identity: str, source_content: str) -> list[str] | None:
    cache = load_ingest_cache(project_path)
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(source_identity)
    if not isinstance(entry, dict):
        return None
    if entry.get("hash") != ingest_cache_hash(source_content):
        return None
    files = entry.get("filesWritten")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        return None
    for rel_path in files:
        full_path = Path(rel_path) if Path(rel_path).is_absolute() else project_path / rel_path
        if not full_path.exists():
            return None
    return files


def save_ingest_cache(project_path: Path, source_identity: str, source_content: str, files_written: list[str]) -> None:
    cache = load_ingest_cache(project_path)
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    entries[source_identity] = {
        "hash": ingest_cache_hash(source_content),
        "timestamp": int(time.time() * 1000),
        "filesWritten": files_written,
    }
    cache["entries"] = entries
    save_ingest_cache_file(project_path, cache)


REVIEW_BLOCK_RE = re.compile(
    r"---\s*REVIEW:\s*(?P<type>\w[\w-]*)\s*\|\s*(?P<title>.+?)\s*---\s*\n(?P<body>.*?)\n---\s*END\s+REVIEW\s*---",
    re.IGNORECASE | re.DOTALL,
)


def parse_review_blocks(text: str, source_path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in REVIEW_BLOCK_RE.finditer(text):
        raw_type = match.group("type").strip().lower()
        review_type = raw_type if raw_type in {"contradiction", "duplicate", "missing-page", "suggestion", "confirm"} else "confirm"
        title = match.group("title").strip()
        body = match.group("body").strip()
        options = parse_review_line(body, "OPTIONS") or "Create Page | Skip"
        pages = parse_review_line(body, "PAGES")
        search = parse_review_line(body, "SEARCH")
        description = re.sub(r"^(OPTIONS|PAGES|SEARCH):.*$", "", body, flags=re.MULTILINE).strip()
        items.append({
            "id": f"review-{int(time.time() * 1000)}-{len(items) + 1}",
            "type": review_type,
            "title": title,
            "description": description,
            "sourcePath": source_path,
            "affectedPages": [item.strip() for item in pages.split(",") if item.strip()] if pages else None,
            "searchQueries": [item.strip() for item in search.split("|") if item.strip()] if search else None,
            "options": [{"label": item.strip(), "action": item.strip()} for item in options.split("|") if item.strip()],
            "resolved": False,
            "createdAt": int(time.time() * 1000),
        })
    return items


def parse_review_line(body: str, key: str) -> str | None:
    match = re.search(rf"^{key}:\s*(.+)$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def save_review_items(project_path: Path, review_items: list[dict[str, Any]]) -> None:
    if not review_items:
        return
    path = project_path / ".llm-wiki" / "reviews.json"
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing = [item for item in raw if isinstance(item, dict)]
        except Exception:
            existing = []
    seen = {(item.get("type"), str(item.get("title", "")).lower()) for item in existing if not item.get("resolved")}
    merged = existing[:]
    for item in review_items:
        key = (item.get("type"), str(item.get("title", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_source_summary(
    project_path: Path,
    source_path: Path,
    source_identity: str,
    title: str,
    digest: str,
    extracted: str,
    status: str,
    error: str | None,
) -> str:
    now = int(time.time() * 1000)
    frontmatter: dict[str, Any] = {
        "type": "source",
        "title": title,
        "sources": [source_identity],
        "sha256": digest,
        "ingested_at": now,
        "ingest_runtime": "fastapi",
        "ingest_status": status,
    }
    if error:
        frontmatter["ingest_error"] = error

    excerpt = extracted.strip()
    if len(excerpt) > 20_000:
        excerpt = excerpt[:20_000] + "\n\n[Source text truncated in source summary]\n"

    return (
        "---\n"
        + "\n".join(f"{key}: {yaml_value(value)}" for key, value in frontmatter.items())
        + "\n---\n\n"
        + f"# {title}\n\n"
        + "## Source\n\n"
        + f"- Path: `{source_identity}`\n"
        + f"- SHA256: `{digest}`\n"
        + f"- Runtime: `fastapi`\n"
        + f"- Status: `{status}`\n\n"
        + "## Extracted Text\n\n"
        + (excerpt or "[No extractable text found.]")
        + "\n"
    )


def update_index(project_path: Path) -> None:
    wiki_root = project_path / "wiki"
    index_path = wiki_root / "index.md"
    pages = sorted(
        path.relative_to(wiki_root).as_posix()
        for path in wiki_root.rglob("*.md")
        if path.name != "index.md" and ".llm-wiki" not in path.parts
    )
    lines = ["# Index", "", "## Pages", ""]
    lines.extend(f"- [[{page.removesuffix('.md')}]]" for page in pages)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_overview(project_path: Path) -> None:
    wiki_root = project_path / "wiki"
    overview_path = wiki_root / "overview.md"
    source_pages = sorted((wiki_root / "sources").glob("*.md")) if (wiki_root / "sources").exists() else []
    lines = [
        "# Overview",
        "",
        "This overview was refreshed by the FastAPI backend.",
        "",
        f"- Source summaries: {len(source_pages)}",
        f"- Last updated: {int(time.time() * 1000)}",
        "",
        "## Recent Sources",
        "",
    ]
    for page in source_pages[-20:]:
        lines.append(f"- [[sources/{page.stem}]]")
    overview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log(project_path: Path, results: list[IngestResult]) -> None:
    log_path = project_path / "wiki" / "log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Log\n"
    lines = [existing.rstrip(), "", f"## FastAPI Ingest {int(time.time() * 1000)}", ""]
    for result in results:
        generated = f" generated={len(result.generated_files or [])}" if result.generated_files else ""
        lines.append(f"- `{result.status}` {result.source_path} -> `{result.wiki_path}`{generated}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def source_identity_for_path(project_path: Path, source_path: Path) -> str:
    try:
        return source_path.relative_to(project_path / "raw" / "sources").as_posix()
    except ValueError:
        return source_path.name


def source_summary_slug_from_identity(source_identity: str) -> str:
    without_ext = re_sub_ext(source_identity)
    parts = [part.strip() for part in without_ext.split("/") if part.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "source"

    hash_value = stable_slug_hash(source_identity)
    slug = "--".join(f"{len(part.encode('utf-8'))}-{percent_encode_path_part(part)}" for part in parts)
    full_slug = f"{slug}--{hash_value}"
    if len(full_slug) <= 120:
        return full_slug
    readable_limit = 120 - len(hash_value) - 2
    readable_prefix = trim_incomplete_percent_encoding(full_slug[:readable_limit]).rstrip("-%")
    return f"{readable_prefix or 'source'}--{hash_value}"


def re_sub_ext(value: str) -> str:
    return re.sub(r"\.[^/.]+$", "", value)


def percent_encode_path_part(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def trim_incomplete_percent_encoding(value: str) -> str:
    return re.sub(r"%(?:[0-9A-F])?$", "", value, flags=re.IGNORECASE)


def stable_slug_hash(value: str) -> str:
    hash_value = 0x811C9DC5
    for char in value:
        hash_value ^= ord(char)
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return base36(hash_value)


def base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def unique_source_slug(project_path: Path, source_path: Path) -> str:
    base = slugify(source_path.name)
    digest = file_sha256(source_path)[:8]
    return f"{base}-{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def yaml_value(value: object) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return compact_json(value)
    return json.dumps(value, ensure_ascii=False)
