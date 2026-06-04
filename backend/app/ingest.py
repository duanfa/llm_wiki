from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extraction import ExtractionError, compact_json, extract_text, slugify
from .llm import two_stage_generate
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


def ingest_sources(
    project_path: Path,
    source_paths: list[Path],
    llm_config: dict[str, Any] | None = None,
    embedding_config: dict[str, Any] | None = None,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for source_path in source_paths:
        results.append(ingest_source(project_path, source_path, llm_config))
    update_index(project_path)
    update_overview(project_path)
    append_log(project_path, results)
    rebuild_vector_index(project_path, embedding_config)
    return results


def ingest_source(
    project_path: Path,
    source_path: Path,
    llm_config: dict[str, Any] | None = None,
) -> IngestResult:
    digest = file_sha256(source_path)
    slug = unique_source_slug(project_path, source_path)
    wiki_path = project_path / "wiki" / "sources" / f"{slug}.md"

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
    content = build_source_summary(project_path, source_path, title, digest, extracted, status, error)
    wiki_path.parent.mkdir(parents=True, exist_ok=True)
    wiki_path.write_text(content, encoding="utf-8")
    generated_files = run_two_stage_generation(project_path, source_path, title, extracted, llm_config)
    return IngestResult(
        source_path=source_path.as_posix(),
        wiki_path=wiki_path.relative_to(project_path).as_posix(),
        status=status,
        title=title,
        digest=digest,
        error=error,
        generated_files=generated_files,
    )


def run_two_stage_generation(
    project_path: Path,
    source_path: Path,
    title: str,
    extracted: str,
    llm_config: dict[str, Any] | None,
) -> list[str]:
    try:
        result = two_stage_generate(title, safe_relative(source_path, project_path), extracted, llm_config)
    except Exception as exc:
        append_backend_note(project_path, f"LLM two-stage generation failed for {source_path.name}: {exc}")
        return []
    if result is None:
        return []

    analysis_path = project_path / ".llm-wiki" / "analysis" / f"{unique_source_slug(project_path, source_path)}.md"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(result.analysis + "\n", encoding="utf-8")

    written: list[str] = []
    for rel_path, content in result.files:
        target = project_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            content = merge_generated_page(target.read_text(encoding="utf-8", errors="replace"), content)
        target.write_text(content, encoding="utf-8")
        written.append(rel_path)
    return written


def merge_generated_page(existing: str, generated: str) -> str:
    if generated.strip() in existing:
        return existing
    return existing.rstrip() + "\n\n---\n\n" + generated.lstrip()


def append_backend_note(project_path: Path, message: str) -> None:
    note_path = project_path / ".llm-wiki" / "backend-notes.log"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with note_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{int(time.time() * 1000)} {message}\n")


def build_source_summary(
    project_path: Path,
    source_path: Path,
    title: str,
    digest: str,
    extracted: str,
    status: str,
    error: str | None,
) -> str:
    rel_source = safe_relative(source_path, project_path)
    now = int(time.time() * 1000)
    frontmatter: dict[str, Any] = {
        "type": "source",
        "title": title,
        "sources": [rel_source],
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
        + f"- Path: `{rel_source}`\n"
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
