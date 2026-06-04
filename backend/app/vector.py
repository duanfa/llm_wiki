from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def has_embedding_config(config: dict[str, Any] | None) -> bool:
    cfg = config or {}
    return bool(cfg.get("enabled") and cfg.get("endpoint") and cfg.get("model"))


def vector_index_path(project_path: Path) -> Path:
    return project_path / ".llm-wiki" / "vector-index.json"


def rebuild_vector_index(project_path: Path, embedding_config: dict[str, Any] | None) -> int:
    if not has_embedding_config(embedding_config):
        return 0
    rows: list[dict[str, Any]] = []
    wiki_root = project_path / "wiki"
    for page in wiki_root.rglob("*.md"):
        rel = page.relative_to(project_path).as_posix()
        content = page.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = fetch_embedding(chunk, embedding_config)
            if embedding:
                rows.append({"path": rel, "chunk": idx, "text": chunk, "embedding": embedding})
    target = vector_index_path(project_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def vector_search(
    project_path: Path,
    query: str,
    embedding_config: dict[str, Any] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if not has_embedding_config(embedding_config):
        return []
    index_path = vector_index_path(project_path)
    if not index_path.exists():
        rebuild_vector_index(project_path, embedding_config)
    if not index_path.exists():
        return []
    query_embedding = fetch_embedding(query, embedding_config)
    if not query_embedding:
        return []
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = cosine_similarity(query_embedding, row.get("embedding") or [])
        if score > 0:
            scored.append({**row, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)

    by_path: dict[str, dict[str, Any]] = {}
    for item in scored:
        path = item["path"]
        current = by_path.get(path)
        if current is None or item["score"] > current["score"]:
            by_path[path] = item
        if len(by_path) >= top_k:
            break
    return list(by_path.values())


def fetch_embedding(text: str, config: dict[str, Any] | None) -> list[float] | None:
    cfg = config or {}
    endpoint = str(cfg.get("endpoint") or "")
    model = str(cfg.get("model") or "")
    if not endpoint or not model:
        return None
    headers = {"Content-Type": "application/json"}
    api_key = str(cfg.get("apiKey") or os.getenv("LLM_WIKI_EMBEDDING_API_KEY", ""))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for key, value in (cfg.get("extraHeaders") or {}).items():
        if isinstance(key, str) and isinstance(value, str):
            headers[key] = value
    body = {"model": model, "input": text[:8000]}
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    data = payload.get("data") or []
    if data and isinstance(data, list):
        embedding = data[0].get("embedding")
        if isinstance(embedding, list):
            return [float(v) for v in embedding]
    return None


def chunk_text(text: str, max_chars: int = 1600) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in normalized.split("\n\n"):
        if len(current) + len(paragraph) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
