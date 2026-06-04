from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LlmResult:
    analysis: str
    files: list[tuple[str, str]]


def has_llm_config(config: dict[str, Any] | None) -> bool:
    cfg = normalized_llm_config(config)
    return bool(cfg.get("endpoint") and cfg.get("model"))


def normalized_llm_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    provider = config.get("provider") or os.getenv("LLM_WIKI_LLM_PROVIDER", "openai")
    endpoint = ""
    api_key = str(config.get("apiKey") or os.getenv("LLM_WIKI_LLM_API_KEY", ""))
    model = str(config.get("model") or os.getenv("LLM_WIKI_LLM_MODEL", ""))

    if provider == "ollama":
        base = str(config.get("ollamaUrl") or os.getenv("LLM_WIKI_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        endpoint = f"{base}/v1/chat/completions"
    elif provider == "custom":
        endpoint = str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))
    elif provider == "openai":
        endpoint = os.getenv("LLM_WIKI_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    elif provider == "minimax":
        endpoint = str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))
    else:
        endpoint = str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))

    return {
        "provider": provider,
        "endpoint": endpoint,
        "apiKey": api_key,
        "model": model,
    }


def two_stage_generate(
    source_title: str,
    source_path: str,
    source_text: str,
    llm_config: dict[str, Any] | None,
) -> LlmResult | None:
    cfg = normalized_llm_config(llm_config)
    if not has_llm_config(cfg):
        return None

    clipped = source_text[:80_000]
    analysis = chat_completion(
        cfg,
        [
            {
                "role": "system",
                "content": (
                    "You are building a durable personal wiki. Analyze the source and identify "
                    "important entities, concepts, claims, and relationships. Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"Source title: {source_title}\nSource path: {source_path}\n\nSOURCE:\n{clipped}",
            },
        ],
        temperature=0.2,
    )

    generation = chat_completion(
        cfg,
        [
            {
                "role": "system",
                "content": (
                    "Generate wiki pages from the analysis. Return only FILE blocks. "
                    "Every path must be under wiki/entities/ or wiki/concepts/. "
                    "Format exactly:\n---FILE: wiki/concepts/example.md---\n# Title\n...\n---END FILE---"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source title: {source_title}\nSource path: {source_path}\n\n"
                    f"ANALYSIS:\n{analysis}\n\nSOURCE EXCERPT:\n{clipped[:20_000]}"
                ),
            },
        ],
        temperature=0.1,
    )
    return LlmResult(analysis=analysis, files=parse_file_blocks(generation))


def chat_completion(config: dict[str, Any], messages: list[dict[str, str]], temperature: float) -> str:
    endpoint = str(config.get("endpoint") or "")
    if not endpoint:
        raise RuntimeError("LLM endpoint is not configured")

    headers = {"Content-Type": "application/json"}
    api_key = str(config.get("apiKey") or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": config.get("model"),
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code} {detail}") from exc

    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


FILE_BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<path>.+?)\s*---\s*\n(?P<content>.*?)\n---\s*END\s+FILE\s*---",
    re.IGNORECASE | re.DOTALL,
)


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for match in FILE_BLOCK_RE.finditer(text):
        path = match.group("path").strip().replace("\\", "/")
        content = match.group("content").strip() + "\n"
        if is_safe_generated_path(path):
            files.append((path, content))
    return files


def is_safe_generated_path(path: str) -> bool:
    if not (path.startswith("wiki/entities/") or path.startswith("wiki/concepts/")):
        return False
    if path.startswith("/") or ".." in path.split("/"):
        return False
    if not path.endswith(".md"):
        return False
    return not bool(re.search(r"[\x00-\x1f<>:\"|?*]", path))
