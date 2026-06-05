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
    generation: str
    files: list[tuple[str, str]]


def has_llm_config(config: dict[str, Any] | None) -> bool:
    if config and config.get("endpoint") and config.get("model"):
        return True
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
        endpoint = normalize_chat_completions_endpoint(
            str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))
        )
    elif provider == "openai":
        endpoint = os.getenv("LLM_WIKI_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    elif provider == "minimax":
        endpoint = str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))
    else:
        endpoint = normalize_chat_completions_endpoint(
            str(config.get("customEndpoint") or os.getenv("LLM_WIKI_LLM_ENDPOINT", ""))
        )

    return {
        "provider": provider,
        "endpoint": endpoint,
        "apiKey": api_key,
        "model": model,
    }


def normalize_chat_completions_endpoint(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if not base or re.search(r"/chat/completions$", base, re.IGNORECASE):
        return base
    return f"{base}/chat/completions"


def two_stage_generate(
    source_title: str,
    source_path: str,
    source_text: str,
    llm_config: dict[str, Any] | None,
    purpose: str = "",
    schema: str = "",
    index: str = "",
    overview: str = "",
    source_summary_path: str | None = None,
) -> LlmResult | None:
    cfg = normalized_llm_config(llm_config)
    if not cfg.get("endpoint") or not cfg.get("model"):
        return None

    clipped = source_text[:80_000]
    analysis = chat_completion(
        cfg,
        [
            {
                "role": "system",
                "content": build_analysis_prompt(purpose, index, source_text),
            },
            {
                "role": "user",
                "content": f"Analyze this source document:\n\n**File:** {source_path}\n\n---\n\n{clipped}",
            },
        ],
        temperature=0.2,
    )

    summary_path = source_summary_path or f"wiki/sources/{source_title}.md"
    generation = chat_completion(
        cfg,
        [
            {
                "role": "system",
                "content": build_generation_prompt(
                    schema,
                    purpose,
                    index,
                    overview,
                    source_path,
                    summary_path,
                    source_text,
                ),
            },
            {
                "role": "user",
                "content": "\n".join([
                    f"Source document to process: **{source_path}**",
                    "",
                    "The Stage 1 analysis below is CONTEXT to inform your output. Do NOT echo it.",
                    "Your output must be FILE blocks as specified in the system prompt, nothing else.",
                    "",
                    "## Stage 1 Analysis",
                    analysis,
                    "",
                    "## Source Context",
                    clipped[:20_000],
                    "",
                    f"Now emit FILE blocks for the wiki files derived from **{source_path}**.",
                    "Your response MUST begin with `---FILE:` as the very first characters.",
                ]),
            },
        ],
        temperature=0.1,
    )
    return LlmResult(analysis=analysis, generation=generation, files=parse_file_blocks(generation))


def language_rule(source_content: str) -> str:
    return (
        "## ⚠️ MANDATORY OUTPUT LANGUAGE: Chinese\n\n"
        "You MUST write your entire response (including wiki page titles, content, descriptions, summaries, and any generated text) in **Chinese**.\n"
        "The source material or wiki content may be in a different language, but this is IRRELEVANT to your output language.\n"
        "Ignore the language of any source content. Generate everything in Chinese only.\n"
        "Proper nouns should use standard Chinese transliteration when appropriate.\n"
        "DO NOT use any other language. This overrides all other instructions."
    )


def build_analysis_prompt(purpose: str, index: str, source_content: str) -> str:
    parts = [
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        language_rule(source_content),
        "",
        "Your analysis should cover:",
        "## Key Entities",
        "List people, organizations, products, datasets, tools mentioned. For each: name/type, role, and whether it likely already exists in the wiki.",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each: definition, why it matters, and whether it likely already exists.",
        "## Main Arguments & Findings",
        "What are the core claims, evidence, and caveats?",
        "## Connections to Existing Wiki",
        "What existing pages does this source relate to?",
        "## Recommendations",
        "What wiki pages should be created or updated?",
    ]
    if purpose:
        parts.extend(["", f"## Wiki Purpose\n{purpose}"])
    if index:
        parts.extend(["", f"## Current Wiki Index\n{index}"])
    return "\n".join(parts)


def build_generation_prompt(
    schema: str,
    purpose: str,
    index: str,
    overview: str,
    source_file_name: str,
    source_summary_path: str,
    source_content: str,
) -> str:
    parts = [
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. Reason internally and output only FILE blocks.",
        "",
        language_rule(source_content),
        "",
        "## IMPORTANT: Source File",
        f"The original source file is: **{source_file_name}**",
        f"All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        "",
    ]
    if schema:
        parts.extend([
            "## Project Schema and Routing",
            schema,
            "",
            "Use schema-defined folders when present. Otherwise use wiki/entities/ and wiki/concepts/.",
            "",
        ])
    parts.extend([
        "## What to generate",
        f"1. A source summary page at **{source_summary_path}** (MUST use this exact path).",
        "2. Entity pages for key named things under wiki/entities/.",
        "3. Concept pages for key ideas, methods, techniques, and abstractions under wiki/concepts/.",
        "4. An updated wiki/index.md preserving existing entries and adding new ones.",
        "5. A log entry for wiki/log.md.",
        "6. An updated wiki/overview.md reflecting the whole wiki.",
        "",
        "## Frontmatter Rules",
        "Every generated page must start with YAML frontmatter delimited by `---`.",
        "Required fields: type, title, created, updated, tags, related, sources.",
        f"`sources` MUST include \"{source_file_name}\".",
        "Use [[wikilink]] syntax in page bodies.",
        "",
    ])
    if purpose:
        parts.extend([f"## Wiki Purpose\n{purpose}", ""])
    if index:
        parts.extend([f"## Current Wiki Index\n{index}", ""])
    if overview:
        parts.extend([f"## Current Overview\n{overview}", ""])
    parts.extend([
        "## Output Format",
        "Your ENTIRE response consists of FILE blocks. Nothing else.",
        "FILE block template:",
        "---FILE: wiki/path/to/page.md---",
        "(complete file content with YAML frontmatter)",
        "---END FILE---",
        "",
        "## Output Requirements",
        "1. The FIRST character of your response MUST be `-` (the opening of `---FILE:`).",
        "2. DO NOT output any preamble.",
        "3. DO NOT echo or restate the analysis.",
        "4. Between blocks, use only blank lines.",
        "5. Every FILE block content must be Chinese.",
        "",
        language_rule(source_content),
    ])
    return "\n".join(parts)


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
        "max_tokens": 4096,
    }
    if is_qwen_model(str(config.get("model") or "")):
        body["chat_template_kwargs"] = {"enable_thinking": False}
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


def is_qwen_model(model: str) -> bool:
    return "qwen" in model.lower()


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
    allowed_exact_paths = {"wiki/index.md", "wiki/log.md", "wiki/overview.md"}
    allowed_prefixes = ("wiki/entities/", "wiki/concepts/", "wiki/sources/")
    if path not in allowed_exact_paths and not path.startswith(allowed_prefixes):
        return False
    if path.startswith("/") or ".." in path.split("/"):
        return False
    if not path.endswith(".md"):
        return False
    return not bool(re.search(r"[\x00-\x1f<>:\"|?*]", path))
