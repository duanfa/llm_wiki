# LLM Wiki Web/FastAPI Runtime

This runtime exposes LLM Wiki through a Python FastAPI backend and a browser-based Vite frontend.

## Development

```bash
cd llm_wiki
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Optional: allow the backend to open existing absolute project paths.
export LLM_WIKI_ALLOW_ABSOLUTE_PATHS=1
export LLM_WIKI_DATA_ROOT="$HOME/llm-wiki-data"

npm run backend
npm run dev:web
```

Open the frontend at `http://localhost:1420`.

For a single-process production run:

```bash
npm run build:web
LLM_WIKI_WEB_DIST=dist npm run serve:web
```

Open `http://localhost:8000`.

## Docker

```bash
docker build -t llm-wiki-web .
docker run --rm -p 8000:8000 \
  -v "$PWD/.web-data:/data" \
  -e LLM_WIKI_API_TOKEN=change-me \
  llm-wiki-web
```

## Environment

- `LLM_WIKI_DATA_ROOT`: root directory for server-managed projects. Defaults to `~/llm-wiki-data`.
- `LLM_WIKI_ALLOW_ABSOLUTE_PATHS=1`: lets the backend access existing absolute paths during migration.
- `LLM_WIKI_API_TOKEN`: enables Bearer-token auth for API requests.
- `LLM_WIKI_CORS_ORIGINS`: comma-separated allowed origins. Defaults to `*`.
- `VITE_LLM_WIKI_API_BASE_URL`: frontend API base URL. Defaults to `http://<current-host>:8000`.
- `VITE_LLM_WIKI_API_TOKEN`: development-only frontend token injection.
- `LLM_WIKI_LLM_ENDPOINT`, `LLM_WIKI_LLM_API_KEY`, `LLM_WIKI_LLM_MODEL`: optional server-side fallback LLM config for two-stage ingest.
- `LLM_WIKI_EMBEDDING_API_KEY`: optional server-side fallback embedding key.

## Current API Surface

- `GET /api/v1/health`
- `GET /api/v1/projects`
- `POST /api/v1/projects/create`
- `POST /api/v1/projects/open`
- `GET /api/v1/projects/{id}/files`
- `GET /api/v1/projects/{id}/files/content`
- `POST /api/v1/projects/{id}/search`
- `GET /api/v1/projects/{id}/graph`
- `POST /api/v1/projects/{id}/sources/upload`
- `POST /api/v1/projects/{id}/ingest`
- `POST /api/v1/projects/{id}/sources/rescan`
- `POST /api/v1/fs/*` for the file operations used by the existing frontend adapter

The first Web runtime keeps the desktop app intact and routes the shared frontend through HTTP when `VITE_LLM_WIKI_RUNTIME=web`.

## Backend Ingest

The FastAPI runtime performs server-side extraction for uploaded sources and writes deterministic source summary pages under `wiki/sources/`. Supported formats include Markdown/text, HTML, PDF, DOCX, PPTX, and XLSX.

When the Web frontend has a usable LLM configuration, `/api/v1/projects/{id}/ingest` runs a two-stage pipeline:

1. Analyze the source into entities, concepts, claims, and relationships.
2. Generate safe `wiki/entities/*.md` and `wiki/concepts/*.md` pages from FILE blocks.

When embedding is enabled, the backend rebuilds `.llm-wiki/vector-index.json` and `/search` returns hybrid keyword/vector results.
