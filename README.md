# DeepGyan

DeepGyan is an AI learning platform for Nepali high‑school students. It lets students upload a textbook PDF, view pages on the left, and ask questions on the right. OCR and embeddings are precomputed so the assistant can answer using either the current page window or whole‑book retrieval.

**Core features**
- PDF viewer with page controls and upload flow
- OCR (Tesseract) for scanned pages
- Whole‑book embeddings stored in Postgres + pgvector
- Chat UI with streamed responses and a separate “Thinking” panel
- Multi‑book catalog with persisted metadata
- Animation generation via Manim plugin jobs (script + MP4 artifacts)

## Requirements
- Python 3.11+ (3.12 works)
- Docker + Docker Compose
- Tesseract OCR installed and available in `PATH`
- (macOS) `brew install tesseract`
- Optional for animation rendering: Manim CE + LaTeX + ffmpeg
  - macOS: `brew install tesseract`
  - Arch Linux: `sudo pacman -S tesseract tesseract-data-eng`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-data-eng`

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` from the example and add your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set INFERENCE_PROVIDER plus the matching model/key settings.
   ```
4. Start services (DB + app):
   ```bash
   ./start_services.sh
   ```
5. Open the app at [http://localhost:8000](http://localhost:8000).

## How It Works
- **Upload** a PDF from the dashboard.
- **OCR** runs automatically for pages without native text.
- **Embeddings** are generated in batches and upserted into pgvector.
- **Ask** a question:
  - *Current page mode* builds a structured (paraphrased) summary of ±5 pages for cleaner context.
  - *Whole book mode* retrieves top‑K chunks from pgvector.
- **Animate** from the chat area:
  - Click **Animate** to create an async plugin job.
  - Job progress is logged in chat.
  - On success, the app returns links to the generated `script.py` and rendered `lesson.mp4`.

## Ingestion Scripts
The one‑off ingestion utilities live under `core/services/ingestion`:
- `pdf_ocr.py` – extract OCR text from a PDF to a text file
- `embedding_pipeline.py` – chunk a text file, generate embeddings, and upsert

Example usage:
```bash
python -m core.services.ingestion.pdf_ocr path/to/book.pdf --out totalBook.txt
python -m core.services.ingestion.embedding_pipeline totalBook.txt --source "grade-5-science"
```

## Configuration
See `.env.example` for all settings. Common ones:
- `INFERENCE_PROVIDER` – `sarvam`, `openai-compatible`, or `ollama`
- `INFERENCE_API_KEY` – required for Sarvam/OpenAI-compatible providers
- `INFERENCE_MODEL` – chat model for Sarvam/OpenAI-compatible providers, or the Ollama model name
- `INFERENCE_BASE_URL` – optional OpenAI-compatible base URL, or the local Ollama URL
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` – backward-compatible Ollama aliases; prefer `INFERENCE_*`
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `SEED_DEMO_CATALOG` (default: true) – registers repo-owned demo PDFs from `pdfs/cdc-grade-10`
- `DEMO_CATALOG_DIR` – override the local demo catalogue folder
- `PRECOMPUTE_ON_SELECT` (default: true) – starts OCR/embedding precompute when a catalogue book is selected
- `MULTIMODAL_PAGE_CONTEXT` – attach the current PDF page image to Ollama current-page chat
- `PAGE_IMAGE_MAX_SIDE`, `PAGE_IMAGE_QUALITY` – bound page image size/quality for multimodal context
- `PRECOMPUTE_OCR_ON_UPLOAD` (default: true)
- `PRECOMPUTE_EMBEDDINGS_ON_UPLOAD` (default: true)
- `EMBEDDING_PROVIDER` (`sentence_transformers` or `openai`)
- `EMBEDDING_MODEL_NAME` (default: `all-MiniLM-L6-v2`)
- `ANIMATION_CONTEXT_MAX_CHARS` (default: `9000`)
- `ANIMATION_RENDER_TIMEOUT_SECONDS` (default: `180`)

For Ollama demos, use `qwen3.5:0.8b` as the small local default. It is a compact Qwen
vision-language model. DeepGyan can attach the current PDF page image alongside OCR/text context
for current-page chat when `MULTIMODAL_PAGE_CONTEXT=true`; whole-book retrieval remains text-first.
If you need the smallest text-only Qwen model, use `qwen3:0.6b`; `qwen2.5:0.5b` is
available but older. The smaller `himalaya-ai/himalayagpt-0.5b-it-gguf` nanochat model can be
pulled by Ollama, but current standard Ollama fails to load it because nanochat is not an upstream
llama.cpp architecture yet. Use the patched HimalayaAI llama.cpp fork for that model.

```bash
ollama serve
ollama pull qwen3.5:0.8b
```

Leave `INFERENCE_REASONING_EFFORT` blank for Ollama demos unless you specifically want the model to
spend time producing a hidden reasoning trace.

For hybrid visual/text page context with Qwen:

```bash
MULTIMODAL_PAGE_CONTEXT=true
PAGE_IMAGE_MAX_SIDE=1400
PAGE_IMAGE_QUALITY=80
```

To test the 0.5B nanochat model with DeepGyan before Ollama supports it, run the patched
`HimalayaAI/llama.cpp` `llama-server`, then start the lightweight OpenAI-compatible bridge:

```bash
/path/to/HimalayaAI/llama.cpp/build/bin/llama-server \
  -m /path/to/himalayagpt-0.5b-it.<quant>.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  -c 2048 \
  -ngl 11 \
  --no-warmup

python -m tools.deepgyan_openai_bridge
```

Point DeepGyan at the bridge with:

```bash
INFERENCE_PROVIDER=openai-compatible
INFERENCE_BASE_URL=http://127.0.0.1:8088/v1
INFERENCE_MODEL=himalayagpt-0.5b-it
INFERENCE_API_KEY=local
INFERENCE_MAX_TOKENS=384
INFERENCE_REASONING_EFFORT=
INFERENCE_TIMEOUT_SECONDS=300
MODEL_CONTEXT_WINDOW=2048
SUMMARY_MAX_TOKENS=160
CONTEXT_WINDOW=1
```

## Troubleshooting
- **DB auth errors**: ensure `.env` has `DB_PASSWORD` and the docker service matches it. If you changed it, remove the old volume and restart:
  ```bash
  docker compose -f core/services/storage/docker/docker-compose.yaml down -v
  ./start_services.sh
  ```
- **OCR missing**: verify Tesseract is installed and available on your shell `PATH`.

## Project Structure (high level)
- `dashboard/backend` – FastAPI backend + OCR + embeddings
- `dashboard/frontend` – HTML/CSS/JS app
- `core/models` – Pydantic models aligned to the DB schema
- `core/agents` – prompt + context managers (agent-ready)
- `core/services/ingestion` – OCR + embedding utilities
- `core/services/inference` – AI inference wrapper (model calls + parsing)
- `core/services/plugins` – plugin runtime + Manim animation plugin
- `core/services/storage` – schema + pgvector helpers
- `tests/` – unit tests

---
Made for Nepal’s classrooms, with students in mind.
