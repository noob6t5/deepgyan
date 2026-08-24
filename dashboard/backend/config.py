"""Dashboard configuration for DeepGyan."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv:
    env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = str(FRONTEND_DIR / "static")
ASSETS_DIR = str(FRONTEND_DIR / "assets")
TEMPLATES_DIR = str(FRONTEND_DIR)
UPLOAD_DIR = str(BASE_DIR / "uploads")
DATA_DIR = str(BASE_DIR / "data")
PLUGIN_ARTIFACTS_DIR = str(Path(DATA_DIR) / "plugin_artifacts")


def _repo_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


DEMO_CATALOG_DIR = _repo_path(
    os.getenv("DEMO_CATALOG_DIR", "pdfs/cdc-grade-10")
)
SEED_DEMO_CATALOG = os.getenv("SEED_DEMO_CATALOG", "true").lower() in {
    "1",
    "true",
    "yes",
}

GLOBAL_CONTEXT_FILE = str(Path(DATA_DIR) / "context.txt")
ENV_CONTEXT_FILE = str(Path(DATA_DIR) / "surrounding_context.txt")

INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "sarvam").strip().lower()
API_KEY_PLACEHOLDER = "YOUR_SARVAM_API_KEY"
DEFAULT_OLLAMA_MODEL = "qwen3.5:0.8b"
_API_KEY_PLACEHOLDERS = {API_KEY_PLACEHOLDER, "YOUR_API_KEY", "your_sarvam_key"}


def _clean_api_key(value: str | None) -> str:
    value = (value or "").strip()
    return "" if not value or value in _API_KEY_PLACEHOLDERS else value


_explicit_inference_api_key = _clean_api_key(os.getenv("INFERENCE_API_KEY"))
if _explicit_inference_api_key:
    INFERENCE_API_KEY = _explicit_inference_api_key
elif INFERENCE_PROVIDER in {"sarvam", "sarvam-ai"}:
    INFERENCE_API_KEY = _clean_api_key(os.getenv("SARVAMAI_KEY"))
elif INFERENCE_PROVIDER in {"openai-compatible", "openai-compat"}:
    INFERENCE_API_KEY = (
        _clean_api_key(os.getenv("OPENAI_COMPAT_API_KEY"))
        or _clean_api_key(os.getenv("OPENAI_API_KEY"))
    )
elif INFERENCE_PROVIDER == "openai":
    INFERENCE_API_KEY = _clean_api_key(os.getenv("OPENAI_API_KEY"))
else:
    INFERENCE_API_KEY = ""
_inference_base_url = os.getenv("INFERENCE_BASE_URL", "").strip()
_openai_compat_base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
if INFERENCE_PROVIDER == "ollama":
    INFERENCE_BASE_URL = _inference_base_url or _ollama_base_url or "http://localhost:11434"
else:
    INFERENCE_BASE_URL = _inference_base_url or _openai_compat_base_url or None

_inference_model = os.getenv("INFERENCE_MODEL", "").strip()
_sarvam_model = os.getenv("SARVAM_MODEL", "").strip()
_ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
if INFERENCE_PROVIDER == "ollama":
    INFERENCE_MODEL = _inference_model or _ollama_model or DEFAULT_OLLAMA_MODEL
else:
    INFERENCE_MODEL = _inference_model or _sarvam_model or "sarvam-m"
INFERENCE_MAX_TOKENS = int(
    os.getenv("INFERENCE_MAX_TOKENS", os.getenv("SARVAM_MAX_TOKENS", "1200"))
)
if INFERENCE_PROVIDER == "ollama":
    _reasoning_effort = os.getenv("INFERENCE_REASONING_EFFORT", "").strip()
else:
    _reasoning_effort = os.getenv(
        "INFERENCE_REASONING_EFFORT", os.getenv("SARVAM_REASONING_EFFORT", "medium")
    ).strip()
INFERENCE_REASONING_EFFORT = _reasoning_effort if _reasoning_effort else None
_default_temp = "0.5" if INFERENCE_REASONING_EFFORT else "0.2"
_temperature = (
    os.getenv("INFERENCE_TEMPERATURE", "").strip()
    or os.getenv("SARVAM_TEMPERATURE", "").strip()
    or _default_temp
)
INFERENCE_TEMPERATURE = float(_temperature)
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "120"))

# Backward-compatible names for older imports/tests.
SARVAMAI_KEY = INFERENCE_API_KEY
SARVAM_MODEL = INFERENCE_MODEL
SARVAM_MAX_TOKENS = INFERENCE_MAX_TOKENS
SARVAM_REASONING_EFFORT = INFERENCE_REASONING_EFFORT
SARVAM_TEMPERATURE = INFERENCE_TEMPERATURE
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "7192"))
CONTEXT_SAFETY_TOKENS = int(os.getenv("CONTEXT_SAFETY_TOKENS", "200"))
CONTEXT_TOKEN_CHAR_RATIO = float(os.getenv("CONTEXT_TOKEN_CHAR_RATIO", "3.0"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "800"))

OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "40"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_FALLBACK_MESSAGE = os.getenv("OCR_FALLBACK_MESSAGE", "OCR failed on this page.")
OCR_SEMAPHORE_LIMIT = int(os.getenv("OCR_SEMAPHORE_LIMIT", "4"))

CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "5"))
ANALYSIS_CHUNK_SIZE = int(os.getenv("ANALYSIS_CHUNK_SIZE", "10"))

PRECOMPUTE_OCR_ON_UPLOAD = os.getenv("PRECOMPUTE_OCR_ON_UPLOAD", "true").lower() in {"1", "true", "yes"}
PRECOMPUTE_EMBEDDINGS_ON_UPLOAD = os.getenv("PRECOMPUTE_EMBEDDINGS_ON_UPLOAD", "true").lower() in {"1", "true", "yes"}
PRECOMPUTE_ON_SELECT = os.getenv("PRECOMPUTE_ON_SELECT", "true").lower() in {"1", "true", "yes"}
EMBEDDING_SOURCE_PREFIX = os.getenv("EMBEDDING_SOURCE_PREFIX", "upload")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
EMBEDDING_WARMUP = os.getenv("EMBEDDING_WARMUP", "false").lower() in {"1", "true", "yes"}

# Chapter-level context: when the current chapter fits the model window, send
# the whole chapter instead of a +/- CONTEXT_WINDOW page slice.
CHAPTER_CONTEXT = os.getenv("CHAPTER_CONTEXT", "true").lower() in {"1", "true", "yes"}
# Chars assumed for a page whose text is not cached yet and that carries no
# native text (i.e. OCR-only), used only for the fits-the-window estimate.
CHAPTER_OCR_PAGE_CHAR_ESTIMATE = int(os.getenv("CHAPTER_OCR_PAGE_CHAR_ESTIMATE", "1800"))
# Tokens reserved for an attached page image when multimodal context is on.
PAGE_IMAGE_TOKEN_ESTIMATE = int(os.getenv("PAGE_IMAGE_TOKEN_ESTIMATE", "1100"))

MULTIMODAL_PAGE_CONTEXT = os.getenv("MULTIMODAL_PAGE_CONTEXT", "false").lower() in {
    "1",
    "true",
    "yes",
}
PAGE_IMAGE_MAX_SIDE = int(os.getenv("PAGE_IMAGE_MAX_SIDE", "1400"))
PAGE_IMAGE_QUALITY = int(os.getenv("PAGE_IMAGE_QUALITY", "80"))

ANIMATION_CONTEXT_MAX_CHARS = int(os.getenv("ANIMATION_CONTEXT_MAX_CHARS", "9000"))
ANIMATION_RENDER_TIMEOUT_SECONDS = int(os.getenv("ANIMATION_RENDER_TIMEOUT_SECONDS", "180"))


SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SSE_MEDIA_TYPE = "text/event-stream"

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

DEFAULT_ANALYSIS_MESSAGE = "No analysis has been generated yet."
API_EMPTY_RESPONSE_MESSAGE = "No response content returned by the API."

ERR_INFERENCE_NOT_CONFIGURED = (
    "Inference provider not configured. Set INFERENCE_PROVIDER plus the required "
    "model/API settings, or use INFERENCE_PROVIDER=ollama with Ollama running locally."
)
ERR_NO_PDF_UPLOADED = "No PDF uploaded yet."
ERR_NO_CONTEXT = "No analysis context found. Generate it first."


def validate_config() -> None:
    if not Path(TEMPLATES_DIR).exists():
        raise RuntimeError(f"Templates directory not found: {TEMPLATES_DIR}")
    if not Path(STATIC_DIR).exists():
        raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLUGIN_ARTIFACTS_DIR, exist_ok=True)
