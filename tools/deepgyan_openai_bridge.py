"""OpenAI-compatible shim for a local llama.cpp server.

This is intentionally small: it exposes the one endpoint DeepGyan needs
(`/v1/chat/completions`) and forwards requests to a patched llama.cpp
`llama-server` `/completion` endpoint. It is useful for nanochat GGUF models
that run in HimalayaAI's llama.cpp fork but not in stock Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


DEFAULT_MODEL_ALIAS = "himalayagpt-0.5b-it"
DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8081"
NANOCHAT_STOP = [
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]


@dataclass(frozen=True)
class BridgeSettings:
    llama_base_url: str = DEFAULT_LLAMA_BASE_URL
    model_alias: str = DEFAULT_MODEL_ALIAS
    timeout_seconds: float = 300.0
    default_max_tokens: int = 256
    default_temperature: float = 0.2
    default_repeat_penalty: float = 1.1

    @classmethod
    def from_env(cls) -> "BridgeSettings":
        return cls(
            llama_base_url=os.getenv(
                "DEEPGYAN_LLAMA_BASE_URL", DEFAULT_LLAMA_BASE_URL
            ).rstrip("/"),
            model_alias=os.getenv("DEEPGYAN_LLAMA_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            timeout_seconds=float(
                os.getenv("DEEPGYAN_LLAMA_TIMEOUT_SECONDS", "300")
            ),
            default_max_tokens=int(
                os.getenv("DEEPGYAN_LLAMA_MAX_TOKENS", "256")
            ),
            default_temperature=float(
                os.getenv("DEEPGYAN_LLAMA_TEMPERATURE", "0.2")
            ),
            default_repeat_penalty=float(
                os.getenv("DEEPGYAN_LLAMA_REPEAT_PENALTY", "1.1")
            ),
        )


class LlamaCompletionClient:
    """Tiny blocking client for llama.cpp server's `/completion` route."""

    def __init__(self, settings: BridgeSettings) -> None:
        self._settings = settings

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self._settings.llama_base_url}/health",
                timeout=min(self._settings.timeout_seconds, 5),
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._settings.llama_base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._settings.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"llama-server returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "could not reach llama-server at "
                f"{self._settings.llama_base_url}: {exc.reason}"
            ) from exc


def _coerce_text(value: Any) -> str:
    """Extract text from OpenAI-style string or content-block arrays."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in {None, "text"}:
                    parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return "" if value is None else str(value)


def nanochat_prompt(messages: list[dict[str, Any]]) -> str:
    """Render OpenAI chat messages with the GGUF's nanochat template."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").lower()
        content = _coerce_text(message.get("content")).strip()
        if not content:
            continue
        if role == "assistant":
            parts.append(f"<|assistant_start|>{content}<|assistant_end|>")
        elif role == "tool":
            parts.append(f"<|user_start|>[Tool result]\n{content}<|user_end|>")
        else:
            parts.append(f"<|user_start|>{content}<|user_end|>")
    parts.append("<|assistant_start|>")
    return "".join(parts)


def _completion_payload(
    body: dict[str, Any], settings: BridgeSettings
) -> dict[str, Any]:
    max_tokens = body.get("max_tokens", settings.default_max_tokens)
    temperature = body.get("temperature", settings.default_temperature)
    payload: dict[str, Any] = {
        "prompt": nanochat_prompt(list(body.get("messages") or [])),
        "n_predict": int(max_tokens or settings.default_max_tokens),
        "temperature": float(temperature),
        "repeat_penalty": float(
            body.get("repeat_penalty", settings.default_repeat_penalty)
        ),
        "parse_special": True,
        "cache_prompt": True,
        "stop": NANOCHAT_STOP,
    }
    if body.get("top_p") is not None:
        payload["top_p"] = float(body["top_p"])
    if body.get("seed") is not None:
        payload["seed"] = int(body["seed"])
    return payload


def _chat_response(
    body: dict[str, Any],
    completion: dict[str, Any],
    settings: BridgeSettings,
) -> dict[str, Any]:
    model = str(body.get("model") or settings.model_alias)
    text = str(completion.get("content") or "").strip()
    prompt_tokens = int(completion.get("tokens_evaluated", 0) or 0)
    completion_tokens = int(completion.get("tokens_predicted", 0) or 0)
    return {
        "id": f"chatcmpl-deepgyan-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_app(
    *,
    settings: BridgeSettings | None = None,
    client: LlamaCompletionClient | None = None,
) -> FastAPI:
    settings = settings or BridgeSettings.from_env()
    client = client or LlamaCompletionClient(settings)
    app = FastAPI(title="DeepGyan Local OpenAI Bridge")

    @app.get("/health")
    def health() -> dict[str, Any]:
        upstream_ready = client.health()
        return {
            "ok": upstream_ready,
            "upstream": settings.llama_base_url,
            "model": settings.model_alias,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "himalaya-ai",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported by the lightweight bridge",
            )
        if not body.get("messages"):
            raise HTTPException(status_code=400, detail="messages are required")
        try:
            completion = client.complete(_completion_payload(body, settings))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(_chat_response(body, completion, settings))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DEEPGYAN_OPENAI_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("DEEPGYAN_OPENAI_BRIDGE_PORT", "8088"))
    uvicorn.run("tools.deepgyan_openai_bridge:app", host=host, port=port)
