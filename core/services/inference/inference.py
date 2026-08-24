from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable


_SARVAM_OPENAI_BASE_URL = "https://api.sarvam.ai/v1"
logger = logging.getLogger(__name__)


@dataclass
class _Message:
    content: str
    reasoning_content: str | None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _ChatCompletionLike:
    choices: list[_Choice]


@dataclass
class InferenceService:
    api_key: str
    api_key_placeholder: str
    model: str
    max_tokens: int
    temperature: float
    reasoning_effort: str | None = None
    provider: str = "sarvam"
    base_url: str | None = None
    timeout_seconds: float = 120.0
    transport: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self._client = None
        self._provider = self.provider.strip().lower().replace("_", "-")
        if self._provider in {"sarvam", "sarvam-ai"}:
            self._provider = "sarvam"
            self.base_url = self.base_url or _SARVAM_OPENAI_BASE_URL
        elif self._provider in {"openai-compatible", "openai-compat"}:
            self._provider = "openai-compatible"
            if not self.base_url:
                return
        elif self._provider == "openai":
            self._provider = "openai"
        elif self._provider == "ollama":
            self.base_url = (self.base_url or "http://localhost:11434").rstrip("/")
            return
        else:
            return

        if not self.api_key or self.api_key == self.api_key_placeholder:
            return

        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        except ImportError:
            self._client = None

    @property
    def client(self):
        return self._client

    def is_configured(self) -> bool:
        if self._provider == "ollama":
            return bool(self.model and self.base_url)
        return self._client is not None

    def build_params(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def chat_completions(self, messages: list[dict], max_tokens: int | None = None):
        if self._provider == "ollama":
            return self._ollama_chat_completions(messages, max_tokens=max_tokens)
        if not self._client:
            raise RuntimeError(f"{self.provider_label} inference not configured")
        request_messages = messages
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self._client.chat.completions.create(
                    **self.build_params(request_messages, max_tokens=max_tokens)
                )
            except Exception as exc:
                last_error = exc
                if not self._is_context_size_error(exc):
                    raise
                next_messages = self._shrink_largest_message(request_messages)
                if next_messages == request_messages:
                    raise
                request_messages = next_messages
                logger.warning(
                    "%s prompt exceeded model context; retrying with shorter context (attempt %s/3)",
                    self.provider_label,
                    attempt + 2,
                )
        if last_error:
            raise last_error
        raise RuntimeError(f"{self.provider_label} inference failed")

    @property
    def provider_label(self) -> str:
        if self._provider == "sarvam":
            return "Sarvam"
        if self._provider == "ollama":
            return "Ollama"
        if self._provider == "openai-compatible":
            return "OpenAI-compatible"
        return self._provider or "Inference"

    def _ollama_chat_completions(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> _ChatCompletionLike:
        if not self.is_configured():
            raise RuntimeError("Ollama inference not configured")

        options: dict[str, Any] = {"temperature": self.temperature}
        token_limit = max_tokens if max_tokens is not None else self.max_tokens
        if token_limit:
            options["num_predict"] = token_limit
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": bool(self.reasoning_effort),
            "options": options,
        }
        if self.transport is not None:
            data = self.transport("/api/chat", payload, self.timeout_seconds)
        else:
            data = self._post_ollama("/api/chat", payload)
        message = data.get("message") or {}
        text = (message.get("content") or data.get("response") or "").strip()
        thinking = (message.get("thinking") or data.get("thinking") or "").strip() or None
        return _ChatCompletionLike(
            choices=[_Choice(message=_Message(content=text, reasoning_content=thinking))]
        )

    def _post_ollama(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"{str(self.base_url).rstrip('/')}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            if details:
                try:
                    details = json.loads(details).get("error", details)
                except json.JSONDecodeError:
                    pass
            suffix = f" Ollama returned: {details}." if details else ""
            raise RuntimeError(
                f"Ollama request failed at {url}.{suffix} Start Ollama with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed at {url}. Start Ollama with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from exc

    @staticmethod
    def _is_context_size_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "context size" in text
            or "context length" in text
            or "maximum context" in text
            or "exceeds the available context" in text
        )

    @staticmethod
    def _shrink_largest_message(messages: list[dict]) -> list[dict]:
        longest_index = -1
        longest_length = 0
        for index, message in enumerate(messages):
            content = message.get("content", "")
            if isinstance(content, str) and len(content) > longest_length:
                longest_index = index
                longest_length = len(content)

        if longest_index < 0 or longest_length <= 900:
            return messages

        shortened = max(900, int(longest_length * 0.55))
        next_messages = [dict(message) for message in messages]
        content = str(next_messages[longest_index].get("content", ""))
        next_messages[longest_index]["content"] = (
            content[:shortened].rstrip()
            + "\n\n[Context shortened locally to fit the model window.]"
        )
        return next_messages

    @staticmethod
    def extract_think_and_final(text: str) -> tuple[str, str]:
        """Extract optional <think> and final content."""
        if not text:
            return "", ""

        import re

        lower = text.lower()
        think_text = ""
        final_text = text

        if "<think>" in lower:
            if "</think>" in lower:
                think_texts = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
                think_text = "\n\n".join(t.strip() for t in think_texts if t.strip())
                final_text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
            else:
                # Fallback: treat first paragraph after <think> as reasoning if present.
                after = re.split(r"<think>", text, flags=re.IGNORECASE, maxsplit=1)[-1]
                parts = re.split(r"\n\s*\n", after, maxsplit=1)
                if len(parts) > 1:
                    think_text = parts[0].strip()
                    final_text = parts[1].strip()
                else:
                    final_text = after.strip()
                    think_text = ""

        final_match = re.search(r"<final>(.*?)</final>", final_text, flags=re.IGNORECASE | re.DOTALL)
        if not final_match:
            final_match = re.search(r"<answer>(.*?)</answer>", final_text, flags=re.IGNORECASE | re.DOTALL)

        if final_match:
            final_text = final_match.group(1).strip()
        else:
            final_text = re.sub(r"</?(final|answer)>", "", final_text, flags=re.IGNORECASE).strip()

        final_text = final_text.replace("<think>", "").replace("</think>", "").strip()
        return final_text, think_text

    def extract_response_payload(self, response) -> tuple[str, str]:
        """Extract answer content and any reasoning content without exposing it."""
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None)
        reasoning = reasoning.strip() if reasoning else ""

        content, think_text = self.extract_think_and_final(content)
        if think_text:
            reasoning = reasoning or think_text

        return content, reasoning
