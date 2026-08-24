from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from core.agents.prompt_manager import PromptManager


def context_char_budget(
    *,
    model_context_window: int,
    safety_tokens: int,
    token_char_ratio: float,
    output_tokens: int,
    reserved_tokens: int = 0,
    overhead_chars: int = 0,
) -> int:
    """Chars of source text that still fit the model window.

    `reserved_tokens` covers anything already claimed in the window that is
    not source text -- today that is the attached page image under
    multimodal context. `overhead_chars` covers the prompt scaffolding the
    context gets wrapped in. Returns 0 when nothing fits, which callers read
    as "this range is too big".
    """
    available_tokens = model_context_window - output_tokens - safety_tokens - reserved_tokens
    if available_tokens <= 0:
        return 0
    return max(0, int(available_tokens * token_char_ratio) - overhead_chars)


class ContextManager:
    """Builds structured context blocks for tutoring."""

    def __init__(
        self,
        inference_service,
        model_context_window: int,
        safety_tokens: int,
        token_char_ratio: float,
        summary_max_tokens: int,
    ):
        self._inference = inference_service
        self._extract_response = inference_service.extract_response_payload
        self._context_window = model_context_window
        self._safety_tokens = safety_tokens
        self._token_char_ratio = token_char_ratio
        self._summary_max_tokens = summary_max_tokens

    def _truncate_raw_text(self, raw_text: str) -> str:
        available_tokens = self._context_window - self._inference.max_tokens - self._safety_tokens
        available_tokens = max(256, available_tokens)
        max_chars = int(available_tokens * self._token_char_ratio)
        max_chars = min(max_chars, 12000)
        if len(raw_text) > max_chars:
            return raw_text[:max_chars]
        return raw_text

    async def build_structured_context(self, raw_text: str) -> str:
        raw_text = raw_text.strip()
        if not raw_text:
            return ""
        raw_text = self._truncate_raw_text(raw_text)
        prompt = PromptManager.env_summary_prompt(raw_text)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
            self._summary_max_tokens,
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""

    async def build_global_chunk_summary(
        self,
        raw_text: str,
        page_start: int,
        page_end: int,
    ) -> str:
        raw_text = raw_text.strip()
        if not raw_text:
            return ""
        raw_text = self._truncate_raw_text(raw_text)
        prompt = PromptManager.global_summary_prompt(raw_text, page_start, page_end)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
            self._summary_max_tokens,
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""
