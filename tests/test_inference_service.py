import io
import json
import urllib.error

import pytest

from core.services.inference.inference import InferenceService


def test_build_params_includes_reasoning_effort():
    service = InferenceService(
        api_key="key",
        api_key_placeholder="placeholder",
        model="sarvam-m",
        max_tokens=123,
        temperature=0.5,
        reasoning_effort="medium",
    )
    params = service.build_params([{"role": "user", "content": "hi"}], max_tokens=77)
    assert params["model"] == "sarvam-m"
    assert params["max_tokens"] == 77
    assert params["temperature"] == 0.5
    assert params["reasoning_effort"] == "medium"


def test_extract_think_and_final_parses_tags():
    text = "<think>plan</think>\n<final>Answer.</final>"
    content, thinking = InferenceService.extract_think_and_final(text)
    assert content == "Answer."
    assert thinking == "plan"


def test_extract_think_and_final_unclosed_think():
    text = "<think>Reasoning line 1.\nReasoning line 2.\n\nFinal answer."
    content, thinking = InferenceService.extract_think_and_final(text)
    assert thinking.startswith("Reasoning line 1")
    assert content == "Final answer."


def test_ollama_provider_uses_native_chat_payload():
    calls = []

    def transport(path, payload, timeout):
        calls.append((path, payload, timeout))
        return {"message": {"content": "<think>plan</think><final>Namaste.</final>"}}

    service = InferenceService(
        api_key="",
        api_key_placeholder="placeholder",
        model="test-ollama-model",
        max_tokens=256,
        temperature=0.1,
        provider="ollama",
        base_url="http://localhost:11434",
        timeout_seconds=9,
        transport=transport,
    )

    assert service.is_configured()
    response = service.chat_completions([{"role": "user", "content": "hi"}], max_tokens=77)
    content, reasoning = service.extract_response_payload(response)

    assert content == "Namaste."
    assert reasoning == "plan"
    assert calls == [
        (
            "/api/chat",
            {
                "model": "test-ollama-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 77},
            },
            9,
        )
    ]


def test_ollama_reasoning_effort_enables_thinking_payload():
    calls = []

    def transport(path, payload, timeout):
        calls.append((path, payload, timeout))
        return {"message": {"content": "Final answer.", "thinking": "private notes"}}

    service = InferenceService(
        api_key="",
        api_key_placeholder="placeholder",
        model="test-ollama-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="medium",
        provider="ollama",
        base_url="http://localhost:11434",
        transport=transport,
    )

    response = service.chat_completions([{"role": "user", "content": "hi"}])
    content, reasoning = service.extract_response_payload(response)

    assert content == "Final answer."
    assert reasoning == "private notes"
    assert calls[0][1]["think"] is True
    assert "think" not in calls[0][1]["options"]


def test_ollama_http_error_includes_response_details(monkeypatch):
    def raise_http_error(*args, **kwargs):
        body = json.dumps({"error": "model 'missing-model' not found"}).encode("utf-8")
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    service = InferenceService(
        api_key="",
        api_key_placeholder="placeholder",
        model="missing-model",
        max_tokens=256,
        temperature=0.1,
        provider="ollama",
        base_url="http://localhost:11434",
    )

    with pytest.raises(RuntimeError, match="missing-model.*not found"):
        service.chat_completions([{"role": "user", "content": "hi"}])


def test_openai_compatible_requires_base_url():
    service = InferenceService(
        api_key="key",
        api_key_placeholder="placeholder",
        model="model",
        max_tokens=100,
        temperature=0.2,
        provider="openai-compatible",
    )

    assert not service.is_configured()


def test_openai_compatible_retries_shorter_prompt_on_context_error():
    calls = []

    class FakeCompletions:
        def create(self, **params):
            calls.append(params)
            if len(calls) == 1:
                raise RuntimeError("request exceeds the available context size")
            return "ok"

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    service = InferenceService(
        api_key="key",
        api_key_placeholder="placeholder",
        model="model",
        max_tokens=100,
        temperature=0.2,
        provider="openai-compatible",
        base_url="http://localhost:8088/v1",
    )
    service._client = FakeClient()

    response = service.chat_completions(
        [
            {"role": "system", "content": "x" * 5000},
            {"role": "user", "content": "question"},
        ]
    )

    assert response == "ok"
    assert len(calls) == 2
    assert len(calls[1]["messages"][0]["content"]) < len(calls[0]["messages"][0]["content"])
    assert calls[1]["messages"][1]["content"] == "question"
