from fastapi.testclient import TestClient

from tools.deepgyan_openai_bridge import (
    BridgeSettings,
    create_app,
    nanochat_prompt,
)


class FakeLlamaClient:
    def __init__(self):
        self.payloads = []

    def health(self):
        return True

    def complete(self, payload):
        self.payloads.append(payload)
        return {
            "content": "नमस्ते!",
            "tokens_evaluated": 12,
            "tokens_predicted": 4,
        }


def test_nanochat_prompt_renders_openai_messages():
    prompt = nanochat_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Namaste"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": [{"type": "text", "text": "फेरि"}]},
        ]
    )

    assert prompt == (
        "<|user_start|>You are helpful.<|user_end|>"
        "<|user_start|>Namaste<|user_end|>"
        "<|assistant_start|>Hello<|assistant_end|>"
        "<|user_start|>फेरि<|user_end|>"
        "<|assistant_start|>"
    )


def test_chat_completions_proxies_to_llama_completion():
    fake = FakeLlamaClient()
    settings = BridgeSettings(
        llama_base_url="http://127.0.0.1:8081",
        model_alias="deepgyan-0.5b",
        timeout_seconds=5,
    )
    client = TestClient(create_app(settings=settings, client=fake))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepgyan-0.5b",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 32,
            "temperature": 0.1,
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "deepgyan-0.5b"
    assert body["choices"][0]["message"]["content"] == "नमस्ते!"
    assert body["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert fake.payloads == [
        {
            "prompt": "<|user_start|>Say hi<|user_end|><|assistant_start|>",
            "n_predict": 32,
            "temperature": 0.1,
            "repeat_penalty": 1.1,
            "parse_special": True,
            "cache_prompt": True,
            "stop": [
                "<|user_start|>",
                "<|user_end|>",
                "<|assistant_start|>",
                "<|assistant_end|>",
            ],
        }
    ]


def test_chat_completions_rejects_streaming():
    client = TestClient(
        create_app(settings=BridgeSettings(), client=FakeLlamaClient())
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepgyan-0.5b",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
