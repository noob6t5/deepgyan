import pytest

pytest.importorskip("jinja2")
from fastapi.testclient import TestClient

from fastapi.testclient import TestClient

from dashboard.backend import app as dashboard_app


def test_dashboard_routes_exist():
    paths = {route.path for route in dashboard_app.app.routes}
    assert "/" in paths
    assert "/api/upload" in paths
    assert "/api/analyze_env" in paths
    assert "/api/analyze_global" in paths
    assert "/api/ask" in paths
    assert "/api/plugins/jobs" in paths
    assert "/api/plugins/jobs/{job_id}" in paths
    assert "/api/plugins/jobs/{job_id}/artifacts/{artifact_type}" in paths
    assert "/static" in paths
    assert "/uploads" in paths


def test_extract_response_content_fallback():
    class Message:
        def __init__(self, content, reasoning_content):
            self.content = content
            self.reasoning_content = reasoning_content

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, message):
            self.choices = [Choice(message)]

    msg = Message(content="", reasoning_content="fallback")
    resp = Response(msg)

    content, reasoning = dashboard_app._extract_response_payload(resp)
    assert content == ""
    assert reasoning == "fallback"


def test_strip_think_tags():
    text = "<think>secret</think>\n<final>Answer.</final>"
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert content == "Answer."
    assert reasoning == "secret"


def test_unclosed_think_block():
    text = "<think>Reasoning line 1.\nReasoning line 2.\n\nFinal answer."
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert reasoning.startswith("Reasoning line 1")
    assert content == "Final answer."


def test_heuristic_reasoning_split():
    text = "Okay, the user is asking about X.\n\nThe answer is Y."
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert reasoning == ""
    assert content == text


def test_create_plugin_job_requires_query():
    client = TestClient(dashboard_app.app)
    response = client.post("/api/plugins/jobs", json={"plugin_id": "manim_video", "query": ""})
    assert response.status_code == 400
    assert "query is required" in response.text


def test_create_plugin_job_rejects_unknown_plugin():
    client = TestClient(dashboard_app.app)
    response = client.post("/api/plugins/jobs", json={"plugin_id": "does_not_exist", "query": "animate this"})
    assert response.status_code == 400
    assert "Unknown plugin" in response.text


def test_seed_demo_catalog_books_registers_manifest_pdfs(monkeypatch, tmp_path):
    fitz = pytest.importorskip("fitz")
    catalog_dir = tmp_path / "catalog"
    upload_dir = tmp_path / "uploads"
    catalog_dir.mkdir()
    upload_dir.mkdir()

    pdf_path = catalog_dir / "sample-grade-10.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    (catalog_dir / "manifest.json").write_text(
        '{"books":[{"filename":"sample-grade-10.pdf"}]}',
        encoding="utf-8",
    )

    upserts = []

    def fake_upsert(filename, file_hash, total_pages):
        upserts.append((filename, file_hash, total_pages))
        return "book-id"

    monkeypatch.setattr(dashboard_app, "SEED_DEMO_CATALOG", True)
    monkeypatch.setattr(dashboard_app, "DEMO_CATALOG_DIR", str(catalog_dir))
    monkeypatch.setattr(dashboard_app, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(dashboard_app, "_upsert_book", fake_upsert)

    seeded = dashboard_app._seed_demo_catalog_books()

    assert seeded == 1
    assert (upload_dir / "sample-grade-10.pdf").exists()
    assert upserts[0][0] == "sample-grade-10.pdf"
    assert upserts[0][2] == 1


def test_select_book_starts_precompute_when_chunks_missing(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "catalog-book.pdf").write_bytes(b"%PDF-1.4\n")

    created = []

    async def fake_precompute():
        return None

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        return None

    old_state = dashboard_app.global_pdf_data.copy()
    monkeypatch.setattr(dashboard_app, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(
        dashboard_app,
        "_get_book_by_id",
        lambda book_id: {
            "id": book_id,
            "filename": "catalog-book.pdf",
            "total_pages": 12,
        },
    )
    monkeypatch.setattr(dashboard_app, "_count_text_chunks", lambda source: 0)
    monkeypatch.setattr(dashboard_app, "_precompute_ocr_and_embeddings", fake_precompute)
    monkeypatch.setattr(dashboard_app.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(dashboard_app, "PRECOMPUTE_ON_SELECT", True)
    monkeypatch.setattr(dashboard_app, "PRECOMPUTE_OCR_ON_UPLOAD", True)
    monkeypatch.setattr(dashboard_app, "PRECOMPUTE_EMBEDDINGS_ON_UPLOAD", True)

    try:
        client = TestClient(dashboard_app.app)
        response = client.post("/api/books/select", json={"book_id": "book-1"})
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert response.json()["precompute_started"] is True
    assert len(created) == 1


def test_select_book_skips_precompute_when_chunks_exist(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "catalog-book.pdf").write_bytes(b"%PDF-1.4\n")

    created = []
    old_state = dashboard_app.global_pdf_data.copy()
    monkeypatch.setattr(dashboard_app, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(
        dashboard_app,
        "_get_book_by_id",
        lambda book_id: {
            "id": book_id,
            "filename": "catalog-book.pdf",
            "total_pages": 12,
        },
    )
    monkeypatch.setattr(dashboard_app, "_count_text_chunks", lambda source: 5)
    monkeypatch.setattr(
        dashboard_app.asyncio, "create_task", lambda coro: created.append(coro)
    )

    try:
        client = TestClient(dashboard_app.app)
        response = client.post("/api/books/select", json={"book_id": "book-1"})
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert response.json()["precompute_started"] is False
    assert created == []


def test_ask_returns_config_error_before_env_context_build(monkeypatch):
    class DummyInference:
        def is_configured(self) -> bool:
            return False

    def _should_not_run(_page_index: int):
        raise AssertionError("Environment context should not be built when API key is missing.")

    old_state = dashboard_app.global_pdf_data.copy()
    dashboard_app.global_pdf_data.update(
        {
            "filename": "test.pdf",
            "filepath": "dashboard/uploads/test.pdf",
            "total_pages": 10,
            "pages": {},
            "book_id": None,
        }
    )

    monkeypatch.setattr(dashboard_app, "inference_service", DummyInference())
    monkeypatch.setattr(dashboard_app, "_build_env_context", _should_not_run)

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/ask",
            json={"query": "hello", "mode": "environment", "current_page": 1},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert dashboard_app.ERR_INFERENCE_NOT_CONFIGURED in response.text


def test_ask_attaches_page_image_for_ollama_hybrid_context(monkeypatch):
    captured = {}

    class DummyInference:
        def is_configured(self) -> bool:
            return True

        def chat_completions(self, messages):
            captured["messages"] = messages
            return object()

        def extract_response_payload(self, response):
            return "Visual answer.", ""

    async def fake_build_env_context(current_page):
        return "Structured page context.", "Raw page context."

    old_state = dashboard_app.global_pdf_data.copy()
    dashboard_app.global_pdf_data.update(
        {
            "filename": "test.pdf",
            "filepath": "dashboard/uploads/test.pdf",
            "total_pages": 10,
            "pages": {},
            "page_images": {},
            "book_id": None,
        }
    )

    monkeypatch.setattr(dashboard_app, "inference_service", DummyInference())
    monkeypatch.setattr(dashboard_app, "MULTIMODAL_PAGE_CONTEXT", True)
    monkeypatch.setattr(dashboard_app, "INFERENCE_PROVIDER", "ollama")
    monkeypatch.setattr(dashboard_app, "_build_env_context", fake_build_env_context)
    monkeypatch.setattr(dashboard_app, "_render_page_image_base64", lambda page: "base64-page")

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/ask",
            json={"query": "explain the diagram", "mode": "environment", "current_page": 2},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert "Visual answer." in response.text
    user_message = captured["messages"][1]
    assert user_message["images"] == ["base64-page"]
    assert "current textbook page image is attached" in user_message["content"]


@pytest.mark.asyncio
async def test_generate_structured_context_falls_back_to_raw_text(monkeypatch):
    class FailingContextManager:
        async def build_structured_context(self, raw_text):
            raise RuntimeError("request exceeds the available context size")

    monkeypatch.setattr(dashboard_app, "context_manager", FailingContextManager())

    raw_text = "--- Page 10 ---\nThroughput is the amount of data sent or received."
    structured = await dashboard_app._generate_structured_context(
        raw_text,
        label="current_page_window",
    )

    assert structured == raw_text


@pytest.mark.asyncio
async def test_generate_structured_context_rejects_prompt_echo(monkeypatch):
    class EchoingContextManager:
        async def build_structured_context(self, raw_text):
            return "Provide the final answer only."

    monkeypatch.setattr(dashboard_app, "context_manager", EchoingContextManager())

    raw_text = "--- Page 10 ---\nThroughput is the actual amount of data sent or received."
    structured = await dashboard_app._generate_structured_context(
        raw_text,
        label="current_page_window",
    )

    assert structured == raw_text
