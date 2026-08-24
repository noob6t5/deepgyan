import pytest

from core.agents.context_manager import context_char_budget
from core.services.ingestion.chapter_index import (
    Chapter,
    chapter_for_page,
    detect_chapters,
)
from core.services.inference.inference import InferenceService


class _FakeSpan(dict):
    pass


def _page(spans, text=""):
    """Minimal stand-in for a PyMuPDF page."""

    class Page:
        def get_text(self, kind=None):
            if kind == "dict":
                return {
                    "blocks": [
                        {"lines": [{"spans": [dict(s) for s in spans]}]}
                    ]
                }
            return text

    return Page()


class _FakeDoc:
    def __init__(self, pages, toc=None):
        self._pages = pages
        self._toc = toc or []

    def __len__(self):
        return len(self._pages)

    def __iter__(self):
        return iter(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def get_toc(self):
        return self._toc


def _body_page(size=10, chars=400):
    return _page([{"text": "x" * chars, "size": size}], text="x" * chars)


def _heading_page(title, size=20, chars=400):
    return _page(
        [{"text": title, "size": size}, {"text": "y" * chars, "size": 10}],
        text=f"{title}\n" + "y" * chars,
    )


# --- budget -----------------------------------------------------------------


def test_budget_subtracts_output_safety_and_reserved_tokens():
    budget = context_char_budget(
        model_context_window=10_000,
        safety_tokens=200,
        token_char_ratio=3.0,
        output_tokens=1_000,
        reserved_tokens=1_100,
    )
    assert budget == int((10_000 - 1_000 - 200 - 1_100) * 3.0)


def test_budget_is_zero_when_window_is_already_spent():
    assert (
        context_char_budget(
            model_context_window=1_000,
            safety_tokens=200,
            token_char_ratio=3.0,
            output_tokens=1_200,
        )
        == 0
    )


def test_budget_never_returns_negative_after_overhead():
    assert (
        context_char_budget(
            model_context_window=2_000,
            safety_tokens=100,
            token_char_ratio=3.0,
            output_tokens=500,
            overhead_chars=999_999,
        )
        == 0
    )


# --- detection --------------------------------------------------------------


def test_detects_chapters_from_outline():
    doc = _FakeDoc(
        [_body_page() for _ in range(40)],
        toc=[[1, "Unit 1", 1], [1, "Unit 2", 15], [1, "Unit 3", 28]],
    )
    chapters = detect_chapters(doc)
    assert [c.title for c in chapters] == ["Unit 1", "Unit 2", "Unit 3"]
    assert chapters[0].start_page == 0
    assert chapters[0].end_page == 13
    assert chapters[-1].end_page == 39


def test_outline_covering_almost_the_whole_book_is_rejected():
    """A single entry spanning the book is a file list, not a chapter index."""
    doc = _FakeDoc(
        [_body_page() for _ in range(40)],
        toc=[[1, "front", 1], [1, "everything", 3], [1, "back", 39]],
    )
    assert detect_chapters(doc) == []


def test_falls_back_to_font_size_when_outline_is_absent():
    pages = []
    for start in range(3):
        pages.append(_heading_page(f"Heading {start}"))
        pages.extend(_body_page() for _ in range(9))
    chapters = detect_chapters(_FakeDoc(pages))
    assert len(chapters) == 3
    assert chapters[0].title == "Heading 0"
    assert chapters[1].start_page == 10


def test_font_size_ignores_a_second_body_style():
    """A large size carrying most of the text is body copy, not a heading."""
    pages = [_page([{"text": "z" * 5000, "size": 20}], text="z" * 5000) for _ in range(30)]
    assert detect_chapters(_FakeDoc(pages)) == []


def test_falls_back_to_text_markers():
    pages = []
    for index in range(3):
        pages.append(_page([{"text": "t", "size": 10}], text=f"Unit {index + 1}\nbody"))
        pages.extend(
            _page([{"text": "t", "size": 10}], text="body") for _ in range(9)
        )
    chapters = detect_chapters(_FakeDoc(pages))
    assert len(chapters) == 3
    assert chapters[0].title.startswith("Unit 1")


def test_short_leading_sections_merge_into_the_previous_chapter():
    pages = [_heading_page("Cover")]
    pages.append(_heading_page("Real Chapter"))
    pages.extend(_body_page() for _ in range(9))
    pages.append(_heading_page("Second Chapter"))
    pages.extend(_body_page() for _ in range(9))
    pages.append(_heading_page("Third Chapter"))
    pages.extend(_body_page() for _ in range(9))
    chapters = detect_chapters(_FakeDoc(pages))
    assert all(c.page_count >= 3 for c in chapters)
    assert chapters[0].start_page == 0


def test_returns_empty_for_a_book_too_short_to_have_chapters():
    assert detect_chapters(_FakeDoc([_body_page() for _ in range(4)])) == []


def test_detection_survives_a_strategy_that_raises():
    class Exploding(_FakeDoc):
        def get_toc(self):
            raise RuntimeError("corrupt outline")

    pages = []
    for start in range(3):
        pages.append(_heading_page(f"Heading {start}"))
        pages.extend(_body_page() for _ in range(9))
    assert len(detect_chapters(Exploding(pages))) == 3


# --- lookup -----------------------------------------------------------------


def test_chapter_for_page_matches_inclusive_bounds():
    chapters = [Chapter("A", 0, 9), Chapter("B", 10, 19)]
    assert chapter_for_page(chapters, 0).title == "A"
    assert chapter_for_page(chapters, 9).title == "A"
    assert chapter_for_page(chapters, 10).title == "B"
    assert chapter_for_page(chapters, 99) is None
    assert chapter_for_page([], 3) is None


# --- num_ctx ----------------------------------------------------------------


def _capture(service, messages):
    seen = {}

    def transport(path, payload, timeout):
        seen.update(payload)
        return {"message": {"content": "ok"}}

    service.transport = transport
    service.chat_completions(messages)
    return seen


def test_ollama_payload_sets_num_ctx_from_context_window():
    service = InferenceService(
        api_key="",
        api_key_placeholder="placeholder",
        model="qwen3.5:0.8b",
        max_tokens=64,
        temperature=0.2,
        provider="ollama",
        context_window=32768,
    )
    payload = _capture(service, [{"role": "user", "content": "hi"}])
    assert payload["options"]["num_ctx"] == 32768
    assert payload["options"]["num_predict"] == 64


def test_ollama_payload_omits_num_ctx_when_unset():
    service = InferenceService(
        api_key="",
        api_key_placeholder="placeholder",
        model="qwen3.5:0.8b",
        max_tokens=64,
        temperature=0.2,
        provider="ollama",
    )
    payload = _capture(service, [{"role": "user", "content": "hi"}])
    assert "num_ctx" not in payload["options"]


def test_unreadable_titles_become_positional_labels():
    """Legacy-font headings mark real boundaries but decode to noise."""
    pages = []
    for index in range(3):
        pages.append(_heading_page("d]\x03/\x03f]\x03 b]\x03zsf]\x03"))
        pages.extend(_body_page() for _ in range(9))
    chapters = detect_chapters(_FakeDoc(pages))
    assert len(chapters) == 3
    assert all(c.title.startswith("Section ") for c in chapters)


def test_readable_titles_are_preserved():
    doc = _FakeDoc(
        [_body_page() for _ in range(40)],
        toc=[[1, "Unit 1", 1], [1, "Unit 2", 15], [1, "Unit 3", 28]],
    )
    assert [c.title for c in detect_chapters(doc)] == ["Unit 1", "Unit 2", "Unit 3"]


def test_bare_unit_numbers_are_kept_as_titles():
    """The CS textbook heads each unit with just its number."""
    pages = []
    for index in range(3):
        pages.append(_heading_page(str(index + 1)))
        pages.extend(_body_page() for _ in range(9))
    assert [c.title for c in detect_chapters(_FakeDoc(pages))] == ["1", "2", "3"]
