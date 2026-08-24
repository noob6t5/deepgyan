"""Chapter boundary detection for textbook PDFs.

Whole-chapter context is only useful if we can find the chapters. The CDC
textbooks in `pdfs/` show the three cases this has to survive:

* `optional-mathematics-grade-10` ships a PDF outline, but it is the print
  shop's file list (`unit 1-5 final`, `unit 9 new layout`) — one entry covers
  five units, so the outline is present but not a chapter index;
* `nepali-grade-10` has no outline at all, and its body text is a legacy
  Preeti-mapped font, so nothing text-based can be matched against;
* `computer-science-grade-10-english` has no outline either, but does start
  each unit with a plain `Unit N` heading.

So detection is a cascade — outline, then heading font size, then a text
marker — and every leg is validated the same way before it is accepted
(`_plausible`). Font size is the leg that carries the Nepali book: glyph
sizes survive a broken encoding even when the decoded characters are junk.

`detect_chapters` returns [] when nothing is trustworthy; callers treat that
as "no chapter structure" and fall back to the page window rather than
guessing at boundaries.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

# A chapter shorter than this is a cover, a colophon, or a heading that the
# font pass caught twice — not a unit worth loading as context.
MIN_CHAPTER_PAGES = 3
# Below this many chapters the detection is more likely noise than structure.
MIN_CHAPTERS = 3

_TEXT_MARKERS = (
    r"^\s*(?:Unit|UNIT|Chapter|CHAPTER|Lesson|LESSON)\s+[\dIVXivx]+",
    r"अध्याय\s*[\d०-९]+",
    r"पाठ\s*[\d०-९]+",
    r"^\s*इकाइ\s*[\d०-९]+",
)


@dataclass(frozen=True)
class Chapter:
    """One chapter, as a 0-indexed inclusive page range."""

    title: str
    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1

    def contains(self, page_index: int) -> bool:
        return self.start_page <= page_index <= self.end_page


def _plausible(starts: list[tuple[int, str]], total_pages: int) -> bool:
    """Reject a candidate set that does not look like a chapter index."""
    if len(starts) < MIN_CHAPTERS:
        return False
    pages = [page for page, _ in starts]
    if pages != sorted(pages) or len(set(pages)) != len(pages):
        return False
    if pages[0] < 0 or pages[-1] >= total_pages:
        return False
    # One entry covering most of the book means the boundaries are wrong,
    # which is exactly how the opt-math outline fails.
    gaps = [b - a for a, b in zip(pages, pages[1:])] + [total_pages - pages[-1]]
    return max(gaps) <= total_pages * 0.6


def _readable_title(title: str) -> bool:
    """Whether a detected title is worth showing to the model.

    Books set in a legacy non-Unicode font (the Nepali textbook uses a Preeti
    mapping) decode to symbol soup. The heading is still a correct *boundary*
    -- glyph size does not depend on encoding -- but the string itself is
    noise, and it would otherwise be pasted into the prompt as a chapter name.
    """
    stripped = title.strip()
    if not stripped:
        return False
    meaningful = sum(1 for char in stripped if char.isalnum() or char.isspace())
    return meaningful / len(stripped) >= 0.6


def _to_chapters(starts: list[tuple[int, str]], total_pages: int) -> list[Chapter]:
    """Turn ordered (page, title) starts into contiguous page ranges."""
    chapters: list[Chapter] = []
    bounds = [page for page, _ in starts] + [total_pages]
    for index, (page, title) in enumerate(starts):
        end = bounds[index + 1] - 1
        if end < page:
            continue
        label = title.strip() if _readable_title(title) else f"Section {index + 1}"
        chapters.append(Chapter(title=label, start_page=page, end_page=end))
    return chapters


def _merge_short(chapters: list[Chapter]) -> list[Chapter]:
    """Fold runs shorter than MIN_CHAPTER_PAGES into the previous chapter.

    Front matter and stray exercise headings arrive as 1-2 page fragments;
    folding them keeps a chapter contiguous with the material it belongs to.
    """
    merged: list[Chapter] = []
    for chapter in chapters:
        if merged and chapter.page_count < MIN_CHAPTER_PAGES:
            previous = merged[-1]
            merged[-1] = Chapter(previous.title, previous.start_page, chapter.end_page)
        else:
            merged.append(chapter)
    return merged


def _from_outline(doc) -> list[tuple[int, str]]:
    """Chapter starts from the embedded PDF outline, shallowest level only."""
    try:
        toc = doc.get_toc() or []
    except Exception as exc:  # pragma: no cover - defensive, varies by file
        logger.warning("Could not read PDF outline: %s", exc)
        return []
    if not toc:
        return []

    top_level = min(entry[0] for entry in toc)
    starts: list[tuple[int, str]] = []
    for level, title, page_number in toc:
        if level != top_level:
            continue
        page_index = int(page_number) - 1
        if page_index >= 0 and (not starts or page_index > starts[-1][0]):
            starts.append((page_index, str(title)))
    return starts


def _span_size_histogram(doc) -> Counter:
    """Character counts keyed by rounded span font size."""
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        sizes[round(span.get("size", 0))] += len(text)
    return sizes


def _from_font_size(doc) -> list[tuple[int, str]]:
    """Chapter starts from headings set noticeably larger than body text.

    Works on the Nepali book because a glyph's size is independent of whether
    its encoding decodes to meaningful characters.
    """
    sizes = _span_size_histogram(doc)
    if not sizes:
        return []

    body_size, body_chars = sizes.most_common(1)[0]
    # A heading size is larger than body and rare — a size carrying a large
    # share of the book's characters is a second body style, not a heading.
    candidates = [
        size
        for size, chars in sizes.items()
        if size > body_size + 2 and chars < body_chars * 0.05
    ]
    if not candidates:
        return []

    threshold = min(candidates)
    starts: list[tuple[int, str]] = []
    for page_index, page in enumerate(doc):
        heading = _first_span_at_size(page, threshold)
        if heading is None:
            continue
        if starts and page_index - starts[-1][0] < MIN_CHAPTER_PAGES:
            continue
        starts.append((page_index, heading))
    return starts


def _first_span_at_size(page, threshold: int) -> str | None:
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text and round(span.get("size", 0)) >= threshold:
                    return text[:80]
    return None


def _from_text_markers(doc) -> list[tuple[int, str]]:
    """Chapter starts from an explicit `Unit N` / `अध्याय N` line near the page top."""
    patterns = [re.compile(pattern, re.MULTILINE) for pattern in _TEXT_MARKERS]
    starts: list[tuple[int, str]] = []
    for page_index, page in enumerate(doc):
        head = "\n".join(page.get_text().split("\n")[:6])
        for pattern in patterns:
            match = pattern.search(head)
            if not match:
                continue
            if starts and page_index - starts[-1][0] < MIN_CHAPTER_PAGES:
                break
            starts.append((page_index, match.group(0).strip()))
            break
    return starts


def detect_chapters(doc) -> list[Chapter]:
    """Best-effort chapter ranges for an open PyMuPDF document.

    Returns [] when no strategy produces a plausible index, which callers
    read as "fall back to the page window".
    """
    total_pages = len(doc)
    if total_pages < MIN_CHAPTER_PAGES * MIN_CHAPTERS:
        return []

    for name, strategy in (
        ("outline", _from_outline),
        ("font-size", _from_font_size),
        ("text-marker", _from_text_markers),
    ):
        try:
            starts = strategy(doc)
        except Exception as exc:  # pragma: no cover - defensive per-strategy
            logger.warning("Chapter detection strategy %s failed: %s", name, exc)
            continue
        if not _plausible(starts, total_pages):
            continue
        chapters = _merge_short(_to_chapters(starts, total_pages))
        if len(chapters) >= MIN_CHAPTERS:
            logger.info("Detected %s chapters via %s", len(chapters), name)
            return chapters

    logger.info("No chapter structure detected; page-window context will be used")
    return []


def chapter_for_page(chapters: list[Chapter], page_index: int) -> Chapter | None:
    for chapter in chapters:
        if chapter.contains(page_index):
            return chapter
    return None
