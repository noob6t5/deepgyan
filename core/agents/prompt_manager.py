from __future__ import annotations


class PromptManager:
    """Centralizes prompt templates for the tutor experience."""

    @staticmethod
    def current_page_prompt(context: str) -> str:
        return f"""You are an advanced, helpful document assistant responding to queries.
You must rely on the structured current-page context below and, when a page image is attached,
use the image as visual evidence for diagrams, tables, equations, headings, and OCR-missed text.
If neither the context nor attached image contains the answer, explicitly state that you cannot find it in the current mode's context.

Format:
<think>Provide a 1-2 sentence high-level summary of how you will answer. Do NOT include step-by-step reasoning.</think>
<final>Provide the final answer only.</final>

Structured Current-Page Context:
{context}
"""

    @staticmethod
    def whole_book_prompt(context: str) -> str:
        return f"""You are an advanced, helpful document assistant responding to queries.
Use the retrieved whole-book chunks below to answer. If the answer is not in the retrieved chunks, say you cannot find it.

Format:
<think>Provide a 1-2 sentence high-level summary of how you will answer. Do NOT include step-by-step reasoning.</think>
<final>Provide the final answer only.</final>

Retrieved Context:
{context}
"""

    @staticmethod
    def env_summary_prompt(raw_text: str) -> str:
        return f"""You are a document analyzer. Convert this raw OCR text into highly structured, clean synthetic data context.
Provide a clear, high-level summary of what this specific section is about and outline its main topics.

Raw OCR Document Text:
{raw_text}
"""

    @staticmethod
    def global_summary_prompt(raw_text: str, page_start: int, page_end: int) -> str:
        return f"""Convert all of the following messy OCR text into clean, structured synthetic data context. 
Organize the overarching concepts strictly and clearly so it can be used for RAG applications.

Raw Document Text (Pages {page_start} to {page_end}):
{raw_text}
"""
