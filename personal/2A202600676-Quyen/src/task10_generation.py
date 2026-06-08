"""
Task 10 - Generation Có Citation.

This module turns retrieved chunks into a cited answer without relying on an
LLM being available. If an API key is present, the structure is ready to be
extended, but the default path is deterministic and offline-safe.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # pragma: no cover - direct execution fallback
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source.

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.
"""


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"
    doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
    return f"{source} ({doc_type})"


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce lost-in-the-middle.
    Keep the strongest chunk first, then alternate from the remaining list.
    """
    if len(chunks) <= 2:
        return list(chunks)

    ordered = [chunks[0]]
    left = 1
    right = len(chunks) - 1
    take_right = True

    while left <= right:
        if take_right:
            ordered.append(chunks[right])
            right -= 1
        else:
            ordered.append(chunks[left])
            left += 1
        take_right = not take_right

    return ordered


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a citation-friendly context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = _source_label(chunk, i)
        content = chunk.get("content", "").strip()
        context_parts.append(f"[{label}]\n{content}")
    return "\n\n---\n\n".join(context_parts)


def _extract_citable_snippet(text: str, max_len: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _compose_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_lines = [
        f"Về câu hỏi: {query}",
        "",
        "Dựa trên các nguồn hiện có:",
    ]

    for i, chunk in enumerate(chunks, 1):
        label = _source_label(chunk, i)
        snippet = _extract_citable_snippet(chunk.get("content", ""))
        if not snippet:
            continue
        answer_lines.append(f"- {snippet} [{label}]")

    if len(answer_lines) == 3:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_lines.extend(
        [
            "",
            "Nếu cần kết luận pháp lý chính xác hơn, cần đối chiếu thêm điều khoản cụ thể trong văn bản gốc.",
        ]
    )
    return "\n".join(answer_lines)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    # Deterministic offline answer generation.
    answer = _compose_answer(query, reordered)
    if not answer.strip():
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    retrieval_source = reordered[0].get("source", "hybrid") if reordered else "none"
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
        "context": context,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
