"""
Task 9 - Retrieval Pipeline Hoàn Chỉnh.

Pipeline:
1. semantic_search + lexical_search
2. Merge bằng RRF
3. Rerank
4. Nếu kết quả hybrid yếu hoặc rỗng, fallback sang PageIndex
"""

from __future__ import annotations

from typing import Callable

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # pragma: no cover - fallback for direct execution
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _safe_call(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return []


def _mark_source(results: list[dict], source: str) -> list[dict]:
    marked = []
    for item in results:
        copied = dict(item)
        copied["source"] = source
        marked.append(copied)
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Unified retrieval pipeline with fallback to PageIndex.
    """
    if top_k <= 0:
        return []

    dense_results = _safe_call(semantic_search, query, top_k=top_k * 2)
    sparse_results = _safe_call(lexical_search, query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    merged = _mark_source(merged, "hybrid")

    final_results = merged[:top_k]
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception:
            final_results = merged[:top_k]

        final_results = _mark_source(final_results, "hybrid")

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback = _safe_call(pageindex_search, query, top_k=top_k)
        if fallback:
            return _mark_source(fallback[:top_k], "pageindex")
        return []

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
