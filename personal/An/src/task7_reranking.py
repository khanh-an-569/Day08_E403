"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Đã hoàn thiện implement cho MMR và RRF kèm giải thích.
"""

from typing import Optional
from collections import defaultdict
import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Helper function
# =============================================================================

def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Tính toán Cosine Similarity giữa 2 vector."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


# =============================================================================
# Reranking Methods
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Fallback implementation.
    Chỉ sort theo score retrieval.
    """
    ranked = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )
    return ranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))
    
    # Đảm bảo top_k không vượt quá số lượng candidate đang có
    k = min(top_k, len(candidates))

    for _ in range(k):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # 1. Relevance: Tính độ tương đồng với query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # 2. Diversity: Tính độ tương đồng lớn nhất với các tài liệu đã chọn
            max_sim_to_selected = 0.0
            if selected:
                max_sim_to_selected = max(
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                )

            # 3. Tính điểm MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            # Cập nhật kết quả tốt nhất
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        # Đưa tài liệu tốt nhất ở vòng lặp này vào danh sách được chọn
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion

    RRF(d) = Σ 1 / (k + rank(d))
    """
    rrf_scores = defaultdict(float)
    item_lookup = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] += 1.0 / (k + rank)
            item_lookup[key] = item

    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for content, score in sorted_docs[:top_k]:
        item = item_lookup[content].copy()
        item["score"] = float(score)  # Cập nhật lại score thành RRF score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
    **kwargs
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    
    elif method == "mmr":
        query_embedding = kwargs.get("query_embedding")
        if not query_embedding:
            raise ValueError("Method 'mmr' requires 'query_embedding' parameter.")
        lambda_param = kwargs.get("lambda_param", 0.7)
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)
        
    elif method == "rrf":
        ranked_lists = kwargs.get("ranked_lists")
        if not ranked_lists:
            raise ValueError("Method 'rrf' requires 'ranked_lists' parameter.")
        return rerank_rrf(ranked_lists, top_k)
        
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print("--- TEST CROSS-ENCODER FALLBACK ---")
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="cross_encoder")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")


    print("\n--- TEST MMR ---")
    # Thêm dummy embeddings để test MMR
    query_emb = [1.0, 0.0, 0.0]
    mmr_candidates = [
        {"content": "Doc 1 (Giống query)", "embedding": [0.9, 0.1, 0.0], "score": 0},
        {"content": "Doc 2 (Giống Doc 1)", "embedding": [0.85, 0.15, 0.0], "score": 0},
        {"content": "Doc 3 (Khác biệt)", "embedding": [0.5, 0.8, 0.0], "score": 0},
    ]
    mmr_results = rerank(
        query="test mmr", 
        candidates=mmr_candidates, 
        top_k=2, 
        method="mmr", 
        query_embedding=query_emb,
        lambda_param=0.5
    )
    for r in mmr_results:
        print(f"[-] {r['content']}")


    print("\n--- TEST RRF ---")
    list_bm25 = [
        {"content": "Doc A", "score": 10.5},
        {"content": "Doc B", "score": 8.2},
        {"content": "Doc C", "score": 5.1}
    ]
    list_vector = [
        {"content": "Doc C", "score": 0.9},
        {"content": "Doc A", "score": 0.8},
        {"content": "Doc D", "score": 0.7}
    ]
    # Truyền rỗng list candidates gốc vì RRF lấy dữ liệu từ ranked_lists
    rrf_results = rerank(
        query="test rrf", 
        candidates=[], 
        top_k=2, 
        method="rrf", 
        ranked_lists=[list_bm25, list_vector]
    )
    for r in rrf_results:
        print(f"[RRF Score: {r['score']:.4f}] {r['content']}")