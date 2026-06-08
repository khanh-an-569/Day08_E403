"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import json
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    import requests

    # Option A: Jina Reranker API
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json().get("results", [])
                results = []
                for r in reranked:
                    idx = r["index"]
                    item = candidates[idx].copy()
                    item["score"] = float(r["relevance_score"])
                    results.append(item)
                return results
        except Exception as e:
            print(f"Lỗi Jina Rerank: {e}. Chuyển sang phương án dự phòng.")

    # Option B: Cohere Rerank API
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {cohere_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "rerank-multilingual-v3.0",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json().get("results", [])
                results = []
                for r in reranked:
                    idx = r["index"]
                    item = candidates[idx].copy()
                    item["score"] = float(r["relevance_score"])
                    results.append(item)
                return results
        except Exception as e:
            print(f"Lỗi Cohere Rerank: {e}. Chuyển sang phương án dự phòng.")

    # Option C: OpenRouter / OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            if openai_key.startswith("sk-or-"):
                client = OpenAI(api_key=openai_key, base_url="https://openrouter.ai/api/v1")
                model_name = "openai/gpt-4o-mini"
            else:
                client = OpenAI(api_key=openai_key)
                model_name = "gpt-4o-mini"

            prompt = f"Yêu cầu: Đánh giá độ liên quan của các tài liệu dưới đây với câu truy vấn: \"{query}\".\n"
            prompt += "Trả về danh sách điểm số (từ 0.0 đến 1.0) cho từng tài liệu dưới định dạng JSON với key là 'scores' chứa list float tương ứng theo thứ tự tài liệu đầu vào.\n\n"
            for i, c in enumerate(candidates):
                prompt += f"Tài liệu {i}: {c['content']}\n"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates document relevance. Output ONLY a valid JSON object in format: {\"scores\": [0.95, 0.4, ...]}"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=15
            )

            res_content = response.choices[0].message.content
            scores_data = json.loads(res_content)
            scores = scores_data.get("scores", [])

            scored_candidates = []
            for idx, score in enumerate(scores):
                if idx < len(candidates):
                    item = candidates[idx].copy()
                    item["score"] = float(score)
                    scored_candidates.append(item)
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            return scored_candidates[:top_k]
        except Exception as e:
            print(f"Lỗi LLM Rerank: {e}. Chuyển sang phương án dự phòng cuối cùng.")

    # Option D: Fallback - Sắp xếp theo score ban đầu
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    import numpy as np

    if not candidates:
        return []

    def cosine_sim(vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # Đảm bảo mỗi candidate có embedding.
    for c in candidates:
        if "embedding" not in c:
            c["embedding"] = []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()

    # Sắp xếp theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        if not candidates:
            return []
        # Tự động tính toán embedding cho query từ OpenAI/OpenRouter
        try:
            from src.task4_chunking_indexing import EMBEDDING_MODEL
            from openai import OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                if openai_key.startswith("sk-or-"):
                    client = OpenAI(api_key=openai_key, base_url="https://openrouter.ai/api/v1")
                    model_name = EMBEDDING_MODEL
                    if not model_name.startswith("openai/") and "/" not in model_name:
                        model_name = f"openai/{model_name}"
                else:
                    client = OpenAI(api_key=openai_key)
                    model_name = EMBEDDING_MODEL
                
                resp = client.embeddings.create(model=model_name, input=[query])
                query_embedding = resp.data[0].embedding
                return rerank_mmr(query_embedding, candidates, top_k)
        except Exception as e:
            print(f"Lỗi tính toán query embedding cho MMR: {e}")
        # Fallback nếu không có embedding
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        if not candidates:
            return []
        if isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k)
        else:
            return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

