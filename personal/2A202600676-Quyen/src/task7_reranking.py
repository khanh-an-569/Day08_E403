"""
Task 7 - Reranking Module.

Design:
- Prefer Jina Reranker API when a key is available.
- Fall back to a deterministic local scoring heuristic if the API is not reachable.
- Also provide MMR and RRF helpers for completeness.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_RERANK_MODEL = os.getenv(
    "JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"
)


def _direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    return [tok for tok in tokens if tok]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(0, 32, 4):
            idx = int.from_bytes(digest[i : i + 4], "little") % dim
            vec[idx] += 1.0
    return _normalize(vec)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _local_rerank_score(query: str, content: str, base_score: float = 0.0) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(content)
    if not query_tokens or not doc_tokens:
        return base_score

    doc_set = set(doc_tokens)
    overlap = sum(1 for token in query_tokens if token in doc_set)
    overlap_ratio = overlap / len(query_tokens)

    phrase_bonus = 0.0
    lowered_query = " ".join(query_tokens)
    lowered_doc = " ".join(doc_tokens)
    if lowered_query and lowered_query in lowered_doc:
        phrase_bonus = 0.35

    query_vec = _hash_embedding(query)
    doc_vec = _hash_embedding(content)
    semantic = max(_cosine_similarity(query_vec, doc_vec), 0.0)

    return float(base_score * 0.4 + overlap_ratio * 0.35 + semantic * 0.2 + phrase_bonus)


def _rerank_with_jina(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    session = _direct_session()
    response = session.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c.get("content", "") for c in candidates],
            "top_n": top_k,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in payload.get("results", []):
        idx = item.get("index")
        if idx is None or idx >= len(candidates):
            continue
        candidate = dict(candidates[idx])
        candidate["score"] = float(item.get("relevance_score", 0.0))
        results.append(candidate)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using a cross-encoder style scorer.

    Returns:
        List of top_k candidates, re-scored and sorted by score descending.
    """
    if top_k <= 0 or not candidates:
        return []

    if JINA_API_KEY:
        try:
            return _rerank_with_jina(query, candidates, top_k)
        except Exception:
            # Fall back to local deterministic scoring if the API is unavailable.
            pass

    rescored = []
    for candidate in candidates:
        item = dict(candidate)
        item["score"] = _local_rerank_score(
            query, item.get("content", ""), float(item.get("score", 0.0))
        )
        rescored.append(item)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]


def _cosine_from_embeddings(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance - selects relevant and diverse candidates.
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            cand_emb = candidates[idx].get("embedding", [])
            if not cand_emb:
                continue

            relevance = _cosine_from_embeddings(query_embedding, cand_emb)
            diversity_penalty = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding", [])
                if sel_emb:
                    diversity_penalty = max(
                        diversity_penalty, _cosine_from_embeddings(cand_emb, sel_emb)
                    )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = dict(candidates[idx])
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results[:top_k]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion - combines multiple ranked lists.
    """
    if top_k <= 0 or not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            content = item.get("content", "")
            if not content:
                continue
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
            item_map[content] = item

    ordered = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results: list[dict] = []
    for content, score in ordered[:top_k]:
        item = dict(item_map[content])
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    if method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
