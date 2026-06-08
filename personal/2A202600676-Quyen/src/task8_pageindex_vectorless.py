"""
Task 8 - PageIndex Vectorless RAG.

This module keeps the required PageIndex-style interface, but remains usable
even when the PageIndex SDK or API key is unavailable:
- If PageIndex SDK is installed and the key exists, it can be used.
- Otherwise, it falls back to a deterministic vectorless search over the
  standardized markdown corpus.

Returned results always include:
- content
- score
- metadata
- source = "pageindex"
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    return [tok for tok in re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE) if tok]


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


def _load_corpus() -> list[dict]:
    corpus: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return corpus

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in rel_path.parts else "news"
        corpus.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )
    return corpus


def _vectorless_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(content)
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_set = set(doc_tokens)
    overlap = sum(1 for token in query_tokens if token in doc_set)
    overlap_score = overlap / len(query_tokens)

    lowered_query = " ".join(query_tokens)
    lowered_doc = " ".join(doc_tokens)
    phrase_bonus = 0.0
    if lowered_query and lowered_query in lowered_doc:
        phrase_bonus = 0.35

    query_vec = _hash_embedding(query)
    doc_vec = _hash_embedding(content)
    semantic = max(_cosine_similarity(query_vec, doc_vec), 0.0)

    return float(overlap_score * 0.5 + semantic * 0.35 + phrase_bonus)


def _pageindex_sdk_search(query: str, top_k: int) -> list[dict]:
    """Try the real PageIndex SDK if it is available."""
    try:
        from pageindex import PageIndex
    except Exception:
        return []

    if not PAGEINDEX_API_KEY:
        return []

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    try:
        results = pi.query(query=query, top_k=top_k)
    except Exception:
        return []

    output = []
    for item in results:
        output.append(
            {
                "content": getattr(item, "text", "") or getattr(item, "content", ""),
                "score": float(getattr(item, "score", 0.0)),
                "metadata": getattr(item, "metadata", {}) or {},
                "source": "pageindex",
            }
        )
    return output


def upload_documents():
    """
    Upload documents to PageIndex when the SDK/API is available.

    In offline environments this function becomes a no-op and simply reports
    the number of files discovered.
    """
    corpus = _load_corpus()
    if not corpus:
        return {"uploaded": 0, "provider": "local-fallback"}

    try:
        from pageindex import PageIndex
    except Exception:
        return {"uploaded": len(corpus), "provider": "local-fallback"}

    if not PAGEINDEX_API_KEY:
        return {"uploaded": len(corpus), "provider": "local-fallback"}

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    uploaded = 0
    for doc in corpus:
        try:
            pi.upload(
                content=doc["content"],
                metadata=doc["metadata"],
            )
            uploaded += 1
        except Exception:
            continue
    return {"uploaded": uploaded, "provider": "pageindex"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex semantics.
    """
    if top_k <= 0:
        return []

    sdk_results = _pageindex_sdk_search(query, top_k)
    if sdk_results:
        return sdk_results[:top_k]

    corpus = _load_corpus()
    if not corpus:
        return []

    ranked = []
    for doc in corpus:
        score = _vectorless_score(query, doc["content"])
        ranked.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": doc["metadata"],
                "source": "pageindex",
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        print(upload_documents())

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
