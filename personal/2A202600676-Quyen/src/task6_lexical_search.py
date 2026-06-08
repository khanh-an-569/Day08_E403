"""
Task 6 - Lexical Search Module (BM25).

This implementation is self-contained:
- Loads markdown documents from data/standardized/
- Builds a BM25 index in memory
- Answers lexical queries without any local vector store
"""

from __future__ import annotations

import math
import re
import sys
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    text = text.lower()
    text = text.replace("#", " ")
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    tokens = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    return [tok for tok in tokens if tok]


def load_corpus() -> list[dict]:
    """Load all markdown files from data/standardized/."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in rel_path.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )
    return documents


class BM25Index:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs: dict[str, int] = {}
        for doc in self.tokenized_corpus:
            for token in set(doc):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.idf = {
            term: math.log(1.0 + (len(self.corpus) - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = [0.0] * len(self.corpus)
        if not self.corpus or not query_tokens:
            return scores

        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            dl = self.doc_lengths[idx] or 1
            tf: dict[str, int] = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(token)
                if idf is None:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores[idx] = score
        return scores


@lru_cache(maxsize=1)
def build_bm25_index(corpus_data: tuple[tuple[str, str, str, str], ...] | None = None) -> BM25Index:
    """
    Build BM25 index from the corpus.

    The optional corpus_data parameter is only used for cache stability.
    """
    corpus = CORPUS if CORPUS else load_corpus()
    return BM25Index(corpus)


def _ensure_corpus_loaded() -> None:
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical search using BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    if top_k <= 0:
        return []

    _ensure_corpus_loaded()
    if not CORPUS:
        return []

    index = build_bm25_index()
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = index.score(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results: list[dict] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        doc = CORPUS[idx]
        results.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc["metadata"],
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
