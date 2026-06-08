"""
Task 6 — Lexical Search Module (BM25)
"""

import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

CORPUS = []
BM25_INDEX = None


def get_weaviate_client():
    import weaviate

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(
                weaviate_api_key
            )
        )

    return weaviate.connect_to_local()


# def tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản.

    Bonus:
        from underthesea import word_tokenize
    """
from underthesea import word_tokenize

def tokenize(text):
    return word_tokenize(
        text.lower()
    )
    return text.lower().split()


def load_corpus_from_weaviate() -> list[dict]:
    """
    Đọc toàn bộ chunks từ collection DrugLawDocs.
    """

    corpus = []

    client = get_weaviate_client()

    try:
        collection = client.collections.get(
            "DrugLawDocs"
        )

        response = collection.iterator()

        for obj in response:

            corpus.append(
                {
                    "content":
                        obj.properties["content"],

                    "metadata": {
                        "source":
                            obj.properties.get(
                                "source"
                            ),

                        "doc_type":
                            obj.properties.get(
                                "doc_type"
                            ),

                        "chunk_index":
                            obj.properties.get(
                                "chunk_index"
                            )
                    }
                }
            )

    finally:
        client.close()

    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Build BM25 index.
    """

    tokenized_corpus = [
        tokenize(doc["content"])
        for doc in corpus
    ]

    return BM25Okapi(tokenized_corpus)


def initialize():
    """
    Load corpus + build BM25.
    """

    global CORPUS
    global BM25_INDEX

    CORPUS = load_corpus_from_weaviate()

    print(
        f"Loaded {len(CORPUS)} chunks"
    )

    BM25_INDEX = build_bm25_index(
        CORPUS
    )

    print("BM25 index ready")


def lexical_search(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    BM25 lexical search.
    """

    global BM25_INDEX
    global CORPUS

    if BM25_INDEX is None:
        initialize()

    tokenized_query = tokenize(query)

    scores = BM25_INDEX.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        score = float(scores[idx])

        if score <= 0:
            continue

        results.append(
            {
                "content":
                    CORPUS[idx]["content"],

                "score":
                    score,

                "metadata":
                    CORPUS[idx]["metadata"]
            }
        )

    return results


if __name__ == "__main__":

    query = (
        "Điều 248 tàng trữ trái phép chất ma tuý"
    )

    results = lexical_search(
        query,
        top_k=5
    )

    print(f"\nQuery: {query}")
    print("-" * 80)

    for i, r in enumerate(
        results,
        start=1
    ):
        print(
            f"{i}. "
            f"Score={r['score']:.4f}"
        )

        print(
            f"Source: "
            f"{r['metadata']['source']}"
        )

        print(
            r["content"][:200]
        )

        print("-" * 80)