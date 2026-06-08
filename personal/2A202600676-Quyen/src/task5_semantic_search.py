"""
Task 5 - Semantic search backed by OpenRouter embeddings and Weaviate.

This module queries the Weaviate Cloud collection built in Task 4.
No local vector index is used.
"""

import hashlib
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
import weaviate
import weaviate.classes.query as wq
from weaviate.auth import AuthApiKey

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

WEAVIATE_COLLECTION = "DrugLawDocs"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
)
OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
OPENROUTER_EMBED_DIMENSIONS = int(os.getenv("OPENROUTER_EMBED_DIMENSIONS", "1536"))


def _direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _weaviate_headers() -> dict:
    if not WEAVIATE_API_KEY:
        raise RuntimeError("WEAVIATE_API_KEY is required")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
    }


def _openrouter_headers() -> dict:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY (or OPENAI_API_KEY alias) is required")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "Day8 RAG Pipeline"),
    }


def _embed_text(text: str) -> list[float]:
    session = _direct_session()
    response = session.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=_openrouter_headers(),
        json={
            "model": OPENROUTER_EMBED_MODEL,
            "input": text,
            "encoding_format": "float",
            "dimensions": OPENROUTER_EMBED_DIMENSIONS,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return [float(x) for x in data["data"][0]["embedding"]]


def _connect_weaviate():
    if not WEAVIATE_URL:
        raise RuntimeError("WEAVIATE_URL is required")
    if not WEAVIATE_API_KEY:
        raise RuntimeError("WEAVIATE_API_KEY is required")

    url = WEAVIATE_URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )


def _content_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = normalized[:500]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    if top_k <= 0:
        return []

    query_vector = _embed_text(query)
    results = []
    seen = set()
    client = _connect_weaviate()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=max(top_k * 3, top_k),
            return_metadata=wq.MetadataQuery(distance=True),
            return_properties=["content", "source", "doc_type", "chunk_index"],
        )

        for obj in response.objects:
            props = obj.properties
            content = props.get("content", "") if isinstance(props, dict) else ""
            if not content:
                continue

            key = _content_key(content)
            if key in seen:
                continue
            seen.add(key)

            distance = getattr(obj.metadata, "distance", None)
            score = float(1.0 - distance) if distance is not None else 0.0
            results.append(
                {
                    "content": content,
                    "score": score,
                "metadata": {
                    "source": props.get("source", "") if isinstance(props, dict) else "",
                    "doc_type": props.get("doc_type", "") if isinstance(props, dict) else "",
                    "chunk_index": props.get("chunk_index", 0) if isinstance(props, dict) else 0,
                },
                }
            )
            if len(results) >= top_k:
                break
    finally:
        client.close()

    return results


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    if not results:
        print("Weaviate collection 'DrugLawDocs' is empty. Run Task 4 indexing first.")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
