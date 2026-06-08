"""
Task 5 — Semantic Search Module
"""

import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


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


def embed_query(query: str) -> list[float]:
    """
    Embed query bằng cùng model với Task 4.
    """

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    if api_key.startswith("sk-or-"):
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        model_name = EMBEDDING_MODEL

        if "/" not in model_name:
            model_name = f"openai/{model_name}"

    else:
        client = OpenAI(api_key=api_key)
        model_name = EMBEDDING_MODEL

    response = client.embeddings.create(
        model=model_name,
        input=query
    )

    return response.data[0].embedding


def semantic_search(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    Dense Retrieval bằng Weaviate.

    Returns:
        [
            {
                "content": str,
                "score": float,
                "metadata": {...}
            }
        ]
    """

    from weaviate.classes.query import MetadataQuery

    query_embedding = embed_query(query)

    client = get_weaviate_client()

    try:
        collection = client.collections.get(
            "DrugLawDocs"
        )

        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(
                distance=True
            )
        )

        retrieved = []

        for obj in results.objects:

            distance = (
                obj.metadata.distance
                if obj.metadata
                else 1.0
            )

            similarity = max(
                0.0,
                1.0 - distance
            )

            retrieved.append(
                {
                    "content":
                        obj.properties["content"],

                    "score":
                        float(similarity),

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

        retrieved.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return retrieved

    finally:
        client.close()


if __name__ == "__main__":

    query = "hình phạt cho tội tàng trữ ma tuý"

    results = semantic_search(
        query=query,
        top_k=5
    )

    print(f"\nQuery: {query}")
    print("-" * 80)

    for i, r in enumerate(results, start=1):

        print(
            f"{i}. Score={r['score']:.4f}"
        )

        print(
            f"Source: "
            f"{r['metadata']['source']}"
        )

        print(
            r["content"][:200]
        )

        print("-" * 80)