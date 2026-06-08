"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def get_weaviate_client():
    """Tạo kết nối tới Weaviate Cloud (WCS) hoặc Weaviate Local dựa trên biến môi trường."""
    import weaviate
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if weaviate_url and weaviate_api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            skip_init_checks=True
        )
    else:
        return weaviate.connect_to_local(skip_init_checks=True)


def get_embedding(text: str) -> list[float]:
    """Sinh vector embedding từ OpenAI hoặc OpenRouter."""
    from openai import OpenAI
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY không được thiết lập trong môi trường / file .env")
        
    if openai_key.startswith("sk-or-"):
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        model_name = "openai/text-embedding-3-small"
    else:
        client = OpenAI(api_key=openai_key)
        model_name = "text-embedding-3-small"
        
    response = client.embeddings.create(
        model=model_name,
        input=[text]
    )
    return response.data[0].embedding


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import weaviate
    from weaviate.classes.query import MetadataQuery
    
    # 1. Sinh embedding cho query
    query_embedding = get_embedding(query)
    
    # 2. Kết nối Weaviate và tìm kiếm
    client = get_weaviate_client()
    try:
        if not client.collections.exists("DrugLawDocs"):
            print("Cảnh báo: Collection 'DrugLawDocs' chưa tồn tại trong Weaviate.")
            return []
            
        collection = client.collections.get("DrugLawDocs")
        
        # Truy vấn vector gần nhất
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        search_results = []
        for obj in results.objects:
            # Cosine similarity = 1 - Cosine distance
            distance = obj.metadata.distance if obj.metadata.distance is not None else 0.0
            score = 1.0 - distance
            
            search_results.append({
                "content": obj.properties.get("content", ""),
                "score": float(score),
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "doc_type": obj.properties.get("doc_type", ""),
                    "chunk_index": int(obj.properties.get("chunk_index", 0))
                }
            })
            
        # Sắp xếp giảm dần theo score
        search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results
    finally:
        client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    # Test thử tìm kiếm
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    print("\n--- Kết quả tìm kiếm ngữ nghĩa: ---")
    for r in results:
        print(f"[{r['score']:.3f}] Source: {r['metadata']['source']} (Index: {r['metadata']['chunk_index']})")
        print(f"Content: {r['content'][:150]}...")
        print("-" * 50)
