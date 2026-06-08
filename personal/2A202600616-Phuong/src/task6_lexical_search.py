"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import sys
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

from pathlib import Path
from src.task4_chunking_indexing import load_documents, chunk_documents

# Corpus và BM25 Index sẽ được load tự động khi gọi lexical_search
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    from underthesea import word_tokenize
    import re
    
    # Tokenize sử dụng underthesea word_tokenize để ghép từ tiếng Việt
    tokenized_corpus = []
    for doc in corpus:
        # Làm sạch ký tự đặc biệt trước khi tách từ
        cleaned_text = re.sub(r'[^\w\s]', ' ', doc["content"].lower())
        words = word_tokenize(cleaned_text, format="text").split()
        tokenized_corpus.append(words)
        
    return BM25Okapi(tokenized_corpus)


def initialize_corpus_and_bm25():
    """Khởi tạo CORPUS và BM25_INDEX nếu chưa được tải."""
    global CORPUS, BM25_INDEX
    if not CORPUS:
        try:
            docs = load_documents()
            CORPUS = chunk_documents(docs)
            BM25_INDEX = build_bm25_index(CORPUS)
            print(f"✓ Đã nạp thành công {len(CORPUS)} chunks vào BM25 corpus.")
        except Exception as e:
            print(f"Lỗi khởi tạo corpus cho BM25: {e}")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import re
    import numpy as np
    
    initialize_corpus_and_bm25()
    
    if not CORPUS or BM25_INDEX is None:
        return []
        
    # Tokenize query sử dụng underthesea
    from underthesea import word_tokenize
    cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
    tokenized_query = word_tokenize(cleaned_query, format="text").split()
    scores = BM25_INDEX.get_scores(tokenized_query)
    
    # Lấy top_k chỉ số có điểm số cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
            
    # Đảm bảo kết quả được sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    # Chạy thử
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    print("\n--- Kết quả tìm kiếm từ khóa (BM25): ---")
    for r in results:
        print(f"[{r['score']:.3f}] Source: {r['metadata']['source']} (Index: {r['metadata']['chunk_index']})")
        print(f"Content: {r['content'][:150]}...")
        print("-" * 50)
