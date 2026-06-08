from src.task7_reranking import RRFReranker
from src.task8_pageindex_vectorless import PageIndexSearcher

class RAGRetriever:
    def __init__(self):
        self.rrf_reranker = RRFReranker()
        self.pageindex_searcher = PageIndexSearcher()
        self.THRESHOLD = 0.005  # Ngưỡng điểm thấp nhất của RRF để quyết định dùng Fallback
        
    def retrieve(self, query: str, top_k: int = 5):
        print(f"\n[RAG Retriever] Searching for: '{query}'")
        
        # 1. Thử dùng Local Hybrid Search (Semantic + BM25)
        local_results = self.rrf_reranker.search(query, top_k=top_k)
        
        # Kiểm tra nếu kết quả trống hoặc điểm quá thấp
        if not local_results or local_results[0]['score'] < self.THRESHOLD:
            print("[RAG Retriever] Local search score is low. Fallback to PageIndex...")
            external_results = self.pageindex_searcher.search(query, top_k=top_k)
            if external_results:
                return external_results
                
        return local_results

if __name__ == "__main__":
    retriever = RAGRetriever()
    res = retriever.retrieve("Ca sĩ Chi Dân bị bắt như thế nào?")
    for r in res:
        print(f"[{r['score']:.4f}] {r['source']}: {r['content'][:100]}...")
