from src.task5_semantic_search import SemanticSearcher
from src.task6_lexical_search import LexicalSearcher

class RRFReranker:
    def __init__(self):
        self.semantic_searcher = SemanticSearcher()
        self.lexical_searcher = LexicalSearcher()
        
    def search(self, query: str, top_k: int = 5, rrf_k: int = 60):
        candidate_count = max(top_k * 4, 50)
        semantic_results = self.semantic_searcher.search(query, top_k=candidate_count)
        lexical_results = self.lexical_searcher.search(query, top_k=candidate_count)
        
        # Hợp nhất và chấm điểm lại bằng thuật toán Reciprocal Rank Fusion
        rrf_scores = {}
        
        # Tính điểm cho Semantic
        for rank, res in enumerate(semantic_results):
            content = res['content']
            if content not in rrf_scores:
                rrf_scores[content] = {"score": 0.0, "source": res['source'], "category": res['category']}
            rrf_scores[content]["score"] += 1.0 / (rrf_k + rank + 1)
            
        # Tính điểm cho Lexical
        for rank, res in enumerate(lexical_results):
            content = res['content']
            if content not in rrf_scores:
                rrf_scores[content] = {"score": 0.0, "source": res['source'], "category": res['category']}
            rrf_scores[content]["score"] += 1.0 / (rrf_k + rank + 1)
            
        # Sắp xếp lại theo điểm RRF giảm dần
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Format kết quả đầu ra
        final_results = []
        for content, data in sorted_rrf[:top_k]:
            final_results.append({
                "content": content,
                "source": data['source'],
                "category": data['category'],
                "score": data['score']
            })
            
        return final_results

if __name__ == "__main__":
    reranker = RRFReranker()
    res = reranker.search("Hình phạt nào cho ca sĩ tổ chức sử dụng ma túy?", top_k=3)
    print("--- RRF HYBRID SEARCH RESULTS ---")
    for r in res:
        print(f"[{r['score']:.4f}] {r['source']}: {r['content'][:100]}...")
