import os
import chromadb
from rank_bm25 import BM25Okapi

VECTOR_DB_DIR = os.path.join("data", "vector_db")

class LexicalSearcher:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.client.get_collection(name="law_and_news_collection")
        
        # Lấy toàn bộ dữ liệu ra để build BM25 index in-memory
        print("Building BM25 Index from ChromaDB corpus...")
        all_data = self.collection.get()
        
        self.documents = all_data['documents']
        self.metadatas = all_data['metadatas']
        
        # Tokenize (tách từ cơ bản bằng khoảng trắng)
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def search(self, query: str, top_k: int = 5):
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Lấy top_k index có score cao nhất
        top_n_idx = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        formatted_results = []
        for idx in top_n_idx:
            if doc_scores[idx] > 0:
                formatted_results.append({
                    "content": self.documents[idx],
                    "source": self.metadatas[idx].get('source', 'unknown'),
                    "category": self.metadatas[idx].get('category', 'unknown'),
                    "score": float(doc_scores[idx])
                })
                
        return formatted_results

if __name__ == "__main__":
    searcher = LexicalSearcher()
    res = searcher.search("nghệ sĩ ma túy")
    print("--- LEXICAL SEARCH RESULTS ---")
    for r in res:
        print(f"[{r['score']:.4f}] {r['source']}: {r['content'][:100]}...")
