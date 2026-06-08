import os
import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = os.path.join("data", "vector_db")

class SemanticSearcher:
    def __init__(self):
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.client.get_collection(name="law_and_news_collection")
        
    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                formatted_results.append({
                    "content": doc,
                    "source": meta.get('source', 'unknown'),
                    "category": meta.get('category', 'unknown'),
                    # Convert distance to similarity score
                    "score": 1.0 / (1.0 + dist)
                })
                
        return formatted_results

if __name__ == "__main__":
    searcher = SemanticSearcher()
    res = searcher.search("Hình phạt cho tội tổ chức sử dụng ma túy")
    print("--- SEMANTIC SEARCH RESULTS ---")
    for r in res:
        print(f"[{r['score']:.4f}] {r['source']}: {r['content'][:100]}...")
