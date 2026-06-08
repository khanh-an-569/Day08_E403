import os
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

class PageIndexSearcher:
    def __init__(self):
        api_key = os.getenv("PAGEINDEX_API_KEY")
        if not api_key:
            raise ValueError("PAGEINDEX_API_KEY not found in .env")
        
        # Initialize PageIndex client
        self.client = PageIndexClient(api_key=api_key)
        
    def search(self, query: str, top_k: int = 3):
        print(f"[PageIndex] Querying external knowledge base for: '{query}'")
        try:
            # Assumed API structure for the course's SDK
            results = self.client.search(query=query, limit=top_k)
            
            formatted_results = []
            for r in results:
                # Some SDKs use dict, some use object attributes. Handle both.
                if isinstance(r, dict):
                    content = r.get("content", r.get("text", ""))
                    source = r.get("source", "PageIndex")
                    score = r.get("score", 0.0)
                else:
                    content = getattr(r, "content", getattr(r, "text", ""))
                    source = getattr(r, "source", "PageIndex")
                    score = getattr(r, "score", 0.0)
                    
                formatted_results.append({
                    "content": content,
                    "source": source,
                    "category": "External API",
                    "score": score
                })
            return formatted_results
        except Exception as e:
            print(f"[PageIndex] Error: {e}")
            return []

if __name__ == "__main__":
    searcher = PageIndexSearcher()
    res = searcher.search("Tin tức về việc nghệ sĩ sử dụng ma túy")
    print("--- PAGEINDEX SEARCH RESULTS ---")
    for r in res:
        print(f"[{r['score']}] {r['source']}: {r['content'][:100]}...")
