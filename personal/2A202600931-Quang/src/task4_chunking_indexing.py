import os
import re
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Setup paths
STANDARDIZED_DIR = os.path.join("data", "standardized")
VECTOR_DB_DIR = os.path.join("data", "vector_db")

# 1. Khởi tạo mô hình embedding (sẽ tự động tải model về máy, hỗ trợ Tiếng Việt)
print("Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

# 2. Khởi tạo ChromaDB client
print("Initializing ChromaDB...")
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

# Xóa dữ liệu cũ trong collection nếu đã tồn tại thay vì xóa collection (để giữ nguyên UUID)
collection = client.get_or_create_collection(name="law_and_news_collection")
try:
    existing_data = collection.get()
    if existing_data['ids']:
        collection.delete(ids=existing_data['ids'])
        print(f"Deleted {len(existing_data['ids'])} existing documents to refresh.")
except Exception as e:
    print(f"Error clearing old data: {e}")

# 3. Cấu hình LangChain Text Splitter
# Chia văn bản thành các đoạn 1000 ký tự, overlap 200 ký tự để không bị đứt đoạn ý
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

def process_and_index():
    docs_processed = 0
    chunks_added = 0
    
    # Quét qua 2 thư mục: legal và news
    for category in ["legal", "news"]:
        category_dir = os.path.join(STANDARDIZED_DIR, category)
        if not os.path.exists(category_dir):
            continue
            
        for filename in os.listdir(category_dir):
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(category_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content.strip():
                continue
                
            # Tiền xử lý: Xóa các dấu xuống dòng dư thừa làm đứt gãy câu chữ
            content = re.sub(r'\n+', ' ', content)
                
            # Chunking (cắt nhỏ file)
            chunks = text_splitter.split_text(content)
            
            # Indexing (Lập chỉ mục)
            ids = [f"{category}_{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "category": category} for _ in range(len(chunks))]
            
            # Mã hóa chunks thành Vectors
            embeddings = embedder.encode(chunks).tolist()
            
            # Lưu vào ChromaDB
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            docs_processed += 1
            chunks_added += len(chunks)
            print(f"  -> {filename}: Indexed {len(chunks)} chunks.")

    print(f"\n[DONE] Processed {docs_processed} documents. Total {chunks_added} chunks saved to ChromaDB.")
    print(f"Database Path: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    process_and_index()
