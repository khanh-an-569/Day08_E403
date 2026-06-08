"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import os
import sys
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNK_SIZE = 500: Chọn 500 ký tự vì nó đủ lớn để lưu trữ ngữ cảnh có nghĩa (khoảng 3-4 câu tiếng Việt),
# nhưng cũng đủ nhỏ để giữ độ chính xác của embedding và tránh vượt quá giới hạn ngữ cảnh của LLM.
CHUNK_SIZE = 500

# CHUNK_OVERLAP = 50: Chọn 50 ký tự gối đầu để tránh mất mát thông tin ngữ nghĩa tại biên của các đoạn chunk.
CHUNK_OVERLAP = 50

# CHUNKING_METHOD = "recursive": Sử dụng RecursiveCharacterTextSplitter vì nó tự động phân tách văn bản
# dựa trên thứ tự các ký tự xuống dòng (\n\n, \n), dấu chấm câu, và dấu cách. Điều này rất phù hợp cho
# cả tài liệu pháp luật (có các điều khoản tách biệt bằng dòng) và bài viết tin tức.
CHUNKING_METHOD = "recursive"

# EMBEDDING_MODEL = "text-embedding-3-small": Sử dụng mô hình OpenAI text-embedding-3-small 
# có hiệu năng cao, chi phí cực kỳ tối ưu và khả năng biểu diễn ngữ nghĩa đa ngôn ngữ (bao gồm tiếng Việt) rất tốt.
EMBEDDING_MODEL = "text-embedding-3-small"

# EMBEDDING_DIM = 1536: text-embedding-3-small mặc định trả về embedding vector với 1536 chiều.
EMBEDDING_DIM = 1536

# VECTOR_STORE = "weaviate": Sử dụng Weaviate vì nó hỗ trợ cơ chế Hybrid Search (Dense + Sparse/BM25)
# tích hợp sẵn, hiệu năng mở rộng tốt và tương thích hoàn hảo với API v4 Client.
VECTOR_STORE = "weaviate"


# =============================================================================
# HELPERS
# =============================================================================

def get_weaviate_client():
    """Tạo kết nối tới Weaviate Cloud (WCS) hoặc Weaviate Local dựa trên biến môi trường."""
    import weaviate
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if weaviate_url and weaviate_api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )
    else:
        return weaviate.connect_to_local()


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
        
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file.relative_to(STANDARDIZED_DIR)) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
        except Exception as e:
            print(f"Error loading {md_file.name}: {e}")
            
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "doc_type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
            
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from openai import OpenAI
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY không được thiết lập trong môi trường / file .env")
        
    if openai_key.startswith("sk-or-"):
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        model_name = EMBEDDING_MODEL
        if not model_name.startswith("openai/") and "/" not in model_name:
            model_name = f"openai/{model_name}"
    else:
        client = OpenAI(api_key=openai_key)
        model_name = EMBEDDING_MODEL
        
    texts = [c["content"] for c in chunks]
    if not texts:
        return chunks
        
    print(f"Đang sinh embeddings sử dụng {model_name} cho {len(texts)} chunks...")
    
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch_texts
        )
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)
        
    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.config import Property, DataType, Configure
    
    client = get_weaviate_client()
    
    try:
        # Xóa collection cũ nếu đã tồn tại để tránh xung đột dữ liệu
        if client.collections.exists("DrugLawDocs"):
            client.collections.delete("DrugLawDocs")
            print("Đã xóa collection 'DrugLawDocs' cũ.")
            
        # Tạo mới collection v4
        collection = client.collections.create(
            name="DrugLawDocs",
            vectorizer_config=Configure.Vectorizer.none(),  # Sử dụng embedding tự tính toán
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        print("Đã tạo mới collection 'DrugLawDocs' trên Weaviate.")
        
        # Batch insert các object
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["doc_type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
        print(f"Đã index thành công {len(chunks)} chunks vào Weaviate.")
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    run_pipeline()