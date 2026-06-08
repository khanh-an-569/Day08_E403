import os
import google.generativeai as genai
from dotenv import load_dotenv
from src.task9_retrieval_pipeline import RAGRetriever

load_dotenv()

class RAGGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-3.5-flash')
        self.retriever = RAGRetriever()
        
    def generate_answer(self, query: str):
        # 1. Truy xuất tài liệu bằng Hybrid Search Pipeline
        retrieved_docs = self.retriever.retrieve(query, top_k=20)
        
        if not retrieved_docs:
            return "Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn trong cơ sở dữ liệu."
            
        # 2. Xây dựng Context và Prompt để ép LLM trả lời kèm Citation (Trích dẫn)
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc['source']
            content = doc['content']
            context_parts.append(f"[Tài liệu {i+1} - Nguồn: {source}]\n{content}\n")
            
        context_str = "\n".join(context_parts)
        
        prompt = f"""
Bạn là một chuyên gia tư vấn Pháp luật và Tin tức báo chí tại Việt Nam.
Dựa vào các tài liệu được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách đầy đủ và chính xác nhất.

*Lưu ý quan trọng:* 
- Tổng hợp các đoạn văn liên quan trong tài liệu để cung cấp định nghĩa, câu trả lời đầy đủ và dễ hiểu nhất có thể. Nếu tài liệu không chứa định nghĩa trực tiếp, hãy kết hợp các thông tin giải thích cấu thành.
- Trích dẫn rõ ràng nguồn tài liệu (Tên file) cho mỗi thông tin bạn sử dụng.
- Nếu thông tin hoàn toàn không có trong tài liệu, hãy nói 'Tôi không tìm thấy thông tin này trong cơ sở dữ liệu'. Không được tự suy diễn thông tin ngoài tài liệu.

TÀI LIỆU CUNG CẤP (CONTEXT):
{context_str}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

TRẢ LỜI:
"""
        print("[LLM] Đang sinh câu trả lời bằng Gemini 3.5 Flash...\n")
        # 3. Gọi Gemini sinh câu trả lời
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    generator = RAGGenerator()
    
    queries = [
        "Hình phạt đối với tội tổ chức sử dụng ma túy là gì?",
        "Ca sĩ Chi Dân bị bắt như thế nào và vì tội gì?",
    ]
    
    for query in queries:
        print("\n==================================================")
        print(f"Câu hỏi: {query}")
        print("==================================================")
        answer = generator.generate_answer(query)
        print(answer)
