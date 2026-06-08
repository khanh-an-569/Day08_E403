import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Thêm thư mục hiện tại vào sys.path để import các module trong src/
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables từ file .env
load_dotenv(dotenv_path=project_root / ".env")

from src.task10_generation import generate_with_citation

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Tra cứu Luật Phòng chống Ma túy & Tin tức",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để giao diện trông premium và hiện đại
st.markdown("""
<style>
    /* Gradient background cho tiêu đề */
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Thiết kế card cho nguồn tham khảo */
    .source-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2a5298;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .source-header {
        font-weight: bold;
        color: #1e3c72;
        margin-bottom: 0.3rem;
    }
    
    .source-content {
        font-size: 0.9rem;
        color: #333333;
    }

    .badge-hybrid {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    .badge-pageindex {
        background-color: #efebe9;
        color: #4e342e;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* CSS cho các nút gợi ý câu hỏi (suggestion chips) */
    .suggestion-btn {
        margin: 5px;
        border-radius: 15px;
        border: 1px solid #2a5298;
        background-color: #ffffff;
        color: #2a5298;
        font-size: 0.85rem;
        padding: 6px 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .suggestion-btn:hover {
        background-color: #2a5298;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONVERSATION ASSISTANT UTILS
# =============================================================================

REWRITE_SYSTEM_PROMPT = """Bạn là một trợ lý ảo chuyên nghiệp. Nhiệm vụ của bạn là đọc câu hỏi mới của người dùng cùng với lịch sử trò chuyện trước đó để quyết định xem câu hỏi này có cần được viết lại hay không.
Nếu câu hỏi cuối cùng liên quan đến ngữ cảnh của lịch sử trò chuyện (sử dụng các đại từ như "đó", "nó", "ông ấy", "điều này", hoặc hỏi tiếp về chủ đề trước đó mà không nêu lại tên cụ thể), hãy viết lại câu hỏi đó thành một câu hỏi đầy đủ, độc lập và rõ ràng hơn để hệ thống tìm kiếm (RAG) có thể truy vấn chính xác thông tin.
Nếu câu hỏi cuối cùng đã là một câu hỏi độc lập và không cần thêm ngữ cảnh từ lịch sử trò chuyện, hãy giữ nguyên câu hỏi đó.

Ví dụ:
Lịch sử trò chuyện:
- User: Luật Phòng chống ma tuý 2021 có hiệu lực từ khi nào?
- Assistant: Luật này có hiệu lực từ ngày 01 tháng 01 năm 2022.
Câu hỏi mới: Ai có thẩm quyền áp dụng biện pháp cai nghiện bắt buộc theo luật đó?
-> Viết lại: Ai có thẩm quyền áp dụng biện pháp cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?

YÊU CẦU: Chỉ trả về duy nhất câu hỏi đã viết lại (hoặc câu hỏi gốc nếu không cần viết lại), không thêm bất kỳ lời dẫn hay giải thích nào khác."""

def rewrite_query(query: str, chat_history: list) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi dựa trên lịch sử hội thoại.
    """
    if not chat_history:
        return query
        
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return query

    try:
        # Xây dựng context lịch sử cho LLM
        history_text = ""
        for msg in chat_history[-5:]: # Chỉ lấy tối đa 5 tin nhắn gần nhất
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        prompt = f"Lịch sử trò chuyện:\n{history_text}\nCâu hỏi mới: {query}\n\n-> Viết lại:"

        if openai_key.startswith("sk-or-"):
            client = OpenAI(api_key=openai_key, base_url="https://openrouter.ai/api/v1")
            model_name = "openai/gpt-4o-mini"
        else:
            client = OpenAI(api_key=openai_key)
            model_name = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten
    except Exception as e:
        return query


SUGGEST_SYSTEM_PROMPT = """Bạn là một chuyên gia tư vấn pháp lý và tin tức về ma túy. Dựa trên câu hỏi của người dùng và câu trả lời của trợ lý dưới đây, hãy gợi ý đúng 3 câu hỏi gợi ý tiếp theo (follow-up questions) ngắn gọn, thực tế và mang tính đào sâu hơn để người dùng có thể click chọn hỏi tiếp.
YÊU CẦU ĐỊNH DẠNG: Chỉ trả về đúng 3 dòng câu hỏi, mỗi dòng là một câu hỏi độc lập, không đánh số thứ tự (không ghi 1. 2. 3.), không có bất kỳ ký tự nào khác ở đầu câu ngoài chữ cái.

Ví dụ:
Q: Quy trình cai nghiện bắt buộc như thế nào?
A: [Nội dung câu trả lời...]
Đầu ra:
Ai có thẩm quyền đưa đi cai nghiện bắt buộc?
Thời gian cai nghiện bắt buộc tối đa là bao lâu?
Chi phí cai nghiện bắt buộc do ai chi trả?"""

def generate_suggested_questions(query: str, answer: str) -> list[str]:
    """
    Tự động sinh 3 câu hỏi gợi ý tiếp theo dựa trên câu trả lời vừa sinh ra.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return []
    try:
        if openai_key.startswith("sk-or-"):
            client = OpenAI(api_key=openai_key, base_url="https://openrouter.ai/api/v1")
            model_name = "openai/gpt-4o-mini"
        else:
            client = OpenAI(api_key=openai_key)
            model_name = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {query}\nAssistant Answer: {answer}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        questions = []
        for line in content.split("\n"):
            line = line.strip()
            # Bỏ dấu gạch đầu dòng hoặc số nếu mô hình lỡ sinh ra
            line = line.lstrip("-*•1234. ")
            if line:
                questions.append(line)
        return questions[:3]
    except Exception:
        return []


# =============================================================================
# MAIN UI
# =============================================================================

# Giao diện Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #1e3c72;'>⚙️ CẤU HÌNH PIPELINE</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Chọn Top K chunks và Reranking
top_k = st.sidebar.slider("Số lượng tài liệu trích xuất (Top K Chunks)", min_value=1, max_value=10, value=5)
use_reranking = st.sidebar.toggle("Sử dụng Reranking (BM25 + Dense Fusion)", value=True)

# Thống kê hệ thống thông tin
st.sidebar.markdown("### 📊 THÔNG TIN PIPELINE")
st.sidebar.info("""
- **Cơ sở dữ liệu:** ChromaDB (Local Store)
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
- **Reranker:** RRF (Reciprocal Rank Fusion)
- **Fallback Engine:** PageIndex Vectorless
""")

if st.sidebar.button("🧹 Xóa lịch sử trò chuyện", use_container_width=True):
    st.session_state.messages = []
    st.session_state.suggested_questions = []
    st.rerun()

# Phần thân ứng dụng
st.markdown("<div class='title-text'>⚖️ Luật Phòng chống Ma túy & Tin tức RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Xây dựng trên nền tảng RAG hoàn chỉnh (Semantic & Lexical Hybrid Search + Reranking + PageIndex Fallback + LLM Generation & Citation)</div>", unsafe_allow_html=True)

# Khởi tạo state tin nhắn và câu hỏi gợi ý nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Hiển thị lịch sử trò chuyện
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Hiển thị nguồn của các tin nhắn từ Assistant nếu có
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Xem nguồn tài liệu tham khảo"):
                st.markdown(f"**Phương thức truy vấn:** "
                            f"<span class='{'badge-hybrid' if msg['retrieval_source'] == 'hybrid' else 'badge-pageindex'}'>"
                            f"{msg['retrieval_source'].upper()}</span>", unsafe_allow_html=True)
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("metadata", {}).get("source", "Nguồn không xác định")
                    doc_type = src.get("metadata", {}).get("type", "unknown")
                    score = src.get("score", 0.0)
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">Tài liệu {idx} — {source_name} (Loại: {doc_type} | Điểm tương đồng: {score:.3f})</div>
                        <div class="source-content">{src['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Khai báo chat input luôn ở cuối trang
input_query = st.chat_input("Hãy hỏi tôi bất kỳ điều gì về luật phòng chống ma túy hoặc tin tức nghệ sĩ...")

# Xác định câu hỏi cần xử lý (từ chat input hoặc từ click nút gợi ý)
active_query = None
if input_query:
    active_query = input_query
    # Xóa gợi ý cũ khi người dùng tự nhập câu hỏi mới
    st.session_state.suggested_questions = []
elif st.session_state.get("clicked_suggestion"):
    active_query = st.session_state.clicked_suggestion
    del st.session_state.clicked_suggestion
    st.session_state.suggested_questions = []

if active_query:
    # Hiển thị tin nhắn người dùng
    st.chat_message("user").markdown(active_query)
    
    # Lưu tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": active_query})
    
    # Thực hiện truy vấn và sinh câu trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Bước 1: Viết lại câu hỏi nếu là follow-up
        with st.spinner("Đang phân tích câu hỏi và liên kết ngữ cảnh..."):
            rewritten_query = rewrite_query(active_query, st.session_state.messages[:-1])
            if rewritten_query != active_query:
                st.caption(f"🔄 Câu hỏi được liên kết ngữ cảnh: *\"{rewritten_query}\"*")
                
        # Bước 2: Gọi RAG Pipeline để sinh câu trả lời
        with st.spinner("Đang tìm kiếm thông tin và tạo câu trả lời..."):
            try:
                # Trích xuất lịch sử trò chuyện sạch (chỉ giữ lại role và content)
                cleaned_history = []
                for msg in st.session_state.messages[:-1]:
                    cleaned_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                # Gọi hàm generator với lịch sử trò chuyện để làm conversation memory
                result = generate_with_citation(rewritten_query, top_k=top_k, chat_history=cleaned_history)
                answer = result["answer"]
                sources = result["sources"]
                ret_source = result["retrieval_source"]
                
                # Hiển thị câu trả lời
                message_placeholder.markdown(answer)
                
                # Hiển thị nguồn tham khảo
                with st.expander("📚 Xem nguồn tài liệu tham khảo"):
                    st.markdown(f"**Phương thức truy vấn:** "
                                f"<span class='{'badge-hybrid' if ret_source == 'hybrid' else 'badge-pageindex'}'>"
                                f"{ret_source.upper()}</span>", unsafe_allow_html=True)
                    for idx, src in enumerate(sources, 1):
                        source_name = src.get("metadata", {}).get("source", "Nguồn không xác định")
                        doc_type = src.get("metadata", {}).get("type", "unknown")
                        score = src.get("score", 0.0)
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">Tài liệu {idx} — {source_name} (Loại: {doc_type} | Điểm tương đồng: {score:.3f})</div>
                            <div class="source-content">{src['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tạo câu hỏi gợi ý tiếp theo (follow-up suggestions)
                with st.spinner("Đang sinh câu hỏi gợi ý tiếp theo..."):
                    suggestions = generate_suggested_questions(rewritten_query, answer)
                    st.session_state.suggested_questions = suggestions
                
                # Lưu câu trả lời cùng thông tin nguồn vào lịch sử trò chuyện
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieval_source": ret_source
                })
                
                # Rerun để hiển thị câu hỏi gợi ý ngay lập tức ở cuối
                st.rerun()
                
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi khi sinh câu trả lời: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Hiển thị nút gợi ý câu hỏi tiếp theo (nếu có) dưới cùng của cuộc hội thoại
if st.session_state.suggested_questions:
    st.markdown("<p style='font-size: 0.9rem; font-weight: bold; color: #1e3c72; margin-top: 15px; margin-bottom: 5px;'>💡 Câu hỏi gợi ý tiếp theo (Follow-up):</p>", unsafe_allow_html=True)
    
    # Hiển thị các nút bấm gợi ý dưới dạng các cột ngang
    cols = st.columns(len(st.session_state.suggested_questions))
    for idx, q in enumerate(st.session_state.suggested_questions):
        if cols[idx].button(q, key=f"suggest_{idx}", use_container_width=True):
            st.session_state.clicked_suggestion = q
            st.rerun()
