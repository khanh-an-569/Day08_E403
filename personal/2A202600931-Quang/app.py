import streamlit as st
from src.task10_generation import RAGGenerator

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ Thống Tra Cứu RAG", page_icon="⚖️", layout="centered")

st.title("⚖️ Tra cứu Pháp Luật & Tin tức AI")
st.markdown("Hệ thống sẽ tự động tìm kiếm thông tin trong các Điều luật và Bài báo đã lưu, sau đó dùng Gemini AI để đúc kết lại thành câu trả lời.")

# Hàm tải mô hình RAG (Dùng cache để không phải load lại mỗi lần người dùng bấm nút)
@st.cache_resource
def load_rag_system():
    return RAGGenerator()

try:
    generator = load_rag_system()
    st.success("✅ Hệ thống RAG đã khởi tạo thành công!")
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo hệ thống (Vui lòng kiểm tra API Key): {e}")
    st.stop()

# Form nhập liệu
st.markdown("---")
query = st.text_input("✍️ Đặt câu hỏi của bạn:", placeholder="Ví dụ: Ca sĩ Chi Dân bị bắt ở đâu? Khung hình phạt là bao nhiêu năm tù?")

if st.button("Tìm Kiếm & Trả Lời", type="primary"):
    if query.strip() == "":
        st.warning("⚠️ Vui lòng nhập câu hỏi trước khi tìm kiếm!")
    else:
        with st.spinner("🤖 Máy đang đọc hàng nghìn trang tài liệu và suy nghĩ..."):
            try:
                answer = generator.generate_answer(query)
                st.markdown("### 💡 Câu trả lời của AI:")
                st.info(answer)
                st.balloons() # Hiệu ứng chúc mừng khi trả lời xong
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
