"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import os
import sys
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() == ".pdf":
            print(f"Converting PDF with pdfplumber: {filepath.name}")
            try:
                import pdfplumber
                text_content = []
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text(x_tolerance=1.5)
                        if page_text:
                            text_content.append(page_text)
                
                full_text = "\n\n".join(text_content)
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(full_text, encoding="utf-8")
                print(f"  [OK] Saved: {output_path}")
            except Exception as e:
                print(f"  [Error] Failed to convert PDF {filepath.name}: {e}")
                
        elif filepath.suffix.lower() in (".docx", ".doc"):
            print(f"Converting Word with MarkItDown: {filepath.name}")
            temp_docx = None
            target_path = filepath
            
            if filepath.suffix.lower() == ".doc":
                try:
                    import win32com.client
                    word = win32com.client.Dispatch("Word.Application")
                    word.Visible = False
                    
                    doc_path = os.path.abspath(filepath)
                    temp_docx = doc_path + "x"
                    
                    doc = word.Documents.Open(doc_path)
                    doc.SaveAs2(temp_docx, FileFormat=16)  # 16 = wdFormatXMLDocument (.docx)
                    doc.Close()
                    word.Quit()
                    
                    target_path = Path(temp_docx)
                    print(f"  [Info] Converted .doc to temporary .docx: {target_path.name}")
                except Exception as doc_err:
                    print(f"  [Error] Failed to convert legacy .doc to .docx via Word COM: {doc_err}")
            
            try:
                result = md.convert(str(target_path))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(result.text_content, encoding="utf-8")
                print(f"  [OK] Saved: {output_path}")
            except Exception as e:
                print(f"  [Error] Failed to convert Word to markdown: {e}")
            finally:
                if temp_docx and os.path.exists(temp_docx):
                    try:
                        os.remove(temp_docx)
                    except Exception:
                        pass


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Thêm metadata header
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                content = header + data.get("content_markdown", "")
                output_path.write_text(content, encoding="utf-8")
                print(f"  [OK] Saved: {output_path}")
            except Exception as e:
                print(f"  [Error] Failed to convert {filepath.name}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output tai:", OUTPUT_DIR)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    convert_all()
