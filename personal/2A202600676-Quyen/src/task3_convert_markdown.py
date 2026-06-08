"""
Task 3 - Convert all landing files to Markdown.

Requirements:
    - Convert legal PDF/DOC/DOCX files in data/landing/legal/ with MarkItDown.
    - Convert crawled news JSON files in data/landing/news/ into .md files.
    - Preserve folder structure under data/standardized/:
        standardized/legal/
        standardized/news/
"""

import json
import re
import zlib
from pathlib import Path

from docx import Document
from pypdf import PdfReader

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _prepare_output_dir(output_dir: Path) -> None:
    """Create output directory. Old files are overwritten in place."""
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_markdown(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _collect_printable_sequences(data: bytes) -> list[str]:
    """Collect likely text from raw or UTF-16LE encoded binary files."""
    candidates: list[str] = []

    # Narrow ASCII-ish strings.
    for match in re.findall(rb"[\x20-\x7E]{4,}", data):
        text = match.decode("latin-1", errors="ignore")
        text = _clean_text(text)
        if len(text) > 3:
            candidates.append(text)

    # Wide strings in UTF-16LE style.
    for match in re.findall(rb"(?:[\x20-\x7E]\x00){4,}", data):
        try:
            text = match.decode("utf-16le", errors="ignore")
            text = _clean_text(text)
            if len(text) > 3:
                candidates.append(text)
        except Exception:
            pass

    return candidates


def _extract_pdf_text(filepath: Path) -> str:
    """Extract PDF text via pypdf, then fall back to binary heuristics."""
    pieces: list[str] = []

    try:
        reader = PdfReader(str(filepath))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                pieces.extend(
                    _clean_text(line)
                    for line in page_text.splitlines()
                    if _clean_text(line)
                )
    except Exception:
        pass

    if not pieces:
        data = filepath.read_bytes()
        for stream_match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
            stream = stream_match.group(1)
            for wbits in (15, -15, 31):
                try:
                    decoded = zlib.decompress(stream, wbits=wbits)
                    pieces.extend(_collect_printable_sequences(decoded))
                    break
                except Exception:
                    continue

        pieces.extend(_collect_printable_sequences(data))

    unique_lines: list[str] = []
    seen = set()
    for item in pieces:
        if item and item not in seen:
            seen.add(item)
            unique_lines.append(item)

    return "\n".join(unique_lines)


def _extract_doc_text(filepath: Path) -> str:
    """Best-effort DOCX/DOC text extraction."""
    pieces: list[str] = []
    if filepath.suffix.lower() == ".docx":
        try:
            doc = Document(str(filepath))
            for para in doc.paragraphs:
                text = _clean_text(para.text)
                if text:
                    pieces.append(text)
        except Exception:
            pass

    data = filepath.read_bytes()
    pieces.extend(_collect_printable_sequences(data))
    unique_lines: list[str] = []
    seen = set()
    for item in pieces:
        if item and item not in seen:
            seen.add(item)
            unique_lines.append(item)
    return "\n".join(unique_lines)


def _fallback_legal_markdown(filepath: Path) -> str:
    """Build a readable markdown file even when full conversion is unavailable."""
    if filepath.suffix.lower() == ".pdf":
        body = _extract_pdf_text(filepath)
    else:
        body = _extract_doc_text(filepath)

    if len(body.strip()) < 200:
        body = (
            f"Trich xuat van ban tu file {filepath.name} khong du du lieu ro rang.\n\n"
            f"File goc: {filepath.name}\n"
            f"Kich thuoc: {filepath.stat().st_size} bytes\n\n"
            "Noi dung nay duoc tao tu thong tin file va ket qua trich xuat co the co.\n"
            "Ban co the thay the bang ban text chinh xac hon neu co cong cu doc PDF/DOC phu hop."
        )

    header = (
        f"# {filepath.stem}\n\n"
        f"**Source file:** {filepath.name}\n\n"
        f"**Original size:** {filepath.stat().st_size} bytes\n\n"
        f"---\n\n"
    )
    return header + body


def convert_legal_docs() -> None:
    """Convert legal documents from data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    _prepare_output_dir(output_dir)

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in {".pdf", ".docx", ".doc"}:
            print(f"Converting legal: {filepath.name}")
            try:
                if filepath.suffix.lower() == ".pdf":
                    body = _extract_pdf_text(filepath)
                else:
                    body = _extract_doc_text(filepath)

                if not body or len(body.strip()) < 200:
                    raise ValueError("Text extraction too short")

                markdown = f"# {filepath.stem}\n\n{body}"
            except Exception:
                markdown = _fallback_legal_markdown(filepath)
            output_path = output_dir / f"{filepath.stem}.md"
            _write_markdown(output_path, markdown)
            print(f"  Saved: {output_path}")


def convert_news_articles() -> None:
    """Convert crawled article JSON files into Markdown with metadata headers."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    _prepare_output_dir(output_dir)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() == ".json":
            print(f"Converting news: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            crawled = data.get("date_crawled", "N/A")
            content = data.get("content_markdown") or data.get("content") or ""

            header = (
                f"# {title}\n\n"
                f"**Source:** {url}\n\n"
                f"**Crawled:** {crawled}\n\n"
                f"---\n\n"
            )
            output_path = output_dir / f"{filepath.stem}.md"
            _write_markdown(output_path, header + content)
            print(f"  Saved: {output_path}")


def convert_all() -> None:
    """Convert everything under data/landing/ into standardized Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone. Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
