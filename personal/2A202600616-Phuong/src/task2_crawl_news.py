"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    # Hãy điền các URL của bạn ở đây, ví dụ:
    "https://laodong.vn/ban-doc/tai-sao-ban-dau-ca-si-miu-le-bi-xu-ly-hanh-chinh-nhung-sau-do-lai-khoi-to-hinh-su-1703111.ldo",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://laodong.vn/phap-luat/bo-cong-an-khoi-to-bat-tam-giam-rapper-binh-gold-1587086.ldo",
    "https://tienphong.vn/hanh-trinh-phe-ma-tuy-roi-giet-nguoi-cua-ca-si-chau-viet-cuong-post1095287.tpo",
    "https://laodong.vn/phap-luat/toan-canh-vu-ca-si-chau-viet-cuong-nhet-toi-vao-mieng-ban-tinh-661444.ldo",
    "https://tuoitre.vn/dien-vien-hai-hiep-ga-bi-bat-qua-tang-tang-tru-ma-tuy-198845.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://tienphong.vn/hanh-trinh-phe-ma-tuy-roi-giet-nguoi-cua-ca-si-chau-viet-cuong-post1095287.tpo",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://laodong.vn/van-hoa-giai-tri/hinh-anh-an-tay-andrea-aybar-khi-mat-het-su-nghiep-1422675.ldo"
]


def extract_clean_text(html_content: str) -> tuple[str, str]:
    """Trích xuất tiêu đề và nội dung chính của bài báo, loại bỏ quảng cáo, sidebar và menu."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Lấy tiêu đề
    title = "Unknown Title"
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        
    # Loại bỏ các thẻ không chứa nội dung văn bản chính
    for element in list(soup(["script", "style", "nav", "header", "footer", "iframe", "noscript", "aside"])):
        element.decompose()
        
    # Loại bỏ sidebar, menu, comment, ads, sharing widgets
    to_decompose = []
    for element in list(soup.find_all(True)):
        if not hasattr(element, "attrs") or element.attrs is None:
            continue
        classes = element.get("class", [])
        if isinstance(classes, list):
            classes = " ".join(classes).lower()
        else:
            classes = str(classes).lower()
            
        id_val = str(element.get("id", "")).lower()
        
        if any(keyword in classes or keyword in id_val for keyword in [
            "sidebar", "menu", "footer", "header", "comment", "ad-", "advertisement",
            "relation", "related", "recommend", "tag", "social", "share", "toolbar", 
            "breadcrumb", "nav", "banner", "popup"
        ]):
            to_decompose.append(element)
            
    for element in to_decompose:
        try:
            element.decompose()
        except Exception:
            pass
            
    # Danh sách các selector chính chứa nội dung bài báo của báo mạng VN
    selectors = [
        ".fck_detail",
        ".content-detail",
        ".detail-content",
        ".detail__content",
        ".article-content",
        ".maincontent",
        ".cms-body",
        ".detail-ccontent",
        "article",
        "[itemprop='articleBody']",
    ]
    
    content_root = None
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            content_root = element
            break
            
    if not content_root:
        content_root = soup.body if soup.body else soup
        
    # Trích xuất toàn bộ text từ các thẻ tiêu đề và đoạn văn
    text_lines = []
    for p in content_root.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = p.get_text().strip()
        if text and text not in text_lines:
            text_lines.append(text)
            
    content_markdown = "\n\n".join(text_lines)
    return title, content_markdown


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from bs4 import BeautifulSoup
    import requests

    title = "Unknown Title"
    content_markdown = ""

    try:
        from crawl4ai import AsyncWebCrawler
        print(f"  Attempting to crawl with Crawl4AI: {url}")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success:
                if result.html:
                    title, content_markdown = extract_clean_text(result.html)
                else:
                    content_markdown = result.markdown
                    title = result.metadata.get("title", "Unknown Title") if result.metadata else "Unknown Title"
            else:
                print(f"  Crawl4AI failed: {result.error_message}. Trying fallback...")
                raise Exception(result.error_message)
    except Exception as e:
        print(f"  Crawl4AI not working or error: {e}. Falling back to requests & BeautifulSoup...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, headers=headers, timeout=15)
            )
            response.raise_for_status()
            title, content_markdown = extract_clean_text(response.text)
        except Exception as fallback_err:
            print(f"  Fallback failed as well: {fallback_err}")
            raise fallback_err

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            if len(article.get("content_markdown", "")) < 200:
                print(f"  ⚠ Noi dung qua ngan ({len(article.get('content_markdown', ''))} ky tu). Bo qua khong luu file.")
                continue
            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath}")
        except Exception as e:
            print(f"  ✗ Failed to crawl {url}: {e}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
