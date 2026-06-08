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
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://laodong.vn/phap-luat/bo-cong-an-khoi-to-bat-tam-giam-rapper-binh-gold-1587086.ldo",
    "https://tienphong.vn/hanh-trinh-phe-ma-tuy-roi-giet-nguoi-cua-ca-si-chau-viet-cuong-post1095287.tpo",
    "https://laodong.vn/phap-luat/toan-canh-vu-ca-si-chau-viet-cuong-nhet-toi-vao-mieng-ban-tinh-661444.ldo",
    "https://tuoitre.vn/dien-vien-hai-hiep-ga-bi-bat-qua-tang-tang-tru-ma-tuy-198845.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://laodong.vn/van-hoa-giai-tri/hinh-anh-an-tay-andrea-aybar-khi-mat-het-su-nghiep-1422675.ldo"
]


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

    title = "Unknown Title"
    content_markdown = ""

    try:
        from crawl4ai import AsyncWebCrawler
        print(f"  Attempting to crawl with Crawl4AI: {url}")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success:
                content_markdown = result.markdown
                if result.html:
                    try:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        if soup.title and soup.title.string:
                            title = soup.title.string.strip()
                    except Exception:
                        pass
                
                if title == "Unknown Title" and result.metadata:
                    title = result.metadata.get("title", "Unknown Title")
            else:
                print(f"  Crawl4AI failed: {result.error_message}. Trying fallback...")
                raise Exception(result.error_message)
    except Exception as e:
        print(f"  Crawl4AI not working or error: {e}. Falling back to requests & BeautifulSoup...")
        import requests
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
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            
            # Remove scripts and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
                
            # Basic conversion to plain/markdown content
            text_lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
            # Filter empty lines
            content_markdown = "\n".join([line for line in text_lines if line])
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
            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())