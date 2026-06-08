"""
Task 2 - Crawl news articles about Vietnamese artists related to drugs.

The goal is to save clean article text, not raw HTML or site chrome.
Each output file should contain article metadata and the extracted body text.
"""

import asyncio
import html
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory() -> None:
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://tienphong.vn/hanh-trinh-phe-ma-tuy-roi-giet-nguoi-cua-ca-si-chau-viet-cuong-post1095287.tpo",
    "https://tuoitre.vn/dien-vien-hai-hiep-ga-bi-bat-qua-tang-tang-tru-ma-tuy-198845.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
]


def _clean_whitespace(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_title(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for selector in [
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            return _clean_whitespace(tag["content"])

    h1 = soup.find("h1")
    if h1:
        text = _clean_whitespace(h1.get_text(" ", strip=True))
        if text:
            return text

    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    if match:
        title = _clean_whitespace(match.group(1))
        if title:
            return title
    return "Unknown Title"


def _looks_like_boilerplate(line: str) -> bool:
    lower = line.lower()
    boilerplate_patterns = [
        "bình luận",
        "chia sẻ",
        "hotline",
        "đăng nhập",
        "đăng ký",
        "đặt báo",
        "theo dõi",
        "liên hệ",
        "quảng cáo",
        "xem thêm",
        "tin liên quan",
        "chuyên mục",
        "copyright",
        "all rights reserved",
        "tối đa:",
        "gửi bình luận",
        "tặng sao",
        "lịch sử giao dịch",
        "thoát",
        "cài đặt tài khoản",
    ]
    return any(pattern in lower for pattern in boilerplate_patterns)


def _select_article_node(soup: BeautifulSoup, domain: str):
    selectors_by_domain = {
        "vnexpress.net": [
            ".fck_detail",
            '[itemprop="articleBody"]',
            "article",
        ],
        "tuoitre.vn": [
            '[itemprop="articleBody"]',
            ".detail-content",
            ".detail__cmain",
            "article",
        ],
        "nld.com.vn": [
            '[itemprop="articleBody"]',
            ".detail-content",
            ".detail__main",
            "article",
        ],
        "vov.vn": [
            ".article-content",
            '[itemprop="articleBody"]',
            "article",
        ],
        "tienphong.vn": [
            "article",
            ".article-content",
            '[itemprop="articleBody"]',
        ],
        "laodong.vn": [
            '[itemprop="articleBody"]',
            ".detail-content",
            ".article-content",
            "article",
        ],
    }

    selectors = selectors_by_domain.get(domain, []) + [
        '[itemprop="articleBody"]',
        ".detail-content",
        ".article-content",
        ".fck_detail",
        "article",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = _clean_whitespace(node.get_text(" ", strip=True))
            if len(text) >= 300:
                return node
    return None


def _extract_article_text(raw_html: str, domain: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    node = _select_article_node(soup, domain)
    if node is None:
        node = soup.body or soup

    # Remove obvious non-article noise before text extraction.
    for tag in node.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    text = node.get_text("\n", strip=True)
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = _clean_whitespace(raw_line)
        if not line or _looks_like_boilerplate(line):
            continue
        if len(line) <= 1:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return a dict containing metadata and content.
    """
    title = "Unknown Title"
    content = ""
    raw_html = ""

    # Try Crawl4AI first when available, then fall back to requests.
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        raw_html = response.text
        domain = urlparse(url).netloc.lower()
        title = _extract_title(raw_html)
        content = _extract_article_text(raw_html, domain)

        # If the first pass is too noisy, use Crawl4AI as a fallback only.
        if len(content.strip()) < 500:
            try:
                from crawl4ai import AsyncWebCrawler

                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    if getattr(result, "success", False):
                        raw_html = getattr(result, "html", "") or raw_html
                        if raw_html:
                            title = _extract_title(raw_html)
                        crawled_text = getattr(result, "markdown", "") or ""
                        if len(crawled_text.strip()) > len(content.strip()):
                            content = crawled_text
                    else:
                        raise RuntimeError(getattr(result, "error_message", "crawl failed"))
            except Exception:
                pass
    except Exception as exc:
        raise RuntimeError(f"Failed to crawl {url}: {exc}") from exc

    if not content:
        content = title or url

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
        "content": content,
    }


async def crawl_all() -> None:
    """Crawl every URL in ARTICLE_URLS and save each article as JSON."""
    setup_directory()

    for stale_file in DATA_DIR.glob("article_*.json"):
        try:
            stale_file.unlink()
        except PermissionError:
            pass

    seen = set()
    unique_urls = []
    for url in ARTICLE_URLS:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    for i, url in enumerate(unique_urls, 1):
        print(f"[{i}/{len(unique_urls)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Saved: {filepath}")
        except Exception as exc:
            print(f"  Failed: {exc}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay.")
    else:
        asyncio.run(crawl_all())
