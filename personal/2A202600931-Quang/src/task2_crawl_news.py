import os
import json
import asyncio
from datetime import datetime
from crawl4ai import AsyncWebCrawler

URLS = [
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://baovephapluat.vn/cong-to-kiem-sat-tu-phap/truy-to/truy-to-ca-si-chi-dan-va-226-bi-can-trong-vu-an-ma-tuy-lien-quan-den-tiep-vien-hang-khong-196299.html",
    "https://pcmatuy.bocongan.gov.vn/Tin-t%E1%BB%A9c-s%E1%BB%B1-ki%E1%BB%87n/articleType/ArticleView/articleId/2019/ca-s-chi-dn-v-anh-rut-r-bn-nhu-t-ma-ty-v-s-dng",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://laodong.vn/van-hoa-giai-tri/hinh-anh-an-tay-andrea-aybar-khi-mat-het-su-nghiep-1422675.ldo",
    "https://vov.vn/phap-luat/vu-an-ma-tuy-lien-quan-den-ca-si-chi-dan-vi-sao-nguoi-mau-an-tay-bi-truy-to-post1281931.vov",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://tienphong.vn/nghe-si-dinh-ma-tuy-khoang-trong-sau-nhung-cu-truot-nga-post1845503.tpo",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://baolaocai.vn/bao-dong-tinh-trang-nghe-si-dung-ma-tuy-va-nhung-he-luy-voi-xa-hoi-post900028.html",
    "https://thanhnien.vn/nghe-si-tu-nguyen-xet-nghiem-ma-tuy-showbiz-dang-bat-an-den-muc-nao-185260526105918638.htm",
    "https://laodong.vn/phap-luat/bo-cong-an-khoi-to-bat-tam-giam-rapper-binh-gold-1587086.ldo",
    "https://tuoitre.vn/rapper-binh-gold-duong-tinh-ma-tuy-khi-lai-xe-co-dau-hieu-gay-roi-trat-tu-cong-cong-20250724080230866.htm",
    "https://vnexpress.net/rapper-binh-gold-tiep-tuc-duong-tinh-voi-ma-tuy-lai-cuop-taxi-4919259.html"
]

OUTPUT_DIR = os.path.join("data", "landing", "news")

async def crawl_articles():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize the AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(URLS):
            try:
                print(f"Crawling [{i+1}/{len(URLS)}]: {url}")
                result = await crawler.arun(url=url)
                
                if result.markdown:
                    # We save the result to a JSON file with metadata
                    article_data = {
                        "url": url,
                        "crawl_date": datetime.now().isoformat(),
                        "content": result.markdown
                    }
                    
                    filename = f"article_{i+1}.json"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(article_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"  -> Saved to {filepath}")
                else:
                    print(f"  -> No content retrieved for {url}")
            except Exception as e:
                print(f"  -> Error crawling {url}: {e}")

if __name__ == "__main__":
    asyncio.run(crawl_articles())
