import os
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://www.gutenberg.org"
START_URL = "https://www.gutenberg.org/ebooks/bookshelf/637?start_index=2651"
OUTPUT_DIR = "dataset"
DELAY = 0.01


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        time.sleep(DELAY)
        return BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"获取页面失败 {url}: {e}")
        return None

def download_book(book_url, title):
    """进入书籍详情页查找并下载纯文本版本"""
    book_id = book_url.split("/")[-1]
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title).strip()
    filename = f"{book_id}_{safe_title}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"文件已存在，跳过: {filename}")
        return

    print(f"正在分析书籍: {title} (ID: {book_id})...")
    soup = get_soup(book_url)
    if not soup:
        return


    download_link = None


    for link in soup.select("table.files td.content a"):
        href = link.get("href")
        text = link.get_text()
        type_cell = link.find_parent("tr").select_one("td.content")


        if ".txt" in href or "Plain Text" in text or "text/plain" in str(link):
            download_link = urljoin(BASE_URL, href)
            break


    if not download_link:
       download_link = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    if download_link:
        try:
            print(f"  -> 找到下载链接: {download_link}")
            r = requests.get(download_link, headers=HEADERS, timeout=15)
            r.raise_for_status()


            r.encoding = r.apparent_encoding if r.encoding == 'ISO-8859-1' else r.encoding

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"  -> 下载成功: {filename}")
        except Exception as e:
            print(f"  -> 下载失败: {e}")
    else:
        print(f"  -> 未找到纯文本下载链接")

def scrape_bookshelf(start_url):
    ensure_dir(OUTPUT_DIR)
    current_url = start_url

    page_num = 1
    while current_url:
        print(f"\n正在处理第 {page_num} 页书籍列表...")
        soup = get_soup(current_url)
        if not soup:
            break


        book_links = soup.select("li.booklink a.link")

        for link in book_links:
            href = link.get("href")
            title_span = link.select_one("span.title")
            title = title_span.get_text() if title_span else "Unknown Title"

            if href and "/ebooks/" in href:
                full_book_url = urljoin(BASE_URL, href)
                download_book(full_book_url, title)


        next_link = None

        candidates = soup.select("a")
        for a_tag in candidates:

            if a_tag.get_text(strip=True) == "Next" or \
               a_tag.get("title") == "Next" or \
               a_tag.get("title") == "Go to the next page of results" or \
               a_tag.get("accesskey") == "n":
                next_link = a_tag
                break


        if not next_link:
             for a_tag in candidates:
                if "Next" in a_tag.get_text(strip=True):
                    next_link = a_tag
                    break

        if next_link:

            next_href = next_link.get("href")
            current_url = urljoin(BASE_URL, next_href)
            print(f"发现下一页: {current_url}")
            page_num += 1
        else:
            print("没有下一页了，爬取结束。")
            current_url = None

if __name__ == "__main__":
    print("开始从 Project Gutenberg 下载 Poetry 类书籍...")
    scrape_bookshelf(START_URL)
    print("任务完成!")
