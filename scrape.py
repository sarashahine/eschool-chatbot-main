#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape a website (static + JS-rendered), extract main content, chunk by tokens, output JSONL ready for embedding.
"""

import asyncio
import json
import time
import hashlib
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path

from bs4 import BeautifulSoup
from readability import Document  # readability-lxml
from playwright.async_api import async_playwright
import tiktoken

# --- CONFIG ---
ROOT_URL = "https://web.myeschoolhome.com/"
OUTPUT_FILE = "eschool_chunks.jsonl"
USER_AGENT = "Mozilla/5.0 (compatible; MyScraper/1.0; +https://example.com/bot)"
REQUEST_DELAY = 1.0  # seconds between page visits
MAX_TOKENS = 400     # max tokens per chunk (embedding-friendly)
OVERLAP_TOKENS = 50  # overlap between chunks

# Which tokenizer/encoding to use
ENCODING = tiktoken.get_encoding("cl100k_base")  # or choose appropriate for your embedding model

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def chunk_text_by_tokens(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """
    Chunk text into pieces with <= max_tokens, using token count (with overlap).
    Returns list of text chunks.
    """
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    total = len(tokens)
    while start < total:
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        if end >= total:
            break
        start = end - overlap
    return chunks

async def fetch_page(playwright, url: str):
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page(user_agent=USER_AGENT)
    try:
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(0.5)
        content = await page.content()
        return content
    finally:
        await browser.close()

def extract_main_text_and_title(html: str):
    """
    Use readability-lxml to extract main article content & title.
    Returns (title, text) or (None, None) if extraction fails / too little text.
    """
    try:
        doc = Document(html)
        summary_html = doc.summary()
        title = doc.short_title()
    except Exception as e:
        # fallback: no readability
        title = None
        summary_html = html

    soup = BeautifulSoup(summary_html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    # normalize whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    # optionally, filter out extremely short content
    if len(text) < 100:  # adjust threshold as needed
        return title, None
    return title, text

def make_id(url: str, chunk_idx: int) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return f"{h}-chunk{chunk_idx}"

async def crawl_site(root_url: str, max_pages=2000):
    visited = set()
    to_visit = {root_url}
    results = []

    async with async_playwright() as playwright:
        while to_visit:
            url = to_visit.pop()
            if url in visited:
                continue
            visited.add(url)
            print("Visiting:", url)
            try:
                html = await fetch_page(playwright, url)
            except Exception as e:
                print("Failed to fetch:", url, e)
                continue

            title, main_text = extract_main_text_and_title(html)
            if main_text:
                chunks = chunk_text_by_tokens(main_text, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS)
                for idx, chunk in enumerate(chunks):
                    if count_tokens(chunk) < 10:
                        continue
                    entry = {
                        "id": make_id(url, idx),
                        "url": url,
                        "page_title": title,
                        "chunk_index": idx,
                        "text": chunk,
                        "crawl_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    results.append(entry)

            # parse links to follow
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                parsed = urlparse(href)
                if parsed.scheme in ("http", "https"):
                    new_url = href
                else:
                    new_url = urljoin(url, href)
                # filter internal only
                if new_url.startswith(root_url) and new_url not in visited:
                    to_visit.add(new_url)

            await asyncio.sleep(REQUEST_DELAY)

            if len(visited) >= max_pages:
                break

    return results

def save_jsonl(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    data = asyncio.run(crawl_site(ROOT_URL))
    print(f"Scraped {len(data)} chunks. Saving to {OUTPUT_FILE}")
    save_jsonl(data, OUTPUT_FILE)

if __name__ == "__main__":
    main()
