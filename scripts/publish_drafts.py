#!/usr/bin/env python3
"""
publish_drafts.py - Check _drafts/ for articles past their publish date.
Move due drafts to articles/, update index.html, en/index.html, and sitemap.xml.
Exit 0 if published, 1 if nothing to publish.
"""

import os
import re
import shutil
import sys
from datetime import datetime, date
from xml.etree import ElementTree as ET

BLOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFTS_DIR = os.path.join(BLOG_DIR, "_drafts")
ARTICLES_DIR = os.path.join(BLOG_DIR, "articles")
EN_ARTICLES_DIR = os.path.join(BLOG_DIR, "en", "articles")
INDEX_CN = os.path.join(BLOG_DIR, "index.html")
INDEX_EN = os.path.join(BLOG_DIR, "en", "index.html")
SITEMAP_PATH = os.path.join(BLOG_DIR, "sitemap.xml")

DRAFT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.html$")


def extract_block(content, tag):
    """Extract content between <!-- TAG ... --> markers."""
    pattern = re.compile(
        rf"<!--\s*{tag}\s*\n(.*?)\n\s*-->", re.DOTALL
    )
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return None


def insert_into_index(index_path, entry_html):
    """Insert an article entry after <main class="article-list">."""
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = '<main class="article-list">'
    if marker not in content:
        print(f"  WARNING: marker not found in {index_path}")
        return False

    # Insert after the marker (with newline)
    new_content = content.replace(
        marker,
        marker + "\n        " + entry_html,
        1,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def clean_draft_content(content):
    """Remove INDEX_ENTRY and EN_INDEX_ENTRY comment blocks from draft."""
    content = re.sub(
        r"<!--\s*INDEX_ENTRY\s*\n.*?\n\s*-->\s*\n?",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"<!--\s*EN_INDEX_ENTRY\s*\n.*?\n\s*-->\s*\n?",
        "",
        content,
        flags=re.DOTALL,
    )
    return content


def publish_draft(filename):
    """Publish a single draft article. Returns True on success."""
    match = DRAFT_PATTERN.match(filename)
    if not match:
        print(f"  SKIP: filename doesn't match pattern: {filename}")
        return False

    publish_date_str, slug = match.groups()
    publish_date = datetime.strptime(publish_date_str, "%Y-%m-%d").date()

    if publish_date > date.today():
        print(f"  SKIP: {filename} scheduled for {publish_date} (future)")
        return False

    print(f"  PUBLISHING: {filename} (date: {publish_date})")

    draft_path = os.path.join(DRAFTS_DIR, filename)
    with open(draft_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract index entries
    cn_entry = extract_block(content, "INDEX_ENTRY")
    en_entry = extract_block(content, "EN_INDEX_ENTRY")

    # Clean draft content (remove comment blocks)
    clean_content = clean_draft_content(content)

    # Write to articles/
    dest_path = os.path.join(ARTICLES_DIR, f"{slug}.html")
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
    print(f"    -> articles/{slug}.html")

    # Check for English draft in _drafts/en/
    en_draft_path = os.path.join(DRAFTS_DIR, "en", filename)
    if os.path.exists(en_draft_path):
        with open(en_draft_path, "r", encoding="utf-8") as f:
            en_content = f.read()
        en_dest_path = os.path.join(EN_ARTICLES_DIR, f"{slug}.html")
        with open(en_dest_path, "w", encoding="utf-8") as f:
            f.write(en_content)
        os.remove(en_draft_path)
        print(f"    -> en/articles/{slug}.html (from en/ draft)")

    # Update Chinese index
    if cn_entry:
        if insert_into_index(INDEX_CN, cn_entry):
            print(f"    -> Updated index.html")
        else:
            print(f"    -> WARNING: failed to update index.html")
    else:
        print(f"    -> WARNING: no INDEX_ENTRY found, index.html not updated")

    # Update English index
    if en_entry:
        if insert_into_index(INDEX_EN, en_entry):
            print(f"    -> Updated en/index.html")
        else:
            print(f"    -> WARNING: failed to update en/index.html")

    # Remove draft file
    os.remove(draft_path)
    print(f"    -> Removed _drafts/{filename}")

    # Update sitemap
    has_en = en_entry is not None
    update_sitemap(slug, publish_date_str, has_en)

    return True


def update_sitemap(slug, publish_date_str, has_en=False):
    """Add new article entry to sitemap.xml."""
    if not os.path.exists(SITEMAP_PATH):
        print(f"    -> WARNING: sitemap.xml not found, skipping")
        return

    # Parse existing sitemap
    ET.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    tree = ET.parse(SITEMAP_PATH)
    root = tree.getroot()

    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    # Add Chinese article
    cn_url = ET.SubElement(root, '{http://www.sitemaps.org/schemas/sitemap/0.9}url')
    ET.SubElement(cn_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text = \
        f'https://worldsensetech.com/articles/{slug}.html'
    ET.SubElement(cn_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod').text = publish_date_str
    ET.SubElement(cn_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq').text = 'monthly'
    ET.SubElement(cn_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}priority').text = '0.8'

    # Add English article if exists
    if has_en:
        en_url = ET.SubElement(root, '{http://www.sitemaps.org/schemas/sitemap/0.9}url')
        ET.SubElement(en_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text = \
            f'https://worldsensetech.com/en/articles/{slug}.html'
        ET.SubElement(en_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod').text = publish_date_str
        ET.SubElement(en_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq').text = 'monthly'
        ET.SubElement(en_url, '{http://www.sitemaps.org/schemas/sitemap/0.9}priority').text = '0.7'

    # Write back with XML declaration
    tree.write(SITEMAP_PATH, encoding='UTF-8', xml_declaration=True)
    print(f"    -> Updated sitemap.xml")


def main():
    if not os.path.isdir(DRAFTS_DIR):
        print("No _drafts/ directory found.")
        sys.exit(1)

    drafts = sorted(os.listdir(DRAFTS_DIR))
    html_drafts = [f for f in drafts if DRAFT_PATTERN.match(f)]

    if not html_drafts:
        print("No draft files found in _drafts/.")
        sys.exit(1)

    published = 0
    for filename in html_drafts:
        if publish_draft(filename):
            published += 1

    print(f"\nDone. Published: {published}")
    sys.exit(0 if published > 0 else 1)


if __name__ == "__main__":
    main()
