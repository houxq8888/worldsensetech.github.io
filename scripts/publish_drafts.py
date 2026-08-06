#!/usr/bin/env python3
"""
publish_drafts.py - Check _drafts/ for articles past their publish date.
Move due drafts to articles/, and auto-update:
  - index.html (add article entry at top)
  - en/index.html (if English draft exists)
  - categories.html (add entry + increment category count)
  - archive.html (add entry + increment total count)
  - sitemap.xml (add URL)
Exit 0 if published, 1 if nothing to publish.
"""

import os
import re
import sys
from datetime import datetime, date
from xml.etree import ElementTree as ET

BLOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFTS_DIR = os.path.join(BLOG_DIR, "_drafts")
ARTICLES_DIR = os.path.join(BLOG_DIR, "articles")
EN_ARTICLES_DIR = os.path.join(BLOG_DIR, "en", "articles")
INDEX_CN = os.path.join(BLOG_DIR, "index.html")
INDEX_EN = os.path.join(BLOG_DIR, "en", "index.html")
CATEGORIES_PATH = os.path.join(BLOG_DIR, "categories.html")
ARCHIVE_PATH = os.path.join(BLOG_DIR, "archive.html")
SITEMAP_PATH = os.path.join(BLOG_DIR, "sitemap.xml")

DRAFT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.html$")

# Tag-to-category mapping for auto-categorization
TAG_TO_CATEGORY = {
    "世界模型": ("world-model", "世界模型理论"),
    "VLA": ("world-model", "世界模型理论"),
    "合成数据": ("world-model", "世界模型理论"),
    "RSSM": ("world-model", "世界模型理论"),
    "DreamerV3": ("world-model", "世界模型理论"),
    "交互式生成": ("world-model", "世界模型理论"),
    "Sim-to-Real": ("embodied-ai", "具身智能行业"),
    "具身智能": ("embodied-ai", "具身智能行业"),
    "强化学习": ("reinforcement-learning", "强化学习"),
    "MuJoCo": ("tutorial", "实战教程"),
    "实战": ("tutorial", "实战教程"),
    "职业": ("career", "职业思考"),
}

# Default category if no tag matches
DEFAULT_CATEGORY = ("world-model", "世界模型理论")


def extract_metadata(html_content, publish_date_str):
    """Extract title, date, reading_time, tags, excerpt from article HTML."""
    meta = {}

    # Title from <h1>
    h1_match = re.search(r"<h1>(.+?)</h1>", html_content)
    meta["title"] = h1_match.group(1).strip() if h1_match else "Untitled"

    # Article info: "2026年8月6日 · 阅读约12分钟 · 世界模型, VLA, 合成数据"
    info_match = re.search(
        r'<div class="article-info">(.+?)</div>', html_content
    )
    if info_match:
        info_text = info_match.group(1).strip()
        meta["article_info"] = info_text

        # Extract date (Chinese format: 2026年8月6日)
        date_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", info_text)
        meta["date_cn"] = date_match.group(1) if date_match else publish_date_str

        # Extract reading time
        time_match = re.search(r"阅读约(\d+)分钟", info_text)
        meta["reading_time"] = time_match.group(1) if time_match else "10"

        # Extract tags (after the last · )
        parts = info_text.split("·")
        if len(parts) >= 3:
            tags_str = parts[-1].strip()
            meta["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        else:
            meta["tags"] = []
    else:
        meta["date_cn"] = publish_date_str
        meta["reading_time"] = "10"
        meta["tags"] = []

    # Excerpt from meta description
    desc_match = re.search(
        r'<meta name="description" content="(.+?)"', html_content
    )
    meta["excerpt"] = desc_match.group(1).strip() if desc_match else ""

    return meta


def generate_index_entry(slug, meta):
    """Generate HTML for index.html article entry."""
    tags_html = "\n        ".join(
        f'<span class="tag">{t}</span>' for t in meta["tags"]
    )
    info_line = f'{meta["date_cn"]} · 阅读约{meta["reading_time"]}分钟'

    return f"""<article class="article-item">
    <div class="article-meta">{info_line}</div>
    <h2><a href="articles/{slug}.html">{meta["title"]}</a></h2>
    <div>
        {tags_html}
    </div>
    <p class="article-excerpt">{meta["excerpt"]}</p>
</article>"""


def insert_into_index(index_path, entry_html):
    """Insert an article entry after <main class="article-list">."""
    if not os.path.exists(index_path):
        print(f"  WARNING: {index_path} not found")
        return False

    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = '<main class="article-list">'
    if marker not in content:
        print(f"  WARNING: marker not found in {index_path}")
        return False

    new_content = content.replace(
        marker,
        marker + "\n        " + entry_html,
        1,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def determine_category(tags):
    """Determine category from tags using mapping."""
    for tag in tags:
        if tag in TAG_TO_CATEGORY:
            return TAG_TO_CATEGORY[tag]
    return DEFAULT_CATEGORY


def update_categories(slug, title, publish_date_str, category_id, category_name):
    """Add article to categories.html and increment count."""
    if not os.path.exists(CATEGORIES_PATH):
        print(f"  WARNING: categories.html not found")
        return False

    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Increment category count: (X篇) -> (X+1篇)
    count_pattern = re.compile(
        rf'(id="{category_id}".*?category-count">\()(\d+)(篇\))',
        re.DOTALL,
    )
    count_match = count_pattern.search(content)
    if count_match:
        old_count = int(count_match.group(2))
        new_count = old_count + 1
        content = content[:count_match.start(2)] + str(new_count) + content[count_match.end(2):]
    else:
        print(f"  WARNING: category count pattern not found for {category_id}")

    # 2. Add article entry at the top of the category's <ul>
    new_entry = f"""                <li>
                    <a href="articles/{slug}.html">{title}</a>
                    <span class="article-date">{publish_date_str}</span>
                </li>"""

    # Find the <ul> after the category section
    section_pattern = re.compile(
        rf'(id="{category_id}".*?<ul class="category-articles">)',
        re.DOTALL,
    )
    section_match = section_pattern.search(content)
    if section_match:
        insert_pos = section_match.end()
        content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]
    else:
        print(f"  WARNING: category section not found for {category_id}")
        return False

    with open(CATEGORIES_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def update_archive(slug, title, publish_date_str):
    """Add article to archive.html and increment total count."""
    if not os.path.exists(ARCHIVE_PATH):
        print(f"  WARNING: archive.html not found")
        return False

    with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Increment total article count
    # Find the first stat-number (total articles)
    count_pattern = re.compile(r'(<div class="stat-item">\s*<div class="stat-number">)(\d+)(</div>)')
    count_match = count_pattern.search(content)
    if count_match:
        old_count = int(count_match.group(2))
        new_count = old_count + 1
        content = content[:count_match.start(2)] + str(new_count) + content[count_match.end(2):]
    else:
        print(f"  WARNING: archive count pattern not found")

    # 2. Parse the publish date for archive format (MM-DD)
    dt = datetime.strptime(publish_date_str, "%Y-%m-%d")
    month = dt.month
    day_str = f"{dt.month:02d}-{dt.day:02d}"
    month_cn = f"{month}月"
    year_str = str(dt.year)

    # 3. Check if the year/month section already exists
    month_pattern = re.compile(
        rf'(<h3 class="archive-month">{re.escape(month_cn)}</h3>\s*<ul class="archive-list">)',
        re.DOTALL,
    )
    month_match = month_pattern.search(content)

    new_entry = f"""            <li class="archive-item">
                <span class="archive-date">{day_str}</span>
                <span class="archive-title"><a href="articles/{slug}.html">{title}</a></span>
            </li>"""

    if month_match:
        # Insert at the top of existing month list
        insert_pos = month_match.end()
        content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]
    else:
        # Need to create new month section - find the year heading
        year_pattern = re.compile(
            rf'(<h2 class="archive-year">{re.escape(year_str)}</h2>)'
        )
        year_match = year_pattern.search(content)
        if year_match:
            new_section = f"""
        <h3 class="archive-month">{month_cn}</h3>
        <ul class="archive-list">
{new_entry}
        </ul>"""
            insert_pos = year_match.end()
            content = content[:insert_pos] + new_section + content[insert_pos:]
        else:
            print(f"  WARNING: year section not found for {year_str}")
            return False

    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def update_sitemap(slug, publish_date_str, has_en=False):
    """Add new article entry to sitemap.xml."""
    if not os.path.exists(SITEMAP_PATH):
        print(f"  WARNING: sitemap.xml not found")
        return

    ET.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    tree = ET.parse(SITEMAP_PATH)
    root = tree.getroot()
    ns = 'http://www.sitemaps.org/schemas/sitemap/0.9'

    # Add Chinese article
    cn_url = ET.SubElement(root, f'{{{ns}}}url')
    ET.SubElement(cn_url, f'{{{ns}}}loc').text = \
        f'https://worldsensetech.com/articles/{slug}.html'
    ET.SubElement(cn_url, f'{{{ns}}}lastmod').text = publish_date_str
    ET.SubElement(cn_url, f'{{{ns}}}changefreq').text = 'monthly'
    ET.SubElement(cn_url, f'{{{ns}}}priority').text = '0.8'

    # Add English article if exists
    if has_en:
        en_url = ET.SubElement(root, f'{{{ns}}}url')
        ET.SubElement(en_url, f'{{{ns}}}loc').text = \
            f'https://worldsensetech.com/en/articles/{slug}.html'
        ET.SubElement(en_url, f'{{{ns}}}lastmod').text = publish_date_str
        ET.SubElement(en_url, f'{{{ns}}}changefreq').text = 'monthly'
        ET.SubElement(en_url, f'{{{ns}}}priority').text = '0.7'

    tree.write(SITEMAP_PATH, encoding='UTF-8', xml_declaration=True)
    print(f"    -> Updated sitemap.xml")


def extract_block(content, tag):
    """Extract content between <!-- TAG ... --> markers (legacy support)."""
    pattern = re.compile(
        rf"<!--\s*{tag}\s*\n(.*?)\n\s*-->", re.DOTALL
    )
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return None


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

    # Try legacy INDEX_ENTRY blocks first
    cn_entry = extract_block(content, "INDEX_ENTRY")
    en_entry = extract_block(content, "EN_INDEX_ENTRY")

    # Extract metadata from HTML
    meta = extract_metadata(content, publish_date_str)
    print(f"    Title: {meta['title']}")
    print(f"    Tags: {', '.join(meta['tags'])}")

    # Clean draft content
    clean_content = clean_draft_content(content)

    # Write to articles/
    dest_path = os.path.join(ARTICLES_DIR, f"{slug}.html")
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
    print(f"    -> articles/{slug}.html")

    # Check for English draft
    en_draft_path = os.path.join(DRAFTS_DIR, "en", filename)
    has_en = False
    if os.path.exists(en_draft_path):
        with open(en_draft_path, "r", encoding="utf-8") as f:
            en_content = f.read()
        en_dest_path = os.path.join(EN_ARTICLES_DIR, f"{slug}.html")
        with open(en_dest_path, "w", encoding="utf-8") as f:
            f.write(en_content)
        os.remove(en_draft_path)
        has_en = True
        print(f"    -> en/articles/{slug}.html")

    # Update index.html
    if cn_entry:
        # Legacy mode: use embedded entry
        if insert_into_index(INDEX_CN, cn_entry):
            print(f"    -> Updated index.html (legacy mode)")
    else:
        # Auto mode: generate entry from metadata
        auto_entry = generate_index_entry(slug, meta)
        if insert_into_index(INDEX_CN, auto_entry):
            print(f"    -> Updated index.html (auto)")

    # Update en/index.html
    if en_entry:
        if insert_into_index(INDEX_EN, en_entry):
            print(f"    -> Updated en/index.html (legacy mode)")
    elif has_en:
        # For English, we'd need English metadata - skip for now
        print(f"    -> en/index.html: manual update needed")

    # Update categories.html
    category_id, category_name = determine_category(meta["tags"])
    if update_categories(slug, meta["title"], publish_date_str, category_id, category_name):
        print(f"    -> Updated categories.html ({category_name})")

    # Update archive.html
    if update_archive(slug, meta["title"], publish_date_str):
        print(f"    -> Updated archive.html")

    # Remove draft file
    os.remove(draft_path)
    print(f"    -> Removed _drafts/{filename}")

    # Update sitemap
    update_sitemap(slug, publish_date_str, has_en)

    return True


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
