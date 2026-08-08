#!/usr/bin/env python3
"""
publish_drafts.py - Check _drafts/ for articles past their publish date.
Move due drafts to articles/, and auto-update:
  - index.html (add article entry at top)
  - en/index.html (if English draft exists)
  - categories.html (add entry + increment category count)
  - archive.html (add entry + increment total count)
  - sitemap.xml (add URL)
  - article-nav (insert nav into new article + update prev/next neighbors)
Navigation is driven by <!-- NAV_PREV: slug --> and <!-- NAV_NEXT: slug -->
comments in each draft. Only links to already-published articles are created.
Exit 0 if published, 1 if nothing to publish.
"""

import os
import re
import sys
import shutil
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


def inject_canonical_tags(html_content, slug, has_en=False):
    """Inject canonical and hreflang tags into HTML content before </head>."""
    SITE_URL = "https://worldsensetech.com"

    # Skip if canonical already exists
    if '<link rel="canonical"' in html_content:
        return html_content

    cn_url = f"{SITE_URL}/articles/{slug}.html"
    tags = [f'    <link rel="canonical" href="{cn_url}">']

    if has_en:
        en_url = f"{SITE_URL}/en/articles/{slug}.html"
        tags.append(f'    <link rel="alternate" hreflang="zh-CN" href="{cn_url}">')
        tags.append(f'    <link rel="alternate" hreflang="en" href="{en_url}">')
        tags.append(f'    <link rel="alternate" hreflang="x-default" href="{cn_url}">')

    tags_str = '\n'.join(tags)
    html_content = html_content.replace('</head>', f'{tags_str}\n</head>')
    return html_content


def inject_en_canonical_tags(html_content, slug):
    """Inject canonical tag into English article HTML."""
    SITE_URL = "https://worldsensetech.com"

    if '<link rel="canonical"' in html_content:
        return html_content

    en_url = f"{SITE_URL}/en/articles/{slug}.html"
    cn_url = f"{SITE_URL}/articles/{slug}.html"
    tags = [
        f'    <link rel="canonical" href="{en_url}">',
        f'    <link rel="alternate" hreflang="zh-CN" href="{cn_url}">',
        f'    <link rel="alternate" hreflang="en" href="{en_url}">',
        f'    <link rel="alternate" hreflang="x-default" href="{cn_url}">',
    ]
    tags_str = '\n'.join(tags)
    html_content = html_content.replace('</head>', f'{tags_str}\n</head>')
    return html_content


def clean_draft_content(content):
    """Remove INDEX_ENTRY, EN_INDEX_ENTRY, and NAV metadata comment blocks from draft."""
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
    # Remove NAV metadata comments (used by publish script, not needed in final article)
    content = re.sub(r"\s*<!--\s*NAV_PREV:\s*.*?-->\s*\n?", "", content)
    content = re.sub(r"\s*<!--\s*NAV_NEXT:\s*.*?-->\s*\n?", "", content)
    return content


def migrate_images(content, slug):
    """Find all img src references in HTML, move local images from _drafts/ to articles/.
    Returns updated content with corrected image paths."""
    # Match img src attributes (both single and double quoted)
    img_pattern = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
    matches = img_pattern.findall(content)

    moved = []
    for src in matches:
        # Skip external URLs and data URIs
        if src.startswith(('http://', 'https://', 'data:')):
            continue

        # Resolve relative path from _drafts/ directory
        src_clean = src.split('?')[0]  # Remove query strings
        draft_img_path = os.path.join(DRAFTS_DIR, src_clean)

        if os.path.isfile(draft_img_path):
            # Determine destination in articles/
            dest_img_path = os.path.join(ARTICLES_DIR, src_clean)

            # Create subdirectories if needed
            dest_img_dir = os.path.dirname(dest_img_path)
            if dest_img_dir and not os.path.isdir(dest_img_dir):
                os.makedirs(dest_img_dir, exist_ok=True)

            shutil.move(draft_img_path, dest_img_path)
            moved.append(src_clean)
            print(f"    -> Moved image: _drafts/{src_clean} -> articles/{src_clean}")

    return content


# ── Navigation chain management ──────────────────────────────────────────

def extract_nav_metadata(content):
    """Extract NAV_PREV and NAV_NEXT slugs from draft HTML comments."""
    prev_match = re.search(r'<!--\s*NAV_PREV:\s*(.+?)\s*-->', content)
    next_match = re.search(r'<!--\s*NAV_NEXT:\s*(.+?)\s*-->', content)
    return {
        'prev': prev_match.group(1).strip() if prev_match else None,
        'next': next_match.group(1).strip() if next_match else None,
    }


def get_article_title(slug):
    """Get the h1 title from a published article by slug."""
    article_path = os.path.join(ARTICLES_DIR, f"{slug}.html")
    if not os.path.exists(article_path):
        return None
    with open(article_path, "r", encoding="utf-8") as f:
        content = f.read()
    h1_match = re.search(r"<h1>(.+?)</h1>", content)
    return h1_match.group(1).strip() if h1_match else None


def generate_nav_html(prev_slug, next_slug):
    """Generate the article-nav HTML block.
    Only includes links for articles that actually exist in articles/."""
    prev_title = get_article_title(prev_slug) if prev_slug else None
    next_title = get_article_title(next_slug) if next_slug else None

    if prev_title:
        left = (
            f'<div style="flex:1; text-align:left;">'
            f'<div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">&#8592; 上一篇</div>'
            f'<a href="{prev_slug}.html" style="color:#2563eb; text-decoration:none; font-weight:500;">{prev_title}</a>'
            f'</div>'
        )
    else:
        left = '<div style="flex:1;"></div>'

    if next_title:
        right = (
            f'<div style="flex:1; text-align:right;">'
            f'<div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">下一篇 &#8594;</div>'
            f'<a href="{next_slug}.html" style="color:#2563eb; text-decoration:none; font-weight:500;">{next_title}</a>'
            f'</div>'
        )
    else:
        right = '<div style="flex:1;"></div>'

    return (
        f'<nav class="article-nav" style="display:flex; justify-content:space-between; '
        f'margin:2rem auto; padding:1.5rem; background:#f8fafc; border-radius:8px; '
        f'border:1px solid #e2e8f0; max-width:800px;">\n'
        f'        {left}\n'
        f'        {right}\n'
        f'    </nav>'
    )


def insert_nav_into_article(content, nav_html):
    """Insert nav HTML into article before the comments section or article-actions."""
    # Try to insert before <!-- Comments -->
    if '<!-- Comments -->' in content:
        return content.replace('<!-- Comments -->', nav_html + '\n\n    <!-- Comments -->', 1)
    # Fallback: insert before article-actions
    if 'class="article-actions"' in content:
        return content.replace(
            '<div class="article-actions"',
            nav_html + '\n\n    <div class="article-actions"',
            1,
        )
    # Fallback: insert before </body>
    return content.replace('</body>', nav_html + '\n\n</body>', 1)


def update_article_nav(article_slug, direction, new_slug):
    """Update the prev/next link in an existing published article.
    direction: 'prev' to update the 上一篇 link, 'next' to update the 下一篇 link.
    new_slug: slug of the article to link to.
    """
    article_path = os.path.join(ARTICLES_DIR, f"{article_slug}.html")
    if not os.path.exists(article_path):
        print(f"    NAV WARN: {article_slug}.html not found, skipping nav update")
        return False

    with open(article_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_title = get_article_title(new_slug)
    if not new_title:
        print(f"    NAV WARN: cannot get title for {new_slug}, skipping nav update")
        return False

    nav_match = re.search(r'<nav class="article-nav".*?</nav>', content, re.DOTALL)
    if not nav_match:
        print(f"    NAV WARN: no nav found in {article_slug}, skipping")
        return False

    old_nav = nav_match.group()
    new_nav = old_nav

    if direction == 'next':
        # Update the 下一篇 link (right side)
        # Right div may have text-align:right (with next link) or be empty (no next link)
        # Use lookahead (?=</nav>) to avoid consuming the closing </nav> tag
        right_pattern = re.compile(
            r'(<div style="flex:1;(?: text-align:right;)?"[^>]*>)'
            r'.*?'
            r'(</div>)'
            r'\s*(?=</nav>)',
            re.DOTALL,
        )
        right_match = right_pattern.search(old_nav)
        if right_match:
            new_right = (
                f'<div style="flex:1; text-align:right;">'
                f'<div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">'
                f'下一篇 &#8594;</div>'
                f'<a href="{new_slug}.html" style="color:#2563eb; text-decoration:none; '
                f'font-weight:500;">{new_title}</a></div>'
            )
            new_nav = old_nav[:right_match.start()] + new_right + old_nav[right_match.end():]
        else:
            print(f"    NAV WARN: right div not found in {article_slug}")
            return False

    elif direction == 'prev':
        # Update the 上一篇 link (left side)
        # Strategy: find the right div, replace everything before it (the left part)
        right_div_pattern = re.compile(
            r'<div style="flex:1;(?: text-align:right;)?"[^>]*>'
        )
        # Find all flex:1 divs; the last one is the right div
        right_divs = list(right_div_pattern.finditer(old_nav))
        if not right_divs:
            print(f"    NAV WARN: right div not found in {article_slug}")
            return False
        right_div_start = right_divs[-1].start()

        # Build new nav: nav opening + new left div + existing right div onwards
        nav_opening_match = re.match(r'<nav class="article-nav"[^>]*>\s*', old_nav)
        if not nav_opening_match:
            print(f"    NAV WARN: nav opening not found in {article_slug}")
            return False

        new_left = (
            f'<div style="flex:1; text-align:left;">'
            f'<div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">'
            f'&#8592; 上一篇</div>'
            f'<a href="{new_slug}.html" style="color:#2563eb; text-decoration:none; '
            f'font-weight:500;">{new_title}</a></div>\n        '
        )
        new_nav = old_nav[:nav_opening_match.end()] + new_left + old_nav[right_div_start:]

    if new_nav != old_nav:
        content = content.replace(old_nav, new_nav, 1)
        with open(article_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"    -> Updated nav in {article_slug} ({direction} -> {new_slug})")
        return True

    return False


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

    # Extract navigation metadata (before cleaning removes NAV comments)
    nav_meta = extract_nav_metadata(content)
    print(f"    Nav: prev={nav_meta['prev']}, next={nav_meta['next']}")

    # Migrate image resources from _drafts/ to articles/
    content = migrate_images(content, slug)

    # Clean draft content
    clean_content = clean_draft_content(content)

    # Write to articles/
    dest_path = os.path.join(ARTICLES_DIR, f"{slug}.html")
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
    print(f"    -> articles/{slug}.html")

    # ── Navigation chain update ──
    # 1. Generate nav HTML for the new article (only link to articles that exist)
    nav_html = generate_nav_html(nav_meta['prev'], nav_meta['next'])
    clean_content = insert_nav_into_article(clean_content, nav_html)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
    print(f"    -> Inserted nav (prev={nav_meta['prev']}, next={nav_meta['next']})")

    # 2. Update prev article's "下一篇" to point to this new article
    if nav_meta['prev']:
        prev_path = os.path.join(ARTICLES_DIR, f"{nav_meta['prev']}.html")
        if os.path.exists(prev_path):
            update_article_nav(nav_meta['prev'], 'next', slug)

    # 3. Update next article's "上一篇" to point to this new article
    if nav_meta['next']:
        next_path = os.path.join(ARTICLES_DIR, f"{nav_meta['next']}.html")
        if os.path.exists(next_path):
            update_article_nav(nav_meta['next'], 'prev', slug)

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

    # ── Inject canonical and hreflang tags ──
    # Re-read and inject into Chinese article
    with open(dest_path, "r", encoding="utf-8") as f:
        clean_content = f.read()
    clean_content = inject_canonical_tags(clean_content, slug, has_en=has_en)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(clean_content)
    print(f"    -> Injected canonical tags (has_en={has_en})")

    # Inject into English article if exists
    if has_en:
        en_dest_path = os.path.join(EN_ARTICLES_DIR, f"{slug}.html")
        with open(en_dest_path, "r", encoding="utf-8") as f:
            en_content = f.read()
        en_content = inject_en_canonical_tags(en_content, slug)
        with open(en_dest_path, "w", encoding="utf-8") as f:
            f.write(en_content)
        print(f"    -> Injected canonical tags into English article")

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
