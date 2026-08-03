#!/usr/bin/env python3
"""
publish_drafts.py - Check _drafts/ for articles past their publish date.
Move due drafts to articles/, update index.html and en/index.html.
Exit 0 if published, 1 if nothing to publish.
"""

import os
import re
import shutil
import sys
from datetime import datetime, date

BLOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFTS_DIR = os.path.join(BLOG_DIR, "_drafts")
ARTICLES_DIR = os.path.join(BLOG_DIR, "articles")
INDEX_CN = os.path.join(BLOG_DIR, "index.html")
INDEX_EN = os.path.join(BLOG_DIR, "en", "index.html")

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
