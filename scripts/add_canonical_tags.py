#!/usr/bin/env python3
"""
为所有HTML页面添加canonical和hreflang标签
用于解决Google Search Console的重复网页问题
"""

import os
import re
from pathlib import Path

BLOG_ROOT = Path(__file__).parent.parent
SITE_URL = "https://worldsensetech.com"

# 中英文对照表：中文slug -> 英文slug
LANG_PAIRS = {
    "world-model-intro": "world-model-intro",
    "embodied-ai-guide": "embodied-ai-guide",
    "rssm-deep-dive": "rssm-deep-dive",
    "sim-to-real-transfer": "sim-to-real-transfer",
    "vla-vs-world-model": "vla-vs-world-model",
    "world-model-lab-setup": "world-model-lab-setup",
}


def add_canonical_to_file(filepath: Path, canonical_url: str, hreflang_pairs: dict = None):
    """为单个HTML文件添加canonical和hreflang标签"""
    content = filepath.read_text(encoding='utf-8')

    # 检查是否已有canonical标签
    if '<link rel="canonical"' in content:
        print(f"  [SKIP] {filepath.name} - 已有canonical标签")
        return False

    # 构建要插入的标签
    tags = [f'    <link rel="canonical" href="{canonical_url}">']

    # 添加hreflang标签（如果有中英文对照）
    if hreflang_pairs:
        zh_url = hreflang_pairs.get('zh')
        en_url = hreflang_pairs.get('en')
        if zh_url and en_url:
            tags.append(f'    <link rel="alternate" hreflang="zh-CN" href="{zh_url}">')
            tags.append(f'    <link rel="alternate" hreflang="en" href="{en_url}">')
            tags.append(f'    <link rel="alternate" hreflang="x-default" href="{zh_url}">')

    tags_str = '\n'.join(tags)

    # 在</head>前插入
    if '</head>' in content:
        content = content.replace('</head>', f'{tags_str}\n</head>')
        filepath.write_text(content, encoding='utf-8')
        print(f"  [OK] {filepath.name}")
        return True
    else:
        print(f"  [ERROR] {filepath.name} - 找不到</head>标签")
        return False


def process_articles():
    """处理articles目录下的所有文章"""
    articles_dir = BLOG_ROOT / "articles"
    en_articles_dir = BLOG_ROOT / "en" / "articles"

    print("=== 处理中文文章 ===")
    for html_file in sorted(articles_dir.glob("*.html")):
        slug = html_file.stem
        canonical_url = f"{SITE_URL}/articles/{slug}.html"

        # 检查是否有英文版本
        hreflang_pairs = None
        if slug in LANG_PAIRS:
            en_slug = LANG_PAIRS[slug]
            en_file = en_articles_dir / f"{en_slug}.html"
            if en_file.exists():
                hreflang_pairs = {
                    'zh': canonical_url,
                    'en': f"{SITE_URL}/en/articles/{en_slug}.html"
                }

        add_canonical_to_file(html_file, canonical_url, hreflang_pairs)

    print("\n=== 处理英文文章 ===")
    for html_file in sorted(en_articles_dir.glob("*.html")):
        slug = html_file.stem
        canonical_url = f"{SITE_URL}/en/articles/{slug}.html"

        # 找到对应的中文版本
        hreflang_pairs = None
        for zh_slug, en_slug in LANG_PAIRS.items():
            if en_slug == slug:
                hreflang_pairs = {
                    'zh': f"{SITE_URL}/articles/{zh_slug}.html",
                    'en': canonical_url
                }
                break

        add_canonical_to_file(html_file, canonical_url, hreflang_pairs)


def process_root_pages():
    """处理根目录下的页面"""
    print("\n=== 处理根目录页面 ===")
    root_pages = ['index.html', 'about.html', 'categories.html', 'archive.html']

    for page in root_pages:
        filepath = BLOG_ROOT / page
        if filepath.exists():
            canonical_url = f"{SITE_URL}/{page}" if page != 'index.html' else f"{SITE_URL}/"
            add_canonical_to_file(filepath, canonical_url)

    # 英文根页面
    en_pages = ['index.html', 'about.html']
    for page in en_pages:
        filepath = BLOG_ROOT / "en" / page
        if filepath.exists():
            canonical_url = f"{SITE_URL}/en/{page}" if page != 'index.html' else f"{SITE_URL}/en/"
            add_canonical_to_file(filepath, canonical_url)


def main():
    print(f"Blog root: {BLOG_ROOT}")
    print(f"Site URL: {SITE_URL}")
    print()

    process_articles()
    process_root_pages()

    print("\n=== 完成 ===")
    print("请运行 git add . && git commit -m 'feat: 添加canonical和hreflang标签' && git push")


if __name__ == "__main__":
    main()
