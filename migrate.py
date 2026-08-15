#!/usr/bin/env python3
"""
Migrate HTML blog articles to Hugo Markdown format
"""

import os
import re
import html
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser


class HTMLToMarkdown(HTMLParser):
    """Convert HTML to Markdown"""
    
    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None
        self.tag_stack = []
        self.in_pre = False
        self.in_code = False
        self.list_type = None
        self.list_counter = 0
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tag_stack.append(tag)
        
        if tag == 'h1':
            self.result.append('\n# ')
        elif tag == 'h2':
            self.result.append('\n## ')
        elif tag == 'h3':
            self.result.append('\n### ')
        elif tag == 'h4':
            self.result.append('\n#### ')
        elif tag == 'p':
            self.result.append('\n\n')
        elif tag == 'br':
            self.result.append('\n')
        elif tag == 'strong' or tag == 'b':
            self.result.append('**')
        elif tag == 'em' or tag == 'i':
            self.result.append('*')
        elif tag == 'code':
            self.result.append('`')
            self.in_code = True
        elif tag == 'pre':
            self.result.append('\n```\n')
            self.in_pre = True
        elif tag == 'a':
            href = attrs_dict.get('href', '')
            self.result.append('[')
            self.current_tag = ('a', href)
        elif tag == 'img':
            src = attrs_dict.get('src', '')
            alt = attrs_dict.get('alt', '')
            self.result.append(f'![{alt}]({src})')
        elif tag == 'ul':
            self.list_type = 'ul'
            self.result.append('\n')
        elif tag == 'ol':
            self.list_type = 'ol'
            self.list_counter = 0
            self.result.append('\n')
        elif tag == 'li':
            if self.list_type == 'ol':
                self.list_counter += 1
                self.result.append(f'\n{self.list_counter}. ')
            else:
                self.result.append('\n- ')
        elif tag == 'blockquote':
            self.result.append('\n> ')
        elif tag == 'hr':
            self.result.append('\n---\n')
            
    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
            
        if tag == 'h1' or tag == 'h2' or tag == 'h3' or tag == 'h4':
            self.result.append('\n')
        elif tag == 'p':
            self.result.append('\n')
        elif tag == 'strong' or tag == 'b':
            self.result.append('**')
        elif tag == 'em' or tag == 'i':
            self.result.append('*')
        elif tag == 'code':
            self.result.append('`')
            self.in_code = False
        elif tag == 'pre':
            self.result.append('\n```\n')
            self.in_pre = False
        elif tag == 'a':
            if self.current_tag and self.current_tag[0] == 'a':
                href = self.current_tag[1]
                self.result.append(f']({href})')
                self.current_tag = None
        elif tag == 'ul' or tag == 'ol':
            self.list_type = None
            self.result.append('\n')
        elif tag == 'blockquote':
            self.result.append('\n')
            
    def handle_data(self, data):
        if self.in_pre:
            self.result.append(data)
        else:
            # Normalize whitespace but preserve meaningful spaces
            text = re.sub(r'\s+', ' ', data)
            self.result.append(text)
            
    def handle_entityref(self, name):
        self.result.append(html.unescape(f'&{name};'))
        
    def handle_charref(self, name):
        self.result.append(html.unescape(f'&#{name};'))
        
    def get_markdown(self):
        text = ''.join(self.result)
        # Clean up multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def extract_title(html_content):
    """Extract title from HTML"""
    match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
    if match:
        title = match.group(1)
        # Remove site name suffix if present
        title = re.sub(r'\s*-\s*WorldSense.*$', '', title, flags=re.IGNORECASE)
        return title.strip()
    return "Untitled"


def extract_date_from_filename(filename):
    """Extract date from filename like 2026-08-15-slug.html"""
    match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def extract_date_from_file(filepath):
    """Extract date from file modification time"""
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')


def guess_categories(filename, title):
    """Guess categories from filename and title"""
    categories = []
    
    filename_lower = filename.lower()
    title_lower = title.lower()
    
    if 'world-model' in filename_lower or '世界模型' in title:
        categories.append('世界模型')
    if 'embodied' in filename_lower or '具身智能' in title:
        categories.append('具身智能')
    if 'sim-to-real' in filename_lower or 'sim-to-real' in title_lower or '迁移' in title:
        categories.append('Sim-to-Real')
    if 'reinforcement' in filename_lower or '强化学习' in title:
        categories.append('强化学习')
    if 'mujoco' in filename_lower or 'isaac' in filename_lower:
        categories.append('仿真')
    if 'dreamer' in filename_lower or 'rssm' in filename_lower:
        categories.append('世界模型')
    if 'vla' in filename_lower:
        categories.append('具身智能')
    if 'training' in filename_lower or '教程' in title:
        categories.append('教程')
    if 'data' in filename_lower or '数据' in title:
        categories.append('数据')
    
    if not categories:
        categories.append('技术')
    
    return list(set(categories))


def guess_tags(filename, title):
    """Guess tags from filename and title"""
    tags = []
    
    filename_lower = filename.lower()
    
    # Map filename patterns to tags
    tag_map = {
        'world-model': 'World Model',
        'embodied': 'Embodied AI',
        'sim-to-real': 'Sim-to-Real',
        'reinforcement': 'Reinforcement Learning',
        'mujoco': 'MuJoCo',
        'isaac': 'Isaac Sim',
        'dreamer': 'DreamerV3',
        'rssm': 'RSSM',
        'vla': 'VLA',
        'td-mpc': 'TD-MPC2',
        'domain-randomization': 'Domain Randomization',
        'data': 'Data',
        'training': 'Training',
    }
    
    for pattern, tag in tag_map.items():
        if pattern in filename_lower:
            tags.append(tag)
    
    if not tags:
        tags.append('AI')
    
    return list(set(tags))


def convert_html_to_markdown(html_content):
    """Convert HTML content to Markdown"""
    # Remove HTML head section, keep only body content
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        body_content = body_match.group(1)
    else:
        body_content = html_content
    
    # Remove navigation, header, footer elements
    body_content = re.sub(r'<nav[^>]*>.*?</nav>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<header[^>]*>.*?</header>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<footer[^>]*>.*?</footer>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<script[^>]*>.*?</script>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<style[^>]*>.*?</style>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert to Markdown
    parser = HTMLToMarkdown()
    parser.feed(body_content)
    markdown = parser.get_markdown()
    
    return markdown


def create_frontmatter(title, date, categories, tags, description=""):
    """Create YAML frontmatter"""
    fm = f"""---
title: "{title}"
date: {date}
draft: false
categories: {categories}
tags: {tags}
description: "{description}"
toc: true
---
"""
    return fm


def migrate_article(html_path, output_dir, is_draft=False):
    """Migrate a single HTML article to Markdown"""
    filename = html_path.name
    slug = filename.replace('.html', '')
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract metadata
    title = extract_title(html_content)
    
    # Get date
    date = extract_date_from_filename(slug)
    if not date:
        date = extract_date_from_file(html_path)
    
    # Guess categories and tags
    categories = guess_categories(slug, title)
    tags = guess_tags(slug, title)
    
    # Convert content
    markdown_content = convert_html_to_markdown(html_content)
    
    # Remove the title from content if it appears at the start
    # (since it's in the frontmatter)
    lines = markdown_content.split('\n')
    if lines and (lines[0].startswith('# ') or title in lines[0]):
        markdown_content = '\n'.join(lines[1:]).strip()
    
    # Create frontmatter
    frontmatter = create_frontmatter(
        title=title,
        date=date,
        categories=str(categories).replace("'", '"'),
        tags=str(tags).replace("'", '"'),
        description=f"{title} - WorldSense 技术笔记"
    )
    
    # Combine
    full_content = frontmatter + '\n\n' + markdown_content
    
    # Write output
    output_path = output_dir / f"{slug}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    return output_path


def main():
    """Main migration function"""
    # Paths
    blog_root = Path("D:/virtualMachine/github/GEOPro-git/GEOPro/idea/worldsense-blog")
    hugo_root = Path("D:/virtualMachine/github/GEOPro-git/GEOPro/idea/worldsense-blog-hugo")
    
    # Directories
    zh_articles_dir = hugo_root / "content" / "zh" / "articles"
    en_articles_dir = hugo_root / "content" / "en" / "articles"
    
    # Ensure directories exist
    zh_articles_dir.mkdir(parents=True, exist_ok=True)
    en_articles_dir.mkdir(parents=True, exist_ok=True)
    
    # Migrate Chinese articles
    print("Migrating Chinese articles...")
    zh_source = blog_root / "articles"
    if zh_source.exists():
        for html_file in zh_source.glob("*.html"):
            print(f"  Converting: {html_file.name}")
            output = migrate_article(html_file, zh_articles_dir)
            print(f"    -> {output.name}")
    
    # Migrate drafts
    print("\nMigrating draft articles...")
    drafts_source = blog_root / "_drafts"
    if drafts_source.exists():
        for html_file in drafts_source.glob("*.html"):
            print(f"  Converting: {html_file.name}")
            output = migrate_article(html_file, zh_articles_dir, is_draft=True)
            print(f"    -> {output.name}")
    
    # Migrate English articles
    print("\nMigrating English articles...")
    en_source = blog_root / "en" / "articles"
    if en_source.exists():
        for html_file in en_source.glob("*.html"):
            print(f"  Converting: {html_file.name}")
            output = migrate_article(html_file, en_articles_dir)
            print(f"    -> {output.name}")
    
    print("\nMigration complete!")
    print(f"Chinese articles: {zh_articles_dir}")
    print(f"English articles: {en_articles_dir}")


if __name__ == "__main__":
    main()
