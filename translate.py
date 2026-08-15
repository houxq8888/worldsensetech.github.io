#!/usr/bin/env python3
"""
WorldSense Blog Translation Script
Translates Chinese Markdown articles to English using Claude API
"""

import os
import sys
import re
import yaml
from pathlib import Path

# 术语表 - 保证翻译一致性
GLOSSARY = {
    "世界模型": "World Model",
    "具身智能": "Embodied AI",
    "强化学习": "Reinforcement Learning",
    "状态空间模型": "State Space Model",
    "仿真": "Simulation",
    "真实机器人": "Real Robot",
    "迁移": "Transfer",
    "策略": "Policy",
    "奖励": "Reward",
    "观测": "Observation",
    "动作": "Action",
    "环境": "Environment",
    "训练": "Training",
    "推理": "Inference",
    "部署": "Deployment",
    "感知": "Perception",
    "决策": "Decision Making",
    "控制": "Control",
    "动力学": "Dynamics",
    "传感器": "Sensor",
    "执行器": "Actuator",
    "标定": "Calibration",
    "延迟": "Latency",
    "噪声": "Noise",
    "鲁棒性": "Robustness",
    "泛化": "Generalization",
    "域随机化": "Domain Randomization",
    "系统辨识": "System Identification",
}

# 专有名词不翻译
PROPER_NOUNS = [
    "MuJoCo", "Isaac Sim", "DreamerV3", "TD-MPC2", "RSSM", 
    "PPO", "SAC", "ManiSkill", "SAPIEN", "Genesis",
    "PyTorch", "TensorFlow", "CUDA", "GPU", "CPU",
    "ROS", "ROS2", "JSON", "API", "SDK",
    "WorldSense", "GitHub", "Hugo"
]


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return frontmatter, body
    return {}, content


def translate_text(text: str) -> str:
    """
    Translate Chinese text to English.
    This is a placeholder - replace with actual API call.
    """
    # TODO: Implement actual translation using Claude API
    # For now, return placeholder
    return f"[TRANSLATED] {text}"


def translate_frontmatter(fm: dict) -> dict:
    """Translate frontmatter fields"""
    translated = fm.copy()
    
    # Translate title
    if "title" in translated:
        translated["title"] = translate_text(translated["title"])
    
    # Translate description
    if "description" in translated:
        translated["description"] = translate_text(translated["description"])
    
    # Translate categories
    category_map = {
        "世界模型": "World Model",
        "具身智能": "Embodied AI",
        "强化学习": "Reinforcement Learning",
        "Sim-to-Real": "Sim-to-Real",
        "仿真": "Simulation",
        "教程": "Tutorial",
    }
    if "categories" in translated:
        translated["categories"] = [
            category_map.get(c, c) for c in translated["categories"]
        ]
    
    return translated


def translate_article(input_path: Path, output_path: Path):
    """Translate a single article"""
    print(f"Translating: {input_path.name}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract frontmatter and body
    frontmatter, body = extract_frontmatter(content)
    
    # Translate frontmatter
    translated_fm = translate_frontmatter(frontmatter)
    
    # Translate body (paragraph by paragraph to preserve structure)
    paragraphs = body.split("\n\n")
    translated_paragraphs = []
    
    for para in paragraphs:
        # Skip code blocks
        if para.startswith("```"):
            translated_paragraphs.append(para)
            continue
        
        # Skip headers (keep structure, translate text)
        if para.startswith("#"):
            # Extract header level and text
            match = re.match(r"(#+)\s*(.*)", para)
            if match:
                level, text = match.groups()
                translated_text = translate_text(text)
                translated_paragraphs.append(f"{level} {translated_text}")
            else:
                translated_paragraphs.append(para)
            continue
        
        # Translate regular paragraphs
        translated = translate_text(para)
        translated_paragraphs.append(translated)
    
    translated_body = "\n\n".join(translated_paragraphs)
    
    # Reconstruct markdown
    output = "---\n"
    output += yaml.dump(translated_fm, allow_unicode=True, default_flow_style=False)
    output += "---\n\n"
    output += translated_body
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    
    print(f"  -> {output_path}")


def main():
    """Main entry point"""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    # Directories
    zh_dir = project_root / "content" / "zh" / "articles"
    en_dir = project_root / "content" / "en" / "articles"
    
    if not zh_dir.exists():
        print(f"Error: Chinese articles directory not found: {zh_dir}")
        sys.exit(1)
    
    # Get list of Chinese articles
    zh_articles = list(zh_dir.glob("*.md"))
    
    if not zh_articles:
        print("No Chinese articles found to translate.")
        sys.exit(0)
    
    print(f"Found {len(zh_articles)} Chinese articles.")
    
    # Translate each article
    for zh_article in zh_articles:
        en_article = en_dir / zh_article.name
        
        # Skip if English version already exists and is newer
        if en_article.exists():
            if en_article.stat().st_mtime >= zh_article.stat().st_mtime:
                print(f"Skipping (already up to date): {zh_article.name}")
                continue
        
        translate_article(zh_article, en_article)
    
    print("\nTranslation complete!")


if __name__ == "__main__":
    main()
