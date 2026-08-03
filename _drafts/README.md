# Drafts

定时发布的文章放在这里。

## 文件命名规则

```
YYYY-MM-DD-article-slug.html
```

日期就是发布日期，到了那天 GitHub Actions 会自动发布。

## 文件格式

在 HTML 文件末尾（`</body>` 之前）加上这段注释，告诉脚本怎么更新首页：

```html
<!-- INDEX_ENTRY
<article class="article-item">
    <div class="article-meta">2026年8月5日 · 阅读约12分钟</div>
    <h2><a href="articles/your-article.html">文章标题</a></h2>
    <div>
        <span class="tag">标签1</span>
        <span class="tag">标签2</span>
    </div>
    <p class="article-excerpt">文章摘要...</p>
</article>
-->

<!-- EN_INDEX_ENTRY
<article class="article-item">
    <div class="article-meta">August 5, 2026 · ~12 min read</div>
    <h2><a href="articles/your-article.html">English Title</a></h2>
    <div>
        <span class="tag">Tag1</span>
        <span class="tag">Tag2</span>
    </div>
    <p class="article-excerpt">English excerpt...</p>
</article>
-->
```

`EN_INDEX_ENTRY` 是可选的，没有的话英文版不会自动添加这篇文章。

## 发布流程

1. 把写好的 HTML 文件放到 `_drafts/`，文件名带日期
2. `git push` 到 GitHub
3. GitHub Actions 每天 UTC 00:00（北京时间 08:00）检查
4. 到期自动发布：移到 `articles/`，更新中英文首页，提交推送
