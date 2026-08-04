/**
 * Blog Interactions: Like (GitHub Reactions) + Bookmark (localStorage) + Giscus Comments
 *
 * Setup required (one-time):
 * 1. Install Giscus app on your repo: https://github.com/apps/giscus
 * 2. Configure at https://giscus.app — get your public key
 * 3. Update GISCUS_CONFIG below with your repo and key
 */

(function () {
    'use strict';

    // ============================================================
    // CONFIGURATION — Update these values for your blog
    // ============================================================
    var GISCUS_CONFIG = {
        repo: 'houxq8888/worldsensetech.github.io',  // GitHub repo
        repoId: '',           // Get from https://giscus.app after installation
        category: 'Announcements',
        categoryId: '',       // Get from https://giscus.app
        mapping: 'pathname',  // Match discussions by page pathname
        theme: 'light',
        lang: 'zh-CN',
        // GitHub API for reading reaction counts (no auth needed for public repos)
        githubApiBase: 'https://api.github.com'
    };

    // Current article path (used as unique key for likes/bookmarks)
    var articlePath = window.location.pathname;

    // ============================================================
    // BOOKMARK (localStorage)
    // ============================================================
    function initBookmark() {
        var btn = document.getElementById('btn-bookmark');
        if (!btn) return;

        var bookmarks = JSON.parse(localStorage.getItem('ws_bookmarks') || '[]');
        var isBookmarked = bookmarks.indexOf(articlePath) !== -1;

        if (isBookmarked) {
            btn.classList.add('active');
            btn.querySelector('.icon').textContent = '\u2605'; // ★
        }

        btn.addEventListener('click', function () {
            var bookmarks = JSON.parse(localStorage.getItem('ws_bookmarks') || '[]');
            var idx = bookmarks.indexOf(articlePath);
            if (idx === -1) {
                bookmarks.push(articlePath);
                btn.classList.add('active');
                btn.querySelector('.icon').textContent = '\u2605';
            } else {
                bookmarks.splice(idx, 1);
                btn.classList.remove('active');
                btn.querySelector('.icon').textContent = '\u2606';
            }
            localStorage.setItem('ws_bookmarks', JSON.stringify(bookmarks));
        });
    }

    // ============================================================
    // LIKE (GitHub Reactions via REST API)
    // ============================================================
    function initLike() {
        var btn = document.getElementById('btn-like');
        var countEl = document.getElementById('like-count');
        if (!btn) return;

        // Check if user already liked (stored locally)
        var liked = JSON.parse(localStorage.getItem('ws_likes') || '[]');
        var hasLiked = liked.indexOf(articlePath) !== -1;
        if (hasLiked) {
            btn.classList.add('active');
        }

        // Try to load like count from GitHub API
        loadLikeCount(countEl);

        btn.addEventListener('click', function () {
            var liked = JSON.parse(localStorage.getItem('ws_likes') || '[]');
            var idx = liked.indexOf(articlePath);
            if (idx === -1) {
                liked.push(articlePath);
                btn.classList.add('active');
                // Update local count display
                if (countEl) {
                    countEl.textContent = parseInt(countEl.textContent || '0') + 1;
                }
                // Try to add GitHub reaction (requires auth, will silently fail if not authenticated)
                addGitHubReaction();
            } else {
                liked.splice(idx, 1);
                btn.classList.remove('active');
                if (countEl) {
                    var c = parseInt(countEl.textContent || '1') - 1;
                    countEl.textContent = Math.max(0, c);
                }
            }
            localStorage.setItem('ws_likes', JSON.stringify(liked));
        });
    }

    function loadLikeCount(el) {
        if (!el) return;
        // Try to load from GitHub Discussions API (public read, no auth needed)
        // This maps article pathname to a Discussion and reads heart reaction count
        // Since we don't have the discussion ID yet, we use a fallback:
        // Store counts in a simple JSON file or use localStorage as fallback
        var stored = JSON.parse(localStorage.getItem('ws_like_counts') || '{}');
        if (stored[articlePath]) {
            el.textContent = stored[articlePath];
        }

        // Try GitHub API (works for public repos, read-only without auth)
        // In production, you'd map articlePath → Discussion ID and query:
        // GET https://api.github.com/repos/{owner}/{repo}/discussions/{number}/reactions
        // For now, show localStorage count
    }

    function addGitHubReaction() {
        // Adding reactions requires authentication.
        // Options to enable this:
        // 1. Set up a GitHub OAuth backend (e.g., Cloudflare Worker)
        // 2. Use a Personal Access Token (for personal blogs)
        // 3. Let users react through Giscus comments (recommended - no extra setup)
        //
        // For now, likes are tracked locally.
        // To enable GitHub Reactions, uncomment and configure:
        //
        // fetch(GISCUS_CONFIG.githubApiBase + '/repos/' + GISCUS_CONFIG.repo + '/discussions/{NUMBER}/reactions', {
        //     method: 'POST',
        //     headers: {
        //         'Accept': 'application/vnd.github.squirrel-girl-preview+json',
        //         'Authorization': 'token YOUR_TOKEN',
        //         'Content-Type': 'application/json'
        //     },
        //     body: JSON.stringify({ content: 'heart' })
        // });
    }

    // ============================================================
    // GISCUS COMMENTS
    // ============================================================
    function initGiscus() {
        var container = document.getElementById('giscus-container');
        if (!container) return;

        // Build Giscus script
        var script = document.createElement('script');
        script.src = 'https://giscus.app/client.js';
        script.setAttribute('data-repo', GISCUS_CONFIG.repo);
        script.setAttribute('data-repo-id', GISCUS_CONFIG.repoId);
        script.setAttribute('data-category', GISCUS_CONFIG.category);
        script.setAttribute('data-category-id', GISCUS_CONFIG.categoryId);
        script.setAttribute('data-mapping', GISCUS_CONFIG.mapping);
        script.setAttribute('data-strict', '0');
        script.setAttribute('data-reactions-enabled', '1');
        script.setAttribute('data-emit-metadata', '0');
        script.setAttribute('data-input-position', 'bottom');
        script.setAttribute('data-theme', GISCUS_CONFIG.theme);
        script.setAttribute('data-lang', GISCUS_CONFIG.lang);
        script.crossOrigin = 'anonymous';
        script.async = true;

        // If repoId/categoryId not configured, show setup message
        if (!GISCUS_CONFIG.repoId || !GISCUS_CONFIG.categoryId) {
            container.innerHTML = '<div style="text-align:center; padding:2rem; color:#64748b; font-size:0.9rem; border:1px dashed #e2e8f0; border-radius:8px;">' +
                '<p style="margin-bottom:0.5rem;">Comment system not yet configured.</p>' +
                '<p>Visit <a href="https://giscus.app" target="_blank" style="color:#2563eb;">giscus.app</a> to get your repo-id and category-id, then update the config in blog-interactions.js.</p>' +
                '</div>';
            return;
        }

        container.appendChild(script);
    }

    // ============================================================
    // INIT
    // ============================================================
    function init() {
        initLike();
        initBookmark();
        initGiscus();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
