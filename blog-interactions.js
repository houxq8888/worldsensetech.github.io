/**
 * Blog Interactions: Like + Bookmark (Supabase global stats) + Giscus Comments
 *
 * Setup:
 * 1. Giscus: https://giscus.app
 * 2. Supabase: https://supabase.com (create project, create blog_stats table)
 */

(function () {
    'use strict';

    // ============================================================
    // CONFIGURATION
    // ============================================================
    var GISCUS_CONFIG = {
        repo: 'houxq8888/worldsensetech.github.io',
        repoId: 'R_kgDOTqqp9g',
        category: 'Announcements',
        categoryId: 'DIC_kwDOTqqp9s4DCnZo',
        mapping: 'pathname',
        strict: '0',
        reactionsEnabled: '1',
        emitMetadata: '0',
        inputPosition: 'bottom',
        theme: 'preferred_color_scheme',
        lang: 'zh-CN'
    };

    var SUPABASE_CONFIG = {
        projectUrl: 'https://vkbbaulinnpgywowgztp.supabase.co',
        apiKey: 'sb_publishable_-LEtCNCKzgK7zd0oTAJ3ow_3dH7Di7H',
        tableName: 'blog_stats'
    };

    var articlePath = window.location.pathname;

    // Get or create visitor ID (stored in localStorage)
    function getVisitorId() {
        var vid = localStorage.getItem('ws_visitor_id');
        if (!vid) {
            vid = 'v_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('ws_visitor_id', vid);
        }
        return vid;
    }

    // ============================================================
    // SUPABASE API HELPERS
    // ============================================================
    function sbHeaders() {
        return {
            'apikey': SUPABASE_CONFIG.apiKey,
            'Authorization': 'Bearer ' + SUPABASE_CONFIG.apiKey,
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        };
    }

    var SB_URL = SUPABASE_CONFIG.projectUrl + '/rest/v1/' + SUPABASE_CONFIG.tableName;

    // Query stats for current article
    function sbQueryStats() {
        return fetch(SB_URL + '?path=eq.' + encodeURIComponent(articlePath), {
            headers: sbHeaders()
        }).then(function (r) { return r.json(); });
    }

    // Create new stats record
    function sbCreateStats(data) {
        return fetch(SB_URL, {
            method: 'POST',
            headers: sbHeaders(),
            body: JSON.stringify(data)
        }).then(function (r) { return r.json(); });
    }

    // Update stats record
    function sbUpdateStats(path, data) {
        return fetch(SB_URL + '?path=eq.' + encodeURIComponent(path), {
            method: 'PATCH',
            headers: sbHeaders(),
            body: JSON.stringify(data)
        }).then(function (r) { return r.json(); });
    }

    // ============================================================
    // LIKE (global via Supabase)
    // ============================================================
    function initLike() {
        var btn = document.getElementById('btn-like');
        var countEl = document.getElementById('like-count');
        if (!btn || !countEl) return;

        // Load current stats
        sbQueryStats().then(function (results) {
            if (results && results.length > 0) {
                var stats = results[0];
                countEl.textContent = stats.likes || 0;
                // Check if current visitor already liked
                var liked = JSON.parse(localStorage.getItem('ws_likes') || '[]');
                if (liked.indexOf(articlePath) !== -1) {
                    btn.classList.add('active');
                }
            } else {
                countEl.textContent = '0';
            }
        }).catch(function () {
            countEl.textContent = '-';
        });

        btn.addEventListener('click', function () {
            var isActive = btn.classList.contains('active');
            var liked = JSON.parse(localStorage.getItem('ws_likes') || '[]');
            var alreadyLiked = liked.indexOf(articlePath) !== -1;

            if (isActive || alreadyLiked) {
                // Unlike: decrement count
                sbQueryStats().then(function (results) {
                    if (results && results.length > 0) {
                        var newCount = Math.max(0, (results[0].likes || 0) - 1);
                        return sbUpdateStats(articlePath, { likes: newCount });
                    } else {
                        return sbCreateStats({ path: articlePath, likes: 0, bookmarks: 0 });
                    }
                }).then(function () {
                    btn.classList.remove('active');
                    // Update localStorage
                    var idx = liked.indexOf(articlePath);
                    if (idx !== -1) liked.splice(idx, 1);
                    localStorage.setItem('ws_likes', JSON.stringify(liked));
                    return sbQueryStats();
                }).then(function (results) {
                    if (results && results.length > 0) {
                        countEl.textContent = results[0].likes || 0;
                    }
                });
            } else {
                // Like: increment count
                sbQueryStats().then(function (results) {
                    if (results && results.length > 0) {
                        var newCount = (results[0].likes || 0) + 1;
                        return sbUpdateStats(articlePath, { likes: newCount });
                    } else {
                        return sbCreateStats({ path: articlePath, likes: 1, bookmarks: 0 });
                    }
                }).then(function () {
                    btn.classList.add('active');
                    // Update localStorage
                    if (liked.indexOf(articlePath) === -1) liked.push(articlePath);
                    localStorage.setItem('ws_likes', JSON.stringify(liked));
                    return sbQueryStats();
                }).then(function (results) {
                    if (results && results.length > 0) {
                        countEl.textContent = results[0].likes || 0;
                    }
                });
            }
        });
    }

    // ============================================================
    // BOOKMARK (global via Supabase)
    // ============================================================
    function initBookmark() {
        var btn = document.getElementById('btn-bookmark');
        if (!btn) return;

        // Load current stats to check bookmark state
        sbQueryStats().then(function (results) {
            if (results && results.length > 0) {
                // Check if current visitor already bookmarked
                var bookmarked = JSON.parse(localStorage.getItem('ws_bookmarks') || '[]');
                if (bookmarked.indexOf(articlePath) !== -1) {
                    btn.classList.add('active');
                    btn.querySelector('.icon').textContent = '\u2605';
                }
            }
        });

        btn.addEventListener('click', function () {
            var isActive = btn.classList.contains('active');
            var bookmarked = JSON.parse(localStorage.getItem('ws_bookmarks') || '[]');
            var alreadyBookmarked = bookmarked.indexOf(articlePath) !== -1;

            if (isActive || alreadyBookmarked) {
                // Remove bookmark: decrement count
                sbQueryStats().then(function (results) {
                    if (results && results.length > 0) {
                        var newCount = Math.max(0, (results[0].bookmarks || 0) - 1);
                        return sbUpdateStats(articlePath, { bookmarks: newCount });
                    } else {
                        return sbCreateStats({ path: articlePath, likes: 0, bookmarks: 0 });
                    }
                }).then(function () {
                    btn.classList.remove('active');
                    btn.querySelector('.icon').textContent = '\u2606';
                    // Update localStorage
                    var idx = bookmarked.indexOf(articlePath);
                    if (idx !== -1) bookmarked.splice(idx, 1);
                    localStorage.setItem('ws_bookmarks', JSON.stringify(bookmarked));
                });
            } else {
                // Add bookmark: increment count
                sbQueryStats().then(function (results) {
                    if (results && results.length > 0) {
                        var newCount = (results[0].bookmarks || 0) + 1;
                        return sbUpdateStats(articlePath, { bookmarks: newCount });
                    } else {
                        return sbCreateStats({ path: articlePath, likes: 0, bookmarks: 1 });
                    }
                }).then(function () {
                    btn.classList.add('active');
                    btn.querySelector('.icon').textContent = '\u2605';
                    // Update localStorage
                    if (bookmarked.indexOf(articlePath) === -1) bookmarked.push(articlePath);
                    localStorage.setItem('ws_bookmarks', JSON.stringify(bookmarked));
                });
            }
        });
    }

    // ============================================================
    // GISCUS COMMENTS
    // ============================================================
    function initGiscus() {
        var container = document.getElementById('giscus-container');
        if (!container) return;

        var script = document.createElement('script');
        script.src = 'https://giscus.app/client.js';
        script.setAttribute('data-repo', GISCUS_CONFIG.repo);
        script.setAttribute('data-repo-id', GISCUS_CONFIG.repoId);
        script.setAttribute('data-category', GISCUS_CONFIG.category);
        script.setAttribute('data-category-id', GISCUS_CONFIG.categoryId);
        script.setAttribute('data-mapping', GISCUS_CONFIG.mapping);
        script.setAttribute('data-strict', GISCUS_CONFIG.strict);
        script.setAttribute('data-reactions-enabled', GISCUS_CONFIG.reactionsEnabled);
        script.setAttribute('data-emit-metadata', GISCUS_CONFIG.emitMetadata);
        script.setAttribute('data-input-position', GISCUS_CONFIG.inputPosition);
        script.setAttribute('data-theme', GISCUS_CONFIG.theme);
        script.setAttribute('data-lang', GISCUS_CONFIG.lang);
        script.crossOrigin = 'anonymous';
        script.async = true;

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
