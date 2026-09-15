/**
 * GA4 Event Tracking for WorldSense Blog
 *
 * Events:
 *   article_like      - article like button (engagement)
 *   article_bookmark   - article bookmark button (conversion)
 *   code_copy          - code block copy button (conversion)
 *   github_click       - GitHub nav link (conversion)
 *   related_click      - "Continue Reading" internal link (engagement)
 *   lang_switch        - language toggle zh/en (engagement)
 *
 * After deploy, mark article_bookmark / code_copy / github_click as
 * conversions in GA4 Admin > Events > toggle "Mark as conversion".
 */

(function () {
    'use strict';

    function send(name, params) {
        if (typeof gtag === 'function') gtag('event', name, params);
    }

    function articleMeta() {
        var h1 = document.querySelector('.article-header h1');
        return {
            article_title: h1 ? h1.textContent.trim() : '',
            article_path: window.location.pathname
        };
    }

    function init() {
        // ---- 1. Like button ----
        var likeBtn = document.getElementById('btn-like');
        if (likeBtn) {
            likeBtn.addEventListener('click', function () {
                // Only fire on like (not unlike): active class is toggled after this handler
                if (!likeBtn.classList.contains('active')) {
                    send('article_like', articleMeta());
                }
            });
        }

        // ---- 2. Bookmark button ----
        var bookmarkBtn = document.getElementById('btn-bookmark');
        if (bookmarkBtn) {
            bookmarkBtn.addEventListener('click', function () {
                if (!bookmarkBtn.classList.contains('active')) {
                    send('article_bookmark', articleMeta());
                }
            });
        }

        // ---- 3. Code copy buttons ----
        // blog-interactions.js creates .code-copy-btn inside <pre> elements.
        // Use MutationObserver to attach tracking when copy buttons appear.
        function attachCopyTracking() {
            var copyBtns = document.querySelectorAll('.code-copy-btn');
            copyBtns.forEach(function (btn) {
                if (btn.dataset.gaTracked) return;
                btn.dataset.gaTracked = '1';
                btn.addEventListener('click', function () {
                    var pre = btn.closest('pre');
                    var code = pre ? pre.querySelector('code') : null;
                    send('code_copy', Object.assign(articleMeta(), {
                        code_language: code ? (code.className.replace(/^language-/, '') || 'unknown') : 'unknown'
                    }));
                });
            });
        }
        // Initial pass
        attachCopyTracking();
        // Observe DOM for dynamically-added copy buttons
        var observer = new MutationObserver(function () { attachCopyTracking(); });
        observer.observe(document.body, { childList: true, subtree: true });
        // Stop observing after 10s to avoid perf overhead
        setTimeout(function () { observer.disconnect(); }, 10000);

        // ---- 4. GitHub link (header nav) ----
        var ghLinks = document.querySelectorAll('a[href*="github.com/houxq8888"]');
        ghLinks.forEach(function (link) {
            link.addEventListener('click', function () {
                send('github_click', {
                    source: link.closest('header') ? 'nav' : 'other',
                    article_path: window.location.pathname
                });
            });
        });

        // ---- 5. Related reading links ----
        var relatedLinks = document.querySelectorAll('.related-reading a');
        relatedLinks.forEach(function (link) {
            link.addEventListener('click', function () {
                send('related_click', Object.assign(articleMeta(), {
                    target_title: link.textContent.trim(),
                    target_href: link.getAttribute('href')
                }));
            });
        });

        // ---- 6. Language switch ----
        var langLinks = document.querySelectorAll('header a[href$="/en/"], header a[href$="/zh/"]');
        langLinks.forEach(function (link) {
            link.addEventListener('click', function () {
                var href = link.getAttribute('href') || '';
                var targetLang = href.indexOf('/en/') !== -1 ? 'en' : 'zh';
                send('lang_switch', {
                    target_language: targetLang,
                    current_path: window.location.pathname
                });
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
