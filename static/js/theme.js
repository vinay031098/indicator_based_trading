// theme.js — light/dark theme toggle persisted in localStorage,
// respecting prefers-color-scheme as the default.

const STORAGE_KEY = 'dash_theme';

function systemPrefersLight() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
}

export function getStoredTheme() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (_e) { return null; }
}

export function resolveInitialTheme() {
    const stored = getStoredTheme();
    if (stored === 'light' || stored === 'dark') return stored;
    return systemPrefersLight() ? 'light' : 'dark';
}

export function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('themeToggle');
    if (btn) {
        const isLight = theme === 'light';
        btn.setAttribute('aria-pressed', String(isLight));
        btn.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
        const icon = btn.querySelector('.theme-icon');
        const label = btn.querySelector('.theme-label');
        if (icon) icon.textContent = isLight ? '\u2600\uFE0F' : '\uD83C\uDF19';
        if (label) label.textContent = isLight ? 'Light' : 'Dark';
    }
    // Let charts (and others) react.
    document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

export function setTheme(theme) {
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (_e) { /* ignore */ }
    applyTheme(theme);
}

export function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || resolveInitialTheme();
    setTheme(current === 'light' ? 'dark' : 'light');
}

export function initTheme() {
    applyTheme(resolveInitialTheme());
    const btn = document.getElementById('themeToggle');
    if (btn) btn.addEventListener('click', toggleTheme);

    // Follow system changes only when the user hasn't chosen explicitly.
    if (window.matchMedia) {
        const mq = window.matchMedia('(prefers-color-scheme: light)');
        const handler = (e) => { if (!getStoredTheme()) applyTheme(e.matches ? 'light' : 'dark'); };
        if (mq.addEventListener) mq.addEventListener('change', handler);
        else if (mq.addListener) mq.addListener(handler);
    }
}
