// chat.js — AI chat widget. Sends the current result set (trimmed) as
// context. Bot replies are HTML-ish; rendered into a sanitized container.

import { chat } from './api.js';

// Minimal HTML sanitizer: allow a small set of formatting tags, strip the rest.
const ALLOWED = new Set(['B', 'STRONG', 'I', 'EM', 'BR', 'UL', 'OL', 'LI', 'P', 'SPAN', 'CODE', 'DIV', 'SMALL']);

function sanitize(html) {
    const tpl = document.createElement('template');
    tpl.innerHTML = String(html);
    const walk = (node) => {
        const children = Array.from(node.childNodes);
        for (const child of children) {
            if (child.nodeType === 1) { // element
                if (!ALLOWED.has(child.tagName)) {
                    // Replace disallowed element with its text content.
                    child.replaceWith(document.createTextNode(child.textContent || ''));
                    continue;
                }
                // Strip all attributes (no on*, href, style, etc.).
                Array.from(child.attributes).forEach((a) => child.removeAttribute(a.name));
                walk(child);
            } else if (child.nodeType === 8) { // comment
                child.remove();
            }
        }
    };
    walk(tpl.content);
    return tpl.innerHTML;
}

export function initChat(getStockContext) {
    const fab = document.getElementById('chatFab');
    const panel = document.getElementById('chatPanel');
    const closeBtn = document.getElementById('chatClose');
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const messages = document.getElementById('chatMessages');
    const typing = document.getElementById('chatTyping');
    if (!fab || !panel) return;

    function toggle(force) {
        const open = force !== undefined ? force : !panel.classList.contains('open');
        panel.classList.toggle('open', open);
        fab.setAttribute('aria-expanded', String(open));
        fab.innerHTML = open ? '\u2715' : '\uD83D\uDCAC';
        if (open && input) input.focus();
    }
    fab.addEventListener('click', () => toggle());
    if (closeBtn) closeBtn.addEventListener('click', () => { toggle(false); fab.focus(); });

    async function send() {
        const msg = (input.value || '').trim();
        if (!msg) return;

        const userEl = document.createElement('div');
        userEl.className = 'chat-msg user';
        userEl.textContent = msg;
        messages.appendChild(userEl);
        input.value = '';
        sendBtn.disabled = true;
        if (typing) typing.classList.add('show');
        messages.scrollTop = messages.scrollHeight;

        try {
            const ctx = (getStockContext ? getStockContext() : []).slice(0, 20).map((s) => ({
                name: s.name, price: s.price, score: s.score, rsi: s.rsi,
                macd_hist: s.macd_hist, change_pct: s.change_pct,
                sma20: s.sma20, sma50: s.sma50, sma200: s.sma200,
                adx: s.adx, vol_ratio: s.vol_ratio, signal: s.signal,
            }));
            const data = await chat(msg, ctx);
            const botEl = document.createElement('div');
            botEl.className = 'chat-msg bot';
            botEl.innerHTML = sanitize(data.reply || data.error || 'No response');
            messages.appendChild(botEl);
        } catch (err) {
            const errEl = document.createElement('div');
            errEl.className = 'chat-msg bot';
            errEl.textContent = 'Error: ' + (err.message || 'request failed');
            messages.appendChild(errEl);
        } finally {
            if (typing) typing.classList.remove('show');
            sendBtn.disabled = false;
            messages.scrollTop = messages.scrollHeight;
        }
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
}
