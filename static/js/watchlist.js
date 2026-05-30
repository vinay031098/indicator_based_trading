// watchlist.js — watchlist state, persistence via API, and drawer UI.

import { getWatchlist, addWatchlist, removeWatchlist } from './api.js';
import { notify } from './toast.js';
import { openModal, closeModal, bindModalDismiss } from './modal.js';
import { esc } from './format.js';

const state = {
    symbols: new Set(),
    items: [],
    listeners: [],
};

export function onWatchlistChange(fn) { state.listeners.push(fn); }
function emit() { state.listeners.forEach((fn) => { try { fn(); } catch (_e) { /* noop */ } }); }

export function isStarred(symbol) { return state.symbols.has(symbol); }

export async function loadWatchlist() {
    try {
        const data = await getWatchlist();
        state.items = data.watchlist || [];
        state.symbols = new Set(state.items.map((w) => w.symbol));
        emit();
    } catch (_e) {
        // Watchlist is optional; ignore load failures silently.
    }
}

export async function toggleWatchlist(stock) {
    const sym = stock.symbol;
    if (!sym) return;
    try {
        if (state.symbols.has(sym)) {
            await removeWatchlist(sym);
            state.symbols.delete(sym);
            state.items = state.items.filter((w) => w.symbol !== sym);
            notify.info(`Removed ${stock.name} from watchlist`);
        } else {
            await addWatchlist(sym, stock.name);
            state.symbols.add(sym);
            state.items.push({ symbol: sym, name: stock.name, added_at: new Date().toISOString() });
            notify.success(`Added ${stock.name} to watchlist`);
        }
        emit();
        renderDrawer();
    } catch (err) {
        notify.error(err.message || 'Watchlist update failed');
    }
}

export function renderDrawer() {
    const body = document.getElementById('watchlistBody');
    const count = document.getElementById('watchlistCount');
    if (count) count.textContent = String(state.items.length);
    if (!body) return;
    if (state.items.length === 0) {
        body.innerHTML = `<div class="empty-state" style="padding:30px"><div class="icon" aria-hidden="true">\u2606</div>
            <h3>Your watchlist is empty</h3><p>Use the star on any stock to add it here.</p></div>`;
        return;
    }
    body.innerHTML = `<div class="watchlist-list">${state.items.map((w) => `
        <div class="watchlist-item">
            <div>
                <div class="wl-name">${esc(w.name || w.symbol)}</div>
                <div class="wl-meta">${esc(w.symbol)}</div>
            </div>
            <div class="watchlist-actions">
                <button class="wl-remove" data-remove="${esc(w.symbol)}" aria-label="Remove ${esc(w.name || w.symbol)} from watchlist">Remove</button>
            </div>
        </div>`).join('')}</div>`;

    body.querySelectorAll('[data-remove]').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const sym = btn.getAttribute('data-remove');
            const item = state.items.find((w) => w.symbol === sym);
            await toggleWatchlist({ symbol: sym, name: item ? item.name : sym });
        });
    });
}

export function initWatchlistUI() {
    const overlay = document.getElementById('watchlistModal');
    const openBtn = document.getElementById('watchlistBtn');
    if (overlay) bindModalDismiss(overlay);
    if (openBtn && overlay) {
        openBtn.addEventListener('click', () => { renderDrawer(); openModal(overlay, { trigger: openBtn }); });
    }
    onWatchlistChange(() => {
        const count = document.getElementById('watchlistCount');
        if (count) count.textContent = String(state.items.length);
    });
}

export function closeWatchlist() {
    const overlay = document.getElementById('watchlistModal');
    if (overlay) closeModal(overlay);
}
