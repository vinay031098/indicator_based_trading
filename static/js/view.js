// view.js — renders the stock result set as either a virtualized card
// grid (windowing) or a sortable/filterable table. Virtualization is
// window-scroll based so 2000+ items stay smooth.

import { inr, num, signed, pct, signalLabel, aiActionLabel, esc, deriveSignal, netScore } from './format.js';

const CARD_H = 420;     // fixed card height (px) — required for windowing
const GAP = 16;
const ROW_H = 42;       // table row height (px)
const BUFFER = 6;       // extra rows above/below the viewport

const TABLE_COLS = [
    { key: 'name', label: 'Name', type: 'text' },
    { key: 'price', label: 'Price', type: 'num', fmt: (s) => inr(s.price) },
    { key: 'change_pct', label: 'Chg %', type: 'num', fmt: (s) => pct(s.change_pct) },
    { key: 'score', label: 'Bull', type: 'num', fmt: (s) => num(s.score, 0) },
    { key: 'bear_score', label: 'Bear', type: 'num', fmt: (s) => num(s.bear_score, 0) },
    { key: 'net_score', label: 'Net', type: 'num', fmt: (s) => signed(netScore(s), 0) },
    { key: 'signal', label: 'Signal', type: 'text', fmt: (s) => signalLabel(deriveSignal(s)).text },
    { key: 'rsi', label: 'RSI', type: 'num', fmt: (s) => num(s.rsi, 1) },
    { key: 'macd', label: 'MACD', type: 'num', fmt: (s) => num(s.macd, 2) },
    { key: 'macd_signal', label: 'MACD Sig', type: 'num', fmt: (s) => num(s.macd_signal, 2) },
    { key: 'macd_hist', label: 'MACD Hist', type: 'num', fmt: (s) => signed(s.macd_hist, 2) },
    { key: 'sma20', label: 'SMA20', type: 'num', fmt: (s) => num(s.sma20, 2) },
    { key: 'sma50', label: 'SMA50', type: 'num', fmt: (s) => num(s.sma50, 2) },
    { key: 'sma200', label: 'SMA200', type: 'num', fmt: (s) => num(s.sma200, 2) },
    { key: 'ema9', label: 'EMA9', type: 'num', fmt: (s) => num(s.ema9, 2) },
    { key: 'ema21', label: 'EMA21', type: 'num', fmt: (s) => num(s.ema21, 2) },
    { key: 'stoch_k', label: 'Stoch K', type: 'num', fmt: (s) => num(s.stoch_k, 1) },
    { key: 'stoch_d', label: 'Stoch D', type: 'num', fmt: (s) => num(s.stoch_d, 1) },
    { key: 'bb_upper', label: 'BB Up', type: 'num', fmt: (s) => num(s.bb_upper, 2) },
    { key: 'bb_lower', label: 'BB Lo', type: 'num', fmt: (s) => num(s.bb_lower, 2) },
    { key: 'bb_width', label: 'BB W', type: 'num', fmt: (s) => num(s.bb_width, 1) },
    { key: 'adx', label: 'ADX', type: 'num', fmt: (s) => num(s.adx, 1) },
    { key: 'plus_di', label: 'DI+', type: 'num', fmt: (s) => num(s.plus_di, 1) },
    { key: 'minus_di', label: 'DI-', type: 'num', fmt: (s) => num(s.minus_di, 1) },
    { key: 'cci', label: 'CCI', type: 'num', fmt: (s) => num(s.cci, 0) },
    { key: 'williams_r', label: 'W%R', type: 'num', fmt: (s) => num(s.williams_r, 0) },
    { key: 'mfi', label: 'MFI', type: 'num', fmt: (s) => num(s.mfi, 1) },
    { key: 'atr', label: 'ATR', type: 'num', fmt: (s) => num(s.atr, 2) },
    { key: 'atr_pct', label: 'ATR %', type: 'num', fmt: (s) => num(s.atr_pct, 1) },
    { key: 'roc', label: 'ROC', type: 'num', fmt: (s) => signed(s.roc, 1) },
    { key: 'cmf', label: 'CMF', type: 'num', fmt: (s) => num(s.cmf, 3) },
    { key: 'vwap', label: 'VWAP', type: 'num', fmt: (s) => num(s.vwap, 2) },
    { key: 'ichimoku_tenkan', label: 'Tenkan', type: 'num', fmt: (s) => num(s.ichimoku_tenkan, 2) },
    { key: 'ichimoku_kijun', label: 'Kijun', type: 'num', fmt: (s) => num(s.ichimoku_kijun, 2) },
    { key: 'pivot', label: 'Pivot', type: 'num', fmt: (s) => num(s.pivot, 2) },
    { key: 'pivot_s1', label: 'S1', type: 'num', fmt: (s) => num(s.pivot_s1, 2) },
    { key: 'pivot_r1', label: 'R1', type: 'num', fmt: (s) => num(s.pivot_r1, 2) },
    { key: 'vol_ratio', label: 'Vol×', type: 'num', fmt: (s) => num(s.vol_ratio, 2) + '\u00D7' },
    { key: 'w52_high', label: '52W Hi', type: 'num', fmt: (s) => num(s.w52_high, 2) },
    { key: 'w52_low', label: '52W Lo', type: 'num', fmt: (s) => num(s.w52_low, 2) },
    { key: 'dist_52w', label: 'Dist52W', type: 'num', fmt: (s) => num(s.dist_52w, 1) },
];

/** All indicator tiles shown on each card (matches backend ``analyze_stock`` output). */
const CARD_INDICATORS = [
    { key: 'rsi', label: 'RSI', fmt: (s) => num(s.rsi, 1) },
    { key: 'macd_hist', label: 'MACD', fmt: (s) => signed(s.macd_hist, 2) },
    { key: 'adx', label: 'ADX', fmt: (s) => num(s.adx, 0) },
    { key: 'mfi', label: 'MFI', fmt: (s) => num(s.mfi, 1) },
    { key: 'cci', label: 'CCI', fmt: (s) => num(s.cci, 0) },
    { key: 'williams_r', label: 'W%R', fmt: (s) => num(s.williams_r, 0) },
    { key: 'stoch_k', label: 'StochK', fmt: (s) => num(s.stoch_k, 1) },
    { key: 'stoch_d', label: 'StochD', fmt: (s) => num(s.stoch_d, 1) },
    { key: 'roc', label: 'ROC', fmt: (s) => signed(s.roc, 1) },
    { key: 'cmf', label: 'CMF', fmt: (s) => num(s.cmf, 2) },
    { key: 'atr_pct', label: 'ATR%', fmt: (s) => num(s.atr_pct, 1) },
    { key: 'vol_ratio', label: 'Vol', fmt: (s) => num(s.vol_ratio, 1) + '×' },
    { key: 'sma20', label: 'SMA20', fmt: (s) => num(s.sma20, 0) },
    { key: 'sma50', label: 'SMA50', fmt: (s) => num(s.sma50, 0) },
    { key: 'ema9', label: 'EMA9', fmt: (s) => num(s.ema9, 0) },
    { key: 'ema21', label: 'EMA21', fmt: (s) => num(s.ema21, 0) },
    { key: 'vwap', label: 'VWAP', fmt: (s) => num(s.vwap, 0) },
    { key: 'dist_52w', label: '52W', fmt: (s) => num(s.dist_52w, 1) },
    { key: 'pivot', label: 'Pivot', fmt: (s) => num(s.pivot, 0) },
    { key: 'tenkan', label: 'Tenkan', fmt: (s) => num(s.ichimoku_tenkan, 0) },
];

export class StockView {
    /**
     * @param {object} cfg
     *  - gridViewport, gridSizer, tableWrap: DOM elements
     *  - getAiRec(name) -> recommendation|null
     *  - isStarred(symbol) -> bool
     *  - onSelect(stock), onToggleStar(stock)
     */
    constructor(cfg) {
        this.grid = cfg.gridViewport;
        this.sizer = cfg.gridSizer;
        this.tableWrap = cfg.tableWrap;
        this.getAiRec = cfg.getAiRec || (() => null);
        this.isStarred = cfg.isStarred || (() => false);
        this.onSelect = cfg.onSelect || (() => {});
        this.onToggleStar = cfg.onToggleStar || (() => {});

        this.items = [];
        this.mode = 'grid';
        this.sortKey = 'score';
        this.sortDir = 'desc';
        this.cols = 1;
        this._raf = null;

        this._onScroll = () => this._schedule();
        this._onResize = () => { this._measure(); this._schedule(); };
        window.addEventListener('scroll', this._onScroll, { passive: true });
        window.addEventListener('resize', this._onResize);

        // Delegated interactions for both views.
        this.grid.addEventListener('click', (e) => this._handleGridClick(e));
        this.grid.addEventListener('keydown', (e) => this._handleGridKey(e));
    }

    setItems(items) {
        this.items = Array.isArray(items) ? items.slice() : [];
        if (this.mode === 'table') this._sortItems();
        this.render();
    }

    setMode(mode) {
        this.mode = mode;
        if (mode === 'grid') {
            this.grid.style.display = '';
            this.tableWrap.style.display = 'none';
        } else {
            this.grid.style.display = 'none';
            this.tableWrap.style.display = '';
            this._sortItems();
        }
        this.render();
    }

    refresh() { this.render(); }

    render() {
        if (this.items.length === 0) {
            this._renderEmpty();
            return;
        }
        if (this.mode === 'grid') {
            this.tableWrap.innerHTML = '';
            this._measure();
            this._renderGridWindow();
        } else {
            this.grid.style.height = '';
            this.sizer.style.height = '';
            this.sizer.innerHTML = '';
            this._renderTable();
        }
    }

    _renderEmpty() {
        const html = `
            <div class="empty-state">
                <div class="icon" aria-hidden="true">\uD83D\uDD0D</div>
                <h3>No stocks match this filter</h3>
                <p>Try adjusting the minimum score, filter, or search.</p>
            </div>`;
        if (this.mode === 'grid') {
            this.sizer.style.height = '';
            this.sizer.innerHTML = html;
            this.tableWrap.innerHTML = '';
        } else {
            this.tableWrap.innerHTML = `<div style="padding:0 32px">${html}</div>`;
            this.sizer.innerHTML = '';
        }
    }

    // ─── Grid windowing ─────────────────────────────────────
    _measure() {
        const width = this.sizer.clientWidth || this.grid.clientWidth || 320;
        this.cols = Math.max(1, Math.floor((width + GAP) / (320 + GAP)));
        this.colW = (width - (this.cols - 1) * GAP) / this.cols;
        const rows = Math.ceil(this.items.length / this.cols);
        this.sizer.style.height = `${rows * (CARD_H + GAP)}px`;
    }

    _schedule() {
        if (this.items.length === 0) return;
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => {
            this._raf = null;
            if (this.mode === 'grid') {
                this._renderGridWindow();
            } else if (this._tbody && this.items.length > 300) {
                this._tbody.innerHTML = this._tableRowWindow();
            }
        });
    }

    _renderGridWindow() {
        const sizerTop = this.sizer.getBoundingClientRect().top + window.scrollY;
        const viewTop = window.scrollY - sizerTop;
        const viewBottom = viewTop + window.innerHeight;
        const rowStride = CARD_H + GAP;
        let firstRow = Math.floor(viewTop / rowStride) - BUFFER;
        let lastRow = Math.ceil(viewBottom / rowStride) + BUFFER;
        firstRow = Math.max(0, firstRow);
        const totalRows = Math.ceil(this.items.length / this.cols);
        lastRow = Math.min(totalRows - 1, lastRow);

        let html = '';
        for (let row = firstRow; row <= lastRow; row++) {
            for (let col = 0; col < this.cols; col++) {
                const idx = row * this.cols + col;
                if (idx >= this.items.length) break;
                const top = row * rowStride;
                const left = col * (this.colW + GAP);
                html += this._cardHTML(this.items[idx], top, left, this.colW);
            }
        }
        this.sizer.innerHTML = html;
    }

    _cardHTML(s, top, left, width) {
        const score = Number(s.score) || 0;
        const tier = score >= 15 ? 'score-tier-top' : score >= 10 ? 'score-tier-high'
            : score >= 7 ? 'score-tier-good' : score >= 4 ? 'score-tier-mid'
            : score >= 2 ? 'score-tier-low' : 'score-tier-weak';
        const scoreClass = score >= 12 ? 'score-high' : score >= 7 ? 'score-med'
            : score >= 4 ? 'score-low' : 'score-none';
        const changeUp = Number(s.change) >= 0;
        const sig = signalLabel(deriveSignal(s));
        const net = netScore(s);

        const ai = this.getAiRec(s.name);
        let aiClass = '';
        let aiBadge = '';
        if (ai) {
            const a = aiActionLabel(ai.action);
            aiClass = a.cls === 'buy' ? 'ai-buy' : a.cls === 'sell' ? 'ai-sell' : 'ai-neutral';
            const tgt = (ai.action === 'BUY' && ai.target)
                ? ` &middot; Target ${esc(inr(ai.target))}` : '';
            aiBadge = `<div class="card-ai-badge ${a.cls}"><span class="ai-label">AI: ${esc(a.text)}</span>
                <span style="color:var(--text-muted)"> ${esc(ai.confidence || '')}${tgt}</span></div>`;
        }

        const starred = this.isStarred(s.symbol);
        const reasons = (s.reasons || []).slice(0, 2).map((r) =>
            `<div class="reason ${esc(r.type)}"><span aria-hidden="true">${esc(r.icon || '')}</span> ${esc(r.text)}</div>`
        ).join('');

        const indicatorTiles = CARD_INDICATORS.map(({ label, fmt }) =>
            `<div class="indicator"><div class="indicator-value">${esc(fmt(s))}</div><div class="indicator-label">${esc(label)}</div></div>`
        ).join('');

        return `
        <article class="stock-card ${tier} ${aiClass}" role="button" tabindex="0"
            aria-label="${esc(s.name)}, score ${score} of 30, signal ${esc(sig.text)}"
            data-symbol="${esc(s.symbol)}" data-name="${esc(s.name)}"
            style="top:${top}px; left:${left}px; width:${width}px; height:${CARD_H}px;">
            ${aiBadge}
            <div class="card-header">
                <div class="card-head-left">
                    <div class="stock-name">${esc(s.name)}</div>
                    <div class="stock-exchange">NSE &middot; Net ${esc(signed(net, 0))} &middot; Bull ${score} / Bear ${Number(s.bear_score) || 0}</div>
                </div>
                <div class="card-head-right">
                    <button class="star-btn ${starred ? 'starred' : ''}" data-action="star"
                        aria-pressed="${starred}" aria-label="${starred ? 'Remove' : 'Add'} ${esc(s.name)} ${starred ? 'from' : 'to'} watchlist"
                        title="Watchlist">${starred ? '\u2605' : '\u2606'}</button>
                    <span class="score-badge ${scoreClass}">${score}</span>
                </div>
            </div>
            <div class="price-row">
                <span class="price">${esc(inr(s.price))}</span>
                <span class="change ${changeUp ? 'up' : 'down'}">${esc(signed(s.change))} (${esc(pct(s.change_pct))})</span>
                <span class="signal-pill ${sig.cls}"><span class="pill-shape" aria-hidden="true">${sig.shape}</span>${esc(sig.text)}</span>
            </div>
            <div class="indicators indicators-full">${indicatorTiles}</div>
            <div class="reasons">${reasons}</div>
        </article>`;
    }

    // ─── Table view ─────────────────────────────────────────
    _sortItems() {
        const col = TABLE_COLS.find((c) => c.key === this.sortKey) || TABLE_COLS[0];
        const dir = this.sortDir === 'asc' ? 1 : -1;
        this.items.sort((a, b) => {
            let av = a[col.key];
            let bv = b[col.key];
            if (col.key === 'net_score') {
                av = netScore(a); bv = netScore(b);
            } else if (col.key === 'signal') {
                av = deriveSignal(a); bv = deriveSignal(b);
            } else if (col.type === 'num') {
                av = Number(av); bv = Number(bv);
                if (!isFinite(av)) av = -Infinity;
                if (!isFinite(bv)) bv = -Infinity;
                return (av - bv) * dir;
            }
            return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
        });
    }

    setSort(key) {
        if (this.sortKey === key) {
            this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortKey = key;
            this.sortDir = key === 'name' || key === 'signal' ? 'asc' : 'desc';
        }
        this._sortItems();
        this.render();
    }

    _renderTable() {
        const head = TABLE_COLS.map((c) => {
            const active = c.key === this.sortKey;
            const ind = active ? (this.sortDir === 'asc' ? '\u25B2' : '\u25BC') : '';
            const aria = active ? (this.sortDir === 'asc' ? 'ascending' : 'descending') : 'none';
            return `<th scope="col" data-sort="${c.key}" role="columnheader" aria-sort="${aria}"
                tabindex="0">${esc(c.label)}<span class="sort-ind" aria-hidden="true">${ind}</span></th>`;
        }).join('');

        // Window the rows for performance.
        const rows = this._tableRowWindow();
        this.tableWrap.innerHTML = `
            <table class="stock-table" aria-label="Stock results table">
                <thead><tr>${head}</tr></thead>
                <tbody>${rows}</tbody>
            </table>`;

        this.tableWrap.querySelectorAll('th[data-sort]').forEach((th) => {
            const handler = () => this.setSort(th.getAttribute('data-sort'));
            th.addEventListener('click', handler);
            th.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
            });
        });
        const tbody = this.tableWrap.querySelector('tbody');
        this._tbody = tbody;
        tbody.addEventListener('click', (e) => this._handleRowActivate(e));
        tbody.addEventListener('keydown', (e) => {
            if ((e.key === 'Enter' || e.key === ' ') && e.target.closest('tr[data-symbol]')) {
                e.preventDefault(); this._handleRowActivate(e);
            }
        });
    }

    _tableRowWindow() {
        // For very large lists, render a windowed slice based on scroll.
        const total = this.items.length;
        if (total <= 300) {
            return this.items.map((s) => this._rowHTML(s)).join('');
        }
        // Anchor windowing to the table position.
        const wrapTop = (this.tableWrap.getBoundingClientRect().top + window.scrollY) || 0;
        const headOffset = 40;
        const viewTop = window.scrollY - wrapTop - headOffset;
        const first = Math.max(0, Math.floor(viewTop / ROW_H) - 20);
        const visible = Math.ceil(window.innerHeight / ROW_H) + 40;
        const last = Math.min(total, first + visible);
        const topPad = first * ROW_H;
        const botPad = (total - last) * ROW_H;
        const colspan = TABLE_COLS.length;
        let html = `<tr class="table-spacer" aria-hidden="true"><td colspan="${colspan}" style="height:${topPad}px"></td></tr>`;
        for (let i = first; i < last; i++) html += this._rowHTML(this.items[i]);
        html += `<tr class="table-spacer" aria-hidden="true"><td colspan="${colspan}" style="height:${botPad}px"></td></tr>`;
        return html;
    }

    _rowHTML(s) {
        const cells = TABLE_COLS.map((c) => {
            let cls = '';
            if (c.key === 'change_pct') cls = Number(s.change_pct) >= 0 ? 'cell-up' : 'cell-down';
            if (c.key === 'signal') {
                const sg = signalLabel(deriveSignal(s));
                cls = sg.cls === 'buy' ? 'cell-up' : sg.cls === 'sell' ? 'cell-down' : '';
            }
            return `<td class="${cls}">${esc(c.fmt ? c.fmt(s) : s[c.key])}</td>`;
        }).join('');
        return `<tr data-symbol="${esc(s.symbol)}" data-name="${esc(s.name)}" tabindex="0"
            aria-label="${esc(s.name)} details">${cells}</tr>`;
    }

    // ─── Interaction handlers ───────────────────────────────
    _findStock(symbol, name) {
        return this.items.find((s) => s.symbol === symbol)
            || this.items.find((s) => s.name === name) || null;
    }

    _handleGridClick(e) {
        const star = e.target.closest('[data-action="star"]');
        const card = e.target.closest('.stock-card');
        if (!card) return;
        const stock = this._findStock(card.getAttribute('data-symbol'), card.getAttribute('data-name'));
        if (!stock) return;
        if (star) { e.stopPropagation(); this.onToggleStar(stock); return; }
        this.onSelect(stock);
    }

    _handleGridKey(e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const card = e.target.closest('.stock-card');
        if (!card || e.target.closest('[data-action="star"]')) return;
        e.preventDefault();
        const stock = this._findStock(card.getAttribute('data-symbol'), card.getAttribute('data-name'));
        if (stock) this.onSelect(stock);
    }

    _handleRowActivate(e) {
        const tr = e.target.closest('tr[data-symbol]');
        if (!tr) return;
        const stock = this._findStock(tr.getAttribute('data-symbol'), tr.getAttribute('data-name'));
        if (stock) this.onSelect(stock);
    }
}
