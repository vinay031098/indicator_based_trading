// format.js — small formatting helpers shared across modules.

export function inr(value) {
    const n = Number(value);
    if (!isFinite(n)) return '-';
    return '\u20B9' + n.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

export function num(value, digits = 2) {
    const n = Number(value);
    if (value === null || value === undefined || !isFinite(n)) return '-';
    return n.toFixed(digits);
}

export function signed(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return '-';
    return (n >= 0 ? '+' : '') + n.toFixed(digits);
}

export function pct(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return '-';
    return (n >= 0 ? '+' : '') + n.toFixed(digits) + '%';
}

// Map backend signal/AI action to an accessible label (text, not color only).
export function signalLabel(signal) {
    switch ((signal || '').toUpperCase()) {
        case 'BUY': return { cls: 'buy', text: 'Buy', shape: '\u25B2' };       // triangle up
        case 'SELL': return { cls: 'sell', text: 'Sell', shape: '\u25BC' };    // triangle down
        default: return { cls: 'neutral', text: 'Neutral', shape: '\u25CF' };  // circle
    }
}

/** Must match ``strategy.STRATEGY.thresholds`` (buy / sell). */
export const SIGNAL_THRESHOLDS = { buy: 3, sell: -3 };

export function netScore(stock) {
    if (typeof stock === 'number') return stock;
    const bull = Number(stock.score) || 0;
    const bear = Number(stock.bear_score) || 0;
    if (stock.net_score != null && stock.net_score !== '') return Number(stock.net_score);
    return bull - bear;
}

/** Derive BUY / NEUTRAL / SELL from net score (single source of truth). */
export function deriveSignal(stockOrNet) {
    const net = typeof stockOrNet === 'object' ? netScore(stockOrNet) : Number(stockOrNet);
    if (net >= SIGNAL_THRESHOLDS.buy) return 'BUY';
    if (net <= SIGNAL_THRESHOLDS.sell) return 'SELL';
    return 'NEUTRAL';
}

export function aiActionLabel(action) {
    switch ((action || '').toUpperCase()) {
        case 'BUY': return { cls: 'buy', text: 'Buy', shape: '\u25B2' };
        case 'AVOID': return { cls: 'sell', text: 'Avoid / Sell', shape: '\u25BC' };
        case 'HOLD': return { cls: 'neutral', text: 'Hold / Neutral', shape: '\u25CF' };
        default: return { cls: 'neutral', text: action || 'Neutral', shape: '\u25CF' };
    }
}

// Escape text for safe insertion into HTML strings.
export function esc(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
