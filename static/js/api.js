/* ============================================================
   api.js — thin wrappers around the same-origin JSON API.
   All requests use credentials:'same-origin' so the session
   cookie is sent. The JSON API is CSRF-exempt (SameSite).
   ============================================================ */

/** Error carrying the HTTP status and parsed body. */
export class ApiError extends Error {
    constructor(message, status, body) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.body = body || {};
    }
}

async function request(path, { method = 'GET', body, signal } = {}) {
    const opts = {
        method,
        credentials: 'same-origin',
        headers: { Accept: 'application/json' },
        signal,
    };
    if (body !== undefined) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }

    const res = await fetch(path, opts);

    // No content
    if (res.status === 204) return {};

    let data = null;
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
        data = await res.json().catch(() => null);
    } else {
        data = { _text: await res.text().catch(() => '') };
    }

    if (!res.ok) {
        const msg = (data && (data.error || data.message)) || `Request failed (${res.status})`;
        throw new ApiError(msg, res.status, data || {});
    }
    return data;
}

/* ─── Auth ─────────────────────────────────────────────── */
export const getSession = () => request('/auth/session');
export const dashboardLogin = (username, password) =>
    request('/auth/dashboard-login', { method: 'POST', body: { username, password } });
export const logout = () => request('/auth/logout', { method: 'POST' });

/* ─── Fyers connection ─────────────────────────────────── */
export const getFyersLoginUrl = () => request('/auth/login');
export const connectFyers = (auth_code) =>
    request('/auth/connect', { method: 'POST', body: { auth_code } });
export const getStatus = () => request('/api/status');

/* ─── Analysis ─────────────────────────────────────────── */
export const analyze = (payload, signal) =>
    request('/api/analyze', { method: 'POST', body: payload, signal });
export const runDaily = (payload) =>
    request('/api/run-daily', { method: 'POST', body: payload });
export const getJob = (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}`);

/* ─── Stored data ──────────────────────────────────────── */
export const getStoredDates = () => request('/api/stored-dates');
export const getStoredData = (payload) =>
    request('/api/stored-data', { method: 'POST', body: payload });

/* ─── History (for charts) ─────────────────────────────── */
export const getHistory = (symbol, { resolution = 'D', days = 365 } = {}) =>
    request(`/api/history/${encodeURIComponent(symbol)}?resolution=${resolution}&days=${days}`);

/* ─── Watchlist ────────────────────────────────────────── */
export const getWatchlist = () => request('/api/watchlist');
export const addWatchlist = (symbol, name) =>
    request('/api/watchlist', { method: 'POST', body: { symbol, name } });
export const removeWatchlist = (symbol) =>
    request(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' });

/* ─── Alerts ───────────────────────────────────────────── */
export const getAlerts = () => request('/api/alerts');
export const addAlert = (payload) => request('/api/alerts', { method: 'POST', body: payload });
export const removeAlert = (id) =>
    request(`/api/alerts/${encodeURIComponent(id)}`, { method: 'DELETE' });

/* ─── Backtest ─────────────────────────────────────────── */
export const runBacktest = (payload) =>
    request('/api/backtest', { method: 'POST', body: payload });

/* ─── Chat / LLM ───────────────────────────────────────── */
export const chat = (message, stock_context) =>
    request('/api/chat', { method: 'POST', body: { message, stock_context } });
export const llmAnalyze = (stocks) =>
    request('/api/llm-analyze', { method: 'POST', body: { stocks } });

/* ─── Export (file download) ───────────────────────────── */
export function exportUrl({ date, category, format = 'csv' }) {
    const p = new URLSearchParams({ date: date || '', category: category || '', format });
    return `/api/export?${p.toString()}`;
}
