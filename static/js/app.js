/* ============================================================
   app.js — application entry point and orchestration.

   Wires the shared modules (theme, modal, toast, view, detail,
   watchlist, chat, disclaimer) to the markup in index.html and
   the same-origin JSON API. Owns auth flow, Fyers connection,
   stored-date loading, live analysis, background-job polling,
   stats, filters/search, view toggle, AI recommendations and
   export.
   ============================================================ */

import * as api from './api.js';
import { StockView } from './view.js';
import { initDetail, openDetail } from './detail.js';
import { loadWatchlist, toggleWatchlist, isStarred, onWatchlistChange, initWatchlistUI } from './watchlist.js';
import { initChat } from './chat.js';
import { initTheme } from './theme.js';
import { openModal, bindModalDismiss } from './modal.js';
import { initDisclaimer } from './disclaimer.js';
import { renderEquityCurve } from './charts.js';
import { esc } from './format.js';
import { toast } from './toast.js';

const $ = (id) => document.getElementById(id);
const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

const CATEGORY_LABELS = {
    nifty50: 'NIFTY 50', nifty100: 'NIFTY 100', nifty200: 'NIFTY 200',
    nifty500: 'NIFTY 500', all: 'All NSE',
};

/** Must match ``strategy.STRATEGY.thresholds`` (buy / sell). */
const SIGNAL_THRESHOLDS = { buy: 3, sell: -3 };

function enrichStock(s) {
    const bull = Number(s.score) || 0;
    const bear = Number(s.bear_score) || 0;
    if (s.net_score == null) s.net_score = bull - bear;
    let sig = (s.signal || '').toUpperCase();
    if (!sig) {
        if (s.net_score >= SIGNAL_THRESHOLDS.buy) sig = 'BUY';
        else if (s.net_score <= SIGNAL_THRESHOLDS.sell) sig = 'SELL';
        else sig = 'NEUTRAL';
        s.signal = sig;
    }
    return s;
}

function enrichAll(stocks) {
    return (stocks || []).map(enrichStock);
}

/* ─── State ────────────────────────────────────────────────── */
const state = {
    allResults: [],
    ai: {},                 // { STOCK_NAME: { action, confidence, reason, target, stoploss, risk_reward } }
    filter: 'qualified',
    search: '',
    view: 'grid',
    mode: 'stored',         // 'stored' | 'live'
    storedDates: [],
    fyersConnected: false,
    lastMeta: { date: null, category: null, category_label: null },
};

let view = null;
let pollTimer = null;
let connecting = false;
let equityChart = null;

const getAiRec = (name) => state.ai[name] || null;

/* ─── Auth flow ────────────────────────────────────────────── */
async function bootstrapAuth() {
    try {
        const s = await api.getSession();
        if (s && s.logged_in) showDashboard();
        else showLogin();
    } catch (_e) {
        showLogin();
    }
}

function showLogin() {
    $('loginOverlay').classList.remove('hidden');
    const u = $('loginUser');
    if (u) u.focus();
}

async function handleLogin(e) {
    e.preventDefault();
    const user = $('loginUser').value.trim();
    const pass = $('loginPass').value;
    const errEl = $('loginError');
    const box = $('loginBox');
    const btn = $('loginBtn');
    errEl.textContent = '';
    btn.disabled = true;
    btn.textContent = 'Signing in…';
    try {
        const res = await api.dashboardLogin(user, pass);
        if (res && res.success) {
            showDashboard();
        } else {
            throw new api.ApiError('Login failed', 401, res || {});
        }
    } catch (err) {
        errEl.textContent = (err.body && err.body.error) || err.message || 'Invalid username or password';
        box.classList.remove('login-shake');
        void box.offsetWidth;
        box.classList.add('login-shake');
        $('loginPass').value = '';
        $('loginPass').focus();
    } finally {
        btn.disabled = false;
        btn.textContent = 'Login';
    }
    return false;
}

async function handleLogout() {
    try { await api.logout(); } catch (_e) { /* ignore */ }
    location.reload();
}

function showDashboard() {
    $('loginOverlay').classList.add('hidden');
    $('dashboardContainer').style.display = '';
    initDisclaimer();
    refreshFyersStatus();
    loadWatchlist();
    loadStoredDates(true);
}

/* ─── Fyers status / connect ───────────────────────────────── */
async function refreshFyersStatus() {
    try {
        const s = await api.getStatus();
        setFyersConnected(!!s.authenticated);
    } catch (_e) { setFyersConnected(false); }
}

function setFyersConnected(connected) {
    state.fyersConnected = connected;
    const status = $('authStatus');
    const dot = $('statusDot');
    const label = $('authLabel');
    if (status) status.className = 'auth-status ' + (connected ? 'ok' : 'no');
    if (dot) dot.className = 'status-dot ' + (connected ? 'ok' : 'no');
    if (label) label.textContent = connected ? 'Fyers Connected' : 'Fyers Not Connected';
}

async function openFyersLogin() {
    const btn = $('fyersLoginBtn');
    btn.disabled = true;
    btn.textContent = 'Loading…';
    try {
        const data = await api.getFyersLoginUrl();
        if (data.auth_url && data.auth_url.includes('client_id=') && !/client_id=&|client_id=$/.test(data.auth_url)) {
            window.open(data.auth_url, '_blank');
            btn.textContent = 'Opened — paste the redirect URL below';
            $('authUrlInput').focus();
            if (data.redirect_uri) {
                toast(`Using redirect: ${data.redirect_uri}`, 'info', 6000);
            }
        } else {
            throw new Error(data.error || 'Server returned an invalid Fyers login URL (missing App ID). Check .env and restart.');
        }
    } catch (err) {
        toast(err.body?.error || err.message || 'Could not start Fyers login', 'error', 8000);
        btn.textContent = 'Open Fyers Login';
    } finally {
        btn.disabled = false;
    }
}

async function connectFyersNow() {
    const input = $('authUrlInput');
    const btn = $('connectBtn');
    const msg = $('connectMsg');
    const val = (input.value || '').trim();
    if (!val) return;
    let authCode = val;
    const m = val.match(/auth_code=([^&]+)/);
    if (m) authCode = decodeURIComponent(m[1]);
    connecting = true;
    btn.disabled = true;
    btn.textContent = 'Connecting…';
    try {
        const data = await api.connectFyers(authCode);
        if (data.success) {
            setFyersConnected(true);
            toast('Connected to Fyers', 'success');
            if (msg) msg.textContent = '';
            $('fyersConnectPanel').style.display = 'none';
        } else {
            throw new Error(data.error || 'Connection failed');
        }
    } catch (err) {
        toast(err.message || 'Connection failed', 'error');
        if (msg) msg.textContent = err.message || 'Connection failed';
    } finally {
        connecting = false;
        btn.disabled = false;
        btn.textContent = 'Connect';
    }
}

/* ─── Stored dates ─────────────────────────────────────────── */
async function loadStoredDates(autoLoad = false) {
    try {
        const data = await api.getStoredDates();
        state.storedDates = data.dates || [];
        renderDateChips();
        if (state.storedDates.length > 0) {
            $('storedDatesBar').classList.add('show');
            if (autoLoad && state.mode === 'stored') {
                const latest = state.storedDates[0];
                loadStoredData(latest.run_date, latest.category, document.querySelector('.date-chip'));
            }
        } else if (autoLoad) {
            showEmpty('No stored analyses yet', 'Switch to Live mode and connect Fyers, or use “Run & Store”.');
        }
    } catch (_e) { /* no stored dates */ }
}

function renderDateChips() {
    const container = $('dateChips');
    if (!container) return;
    if (!state.storedDates.length) {
        container.innerHTML = '<span style="color:var(--text-muted); font-size:13px;">No stored analyses yet. Use “Run &amp; Store” to create one.</span>';
        return;
    }
    container.innerHTML = '';
    state.storedDates.forEach((d) => {
        const dateObj = new Date(d.run_date + 'T00:00:00');
        const dayName = dateObj.toLocaleDateString('en-US', { weekday: 'short' });
        const dateStr = dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const chip = document.createElement('button');
        chip.className = 'date-chip';
        chip.type = 'button';
        chip.dataset.date = d.run_date;
        chip.setAttribute('aria-label', `Load stored analysis for ${dayName} ${dateStr}, ${d.category}`);
        const ai = d.ai_completed
            ? `<span class="chip-ai"><span style="color:var(--green)">B ${d.ai_buy_count ?? 0}</span> <span style="color:var(--yellow)">H ${d.ai_hold_count ?? 0}</span> <span style="color:var(--red)">A ${d.ai_avoid_count ?? 0}</span></span>`
            : '<span class="chip-ai" style="color:var(--text-muted)">No AI</span>';
        chip.innerHTML = `<span class="chip-date">${dayName}, ${dateStr}</span>
            <span class="chip-meta">${d.total_stocks} stocks · ${escAttr(d.category)}</span>${ai}`;
        chip.addEventListener('click', () => loadStoredData(d.run_date, d.category, chip));
        container.appendChild(chip);
    });
}

function escAttr(s) {
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ─── Mode toggle ──────────────────────────────────────────── */
function switchMode(mode) {
    state.mode = mode;
    $('modeStored').classList.toggle('active', mode === 'stored');
    $('modeStored').setAttribute('aria-pressed', String(mode === 'stored'));
    $('modeLive').classList.toggle('active', mode === 'live');
    $('modeLive').setAttribute('aria-pressed', String(mode === 'live'));
    $('runStoreBtn').style.display = mode === 'stored' ? '' : 'none';

    if (mode === 'live' && !state.fyersConnected && !state.allResults.length) {
        $('fyersConnectPanel').style.display = 'block';
        hideResultsChrome();
    } else {
        $('fyersConnectPanel').style.display = 'none';
    }
}

/* ─── Loading / progress ───────────────────────────────────── */
function showLoading(text) {
    const banner = $('progressBanner');
    if (banner) banner.classList.add('show');
    const t = $('progressText');
    if (t) t.textContent = text || 'Working…';
    setProgress(8);
    showSkeleton();
}
function hideLoading() {
    const banner = $('progressBanner');
    if (banner) banner.classList.remove('show');
    hideSkeleton();
}
function setProgress(pct) {
    const fill = $('progressFill');
    if (fill) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
}
function showSkeleton() {
    const sk = $('skeletonGrid');
    if (sk) sk.style.display = 'grid';
    $('emptyState').style.display = 'none';
    $('gridViewport').style.display = 'none';
    $('tableWrap').style.display = 'none';
    $('fyersConnectPanel').style.display = 'none';
}
function hideSkeleton() {
    const sk = $('skeletonGrid');
    if (sk) sk.style.display = 'none';
}
function showEmpty(title, msg) {
    const es = $('emptyState');
    if (!es) return;
    es.style.display = '';
    const h = $('emptyStateTitle');
    if (h) h.textContent = title;
    const p = es.querySelector('p');
    if (p && msg) p.textContent = msg;
    $('gridViewport').style.display = 'none';
    $('tableWrap').style.display = 'none';
}
function hideResultsChrome() {
    ['statsBar', 'filterBar', 'aiBar', 'gridViewport', 'tableWrap'].forEach((id) => {
        const el = $(id);
        if (el) el.style.display = 'none';
    });
    $('aiBar').classList.remove('show');
    $('emptyState').style.display = 'none';
}

/* ─── Live analysis ────────────────────────────────────────── */
async function runAnalysis() {
    const date = $('analysisDate').value;
    const minScore = $('minScore').value;
    const category = $('stockCategory').value;
    const btn = $('analyzeBtn');
    const label = CATEGORY_LABELS[category] || '';

    showLoading(`Fetching ${label} stocks from Fyers…`);
    setProgress(15);
    btn.disabled = true;
    btn.textContent = 'Analyzing…';

    const controller = new AbortController();
    const timeoutMap = { nifty50: 120, nifty100: 180, nifty200: 300, nifty500: 600, all: 900 };
    const timeoutId = setTimeout(() => controller.abort(), (timeoutMap[category] || 300) * 1000);
    try {
        const data = await api.analyze({ date, min_score: minScore, category }, controller.signal);
        setProgress(90);
        loadPayload(data);
        const buys = state.allResults.filter((s) => (s.signal || '').toUpperCase() === 'BUY').length;
        const sells = state.allResults.filter((s) => (s.signal || '').toUpperCase() === 'SELL').length;
        toast(`Analysis complete — ${buys} Buy, ${sells} Sell signals`, 'success');
    } catch (err) {
        hideSkeleton();
        if (err.name === 'AbortError') {
            toast('Request timed out — try a smaller universe or retry.', 'error', 7000);
        } else if (err.status === 401) {
            setFyersConnected(false);
            $('fyersConnectPanel').style.display = 'block';
            toast('Fyers token expired. Reconnect for live analysis (stored data still works).', 'error', 7000);
        } else {
            toast(err.message || 'Analysis failed', 'error');
        }
    } finally {
        clearTimeout(timeoutId);
        hideLoading();
        btn.disabled = false;
        btn.textContent = 'Analyze';
    }
}

/* ─── Run & Store (background job with progress) ───────────── */
async function runAndStore() {
    const date = $('analysisDate').value;
    const minScore = $('minScore').value;
    const category = $('stockCategory').value;
    const btn = $('runStoreBtn');
    const label = CATEGORY_LABELS[category] || '';

    showLoading(`Run & Store: analyzing ${label} + AI…`);
    btn.disabled = true;
    btn.textContent = 'Running…';
    try {
        const resp = await api.runDaily({ date, min_score: minScore, category, force: true });
        if (resp.exists) {
            toast(resp.message || 'A run for this date already exists.', 'info');
            await loadStoredData(date, category, null);
            return;
        }
        if (resp.job_id) {
            await pollJob(resp.job_id, date, category);
        } else {
            loadPayload(resp);
            toast('Analysis stored', 'success');
        }
        await loadStoredDates(false);
    } catch (err) {
        if (err.status === 401) toast('Fyers not connected. Connect first.', 'error');
        else toast(err.message || 'Run failed', 'error');
    } finally {
        hideLoading();
        btn.disabled = false;
        btn.textContent = 'Run & Store';
    }
}

function pollJob(jobId, date, category) {
    return new Promise((resolve, reject) => {
        const tick = async () => {
            try {
                const job = await api.getJob(jobId);
                const total = job.total || 0;
                const done = job.progress || 0;
                const pct = total > 0 ? (done / total) * 100 : 12;
                setProgress(pct);
                const t = $('progressText');
                if (t) {
                    const found = job.found != null ? ` · ${job.found} qualified` : '';
                    t.textContent = `Analyzing… (${done}/${total || '?'})${found}`;
                }
                if (job.status === 'finished') {
                    clearTimeout(pollTimer);
                    if (job.result) loadPayload(job.result);
                    else await loadStoredData(date, category, null);
                    toast('Analysis stored', 'success');
                    resolve(job);
                    return;
                }
                if (job.status === 'failed') {
                    clearTimeout(pollTimer);
                    reject(new Error(job.error || 'Job failed'));
                    return;
                }
                pollTimer = setTimeout(tick, 1500);
            } catch (err) {
                clearTimeout(pollTimer);
                reject(err);
            }
        };
        tick();
    });
}

/* ─── Stored data ──────────────────────────────────────────── */
async function loadStoredData(date, category, chipEl) {
    document.querySelectorAll('.date-chip').forEach((c) => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');
    showLoading(`Loading stored analysis for ${date}…`);
    setProgress(40);
    try {
        const minScore = parseInt($('minScore').value, 10);
        const data = await api.getStoredData({ date, category, min_score: minScore });
        setProgress(90);
        loadPayload(data, { stored: true });
    } catch (err) {
        hideSkeleton();
        toast(err.message || 'Failed to load stored data', 'error');
    } finally {
        hideLoading();
    }
}

/* ─── Load a payload into the dashboard ────────────────────── */
function loadPayload(data, { stored = false } = {}) {
    hideSkeleton();
    const qualified = enrichAll(data.qualified || []);
    const unqualified = enrichAll(data.unqualified || []);
    state.allResults = [...qualified, ...unqualified];
    state.lastMeta = { date: data.date, category: data.category, category_label: data.category_label };
    if (data.ai_recommendations) state.ai = data.ai_recommendations;
    else if (!stored) state.ai = {};

    updateStats(data);
    updateAiUi();

    $('statsBar').style.display = 'grid';
    $('filterBar').style.display = 'flex';
    const aiBar = $('aiBar');
    aiBar.classList.add('show');
    aiBar.style.display = 'flex';
    $('fyersConnectPanel').style.display = 'none';
    $('emptyState').style.display = 'none';

    applyFilter(state.filter);
}

function updateStats(data) {
    $('statTotal').textContent = data.total_stocks ?? state.allResults.length;
    const buys = state.allResults.filter((s) => (s.signal || '').toUpperCase() === 'BUY').length;
    const sells = state.allResults.filter((s) => (s.signal || '').toUpperCase() === 'SELL').length;
    const neutrals = state.allResults.filter((s) => (s.signal || '').toUpperCase() === 'NEUTRAL').length;
    $('statSignalBuy').textContent = buys;
    $('statSignalSell').textContent = sells;
    $('statSignalNeutral').textContent = neutrals;
    $('statQualified').textContent = data.qualified_count ?? (data.qualified || []).length;
    $('statBullish').textContent = state.allResults.filter((s) => s.score >= 10).length;
    $('statStrong').textContent = state.allResults.filter((s) => s.score >= 5 && s.score < 10).length;
    const label = data.category_label || CATEGORY_LABELS[data.category] || '';
    const dateEl = $('statDate');
    dateEl.textContent = `${data.date || ''} · ${label}`;
    dateEl.style.fontSize = '13px';
}

function updateAiUi() {
    const recs = Object.values(state.ai);
    const has = recs.length > 0;
    const buy = recs.filter((r) => r.action === 'BUY').length;
    const hold = recs.filter((r) => r.action === 'HOLD').length;
    const avoid = recs.filter((r) => r.action === 'AVOID').length;

    ['statAiBuyCard', 'statAiSellCard', 'statAiNeutralCard'].forEach((id) => {
        const el = $(id);
        if (el) el.style.display = has ? '' : 'none';
    });
    $('statAiBuy').textContent = buy;
    $('statAiSell').textContent = avoid;
    $('statAiNeutral').textContent = hold;

    const setDisp = (id, disp) => { const el = $(id); if (el) el.style.display = disp; };
    setDisp('aiFilterDivider', has ? '' : 'none');
    setDisp('aiFilterBuy', has ? 'inline-flex' : 'none');
    setDisp('aiFilterSell', has ? 'inline-flex' : 'none');
    setDisp('aiFilterNeutral', has ? 'inline-flex' : 'none');

    const aiBtn = $('aiBtn');
    const aiStatus = $('aiStatus');
    if (has) {
        aiBtn.textContent = 'AI Done ✓';
        aiStatus.innerHTML = `<span style="color:var(--green)">${buy} Buy</span> · <span style="color:var(--yellow)">${hold} Neutral</span> · <span style="color:var(--red)">${avoid} Avoid</span>`;
    } else {
        aiBtn.textContent = 'Get AI Buy / Sell / Neutral';
        aiStatus.textContent = '';
    }
}

/* ─── AI recommendations ───────────────────────────────────── */
async function getAiRecommendations() {
    if (!state.allResults.length) { toast('Run or load an analysis first', 'info'); return; }
    const btn = $('aiBtn');
    const aiStatus = $('aiStatus');
    btn.disabled = true;
    btn.textContent = 'Analyzing with AI…';
    aiStatus.textContent = 'Sending stocks to AI — this can take ~20s…';
    try {
        const data = await api.llmAnalyze(state.allResults);
        state.ai = data.recommendations || {};
        updateAiUi();
        if (view) view.refresh();
        toast('AI analysis complete', 'success');
    } catch (err) {
        aiStatus.textContent = '';
        toast(err.message || 'AI analysis failed', 'error');
        btn.textContent = 'Retry AI Analysis';
    } finally {
        btn.disabled = false;
    }
}

/* ─── Filtering / search ───────────────────────────────────── */
function applyFilter(filter) {
    state.filter = filter;
    document.querySelectorAll('.filter-tab').forEach((t) => {
        const active = t.dataset.filter === filter;
        t.classList.toggle('active', active);
        t.setAttribute('aria-pressed', String(active));
    });
    renderResults();
}

function computeRows() {
    const minScore = parseInt($('minScore').value, 10) || 0;
    let rows = state.allResults;
    switch (state.filter) {
        case 'all': break;
        case 'qualified': rows = rows.filter((s) => s.score >= minScore); break;
        case 'signal-buy': rows = rows.filter((s) => (s.signal || '').toUpperCase() === 'BUY'); break;
        case 'signal-sell': rows = rows.filter((s) => (s.signal || '').toUpperCase() === 'SELL'); break;
        case 'signal-neutral': rows = rows.filter((s) => (s.signal || '').toUpperCase() === 'NEUTRAL'); break;
        case 'oversold': rows = rows.filter((s) => s.rsi < 35); break;
        case 'momentum': rows = rows.filter((s) => (s.dist_52w ?? 100) < 5); break;
        case 'golden': rows = rows.filter((s) => (s.reasons || []).some((r) => /golden cross/i.test(r.text || ''))); break;
        case 'ai-buy': rows = rows.filter((s) => getAiRec(s.name)?.action === 'BUY'); break;
        case 'ai-sell': rows = rows.filter((s) => getAiRec(s.name)?.action === 'AVOID'); break;
        case 'ai-neutral': rows = rows.filter((s) => getAiRec(s.name)?.action === 'HOLD'); break;
        default: break;
    }
    if (state.search) {
        const q = state.search.toLowerCase();
        rows = rows.filter((s) =>
            (s.name || '').toLowerCase().includes(q) || (s.symbol || '').toLowerCase().includes(q));
    }
    // Default ordering: signal filters by net score; otherwise bull score.
    const signalSort = ['signal-buy', 'signal-sell', 'signal-neutral'].includes(state.filter);
    return rows.slice().sort((a, b) => {
        if (signalSort) return (Number(b.net_score) || 0) - (Number(a.net_score) || 0);
        return (Number(b.score) || 0) - (Number(a.score) || 0);
    });
}

function renderResults() {
    const rows = computeRows();
    $('emptyState').style.display = 'none';
    if (state.view === 'grid') {
        $('gridViewport').style.display = '';
        $('tableWrap').style.display = 'none';
    } else {
        $('gridViewport').style.display = 'none';
        $('tableWrap').style.display = '';
    }
    view.setItems(rows);
}

/* ─── View toggle ──────────────────────────────────────────── */
function setView(v) {
    state.view = v;
    $('viewGrid').classList.toggle('active', v === 'grid');
    $('viewGrid').setAttribute('aria-pressed', String(v === 'grid'));
    $('viewTable').classList.toggle('active', v === 'table');
    $('viewTable').setAttribute('aria-pressed', String(v === 'table'));
    view.setMode(v);
    renderResults();
}

/* ─── Export ───────────────────────────────────────────────── */
function doExport(format) {
    if (!state.lastMeta.date) { toast('Load or run an analysis first', 'info'); return; }
    const url = api.exportUrl({ date: state.lastMeta.date, category: state.lastMeta.category, format });
    window.open(url, '_blank', 'noopener');
}

/* ─── Alerts (Task 46) ─────────────────────────────────────── */
async function refreshAlerts() {
    const list = $('alertsList');
    if (!list) return;
    list.innerHTML = '<p style="color:var(--text-muted)">Loading…</p>';
    try {
        const data = await api.getAlerts();
        const items = data.alerts || [];
        if (!items.length) {
            list.innerHTML = '<p style="color:var(--text-muted)">No alerts yet. Create one above.</p>';
            return;
        }
        list.innerHTML = '';
        items.forEach((a) => {
            const row = document.createElement('div');
            row.className = 'rule-row';
            const name = a.name || `${a.metric} ${a.operator} ${a.threshold}`;
            row.innerHTML = `<div>
                    <div class="rr-name">${esc(name)}</div>
                    <div class="rr-meta">${esc(a.metric)} ${esc(a.operator)} ${esc(String(a.threshold))} · ${esc(a.channel || 'email')}${a.target ? ' → ' + esc(a.target) : ''}</div>
                </div>
                <button class="wl-remove" type="button" aria-label="Delete alert">Delete</button>`;
            row.querySelector('button').addEventListener('click', async () => {
                try { await api.removeAlert(a.id); row.remove(); toast('Alert deleted', 'info'); refreshAlerts(); }
                catch (err) { toast(err.message || 'Delete failed', 'error'); }
            });
            list.appendChild(row);
        });
    } catch (err) {
        list.innerHTML = `<p style="color:var(--red)">${esc(err.message || 'Failed to load alerts')}</p>`;
    }
}

async function submitAlert(e) {
    e.preventDefault();
    const payload = {
        name: $('alertName').value.trim(),
        metric: $('alertMetric').value,
        operator: $('alertOperator').value,
        threshold: parseFloat($('alertThreshold').value),
        channel: $('alertChannel').value,
        target: $('alertTarget').value.trim(),
    };
    if (!isFinite(payload.threshold)) { toast('Enter a valid threshold', 'error'); return; }
    const btn = $('alertSubmit');
    btn.disabled = true;
    try {
        await api.addAlert(payload);
        toast('Alert created', 'success');
        $('alertForm').reset();
        refreshAlerts();
    } catch (err) {
        toast(err.message || 'Could not create alert', 'error');
    } finally {
        btn.disabled = false;
    }
}

/* ─── Backtest (Task 45) ───────────────────────────────────── */
function openBacktest() {
    const end = $('btEnd');
    const start = $('btStart');
    if (end && !end.value) end.value = new Date().toISOString().slice(0, 10);
    if (start && !start.value) {
        const d = new Date();
        d.setDate(d.getDate() - 180);
        start.value = d.toISOString().slice(0, 10);
    }
    openModal($('backtestModal'), { trigger: $('backtestBtn') });
}

async function submitBacktest(e) {
    e.preventDefault();
    const payload = {
        category: $('btCategory').value,
        start_date: $('btStart').value,
        end_date: $('btEnd').value,
        min_score: parseInt($('btMinScore').value, 10),
        hold_days: parseInt($('btHoldDays').value, 10),
        target_pct: parseFloat($('btTarget').value),
        stop_pct: parseFloat($('btStop').value),
    };
    const btn = $('btRunBtn');
    const out = $('btResults');
    btn.disabled = true;
    btn.textContent = 'Running…';
    out.innerHTML = '<p style="color:var(--text-muted)">Running backtest… this can take a while for large universes.</p>';
    try {
        const r = await api.runBacktest(payload);
        if (r.error) { out.innerHTML = `<p style="color:var(--red)">${esc(r.error)}</p>`; return; }
        renderBacktest(r);
    } catch (err) {
        out.innerHTML = `<p style="color:var(--red)">${esc(err.message || 'Backtest failed')}</p>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Backtest';
    }
}

function fmtPct(v) {
    const n = Number(v);
    if (!isFinite(n)) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
}

function renderBacktest(r) {
    const out = $('btResults');
    out.innerHTML = `
        <div class="bt-stats">
            <div class="bt-stat"><div class="v">${r.trades ?? 0}</div><div class="l">Trades</div></div>
            <div class="bt-stat"><div class="v" style="color:var(--green)">${Number(r.win_rate ?? 0).toFixed(1)}%</div><div class="l">Win rate</div></div>
            <div class="bt-stat"><div class="v">${fmtPct(r.avg_return_pct)}</div><div class="l">Avg return</div></div>
            <div class="bt-stat"><div class="v" style="color:${(r.total_return_pct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtPct(r.total_return_pct)}</div><div class="l">Total return</div></div>
            <div class="bt-stat"><div class="v" style="color:var(--red)">-${Number(r.max_drawdown_pct ?? 0).toFixed(2)}%</div><div class="l">Max drawdown</div></div>
        </div>
        <div class="detail-section-title">Equity curve</div>
        <div class="chart-host" id="equityChart" style="height:260px;"></div>`;
    const el = $('equityChart');
    if (equityChart) { equityChart.destroy(); equityChart = null; }
    const curve = r.equity_curve || [];
    if (curve.length) {
        renderEquityCurve(el, curve)
            .then((c) => { equityChart = c; })
            .catch((err) => { el.innerHTML = `<div class="chart-msg">${esc(err.message || 'Could not render chart')}</div>`; });
    } else {
        el.innerHTML = '<div class="chart-msg">No trades in this period.</div>';
    }
}

/* ─── Init ─────────────────────────────────────────────────── */
function init() {
    initTheme();
    initWatchlistUI();
    initChat(() => state.allResults);
    initDetail({ getAiRec });

    // Rules modal (detail/watchlist modals are wired by their own modules).
    const rulesModal = $('rulesModal');
    if (rulesModal) bindModalDismiss(rulesModal);
    on($('rulesBtn'), 'click', () => openModal(rulesModal, { trigger: $('rulesBtn') }));

    // Alerts + Backtest modals.
    const alertsModal = $('alertsModal');
    if (alertsModal) bindModalDismiss(alertsModal);
    on($('alertsBtn'), 'click', () => { openModal(alertsModal, { trigger: $('alertsBtn') }); refreshAlerts(); });
    on($('alertForm'), 'submit', submitAlert);

    const backtestModal = $('backtestModal');
    if (backtestModal) bindModalDismiss(backtestModal);
    on($('backtestBtn'), 'click', openBacktest);
    on($('backtestForm'), 'submit', submitBacktest);

    view = new StockView({
        gridViewport: $('gridViewport'),
        gridSizer: $('gridSizer'),
        tableWrap: $('tableWrap'),
        getAiRec,
        isStarred,
        onSelect: (stock) => openDetail(stock),
        onToggleStar: (stock) => toggleWatchlist(stock),
    });

    // Re-render stars/cards when the watchlist changes.
    onWatchlistChange(() => { if (state.allResults.length) view.refresh(); });
    // Re-render charts/cards on theme change is handled inside their modules.

    on($('loginForm'), 'submit', handleLogin);
    on($('logoutBtn'), 'click', handleLogout);

    on($('analyzeBtn'), 'click', runAnalysis);
    on($('runStoreBtn'), 'click', runAndStore);
    on($('aiBtn'), 'click', getAiRecommendations);
    on($('modeStored'), 'click', () => switchMode('stored'));
    on($('modeLive'), 'click', () => switchMode('live'));
    on($('exportCsv'), 'click', () => doExport('csv'));
    on($('exportXlsx'), 'click', () => doExport('xlsx'));

    on($('fyersLoginBtn'), 'click', openFyersLogin);
    on($('connectBtn'), 'click', connectFyersNow);
    on($('authUrlInput'), 'input', (e) => {
        if (e.target.value.includes('auth_code=') && !connecting) connectFyersNow();
    });

    document.querySelectorAll('.filter-tab').forEach((t) =>
        on(t, 'click', () => applyFilter(t.dataset.filter)));
    on($('searchBox'), 'input', (e) => { state.search = e.target.value.trim(); renderResults(); });
    on($('viewGrid'), 'click', () => setView('grid'));
    on($('viewTable'), 'click', () => setView('table'));

    switchMode('stored');
    bootstrapAuth();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
