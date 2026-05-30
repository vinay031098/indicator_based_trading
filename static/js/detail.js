// detail.js — stock detail modal: full indicator table, interactive
// candlestick chart with overlays + oscillator, AI recommendation, and
// target/stop-loss markers on the chart. Includes a watchlist star.

import { getHistory } from './api.js';
import { notify } from './toast.js';
import { openModal, closeModal, bindModalDismiss } from './modal.js';
import { inr, num, signed, pct, aiActionLabel, signalLabel, esc, deriveSignal, netScore } from './format.js';
import { createStockChart } from './charts.js';
import { isStarred, toggleWatchlist } from './watchlist.js';

let chartCtrl = null;
let getAiRecFn = () => null;
let currentStock = null;

const INDICATOR_FIELDS = [
    ['rsi', 'RSI', 1], ['macd', 'MACD', 3], ['macd_signal', 'MACD Signal', 3], ['macd_hist', 'MACD Hist', 3],
    ['sma20', 'SMA 20', 2], ['sma50', 'SMA 50', 2], ['sma200', 'SMA 200', 2],
    ['ema9', 'EMA 9', 2], ['ema21', 'EMA 21', 2],
    ['stoch_k', 'Stoch %K', 1], ['stoch_d', 'Stoch %D', 1],
    ['bb_upper', 'BB Upper', 2], ['bb_lower', 'BB Lower', 2], ['bb_width', 'BB Width', 2],
    ['adx', 'ADX', 1], ['plus_di', 'DI+', 1], ['minus_di', 'DI-', 1],
    ['cci', 'CCI', 1], ['williams_r', 'Williams %R', 1], ['mfi', 'MFI', 1],
    ['atr', 'ATR', 2], ['atr_pct', 'ATR %', 2], ['roc', 'ROC %', 2],
    ['cmf', 'CMF', 3], ['vwap', 'VWAP', 2],
    ['ichimoku_tenkan', 'Tenkan', 2], ['ichimoku_kijun', 'Kijun', 2],
    ['pivot', 'Pivot', 2], ['pivot_s1', 'Pivot S1', 2], ['pivot_r1', 'Pivot R1', 2],
    ['vol_ratio', 'Vol Ratio', 2], ['w52_high', '52W High', 2], ['w52_low', '52W Low', 2],
    ['dist_52w', 'Dist 52W %', 2],
];

export function initDetail(cfg = {}) {
    getAiRecFn = cfg.getAiRec || (() => null);
    const overlay = document.getElementById('detailModal');
    if (overlay) {
        bindModalDismiss(overlay);
        overlay.addEventListener('detail-closed', destroyChart);
        // Tear down the chart when the modal is dismissed.
        const observer = new MutationObserver(() => {
            if (!overlay.classList.contains('show')) destroyChart();
        });
        observer.observe(overlay, { attributes: true, attributeFilter: ['class'] });
    }
}

function destroyChart() {
    if (chartCtrl) { chartCtrl.destroy(); chartCtrl = null; }
}

export async function openDetail(stock) {
    currentStock = stock;
    const overlay = document.getElementById('detailModal');
    const titleEl = document.getElementById('detailTitle');
    const body = document.getElementById('detailBody');
    if (!overlay || !body) return;

    const ai = getAiRecFn(stock.name);
    titleEl.textContent = stock.name;
    body.innerHTML = buildBody(stock, ai);
    openModal(overlay, { trigger: document.activeElement });

    wireBody(stock, ai);
    loadChart(stock, ai);
}

function buildBody(s, ai) {
    const sig = signalLabel(deriveSignal(s));
    const net = netScore(s);
    const changeUp = Number(s.change) >= 0;
    const starred = isStarred(s.symbol);

    const indicatorCells = INDICATOR_FIELDS.map(([key, label, dg]) => {
        const val = s[key];
        const display = (val === null || val === undefined) ? '-' : num(val, dg);
        return `<div class="detail-ind"><div class="di-label">${esc(label)}</div><div class="di-value">${esc(display)}</div></div>`;
    }).join('');

    let aiHtml = '';
    if (ai) {
        const a = aiActionLabel(ai.action);
        aiHtml = `
        <div class="detail-section-title">AI Recommendation</div>
        <div class="detail-ai ${a.cls}">
            <div class="detail-ai-head">
                <span class="detail-ai-action ${a.cls}">${esc(a.text)}</span>
                <span style="color:var(--text-muted)">Confidence: <b>${esc(ai.confidence || 'n/a')}</b></span>
            </div>
            <div>${esc(ai.reason || '')}</div>
            <div class="detail-ai-meta">
                ${ai.target ? `<span>Target: <b>${esc(inr(ai.target))}</b></span>` : ''}
                ${ai.stoploss ? `<span>Stop-loss: <b>${esc(inr(ai.stoploss))}</b></span>` : ''}
                ${ai.risk_reward ? `<span>Risk/Reward: <b>${esc(num(ai.risk_reward, 2))}</b></span>` : ''}
            </div>
        </div>`;
    }

    return `
    <div class="detail-top">
        <div>
            <div class="detail-price">${esc(inr(s.price))}
                <span class="change ${changeUp ? 'up' : 'down'}" style="font-size:14px">${esc(signed(s.change))} (${esc(pct(s.change_pct))})</span>
            </div>
            <div class="detail-sub">NSE &middot; ${esc(s.symbol)} &middot; Net ${esc(signed(net, 0))} &middot; Bull ${esc(num(s.score, 0))}/30
                <span class="signal-pill ${sig.cls}" style="margin-left:8px"><span class="pill-shape" aria-hidden="true">${sig.shape}</span>${esc(sig.text)}</span>
            </div>
        </div>
        <button class="star-btn ${starred ? 'starred' : ''}" id="detailStar"
            aria-pressed="${starred}" aria-label="${starred ? 'Remove from' : 'Add to'} watchlist"
            style="width:auto; padding:8px 14px">${starred ? '\u2605 Watching' : '\u2606 Watchlist'}</button>
    </div>

    <div class="detail-section-title">Price Chart</div>
    <div class="chart-controls" role="group" aria-label="Chart overlays">
        <button class="chart-toggle active" data-overlay="sma" aria-pressed="true">SMA 20/50</button>
        <button class="chart-toggle active" data-overlay="ema" aria-pressed="true">EMA 21</button>
        <button class="chart-toggle active" data-overlay="bollinger" aria-pressed="true">Bollinger</button>
        <span style="width:1px;background:var(--border);margin:0 4px"></span>
        <button class="chart-toggle active" data-osc="rsi" aria-pressed="true">RSI</button>
        <button class="chart-toggle" data-osc="macd" aria-pressed="false">MACD</button>
    </div>
    <div class="chart-host" id="detailChart"><div class="chart-msg">Loading chart\u2026</div></div>
    <div class="subchart-host" id="detailSubChart"></div>

    ${aiHtml}

    <div class="detail-section-title">All Indicators</div>
    <div class="detail-indicator-grid">${indicatorCells}</div>`;
}

function wireBody(stock, ai) {
    const star = document.getElementById('detailStar');
    if (star) {
        star.addEventListener('click', async () => {
            await toggleWatchlist(stock);
            const now = isStarred(stock.symbol);
            star.classList.toggle('starred', now);
            star.setAttribute('aria-pressed', String(now));
            star.textContent = now ? '\u2605 Watching' : '\u2606 Watchlist';
        });
    }

    document.querySelectorAll('#detailBody [data-overlay]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const group = btn.getAttribute('data-overlay');
            const active = btn.classList.toggle('active');
            btn.setAttribute('aria-pressed', String(active));
            if (chartCtrl) chartCtrl.setOverlayVisible(group, active);
        });
    });

    const oscBtns = document.querySelectorAll('#detailBody [data-osc]');
    oscBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            oscBtns.forEach((b) => { b.classList.remove('active'); b.setAttribute('aria-pressed', 'false'); });
            btn.classList.add('active');
            btn.setAttribute('aria-pressed', 'true');
            if (chartCtrl) chartCtrl.renderOscillator(btn.getAttribute('data-osc'));
        });
    });
}

async function loadChart(stock, ai) {
    const host = document.getElementById('detailChart');
    const sub = document.getElementById('detailSubChart');
    if (!host) return;
    destroyChart();
    try {
        const data = await getHistory(stock.symbol, { resolution: 'D', days: 365 });
        const candles = data.candles || [];
        if (candles.length === 0) {
            host.innerHTML = '<div class="chart-msg">No historical data available for this symbol.</div>';
            return;
        }
        host.innerHTML = '';
        chartCtrl = createStockChart(host, sub, candles);

        // Target / stop-loss markers from the AI recommendation.
        const lines = [];
        if (ai && ai.target) lines.push({ price: Number(ai.target), color: 'var(--green)', title: 'Target' });
        if (ai && ai.stoploss) lines.push({ price: Number(ai.stoploss), color: 'var(--red)', title: 'Stop-loss' });
        if (lines.length) chartCtrl.setPriceLines(lines);
    } catch (err) {
        host.innerHTML = `<div class="chart-msg">Could not load chart: ${esc(err.message || 'error')}</div>`;
        if (err.status === 401) notify.warning('Connect to Fyers (Live mode) to load price history.');
    }
}

export function closeDetail() {
    const overlay = document.getElementById('detailModal');
    if (overlay) closeModal(overlay);
    destroyChart();
}
