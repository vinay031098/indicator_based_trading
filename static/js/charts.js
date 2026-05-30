/* ============================================================
   charts.js — TradingView Lightweight Charts integration.

   Exposes createStockChart(): a controller around a candlestick
   chart with toggleable SMA/EMA/Bollinger overlays, a separate
   oscillator subpanel (RSI or MACD), and target/stop price lines.
   The UMD library is loaded from the CDN in index.html; if it is
   not yet present, ensureChartsLib() lazy-loads it.
   ============================================================ */

const LIB_URL =
    'https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';
let libPromise = null;

/** Lazy-load the Lightweight Charts UMD bundle once (fallback if the
 *  synchronous <script> in index.html has not executed yet). */
export function ensureChartsLib() {
    if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts);
    if (libPromise) return libPromise;
    libPromise = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = LIB_URL;
        s.async = true;
        s.onload = () =>
            window.LightweightCharts
                ? resolve(window.LightweightCharts)
                : reject(new Error('Lightweight Charts failed to initialise'));
        s.onerror = () => reject(new Error('Could not load charting library from CDN'));
        document.head.appendChild(s);
    });
    return libPromise;
}

/* ─── Theme-aware colors (read CSS variables) ──────────────── */
function themeColors() {
    const css = getComputedStyle(document.documentElement);
    const v = (name, fallback) => (css.getPropertyValue(name) || fallback).trim();
    return {
        text: v('--text-muted', '#9aa0b4'),
        grid: v('--chart-grid', 'rgba(255,255,255,0.06)'),
        green: v('--green', '#34d27b'),
        red: v('--red', '#f76d6d'),
        blue: v('--blue', '#5b9dff'),
        accent: v('--accent', '#8b85ff'),
        yellow: v('--yellow', '#f6b545'),
    };
}

function baseLayout(LC, colors) {
    return {
        layout: {
            background: { type: LC.ColorType.Solid, color: 'transparent' },
            textColor: colors.text,
            fontSize: 11,
        },
        grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
        rightPriceScale: { borderColor: colors.grid },
        timeScale: { borderColor: colors.grid, timeVisible: false },
        crosshair: { mode: LC.CrosshairMode ? LC.CrosshairMode.Normal : 0 },
        handleScale: true,
        handleScroll: true,
    };
}

function dashed(LC) {
    return LC.LineStyle ? LC.LineStyle.Dashed : 2;
}

/* ─── Indicator math (client-side overlays) ────────────────── */
function sma(values, period) {
    const out = new Array(values.length).fill(null);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += values[i];
        if (i >= period) sum -= values[i - period];
        if (i >= period - 1) out[i] = sum / period;
    }
    return out;
}

function ema(values, period) {
    const out = new Array(values.length).fill(null);
    const k = 2 / (period + 1);
    let prev = null;
    for (let i = 0; i < values.length; i++) {
        const val = values[i];
        if (prev === null) {
            // Seed with an SMA once enough points exist.
            if (i >= period - 1) {
                let s = 0;
                for (let j = i - period + 1; j <= i; j++) s += values[j];
                prev = s / period;
                out[i] = prev;
            }
        } else {
            prev = val * k + prev * (1 - k);
            out[i] = prev;
        }
    }
    return out;
}

function bollinger(values, period = 20, mult = 2) {
    const mid = sma(values, period);
    const upper = new Array(values.length).fill(null);
    const lower = new Array(values.length).fill(null);
    for (let i = period - 1; i < values.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += (values[j] - mid[i]) ** 2;
        const sd = Math.sqrt(sum / period);
        upper[i] = mid[i] + mult * sd;
        lower[i] = mid[i] - mult * sd;
    }
    return { mid, upper, lower };
}

function rsi(values, period = 14) {
    const out = new Array(values.length).fill(null);
    if (values.length <= period) return out;
    let gain = 0;
    let loss = 0;
    for (let i = 1; i <= period; i++) {
        const d = values[i] - values[i - 1];
        if (d >= 0) gain += d; else loss -= d;
    }
    let avgGain = gain / period;
    let avgLoss = loss / period;
    out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    for (let i = period + 1; i < values.length; i++) {
        const d = values[i] - values[i - 1];
        avgGain = (avgGain * (period - 1) + (d > 0 ? d : 0)) / period;
        avgLoss = (avgLoss * (period - 1) + (d < 0 ? -d : 0)) / period;
        out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    }
    return out;
}

function macd(values, fast = 12, slow = 26, signalP = 9) {
    const emaFast = ema(values, fast);
    const emaSlow = ema(values, slow);
    const macdLine = values.map((_, i) =>
        emaFast[i] !== null && emaSlow[i] !== null ? emaFast[i] - emaSlow[i] : null);
    // Signal = EMA of the defined macd values.
    const defined = macdLine.map((v) => (v === null ? 0 : v));
    const sigRaw = ema(defined, signalP);
    const signal = macdLine.map((v, i) => (v === null ? null : sigRaw[i]));
    const hist = macdLine.map((v, i) =>
        v !== null && signal[i] !== null ? v - signal[i] : null);
    return { macdLine, signal, hist };
}

function lineData(times, series) {
    const out = [];
    for (let i = 0; i < series.length; i++) {
        const v = series[i];
        if (v !== null && v !== undefined && isFinite(v)) {
            out.push({ time: times[i], value: Number(v.toFixed(4)) });
        }
    }
    return out;
}

/**
 * Build an interactive stock chart controller.
 * @param {HTMLElement} host  candlestick container
 * @param {HTMLElement} sub   oscillator subpanel container (optional)
 * @param {Array} candles     [{ time, open, high, low, close, volume? }]
 * @returns controller { destroy, resize, setOverlayVisible, renderOscillator, setPriceLines, hasData }
 */
export function createStockChart(host, sub, candles) {
    const LC = window.LightweightCharts;
    if (!LC) throw new Error('Charting library not loaded');
    const colors = themeColors();

    const rows = (candles || [])
        .filter((c) => c && c.time != null)
        .map((c) => ({
            time: Math.floor(Number(c.time)),
            open: +c.open, high: +c.high, low: +c.low, close: +c.close,
        }))
        .filter((c) => isFinite(c.open) && isFinite(c.close))
        .sort((a, b) => a.time - b.time);

    // De-duplicate timestamps (Lightweight Charts requires strictly ascending).
    const deduped = [];
    let lastT = null;
    for (const r of rows) {
        if (r.time === lastT) deduped[deduped.length - 1] = r;
        else { deduped.push(r); lastT = r.time; }
    }

    const times = deduped.map((r) => r.time);
    const closes = deduped.map((r) => r.close);

    const priceChart = LC.createChart(host, {
        ...baseLayout(LC, colors),
        width: host.clientWidth,
        height: host.clientHeight || 320,
    });
    const candleSeries = priceChart.addCandlestickSeries({
        upColor: colors.green, downColor: colors.red,
        borderUpColor: colors.green, borderDownColor: colors.red,
        wickUpColor: colors.green, wickDownColor: colors.red,
    });
    candleSeries.setData(deduped);

    // ── Overlays, grouped so they can be toggled together ──
    const overlays = { sma: [], ema: [], bollinger: [] };

    const sma20 = priceChart.addLineSeries({ color: colors.blue, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
    sma20.setData(lineData(times, sma(closes, 20)));
    const sma50 = priceChart.addLineSeries({ color: colors.yellow, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
    sma50.setData(lineData(times, sma(closes, 50)));
    overlays.sma.push(sma20, sma50);

    const ema21 = priceChart.addLineSeries({ color: colors.accent, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
    ema21.setData(lineData(times, ema(closes, 21)));
    overlays.ema.push(ema21);

    const bb = bollinger(closes, 20, 2);
    const bbUpper = priceChart.addLineSeries({ color: colors.grid, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    bbUpper.setData(lineData(times, bb.upper));
    const bbLower = priceChart.addLineSeries({ color: colors.grid, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    bbLower.setData(lineData(times, bb.lower));
    overlays.bollinger.push(bbUpper, bbLower);

    priceChart.timeScale().fitContent();

    // ── Oscillator subpanel ──
    let subChart = null;
    let oscSeries = [];
    if (sub) {
        subChart = LC.createChart(sub, {
            ...baseLayout(LC, colors),
            width: sub.clientWidth,
            height: sub.clientHeight || 130,
        });
        // Keep both time scales in sync.
        let syncing = false;
        const link = (from, to) => from.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (syncing || !range) return;
            syncing = true;
            try { to.timeScale().setVisibleLogicalRange(range); } finally { syncing = false; }
        });
        link(priceChart, subChart);
        link(subChart, priceChart);
    }

    function clearOscillator() {
        if (!subChart) return;
        oscSeries.forEach((s) => { try { subChart.removeSeries(s); } catch (_e) { /* noop */ } });
        oscSeries = [];
    }

    function renderOscillator(name) {
        if (!subChart) return;
        clearOscillator();
        if (name === 'macd') {
            const m = macd(closes);
            const hist = subChart.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
            hist.setData(times.map((t, i) => (
                m.hist[i] != null && isFinite(m.hist[i])
                    ? { time: t, value: Number(m.hist[i].toFixed(4)), color: m.hist[i] >= 0 ? colors.green : colors.red }
                    : null
            )).filter(Boolean));
            const line = subChart.addLineSeries({ color: colors.blue, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
            line.setData(lineData(times, m.macdLine));
            const sig = subChart.addLineSeries({ color: colors.yellow, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
            sig.setData(lineData(times, m.signal));
            oscSeries = [hist, line, sig];
        } else {
            const r = subChart.addLineSeries({ color: colors.accent, lineWidth: 2, priceLineVisible: false });
            r.setData(lineData(times, rsi(closes, 14)));
            r.createPriceLine({ price: 70, color: colors.red, lineWidth: 1, lineStyle: dashed(LC), axisLabelVisible: true, title: '70' });
            r.createPriceLine({ price: 30, color: colors.green, lineWidth: 1, lineStyle: dashed(LC), axisLabelVisible: true, title: '30' });
            oscSeries = [r];
        }
        subChart.timeScale().fitContent();
    }
    // Default oscillator.
    renderOscillator('rsi');

    // ── Price lines (target / stop-loss) ──
    let priceLines = [];
    function setPriceLines(lines) {
        priceLines.forEach((pl) => { try { candleSeries.removePriceLine(pl); } catch (_e) { /* noop */ } });
        priceLines = (lines || []).map((l) => candleSeries.createPriceLine({
            price: Number(l.price),
            color: l.color || colors.accent,
            lineWidth: 1,
            lineStyle: dashed(LC),
            axisLabelVisible: true,
            title: l.title || '',
        }));
    }

    function setOverlayVisible(group, visible) {
        (overlays[group] || []).forEach((s) => s.applyOptions({ visible }));
    }

    const resize = () => {
        priceChart.applyOptions({ width: host.clientWidth, height: host.clientHeight || 320 });
        if (subChart && sub) subChart.applyOptions({ width: sub.clientWidth, height: sub.clientHeight || 130 });
    };
    window.addEventListener('resize', resize);

    return {
        hasData: deduped.length > 0,
        setOverlayVisible,
        renderOscillator,
        setPriceLines,
        resize,
        destroy() {
            window.removeEventListener('resize', resize);
            try { priceChart.remove(); } catch (_e) { /* noop */ }
            try { subChart && subChart.remove(); } catch (_e) { /* noop */ }
        },
    };
}

/** Area chart for an equity curve (backtest panel, when present). */
export async function renderEquityCurve(el, equityCurve) {
    const LC = await ensureChartsLib();
    const colors = themeColors();
    const chart = LC.createChart(el, {
        ...baseLayout(LC, colors),
        width: el.clientWidth,
        height: el.clientHeight || 260,
    });
    const series = chart.addAreaSeries({
        lineColor: colors.accent, topColor: colors.accent + '55', bottomColor: 'transparent', lineWidth: 2,
    });
    const data = (equityCurve || []).map((p, i) => {
        if (typeof p === 'number') return { time: i + 1, value: p };
        const t = p.time || p.date || p.t || (i + 1);
        const time = typeof t === 'string' ? t : Math.floor(Number(t));
        return { time, value: Number(p.value ?? p.equity ?? p.v ?? 0) };
    });
    series.setData(data);
    chart.timeScale().fitContent();
    const resize = () => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight || 260 });
    window.addEventListener('resize', resize);
    return {
        destroy() { window.removeEventListener('resize', resize); try { chart.remove(); } catch (_e) { /* noop */ } },
        resize,
    };
}
