// indicators.js — client-side technical indicator math for chart overlays.
// Inputs are arrays of normalized candles: {time, open, high, low, close, volume}.
// Outputs are arrays of {time, value} aligned to candle time, suitable for
// Lightweight Charts line/histogram series (gaps omitted via whitespace-free arrays).

export function sma(candles, period) {
    const out = [];
    let sum = 0;
    for (let i = 0; i < candles.length; i++) {
        sum += candles[i].close;
        if (i >= period) sum -= candles[i - period].close;
        if (i >= period - 1) out.push({ time: candles[i].time, value: sum / period });
    }
    return out;
}

export function ema(candles, period) {
    const out = [];
    if (candles.length < period) return out;
    const k = 2 / (period + 1);
    // Seed with SMA of first `period` closes.
    let seed = 0;
    for (let i = 0; i < period; i++) seed += candles[i].close;
    let prev = seed / period;
    out.push({ time: candles[period - 1].time, value: prev });
    for (let i = period; i < candles.length; i++) {
        prev = candles[i].close * k + prev * (1 - k);
        out.push({ time: candles[i].time, value: prev });
    }
    return out;
}

export function bollinger(candles, period = 20, mult = 2) {
    const upper = [];
    const middle = [];
    const lower = [];
    for (let i = period - 1; i < candles.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += candles[j].close;
        const mean = sum / period;
        let variance = 0;
        for (let j = i - period + 1; j <= i; j++) {
            const d = candles[j].close - mean;
            variance += d * d;
        }
        const sd = Math.sqrt(variance / period);
        const t = candles[i].time;
        middle.push({ time: t, value: mean });
        upper.push({ time: t, value: mean + mult * sd });
        lower.push({ time: t, value: mean - mult * sd });
    }
    return { upper, middle, lower };
}

// RSI with Wilder smoothing.
export function rsi(candles, period = 14) {
    const out = [];
    if (candles.length <= period) return out;
    let gain = 0;
    let loss = 0;
    for (let i = 1; i <= period; i++) {
        const diff = candles[i].close - candles[i - 1].close;
        if (diff >= 0) gain += diff; else loss -= diff;
    }
    let avgGain = gain / period;
    let avgLoss = loss / period;
    const rsiVal = (ag, al) => (al === 0 ? 100 : 100 - 100 / (1 + ag / al));
    out.push({ time: candles[period].time, value: rsiVal(avgGain, avgLoss) });
    for (let i = period + 1; i < candles.length; i++) {
        const diff = candles[i].close - candles[i - 1].close;
        const g = diff > 0 ? diff : 0;
        const l = diff < 0 ? -diff : 0;
        avgGain = (avgGain * (period - 1) + g) / period;
        avgLoss = (avgLoss * (period - 1) + l) / period;
        out.push({ time: candles[i].time, value: rsiVal(avgGain, avgLoss) });
    }
    return out;
}

// MACD line, signal line and histogram.
export function macd(candles, fast = 12, slow = 26, signalPeriod = 9) {
    const emaFast = ema(candles, fast);
    const emaSlow = ema(candles, slow);
    // Index ema series by time for alignment.
    const fastMap = new Map(emaFast.map((p) => [p.time, p.value]));
    const macdLine = [];
    for (const p of emaSlow) {
        const f = fastMap.get(p.time);
        if (f !== undefined) macdLine.push({ time: p.time, value: f - p.value });
    }
    // Signal = EMA of MACD line.
    const signal = emaOfSeries(macdLine, signalPeriod);
    const sigMap = new Map(signal.map((p) => [p.time, p.value]));
    const hist = [];
    for (const p of macdLine) {
        const s = sigMap.get(p.time);
        if (s !== undefined) hist.push({ time: p.time, value: p.value - s });
    }
    return { macdLine, signal, hist };
}

function emaOfSeries(series, period) {
    const out = [];
    if (series.length < period) return out;
    const k = 2 / (period + 1);
    let seed = 0;
    for (let i = 0; i < period; i++) seed += series[i].value;
    let prev = seed / period;
    out.push({ time: series[period - 1].time, value: prev });
    for (let i = period; i < series.length; i++) {
        prev = series[i].value * k + prev * (1 - k);
        out.push({ time: series[i].time, value: prev });
    }
    return out;
}
