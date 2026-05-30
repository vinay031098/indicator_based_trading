"""
Strategy configuration for the indicator engine.

This module holds *all* tunable parameters used by ``indicators.py``:

* indicator look-back periods (RSI 14, MACD 12/26/9, Bollinger 20/2, ...)
* the point weights used by the bullish / bearish scoring engine
* the thresholds that turn a net score into a BUY / NEUTRAL / SELL signal

It is intentionally dependency-free: it does NOT read ``config.py`` or the
environment. The defaults live here as plain Python dataclasses so the values
can be imported, introspected and overridden in tests without any I/O.

``indicators.py`` imports the module-level :data:`STRATEGY` singleton. Tests
may build their own :class:`Strategy` instance and pass it explicitly to keep
results deterministic and decoupled from these defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


# ─── Indicator look-back periods ─────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorPeriods:
    """Look-back windows / parameters for every indicator."""

    rsi: int = 14
    atr: int = 14
    adx: int = 14

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Stochastic oscillator: %K window, %K smoothing, %D (SMA of %K) window.
    stoch_k: int = 14
    stoch_smooth_k: int = 1
    stoch_d: int = 3

    bb_period: int = 20
    bb_std: float = 2.0

    sma_short: int = 20
    sma_mid: int = 50
    sma_long: int = 200

    ema_fast: int = 9
    ema_slow: int = 21

    cci: int = 20
    williams: int = 14
    mfi: int = 14
    roc: int = 12
    cmf: int = 20

    obv_slope: int = 5
    volume_avg: int = 20

    w52: int = 252  # ~one trading year

    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52

    def validate(self) -> None:
        # Every window must be a positive integer.
        int_fields = (
            "rsi", "atr", "adx", "macd_fast", "macd_slow", "macd_signal",
            "stoch_k", "stoch_smooth_k", "stoch_d", "bb_period",
            "sma_short", "sma_mid", "sma_long", "ema_fast", "ema_slow",
            "cci", "williams", "mfi", "roc", "cmf", "obv_slope",
            "volume_avg", "w52", "ichimoku_tenkan", "ichimoku_kijun",
            "ichimoku_senkou_b",
        )
        for name in int_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"IndicatorPeriods.{name} must be a positive int, got {value!r}")

        if self.bb_std <= 0:
            raise ValueError(f"IndicatorPeriods.bb_std must be > 0, got {self.bb_std!r}")

        if self.macd_fast >= self.macd_slow:
            raise ValueError(
                f"macd_fast ({self.macd_fast}) must be < macd_slow ({self.macd_slow})"
            )

        if not (self.sma_short < self.sma_mid < self.sma_long):
            raise ValueError(
                "SMA windows must satisfy sma_short < sma_mid < sma_long, got "
                f"{self.sma_short}/{self.sma_mid}/{self.sma_long}"
            )

        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"ema_fast ({self.ema_fast}) must be < ema_slow ({self.ema_slow})"
            )


# ─── Score weights ───────────────────────────────────────────────────────────

def _default_bull_weights() -> Dict[str, int]:
    """Points awarded to each bullish condition (kept identical to the
    original engine for backward-compatible ``score`` values)."""
    return {
        "rsi_oversold": 2,
        "rsi_low": 1,
        "macd_bull_cross": 1,
        "macd_hist_rising": 1,
        "below_sma20_dip": 1,
        "stoch_bull_cross": 2,
        "stoch_oversold": 1,
        "near_52w_high": 2,
        "close_52w_high": 1,
        "golden_cross": 2,
        "below_bb_lower": 1,
        "ema_bull_cross": 2,
        "ema_fast_above_slow": 1,
        "adx_strong_up": 2,
        "adx_moderate_up": 1,
        "cci_oversold": 1,
        "williams_oversold": 1,
        "mfi_oversold": 2,
        "mfi_low": 1,
        "obv_rising": 1,
        "below_vwap": 1,
        "roc_strong": 1,
        "cmf_positive": 1,
        "above_cloud": 1,
        "ichimoku_tk_bull": 1,
        "above_r1": 1,
        "bb_squeeze": 1,
        "near_52w_low": 1,
        "volume_spike_up": 1,
        "green_candles": 1,
        "above_sma200": 1,
    }


def _default_bear_weights() -> Dict[str, int]:
    """Points awarded to each bearish condition (symmetric to the bull side)."""
    return {
        "rsi_overbought": 2,
        "rsi_high": 1,
        "macd_bear_cross": 1,
        "macd_hist_falling": 1,
        "stoch_bear_cross": 2,
        "stoch_overbought": 1,
        "near_52w_low_risk": 2,
        "death_cross": 2,
        "above_bb_upper": 1,
        "ema_bear_cross": 2,
        "ema_fast_below_slow": 1,
        "adx_strong_down": 2,
        "adx_moderate_down": 1,
        "cci_overbought": 1,
        "williams_overbought": 1,
        "mfi_overbought": 2,
        "mfi_high": 1,
        "obv_falling": 1,
        "above_vwap": 1,
        "roc_weak": 1,
        "cmf_negative": 1,
        "below_cloud": 1,
        "ichimoku_tk_bear": 1,
        "below_s1": 1,
        "red_candles": 1,
        "below_sma200": 1,
    }


@dataclass(frozen=True)
class ScoreWeights:
    bull: Dict[str, int] = field(default_factory=_default_bull_weights)
    bear: Dict[str, int] = field(default_factory=_default_bear_weights)

    def validate(self) -> None:
        for side, weights in (("bull", self.bull), ("bear", self.bear)):
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"ScoreWeights.{side} must be a non-empty dict")
            for key, value in weights.items():
                if not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"ScoreWeights.{side}[{key!r}] must be a non-negative int, got {value!r}"
                    )


# ─── Signal thresholds ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalThresholds:
    """Net score (= bull - bear) cut-offs for the categorical signal.

    ``net >= buy`` -> "BUY", ``net <= sell`` -> "SELL", otherwise "NEUTRAL".
    """

    buy: int = 3
    sell: int = -3

    def validate(self) -> None:
        if not isinstance(self.buy, int) or not isinstance(self.sell, int):
            raise ValueError("SignalThresholds.buy/sell must be ints")
        if self.sell >= self.buy:
            raise ValueError(
                f"SignalThresholds.sell ({self.sell}) must be < buy ({self.buy})"
            )


# ─── Indicator decision thresholds (oversold/overbought levels etc.) ─────────

@dataclass(frozen=True)
class Levels:
    """Threshold constants used by the scoring rules."""

    rsi_oversold: float = 30.0
    rsi_low: float = 40.0
    rsi_high: float = 60.0
    rsi_overbought: float = 70.0

    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    near_52w_high_pct: float = 5.0
    close_52w_high_pct: float = 10.0
    near_52w_low_pct: float = 10.0

    adx_strong: float = 25.0
    adx_moderate: float = 20.0

    cci_oversold: float = -100.0
    cci_overbought: float = 100.0

    williams_oversold: float = -80.0
    williams_overbought: float = -20.0

    mfi_oversold: float = 20.0
    mfi_low: float = 40.0
    mfi_high: float = 60.0
    mfi_overbought: float = 80.0

    roc_strong: float = 5.0
    roc_weak: float = -10.0

    cmf_positive: float = 0.1
    cmf_negative: float = -0.1

    bb_squeeze_pct: float = 5.0
    vol_spike_ratio: float = 1.5

    dist_sma200_pct: float = 5.0
    green_candles_min: int = 3
    red_candles_min: int = 3


@dataclass(frozen=True)
class Strategy:
    periods: IndicatorPeriods = field(default_factory=IndicatorPeriods)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    thresholds: SignalThresholds = field(default_factory=SignalThresholds)
    levels: Levels = field(default_factory=Levels)

    def validate(self) -> "Strategy":
        self.periods.validate()
        self.weights.validate()
        self.thresholds.validate()
        return self

    def with_overrides(self, **kwargs) -> "Strategy":
        """Return a copy with top-level sections replaced (handy in tests)."""
        return replace(self, **kwargs).validate()


# Module-level singleton consumed by indicators.py. Validated at import time so
# a misconfigured strategy fails fast rather than silently scoring wrong.
STRATEGY: Strategy = Strategy().validate()
