"""Shared pytest fixtures and import-path setup for indicator tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root (which holds indicators.py / strategy.py) is importable
# regardless of where pytest is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def make_ohlcv(closes, *, high_offset=0.5, low_offset=0.5, volume=1000, start="2024-01-01"):
    """Build a daily OHLCV DataFrame from a list of close prices.

    High/Low are derived from close +/- offsets so the frame is internally
    consistent. Indexed by consecutive business-ish daily dates.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + high_offset
    lows = np.minimum(opens, closes) - low_offset
    vols = np.full(n, volume, dtype=float) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    idx = pd.date_range(start=start, periods=n, freq="D")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


@pytest.fixture
def bullish_frame():
    """A strongly, steadily rising series → should score clearly bullish."""
    closes = list(np.linspace(100, 200, 260))
    # Last bar closes up on a volume spike to trigger volume/OBV bull rules.
    vols = np.full(len(closes), 1000.0)
    vols[-1] = 3000.0
    return make_ohlcv(closes, volume=vols)


@pytest.fixture
def bearish_frame():
    """A strongly, steadily falling series → should score clearly bearish."""
    closes = list(np.linspace(200, 100, 260))
    vols = np.full(len(closes), 1000.0)
    vols[-1] = 3000.0
    return make_ohlcv(closes, volume=vols)
