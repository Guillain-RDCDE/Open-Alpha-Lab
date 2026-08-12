"""Data-layer tests. The synthetic generator is exercised offline; the real-tape checks are
skipped whenever the git-ignored _cache/ is absent (as on CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pref_reset import data, strategy as st  # noqa: E402

CACHE = data.PRICES_CACHE


# --------------------------------------------------------------------------- #
# Offline synthetic-only
# --------------------------------------------------------------------------- #
def test_synthetic_fingerprint_stable():
    w = data.synthetic_world(edge=0.0030, seed=909)
    fp1 = data.fingerprint(w[["variable", "fixed", "cash"]])
    fp2 = data.fingerprint(data.synthetic_world(edge=0.0030, seed=909)[["variable", "fixed", "cash"]])
    assert fp1 == fp2 and len(fp1) == 12


def test_edge_knob_monotone():
    lo = st.synthetic_detect(data.synthetic_world(edge=0.0010, seed=909))["spread_ann_pct"]
    hi = st.synthetic_detect(data.synthetic_world(edge=0.0060, seed=909))["spread_ann_pct"]
    assert hi > lo


def test_tickers_partition():
    assert set(data.VARIABLE).isdisjoint(data.FIXED)
    assert data.CASH not in data.VARIABLE + data.FIXED
    assert set(data.TICKERS) == set(data.VARIABLE + data.FIXED + [data.CASH])


# --------------------------------------------------------------------------- #
# Real-tape — only when the cache is present locally
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent (CI) — offline test")
def test_real_cache_loads_and_stamps():
    prices = data.load_prices()
    assert set(data.TICKERS).issubset(prices.columns)
    assert prices.index.max() <= pd.Timestamp(data.AS_OF)
    monthly = data.monthly_returns(prices)
    sleeves = data.sleeve_returns(monthly)
    assert {"variable", "fixed", "cash", "VRP", "PFF"}.issubset(sleeves.columns)


@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent (CI) — offline test")
def test_real_high_rate_edge_positive():
    prices = data.load_prices()
    sleeves = data.sleeve_returns(data.monthly_returns(prices))
    rf = st.race_frame(sleeves, var_col="VRP", fix_col="PFF")
    era = st.era_cut(rf, data.HIGH_RATE_SPLIT)
    # the reset premium shows up in the high-rate regime and not the low-rate one
    assert era["high_rate"]["spread_ann_pct"] > era["low_rate"]["spread_ann_pct"]
    assert np.isfinite(era["high_rate"]["t_nw"])
