"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tips_roll import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=949)
    b, _ = data.synthetic_daily(seed=949)
    for c in ("tips", "nominal", "cash"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_daily(seed=949)
    b, _ = data.synthetic_daily(seed=950)
    assert not np.allclose(a["tips"].to_numpy(), b["tips"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=10, seed=949)
    assert {"tips", "nominal", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert (prices > 0).all().all()


def test_synthetic_cash_is_monotone_growing():
    prices, truth = data.synthetic_daily(seed=949)
    assert (prices["cash"].diff().dropna() > 0).all()
    grew = prices["cash"].iloc[-1] / prices["cash"].iloc[0]
    expected = (1.0 + truth["cash_rate_ann"]) ** truth["n_years"]
    assert abs(grew / expected - 1.0) < 0.02


def test_signal_strength_scales_the_planted_carry():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=949)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=949)
    _, th = data.synthetic_daily(signal_strength=0.5, seed=949)
    assert t0["carry_ann_planted"] == 0.0
    assert t1["carry_ann_planted"] == pytest.approx(t1["carry_ann"])
    assert th["carry_ann_planted"] == pytest.approx(0.5 * t1["carry_ann"])


def test_panel_shape_and_duration_ordering(planted_panel):
    prices, truth = planted_panel
    k = truth["n_buckets"]
    assert k == 3
    for i in range(k):
        assert f"tips_{i}" in prices.columns and f"nom_{i}" in prices.columns
    ex = st.excess_returns(prices, cash_col="cash")
    vols = [ex[f"tips_{i}"].std() for i in range(k)]
    # Longer buckets must be more volatile — the real tape's defining feature.
    assert vols[0] < vols[1] < vols[2]


def test_panel_plants_the_same_carry_in_every_bucket(planted_panel):
    _, truth = planted_panel
    assert truth["carry_ann_planted"] == pytest.approx(truth["carry_ann"])
    assert len(truth["durations"]) == truth["n_buckets"]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=949)
    b, _ = data.synthetic_daily(seed=950)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_ticker_maps_are_consistent():
    assert set(data.NOMINAL_MATCH) == set(data.TIPS_LEGS)
    for leg, nom in data.NOMINAL_MATCH.items():
        assert leg in data.TICKERS and nom in data.TICKERS
        assert nom in data.ALT_MATCH[leg]
    assert data.CASH in data.TICKERS
    assert set(data.HEADLINE_COLS).issubset(set(data.TICKERS))


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — "
                           "synthetic tests cover all the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    ex = st.excess_returns(px, cash_col=data.CASH)
    race = st.ladder_race(ex, data.TIPS_LEGS)
    for lg in data.TIPS_LEGS:
        assert np.isfinite(race["legs"][lg]["sharpe"])
    # Longer real duration must carry more volatility — a sanity check on the pairing.
    assert race["legs"]["LTPZ"]["vol_ann"] > race["legs"]["VTIP"]["vol_ann"]
