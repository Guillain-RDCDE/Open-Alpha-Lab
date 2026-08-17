"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cash_lag import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=923)
    b, _ = data.synthetic_panel(seed=923)
    assert list(a.columns) == list(b.columns)
    for c in a.columns:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_panel(seed=923)
    b, _ = data.synthetic_panel(seed=924)
    assert not np.allclose(a["LONG"].to_numpy(), b["LONG"].to_numpy())


def test_synthetic_shape_and_columns(laddered):
    prices, truth = laddered
    assert set(truth["names"]) | {data.RATE} == set(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"]
    # OOB-safe: the synthetic index must stay well inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rate_is_non_negative_and_moves(laddered):
    prices, _ = laddered
    r = prices[data.RATE]
    assert (r >= 0.0).all()
    assert r.std() > 0.05


def test_synthetic_navs_grow_like_cash(laddered):
    """A bill portfolio in a positive-rate world compounds up, slowly."""
    prices, truth = laddered
    for v in truth["names"]:
        s = prices[v]
        assert s.iloc[-1] > s.iloc[0]
        ann = (s.iloc[-1] / s.iloc[0]) ** (252.0 / len(s)) - 1.0
        assert 0.0 < ann < 0.10


def test_signal_strength_zero_collapses_the_ladder():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=923)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=923)
    assert len(set(t1["effective_wam_days"])) > 1
    assert len(set(t0["effective_wam_days"])) == 1
    assert t0["trend_phi_effective"] == 0.0


def test_fast_and_slow_names_are_stable_across_strength():
    """Tie-breaking on NOMINAL WAM keeps the same pair raced at every strength."""
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=923)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=923)
    assert t1["fastest"] == t0["fastest"] and t1["slowest"] == t0["slowest"]
    assert t1["fastest"] != t1["slowest"]


def test_synthetic_daily_single_vehicle():
    prices, truth = data.synthetic_daily(wam_days=45, seed=923)
    assert list(prices.columns) == ["VEHICLE", data.RATE]
    assert truth["effective_wam_days"] == [45]


def test_fingerprint_stable_and_sensitive(laddered):
    prices, _ = laddered
    fp1 = data.fingerprint(prices)
    other, _ = data.synthetic_panel(seed=924)
    assert fp1 == data.fingerprint(prices) and len(fp1) == 12
    assert fp1 != data.fingerprint(other)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_nominal_wam_covers_every_vehicle():
    assert set(data.NOMINAL_WAM_DAYS) == set(data.VEHICLES)
    assert data.BENCH in data.VEHICLES
    assert set(data.HEADLINE_VEHICLES).issubset(set(data.VEHICLES))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_headline_frame_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    hf = data.headline_frame(px)
    assert hf.index[0] >= pd.Timestamp(data.HEADLINE_START)
    assert not hf.isna().any().any()
    d = st.effective_duration(hf["SHV"], hf[data.RATE])
    assert np.isfinite(d["duration_years"]) and np.isfinite(d["tstat"])
