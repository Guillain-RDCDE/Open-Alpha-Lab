"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frn_front import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=922)
    b, _ = data.synthetic_panel(seed=922)
    for c in ("frn", "bills", "fixed", "IRX"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=8, seed=922)
    assert {"frn", "bills", "fixed", "IRX"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 8 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rate_path_has_a_full_cycle():
    prices, truth = data.synthetic_panel(seed=922)
    irx = prices["IRX"]
    assert truth["rate_max"] - truth["rate_min"] == pytest.approx(truth["amplitude"], rel=1e-6)
    # rises then falls: the top is in the middle of the sample, not at either end.
    top = irx.to_numpy().argmax()
    assert 0.2 * len(irx) < top < 0.9 * len(irx)


def test_signal_strength_only_touches_the_fixed_leg():
    """The knob is the fixed leg's duration — the rate cycle is identical either way."""
    p1, t1 = data.synthetic_panel(signal_strength=1.0, seed=922)
    p0, t0 = data.synthetic_panel(signal_strength=0.0, seed=922)
    assert np.allclose(p1["IRX"].to_numpy(), p0["IRX"].to_numpy())
    assert np.allclose(p1["frn"].to_numpy(), p0["frn"].to_numpy())
    assert t1["duration_eff"] > 0 and t0["duration_eff"] == 0.0
    # Switching the duration on has to hurt the fixed leg through the hiking phase: the
    # daily vol barely moves (a slow ramp is a few bps a day), but the path does.
    dd1 = st.max_drawdown(p1["fixed"].pct_change())
    dd0 = st.max_drawdown(p0["fixed"].pct_change())
    assert dd1 < dd0 - 0.01


def test_floater_tracks_the_short_rate_and_fixed_does_not():
    prices, truth = data.synthetic_panel(seed=922)
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    hi = prices["IRX"] > prices["IRX"].median()
    # the floater earns visibly more per day when the rate is high; the fixed leg,
    # carrying duration, does not have to.
    assert rets["frn"][hi.reindex(rets.index).fillna(False)].mean() > rets["frn"][~hi.reindex(rets.index).fillna(True)].mean()


def test_bill_ladder_lags_the_short_rate(planted):
    """The bill leg's yield is a trailing average, so it trails the floater on the way up."""
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    reg = st.irx_regime(prices["IRX"])
    tbl = st.regime_table(rets, reg, pairs=[("frn", "bills")])
    assert tbl.loc["rising", "frn-bills"] > tbl.loc["falling", "frn-bills"]


def test_synthetic_daily_view():
    prices, truth = data.synthetic_daily(n_years=6, seed=922)
    assert list(prices.columns) == ["asset", "cash"]
    assert len(prices) == truth["n_days"]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=922)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_panel(seed=923)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_cache_path_strips_caret():
    p = data._cache_path("^IRX", "/tmp/x")
    assert p.endswith("prices_IRX_1d.parquet")


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    assert set(data.FUNDS).issubset(px.columns) and "IRX" in px.columns
    rets = st.daily_returns(px)
    res = st.pair_race(rets, "USFR", "SHY")
    assert np.isfinite(res["ann_diff"]) and np.isfinite(res["tstat"])
    # the fixed 1-3y leg carries duration, so it must be the more volatile sleeve.
    assert rets["SHY"].std() > rets["BIL"].std()
