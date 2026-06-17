"""Tests for the strategy layer of Study 271 (Cardboard-Box).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded freight table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardboard_box import data, strategy as st  # noqa: E402


def _synth_frame(beta: float = 0.0, n_years: int = 400, seed: int = 271) -> pd.DataFrame:
    df, _ = data.synthetic_panel(n_years=n_years, beta=beta, seed=seed)
    return df.rename(columns={"box_growth": "predictor"})


# ---------------------------------------------------------------------------
# ols_hac
# ---------------------------------------------------------------------------
def test_ols_hac_recovers_slope():
    """On a clean linear DGP, ols_hac recovers the slope and a big t-stat."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 500)
    res = st.ols_hac(y, x, lags=1)
    assert abs(res["beta"] - 3.0) < 0.1
    assert abs(res["alpha"] - 2.0) < 0.1
    assert abs(res["t_beta"]) > 10
    assert 0.0 <= res["r2"] <= 1.0


def test_ols_hac_null_small_t():
    """With no relationship, the slope t-stat is modest."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 600)
    y = rng.normal(0, 1, 600)
    res = st.ols_hac(y, x, lags=1)
    assert abs(res["t_beta"]) < 3


def test_ols_hac_handles_short_input():
    res = st.ols_hac(np.array([1.0]), np.array([2.0]), lags=1)
    assert np.isnan(res["t_beta"])
    assert res["n"] == 1


# ---------------------------------------------------------------------------
# predictive_regression
# ---------------------------------------------------------------------------
def test_predictive_regression_keys():
    frame = _synth_frame(beta=0.0, n_years=100)
    res = st.predictive_regression(frame, lags=1)
    for k in ("alpha", "beta", "t_beta", "r2", "n", "pearson_r", "pearson_p"):
        assert k in res


def test_predictive_regression_null_insignificant():
    """beta=0 synthetic tape -> forecasting slope is not significant."""
    frame = _synth_frame(beta=0.0, n_years=2000, seed=271)
    res = st.predictive_regression(frame, lags=1)
    assert abs(res["t_beta"]) < 2.5


def test_predictive_regression_planted_signal_detected():
    """A real planted beta -> |t| clears 2 (the positive control)."""
    frame = _synth_frame(beta=4.0, n_years=500, seed=271)
    res = st.predictive_regression(frame, lags=1)
    assert res["t_beta"] > 2
    assert res["beta"] > 0


# ---------------------------------------------------------------------------
# timing_strategy
# ---------------------------------------------------------------------------
def test_timing_strategy_keys():
    frame = _synth_frame(beta=0.0, n_years=100)
    frame["ret_year"] = frame["year"] + 1
    r = st.timing_strategy(frame, threshold=0.0, cost_bps=10.0)
    for k in ("bah_ann", "gross_ann", "net_ann", "turnover", "hit_rate",
              "uncond_up_rate", "frac_invested", "net_minus_bah"):
        assert k in r


def test_timing_costs_reduce_return():
    """Net return is never above gross (costs are non-negative)."""
    frame = _synth_frame(beta=0.0, n_years=200)
    frame["ret_year"] = frame["year"] + 1
    r0 = st.timing_strategy(frame, threshold=0.0, cost_bps=0.0)
    r1 = st.timing_strategy(frame, threshold=0.0, cost_bps=50.0)
    assert r1["net_ann"] <= r0["net_ann"] + 1e-12
    assert r1["gross_ann"] == pytest.approx(r0["gross_ann"], abs=1e-12)


def test_timing_turnover_and_frac_in_range():
    frame = _synth_frame(beta=0.0, n_years=200)
    frame["ret_year"] = frame["year"] + 1
    r = st.timing_strategy(frame, threshold=0.0, cost_bps=10.0)
    assert 0.0 <= r["turnover"] <= 2.0
    assert 0.0 <= r["frac_invested"] <= 1.0
    assert 0.0 <= r["hit_rate"] <= 1.0


# ---------------------------------------------------------------------------
# coincident_regression
# ---------------------------------------------------------------------------
def test_coincident_regression_runs(freight):
    gspc = pd.DataFrame({"year": list(range(1970, 2025)),
                         "gspc_return": np.linspace(-0.1, 0.2, 55)})
    res = st.coincident_regression(freight, gspc, predictor="box_growth", lags=1)
    assert "t_beta" in res and "pearson_r" in res
    assert res["n"] == 55


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(freight):
    gspc = pd.DataFrame({"year": list(range(1971, 2026)),
                         "gspc_return": np.linspace(-0.05, 0.18, 55)})
    s = st.summarize(freight, gspc, cost_bps=10.0, lags=1)
    required = {
        "n", "box_beta", "box_t", "box_r2", "rail_t",
        "coin_box_t", "uncond_up_pct", "hit_rate_pct", "turnover",
        "bah_ann_pct", "gross_ann_pct", "net_ann_pct", "net_minus_bah_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches(freight):
    gspc = pd.DataFrame({"year": list(range(1971, 2026)),
                         "gspc_return": np.linspace(-0.05, 0.18, 55)})
    s = st.summarize(freight, gspc, cost_bps=10.0, lags=1)
    assert s["n"] == 55
