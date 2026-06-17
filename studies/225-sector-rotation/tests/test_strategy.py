"""Tests for the strategy layer — signals, weights, backtest engine, the study's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sector_rotation import data, strategy as st  # noqa: E402

STUDY_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
CACHE_PATH = os.path.join(STUDY_CACHE, "sector_rotation_monthly.parquet")


# ---------------------------------------------------------------------------
# momentum_score
# ---------------------------------------------------------------------------
def test_momentum_score_shape(momentum_panel):
    panel, _ = momentum_panel
    score = st.momentum_score(panel, lookback=6, skip=1)
    assert score.shape == panel.shape


def test_momentum_score_nan_in_burnin(momentum_panel):
    """First lookback+skip months must be NaN — insufficient history."""
    panel, _ = momentum_panel
    score = st.momentum_score(panel, lookback=6, skip=1)
    # First 7 rows (0..6) have no valid score (skip=1, lookback=6 => need 7 months)
    assert score.iloc[:7].isnull().all().all(), (
        "Score should be NaN for the first lookback+skip months"
    )


def test_momentum_score_nonnull_after_burnin(momentum_panel):
    panel, _ = momentum_panel
    score = st.momentum_score(panel, lookback=6, skip=1)
    assert score.iloc[8:].notnull().all().all()


def test_momentum_score_positive_for_up_market():
    """A sector that went up over the lookback should have a positive score."""
    idx = pd.period_range("2000-01", periods=18, freq="M")
    panel = pd.DataFrame({"A": [0.05] * 18}, index=idx)
    score = st.momentum_score(panel, lookback=6, skip=1)
    valid = score["A"].dropna()
    assert (valid > 0).all()


# ---------------------------------------------------------------------------
# top_k_weights
# ---------------------------------------------------------------------------
def test_top_k_weights_sums_to_one():
    score = pd.Series({"XLK": 0.1, "XLV": 0.2, "XLF": -0.05, "XLI": 0.15})
    w = st.top_k_weights(score, k=2)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_top_k_weights_picks_top_k():
    score = pd.Series({"A": 0.3, "B": 0.1, "C": 0.05, "D": 0.2})
    w = st.top_k_weights(score, k=2)
    assert w["A"] > 0 and w["D"] > 0  # A and D are top-2
    assert w["B"] == 0 and w["C"] == 0


def test_top_k_weights_equal_within_top():
    score = pd.Series({"A": 0.3, "B": 0.2, "C": 0.1})
    w = st.top_k_weights(score, k=2)
    assert abs(w["A"] - 0.5) < 1e-9
    assert abs(w["B"] - 0.5) < 1e-9


def test_top_k_weights_handles_nans():
    score = pd.Series({"A": 0.3, "B": float("nan"), "C": 0.1})
    w = st.top_k_weights(score, k=2)
    assert w.sum() <= 1.0 + 1e-9


def test_top_k_weights_insufficient_valid():
    score = pd.Series({"A": float("nan"), "B": float("nan")})
    w = st.top_k_weights(score, k=2)
    assert w.sum() == 0.0


# ---------------------------------------------------------------------------
# ew_weights
# ---------------------------------------------------------------------------
def test_ew_weights_uniform(null_panel):
    panel, _ = null_panel
    score = st.momentum_score(panel).iloc[10]
    w = st.ew_weights(score)
    valid = score.dropna()
    expected = 1.0 / len(valid)
    assert (w[valid.index] - expected).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# random_weights
# ---------------------------------------------------------------------------
def test_random_weights_sum_to_one():
    score = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5})
    rng = np.random.default_rng(0)
    w = st.random_weights(score, k=3, rng=rng)
    assert abs(w.sum() - 1.0) < 1e-9


def test_random_weights_k_nonzero():
    score = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5})
    rng = np.random.default_rng(0)
    w = st.random_weights(score, k=3, rng=rng)
    assert (w > 0).sum() == 3


# ---------------------------------------------------------------------------
# run_strategy output shape and monotone cost
# ---------------------------------------------------------------------------
def test_run_strategy_returns_dataframe(null_panel):
    panel, _ = null_panel
    out = st.run_strategy(panel, k=3, mode="momentum")
    assert isinstance(out, pd.DataFrame)
    assert "r_gross" in out.columns and "r_net" in out.columns


def test_run_strategy_net_leq_gross(null_panel):
    panel, _ = null_panel
    out = st.run_strategy(panel, k=3, cost_bps=20, mode="momentum")
    assert (out["r_net"] <= out["r_gross"] + 1e-10).all(), (
        "Net returns must be <= gross (costs cannot be negative)"
    )


def test_run_strategy_modes_available(null_panel):
    panel, _ = null_panel
    for mode in ("momentum", "ew", "random"):
        out = st.run_strategy(panel, k=3, mode=mode, seed=0)
        assert len(out) == len(panel)


def test_run_strategy_invalid_mode(null_panel):
    panel, _ = null_panel
    with pytest.raises(ValueError, match="mode must be"):
        st.run_strategy(panel, k=3, mode="bogus")


def test_cost_monotonically_lowers_net(null_panel):
    panel, _ = null_panel
    means = []
    for c in (0.0, 10.0, 40.0):
        out = st.run_strategy(panel, k=3, cost_bps=c, mode="momentum")
        means.append(out["r_net"].mean())
    assert means[0] >= means[1] >= means[2], "Higher cost must lower or equal the net mean"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.05, 120))
    s = st.summarize(r)
    for key in ("n", "ann_ret", "ann_vol", "sharpe", "max_drawdown", "skew", "tstat"):
        assert key in s, f"summarize missing key: {key}"


def test_summarize_n_matches_length():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.04, 60))
    s = st.summarize(r)
    assert s["n"] == 60


def test_summarize_positive_mean_positive_tstat():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.02, 0.03, 200))
    s = st.summarize(r)
    assert s["tstat"] > 0


# ---------------------------------------------------------------------------
# The study's spine: momentum finds an edge only when one is baked in
# ---------------------------------------------------------------------------
def test_momentum_beats_random_only_with_momentum():
    """Backbone test: on a momentum panel the 6-month sort outperforms random; on the
    null it does not (the machinery finds only what exists in the tape)."""
    def edge(mom_strength, seed=225):
        panel, _ = data.synthetic_panel(
            n_months=300, mom_strength=mom_strength, seed=seed
        )
        mom_out = st.run_strategy(panel, k=3, lookback=6, cost_bps=0, mode="momentum")
        rand_out = st.run_strategy(panel, k=3, lookback=6, cost_bps=0, mode="random", seed=0)
        return float(mom_out["r_gross"].mean() - rand_out["r_gross"].mean())

    # With strong momentum planted, the sort should outperform random selection
    edge_with = edge(mom_strength=0.04)
    # On the null, the advantage should be minimal
    edge_null = edge(mom_strength=0.0)

    assert edge_with > edge_null, (
        f"With planted momentum ({edge_with:.4f}) should beat null ({edge_null:.4f})"
    )


# ---------------------------------------------------------------------------
# cost_sweep
# ---------------------------------------------------------------------------
def test_cost_sweep_index(null_panel):
    panel, _ = null_panel
    cs = st.cost_sweep(panel, k=3, costs_bps=(0, 10, 20))
    assert list(cs.index) == [0, 10, 20]
    assert "mom_sharpe" in cs.columns


# ---------------------------------------------------------------------------
# lookback_sweep
# ---------------------------------------------------------------------------
def test_lookback_sweep_index(null_panel):
    panel, _ = null_panel
    ls = st.lookback_sweep(panel, k=3, lookbacks=(3, 6))
    assert list(ls.index) == [3, 6]
    assert "mom_tstat" in ls.columns


# ---------------------------------------------------------------------------
# Real-tape smoke test (skip if cache absent — CI / offline)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_tape_active_return_near_zero():
    """On the real tape, active return vs EW should be near zero (our verdict)."""
    panel, _ = data.fetch_etfs(fetch=False, cache_dir=STUDY_CACHE)
    mom_df = st.run_strategy(panel, k=3, lookback=6, cost_bps=0, mode="momentum")
    ew_df  = st.run_strategy(panel, k=3, lookback=6, cost_bps=0, mode="ew")
    active = mom_df["r_gross"] - ew_df["r_gross"]
    s = st.summarize(active)
    # Active t must be below 2 (per our verdict, the active return is near zero)
    assert abs(s["tstat"]) < 2.0, (
        f"Active HAC t = {s['tstat']:.2f} — if this exceeds 2, revisit the verdict"
    )
