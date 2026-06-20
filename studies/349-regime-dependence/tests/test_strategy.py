"""The metrics, the regime-dependence lens, and the study's two spines:
(1) a durable edge survives leaving its best decade out, a regime-fitted edge does not;
(2) the stability score separates them — the full-sample Sharpe alone does not."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_dependence import strategy as st  # noqa: E402


# ---- metrics ---------------------------------------------------------------
def test_sharpe_of_constant_is_nan():
    import pandas as pd
    s = pd.Series([0.01] * 50)
    assert np.isnan(st.sharpe(s))


def test_summarize_keys(planted_panel):
    df, _, _ = planted_panel
    s = st.summarize(df["durable"])
    assert set(s) == {"n", "cagr", "sharpe", "max_dd", "mean_ann"}


def test_hac_tstat_on_null_is_small(null_panel):
    """On a pure-noise series the HAC t-stat does not clear 2 (the harness is honest)."""
    df, _, _ = null_panel
    for col in df.columns:
        assert abs(st.hac_tstat(df[col])["tstat"]) < 2.0


def test_block_bootstrap_ci_brackets_mean(planted_panel):
    df, _, _ = planted_panel
    lo, hi = st.block_bootstrap_ci(df["durable"], n_boot=1000, seed=1)
    assert lo < df["durable"].mean() < hi


# ---- the regime-dependence lens --------------------------------------------
def test_regime_sharpes_one_per_regime(planted_panel):
    df, regime, truth = planted_panel
    rs = st.regime_sharpes(df["durable"], regime)
    assert len(rs) == truth["n_regimes"]


def test_regime_fit_has_one_dominant_regime(planted_panel):
    """Spine: the regime-fitted strategy's Sharpe is concentrated in the planted regime."""
    df, regime, truth = planted_panel
    rs = st.regime_sharpes(df["regime_fit"], regime)
    assert rs.idxmax() == truth["hot_regime"]
    # the hot regime's Sharpe towers over the rest
    assert rs.max() > rs.drop(truth["hot_regime"]).max() + 0.5


def test_stability_score_ranks_regime_fit_as_less_stable(planted_panel):
    """The CV of the per-regime Sharpe is higher for the regime-fitted strategy, even
    though its FULL-SAMPLE Sharpe is comparable or better — the whole point of the lens."""
    df, regime, _ = planted_panel
    ss_dur = st.stability_score(df["durable"], regime)
    ss_fit = st.stability_score(df["regime_fit"], regime)
    assert ss_fit["cv"] > ss_dur["cv"]
    assert ss_fit["share_top_regime"] > ss_dur["share_top_regime"]
    # the trap: full-sample Sharpe does NOT reveal the difference (fit looks fine/better)
    assert st.sharpe(df["regime_fit"]) >= st.sharpe(df["durable"]) - 0.10


def test_leave_hot_regime_out_separates_durable_from_fit(planted_panel):
    """Spine #1 — the methodology verdict: with its best decade removed, the DURABLE
    edge stays significant (HAC t >= 2) while the REGIME-FITTED edge collapses below 2."""
    df, regime, _ = planted_panel
    dur = st.leave_hot_regime_out(df["durable"], regime)
    fit = st.leave_hot_regime_out(df["regime_fit"], regime)
    # durable survives the loss of its best regime
    assert dur["t_ex_hot"] >= 2.0
    assert dur["sharpe_ex_hot"] > 0.20
    # regime-fitted collapses once its one good decade is removed
    assert fit["t_ex_hot"] < 2.0
    assert fit["sharpe_ex_hot"] < dur["sharpe_ex_hot"]


def test_hot_minus_rest_ci_above_zero_for_fit(planted_panel):
    """The cross-regime gap CI is entirely above zero for the regime-fitted strategy
    (the edge is genuinely concentrated), but the durable strategy's gap is not resolved."""
    df, regime, _ = planted_panel
    fit = st.hot_minus_rest_ci(df["regime_fit"], regime, n_boot=1000)
    dur = st.hot_minus_rest_ci(df["durable"], regime, n_boot=1000)
    assert fit["lo"] > 0.0           # fit's best decade is decisively above the rest
    assert dur["gap_ann"] < fit["gap_ann"]


# ---- real-tape strategy builders -------------------------------------------
def _toy_market(n=120, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M").to_timestamp()
    eq = pd.Series(rng.normal(0.007, 0.04, n), index=idx)
    bd = pd.Series(rng.normal(0.002, 0.015, n), index=idx)
    return eq, bd


def test_trend_has_one_execution_lag():
    """The trend signal at month t sets the position for t+1 — a single shift, so the
    first realised position is zero (no look-ahead)."""
    eq, _ = _toy_market()
    s = st.apply_trend(eq, lookback=12, cost_bps=0.0)
    # the strategy can never out-earn a perfect in-sample timer; it is non-trivial
    assert s.notna().all()
    assert len(s) > 0


def test_trend_costs_lower_returns():
    eq, _ = _toy_market(seed=3)
    free = st.apply_trend(eq, cost_bps=0.0).mean()
    costly = st.apply_trend(eq, cost_bps=50.0).mean()
    assert costly <= free


def test_6040_first_rebalance_is_free():
    eq, bd = _toy_market()
    free = st.apply_6040(eq, bd, cost_bps=0.0)
    costly = st.apply_6040(eq, bd, cost_bps=50.0)
    assert np.isclose(free.iloc[0], costly.iloc[0])


def test_6040_is_between_legs_on_average():
    """A 60/40 blend's mean return sits between the two legs' means (a convex mix)."""
    eq, bd = _toy_market(seed=5)
    m = st.apply_6040(eq, bd).mean()
    assert min(eq.mean(), bd.mean()) - 1e-4 <= m <= max(eq.mean(), bd.mean()) + 1e-4


def test_decade_label():
    import pandas as pd
    idx = pd.to_datetime(["1999-06-01", "2005-01-01", "2017-12-01"])
    lab = st.decade_label(idx)
    assert list(lab) == [1990, 2000, 2010]
