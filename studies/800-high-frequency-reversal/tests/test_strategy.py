"""Strategy tests — signal alignment (no look-ahead), portfolio construction, inference
primitives, the costed timer / haircut, and the study's spine: a planted REAL reversal
survives a one-week skip while a planted PURE bounce dies. Offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_reversal import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Signal construction — no look-ahead
# --------------------------------------------------------------------------- #
def test_trailing_return_burn_in_nan(null_panel):
    px, _ = null_panel
    sig = st.trailing_return(px, skip=0)
    assert sig.iloc[:2].isnull().all().all(), "signal must be NaN for burn-in rows"


def test_trailing_return_skip_adds_one_lag(null_panel):
    px, _ = null_panel
    s0 = st.trailing_return(px, skip=0)
    s1 = st.trailing_return(px, skip=1)
    pd.testing.assert_frame_equal(s1.iloc[3:], s0.shift(1).iloc[3:])


def test_trailing_return_deterministic(null_panel):
    px, _ = null_panel
    pd.testing.assert_frame_equal(st.trailing_return(px), st.trailing_return(px))


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def test_quintile_columns_and_nonempty(null_panel):
    px, _ = null_panel
    res = st.quintile_returns(st.trailing_return(px), px, q=0.20)
    assert {"loser", "winner", "market", "spread", "loser_turn", "winner_turn",
            "turn", "n_total"}.issubset(res.columns)
    assert len(res) > 0


def test_spread_is_loser_minus_winner(null_panel):
    px, _ = null_panel
    res = st.quintile_returns(st.trailing_return(px), px, q=0.20)
    np.testing.assert_allclose(res["spread"].values,
                               res["loser"].values - res["winner"].values, atol=1e-10)


def test_market_between_legs_most_weeks(null_panel):
    px, _ = null_panel
    res = st.quintile_returns(st.trailing_return(px), px, q=0.20)
    lo, wi, mkt = res["loser"].values, res["winner"].values, res["market"].values
    pct = np.mean((mkt >= np.minimum(lo, wi)) & (mkt <= np.maximum(lo, wi)))
    assert pct > 0.5


def test_turnover_in_unit_interval(null_panel):
    px, _ = null_panel
    res = st.quintile_returns(st.trailing_return(px), px, q=0.20)
    for col in ("loser_turn", "winner_turn", "turn"):
        assert (res[col] >= 0).all() and (res[col] <= 1).all()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_welch_and_one_sample_signs():
    rng = np.random.default_rng(0)
    a = rng.normal(0.01, 0.02, 400); b = rng.normal(0.0, 0.02, 400)
    assert st.welch_t(a, b) > 2
    assert st.one_sample_t(a) > 2
    assert abs(st.one_sample_t(rng.normal(0, 0.02, 400))) < 3


def test_newey_west_returns_t_and_lags():
    rng = np.random.default_rng(1)
    t, lags = st.newey_west_t(rng.normal(0.005, 0.02, 300))
    assert isinstance(t, float) and lags >= 1


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_summarize_keys_and_short_series():
    s = st.summarize(pd.Series(np.random.default_rng(0).standard_normal(60) * 0.02))
    assert {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}.issubset(s)
    short = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(short["tstat"])


def test_beta_alpha_recovers_planted():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.002, 0.02, 400))
    leg = 0.7 * mkt + 0.001 + pd.Series(rng.normal(0, 0.005, 400))
    beta, alpha = st.beta_alpha(leg, mkt)
    assert abs(beta - 0.7) < 0.15


# --------------------------------------------------------------------------- #
# Costed timer / haircut
# --------------------------------------------------------------------------- #
def test_net_spread_below_gross_and_borrow_bites():
    df = pd.DataFrame({"spread": [0.01] * 10, "turn": [0.5] * 10,
                       "loser_turn": [0.5] * 10, "winner_turn": [0.5] * 10,
                       "loser_sprd": [0.004] * 10, "winner_sprd": [0.004] * 10})
    net = st.net_spread(df, one_way_bps=10.0, borrow_bps_annual=100.0)
    assert (net < df["spread"]).all()


def test_bounce_haircut_charges_leg_spreads():
    df = pd.DataFrame({"spread": [0.01] * 10, "turn": [0.8] * 10,
                       "loser_turn": [0.8] * 10, "winner_turn": [0.8] * 10,
                       "loser_sprd": [0.006] * 10, "winner_sprd": [0.003] * 10})
    hc = st.bounce_haircut_spread(df, spread_mult=1.0)
    # each leg pays its own spread x turnover: 0.006*0.8 + 0.003*0.8 = 0.0072
    np.testing.assert_allclose(hc.values, 0.01 - 0.0072, atol=1e-9)
    # the illiquid loser leg must dominate the haircut
    hc_lo = st.bounce_haircut_spread(df.assign(winner_sprd=0.0), spread_mult=1.0)
    assert (hc_lo > hc).all()


def test_break_even_positive_for_positive_spread():
    df = pd.DataFrame({"spread": [0.005] * 10, "turn": [0.5] * 10})
    assert st.break_even_cost(df) > 0


# --------------------------------------------------------------------------- #
# The spine: skip separates real reversal from bid-ask bounce
# --------------------------------------------------------------------------- #
def test_null_panel_spread_small(null_panel):
    px, _ = null_panel
    d = st.detect_spread(px, skip=0)
    assert np.isfinite(d["tstat"]) and abs(d["tstat"]) < 4.0


def test_real_reversal_survives_skip(reversal_panel):
    px, _ = reversal_panel
    d0 = st.detect_spread(px, skip=0)
    d1 = st.detect_spread(px, skip=1)
    assert d0["tstat"] > 3, f"planted reversal should fire at skip=0, got {d0['tstat']:.2f}"
    assert d1["tstat"] > 2, f"a REAL reversal must survive the skip, got {d1['tstat']:.2f}"


def test_pure_bounce_dies_under_skip(bounce_panel):
    px, _ = bounce_panel
    d0 = st.detect_spread(px, skip=0)
    d1 = st.detect_spread(px, skip=1)
    assert d0["tstat"] > 3, f"bounce should fake a skip=0 reversal, got {d0['tstat']:.2f}"
    assert d1["tstat"] < 2, f"pure bounce must DIE under the skip, got {d1['tstat']:.2f}"


# --------------------------------------------------------------------------- #
# Real-data smoke test — guarded by cache/panel presence
# --------------------------------------------------------------------------- #
def _have_panel() -> bool:
    return data.have_real() or data._shared_daily_path() is not None


requires_panel = pytest.mark.skipif(
    not _have_panel(),
    reason="no weekly cache or shared daily panel (offline CI); covered by synthetic tests",
)


@requires_panel
def test_real_pipeline_runs():
    close, spread = data.load_real()
    res = st.quintile_returns(st.trailing_return(close), close, spreads=spread, q=0.20)
    assert res["spread"].notna().sum() > 200
    assert isinstance(st.summarize(res["spread"].dropna())["tstat"], float)
