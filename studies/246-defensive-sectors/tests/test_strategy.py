"""Signal construction, quintile logic, timing overlay, and the study spine:
the defensive-sector timer only outperforms buy-and-hold when a signal is planted,
and reads ~0 without."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defensive_sectors import strategy as st  # noqa: E402


# ---- Combined RS momentum --------------------------------------------------

def test_rs_momentum_is_20day_diff(null_tape):
    df, _ = null_tape
    rs_xlp = np.log(df["XLP_close"]) - np.log(df["SPY_close"])
    rs_xlu = np.log(df["XLU_close"]) - np.log(df["SPY_close"])
    combined = 0.5 * (rs_xlp + rs_xlu)
    expected = combined.diff(20)
    result = st.combined_rs_momentum(df, window=20)
    assert np.allclose(result.dropna().to_numpy(), expected.dropna().to_numpy(), atol=1e-12)


def test_rs_momentum_rank_in_unit_interval(null_tape):
    df, _ = null_tape
    rs_mom = st.combined_rs_momentum(df)
    rank = st.rs_momentum_rank(rs_mom)
    valid = rank.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


# ---- forward return --------------------------------------------------------

def test_forward_return_no_lookahead(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    # The last row must be NaN (we can't look forward)
    assert np.isnan(fwd.iloc[-1])
    # The second-to-last row is the 1-day return of the last close
    expected_last = np.log(df["SPY_close"].iloc[-1]) - np.log(df["SPY_close"].iloc[-2])
    assert np.isclose(fwd.iloc[-2], expected_last)


# ---- quintile table --------------------------------------------------------

def test_quintile_table_shape(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1, 5])
    assert set(tbl.columns) >= {"quintile", "horizon", "n", "mean_bps", "t_stat"}
    assert set(tbl["quintile"].unique()) == {1, 2, 3, 4, 5}
    assert set(tbl["horizon"].unique()) == {1, 5}


def test_quintile_table_counts_sum_to_sample(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1])
    total = tbl[tbl["horizon"] == 1]["n"].sum()
    assert total > 4000


# ---- top_minus_bottom ------------------------------------------------------

def test_top_minus_bottom_keys(null_tape):
    df, _ = null_tape
    result = st.top_minus_bottom(df, horizon=1)
    for key in ("n_alert", "n_calm", "mean_alert_bps", "mean_calm_bps",
                "spread_bps", "t_spread", "t_alert", "t_calm"):
        assert key in result


def test_defensive_signal_detected_with_planted_effect(defensive_tape, null_tape):
    """The spine: a planted negative defensive signal produces a positive Q1-Q5 spread
    (calm days have higher forward returns than defensive-alert days); on the null tape
    the spread is ~0."""
    df_d, _ = defensive_tape
    df_n, _ = null_tape
    spread_d = st.top_minus_bottom(df_d, horizon=5)["spread_bps"]
    spread_n = st.top_minus_bottom(df_n, horizon=5)["spread_bps"]
    # With a planted signal the spread (calm - alert) should be positive
    assert spread_d > 0
    # On the null tape the absolute spread should be much smaller
    assert abs(spread_n) < abs(spread_d)


# ---- timing overlay --------------------------------------------------------

def test_timing_overlay_columns(null_tape):
    df, _ = null_tape
    overlay = st.timing_overlay(df, cost_bps=0.0)
    for col in ("r_passive", "r_active", "r_spread", "is_alert", "switched"):
        assert col in overlay.columns


def test_timing_overlay_flat_when_alert(null_tape):
    df, _ = null_tape
    overlay = st.timing_overlay(df, cost_bps=0.0)
    alert_rows = overlay[overlay["is_alert"].fillna(False)]
    if len(alert_rows) > 0:
        assert np.allclose(alert_rows["r_active"].to_numpy(), 0.0, atol=1e-12)


def test_timing_overlay_equals_passive_when_calm(null_tape):
    df, _ = null_tape
    overlay = st.timing_overlay(df, cost_bps=0.0)
    calm_rows = overlay[~overlay["is_alert"].fillna(True) & ~overlay["switched"]]
    if len(calm_rows) > 0:
        assert np.allclose(
            calm_rows["r_active"].to_numpy(),
            calm_rows["r_passive"].to_numpy(),
            atol=1e-12,
        )


# ---- regime stats ----------------------------------------------------------

def test_regime_stats_three_rows(null_tape):
    df, _ = null_tape
    rs = st.regime_stats(df)
    assert len(rs) == 3
    assert set(rs["regime"]) == {"full sample", "calm (def Q1-Q2)", "alert (def Q3-Q5)"}


def test_regime_stats_sharpe_finite(null_tape):
    df, _ = null_tape
    rs = st.regime_stats(df)
    for _, row in rs.iterrows():
        assert np.isfinite(row["sharpe_ann"])


# ---- summarize -------------------------------------------------------------

def test_summarize_keys():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    s = st.summarize(r, label="test")
    for key in ("label", "n", "mean_bps", "std_bps", "sharpe_ann", "skew", "t_stat"):
        assert key in s


def test_summarize_empty():
    s = st.summarize(np.array([]), label="empty")
    assert s["n"] == 0
    assert np.isnan(s["mean_bps"])
