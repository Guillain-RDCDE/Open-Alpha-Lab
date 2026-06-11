"""The synthetic world is deterministic; the predictive regression recovers a significant negative
slope only when oil really forecasts equities; the null shows nothing; the timing rule beats
buy-and-hold when the signal is real; the rule is in cash unless oil fell — and the cash leg earns the
T-bill. The lag is calendar (a hole in the grid yields NaN, never a 2–3-month-old value), and a gapped
cache fails loudly. Offline, seeded."""

import numpy as np
import pandas as pd
import pytest

from black_gold import data, strategy as st


def test_world_deterministic(predictive_world):
    df, truth = predictive_world
    df2, _ = data.synthetic_world(oil_loads=-0.10, seed=49)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.oil_predicts


def test_predictive_world_negative_significant_slope(predictive_world):
    df, _ = predictive_world
    reg = st.predict_regression(df["oil"], df["eq"])
    assert reg["slope"] < 0.0       # the Driesprong sign
    assert reg["tstat"] < -2.0      # and significant


def test_null_world_no_relationship(null_world):
    df, _ = null_world
    reg = st.predict_regression(df["oil"], df["eq"])
    assert abs(reg["tstat"]) < 2.0


def test_timing_beats_buyhold_when_signal_real(predictive_world):
    df, _ = predictive_world
    timing = st.summary(st.oil_timing(df["oil"], df["eq"]))
    bh = st.summary(st.buy_hold(df["eq"]))
    assert timing["sharpe"] > bh["sharpe"]   # real signal ⇒ timing adds value


def test_timing_in_cash_unless_oil_fell(predictive_world):
    df, _ = predictive_world
    t = st.oil_timing(df["oil"], df["eq"])
    rose = (df["oil"].shift(1) >= 0).reindex(t.index)
    assert (t[rose] == 0.0).all()            # cash after an up-oil month (no T-bill passed → 0)


def test_timing_cash_leg_earns_tbill(predictive_world):
    df, _ = predictive_world
    tbill = pd.Series(0.003, index=df.index)                 # a flat 0.3%/month cash rate
    t = st.oil_timing(df["oil"], df["eq"], tbill)
    rose = (df["oil"].shift(1) >= 0).reindex(t.index).fillna(False)
    assert (t[rose] == 0.003).all()          # out-of-market months are credited, not zeroed
    fell = ~rose
    assert np.allclose(t[fell], df["eq"].reindex(t.index)[fell])  # in-market months are pure equity


def test_lag_is_calendar_not_positional():
    """A hole in the monthly grid must yield NaN — never a stale 2-month-old value."""
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"])  # March missing
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    lag = st.lag_one_month(s)
    assert np.isnan(lag.iloc[0])             # no month before the first
    assert lag.iloc[1] == 1.0                # Feb sees Jan
    assert np.isnan(lag.iloc[2])             # Apr must NOT see Feb (positional shift would)


def test_summary_sharpe_convention(predictive_world):
    """Raw Sharpe by default; excess-of-cash when rf is given — and a positive rf lowers it."""
    df, _ = predictive_world
    bh = st.buy_hold(df["eq"])
    rf = pd.Series(0.002, index=df.index)
    raw, excess = st.summary(bh), st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):            # raw-series stats are rf-independent
        assert excess[k] == raw[k]


def test_fetch_pair_rejects_gapped_cache(tmp_path, predictive_world):
    """A stale cache with interior holes must fail loudly, not silently mis-lag."""
    df, _ = predictive_world
    frame = pd.DataFrame({"oil": df["oil"], "eq": df["eq"], "tbill": 0.001})
    gapped = frame.drop(frame.index[5:8])                    # punch a 3-month hole
    gapped.to_parquet(tmp_path / "black_gold_pair.parquet")
    with pytest.raises(AssertionError, match="hole"):
        data.fetch_pair(cache_dir=str(tmp_path))


def test_fetch_pair_accepts_continuous_cache(tmp_path, predictive_world):
    df, _ = predictive_world
    frame = pd.DataFrame({"oil": df["oil"], "eq": df["eq"], "tbill": 0.001})
    frame.to_parquet(tmp_path / "black_gold_pair.parquet")
    out = data.fetch_pair(cache_dir=str(tmp_path))
    assert list(out.columns) == ["oil", "eq", "tbill"] and len(out) == len(frame)
