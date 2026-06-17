"""The synthetic world is deterministic; the spring-premium engine recovers the planted signal when
it is real; the null shows nothing; the timer beats buy-and-hold only when the signal is genuine;
and a gapped cache fails loudly. Offline, seeded."""

import os

import numpy as np
import pandas as pd
import pytest

from crude_seasonality import data, strategy as st

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "crude_seasonality.parquet",
)


def test_world_deterministic(seasonal_world):
    df, truth = seasonal_world
    df2, _ = data.synthetic_world(spring_premium=0.08, seed=226)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_seasonality


def test_seasonal_world_spring_above_autumn(seasonal_world):
    """When a spring premium is planted, spring months must be higher than autumn months."""
    df, _ = seasonal_world
    result = st.spring_autumn_tstat(df["crude"])
    # Spring mean must exceed autumn mean (the planted premium)
    assert result["spring_mean"] > result["autumn_mean"]
    # And the t-stat should be positive and significant with a 0.03 premium over 26 years
    assert result["tstat"] > 2.0


def test_null_world_no_seasonality(null_world):
    """With no planted premium, spring vs autumn t-stat must be below the signal bar."""
    df, _ = null_world
    result = st.spring_autumn_tstat(df["crude"])
    assert abs(result["tstat"]) < 2.0


def test_timer_beats_buyhold_when_signal_real(seasonal_world):
    """A genuine spring premium means the calendar timer outperforms buy-and-hold."""
    df, _ = seasonal_world
    timer = st.seasonal_timer(df["crude"])
    bh = st.buy_hold(df["crude"])
    assert st.summary(timer)["sharpe"] > st.summary(bh)["sharpe"]


def test_timer_in_spring_earns_crude_return(seasonal_world):
    """In spring months the timer must be long (return == crude return)."""
    df, _ = seasonal_world
    timer = st.seasonal_timer(df["crude"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    spring = timer[timer.index.month.isin(st.SPRING_MONTHS)]
    crude_sp = df["crude"].reindex(spring.index)
    assert np.allclose(spring.values, crude_sp.values)


def test_timer_in_autumn_earns_negative_crude(seasonal_world):
    """In autumn months the timer must be short (return == −crude return)."""
    df, _ = seasonal_world
    timer = st.seasonal_timer(df["crude"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    autumn = timer[timer.index.month.isin(st.AUTUMN_MONTHS)]
    crude_au = df["crude"].reindex(autumn.index)
    assert np.allclose(autumn.values, -crude_au.values)


def test_timer_flat_months_earn_tbill(seasonal_world):
    """Outside spring and autumn the timer must earn the passed T-bill rate."""
    df, _ = seasonal_world
    tbill = pd.Series(0.003, index=df.index)
    timer = st.seasonal_timer(df["crude"], tbill=tbill)
    timer.index = pd.DatetimeIndex(timer.index)
    flat_months = [m for m in range(1, 13) if m not in st.SPRING_MONTHS + st.AUTUMN_MONTHS]
    flat = timer[timer.index.month.isin(flat_months)]
    assert (flat == 0.003).all()


def test_month_stats_shape(seasonal_world):
    """month_stats returns a 12-row DataFrame with the expected columns."""
    df, _ = seasonal_world
    ms = st.month_stats(df["crude"])
    assert len(ms) == 12
    assert set(["mean", "std", "n", "tstat"]).issubset(ms.columns)


def test_summary_sharpe_convention(seasonal_world):
    """Raw Sharpe by default; excess-of-cash Sharpe when rf is given — a positive rf lowers it."""
    df, _ = seasonal_world
    bh = st.buy_hold(df["crude"])
    rf = pd.Series(0.002, index=df.index)
    raw = st.summary(bh)
    excess = st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert excess[k] == raw[k]


def test_fetch_data_rejects_gapped_cache(tmp_path, seasonal_world):
    """A stale cache with interior holes must fail loudly."""
    df, _ = seasonal_world
    frame = pd.DataFrame({"crude": df["crude"], "energy": df["energy"], "tbill": 0.001})
    gapped = frame.drop(frame.index[5:8])  # punch a 3-month hole
    gapped.to_parquet(tmp_path / "crude_seasonality.parquet")
    with pytest.raises(AssertionError, match="hole"):
        data.fetch_data(cache_dir=str(tmp_path))


def test_fetch_data_accepts_continuous_cache(tmp_path, seasonal_world):
    df, _ = seasonal_world
    frame = pd.DataFrame({"crude": df["crude"], "energy": df["energy"], "tbill": 0.001})
    frame.to_parquet(tmp_path / "crude_seasonality.parquet")
    out = data.fetch_data(cache_dir=str(tmp_path))
    assert list(out.columns) == ["crude", "energy", "tbill"]
    assert len(out) == len(frame)


@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_is_continuous():
    """The real cache must be hole-free (asserted inside fetch_data, also checked here)."""
    d = data.fetch_data()
    assert not d.empty
    months = pd.PeriodIndex(d.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    assert len(expected.difference(months)) == 0
