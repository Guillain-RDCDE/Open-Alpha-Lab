"""Data for the pumpkin-spice-season study — an offline synthetic world and cached SBUX/SPY tapes.

  * :func:`synthetic_world` — **offline, deterministic**. Monthly SBUX-minus-SPY *excess* returns with
    a tunable ``psl_premium`` injected into the "pumpkin-spice season" (Aug–Nov, from the late-August
    PSL launch through Thanksgiving). ``psl_premium = 0`` is the null. Pins the seasonality machinery
    offline. NEVER backs a Signal stamp.
  * :func:`fetch_data` — monthly **total-return** series for Starbucks (``SBUX``), the market (``SPY``)
    and the 13-week T-bill (``^IRX``, the cash leg), **cache-first**, built from *daily* auto-adjusted
    closes resampled to month-end. The headline ``excess`` column is SBUX − SPY. Fingerprinted run in
    ``docs/results.md``.
  * :func:`fetch_basket` — a small equal-weight coffee/QSR basket (SBUX, MCD, YUM, CMG) excess over
    SPY, a robustness leg that only starts once every name is listed (CMG IPO'd 2006). Cache-first.

All equity series are **total-return** (yfinance ``auto_adjust=True`` reinvests dividends & splits) —
labelled as such, never as price-only. Cache lives in the study-local ``_cache/`` (gitignored).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

SBUX, SPY, TBILL = "SBUX", "SPY", "^IRX"
QSR_BASKET = ["SBUX", "MCD", "YUM", "CMG"]  # coffee/QSR names; CMG IPO'd Jan-2006 (sets the start)
MONTHS_PER_YEAR = 12

# The "pumpkin-spice season": Aug–Nov. Starbucks launches the Pumpkin Spice Latte in late August
# (Aug 24–28 in recent years) and the pumpkin-spice line runs through Thanksgiving. The believers'
# window is "beat the market Aug–Nov"; the complement (Dec–Jul) is the off-season.
SEASON_MONTHS = [8, 9, 10, 11]
OFF_MONTHS = [m for m in range(1, 13) if m not in SEASON_MONTHS]


@dataclass(frozen=True)
class WorldTruth:
    psl_premium: float

    @property
    def has_seasonality(self) -> bool:
        return self.psl_premium != 0.0


def synthetic_world(
    n_years: int = 33,
    psl_premium: float = 0.02,
    excess_vol: float = 0.28,
    seed: int = 724,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly SBUX-minus-SPY *excess* world — deterministic given ``seed``.

    Monthly excess returns are i.i.d. with annual vol ``excess_vol`` (a single high-beta name's
    tracking error to the market) plus a ``psl_premium`` added to each pumpkin-spice-season month
    (Aug–Nov) and a symmetric drag spread across the off-season so the annual mean is unchanged.
    ``psl_premium = 0`` is the null. Returns a frame with columns ``excess``, ``sbux``, ``spy``,
    ``tbill`` (``sbux``/``spy`` are a plausible decomposition used only by the offline control).
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1993-01-31", periods=n, freq="ME", name="date")
    months = idx.month

    base = (excess_vol / np.sqrt(12)) * rng.standard_normal(n)
    n_season, n_off = len(SEASON_MONTHS), len(OFF_MONTHS)
    seasonal = np.where(
        np.isin(months, SEASON_MONTHS),
        psl_premium / n_season,
        -psl_premium / n_off,  # symmetric drag → same annual mean under any premium
    )
    excess = base + seasonal

    spy = pd.Series(0.007 + 0.04 / np.sqrt(12) * rng.standard_normal(n), index=idx)  # ~market
    sbux = spy.values + excess
    tbill = pd.Series(0.02 / 12, index=idx)  # flat 2%/yr cash leg
    return (
        pd.DataFrame({"excess": excess, "sbux": sbux, "spy": spy.values, "tbill": tbill.values}, index=idx),
        WorldTruth(psl_premium),
    )


def _assert_continuous_monthly(df: pd.DataFrame) -> None:
    """Fail loudly if the monthly grid has interior holes."""
    months = pd.PeriodIndex(df.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])}) — refetch with fetch_data(fetch=True)"
        )


def _monthly_returns(tickers: list[str], fetch: bool) -> pd.DataFrame:
    """Total-return monthly pct-change for ``tickers`` from daily auto-adjusted closes (network)."""
    import yfinance as yf  # lazy

    px = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()
    return me.pct_change()


def fetch_data(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly total-returns for SBUX, SPY and the T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``excess`` (SBUX − SPY), ``sbux``, ``spy``,
    ``tbill`` (empty DataFrame on a cache miss with ``fetch=False``). Built from **daily**
    auto-adjusted closes resampled to month-end (last close of each calendar month) — total-return,
    not price-only. The in-progress month is dropped so the last bar is always complete. Grid is
    asserted hole-free on every read.
    """
    cache = os.path.join(cache_dir, "pumpkin_spice_season.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()

    ret = _monthly_returns([SBUX, SPY], fetch=True)
    import yfinance as yf  # lazy — the T-bill level, not a return

    tb = yf.download(TBILL, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    tb.index = pd.DatetimeIndex(tb.index).tz_localize(None)
    tb_me = tb.resample("ME").last()
    tb_col = tb_me.iloc[:, 0] if isinstance(tb_me, pd.DataFrame) else tb_me

    out = pd.DataFrame(
        {
            "sbux": ret[SBUX],
            "spy": ret[SPY],
            "tbill": (tb_col.ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["sbux", "spy"])
    out["excess"] = out["sbux"] - out["spy"]
    out = out[["excess", "sbux", "spy", "tbill"]]

    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out


def fetch_basket(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Equal-weight coffee/QSR basket (SBUX, MCD, YUM, CMG) *excess* over SPY, cache-first.

    A robustness leg for the notebooks: is any "pumpkin-spice-season" pop a Starbucks quirk or a
    broad QSR-autumn thing? Columns ``basket`` (eq-weight QSR return), ``spy``, ``excess``
    (basket − SPY), ``tbill``. Starts once every name is listed (CMG IPO'd Jan-2006). Cache-only
    unless ``fetch=True``; empty DataFrame on a cache miss with ``fetch=False``.
    """
    cache = os.path.join(cache_dir, "pumpkin_spice_basket.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()

    ret = _monthly_returns(QSR_BASKET + [SPY], fetch=True)
    import yfinance as yf  # lazy

    tb = yf.download(TBILL, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    tb.index = pd.DatetimeIndex(tb.index).tz_localize(None)
    tb_me = tb.resample("ME").last()
    tb_col = tb_me.iloc[:, 0] if isinstance(tb_me, pd.DataFrame) else tb_me

    qsr = ret[QSR_BASKET].dropna()  # start once all four names have data
    basket = qsr.mean(axis=1)
    out = pd.DataFrame(
        {
            "basket": basket,
            "spy": ret[SPY].reindex(basket.index),
            "tbill": (tb_col.ffill() / 100.0 / MONTHS_PER_YEAR).reindex(basket.index),
        }
    ).dropna(subset=["basket", "spy"])
    out["excess"] = out["basket"] - out["spy"]
    out = out[["excess", "basket", "spy", "tbill"]]

    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
