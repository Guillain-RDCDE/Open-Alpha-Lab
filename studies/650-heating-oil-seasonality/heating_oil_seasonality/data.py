"""Data layer for Study 650 — Heating-Oil-Seasonality.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily HO=F (NY Harbor ULSD / "heating oil" futures) raw OHLC — Yahoo's
  continuous front-month chain, which splices to the next active contract at/near expiry with
  **no back-adjustment**. That splice jump is not an artefact to apologize for: it is exactly
  the term-structure cost (contango drag) or gain (backwardation) a real futures roller pays on
  the same day — the "roll/contango caveat" the brief asks for is *baked into this series by
  construction*, not bolted on afterward. Daily UHN (United States Heating Oil Fund, the retail
  ETF wrapper) adjusted closes for the third axis, and ^IRX (13-week T-bill) for the cash leg of
  the seasonal timer. All from yfinance (no key), cached as CSV under the study's own ``_cache/``.

* **The heating-season calendar, hardcoded.** Three windows, defined once and used everywhere:
  ``AUTUMN_BUILD_MONTHS`` (Sep-Nov, the anticipatory "build" ahead of peak demand),
  ``WINTER_DRAW_MONTHS`` (Dec-Feb, the "draw" — physical inventories fall as furnaces run) and
  their union ``HEATING_MONTHS`` (Sep-Feb) versus ``OFF_SEASON_MONTHS`` (Mar-Aug). This is a
  calendar fact, not a fitted parameter — no network needed to define it.

* **Synthetic world.** A deterministic, seeded i.i.d. monthly-return generator with a TUNABLE
  planted heating-season premium (knob ``seasonal``, spread evenly over the 6 heating months).
  ``seasonal = 0`` is the null world — heating months statistically identical to the rest; the
  Welch machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
HO_CACHE = os.path.join(CACHE_DIR, "hos_ho.csv")
UHN_CACHE = os.path.join(CACHE_DIR, "hos_uhn.csv")
IRX_CACHE = os.path.join(CACHE_DIR, "hos_irx.csv")

START = "2000-09-01"        # HO=F's first printed session on Yahoo
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --------------------------------------------------------------------------- #
# The heating-season calendar (a calendar fact, hardcoded — no network).
# --------------------------------------------------------------------------- #
AUTUMN_BUILD_MONTHS = [9, 10, 11]          # Sep-Nov: the anticipatory "build" ahead of winter
WINTER_DRAW_MONTHS = [12, 1, 2]            # Dec-Feb: peak heating demand, inventories drawn down
HEATING_MONTHS = AUTUMN_BUILD_MONTHS + WINTER_DRAW_MONTHS   # Sep-Feb, the folklore's window
OFF_SEASON_MONTHS = [3, 4, 5, 6, 7, 8]     # Mar-Aug: the control group

# UHN (United States Heating Oil Fund, USCF) held front-month HO=F futures for retail; Yahoo's
# last printed session is 2018-09-11 — USCF wound the fund down for lack of AUM, so the vehicle
# is simply gone today. A hardcoded fact (product discontinuation), not a fetched one.
UHN_LAST_TRADE = "2018-09-11"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download HO=F raw OHLC, UHN adjusted closes and ^IRX closes; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    # HO=F — Yahoo's continuous front-month chain, raw (no back-adjustment): the splice at
    # rollover *is* the roll/contango cost a real futures holder pays.
    ho = yf.download("HO=F", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(ho.columns, pd.MultiIndex):
        ho.columns = ho.columns.get_level_values(0)
    ho[["Open", "High", "Low", "Close"]].dropna(how="all").to_csv(HO_CACHE)

    # UHN — the retail ETF wrapper (inception 2008-04-10, wound down 2018-09-11); total-return
    # adjusted closes over its whole trading life.
    uhn = yf.download("UHN", start="2008-01-01", end=end, auto_adjust=True, progress=False)
    if isinstance(uhn.columns, pd.MultiIndex):
        uhn.columns = uhn.columns.get_level_values(0)
    uhn[["Close"]].dropna().to_csv(UHN_CACHE)

    # ^IRX — 13-week T-bill discount rate (annualized %), the cash leg of the seasonal timer.
    irx = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(irx.columns, pd.MultiIndex):
        irx.columns = irx.columns.get_level_values(0)
    irx[["Close"]].dropna().to_csv(IRX_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (HO_CACHE, UHN_CACHE, IRX_CACHE))


def load_real(start: str = START, asof: str = AS_OF
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached (ho, uhn, irx) daily frames, sliced to [start, asof]. UHN keeps its own life
    (2008-04-10 -> 2018-09-11); it is never extended past its last printed session."""
    ho = pd.read_csv(HO_CACHE, index_col=0, parse_dates=True).sort_index()
    ho = ho.loc[(ho.index >= start) & (ho.index <= asof)].copy()
    uhn = pd.read_csv(UHN_CACHE, index_col=0, parse_dates=True).sort_index()
    uhn = uhn.loc[uhn.index <= asof].copy()
    irx = pd.read_csv(IRX_CACHE, index_col=0, parse_dates=True).sort_index()
    irx = irx.loc[(irx.index >= start) & (irx.index <= asof)].copy()
    return ho, uhn, irx


def _assert_continuous_monthly(idx: pd.DatetimeIndex) -> None:
    """Fail loudly if the monthly grid has interior holes."""
    months = pd.PeriodIndex(idx, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])})"
        )


def monthly_returns(daily_close: pd.Series, asof: str = AS_OF) -> pd.Series:
    """Month-end (last daily close per calendar month) simple returns, hole-free, asof-pinned.

    Drops the in-progress current month so the last bar is always a *complete* calendar month —
    never a partial one — matching the desk's as-of convention.
    """
    px = daily_close.dropna()
    px = px[px.index <= pd.Timestamp(asof)]
    me = px.resample("ME").last()
    last_complete = pd.Timestamp(asof).to_period("M")
    me = me[me.index.to_period("M") <= last_complete]
    ret = me.pct_change().dropna()
    _assert_continuous_monthly(ret.index)
    return ret


def monthly_cash_rate(irx_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Monthly T-bill cash return (annualized ^IRX % / 12), forward-filled onto ``index``."""
    me = irx_close.resample("ME").last()
    rate = (me.ffill() / 100.0) / 12.0
    return rate.reindex(index).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# Synthetic world — planted heating-season premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seasonal: float = 0.0, seed: int = 650, n_years: int = 26,
                     vol: float = 0.30) -> pd.Series:
    """Deterministic i.i.d. monthly-return world with a TUNABLE planted heating-season premium.

    Each month is drawn i.i.d. N(0, vol/sqrt(12)); on the 6 heating months (Sep-Feb) an EXTRA
    ``seasonal / 6`` is added (spread evenly across the window). ``seasonal = 0`` is the null:
    heating months are statistically identical to the rest, and the Welch split must NOT reach
    significance. Monthly PeriodIndex converted to timestamps, span ~26 years — far below the
    ~250-year pandas ns-timestamp trap.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.period_range("2000-09", periods=n, freq="M")
    months = idx.month
    base = (vol / np.sqrt(12)) * rng.standard_normal(n)
    add = np.where(np.isin(months, HEATING_MONTHS), seasonal / len(HEATING_MONTHS), 0.0)
    return pd.Series(base + add, index=idx.to_timestamp(), name="synthetic_ret")
