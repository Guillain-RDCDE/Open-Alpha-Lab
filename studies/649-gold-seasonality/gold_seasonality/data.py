"""Data layer for Study 649 — Gold Seasonality.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily adjusted closes of **GLD** (SPDR Gold Shares, the tradable, physically
  backed gold ETF, inception 2004-11-18) and the **13-week T-bill discount yield (^IRX)**, used
  only to build an excess-of-cash return for the timer/buy-and-hold Sharpe race. GLD holds
  allocated bullion (no futures roll, no dividends to adjust for) — its daily close *is* a spot
  gold-price proxy anyone could actually have bought. Both from yfinance (no key), cached as CSV
  under the study's own ``_cache/``.

* **Synthetic world.** A deterministic, seeded i.i.d. monthly-return series with a TUNABLE
  planted September premium / summer (May-Aug) discount pair (knob ``seasonal``). ``seasonal =
  0`` is the null world — every calendar month is statistically identical; the Welch/HAC
  machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
GLD_CACHE = os.path.join(CACHE_DIR, "gsn_gld.csv")
IRX_CACHE = os.path.join(CACHE_DIR, "gsn_irx.csv")

# GLD's actual inception (2004-11-18) sets the start; the first COMPLETE calendar month on tape
# is December 2004, so the monthly-return series (a diff of month-end closes) begins there.
START = "2004-11-18"
AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)

# The folklore's claimed calendar. "Strong" = gold's textbook "best month" (September: Indian
# wedding-season + pre-Diwali physical demand, Northern-hemisphere jewellery restocking ahead of
# year-end). "Weak"/lull = the quiet Northern-hemisphere summer (low physical/jewellery/ETF
# demand between the spring Akshaya Tritiya buying and the autumn wedding season). Both windows
# are stated by the claim itself, not fitted to this sample -- see docs/references.md.
STRONG_MONTHS = (9,)              # September -- "gold's best month"
SUMMER_MONTHS = (5, 6, 7, 8)      # the claimed summer lull

# The 2013 gold crash (2013-04-12/15, gold's worst two-day drop in 30 years, spot -13% in two
# sessions) is the desk's own justified, externally-dated era split for gold: the end of the
# 2001-2012 bull "supercycle" and the start of the decade-long range-bound regime, chosen because
# it is a documented historical event, not because it flatters either side of the split.
ERA_SPLIT = "2013-04-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-10-01", end: str = "2026-07-01") -> None:
    """Download GLD adjusted closes and the ^IRX 13-week T-bill yield; cache. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    gld = yf.download("GLD", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(gld.columns, pd.MultiIndex):
        gld.columns = gld.columns.get_level_values(0)
    gld[["Close"]].dropna().to_csv(GLD_CACHE)

    irx = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(irx.columns, pd.MultiIndex):
        irx.columns = irx.columns.get_level_values(0)
    irx[["Close"]].dropna().to_csv(IRX_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (GLD_CACHE, IRX_CACHE))


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.Series, pd.Series]:
    """Cached (GLD close, ^IRX close) daily series, sliced to [start, asof]."""
    gld = pd.read_csv(GLD_CACHE, index_col=0, parse_dates=True).sort_index()["Close"].astype(float)
    irx = pd.read_csv(IRX_CACHE, index_col=0, parse_dates=True).sort_index()["Close"].astype(float)
    gld = gld.loc[(gld.index >= start) & (gld.index <= asof)]
    irx = irx.loc[(irx.index >= start) & (irx.index <= asof)]
    return gld, irx


# --------------------------------------------------------------------------- #
# Synthetic world -- planted September premium / summer discount (positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seasonal: float = 0.0, seed: int = 649, n_years: int = 30,
                    vol: float = 0.17,
                    strong: tuple[int, ...] = STRONG_MONTHS,
                    weak: tuple[int, ...] = SUMMER_MONTHS,
                    ) -> pd.DataFrame:
    """Deterministic i.i.d. monthly gold-return world with a TUNABLE planted Sept/summer calendar.

    i.i.d. monthly log returns at annualised vol ``vol`` (~GLD's own realised monthly vol), plus a
    ``seasonal`` premium added to ``strong`` months and an equal-and-opposite discount spread over
    ``weak`` months. ``seasonal = 0`` is the null world: every month is statistically identical,
    and the Welch/HAC machinery must NOT reach significance.

    Monthly PeriodIndex grid (n_years * 12 points, far below the pandas ns-timestamp Timestamp
    trap even before conversion) -> converted to a month-end DatetimeIndex for downstream reuse.
    Returns a one-column ("ret") frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    pidx = pd.period_range("1996-01", periods=n, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    months = idx.month

    base = (vol / np.sqrt(12)) * rng.standard_normal(n)
    add = np.where(np.isin(months, strong), seasonal,
                   np.where(np.isin(months, weak), -seasonal / len(weak), 0.0))
    return pd.DataFrame({"ret": base + add}, index=idx)
