"""Data layer for Study 719 — Met-Gala-Luxury.

The claim under test: the big European luxury houses — LVMH, Kering, Hermes, Richemont
— get a "spotlight bump" around the Met Gala, fashion's biggest night, held on the
first Monday in May. Every May the red carpet is wall-to-wall with these brands; the
folklore (a fixture of markets-meets-culture pieces) says the free global attention
lifts the luxury complex in the days around the event.

Three ingredients:

* **The calendar, hardcoded.** Every Met Gala (Costume Institute Benefit) 2000->2025,
  its exact date. The event settled onto the **first Monday in May from 2005 onward**;
  before that its date wandered, and three years have no gala at all (rich texture we
  keep, not smooth over):
    - 2000 — NO gala (the planned Chanel exhibition was cancelled).
    - 2002 — NO gala (cancelled in the aftermath of 9/11; the calendar had just moved
      from December to April in 2001).
    - 2020 — NO gala (COVID-19; the planned 2020-05-04 edition was called off).
    - 2021 — held **2021-09-13** (a Monday, but in SEPTEMBER, not May) as the make-up
      for the cancelled 2020 night — a named quirk, like study 708's UK-hosts-for-
      Ukraine year.
  Source: Wikipedia "Met Gala" and the Costume Institute benefit's per-year coverage,
  cross-checked for the exact Monday date.
* **A luxury basket.** The four largest listed European luxury houses, each on its home
  exchange: ``MC.PA`` (LVMH, Euronext Paris), ``KER.PA`` (Kering, Paris), ``RMS.PA``
  (Hermes, Paris), ``CFR.SW`` (Richemont, SIX Swiss). Equal-weighted daily returns of
  whichever names trade that day (all four cover the entire tested window).
* **A Europe benchmark**, ``VGK`` (Vanguard FTSE Europe — spans euro AND non-euro
  Europe, so it is a fair counterfactual for a France+Switzerland basket, exactly as in
  study 708). VGK's inception (**2005-03-10**) is the hard floor on how far back the
  test can reach — which conveniently coincides with the first year the Met Gala firmly
  adopted the first-Monday-in-May slot, so the tested sample (2005->2025) is precisely
  the "modern, calendar-regular" Met Gala.
* **Synthetic world.** A deterministic, seeded pair of (basket, benchmark) log-return
  series with a TUNABLE planted "spotlight bump" on a synthetic event calendar.
  ``bump = 0`` is the null world; the one-sample-t machinery must not manufacture
  significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-12)
EUROPE_BENCHMARK = "VGK"    # Vanguard FTSE Europe -- spans euro AND non-euro Europe

# --------------------------------------------------------------------------- #
# The luxury basket: ticker -> house name. Equal-weighted; every name covers the
# whole tested window (2005->2025), so no survivorship backfill is ever needed.
# --------------------------------------------------------------------------- #
LUXURY: dict[str, str] = {
    "MC.PA":  "LVMH",       # Moet Hennessy Louis Vuitton, Euronext Paris
    "KER.PA": "Kering",     # Kering (ex-PPR), Euronext Paris
    "RMS.PA": "Hermes",     # Hermes International, Euronext Paris
    "CFR.SW": "Richemont",  # Cie Financiere Richemont, SIX Swiss
}

# --------------------------------------------------------------------------- #
# The Met Gala calendar, hardcoded: year, date (Monday) or None when no gala was
# held that year. The date is the first Monday in May from 2005 onward; 2001/2003/2004
# predate that convention (and predate VGK, so they never enter the tested sample);
# 2000/2002/2020 have no gala; 2021 was held in September.
# Source: Wikipedia "Met Gala" + Costume Institute per-year coverage, dates cross-checked.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, gala_date (Monday) or None, note
    (2000, None,          "no gala (planned Chanel exhibition cancelled)"),
    (2001, "2001-04-23",  "April (calendar just moved from December); pre-VGK"),
    (2002, None,          "no gala (cancelled after 9/11)"),
    (2003, "2003-04-28",  "April; pre-VGK"),
    (2004, "2004-04-26",  "April; pre-VGK"),
    (2005, "2005-05-02",  "first firm 'first Monday in May'"),
    (2006, "2006-05-01",  ""),
    (2007, "2007-05-07",  ""),
    (2008, "2008-05-05",  ""),
    (2009, "2009-05-04",  ""),
    (2010, "2010-05-03",  ""),
    (2011, "2011-05-02",  ""),
    (2012, "2012-05-07",  ""),
    (2013, "2013-05-06",  ""),
    (2014, "2014-05-05",  ""),
    (2015, "2015-05-04",  ""),
    (2016, "2016-05-02",  ""),
    (2017, "2017-05-01",  ""),
    (2018, "2018-05-07",  ""),
    (2019, "2019-05-06",  ""),
    (2020, None,          "no gala (COVID-19; planned 2020-05-04 cancelled)"),
    (2021, "2021-09-13",  "held in SEPTEMBER as the make-up for 2020 (named quirk)"),
    (2022, "2022-05-02",  ""),
    (2023, "2023-05-01",  ""),
    (2024, "2024-05-06",  ""),
    (2025, "2025-05-05",  ""),
]


def all_tickers() -> list[str]:
    """Every distinct ticker this study needs: the four luxury names plus the benchmark."""
    return list(LUXURY.keys()) + [EUROPE_BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"metgala_{ticker.lower().replace('.', '_')}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2003-01-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for every ticker; cache them.

    ``auto_adjust=True`` -- these are equities/ETFs, so total-return (dividends
    reinvested) is the honest, like-for-like comparison for both the luxury basket and
    the VGK benchmark.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        d.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


def basket_close(prices: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted luxury-basket total-return index from the four names.

    Built by averaging the four names' daily simple returns and compounding — a
    rebalanced equal-weight basket. We require **all four names to trade** on a day for it
    to count (``dropna(how="any")``): a day where one home exchange is shut (e.g. May 1
    Labour Day on Euronext Paris) would otherwise let a single name stand in for the whole
    basket and distort the anchor. The index is arbitrary-based (starts at 1.0 on the
    first common date); only its returns are ever used.
    """
    rets = pd.DataFrame({t: prices[t].pct_change() for t in LUXURY}).dropna(how="any")
    daily = rets.mean(axis=1)
    return (1.0 + daily).cumprod()


# --------------------------------------------------------------------------- #
# Synthetic world -- planted spotlight bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 719, n_events: int = 20,
                    n_days: int = 6000, spacing: int = 260,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (basket, benchmark) log-return world with a planted bump.

    Both series are correlated (rho ~ 0.80, like a luxury basket vs a regional Europe
    benchmark) zero-mean noise; on a synthetic "event day" (every ``spacing``-th
    business day) the basket gets an EXTRA ``bump`` log-return (in addition to its normal
    correlated draw) while the benchmark does not. ``bump = 0`` is the null world --
    event days statistically identical to the rest.

    Business-day integer index (positions 0..n_days), far below the 250-year
    ns-timestamp trap (no calendar dates are generated at all here).
    Returns (basket_logret, bench_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.80
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.011, n_days)
    idio_b = rng.normal(0.0, 0.011, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    event_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in event_pos:
        a[p] += bump

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), event_pos
