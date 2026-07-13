"""Data layer for Study 732 — Tour-de-France-Effect.

The claim under test: French equities enjoy a **July "Grande Boucle" seasonal** — a
feel-good, summer-holiday bump while the whole country watches the Tour de France roll
through the countryside toward the Champs-Elysees. It is the summer-holiday cousin of
the sports-sentiment folklore (Edmans-Garcia-Norli's real World-Cup elimination effect),
recast as a three-week *calendar window* on the CAC / EWQ rather than a surprise result.

The honest twist that makes this an interesting test: the Tour de France runs in **July**
— smack in the middle of the "Sell in May and go away" window (May->October), the single
best-documented seasonal *weakness* in equities. So a naive "buy French stocks during the
Tour" seasonal is running straight into the summer doldrums, and any France-specific
"bump" has to be measured *against Europe*, not in absolute terms, or it just re-discovers
pan-European summer beta. That raw-vs-abnormal split is the study's third axis.

Three ingredients:

* **The calendar, hardcoded.** Every Tour de France 1996->2025 (30 editions), its Grand
  Depart date and its final-stage date. Two named quirks: **2020** was pushed by COVID-19
  from July to **Aug 29 -> Sep 20** (a natural "does the effect follow the RACE or the
  CALENDAR MONTH?" probe), and **2024** finished in Nice, not Paris, because of the Paris
  Olympics (no market consequence). Source: Wikipedia "List of Tour de France editions" /
  the individual per-year race pages for the exact Grand Depart and final-stage dates.
* **The tradable French instruments.** ``EWQ`` (iShares MSCI France, US-listed,
  total-return) is the vehicle a retail believer could actually buy; ``FCHI`` (the CAC 40
  *price* index, no dividends -- labelled price-only) is a longer-history cross-check on
  the raw seasonal. Because the Tour dates are public a *year* in advance, this is a
  **calendar-known** window: no surprise, no information lag, no un-tradable weekend jump
  to strip out (contrast study 708's Saturday-night Eurovision result).
* **A Europe benchmark**, ``VGK`` (Vanguard FTSE Europe -- spans euro AND non-euro
  Europe), for the *abnormal* (France-minus-Europe) measurement that separates a genuine
  French-sentiment effect from ordinary summer beta. VGK's inception (2005-03-10) is a
  hard floor on how far back the abnormal test can reach; the raw EWQ seasonal reaches to
  EWQ's 1996 inception, and the CAC price cross-check to 1996 as well (FCHI to 1990, but
  we align the CAC window to the EWQ sample for a like-for-like count).
* **Synthetic world.** A deterministic, seeded pair of (France, Europe) log-return series
  with a TUNABLE planted "July seasonal bump" on a scheduled annual window. ``bump = 0``
  is the null world; the one-sample-t machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-13)
FRANCE_ETF = "EWQ"          # iShares MSCI France -- US-listed, total-return, tradable
CAC_INDEX = "^FCHI"         # CAC 40 -- PRICE index (no dividends), long-history cross-check
EUROPE_BENCHMARK = "VGK"    # Vanguard FTSE Europe -- euro AND non-euro Europe

# --------------------------------------------------------------------------- #
# The Tour de France calendar, hardcoded: year, Grand Depart date, final-stage date.
# Held every July since the modern era; 2020 shifted to Aug-Sep (COVID-19). Source:
# Wikipedia "List of Tour de France editions" and each year's "<YEAR> Tour de France"
# page for the exact Grand Depart / final-stage (Champs-Elysees, except 2024 Nice) dates.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, grand_depart, final_stage,  note
    (1996, "1996-06-29", "1996-07-21", ""),
    (1997, "1997-07-05", "1997-07-27", ""),
    (1998, "1998-07-11", "1998-08-02", ""),
    (1999, "1999-07-03", "1999-07-25", ""),
    (2000, "2000-07-01", "2000-07-23", ""),
    (2001, "2001-07-07", "2001-07-29", ""),
    (2002, "2002-07-06", "2002-07-28", ""),
    (2003, "2003-07-05", "2003-07-27", ""),
    (2004, "2004-07-03", "2004-07-25", ""),
    (2005, "2005-07-02", "2005-07-24", ""),
    (2006, "2006-07-01", "2006-07-23", ""),
    (2007, "2007-07-07", "2007-07-29", ""),
    (2008, "2008-07-05", "2008-07-27", ""),
    (2009, "2009-07-04", "2009-07-26", ""),
    (2010, "2010-07-03", "2010-07-25", ""),
    (2011, "2011-07-02", "2011-07-24", ""),
    (2012, "2012-06-30", "2012-07-22", ""),
    (2013, "2013-06-29", "2013-07-21", ""),
    (2014, "2014-07-05", "2014-07-27", ""),
    (2015, "2015-07-04", "2015-07-26", ""),
    (2016, "2016-07-02", "2016-07-24", ""),
    (2017, "2017-07-01", "2017-07-23", ""),
    (2018, "2018-07-07", "2018-07-29", ""),
    (2019, "2019-07-06", "2019-07-28", ""),
    (2020, "2020-08-29", "2020-09-20", "COVID-19: shifted July->Aug/Sep"),
    (2021, "2021-06-26", "2021-07-18", ""),
    (2022, "2022-07-01", "2022-07-24", ""),
    (2023, "2023-07-01", "2023-07-23", ""),
    (2024, "2024-06-29", "2024-07-21", "finished in Nice (Paris Olympics)"),
    (2025, "2025-07-05", "2025-07-27", ""),
]


def all_tickers() -> list[str]:
    """Every distinct ticker this study needs: France ETF, CAC price index, Europe bench."""
    return [FRANCE_ETF, CAC_INDEX, EUROPE_BENCHMARK]


def _slug(ticker: str) -> str:
    return ticker.lower().lstrip("^")


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"tdf_{_slug(ticker)}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1990-01-01", end: str = "2026-07-01") -> None:
    """Download daily closes for every ticker and cache them.

    ``EWQ`` and ``VGK`` are equity ETFs -> ``auto_adjust=True`` (total-return, dividends
    reinvested) is the honest comparison. ``^FCHI`` (CAC 40) is a **price index** with no
    dividend adjustment concept; auto_adjust is a no-op on it and it is *labelled
    price-only* everywhere it appears (it understates true French equity returns and is
    used only as a long-history cross-check on the raw seasonal, never mixed into the
    total-return abnormal test).
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
    """Cached {ticker: close Series}, each sliced to <= asof.

    EWQ/VGK are total-return; FCHI is price-only (labelled). Keyed by the raw ticker
    string (``"EWQ"``, ``"^FCHI"``, ``"VGK"``).
    """
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted July seasonal bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 732, n_events: int = 30,
                    win: int = 16, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[tuple[int, int]]]:
    """Deterministic paired (France, Europe) log-return world with a planted seasonal.

    Both series are correlated (rho ~ 0.85, like a single-country ETF vs its regional
    benchmark) zero-mean noise; during a scheduled synthetic "Tour window" (``win``
    sessions every ``spacing``-th business day) the France leg earns a small EXTRA
    per-day ``bump`` log-return that the Europe leg does not. ``bump = 0`` is the null
    world -- Tour windows statistically identical to the rest of the year.

    Business-day integer index (positions 0..n), far below the ns-timestamp trap.
    Returns (france_logret, europe_logret, [(entry, exit), ...] window positions).
    """
    rng = np.random.default_rng(seed)
    n_days = spacing * (n_events + 1)
    rho = 0.85
    common = rng.normal(0.0, 0.011, n_days)
    idio_f = rng.normal(0.0, 0.008, n_days)
    idio_e = rng.normal(0.0, 0.008, n_days)
    f = rho * common + np.sqrt(1 - rho**2) * idio_f
    e = rho * common + np.sqrt(1 - rho**2) * idio_e

    windows = []
    for k in range(1, n_events + 1):
        entry = k * spacing
        exit_ = entry + win
        if exit_ >= n_days:
            break
        f[entry + 1:exit_ + 1] += bump   # extra France drift during the window
        windows.append((entry, exit_))

    idx = pd.RangeIndex(n_days)
    return pd.Series(f, index=idx), pd.Series(e, index=idx), windows
