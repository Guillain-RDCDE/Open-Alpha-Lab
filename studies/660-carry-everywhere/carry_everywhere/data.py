"""Data layer for Study 660 — Carry-Everywhere.

Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, JFE) show that a simple carry
signal — "the return you'd earn if prices never moved" — predicts returns not just in
FX (the textbook carry trade) but in **every** major asset class: equities (dividend
yield), bonds (the term spread / roll-down), commodities (the futures roll / basis) and
currencies (the interest-rate differential). A single, low-correlation "carry
everywhere" factor is one of their headline results.

We proxy one sleeve per asset class with liquid, free-data ETFs/FX crosses — a
transparent, static, long-run classification (not a monthly re-ranked signal; the free
tape has no daily deposit-rate or futures-curve panel), exactly the same honesty
convention as sibling studies 364/612/638:

* **FX carry** — long the classic high-yielders **AUD, NZD** vs USD, short the classic
  funding currencies **JPY, CHF** vs USD (the textbook "AUD/JPY, NZD/JPY" carry pairs).
  Dollar-neutral, equal-weighted across the 4 legs.
* **Bond carry** — long **IEF** (7-10y UST, more term premium / roll-down), short
  **SHY** (1-3y UST, funds the trade near cash). This *is* the term-spread carry trade:
  duration financed at the short end.
* **Equity carry** — long **VYM** (high dividend yield), short **VUG** (low/no
  dividend, growth) — a dividend-yield tilt, the equity analogue of carry.
* **Commodity carry** — long **DBC** (Invesco's "Optimum Yield" methodology, which
  actively selects contract months to harvest backwardation / minimise contango
  drag), short **GSG** (a naive front-month roll) — the spread isolates the
  *roll-yield* component KMPV call commodity carry.

All four legs are yfinance **total-return (dividend-adjusted) closes**, no key,
cached under the study's own ``_cache/``. **BIL** (T-Bill ETF) stands in for cash — used
only to price a modest short-leg financing/borrow spread, never subtracted from the
already dollar-neutral sleeve returns (each sleeve is self-financing by construction).

Two crisis windows are hardcoded as **facts** (no network, no fitting) for the honest
"where does carry crash" tail check the brief asks for: the 2008 Lehman collapse and
the 2020 COVID crash — the two textbook "carry unwind" episodes.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
CLOSES_CACHE = os.path.join(CACHE_DIR, "cev_closes.csv")

START = "2007-06-01"        # first full month after BIL's 2007-05-30 inception
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)

# --------------------------------------------------------------------------- #
# The four carry sleeves — static long/short legs, one proxy per asset class
# --------------------------------------------------------------------------- #
FX_LONG = ["AUDUSD=X", "NZDUSD=X"]      # classic high-carry currencies vs USD
FX_SHORT = ["JPY=X", "CHF=X"]           # USD/JPY, USD/CHF: long = short the funder
BOND_LONG, BOND_SHORT = "IEF", "SHY"    # term-spread / roll-down carry
EQ_LONG, EQ_SHORT = "VYM", "VUG"        # dividend-yield carry
CMD_LONG, CMD_SHORT = "DBC", "GSG"      # roll-yield / backwardation carry
CASH = "BIL"                             # financing / borrow-spread reference only

ALL_TICKERS = sorted(set(FX_LONG + FX_SHORT + [BOND_LONG, BOND_SHORT, EQ_LONG,
                                                EQ_SHORT, CMD_LONG, CMD_SHORT, CASH]))

# --------------------------------------------------------------------------- #
# Hardcoded crisis windows — the two textbook "carry unwind" episodes, facts only.
# --------------------------------------------------------------------------- #
GFC_WINDOW = ("2008-08", "2008-11")     # Lehman (2008-09-15) and the aftermath
COVID_WINDOW = ("2020-02", "2020-04")   # the COVID crash and initial rebound
CRISIS_WINDOWS = {"GFC 2008": GFC_WINDOW, "COVID 2020": COVID_WINDOW}


def crisis_months(window: tuple[str, str]) -> pd.PeriodIndex:
    lo, hi = pd.Period(window[0], freq="M"), pd.Period(window[1], freq="M")
    return pd.period_range(lo, hi, freq="M")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2003-01-01", end: str = "2026-07-01") -> None:
    """Download total-return (dividend-adjusted) daily closes for every ticker;
    cache as one wide CSV. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = {}
    for t in ALL_TICKERS:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[t] = df["Close"].dropna()
    wide = pd.DataFrame(frames).sort_index()
    wide.to_csv(CLOSES_CACHE)


def have_real() -> bool:
    return os.path.exists(CLOSES_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide daily-close frame, sliced to [start, asof]."""
    df = pd.read_csv(CLOSES_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


def monthly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns per ticker (PeriodIndex, so no long-span Timestamp
    range is ever built). Forward-fills first: the union tape has extra rows on days
    only *some* tickers trade (e.g. FX trades on US equity holidays), which would
    otherwise leave a spurious NaN at a month-end that a holiday happens to land on."""
    m = closes.ffill().resample("ME").last()
    ret = m.pct_change().dropna(how="all")
    ret.index = ret.index.to_period("M")
    return ret


# --------------------------------------------------------------------------- #
# Synthetic world — a faithful-engine / power check, no network, deterministic.
# --------------------------------------------------------------------------- #
def synthetic_sleeves(carry_bps_mo: float = 0.0, crash_beta: float = 0.0,
                      seed: int = 660, n_months: int = 228,
                      vol_mo: float = 0.02) -> pd.DataFrame:
    """Four synthetic monthly sleeve return series (FX/BOND/EQ/CMD), independent
    Gaussian noise around a common TUNABLE planted per-sleeve carry mean, each
    sleeve loaded on a shared "crash" factor that fires in a handful of extreme
    months (so a planted crash tail can be recovered too).

    ``carry_bps_mo = 0`` and ``crash_beta = 0`` is the null world: no premium, no
    tail loading. ``n_months = 228`` (19 years) mirrors the real sample length;
    PeriodIndex throughout, so no ns-timestamp risk.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2007-06", periods=n_months, freq="M")
    sleeves = ["FX", "BOND", "EQ", "CMD"]
    # a shared latent "risk-off" factor: mostly zero, occasionally a sharp negative
    # shock (a crude stand-in for a Lehman/COVID-style carry-unwind month)
    crash = np.zeros(n_months)
    crash_idx = rng.choice(n_months, size=max(1, n_months // 40), replace=False)
    crash[crash_idx] = -rng.uniform(2.0, 5.0, size=len(crash_idx))
    out = {}
    for s in sleeves:
        eps = rng.normal(0.0, vol_mo, size=n_months)
        out[s] = carry_bps_mo / 1e4 + eps + crash_beta * crash * vol_mo
    return pd.DataFrame(out, index=idx)
