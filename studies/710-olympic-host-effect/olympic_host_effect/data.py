"""Data layer for Study 710 — Olympic-Host-Effect.

The claim: hosting the Summer Olympics gives the HOST country's stock market a lift —
national pride, an infrastructure boom, a tourism bump — visible in the host's equity
market around the Games. Two ingredients:

* **Real tape.** Daily total-return (adjusted) closes for six single-country ETFs that
  each track a Summer Olympics HOST — EWA (Australia, Sydney 2000), FXI (China,
  Beijing 2008), EWU (United Kingdom, London 2012), EWZ (Brazil, Rio 2016), EWJ (Japan,
  Tokyo 2020/21) and EWQ (France, Paris 2024) — plus a world-market benchmark, all from
  yfinance (no key), cached as CSV under the study's own ``_cache/``.

  **Athens 2004 is in the hardcoded host table but carries no ticker.** No single-country
  Greece ETF existed in 2004 — Global X's GREK, the first one, launched 2011-12-08, seven
  years after the Athens Games closed. Rather than force a post-hoc proxy onto a market
  that had no listed vehicle at the time, we name the gap and run the real-tape panel on
  the **n = 6** hosts that do have a contemporaneous ETF. This is exactly the "genuinely
  no free data" case METHODOLOGY.md asks studies to name rather than paper over — it
  shrinks an already-tiny sample by one, and we say so on every axis.

  **The world benchmark is the S&P 500 (^GSPC), not URTH/ACWI — a named substitution.**
  iShares' MSCI World ETF (URTH) launched 2012-01-04 and the MSCI ACWI ETF (ACWI)
  launched 2008-03-26 — both postdate the Sydney 2000 window outright, and ACWI's
  inception sits *inside* the Beijing 2008 pre-Games window. Neither covers the full
  2000->2024 sample the claim spans, so a single benchmark that DOES (^GSPC, continuous
  since 1927, no survivorship) is used throughout instead — a defensible but imperfect
  proxy for "the world" (US equities, not global-cap-weighted; ^GSPC is a **price index**
  with no dividends, so the comparison is host **total return** vs benchmark **price-only**
  return, named here and in every table it appears in).

* **The hardcoded host calendar.** Seven editions, opening/closing dates from the IOC's
  own results archive (facts, no network) — host cities are awarded 7+ years ahead, so
  every window below involves zero look-ahead.

* **Synthetic world.** A deterministic, seeded draw of ``n`` abnormal returns from
  Normal(effect, sd), sd calibrated to the real panel's observed cross-host dispersion.
  This is a **faithful-engine / power check only**: it proves the one-sample-t detector
  is unbiased (the null must not fire) and quantifies how large a *true* effect would
  need to be to clear the desk's t >= 2 bar at n = 6 — never cited as market evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07-10)
FETCH_START = "1998-01-01"    # comfortably before the earliest window (Sydney, Mar 2000)
FETCH_END = "2026-07-01"

WINDOW_PRE_MONTHS = 6         # the claim's window: [-6mo .. +2mo] around the Games
WINDOW_POST_MONTHS = 2

BENCH_TICKER = "^GSPC"        # named substitute for URTH/ACWI — see module docstring


@dataclass(frozen=True)
class Host:
    year: int
    city: str
    country: str
    ticker: str | None       # None = no contemporaneous single-country ETF existed
    games_start: str          # opening ceremony (ISO date)
    games_end: str             # closing ceremony (ISO date)
    note: str                  # named confounder / data quirk, "" if none


# --------------------------------------------------------------------------- #
# Hardcoded Summer Olympics host calendar, 2000 -> 2024.
# Source: IOC Olympic Games results archive (olympics.com/en/olympic-games); opening/
# closing-ceremony dates. Host cities are announced by IOC vote 7-9 years ahead of the
# Games, so a host-country flag involves zero look-ahead by construction.
# --------------------------------------------------------------------------- #
HOSTS: tuple[Host, ...] = (
    Host(2000, "Sydney", "Australia", "EWA", "2000-09-15", "2000-10-01", ""),
    Host(2004, "Athens", "Greece", None, "2004-08-13", "2004-08-29",
         "no contemporaneous Greece ETF (GREK launched 2011-12-08) -- excluded from the "
         "real-tape panel, n effectively 6 not 7"),
    Host(2008, "Beijing", "China", "FXI", "2008-08-08", "2008-08-24",
         "sits inside the Global Financial Crisis (Lehman failed 2008-09-15, inside the "
         "post-Games leg of this very window) -- a severe, named confounder"),
    Host(2012, "London", "United Kingdom", "EWU", "2012-07-27", "2012-08-12", ""),
    Host(2016, "Rio de Janeiro", "Brazil", "EWZ", "2016-08-05", "2016-08-21",
         "coincides with Brazil's 2016 rebound off the commodity-bust trough (Bovespa "
         "bottomed Jan 2016) -- plausibly the dominant driver, not the Games"),
    Host(2021, "Tokyo", "Japan", "EWJ", "2021-07-23", "2021-08-08",
         "the 'Tokyo 2020' Games, held in 2021 after a COVID-19 postponement"),
    Host(2024, "Paris", "France", "EWQ", "2024-07-26", "2024-08-11", ""),
)

REAL_HOSTS: tuple[Host, ...] = tuple(h for h in HOSTS if h.ticker is not None)


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("^", "idx_")
    return os.path.join(CACHE_DIR, f"ohe_{safe}.csv")


def tickers() -> list[str]:
    """All tickers this study needs: the real hosts' ETFs plus the benchmark."""
    return sorted({h.ticker for h in REAL_HOSTS} | {BENCH_TICKER})


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = FETCH_START, end: str = FETCH_END) -> None:
    """Download total-return (adjusted) daily closes for every host ETF + the
    benchmark; cache each as its own CSV. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in tickers():
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: Close series} for every host ETF + benchmark, sliced to asof."""
    out = {}
    hi = pd.Timestamp(asof)
    for t in tickers():
        s = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s.loc[s.index <= hi]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- a calibrated null / power check on the SAME statistic
# (a one-sample t across n abnormal returns), not a simulated price tape.
# --------------------------------------------------------------------------- #
CALIBRATED_SD = 36.16   # cross-host sd of the real n=6 abnormal returns, %  (see results.md)


def synthetic_world(effect: float = 0.0, seed: int = 710, n: int = 6,
                     sd: float = CALIBRATED_SD) -> np.ndarray:
    """``n`` draws from Normal(effect, sd) -- a deterministic, seeded stand-in for the
    "n abnormal host-vs-benchmark returns" the real study measures, calibrated to the
    REAL panel's own dispersion (36.16 pp). ``effect = 0`` is the null world: the
    one-sample-t detector must not fire on it. A nonzero ``effect`` is a planted average
    abnormal return (percentage points), used only to check the detector's power at the
    real study's tiny n -- never as market evidence.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=effect, scale=sd, size=n)
