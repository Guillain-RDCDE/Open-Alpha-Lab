"""Data layer for Study 743 — Lucky-Number-8.

The claim under test has two heads, and the study tests both:

* **Price clustering.** In Chinese culture 8 (八, *ba*) sounds like 發 (*fa*,
  "wealth/prosper") and 4 (四, *si*) sounds like 死 (*si*, "death"). The folklore —
  backed by real academic work on mainland A-shares (Brown, Chua & Mitchell 2002;
  Brown & Mitchell 2008) — is that superstitious order flow makes prices *cluster* on
  8-ending values and *avoid* 4-ending ones. We test whether that fingerprint survives
  onto the US tape: does a basket of **US-listed Chinese ADRs** show an 8-excess (and a
  4-deficit) in the trailing cent digit of its raw closing price, *relative to a matched
  basket of US-domestic large-caps* that shares the same $0.01 tick but none of the
  culture?

* **A superstition premium around 8/8.** The single most auspicious calendar date is
  the **8th of August** (8/8 — doubly "prosperous"; 2008-08-08 was chosen for the
  Beijing Olympics opening for exactly this reason). The folklore says Chinese equities
  get a feel-good/buying bump around it. We run an event study on a China large-cap ETF
  (`FXI`) minus a broad emerging-markets benchmark (`EEM`), one lucky date per year.

Three ingredients:

* **The lucky-date calendar, hardcoded.** ``LUCKY_DATES`` = the 8th of August of every
  year `FXI` can cover (2005→2025). Each is a single, independent, non-overlapping
  calendar event. 8/8 is *calendar-known* — nobody needs private information to know it
  is coming — so a trade around it carries **no look-ahead at all** (unlike an earnings
  or crash date). The snap rule (day(-1) = last session strictly before Aug 8, day(0) =
  first session on/after) is the single documented execution convention.

* **Real tape.** yfinance daily bars, cached under ``_cache/``. We keep BOTH the **raw
  Close** (``auto_adjust=False`` — the actual last traded price, whose trailing digit is
  the object of the clustering test; a split/dividend-*adjusted* close has a meaningless
  trailing digit) AND the **Adj Close** (total-return, used for the event-study returns).
  `FXI`/`EEM` for the event study; a China-ADR basket + a US-control basket for the
  clustering test.

* **Synthetic worlds.** (a) A seeded paired (asset, benchmark) log-return world with a
  TUNABLE planted 8/8 bump — the event-study positive control. (b) A seeded trailing-
  digit generator with a TUNABLE planted 8-excess — the clustering positive control.
  Both null (bump/excess = 0) worlds must NOT fire.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication
CHINA_PROXY = "FXI"         # iShares China Large-Cap ETF (inception 2004-10)
EM_BENCHMARK = "EEM"        # iShares MSCI Emerging Markets — the fair China counterfactual

# --------------------------------------------------------------------------- #
# The lucky-date calendar, hardcoded: the 8th of August (8/8), the single most
# auspicious date in Chinese numerology (八八, doubly "prosperous" — the reason the
# 2008 Beijing Olympics opened at 8:08 pm on 2008-08-08). One event per year, each an
# independent, non-overlapping calendar date. 8/8 is calendar-known: a trade around it
# needs NO private information and carries zero look-ahead. We start in 2005 (the first
# 8th of August after FXI's 2004-10 inception) and run to 2025.
# 2008-08-08 (the triple-8 Olympics opening) is flagged for the notebook's callout.
# --------------------------------------------------------------------------- #
LUCKY_DATES: list[tuple[int, str, str]] = [
    (2005, "2005-08-08", ""),
    (2006, "2006-08-08", ""),
    (2007, "2007-08-08", ""),
    (2008, "2008-08-08", "triple-8: Beijing Olympics opening (08/08/08, 8:08pm)"),
    (2009, "2009-08-08", ""),
    (2010, "2010-08-08", ""),
    (2011, "2011-08-08", ""),
    (2012, "2012-08-08", ""),
    (2013, "2013-08-08", ""),
    (2014, "2014-08-08", ""),
    (2015, "2015-08-08", ""),
    (2016, "2016-08-08", ""),
    (2017, "2017-08-08", ""),
    (2018, "2018-08-08", ""),
    (2019, "2019-08-08", ""),
    (2020, "2020-08-08", ""),
    (2021, "2021-08-08", ""),
    (2022, "2022-08-08", ""),
    (2023, "2023-08-08", ""),
    (2024, "2024-08-08", ""),
    (2025, "2025-08-08", ""),
]

# --------------------------------------------------------------------------- #
# Clustering-test baskets. US-listed Chinese ADRs (the "culture" group) vs US-domestic
# large-caps (the "control" group). BOTH trade on US venues with a uniform $0.01 tick,
# so the trailing cent digit is uniform under the null for either group — the round-
# number (0/5) preference is common to both and cancels in the China-minus-control
# contrast. Only a CHINA-SPECIFIC 8-excess would support the superstition claim.
# Raw closing prices (not adjusted) are what carry the meaningful trailing digit.
# --------------------------------------------------------------------------- #
CHINA_ADRS: list[str] = [
    "BABA", "JD", "PDD", "BIDU", "NTES", "TCOM", "NIO", "LI",
    "XPEV", "BILI", "TME", "VIPS", "YUMC", "ZTO", "HTHT",
]
CONTROL_US: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "KO",
    "DIS", "INTC", "CSCO", "PFE", "WMT", "HD", "XOM",
]


def all_tickers() -> list[str]:
    """Every distinct ticker the study needs."""
    return [CHINA_PROXY, EM_BENCHMARK] + CHINA_ADRS + CONTROL_US


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"lucky8_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-01-01", end: str = "2026-07-01") -> None:
    """Download raw Close AND Adj Close for every ticker; cache them.

    ``auto_adjust=False`` so we keep the **raw** last-traded Close (its trailing cent
    digit is the object of the clustering test) alongside the total-return **Adj Close**
    (used for the event-study abnormal returns). Network; run once.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        d = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols = {}
        if "Close" in d:
            cols["close"] = d["Close"]
        if "Adj Close" in d:
            cols["adj"] = d["Adj Close"]
        out = pd.DataFrame(cols).dropna()
        out.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached {ticker: DataFrame[['close','adj']]}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        out[t] = df[df.index <= pd.Timestamp(asof)]
    return out


def adj_close(prices: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    """Total-return (adjusted) close for the event study."""
    return prices[ticker]["adj"]


def raw_close(prices: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    """Raw last-traded close for the trailing-digit clustering test."""
    return prices[ticker]["close"]


# --------------------------------------------------------------------------- #
# Synthetic world A — planted 8/8 bump (the event-study positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 743, n_events: int = 21,
                    n_days: int = 5400, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted bump.

    Correlated (rho ~ 0.85, like a China ETF vs a broad EM benchmark) zero-mean noise;
    on each synthetic "lucky day" (every ``spacing``-th business day) the asset gets an
    EXTRA ``bump`` log-return the benchmark does not. ``bump = 0`` is the null world.
    Integer index (positions 0..n_days), no calendar dates generated (avoids the pandas
    250-year ns-timestamp trap). Returns (asset_logret, bench_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.85
    common = rng.normal(0.0, 0.012, n_days)
    idio_a = rng.normal(0.0, 0.012, n_days)
    idio_b = rng.normal(0.0, 0.012, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    event_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in event_pos:
        a[p] += bump

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), event_pos


# --------------------------------------------------------------------------- #
# Synthetic world B — planted trailing-digit 8-excess (the clustering positive control)
# --------------------------------------------------------------------------- #
def synthetic_digits(excess: float = 0.0, seed: int = 743, n: int = 40000) -> np.ndarray:
    """Deterministic trailing-digit sample with a TUNABLE 8-excess.

    Baseline is uniform over 0..9 (probability 0.1 each). ``excess`` shifts probability
    mass onto digit 8 (and, symmetrically, off digit 4), so ``excess = 0`` is the null
    (uniform) world the chi-square / two-proportion detector must NOT flag. Returns an
    integer array of ``n`` trailing digits.
    """
    rng = np.random.default_rng(seed)
    p = np.full(10, 0.1)
    p[8] += excess
    p[4] -= excess
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return rng.choice(10, size=n, p=p)
