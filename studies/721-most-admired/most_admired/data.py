"""Data layer for Study 721 — Most-Admired (Fortune's World's Most Admired Companies).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of the list's perennial **All-Stars** — the
  mega-caps Fortune's survey has crowned "most admired" year after year (Apple has topped the
  overall ranking every year 2008-2024). For each row we record the *currently-traded ticker*
  and the approximate year the firm first became a fixture near the top of the list, so a
  **publication-lagged** variant can own a name only *after* Fortune first crowns it (no
  look-ahead). From yfinance **month-end** adjusted closes we build an equal-weight admired
  book and benchmark it against the market (``SPY``). Cached under
  ``_cache/admired_prices.csv`` (a wide CSV, one column per ticker).

  The table is the honest part of this study, and its bias is named **loudly** and on the
  Signal axis: a *current* most-admired list is **look-ahead selection**. A firm lands near
  the top of the survey only after a long run of good news and a soaring stock — Apple,
  Microsoft, Nvidia are admired *because* they already won. So the admired book is
  survivorship-and-hindsight-tilted **toward** past winners, and any raw out-performance is a
  quality/tech **factor beta** and a selection artefact before it is an "admiration premium."

* **Spurned proxy (labelled).** The contrarian claim (Statman-Fisher-Anginer 2008) is that
  the *least*-admired stocks out-earn the most-admired. The genuine bottom of the survey is
  dominated by firms that later **delisted** (airlines, retailers, autos in bankruptcy), so a
  priced "spurned" book is unavoidably survivor-only. We hardcode a **small, cited, survivor**
  set of perennially low-reputation names as a **labelled proxy** (never a clean feed) and use
  it only for a directional long/short read, with the survivorship caveat travelling with it.

* **Synthetic.** A deterministic, fixed-seed generator that plants a controllable monthly
  **admiration premium** into an otherwise market-like book. The positive control: with the
  ``edge`` knob at 0 the HAC inference must NOT manufacture a significant premium; with a
  large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "admired_prices.csv")

# ---------------------------------------------------------------------------- #
# The hardcoded admired table — Fortune "World's Most Admired Companies" All-Stars.
#
# Each row: (ticker, first_year, label)
#   * ticker      — the symbol carrying the price on yfinance today.
#   * first_year  — approximate first year the firm became a fixture at/near the TOP of the
#                   overall "All-Stars" ranking. Used by the publication-lagged variant, which
#                   owns a name only from Feb of first_year (the survey prints late January),
#                   so the lagged book has NO look-ahead into which firms would later be admired.
#   * label       — for the charts.
#
# Source: Fortune, "World's Most Admired Companies" annual All-Stars (see docs/references.md).
# The set is the survey's perennial top tier; it is deliberately transparent and mega-cap.
# ---------------------------------------------------------------------------- #
ADMIRED = [
    ("AAPL",  2008, "Apple — #1 overall 2008-2024"),
    ("AMZN",  2010, "Amazon"),
    ("MSFT",  2010, "Microsoft"),
    ("BRK-B", 2010, "Berkshire Hathaway"),
    ("DIS",   2009, "Walt Disney"),
    ("GOOGL", 2011, "Alphabet (Google)"),
    ("SBUX",  2011, "Starbucks"),
    ("NKE",   2011, "Nike"),
    ("COST",  2012, "Costco"),
    ("JPM",   2012, "JPMorgan Chase"),
    ("NFLX",  2018, "Netflix"),
    ("CRM",   2016, "Salesforce"),
    ("AXP",   2010, "American Express"),
    ("FDX",   2010, "FedEx"),
    ("NVDA",  2023, "Nvidia (recent All-Star)"),
]

# A small, cited, SURVIVOR set of perennially low-reputation ("spurned") large caps — a
# LABELLED PROXY for the bottom of the survey, never a clean feed. The true least-admired
# cohort is survivorship-riddled (many delisted in bankruptcy), named in DELISTED below.
SPURNED = [
    ("T",   "AT&T — perennial low-reputation telecom"),
    ("F",   "Ford — chronically out-of-favour legacy auto"),
    ("KHC", "Kraft Heinz — post-2015 value trap / brand impairments"),
    ("M",   "Macy's — declining mall retailer"),
    ("GM",  "General Motors — legacy auto, 2009 bankruptcy alumnus"),
    ("C",   "Citigroup — perennially discounted money-center bank"),
]

# Famously LEAST-admired / low-reputation firms that DELISTED or went bankrupt (no clean
# yfinance series). Named for the survivorship caveat on the spurned/long-short leg: the
# genuine bottom of the reputation ranking deletes itself, biasing a priced spurned book UP.
DELISTED = [
    "Sears / Sears Holdings — 2018 bankruptcy",
    "Lehman Brothers — 2008 failure",
    "General Motors 'old GM' (GM->MTLQQ) — 2009 bankruptcy",
    "Eastman Kodak — 2012 bankruptcy",
    "Enron — 2001 fraud / collapse",
    "Blockbuster — 2010 liquidation",
    "American Airlines 'AMR' — 2011 bankruptcy",
]

TICKERS = sorted({t for t, *_ in ADMIRED} | {t for t, *_ in SPURNED})

# A broad large-cap POOL for the placebo null: random equal-weight books of the same size
# drawn from here size "would a random basket of famous large caps beat SPY this much?".
# A transparent, hardcoded roster of well-known large caps (not the admired names alone).
POOL = sorted(set(TICKERS) | {
    "JNJ", "PG", "KO", "PEP", "WMT", "HD", "MCD", "VZ", "XOM", "CVX",
    "PFE", "MRK", "ABT", "TMO", "CSCO", "ORCL", "IBM", "INTC", "QCOM", "TXN",
    "CAT", "BA", "GE", "MMM", "HON", "UPS", "LOW", "TGT", "BAC", "WFC",
    "GS", "MS", "USB", "UNH", "CVS", "MDT", "DHR", "LIN", "ADBE", "AMD",
})


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2004-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the admired + spurned + pool tickers + SPY via yfinance (month-end) and cache.

    Network-only; used once to build ``_cache/admired_prices.csv``. Never imported by the
    offline notebook cells. Keeps every column with any usable history.
    """
    import yfinance as yf

    tickers = sorted(set(POOL) | {"SPY"})
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    monthly = raw.resample("ME").last().dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    monthly.to_csv(path)
    return monthly


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide month-end adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def entry_dates(admired: list | None = None) -> dict:
    """Map each admired ticker to its publication-lagged entry date (Feb 1 of first_year).

    The survey prints late January; a name is ownable only from the following month, so the
    lagged book has no look-ahead into which firms Fortune would *later* crown.
    """
    admired = admired if admired is not None else ADMIRED
    return {t: pd.Timestamp(f"{y}-02-01") for t, y, *_ in admired}


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience bundle for the strategy layer: month-end prices + the admired/spurned tables."""
    return {"prices": load_prices(path), "admired": ADMIRED, "spurned": SPURNED,
            "entry": entry_dates()}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_admired(n_names: int = 15, edge_ann: float = 0.0, seed: int = 721,
                      n_months: int = 200, mkt_mu: float = 0.007,
                      mkt_sig: float = 0.043, idio_sig: float = 0.05) -> dict:
    """Deterministic month-end book with a planted **annual** admiration premium ``edge_ann``.

    Each of ``n_names`` "admired" stocks is market (beta 1) + idiosyncratic noise + a constant
    monthly abnormal return ``edge_ann/12``. ``edge_ann = 0`` plants nothing: the HAC
    inference must then find no significant premium, however the noise falls. A large
    ``edge_ann`` (e.g. 0.06 = 6%/yr) must light up. Returns a bundle shaped like
    :func:`load_real` with an all-in-sample entry (no lag) so the control isolates power.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2004-01-31", periods=n_months, freq="M").to_timestamp("M")
    mkt = rng.normal(mkt_mu, mkt_sig, n_months)
    cols = {"SPY": 100.0 * np.cumprod(1 + mkt)}
    admired = []
    for k in range(n_names):
        r = mkt + rng.normal(edge_ann / 12.0, idio_sig, n_months)
        tkr = f"AD{k:02d}"
        cols[tkr] = 100.0 * np.cumprod(1 + r)
        admired.append((tkr, 2004, f"synthetic admired {k}"))
    prices = pd.DataFrame(cols, index=idx)
    entry = {t: idx[0] for t, *_ in admired}
    return {"prices": prices, "admired": admired, "spurned": [], "entry": entry}
