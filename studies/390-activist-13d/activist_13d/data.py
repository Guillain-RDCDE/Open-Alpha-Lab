"""Data layer for Study 390 — Activist-13D (target prices + SPY + synthetic control).

Two sources, both offline-friendly:

* **Real tape.** A transparent, **hardcoded table of famous activist 13D campaigns**
  (target ticker, activist, announcement date) — the kind of headline filing the
  folklore is built on. For each campaign we pull the target's and SPY's daily adjusted
  closes (yfinance, no key), cached under ``_cache/event_prices.csv`` (a wide CSV indexed
  by date, one column per ticker plus ``SPY``). True EDGAR coverage of *every* 13D is not
  freely/cleanly available, so this is an explicit, transparent **selected basket** of
  well-known events — and that selection (only famous, mostly-successful campaigns get
  remembered) is named on the Signal axis as survivorship/selection bias.

* **Synthetic.** A deterministic, fixed-seed generator that produces a set of synthetic
  13D "events": each target follows a GBM-like path, an announcement is dropped on a
  known date, and a controllable **planted drift edge** is added over the following months.
  It is the positive control: the engine must recover a planted edge (and, with the edge
  set to zero, must NOT manufacture significance out of ~25 events).

Pure numpy + pandas + stdlib for the offline path. ``fetch_events`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "event_prices.csv")

# --------------------------------------------------------------------------- #
# The hardcoded 13D event table — ~25 famous activist campaigns.
# (target_ticker, activist, announcement_date). Dates are the approximate public
# announcement / Schedule-13D-filing dates of widely-reported campaigns. This is an
# explicit, transparent SELECTED basket: campaigns enter folklore precisely because they
# were prominent (and often profitable), so the set carries selection/survivorship bias,
# which we name on the Signal axis. The point of the study is to ask whether, even on a
# basket tilted *toward* memorable wins, the post-announcement drift you can actually buy
# beats the market by more than luck.
# --------------------------------------------------------------------------- #
EVENTS = [
    ("AAPL", "Carl Icahn",            "2013-08-13"),  # Icahn tweets a big Apple stake
    ("EBAY", "Carl Icahn",            "2014-01-22"),  # Icahn pushes PayPal spinoff
    ("PYPL", "Elliott Management",    "2022-08-30"),  # Elliott 13D-ish stake in PayPal
    ("TWTR", "Elliott Management",    "2020-02-28"),  # Elliott targets Twitter / Dorsey
    ("PINS", "Elliott Management",    "2022-07-18"),  # Elliott becomes largest Pinterest holder
    ("SBUX", "Elliott Management",    "2024-08-13"),  # Elliott builds Starbucks stake
    ("SWK",  "Elliott Management",    "2024-04-22"),  # Elliott stake in Stanley Black & Decker
    ("CRM",  "Elliott Management",    "2023-01-22"),  # Elliott stake in Salesforce
    ("DELL", "Elliott Management",    "2016-12-19"),  # Elliott stake (placeholder famous name)
    ("MSFT", "ValueAct",              "2013-04-22"),  # ValueAct discloses Microsoft stake
    ("ADBE", "ValueAct",              "2012-01-23"),  # ValueAct Adobe-era activism
    ("DIS",  "Third Point",           "2020-08-19"),  # Loeb pushes Disney to cut dividend
    ("YUM",  "Third Point",           "2015-10-15"),  # Third Point Yum! Brands stake
    ("CPB",  "Third Point",           "2018-08-09"),  # Loeb targets Campbell Soup
    ("HLF",  "Pershing Square",       "2012-12-19"),  # Ackman's famous Herbalife short thesis
    ("CMG",  "Pershing Square",       "2016-09-06"),  # Ackman builds Chipotle stake
    ("LOW",  "Pershing Square",       "2018-05-07"),  # Ackman stake in Lowe's
    ("QSR",  "Pershing Square",       "2014-09-08"),  # Ackman / Burger King-Tim Hortons era
    ("PG",   "Trian (Peltz)",         "2017-07-17"),  # Peltz proxy fight at P&G
    ("GE",   "Trian (Peltz)",         "2015-10-05"),  # Trian discloses GE stake
    ("DD",   "Trian (Peltz)",         "2013-07-17"),  # Trian DuPont campaign
    ("EMR",  "Trian (Peltz)",         "2024-02-12"),  # Trian Emerson-era (placeholder)
    ("WDC",  "Elliott Management",    "2022-05-09"),  # Elliott pushes Western Digital split
    ("MRVL", "Starboard Value",       "2017-03-01"),  # Starboard activism era
    ("BHC",  "Icahn / Glenview",      "2020-08-10"),  # Icahn Bausch Health stake
    ("HRB",  "Starboard Value",       "2022-09-12"),  # Starboard H&R Block-era (placeholder)
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def event_table() -> pd.DataFrame:
    """The hardcoded 13D campaign table as a tidy frame (one row per event)."""
    df = pd.DataFrame(EVENTS, columns=["ticker", "activist", "announce"])
    df["announce"] = pd.to_datetime(df["announce"])
    return df


def fetch_events(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + every target via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/event_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a usable amount of history.
    """
    import yfinance as yf

    tickers = sorted({"SPY", *(t for t, _, _ in EVENTS)})
    raw = yf.download(tickers, start="2010-01-01", end="2026-06-22",
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().sum() >= 250]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real(path: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (wide price frame, event table) — only events whose target priced."""
    prices = load_prices(path)
    tbl = event_table()
    have = set(prices.columns)
    tbl = tbl[tbl["ticker"].isin(have)].reset_index(drop=True)
    return prices, tbl


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 25, drift_6m: float = 0.0, seed: int = 390,
                     n_days: int = 1_500, mu_daily: float = 0.0003,
                     sig_daily: float = 0.018, pop: float = 0.06) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic synthetic 13D events with a controllable planted drift edge.

    Builds ``n_events`` synthetic targets, each a GBM-like price path plus a common SPY
    benchmark sharing the same market factor. Each target gets one announcement on a known
    date: on day 0 the price jumps by ``pop`` (the announcement bump), and if
    ``drift_6m`` != 0 an *extra* cumulative excess drift of ``drift_6m`` is spread over the
    ~126 trading days *after* the announcement (the planted edge the inference must recover).

    With ``drift_6m = 0`` the post-announcement excess is pure noise: ~25 events cannot
    reject the null however the draws fall — the whole small-sample / selection lesson,
    demonstrated where we *know* the truth. The pop is always present (announcement effects
    are real and immediate); the question the control mirrors is the *drift after you can buy*.

    Returns ``(prices, table)`` matching the real-tape shape: a wide adjusted-close frame
    with columns ``SYN00..`` + ``SPY`` and an event table (ticker, activist, announce).
    """
    rng = np.random.default_rng(seed)
    # Use a decorative period index (avoids OutOfBounds; the dates are only labels here).
    idx = pd.period_range("2005-01-03", periods=n_days, freq="D").to_timestamp()

    # common market factor -> SPY and a shared component of each target
    mkt = rng.normal(mu_daily, sig_daily * 0.6, size=n_days)
    spy = 100.0 * np.exp(np.cumsum(mkt))

    cols = {"SPY": spy}
    rows = []
    H = 126  # ~6 months of trading days over which the planted edge is spread
    gap = n_days // (n_events + 2)
    for k in range(n_events):
        idio = rng.normal(0.0, sig_daily, size=n_days)
        ret = 0.9 * mkt + idio                       # beta ~0.9 to the market
        ann = gap * (k + 1) + int(rng.integers(0, gap // 3))
        ann = min(ann, n_days - H - 5)
        ret[ann] += np.log1p(pop)                     # the announcement pop on day 0
        if drift_6m != 0.0:
            ret[ann + 1:ann + 1 + H] += drift_6m / H  # planted excess drift, post-announce
        price = 100.0 * np.exp(np.cumsum(ret))
        name = f"SYN{k:02d}"
        cols[name] = price
        rows.append((name, "synthetic", idx[ann]))

    prices = pd.DataFrame(cols, index=idx)
    table = pd.DataFrame(rows, columns=["ticker", "activist", "announce"])
    return prices, table
