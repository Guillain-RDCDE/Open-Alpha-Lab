"""Data layer for Study 563 — Secondary-Offering-Drift (event table + abnormal returns vs SPY).

Two sources, both offline-friendly:

* **Real tape.** A hard-coded, transparent table of ~34 notable **seasoned / follow-on equity
  offerings** (``ticker``, pricing ``date``, deal ``size_bn`` in USD billions). Each row is a
  marketed/primary follow-on or large secondary in which the company (or a large holder) sold a
  fresh slug of shares to the public — the dilution event the "offering drift" claim is about.
  We download daily adjusted closes for the named tickers + SPY (yfinance, no key), cache them
  under ``_cache/event_prices.csv`` (a wide CSV, one column per ticker plus ``SPY``), and build
  the **abnormal** forward return: the event stock's return minus SPY's over the same window.

* **Synthetic.** A deterministic, fixed-seed generator that produces ``n_events`` event windows
  with a **planted (negative) drift edge** (``edge`` = the true cumulative abnormal drift injected
  over the ~126-day window after each offering). It is the positive control: with ``edge=0`` the
  inference must NOT manufacture significance from a few-dozen noisy events; with a large planted
  (negative) ``edge`` it must light up on the *downside*.

Pure numpy + pandas + stdlib for the offline path. ``fetch_event_prices`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.

Survivorship / selection note (named on the Signal axis): the table is hand-picked, recognisable
offerings by firms that (mostly) survived, and the pricing dates are approximate public pricing
dates rather than an SEC-424B point-in-time feed. That biases the sample toward the *tradable,
orderly* offerings and away from the distressed dilutions where the dilution signal is strongest —
so if a drift shows up it is real-as-lore; if it doesn't, the small-sample lesson still holds a
fortiori for the true population.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "event_prices.csv")

# A transparent, hand-curated table of ~34 notable, well-documented SEASONED / FOLLOW-ON EQUITY
# OFFERINGS: the company (or a large holder) sold a fresh slug of shares to the public — a
# marketed follow-on, a primary raise, or a large registered secondary. (ticker, public pricing
# date, approximate deal size in USD billions.) These are the dilution events the "offering
# drift" claim is about. Chosen for size, recognisability, sector spread, and long post-event
# price history. This is an explicit, named SAMPLE (point-in-time 424B feeds for thousands of
# offerings are not on yfinance) — survivorship + visibility named on the Signal axis.
EVENTS = [
    ("TSLA", "2013-05-16", 1.02),   # $1bn+ follow-on (raised to fund growth)
    ("TSLA", "2015-08-19", 0.74),   # follow-on
    ("TSLA", "2016-05-19", 1.70),   # $1.7bn follow-on
    ("TSLA", "2019-05-02", 2.35),   # $2.35bn equity+notes raise
    ("TSLA", "2020-02-13", 2.31),   # $2bn+ follow-on
    ("TSLA", "2020-09-01", 5.00),   # $5bn at-the-market program
    ("TSLA", "2020-12-08", 5.00),   # $5bn at-the-market program
    ("CCL", "2020-04-01", 0.50),    # pandemic dilution raise
    ("CCL", "2020-11-16", 1.02),    # follow-on
    ("NCLH", "2020-05-06", 0.40),   # pandemic equity raise
    ("AAL", "2020-06-25", 1.10),    # $1.1bn stock+notes raise
    ("UAL", "2020-04-21", 1.10),    # equity raise
    ("BA", "2009-06-04", 0.60),     # secondary
    ("GM", "2010-11-18", 20.10),    # GM re-IPO / secondary
    ("F", "2009-05-11", 1.40),      # $1.4bn follow-on during crisis
    ("BAC", "2009-05-08", 13.50),   # $13.5bn share sale (stress-test raise)
    ("WFC", "2009-05-08", 8.60),    # $8.6bn common raise
    ("C", "2009-04-30", 0.00),      # capital raise (placeholder size below cleaned out)
    ("MS", "2009-05-08", 4.00),     # $4bn common raise
    ("PLUG", "2021-01-25", 1.00),   # $1bn follow-on at highs
    ("PLUG", "2020-11-24", 0.36),   # follow-on
    ("BYND", "2019-07-29", 0.51),   # secondary at post-IPO highs
    ("SNAP", "2021-04-29", 1.05),   # convertible/equity raise
    ("COIN", "2024-03-18", 1.00),   # convertible/equity raise
    ("MSTR", "2024-06-14", 0.70),   # convertible raise to buy BTC
    ("RIOT", "2021-02-03", 0.60),   # ATM equity raise
    ("MARA", "2021-01-20", 0.25),   # ATM equity raise
    ("DKNG", "2020-10-14", 1.66),   # $1.6bn secondary
    ("PTON", "2020-09-15", 1.00),   # follow-on at highs
    ("ZM", "2021-01-11", 1.75),     # $1.75bn follow-on
    ("MRNA", "2020-05-18", 1.34),   # $1.3bn follow-on (vaccine news)
    ("NVAX", "2020-09-09", 0.20),   # dilution raise
    ("SPCE", "2021-07-06", 0.50),   # $500m ATM after spaceflight
    ("LCID", "2023-05-31", 3.00),   # $3bn raise (PIF-backed)
    ("RIVN", "2024-03-05", 1.50),   # convertible raise
]


def event_table() -> pd.DataFrame:
    """The hard-coded offering table as a frame (ticker, date, size_bn), sorted by date.

    Rows with a non-positive placeholder size are dropped (a size we could not verify cleanly);
    the drift test itself does not use the size except for the size-split robustness cut.
    """
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "size_bn"])
    df = df[df["size_bn"] > 0.0].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_event_prices(start: str = "2008-01-01", end: str | None = None,
                       path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the event tickers + SPY via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/event_prices.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    tickers = sorted({t for t, _, _ in EVENTS} | {"SPY"})
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real(path: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: returns (prices, event_table) for the strategy layer."""
    return load_prices(path), event_table()


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the cached price panel for the as-of stamp."""
    import hashlib

    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float)).tobytes()
    return hashlib.sha1(arr).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 34, edge: float = 0.0, seed: int = 563,
                     sig_daily: float = 0.030, horizon: int = 252,
                     pre: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic event panel with a **planted abnormal drift** of ``edge`` per event.

    Builds ``n_events`` independent event windows. Each window is a synthetic *abnormal*
    (already SPY-subtracted) log-return path of length ``pre + horizon`` days with daily vol
    ``sig_daily`` and **zero** unconditional drift; if ``edge`` != 0 an *extra* cumulative
    abnormal drift of ``edge`` is spread evenly over the ``horizon`` days *after* the offering
    (day ``pre``). For the offering claim the planted edge is **negative** (the stock slides).
    Returns ``(prices, events)`` in the same shape the strategy layer consumes: a wide price
    frame whose columns are ``EV00..`` plus a flat ``SPY`` (==1.0, so "abnormal" == raw for the
    synthetic), and an event table pointing each ``EVkk`` column at its offering date.

    With ``edge = 0`` the control must NOT find significance — a few-dozen noisy windows cannot
    reject the null however the draws fall (the sample-size lesson on data where we *know* the
    truth). A large planted (negative) ``edge`` MUST light the test up on the downside. The
    daily vol here (3%/day) is deliberately set to the high single-name volatility of the real
    offering names (many are high-beta growth/crypto-adjacent), so the control's power matches
    the real tape.
    """
    rng = np.random.default_rng(seed)
    win = pre + horizon
    idx = pd.bdate_range("2005-01-03", periods=n_events * (win + 5))
    cols = {}
    rows = []
    cursor = 5
    for k in range(n_events):
        r = rng.normal(0.0, sig_daily, size=win)
        if edge != 0.0:
            r[pre:] += edge / horizon          # spread the planted edge over the horizon
        path = 100.0 * np.exp(np.cumsum(r))
        series = pd.Series(np.nan, index=idx)
        seg = idx[cursor:cursor + win]
        series.loc[seg] = path
        name = f"EV{k:02d}"
        cols[name] = series
        ann_date = idx[cursor + pre]           # offering = day `pre` into the window
        rows.append((name, ann_date, 1.0))
        cursor += win + 5
    cols["SPY"] = pd.Series(1.0, index=idx)    # flat market => abnormal == raw here
    prices = pd.DataFrame(cols)
    events = pd.DataFrame(rows, columns=["ticker", "date", "size_bn"])
    events["date"] = pd.to_datetime(events["date"])
    return prices, events
