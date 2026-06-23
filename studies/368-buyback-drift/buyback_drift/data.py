"""Data layer for Study 368 — Buyback-Drift (event table + abnormal returns vs SPY).

Two sources, both offline-friendly:

* **Real tape.** A hard-coded, transparent table of ~30 notable share-**buyback
  authorization** announcements (``ticker``, ``date``, program ``size_bn`` in USD billions).
  Each row is a board/press-release *authorization* — the announcement the market reacts to —
  **not** execution (actual repurchases trickle out over years and are not in the headline).
  We download daily adjusted closes for the named tickers + SPY (yfinance, no key), cached
  under ``_cache/event_prices.csv`` (a wide CSV, one column per ticker plus ``SPY``), and build
  the **abnormal** forward return: the event stock's return minus SPY's over the same window.

* **Synthetic.** A deterministic, fixed-seed generator that produces ``n_events`` event windows
  with a **planted drift edge** (``edge`` = the true cumulative abnormal drift injected over the
  ~126-day window after each announcement). It is the positive control: with ``edge=0`` the
  inference must NOT manufacture significance from a few-dozen noisy events; with a large planted
  ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_event_prices`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "event_prices.csv")

# A transparent, hand-curated table of ~30 notable, well-documented share-buyback
# AUTHORIZATION announcements: the board/press-release OK of a $X repurchase program — the
# event the "buyback drift" folklore is about — NOT execution. (ticker, announcement date,
# program size in USD billions.) Dates are the public announcement dates; sizes are the
# headline authorized amounts. This is an explicit, named SAMPLE (point-in-time press-release
# timestamps for thousands of authorizations are not on yfinance), and we say so on the Signal
# axis. Chosen for size, recognisability, sector spread, and long post-event price history.
EVENTS = [
    ("AAPL", "2013-04-23", 60.0),   # raised total program to $60bn
    ("AAPL", "2014-04-23", 30.0),   # expanded by $30bn
    ("AAPL", "2018-05-01", 100.0),  # $100bn authorization
    ("AAPL", "2023-05-04", 90.0),   # $90bn authorization
    ("MSFT", "2013-09-16", 40.0),   # $40bn program
    ("MSFT", "2016-09-20", 40.0),   # $40bn program
    ("MSFT", "2021-09-14", 60.0),   # $60bn program
    ("GOOGL", "2018-07-23", 25.0),  # $25bn class C
    ("GOOGL", "2022-04-26", 70.0),  # $70bn authorization
    ("META", "2021-10-25", 50.0),   # $50bn authorization
    ("META", "2023-02-01", 40.0),   # $40bn add to program
    ("ORCL", "2017-06-21", 12.0),   # $12bn add
    ("CSCO", "2018-02-14", 25.0),   # raised by $25bn
    ("INTC", "2011-09-22", 10.0),   # $10bn add
    ("QCOM", "2018-07-26", 30.0),   # $30bn authorization
    ("IBM", "2013-10-29", 15.0),    # $15bn add
    ("HPQ", "2015-09-29", 7.0),     # multi-bn program post-split
    ("WMT", "2017-10-10", 20.0),    # $20bn program
    ("HD", "2017-02-21", 15.0),     # $15bn authorization
    ("PG", "2014-04-23", 6.0),      # multi-year program
    ("KO", "2012-10-16", 6.0),      # program expansion
    ("PEP", "2018-02-13", 15.0),    # $15bn authorization
    ("JNJ", "2017-01-03", 10.0),    # $10bn program
    ("PFE", "2014-02-03", 5.0),     # added $5bn
    ("MRK", "2018-05-01", 10.0),    # $10bn authorization
    ("JPM", "2018-06-28", 20.7),    # ~$20.7bn (CCAR)
    ("BAC", "2019-06-27", 30.0),    # ~$30bn (CCAR)
    ("XOM", "2021-12-01", 10.0),    # up to $10bn program
    ("CVX", "2023-01-25", 75.0),    # $75bn authorization
    ("BA", "2014-12-15", 12.0),     # $12bn program
    ("DIS", "2016-08-09", 8.0),     # program continuation
    ("NKE", "2018-06-28", 15.0),    # $15bn program
]


def event_table() -> pd.DataFrame:
    """The hard-coded buyback-authorization table as a frame (ticker, date, size_bn)."""
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "size_bn"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_event_prices(start: str = "2010-01-01", end: str | None = None,
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


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 31, edge: float = 0.0, seed: int = 368,
                     sig_daily: float = 0.018, horizon: int = 252,
                     pre: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic event panel with a **planted abnormal drift** of ``edge`` per event.

    Builds ``n_events`` independent event windows. Each window is a synthetic *abnormal*
    (already SPY-subtracted) log-return path of length ``pre + horizon`` days with daily vol
    ``sig_daily`` and **zero** unconditional drift; if ``edge`` != 0 an *extra* cumulative
    abnormal drift of ``edge`` is spread evenly over the ``horizon`` days *after* the
    announcement (day ``pre``). Returns ``(prices, events)`` in the same shape the strategy
    layer consumes: a wide price frame whose columns are ``EV00..`` plus a flat ``SPY``
    (==1.0, so "abnormal" == raw for the synthetic), and an event table pointing each ``EVkk``
    column at its announcement date.

    With ``edge = 0`` the control must NOT find significance — a few-dozen noisy windows
    cannot reject the null however the draws fall (the sample-size lesson on data where we
    *know* the truth). A large planted ``edge`` MUST light the test up.
    """
    rng = np.random.default_rng(seed)
    win = pre + horizon
    # decorative monthly-free daily index; long enough to host all windows back-to-back
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
        ann_date = idx[cursor + pre]           # announcement = day `pre` into the window
        rows.append((name, ann_date, 1.0))
        cursor += win + 5
    cols["SPY"] = pd.Series(1.0, index=idx)    # flat market => abnormal == raw here
    prices = pd.DataFrame(cols)
    events = pd.DataFrame(rows, columns=["ticker", "date", "size_bn"])
    events["date"] = pd.to_datetime(events["date"])
    return prices, events
