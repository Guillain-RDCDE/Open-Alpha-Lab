"""Data layer for Study 389 — Name-Change-Effect (theme-chasing rebrands).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~25 real theme-chasing corporate
  rebrands — companies that renamed/retickered toward whatever was hot (``.com`` in
  1998-2000, ``blockchain`` in 2017-2018, ``AI`` in 2023-2024). For each event we record
  the *currently-traded ticker* (post-rebrand, or the survivor symbol) and the approximate
  rebrand-announcement date; from yfinance daily adjusted closes we then measure the
  event-window drift around the rename (a short pop, then the give-back). Cached under
  ``_cache/rebrand_prices.csv`` (a wide CSV, one column per ticker).

  The table is the honest part of this study: every row is a real, documented rename, and
  the *survivorship* is named loudly — many of the loudest dot-com/blockchain rebrands
  delisted and leave **no** yfinance series, so the surviving tape is biased *toward* the
  names that did not go to zero. That bias works **against** the believers' "and then give
  it back" claim, so a survivor-only pop-and-fade is a conservative read.

* **Synthetic.** A deterministic, fixed-seed generator that plants a controllable
  rebrand "pop + fade" into otherwise-random event windows. The positive control: with the
  ``edge`` knob at 0 the inference must NOT manufacture a significant pop/fade out of a
  handful of events; with a large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "rebrand_prices.csv")

# ---------------------------------------------------------------------------- #
# The hardcoded rebrand table — real theme-chasing name/ticker changes.
#
# Each row: (ticker, announce_date, wave, label)
#   * ticker       — the symbol that carries the post-rebrand price on yfinance today
#                    (the survivor symbol; delisted rebrands cannot be priced and are
#                    listed in DELISTED below, named for the survivorship caveat).
#   * announce_date— approximate date the theme-chasing rename was announced/effective.
#   * wave         — "dotcom" (1998-2001), "blockchain" (2017-2018), "ai" (2023-2024).
#   * label        — old name -> new name, for the charts.
#
# Dates are the documented rebrand month (rounded to a trading-day-ish date); the engine
# snaps each to the nearest available price date, so day-level precision is not required.
# ---------------------------------------------------------------------------- #
REBRANDS = [
    # ---- dot-com wave (append ".com" / internet branding) --------------------
    ("DDS",  "1999-06-15", "dotcom", "(dot-com era branding pivot)"),
    ("BBY",  "2000-04-03", "dotcom", "bricks-to-clicks .com push"),
    ("IBM",  "1999-03-01", "dotcom", "e-business .com positioning"),
    ("INTC", "1999-09-01", "dotcom", "internet-economy repositioning"),
    ("ORCL", "2000-01-10", "dotcom", "internet-platform rebrand"),
    ("CSCO", "1999-11-01", "dotcom", "the-internet-is-the-network branding"),
    ("XRX",  "1999-09-15", "dotcom", "Xerox -> 'The Document Company' / .com"),
    # ---- blockchain wave (rename toward 'blockchain'/'crypto') ---------------
    ("RIOT", "2017-10-04", "blockchain", "Bioptix -> Riot Blockchain"),
    ("MARA", "2018-02-28", "blockchain", "Marathon Patent -> Marathon (crypto pivot)"),
    ("SOS",  "2017-12-01", "blockchain", "China Rapid Finance -> blockchain pivot"),
    ("GREE", "2017-11-01", "blockchain", "name/ticker pivot to crypto mining"),
    ("OSTK", "2017-08-07", "blockchain", "Overstock -> tZERO blockchain push"),
    ("BTBT", "2020-03-09", "blockchain", "Bit Digital crypto-mining rebrand"),
    ("CAN",  "2019-11-21", "blockchain", "Canaan crypto-ASIC listing"),
    ("MSTR", "2020-08-11", "blockchain", "MicroStrategy -> bitcoin treasury pivot"),
    # ---- AI wave (rename/positioning toward 'AI') ----------------------------
    ("AI",   "2020-12-09", "ai", "C3.ai -> ticker 'AI'"),
    ("BBAI", "2021-12-08", "ai", "BigBear.ai rebrand/listing"),
    ("SOUN", "2022-04-28", "ai", "SoundHound AI listing"),
    ("PLTR", "2023-05-08", "ai", "Palantir -> 'AIP' AI positioning"),
    ("SYM",  "2022-06-08", "ai", "Symbotic AI-robotics listing"),
    ("NVDA", "2023-05-24", "ai", "Nvidia -> 'we are an AI company' pivot"),
    ("IBM",  "2023-05-09", "ai", "IBM -> watsonx AI rebrand"),
    ("AAOI", "2024-03-01", "ai", "Applied Optoelectronics AI-datacenter pivot"),
    ("AMD",  "2023-06-13", "ai", "AMD -> AI-accelerator repositioning"),
    ("MNDY", "2024-01-15", "ai", "monday.com -> 'monday AI' rebrand"),
]

# Famous theme-chasing rebrands that DELISTED / went to zero (no yfinance series).
# Named for the survivorship caveat: the survivor tape is biased AGAINST the "give it
# back" story, because the worst give-backs left the tape entirely.
DELISTED = [
    "Computer Literacy -> fatbrain.com (dot-com)",
    "Books-A-Million 'booksamillion.com' pop (dot-com)",
    "K-Tel International -> ktel.com (~+700% then collapse, dot-com)",
    "Zap.com / NetSol (dot-com)",
    "Long Island Iced Tea -> Long Blockchain Corp (2017, +200% then delisted)",
    "UBI Blockchain Internet (2017, halted by SEC)",
    "On-line Plc / On-line Blockchain (2017, UK)",
    "Eastman Kodak -> KodakCoin (2018, +300% then faded)",
]

TICKERS = sorted({t for t, *_ in REBRANDS})


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1995-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the rebrand tickers + a market benchmark (SPY) via yfinance and cache.

    Network-only; used once to build ``_cache/rebrand_prices.csv``. Never imported by the
    offline notebook cells. Keeps every column with any usable history.
    """
    import yfinance as yf

    tickers = sorted(set(TICKERS) | {"SPY"})
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


def event_table(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """The rebrand table joined to whether the cached tape actually covers the window.

    Returns one row per rebrand with the announce date snapped to the nearest available
    price date for that ticker, or NaT if the ticker isn't priced around the event (e.g.
    the company listed after the announce date in the cache). Offline if the cache exists.
    """
    prices = load_prices(path) if os.path.exists(path) else None
    rows = []
    for tkr, dt, wave, label in REBRANDS:
        andt = pd.Timestamp(dt)
        snapped = pd.NaT
        if prices is not None and tkr in prices.columns:
            s = prices[tkr].dropna()
            if len(s):
                near = s.index[(s.index >= andt - pd.Timedelta(days=10)) &
                               (s.index <= andt + pd.Timedelta(days=10))]
                if len(near):
                    snapped = near[np.argmin(np.abs(near - andt))]
        rows.append({"ticker": tkr, "announce": andt, "snapped": snapped,
                     "wave": wave, "label": label})
    return pd.DataFrame(rows)


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience bundle for the strategy layer: prices + the snapped event table."""
    return {"prices": load_prices(path), "events": event_table(path)}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_rebrands(n_events: int = 25, edge: float = 0.0, seed: int = 389,
                       n_days: int = 1_500, mu_daily: float = 0.0003,
                       sig_daily: float = 0.028, pop_days: int = 5,
                       fade_days: int = 60) -> dict:
    """Deterministic event windows with a planted "pop + fade" of size ``edge``.

    For each of ``n_events`` synthetic stocks we draw a random walk (drift ``mu_daily``,
    vol ``sig_daily`` — small-cap-ish), place a rebrand at the midpoint, and — when
    ``edge`` != 0 — add a cumulative **+edge** abnormal return spread over ``pop_days``
    trading days right after the rename, then **give it all back** (−edge) spread over the
    next ``fade_days``. ``edge = 0`` plants nothing: the inference must then find no
    significant pop and no significant fade, however the noise falls. A large ``edge``
    (e.g. 0.40 = a 40% pop) must light up both legs.

    Returns a bundle shaped like :func:`load_real`: a wide price frame (one column per
    synthetic ticker plus ``SPY``) and an event table with the rename date per ticker.
    Uses ``pd.period_range`` -> timestamp to stay clear of datetime overflow.
    """
    rng = np.random.default_rng(seed)
    # pd.period_range avoids datetime overflow on long synthetic series; the date index is
    # only a decorative label here (the event study works in integer offsets).
    idx = pd.period_range("2000-01-03", periods=n_days, freq="D").to_timestamp()
    cols, events = {}, []
    # a benign market benchmark
    cols["SPY"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n_days)))
    mid = n_days // 2
    for k in range(n_events):
        r = rng.normal(mu_daily, sig_daily, n_days)
        e0 = mid + int(rng.integers(-n_days // 6, n_days // 6))   # jittered rename day
        if edge != 0.0:
            # pop spread over pop_days, then full give-back over fade_days
            r[e0 + 1:e0 + 1 + pop_days] += edge / pop_days
            f0 = e0 + 1 + pop_days
            r[f0:f0 + fade_days] += (-edge) / fade_days
        price = 50.0 * np.exp(np.cumsum(r))
        tkr = f"SY{k:02d}"
        cols[tkr] = price
        events.append({"ticker": tkr, "announce": idx[e0], "snapped": idx[e0],
                       "wave": "synthetic", "label": f"synthetic rebrand {k}"})
    prices = pd.DataFrame(cols, index=idx)
    return {"prices": prices, "events": pd.DataFrame(events)}
