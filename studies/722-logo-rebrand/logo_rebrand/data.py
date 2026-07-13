"""Data layer for Study 722 — Logo-Rebrand (renewal, or a floundering firm's vanity?).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~26 real corporate rebrands / logo
  changes 2010-2025 — a firm that changed its *name* (Facebook -> Meta, Google -> Alphabet,
  Weight Watchers -> WW), its *identity/structure* (ConAgra -> Conagra Brands), or just its
  *logo* (Starbucks drops "Coffee", Mastercard drops the name, Pepsi's 2023 redesign). For
  each event we record the *currently-traded ticker* and the approximate rebrand-announcement
  date; from yfinance daily adjusted closes we then measure the event-window drift around the
  reveal (a short **announce** reaction, then a longer **drift**). Cached under
  ``_cache/rebrand_prices.csv`` (a wide CSV, one column per ticker).

  The table is the honest part of this study: every row is a real, documented rebrand, and
  the *survivorship* is named loudly — several firms rebranded and then **floundered into
  bankruptcy or delisting** (Twitter -> X went private; Weight Watchers -> WW later filed
  Chapter 11 in 2025; Overstock -> Bed Bath & Beyond went bust) — no clean yfinance series.
  So the surviving tape is biased *toward* the rebrands that did NOT collapse, i.e. **against**
  the cynical "a rebrand is a floundering firm's tell" story. A survivor-only drift that is
  *not negative* is therefore a conservative refutation of the floundering thesis; a survivor
  drift that is *positive* can't distinguish renewal from survivorship. Named on the Signal
  axis.

* **Synthetic.** A deterministic, fixed-seed generator that plants a controllable
  post-rebrand "renewal drift" of size ``edge`` into otherwise-random event windows. The
  positive control: with the ``edge`` knob at 0 the inference must NOT manufacture a
  significant announce/drift out of a handful of events; with a large planted edge it must
  light up.

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
# The hardcoded rebrand table — real corporate rebrands / logo changes 2010-2025.
#
# Each row: (ticker, announce_date, kind, label)
#   * ticker       — the symbol that carries the post-rebrand price on yfinance today
#                    (the survivor symbol; rebrands that delisted / went private cannot be
#                    priced and are listed in DELISTED below, named for survivorship).
#   * announce_date— approximate date the rebrand / new logo was announced or revealed.
#   * kind         — "name" (new corporate name), "identity" (name + structure/holdco),
#                    or "logo" (visual identity refresh only).
#   * label        — old -> new, for the charts.
#
# Dates are the documented reveal (rounded to a trading-day-ish date); the engine snaps each
# to the nearest available price date, so day-level precision is not required. Sources are
# public press announcements — see docs/references.md.
# ---------------------------------------------------------------------------- #
REBRANDS = [
    # ---- name changes -------------------------------------------------------
    ("GOOGL", "2015-08-10", "identity", "Google -> Alphabet holding co"),
    ("META",  "2021-10-28", "name",     "Facebook -> Meta Platforms"),
    ("BB",    "2013-01-30", "name",     "Research In Motion -> BlackBerry"),
    ("WW",    "2018-09-24", "name",     "Weight Watchers -> WW"),
    ("TTE",   "2021-05-28", "name",     "Total -> TotalEnergies"),
    ("ALLY",  "2010-05-10", "name",     "GMAC -> Ally Financial"),
    ("BKNG",  "2018-02-21", "name",     "Priceline -> Booking Holdings"),
    ("BBWI",  "2021-08-02", "name",     "L Brands -> Bath & Body Works"),
    ("TPR",   "2017-10-31", "name",     "Coach -> Tapestry"),
    ("CPRI",  "2018-12-28", "name",     "Michael Kors -> Capri Holdings"),
    ("PARA",  "2022-02-16", "name",     "ViacomCBS -> Paramount Global"),
    ("MDLZ",  "2012-10-02", "name",     "Kraft Foods -> Mondelez"),
    ("KHC",   "2015-07-06", "name",     "Kraft + Heinz -> Kraft Heinz"),
    # ---- identity / structure refresh --------------------------------------
    ("CAG",   "2016-04-07", "identity", "ConAgra -> Conagra Brands"),
    ("DOW",   "2019-04-01", "identity", "Dow relists as standalone Dow Inc"),
    ("HPQ",   "2015-11-02", "identity", "HP splits -> HP Inc"),
    # ---- logo-only refreshes ------------------------------------------------
    ("SBUX",  "2011-01-05", "logo",     "Starbucks drops 'Coffee' from logo"),
    ("MA",    "2019-01-07", "logo",     "Mastercard removes name from logo"),
    ("VZ",    "2015-09-01", "logo",     "Verizon flat checkmark logo"),
    ("DIN",   "2018-06-04", "logo",     "IHOP -> 'IHOb' burger stunt"),
    ("GM",    "2021-01-08", "logo",     "General Motors lowercase logo"),
    ("INTC",  "2020-09-02", "logo",     "Intel drops the swoosh, new logo"),
    ("PEP",   "2023-03-28", "logo",     "Pepsi first global logo in 14y"),
    ("JNJ",   "2023-09-15", "logo",     "J&J first logo change in 130y"),
    ("WMT",   "2025-01-13", "logo",     "Walmart 'spark' logo refresh"),
    ("NFLX",  "2011-09-19", "identity", "Netflix 'Qwikster' split fiasco"),
]

# Famous rebrands that FLOUNDERED into delisting / bankruptcy / going private (no clean
# yfinance series). Named for the survivorship caveat: the survivor tape is biased AGAINST
# the "a rebrand is a floundering firm's tell" story, because the worst-outcome rebrands
# left the tape entirely.
DELISTED = [
    "Twitter -> X (2023, taken private by Musk — no listed series)",
    "Weight Watchers -> WW (2018 rename; filed Chapter 11 in 2025)",
    "Overstock -> Bed Bath & Beyond (2023 rename; the acquired brand had gone bankrupt)",
    "WeWork rebrand / 'We Company' (2019; bankrupt 2023)",
    "Tribune Publishing -> tronc (2016; reverted, then acquired/delisted)",
    "Sears Holdings 'Shop Your Way' pivot (delisted 2018)",
    "Xerox 'The Document Company' era brands (later split/delisted)",
    "Dunkin' Donuts -> Dunkin' (2019 rename; taken private 2020)",
]

TICKERS = sorted({t for t, *_ in REBRANDS})


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2008-01-01", end: str | None = None,
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
    for tkr, dt, kind, label in REBRANDS:
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
                     "kind": kind, "label": label})
    return pd.DataFrame(rows)


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience bundle for the strategy layer: prices + the snapped event table."""
    return {"prices": load_prices(path), "events": event_table(path)}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_rebrands(n_events: int = 26, edge: float = 0.0, seed: int = 722,
                       n_days: int = 1_500, mu_daily: float = 0.0003,
                       sig_daily: float = 0.020, announce_days: int = 5,
                       drift_days: int = 120) -> dict:
    """Deterministic event windows with a planted post-rebrand "renewal drift" of ``edge``.

    For each of ``n_events`` synthetic stocks we draw a random walk (drift ``mu_daily``,
    vol ``sig_daily`` — large-cap-ish, the rebrand names here are mostly big caps), place a
    rebrand at the midpoint, and — when ``edge`` != 0 — add a cumulative **+edge** abnormal
    return spread over ``drift_days`` trading days after the reveal (the "renewal" the story
    predicts). ``edge = 0`` plants nothing: the inference must then find no significant
    announce reaction and no significant drift, however the noise falls. A large ``edge``
    (e.g. 0.30) must light up the drift leg.

    Returns a bundle shaped like :func:`load_real`: a wide price frame (one column per
    synthetic ticker plus ``SPY``) and an event table with the reveal date per ticker.
    Uses ``pd.period_range`` -> timestamp to stay clear of datetime overflow.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01-03", periods=n_days, freq="D").to_timestamp()
    cols, events = {}, []
    cols["SPY"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n_days)))
    mid = n_days // 2
    for k in range(n_events):
        r = rng.normal(mu_daily, sig_daily, n_days)
        e0 = mid + int(rng.integers(-n_days // 6, n_days // 6))   # jittered reveal day
        if edge != 0.0:
            d0 = e0 + 1 + announce_days
            r[d0:d0 + drift_days] += edge / drift_days            # renewal drift only
        price = 50.0 * np.exp(np.cumsum(r))
        tkr = f"SY{k:02d}"
        cols[tkr] = price
        events.append({"ticker": tkr, "announce": idx[e0], "snapped": idx[e0],
                       "kind": "synthetic", "label": f"synthetic rebrand {k}"})
    prices = pd.DataFrame(cols, index=idx)
    return {"prices": prices, "events": pd.DataFrame(events)}
