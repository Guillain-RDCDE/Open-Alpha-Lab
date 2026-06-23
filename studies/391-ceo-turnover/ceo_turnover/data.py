"""Data layer for Study 391 — CEO-Turnover (event-window abnormal returns).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~25 notable large-cap CEO changes
  (``CEO_EVENTS``: ticker, announcement date, forced/planned flag), plus daily adjusted
  closes for each ticker and SPY (yfinance, no key), cached under ``_cache/`` as one
  parquet per ticker. From those we compute, for each event, the **abnormal return**
  (stock return minus a market-model fit on a clean pre-event estimation window) and
  cumulate it over short event windows (CAR) around the announcement. True CEO-change
  databases (ExecuComp/BoardEx) are not free, so the dated, labelled table is the
  transparent stand-in — every input is a public price and a public, citable headline.

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* CAR edge (``car_bps``). It is the positive control:
  with the edge set to zero the inference must NOT manufacture significance out of a
  dozen events; with a large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# --------------------------------------------------------------------------- #
# Hardcoded CEO-change event table.
# Columns: ticker, announce_date (the trading day the change hit the tape),
#          forced (True = ousted / pushed out / abrupt; False = planned/orderly).
# Sources: company press releases & contemporaneous financial-press coverage
# (WSJ / Reuters / Bloomberg / FT). "Forced" = activist/board-driven ouster,
# abrupt resignation under pressure, or termination; "Planned" = announced
# succession, retirement, or scheduled hand-off. The forced/planned label is the
# believers' own framing ("firing the CEO") and is, of course, somewhat subjective
# at the margin — we say so on the Signal axis. Large, long-listed names only, so
# the price history is clean around the date.
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # (ticker, announce_date, forced)
    # --- forced / pressured departures ---
    ("HPQ",  "2010-08-06", True),   # Mark Hurd resigns under pressure
    ("HPQ",  "2011-09-22", True),   # Léo Apotheker ousted
    ("YHOO", "2012-05-13", True),   # Scott Thompson out (resume scandal)
    ("UBER", "2017-06-21", True),   # Travis Kalanick forced out
    ("WFC",  "2016-10-12", True),   # John Stumpf abrupt exit (accounts scandal)
    ("EQT",  "2019-07-10", True),   # Rice-team activist ouster of leadership
    ("KHC",  "2019-06-24", True),   # Bernardo Hees pushed out after write-down era
    ("GE",   "2018-10-01", True),   # John Flannery ousted, Culp installed
    ("INTC", "2021-01-13", True),   # Bob Swan replaced amid pressure
    ("TWTR", "2015-06-11", True),   # Dick Costolo steps down under pressure
    ("BA",   "2019-12-23", True),   # Dennis Muilenburg fired (737 MAX)
    ("AIG",  "2017-03-09", True),   # Peter Hancock out after losses
    ("F",    "2017-05-22", True),   # Mark Fields ousted
    ("UAA",  "2019-10-22", True),   # Kevin Plank steps aside under pressure
    # --- planned / orderly successions ---
    ("AAPL", "2011-08-24", False),  # Steve Jobs -> Tim Cook (planned hand-off)
    ("MSFT", "2014-02-04", False),  # Satya Nadella named (orderly succession)
    ("GOOGL","2015-08-10", False),  # Alphabet reorg, Pichai to Google CEO
    ("AMZN", "2021-02-02", False),  # Bezos -> Jassy (planned transition)
    ("JPM",  "2021-05-18", False),  # succession planning disclosed (orderly)
    ("DIS",  "2020-02-25", False),  # Iger -> Chapek (planned, then noisy)
    ("KO",   "2016-12-09", False),  # Muhtar Kent -> James Quincey (planned)
    ("PG",   "2015-08-04", False),  # A.G. Lafley -> David Taylor (planned)
    ("IBM",  "2020-01-30", False),  # Ginni Rometty -> Arvind Krishna (planned)
    ("NKE",  "2019-10-22", False),  # Mark Parker -> John Donahoe (planned)
    ("MCD",  "2019-11-04", False),  # Steve Easterbrook out, Kempczinski (planned-named)
    ("WMT",  "2013-11-25", False),  # Mike Duke -> Doug McMillon (planned)
]

# De-duplicate by (ticker, announce_date) and build the canonical table.
_seen: set = set()
CEO_EVENTS: list[dict] = []
for _t, _d, _f in _RAW_EVENTS:
    _key = (_t, _d)
    if _key in _seen:
        continue
    _seen.add(_key)
    CEO_EVENTS.append(
        {"ticker": _t, "announce_date": pd.Timestamp(_d), "forced": bool(_f)}
    )
CEO_EVENTS.sort(key=lambda r: r["announce_date"])

TICKERS = sorted({r["ticker"] for r in CEO_EVENTS})


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_391_{safe}_1d.parquet")


def fetch_prices(start: str = "2008-01-01", end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download daily adjusted closes for every event ticker + SPY and cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. One parquet per ticker (column ``close``, index ``date``).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in TICKERS + ["SPY"]:
        raw = yf.download(ticker, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.empty or "close" not in raw.columns:
            continue
        out = raw[["close"]].copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out.index.name = "date"
        out.to_parquet(_cache_path(ticker, cache_dir))


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff SPY and at least most event tickers are cached."""
    if not os.path.exists(_cache_path("SPY", cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in TICKERS)
    return have >= max(1, int(0.6 * len(TICKERS)))


def load_prices(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load cached closes into a wide frame (index = date, columns = tickers + SPY)."""
    series = {}
    for ticker in TICKERS + ["SPY"]:
        p = _cache_path(ticker, cache_dir)
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)["close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        series[ticker] = s
    df = pd.DataFrame(series).sort_index()
    return df


def load_real(cache_dir: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, list[dict]]:
    """Convenience: cached wide-price frame + the event table (only events with data)."""
    prices = load_prices(cache_dir)
    events = [e for e in CEO_EVENTS if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_forced: int = 14, n_planned: int = 12,
                     car_bps: float = 0.0, seed: int = 391,
                     est_days: int = 120, sig_daily: float = 0.018,
                     beta: float = 1.1) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable CAR edge.

    For each synthetic event we draw an estimation window of market + idiosyncratic
    returns and a short event window. The stock return is ``alpha + beta*mkt + eps``;
    on the event day a **planted abnormal jump** of ``car_bps`` basis points is added to
    the FORCED bucket only (the believers' "firing is good/bad news" effect we want the
    engine to recover). With ``car_bps = 0`` there is no planted effect and the inference
    must NOT find significance out of ~a dozen events per bucket.

    Returns a dict with arrays keyed by bucket:
      ``forced_car``, ``planned_car``  — event-window CAR per event (market-model abnormal)
      ``forced_win``, ``planned_win``  — same, sign only (for the win-rate)
      ``base_car``                     — abnormal CAR on random non-event windows (base rate)
      ``truth``                        — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    jump = car_bps * 1e-4
    win = 3                      # event-window length in trading days (CAR[0..2])

    def one_event(is_forced: bool) -> tuple[float, float]:
        # market + idiosyncratic returns over estimation + event window
        n = est_days + win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        alpha = 0.0
        stock = alpha + beta * mkt + eps
        # fit the market model on the estimation window (clean, pre-event)
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], stock[est], 1)
        ev = slice(est_days, est_days + win)
        abn = stock[ev] - (a + b * mkt[ev])
        if is_forced and jump != 0.0:
            abn[0] += jump          # plant the abnormal jump on the announcement day
        car = float(abn.sum())
        return car, float(np.sign(car) > 0)

    forced = [one_event(True) for _ in range(n_forced)]
    planned = [one_event(False) for _ in range(n_planned)]
    # base rate: random non-event windows (no planted jump, any bucket structure)
    base = []
    for _ in range(2000):
        n = est_days + win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        b, a = np.polyfit(mkt[:est_days], stock[:est_days], 1)
        ev = slice(est_days, est_days + win)
        abn = stock[ev] - (a + b * mkt[ev])
        base.append(float(abn.sum()))

    return {
        "forced_car": np.array([c for c, _ in forced]),
        "planned_car": np.array([c for c, _ in planned]),
        "forced_win": np.array([w for _, w in forced]),
        "planned_win": np.array([w for _, w in planned]),
        "base_car": np.array(base),
        "truth": {"n_forced": n_forced, "n_planned": n_planned,
                  "car_bps": car_bps, "seed": seed, "win": win},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["announce_date"]).value for e in events],
                   dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
