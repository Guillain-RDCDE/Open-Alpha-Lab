"""Data layer for Study 717 — Person-of-the-Year (magazine-cover-curse event study).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of the TIME *Person of the Year* honorees
  who ran (or were the public face of) a tradable company (``POY_EVENTS``: ticker,
  announcement date, honoree, ``direct`` flag), plus daily adjusted (total-return) closes
  for each ticker and SPY (yfinance, no key), cached under ``_cache/`` as one parquet per
  ticker. From those we compute, for each coronation, the **abnormal return** (stock return
  minus a market-model fit on a clean pre-event estimation window) cumulated over long
  post-announcement windows (1 / 3 / 6 / 12 months). Most Persons of the Year are
  politicians or abstract groups with no ticker at all; the dated, labelled table is the
  transparent census of the *tradable* ones — every input is a public price and a public,
  citable cover.

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* post-announcement drift edge (``curse_bps``). It is the
  positive control: with the edge set to zero the inference must NOT manufacture a "curse"
  out of four events; with a large planted decline it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# --------------------------------------------------------------------------- #
# Hardcoded Person-of-the-Year honoree table (the tradable ones).
# Columns: ticker, announce_date (the mid-December day TIME's pick hit the tape),
#          honoree, direct (True = honored primarily AS the company's boss;
#          False = a public-company link but honored for something else).
# Sources: TIME's own Person-of-the-Year archive (time.com/person-of-the-year), plus
# contemporaneous coverage. Announcement dates are the public reveal (mid/late December);
# the exact trading day is resolved by searchsorted, so being within a day is harmless.
# The ``direct`` label is the believers' framing ("the CEO on the cover") and is somewhat
# subjective at the margin — we say so on the Signal axis. Only names with a public stock
# AT the announcement are kept; the most business-y honorees who were NOT yet public are
# listed in _DROPPED (named on the Signal axis — a survivorship note).
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # (ticker, announce_date, honoree, direct)
    ("AMZN", "1999-12-27", "Jeff Bezos (Amazon)",          True),   # dot-com cover icon
    ("MSFT", "2005-12-19", "Bill Gates (Good Samaritans)", False),  # honored for philanthropy
    ("TSLA", "2021-12-13", "Elon Musk (Tesla/SpaceX)",     True),   # EV-bubble zenith
    ("DJT",  "2024-12-12", "Donald Trump (Trump Media)",   False),  # president-elect, namesake co
]

# Named-but-untradable business honorees (no price history at the announcement) — the
# survivorship note, spelled out on the Signal axis. These are the biggest business picks
# of their era, and dropping them tilts the tiny sample; we say so, in the open.
_DROPPED = [
    ("META", "2010-12-15", "Mark Zuckerberg (Facebook)", "IPO 2012-05-18 — not public at the coronation"),
    ("DJT",  "2016-12-07", "Donald Trump (president-elect)", "no public company until the 2024 DWAC/DJT merger"),
]

_seen: set = set()
POY_EVENTS: list[dict] = []
for _t, _d, _h, _dir in _RAW_EVENTS:
    _key = (_t, _d)
    if _key in _seen:
        continue
    _seen.add(_key)
    POY_EVENTS.append(
        {"ticker": _t, "announce_date": pd.Timestamp(_d),
         "honoree": _h, "direct": bool(_dir)}
    )
POY_EVENTS.sort(key=lambda r: r["announce_date"])

TICKERS = sorted({r["ticker"] for r in POY_EVENTS})


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_717_{safe}_1d.parquet")


def fetch_prices(start: str = "1997-01-01", end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download daily adjusted (total-return) closes for every honoree ticker + SPY.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. One parquet per ticker (column ``close``, index ``date``). ``auto_adjust=True``
    so the series are **total-return** (dividends reinvested) — labelled as such wherever
    a return is quoted; SPY is the total-return benchmark on the same convention.
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
    """True iff SPY and at least most honoree tickers are cached."""
    if not os.path.exists(_cache_path("SPY", cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in TICKERS)
    return have >= max(1, int(0.75 * len(TICKERS)))


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
    """Convenience: cached wide-price frame + the honoree table (only events with data)."""
    prices = load_prices(cache_dir)
    events = [e for e in POY_EVENTS if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 4, curse_bps: float = 0.0, seed: int = 717,
                     est_days: int = 120, horizon: int = 252,
                     sig_daily: float = 0.028, beta: float = 1.3) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable post-drift edge.

    For each synthetic honoree we draw an estimation window of market + idiosyncratic
    returns and a long post-announcement horizon. The stock return is
    ``alpha + beta*mkt + eps``; over the horizon a **planted cumulative abnormal drift** of
    ``curse_bps`` basis points is spread across the days (the believers' "cover curse"). With
    ``curse_bps = 0`` there is no planted effect and the inference must NOT find a curse out
    of four events; with a large negative planted drift it must light up.

    The idiosyncratic vol (``sig_daily``) and beta are deliberately meme-stock-sized — the
    real honorees are AMZN/TSLA/DJT-class names — so the control honestly reflects how much
    single-name noise a four-event mean CAR carries.

    Returns a dict:
      ``car``      — post-announcement CAR per event (market-model abnormal)
      ``base_car`` — abnormal CAR on random non-event windows (base rate)
      ``truth``    — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    drift_total = curse_bps * 1e-4

    def one_event() -> float:
        n = est_days + horizon + 5
        mkt = rng.normal(0.0003, 0.011, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], stock[est], 1)
        ev = slice(est_days, est_days + horizon)
        abn = stock[ev] - (a + b * mkt[ev])
        if drift_total != 0.0:
            abn += drift_total / horizon        # spread the planted drift across the window
        return float(abn.sum())

    car = np.array([one_event() for _ in range(n_events)])
    base = []
    for _ in range(3000):
        n = est_days + horizon + 5
        mkt = rng.normal(0.0003, 0.011, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        b, a = np.polyfit(mkt[:est_days], stock[:est_days], 1)
        ev = slice(est_days, est_days + horizon)
        abn = stock[ev] - (a + b * mkt[ev])
        base.append(float(abn.sum()))

    return {
        "car": car,
        "base_car": np.array(base),
        "truth": {"n_events": n_events, "curse_bps": curse_bps, "seed": seed,
                  "horizon": horizon},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the honoree table (dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["announce_date"]).value for e in events],
                   dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
