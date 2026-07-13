"""Data layer for Study 749 — Layoff-Drift (event-window abnormal returns).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~28 well-known mass-layoff
  announcements from large-cap US firms, 2015–2025 (``LAYOFF_EVENTS``: ticker,
  announcement date, approximate headcount cut, one-line source), plus daily adjusted
  closes for each ticker and SPY (yfinance, no key), cached under ``_cache/`` as one
  parquet per ticker. From those we compute, for each event, the **abnormal return**
  (stock return minus a market-model fit on a clean pre-event estimation window) and
  cumulate it over a short **"restructuring pop"** window and a longer **PEAD-style
  drift** window after the announcement. There is no free, survivorship-clean database
  of mass-layoff dates, so the dated table is the transparent stand-in — every input is
  a public price and a public, citable headline (WSJ / Reuters / company press release).

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* pop and drift edge (``pop_bps``, ``drift_bps``). It is
  the positive control: with the edges set to zero the inference must NOT manufacture
  significance out of ~two dozen events; with a large planted drift it must light up.

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
# Hardcoded mass-layoff announcement table.
# Columns: ticker, announce_date (the trading day the cut hit the tape),
#          cut (approximate headcount reduction, for context / weighting only).
# Sources: company press releases & contemporaneous financial-press coverage
# (WSJ / Reuters / Bloomberg / FT / Layoffs.fyi). We pick large, long-listed names so
# the price history is clean around the date, and cuts that were *headline* events (a
# named number, a scheduled restructuring), not quiet attrition. The "restructuring pop"
# folklore is loudest for the 2022–2024 tech "efficiency" wave, so it is well represented,
# balanced by industrial / energy / COVID-era cuts for contrast. Dates are the first
# trading day the market could act on the announcement.
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # (ticker, announce_date, approx_cut)
    # --- 2022–2024 tech "efficiency wave" (where the pop folklore is loudest) ---
    ("META",  "2022-11-09", 11000),  # Meta's first mass layoff, 13% of staff
    ("META",  "2023-03-14", 10000),  # "year of efficiency" second round
    ("AMZN",  "2023-01-04", 18000),  # largest in Amazon history at the time
    ("GOOGL", "2023-01-20", 12000),  # Alphabet, ~6% of workforce
    ("MSFT",  "2023-01-18", 10000),  # ~5% of workforce
    ("CRM",   "2023-01-04", 8000),   # Salesforce, ~10% of staff
    ("DIS",   "2023-02-08", 7000),   # Disney restructuring under Iger
    ("SPOT",  "2023-12-04", 1500),   # Spotify, ~17% of staff (third 2023 round)
    ("SNAP",  "2022-08-31", 1300),   # Snap, ~20% of staff
    ("PYPL",  "2023-01-31", 2000),   # PayPal, ~7% of workforce
    ("ZM",    "2023-02-07", 1300),   # Zoom, ~15% of staff
    ("DELL",  "2023-02-06", 6650),   # Dell, ~5% of workforce
    ("HPQ",   "2022-11-22", 5000),   # HP, 4,000–6,000 over three years
    ("CSCO",  "2024-08-14", 7000),   # Cisco, ~7% of workforce (second 2024 round)
    ("INTC",  "2024-08-01", 15000),  # Intel, ~15% cut with cost-reduction plan
    ("NKE",   "2024-02-16", 1600),   # Nike, ~2% of workforce
    # --- financials / industrials ---
    ("GS",    "2023-01-09", 3200),   # Goldman Sachs, largest since 2008
    ("MMM",   "2023-01-24", 2500),   # 3M restructuring with Q4 results
    ("GE",    "2017-12-07", 12000),  # GE Power division
    ("F",     "2019-05-20", 7000),   # Ford salaried "Smart Redesign"
    ("GM",    "2018-11-26", 14000),  # GM plant closures / restructuring
    ("XOM",   "2020-10-29", 14000),  # ExxonMobil global headcount reduction
    ("CVX",   "2020-05-29", 6000),   # Chevron, 10–15% of workforce
    # --- earlier tech / COVID-era ---
    ("MSFT",  "2015-07-08", 7800),   # Microsoft phone-hardware writedown cut
    ("INTC",  "2016-04-19", 12000),  # Intel, ~11% of workforce
    ("UBER",  "2020-05-06", 3700),   # Uber, first COVID cut
    ("BA",    "2020-04-29", 16000),  # Boeing, ~10% of workforce (COVID)
    ("IBM",   "2023-01-25", 3900),   # IBM, ~1.5% with Q4 results
]

# De-duplicate by (ticker, announce_date) and build the canonical table.
_seen: set = set()
LAYOFF_EVENTS: list[dict] = []
for _t, _d, _c in _RAW_EVENTS:
    _key = (_t, _d)
    if _key in _seen:
        continue
    _seen.add(_key)
    LAYOFF_EVENTS.append(
        {"ticker": _t, "announce_date": pd.Timestamp(_d), "cut": int(_c)}
    )
LAYOFF_EVENTS.sort(key=lambda r: r["announce_date"])

TICKERS = sorted({r["ticker"] for r in LAYOFF_EVENTS})


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_749_{safe}_1d.parquet")


def fetch_prices(start: str = "2013-01-01", end: str | None = None,
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
    events = [e for e in LAYOFF_EVENTS if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 24, pop_bps: float = 0.0, drift_bps: float = 0.0,
                     seed: int = 749, est_days: int = 120, sig_daily: float = 0.020,
                     beta: float = 1.15, pop_win: int = 3, drift_win: int = 60) -> dict:
    """Deterministic per-event abnormal-return panel with plantable pop and drift edges.

    For each synthetic event we draw an estimation window of market + idiosyncratic
    returns plus a post-announcement window. The stock return is ``alpha + beta*mkt +
    eps``; on the pop window a **planted abnormal jump** of ``pop_bps`` basis points is
    spread over the first ``pop_win`` days, and a **planted drift** of ``drift_bps`` bps
    is spread over the following ``drift_win`` days (the PEAD-style continuation the
    believers claim). With both set to zero there is no planted effect and the inference
    must NOT find significance out of ~two dozen events.

    Returns a dict with per-event pop / drift CARs, plus the pooled daily abnormal-return
    drift series (for a HAC t-stat), and the planted ``truth``.
    """
    rng = np.random.default_rng(seed)
    pop_jump = pop_bps * 1e-4
    drift_jump = drift_bps * 1e-4

    pops, drifts = [], []
    daily_drift = []  # pooled daily abnormal returns over the drift window
    for _ in range(n_events):
        n = est_days + pop_win + drift_win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], stock[est], 1)
        pop_sl = slice(est_days, est_days + pop_win)
        drift_sl = slice(est_days + pop_win, est_days + pop_win + drift_win)
        abn_pop = stock[pop_sl] - (a + b * mkt[pop_sl])
        abn_drift = stock[drift_sl] - (a + b * mkt[drift_sl])
        # plant the effects
        abn_pop = abn_pop + pop_jump / pop_win
        abn_drift = abn_drift + drift_jump / drift_win
        pops.append(float(abn_pop.sum()))
        drifts.append(float(abn_drift.sum()))
        daily_drift.append(abn_drift)

    return {
        "pop": np.array(pops),
        "drift": np.array(drifts),
        "daily_drift": np.concatenate(daily_drift),
        "truth": {"n_events": n_events, "pop_bps": pop_bps, "drift_bps": drift_bps,
                  "seed": seed, "pop_win": pop_win, "drift_win": drift_win},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["announce_date"]).value for e in events],
                   dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
