"""Data layer for Study 746 — HQ-Relocation (event-window abnormal returns).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~20 documented corporate
  **headquarters relocations, 2010-2025** (``HQ_MOVES``: ticker, announcement date,
  ``tax`` flag = the move's stated primary motive was a tax/incentive advantage —
  an inversion abroad or a jump to a lower-tax U.S. state — versus some *other*
  rationale, talent/cost/proximity). Plus daily adjusted closes for each ticker and
  SPY (yfinance, no key), cached under ``_cache/`` as one parquet per ticker. From
  those we compute, for each event, the **abnormal return** (stock return minus a
  market-model fit on a clean pre-event estimation window) and cumulate it over short
  event windows (CAR) around the announcement, plus a longer post-announcement
  **drift** leg. A tidy, machine-readable relocation database is not freely available,
  so the dated, labelled table is the transparent stand-in — every input is a public
  price and a public, citable headline.

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* CAR edge (``car_bps``). It is the positive control:
  with the edge set to zero the inference must NOT manufacture significance out of a
  couple-dozen events; with a large planted edge it must light up.

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
# Hardcoded HQ-relocation event table.
# Columns: ticker, announce_date (the trading day the move hit the tape),
#          tax (True  = the stated primary motive was a tax / incentive edge —
#                       an inversion abroad or a move to a lower-tax U.S. state;
#               False = some other rationale: talent, cost of living, proximity).
# Sources: company press releases, proxy/8-K filings, and contemporaneous
# financial-press coverage (WSJ / Reuters / Bloomberg / FT). The tax/other label is
# the believers' own framing ("they only moved to dodge taxes") and is, of course,
# somewhat subjective at the margin (a Texas move cuts both cost-of-living AND state
# tax) — we say so, loudly, on the Signal axis. Announcement dates are the documented
# announcement day rounded to a trading-day-ish date; the engine snaps each event to
# the nearest available price date, so day-level precision is not required. Large,
# long-listed names only, so the market-model estimation window is clean around the
# date.
# --------------------------------------------------------------------------- #
_RAW_MOVES = [
    # (ticker, announce_date, tax)
    # --- tax / incentive-motivated moves (inversion abroad or low-tax U.S. state) ---
    ("AON",  "2012-01-13", True),   # Aon -> London (UK tax domicile)
    ("ETN",  "2012-05-21", True),   # Eaton -> Ireland (Cooper inversion)
    ("PRGO", "2013-07-29", True),   # Perrigo -> Ireland (Elan inversion)
    ("ADM",  "2013-09-30", True),   # ADM -> Chicago (sought state tax incentives)
    ("MDT",  "2014-06-16", True),   # Medtronic -> Ireland (Covidien inversion)
    ("PFE",  "2015-11-23", True),   # Pfizer -> Ireland (Allergan inversion; later blocked)
    ("JCI",  "2016-01-25", True),   # Johnson Controls -> Ireland (Tyco inversion)
    ("SCHW", "2019-11-25", True),   # Schwab -> Texas (Westlake, with TD deal)
    ("CBRE", "2020-10-30", True),   # CBRE -> Texas (Dallas)
    ("HPE",  "2020-12-01", True),   # HPE -> Texas (Houston)
    ("ORCL", "2020-12-11", True),   # Oracle -> Texas (Austin)
    ("TSLA", "2021-10-07", True),   # Tesla -> Texas (Austin)
    ("CAT",  "2022-06-14", True),   # Caterpillar -> Texas (Irving)
    ("CVX",  "2024-08-02", True),   # Chevron -> Texas (Houston)
    # --- other rationale (talent / cost / proximity, not primarily tax) ---
    ("WY",   "2014-06-25", False),  # Weyerhaeuser -> downtown Seattle (urban/talent)
    ("EXPE", "2015-04-08", False),  # Expedia -> Seattle Interbay campus (talent)
    ("GE",   "2016-01-13", False),  # GE -> Boston (innovation/talent ecosystem)
    ("MCD",  "2016-06-13", False),  # McDonald's -> downtown Chicago (talent/urban)
    ("HON",  "2018-11-09", False),  # Honeywell -> Charlotte NC (cost/talent)
    ("BA",   "2022-05-05", False),  # Boeing -> Arlington VA (proximity to regulators/DC)
]

# Famous HQ-relocation stories that we CANNOT price cleanly around the event, listed
# for the selection/survivorship caveat (named on the Signal axis): the tape we test
# is the set of moves by names that were already public with a clean pre-event window.
UNPRICEABLE = [
    "Burger King -> Canada (2014, Tim Hortons inversion; QSR IPO'd only that Dec)",
    "Mylan -> Netherlands (2015 Abbott-generics inversion; MYL folded into VTRS 2020)",
    "Walgreens abandoned Swiss inversion (2014 — a move that reversed, a non-event)",
    "Toyota N.A. -> Plano, Texas (2014); Nestle USA -> Arlington (2017) — foreign parents",
    "Chiquita -> Ireland (2014 Fyffes inversion, later abandoned)",
]

TICKERS = sorted({t for t, *_ in _RAW_MOVES})

# De-duplicate by (ticker, announce_date) and build the canonical table.
_seen: set = set()
HQ_MOVES: list[dict] = []
for _t, _d, _tax in _RAW_MOVES:
    _key = (_t, _d)
    if _key in _seen:
        continue
    _seen.add(_key)
    HQ_MOVES.append(
        {"ticker": _t, "announce_date": pd.Timestamp(_d), "tax": bool(_tax)}
    )
HQ_MOVES.sort(key=lambda r: r["announce_date"])


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_746_{safe}_1d.parquet")


def fetch_prices(start: str = "2010-01-01", end: str | None = None,
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
    events = [e for e in HQ_MOVES if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_tax: int = 14, n_other: int = 12,
                     car_bps: float = 0.0, seed: int = 746,
                     est_days: int = 120, sig_daily: float = 0.018,
                     beta: float = 1.05) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable CAR edge.

    For each synthetic event we draw an estimation window of market + idiosyncratic
    returns and a short event window. The stock return is ``alpha + beta*mkt + eps``;
    on the event day a **planted abnormal jump** of ``car_bps`` basis points is added to
    the TAX bucket only (the believers' "the market prices the tax saving" effect we want
    the engine to recover). With ``car_bps = 0`` there is no planted effect and the
    inference must NOT find significance out of a couple-dozen events per bucket.

    Returns a dict with arrays keyed by bucket:
      ``tax_car``, ``other_car``  — event-window CAR per event (market-model abnormal)
      ``tax_win``, ``other_win``  — same, sign only (for the win-rate)
      ``base_car``                — abnormal CAR on random non-event windows (base rate)
      ``truth``                   — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    jump = car_bps * 1e-4
    win = 3                      # event-window length in trading days (CAR[0..2])

    def one_event(is_tax: bool) -> tuple[float, float]:
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
        if is_tax and jump != 0.0:
            abn[0] += jump          # plant the abnormal jump on the announcement day
        car = float(abn.sum())
        return car, float(np.sign(car) > 0)

    tax = [one_event(True) for _ in range(n_tax)]
    other = [one_event(False) for _ in range(n_other)]
    # base rate: random non-event windows (no planted jump)
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
        "tax_car": np.array([c for c, _ in tax]),
        "other_car": np.array([c for c, _ in other]),
        "tax_win": np.array([w for _, w in tax]),
        "other_win": np.array([w for _, w in other]),
        "base_car": np.array(base),
        "truth": {"n_tax": n_tax, "n_other": n_other,
                  "car_bps": car_bps, "seed": seed, "win": win},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["announce_date"]).value for e in events],
                   dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
