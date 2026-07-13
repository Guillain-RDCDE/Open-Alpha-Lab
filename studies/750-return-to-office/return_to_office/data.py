"""Data layer for Study 750 — Return-to-Office (did RTO mandates move office REITs?).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~26 documented **RTO-mandate
  announcements by big employers** (``RTO_EVENTS``: date, employer, strict/hybrid flag),
  plus daily adjusted closes for a small **office-REIT basket** and benchmarks (SPY and the
  broad-REIT ETF VNQ), from yfinance (no key), cached under ``_cache/`` as one parquet per
  ticker. The believers' claim is a *sector* reaction, not a single stock: when a marquee
  employer orders workers back, office landlords (desks refill, leases firm up) should pop.
  So the object we price is the **office-REIT basket's abnormal return** around each mandate
  — a market-model CAR (basket vs SPY), estimated on a clean pre-event window.

  The RTO calendar is the honest part: every row is a real, dated, citable mandate. The
  *survivorship* is named loudly on the Signal axis — the worst-hit office landlords left
  the tape entirely (**WeWork** delisted in 2023; scores of private towers handed the keys
  back via CMBS default), so the surviving basket is biased *toward* the names that did not
  collapse. That bias trims the downside, but for a short-window *reaction* study it mostly
  pushes the estimate **toward zero**, so a survivor basket that fails to pop is a
  conservative read of "RTO rescued the office."

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* CAR edge (``car_bps``) on the strict bucket. It is the
  positive control: with the edge at zero the inference must NOT manufacture significance
  out of ~two dozen events; with a large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.

An additional **labelled PROXY** (``KASTLE_OCCUPANCY``) carries the physical
back-to-the-office trend (Kastle Systems' 10-city "Back to Work Barometer"). It is a
small, cited, *approximate* hardcoded series used only for context — never priced, never
under a real-tape banner. See ``docs/references.md``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# --------------------------------------------------------------------------- #
# The office-REIT basket + benchmarks.
#   * Pure-ish office landlords with long, clean daily history on yfinance.
#   * SPY  = the broad-market benchmark for the market model.
#   * VNQ  = broad-REIT ETF, used in a robustness pass ("does office react
#            beyond what ALL REITs did that day?").
# --------------------------------------------------------------------------- #
OFFICE_REITS = ["SLG", "BXP", "VNO", "KRC", "HIW", "DEI", "CUZ", "HPP", "BDN", "ESRT"]
MARKET = "SPY"
BROAD_REIT = "VNQ"
BENCH_TICKERS = [MARKET, BROAD_REIT]

# Famous office-exposure names that DELISTED / collapsed (no continuing series).
# Named for the survivorship caveat: the priced basket is biased AGAINST the office
# distress the RTO story is supposed to reverse — the worst casualties left the tape.
DELISTED = [
    "WeWork Inc. (WE) — bankrupt Nov 2023, delisted (the flex-office poster child)",
    "hundreds of private office towers — handed to lenders via CMBS default 2023-24",
    "Office Properties Income Trust (OPI) — distressed debt exchange, reverse split 2024",
]

# --------------------------------------------------------------------------- #
# Hardcoded RTO-mandate event table.
# Columns: date (the trading day the mandate hit the tape), employer, strict.
#   * strict = True  -> a FULL in-office mandate (5 days, or an unambiguous "come back"
#                       with no hybrid carve-out) — the strongest form of the signal.
#   * strict = False -> a HYBRID mandate (2-4 days/week, "most of the week", flexible).
# Sources: company memos & contemporaneous financial-press coverage (WSJ / Reuters /
# Bloomberg / CNBC / FT). Dates are the documented announcement day (approximate to the
# trading day); the engine snaps each to the nearest available price date, so day-level
# precision is not required. The strict/hybrid label is the believers' own framing
# ("a REAL mandate should pop offices more than a soft hybrid") and is subjective at the
# margin — we say so on the Signal axis.
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # (date, employer, strict)
    # ---- 2021: the first "everyone back" wave (finance leads) ----------------
    ("2021-02-25", "Goldman Sachs (Solomon: WFH an 'aberration')", True),
    ("2021-05-04", "JPMorgan (Dimon orders U.S. staff back)", True),
    ("2021-06-14", "Morgan Stanley (Gorman: 'if you can go to a restaurant…')", True),
    ("2021-09-07", "Bank of America (back after Labor Day push)", True),
    # ---- 2022: big tech sets a hybrid template, then Musk breaks it ----------
    ("2022-03-04", "Apple (3-day hybrid mandate)", False),
    ("2022-03-30", "Google (3-day hybrid from April)", False),
    ("2022-05-03", "Microsoft (hybrid, 50%+ in office)", False),
    ("2022-06-01", "Tesla (Musk email: 40h in office or leave)", True),
    ("2022-09-16", "Comcast/NBCU (4-day in office)", True),
    ("2022-11-10", "Twitter/X (Musk ends remote work)", True),
    # ---- 2023: the enforcement year -----------------------------------------
    ("2023-01-09", "Disney (Iger: 4 days in office)", True),
    ("2023-01-30", "Starbucks (corporate 3-day)", False),
    ("2023-02-17", "Amazon (3-day RTO from May)", False),
    ("2023-05-04", "AT&T (relocate to 9 core hubs)", True),
    ("2023-06-01", "BlackRock (4 days in office)", True),
    ("2023-06-15", "Meta (3-day RTO from September)", False),
    ("2023-08-10", "Salesforce (3+ days for most roles)", False),
    ("2023-09-11", "Google (tightens 3-day, badge tracking)", False),
    # ---- 2024: the 5-day-return escalation ----------------------------------
    ("2024-01-16", "IBM (managers RTO or resign)", True),
    ("2024-02-05", "Dell (sales/most staff full RTO)", True),
    ("2024-05-14", "Walmart (relocate + corporate RTO)", True),
    ("2024-09-16", "Amazon (full 5-day RTO from Jan 2025)", True),
    ("2024-10-14", "AT&T (5 days a week)", True),
    # ---- 2025: banks + the federal government go all-in ----------------------
    ("2025-01-10", "JPMorgan (all employees 5 days)", True),
    ("2025-01-20", "US federal government (Trump RTO order)", True),
    ("2025-02-27", "Starbucks (corporate 4-day escalation)", True),
]

RTO_EVENTS: list[dict] = []
for _d, _emp, _s in _RAW_EVENTS:
    RTO_EVENTS.append({"date": pd.Timestamp(_d), "employer": _emp, "strict": bool(_s)})
RTO_EVENTS.sort(key=lambda r: r["date"])

TICKERS = sorted(set(OFFICE_REITS))


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus benchmarks
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_750_{safe}_1d.parquet")


def fetch_prices(start: str = "2018-06-01", end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download daily adjusted closes for the office basket + SPY + VNQ and cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. One parquet per ticker (column ``close``, index ``date``).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in TICKERS + BENCH_TICKERS:
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
    """True iff SPY and most office-REIT tickers are cached."""
    if not os.path.exists(_cache_path(MARKET, cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in TICKERS)
    return have >= max(1, int(0.7 * len(TICKERS)))


def load_prices(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load cached closes into a wide frame (index = date, columns = tickers + benchmarks)."""
    series = {}
    for ticker in TICKERS + BENCH_TICKERS:
        p = _cache_path(ticker, cache_dir)
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)["close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        series[ticker] = s
    return pd.DataFrame(series).sort_index()


def load_real(cache_dir: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, list[dict]]:
    """Convenience: cached wide-price frame + the RTO event table."""
    prices = load_prices(cache_dir)
    return prices, list(RTO_EVENTS)


def members_present(prices: pd.DataFrame) -> list[str]:
    """The office-REIT basket members that actually have a column in the cached frame."""
    return [t for t in OFFICE_REITS if t in prices.columns]


# --------------------------------------------------------------------------- #
# Labelled PROXY — physical office occupancy (Kastle "Back to Work Barometer")
# --------------------------------------------------------------------------- #
# APPROXIMATE, CITED, hardcoded. Kastle Systems' 10-city average of badge swipes vs a
# pre-COVID (Feb 2020 = 100) baseline, read off Kastle's public weekly chart at roughly
# quarter-ends. This is a PROXY for the physical RTO trend, used ONLY for a context chart
# (how far desks actually refilled) — it is never priced and never under a real-tape
# banner. Source: Kastle Systems, kastle.com/safety-wellness/getting-america-back-to-work.
KASTLE_OCCUPANCY = [
    ("2021-03-31", 24), ("2021-06-30", 32), ("2021-09-30", 34), ("2021-12-31", 33),
    ("2022-03-31", 43), ("2022-06-30", 44), ("2022-09-30", 47), ("2022-12-31", 47),
    ("2023-03-31", 49), ("2023-06-30", 50), ("2023-09-30", 50), ("2023-12-31", 51),
    ("2024-03-31", 52), ("2024-06-30", 51), ("2024-09-30", 52), ("2024-12-31", 53),
    ("2025-03-31", 54),
]


def kastle_proxy() -> pd.Series:
    """The approximate, cited Kastle 10-city office-occupancy proxy (Feb-2020 = 100).

    LABELLED A PROXY: reconstructed from Kastle's public weekly chart at quarter-ends, not
    a live data feed. Used for context only (the physical RTO trend), never priced.
    """
    idx = pd.to_datetime([d for d, _ in KASTLE_OCCUPANCY])
    return pd.Series([v for _, v in KASTLE_OCCUPANCY], index=idx, name="kastle_occupancy")


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_strict: int = 15, n_hybrid: int = 11,
                     car_bps: float = 0.0, seed: int = 750,
                     est_days: int = 120, sig_daily: float = 0.014,
                     beta: float = 1.05) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable strict-bucket edge.

    For each synthetic RTO event we draw an estimation window of market + idiosyncratic
    basket returns and a short event window. The basket return is ``alpha + beta*mkt +
    eps``; on the event day a **planted abnormal jump** of ``car_bps`` basis points is
    added to the STRICT bucket only (the believers' "a real mandate refills the towers"
    effect we want the engine to recover). With ``car_bps = 0`` there is no planted effect
    and the inference must NOT find significance out of ~two dozen events.

    Returns a dict with arrays keyed by bucket:
      ``strict_car``, ``hybrid_car``  — event-window CAR per event (market-model abnormal)
      ``strict_win``, ``hybrid_win``  — sign only (for the win-rate)
      ``base_car``                    — abnormal CAR on random non-event windows (base rate)
      ``truth``                       — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    jump = car_bps * 1e-4
    win = 3                      # event-window length in trading days (CAR[0..2])

    def one_event(is_strict: bool) -> tuple[float, float]:
        n = est_days + win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        basket = beta * mkt + eps
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], basket[est], 1)
        ev = slice(est_days, est_days + win)
        abn = basket[ev] - (a + b * mkt[ev])
        if is_strict and jump != 0.0:
            abn[0] += jump          # plant the abnormal jump on the announcement day
        car = float(abn.sum())
        return car, float(np.sign(car) > 0)

    strict = [one_event(True) for _ in range(n_strict)]
    hybrid = [one_event(False) for _ in range(n_hybrid)]
    base = []
    for _ in range(2000):
        n = est_days + win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        basket = beta * mkt + eps
        b, a = np.polyfit(mkt[:est_days], basket[:est_days], 1)
        ev = slice(est_days, est_days + win)
        abn = basket[ev] - (a + b * mkt[ev])
        base.append(float(abn.sum()))

    return {
        "strict_car": np.array([c for c, _ in strict]),
        "hybrid_car": np.array([c for c, _ in hybrid]),
        "strict_win": np.array([w for _, w in strict]),
        "hybrid_win": np.array([w for _, w in hybrid]),
        "base_car": np.array(base),
        "truth": {"n_strict": n_strict, "n_hybrid": n_hybrid,
                  "car_bps": car_bps, "seed": seed, "win": win},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["date"]).value for e in events], dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
