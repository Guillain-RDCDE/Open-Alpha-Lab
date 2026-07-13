"""Data layer for Study 751 — Fortune-500-Inclusion (event-window abnormal returns).

Two sources, both offline-friendly.

* **Real tape.** A hardcoded, transparent, *cited* table of ~26 notable Fortune-500
  **debuts** (first-time entrants) and **exits** (notable drops), each snapped to that
  year's **list-reveal date** (``FORTUNE_EVENTS``: ticker, list date, added-flag), plus
  daily adjusted closes for each ticker and SPY (yfinance, no key), cached under ``_cache/``
  as one parquet per ticker. From those we compute, for each event, the **abnormal return**
  (stock return minus a market-model fit on a clean pre-event estimation window) and cumulate
  it over short event windows (CAR) around the reveal.

  A **LABELLED PROXY, said plainly.** Fortune does not sell a free point-in-time
  membership feed, so the debut/exit *year* is **curated & approximate** — a company's exact
  first (or last) appearance can be off by a year, and we snap each to its list-reveal date.
  This is the transparent stand-in (exactly as Study 391 uses a hardcoded CEO-change table in
  place of a paid ExecuComp feed). Every input is a public price and a public, citable list.
  The point-estimate direction is not the load-bearing number; the *sample size* and the
  *absence of a demand shock* are.

  **Survivorship, named.** The worst exits are **bankruptcies** whose series ends — J.C.
  Penney (2020) and, later, Bed Bath & Beyond (2023) — see :data:`DELISTED_EXITS`. A tape
  that keeps only the survivors is biased *against* a negative drop-reaction, so a
  survivor-only "exit CAR" that is not sharply negative is a **conservative** read.

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-return
  paths with a *plantable* CAR edge (``car_bps``) on the ADDED bucket. It is the positive
  control: with the edge set to zero the inference must NOT manufacture significance out of a
  dozen events; with a large planted edge it must light up.

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
# Fortune-500 list-reveal dates (the trading day the annual ranking went public).
# Fortune publishes the 500 once a year, ranked on the PRIOR fiscal year's revenue
# (already-public information). Dates are the magazine's public reveal for that year.
# --------------------------------------------------------------------------- #
LIST_DATE = {
    2017: "2017-06-07",
    2018: "2018-05-21",
    2019: "2019-05-16",
    2020: "2020-05-18",
    2021: "2021-06-02",
    2022: "2022-05-23",
    2023: "2023-06-05",
    2024: "2024-06-04",
}

# --------------------------------------------------------------------------- #
# Hardcoded add/drop event table.
# Columns: ticker, list_year, added (True = notable DEBUT / first appearance;
#          False = notable EXIT / drop off the list that year).
# Sources: Fortune's annual Fortune-500 list & contemporaneous coverage of new
# entrants and drop-offs (Fortune.com, CNBC, company filings). "Debut" = widely
# reported first appearance; "Exit" = widely reported fall-off. The year is curated
# and approximate (a labelled proxy — see the module docstring); each event is
# snapped to that year's LIST_DATE. Large, listed names only, so price history is
# clean around the reveal.
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # --- notable DEBUTS (first appearance on the Fortune 500) ---
    ("TSLA", 2017, True),   # Tesla debuts (~#383, on 2016 revenue)
    ("NFLX", 2018, True),   # Netflix debuts (~#261)
    ("UBER", 2020, True),   # Uber debuts (~#228, first list after 2019 IPO)
    ("CVNA", 2021, True),   # Carvana debuts
    ("MRNA", 2022, True),   # Moderna debuts (~#195, COVID-vaccine revenue)
    ("ABNB", 2022, True),   # Airbnb debuts
    ("COIN", 2022, True),   # Coinbase debuts
    ("ZM",   2022, True),   # Zoom debuts
    ("ETSY", 2022, True),   # Etsy debuts
    ("DASH", 2023, True),   # DoorDash debuts
    ("PANW", 2023, True),   # Palo Alto Networks debuts
    ("CRWD", 2023, True),   # CrowdStrike debuts
    ("SNAP", 2023, True),   # Snap debuts
    ("HOOD", 2024, True),   # Robinhood debuts
    # --- notable EXITS (fell off the list) ---
    ("RL",   2020, False),  # Ralph Lauren drops off (pandemic revenue hit)
    ("MAT",  2019, False),  # Mattel drops off
    ("HOG",  2020, False),  # Harley-Davidson drops off
    ("GME",  2021, False),  # GameStop drops off
    ("XRX",  2022, False),  # Xerox drops off (post-split)
    ("BBBY", 2022, False),  # Bed Bath & Beyond drops off (later bankrupt — see DELISTED_EXITS)
    ("LEG",  2023, False),  # Leggett & Platt drops off
    ("NWL",  2023, False),  # Newell Brands drops off
    ("WU",   2023, False),  # Western Union drops off
    ("HAS",  2024, False),  # Hasbro drops off (revenue decline)
    ("GAP",  2024, False),  # Gap drops off (ticker GPS -> GAP in 2024)
    ("VFC",  2024, False),  # VF Corp drops off
]

# Famous EXITS that DELISTED / went bankrupt and leave no clean continuing series.
# Named for the survivorship caveat: the worst drop-reactions left the tape entirely,
# so the surviving exit sample is biased AGAINST a negative drop CAR.
DELISTED_EXITS = [
    "J.C. Penney (JCP) — fell off ~2020, Chapter 11 in 2020, delisted",
    "Bed Bath & Beyond (BBBY) — fell off ~2022, Chapter 11 in 2023, delisted",
    "Sears Holdings — long gone from the list, bankrupt 2018",
]

# De-duplicate by (ticker, list_year) and build the canonical table.
_seen: set = set()
FORTUNE_EVENTS: list[dict] = []
for _t, _y, _a in _RAW_EVENTS:
    _key = (_t, _y)
    if _key in _seen:
        continue
    _seen.add(_key)
    FORTUNE_EVENTS.append(
        {"ticker": _t, "list_year": int(_y),
         "list_date": pd.Timestamp(LIST_DATE[_y]), "added": bool(_a)}
    )
FORTUNE_EVENTS.sort(key=lambda r: (r["list_date"], r["ticker"]))

TICKERS = sorted({r["ticker"] for r in FORTUNE_EVENTS})


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_751_{safe}_1d.parquet")


def fetch_prices(start: str = "2015-01-01", end: str | None = None,
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
    events = [e for e in FORTUNE_EVENTS if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_added: int = 14, n_dropped: int = 12,
                     car_bps: float = 0.0, seed: int = 751,
                     est_days: int = 120, sig_daily: float = 0.024,
                     beta: float = 1.15) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable CAR edge.

    For each synthetic event we draw an estimation window of market + idiosyncratic
    returns and a short event window. The stock return is ``alpha + beta*mkt + eps``; on the
    reveal day a **planted abnormal jump** of ``car_bps`` basis points is added to the ADDED
    bucket only (the believers' "making the list lifts the stock" effect we want the engine
    to recover). With ``car_bps = 0`` there is no planted effect and the inference must NOT
    find significance out of ~a dozen events per bucket.

    Returns a dict with arrays keyed by bucket:
      ``added_car``, ``dropped_car``  — event-window CAR per event (market-model abnormal)
      ``added_win``, ``dropped_win``  — same, sign only (for the win-rate)
      ``base_car``                    — abnormal CAR on random non-event windows (base rate)
      ``truth``                       — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    jump = car_bps * 1e-4
    win = 3                      # event-window length in trading days (CAR[0..2])

    def one_event(is_added: bool) -> tuple[float, float]:
        n = est_days + win + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        alpha = 0.0
        stock = alpha + beta * mkt + eps
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], stock[est], 1)
        ev = slice(est_days, est_days + win)
        abn = stock[ev] - (a + b * mkt[ev])
        if is_added and jump != 0.0:
            abn[0] += jump          # plant the abnormal jump on the reveal day
        car = float(abn.sum())
        return car, float(np.sign(car) > 0)

    added = [one_event(True) for _ in range(n_added)]
    dropped = [one_event(False) for _ in range(n_dropped)]
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
        "added_car": np.array([c for c, _ in added]),
        "dropped_car": np.array([c for c, _ in dropped]),
        "added_win": np.array([w for _, w in added]),
        "dropped_win": np.array([w for _, w in dropped]),
        "base_car": np.array(base),
        "truth": {"n_added": n_added, "n_dropped": n_dropped,
                  "car_bps": car_bps, "seed": seed, "win": win},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (dates + direction), for as-of stamps."""
    arr = np.array(
        [pd.Timestamp(e["list_date"]).value + (1 if e["added"] else 0)
         for e in events], dtype=np.int64)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
