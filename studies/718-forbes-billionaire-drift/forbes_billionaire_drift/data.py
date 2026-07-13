"""Data layer for Study 718 — Forbes-Billionaire-Drift (event-window abnormal returns).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~27 *newly-minted* Forbes billionaires
  whose fortune sits in a **publicly-traded vehicle** (``FORBES_EVENTS``: ticker, founder,
  the year they first appeared on the annual World's Billionaires list, and that list's
  publication date), plus daily adjusted closes for each ticker and SPY (yfinance, no key),
  cached under ``_cache/`` as one parquet per ticker. From those we compute, for each event,
  the **abnormal return** (stock return minus a market-model fit on a clean pre-event
  estimation window) cumulated over event windows (CAR) around the list date.

  Forbes does not license a machine-readable "new entrants" feed, so the dated, labelled
  table is the transparent stand-in — every input is a public price and a public, citable
  Forbes headline. The "newly-minted" call is Forbes' own framing (first appearance on the
  annual list) and, like any hand-compiled sample, is a judgement call at the margin (some
  founders were paper-billionaires pre-IPO; we date the event to the first annual list on
  which the *tradable vehicle* had public price history). We say so on the Signal axis.

* **Synthetic.** A deterministic, fixed-seed generator that builds per-event abnormal-
  return paths with a *plantable* post-list drift edge (``drift_bps``). It is the positive
  control: with the edge set to zero the inference must NOT manufacture significance out of
  ~two dozen events; with a large planted edge it must light up.

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
# Publication dates of the annual Forbes World's Billionaires list.
# The list drops in early spring (historically early March; April since 2020).
# Sources: Forbes press releases / contemporaneous coverage of each year's list.
# The event-study code snaps to the nearest trading day, so a calendar date near
# the true publication is what matters.
# --------------------------------------------------------------------------- #
LIST_DATES = {
    2013: "2013-03-04", 2014: "2014-03-03", 2015: "2015-03-02", 2016: "2016-03-01",
    2017: "2017-03-20", 2018: "2018-03-06", 2019: "2019-03-05", 2020: "2020-04-07",
    2021: "2021-04-06", 2022: "2022-04-05", 2023: "2023-04-04", 2024: "2024-04-02",
    2025: "2025-04-01",
}

# --------------------------------------------------------------------------- #
# Hardcoded table of newly-minted-billionaire vehicles.
# Columns: ticker, founder, list_year (first annual Forbes list on which the
# founder appears AS a billionaire with a tradable public vehicle).
# Sources: Forbes World's Billionaires list + company IPO/listing records &
# contemporaneous financial-press coverage. The believers' trade is "buy the
# vehicle behind the fresh new name." The "newly-minted" label is Forbes' own
# framing and, at the margin, a judgement call (paper-billionaire pre-IPO vs.
# first tradable appearance) — named on the Signal axis. Large, liquid names so
# there is clean price history for the market-model estimation window.
# --------------------------------------------------------------------------- #
_RAW_EVENTS = [
    # (ticker, founder, list_year)
    ("TEAM", "Mike Cannon-Brookes (Atlassian)", 2017),   # Atlassian IPO Dec 2015
    ("SNAP", "Evan Spiegel (Snap)",             2018),   # IPO Mar 2017
    ("SPOT", "Daniel Ek (Spotify)",             2019),   # direct listing Apr 2018
    ("SHOP", "Tobias Lutke (Shopify)",          2019),   # IPO May 2015
    ("ROKU", "Anthony Wood (Roku)",             2019),   # IPO Sep 2017
    ("WDAY", "Aneel Bhusri (Workday)",          2019),   # IPO Oct 2012
    ("ZM",   "Eric Yuan (Zoom)",                2020),   # IPO Apr 2019
    ("ZS",   "Jay Chaudhry (Zscaler)",          2020),   # IPO Mar 2018
    ("PINS", "Ben Silbermann (Pinterest)",      2021),   # IPO Apr 2019
    ("NET",  "Matthew Prince (Cloudflare)",     2021),   # IPO Sep 2019
    ("TWLO", "Jeff Lawson (Twilio)",            2021),   # IPO Jun 2016
    ("ABNB", "Brian Chesky (Airbnb)",           2021),   # IPO Dec 2020
    ("LAZR", "Austin Russell (Luminar)",        2021),   # SPAC Dec 2020
    ("NKLA", "Trevor Milton (Nikola)",          2021),   # SPAC Jun 2020 (brief; later fraud)
    ("HOOD", "Vlad Tenev (Robinhood)",          2022),   # IPO Jul 2021
    ("BMBL", "Whitney Wolfe Herd (Bumble)",     2022),   # IPO Feb 2021
    ("RIVN", "RJ Scaringe (Rivian)",            2022),   # IPO Nov 2021
    ("COIN", "Brian Armstrong (Coinbase)",      2022),   # direct listing Apr 2021
    ("DASH", "Tony Xu (DoorDash)",              2022),   # IPO Dec 2020
    ("DDOG", "Olivier Pomel (Datadog)",         2022),   # IPO Sep 2019
    ("CRWD", "George Kurtz (CrowdStrike)",      2022),   # IPO Jun 2019
    ("OKTA", "Todd McKinnon (Okta)",            2022),   # IPO Apr 2017
    ("DKNG", "Jason Robins (DraftKings)",       2022),   # SPAC Apr 2020
    ("U",    "David Helgason (Unity)",          2022),   # IPO Sep 2020
    ("PLTR", "Alex Karp (Palantir)",            2022),   # direct listing Sep 2020
    ("RBLX", "David Baszucki (Roblox)",         2022),   # direct listing Mar 2021
    ("HUBS", "Brian Halligan (HubSpot)",        2022),   # IPO Oct 2014
]

# De-duplicate by (ticker, list_year) and build the canonical table.
_seen: set = set()
FORBES_EVENTS: list[dict] = []
for _t, _f, _y in _RAW_EVENTS:
    _key = (_t, _y)
    if _key in _seen:
        continue
    _seen.add(_key)
    FORBES_EVENTS.append(
        {"ticker": _t, "founder": _f, "list_year": int(_y),
         "list_date": pd.Timestamp(LIST_DATES[_y])}
    )
FORBES_EVENTS.sort(key=lambda r: (r["list_date"], r["ticker"]))

TICKERS = sorted({r["ticker"] for r in FORBES_EVENTS})

# Benchmarks: SPY = broad market (primary), QQQ = tech/growth (the alpha-vs-beta check —
# how much of the "abnormal" drift is just the growth factor SPY doesn't hedge).
BENCHMARKS = ["SPY", "QQQ"]


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_718_{safe}_1d.parquet")


def fetch_prices(start: str = "2012-01-01", end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download daily adjusted closes for every event ticker + SPY and cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. One parquet per ticker (column ``close``, index ``date``).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in TICKERS + BENCHMARKS:
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
    for ticker in TICKERS + BENCHMARKS:
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
    events = [e for e in FORBES_EVENTS if e["ticker"] in prices.columns]
    return prices, events


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 26, drift_bps: float = 0.0, seed: int = 718,
                     est_days: int = 60, post_days: int = 63,
                     sig_daily: float = 0.030, beta: float = 1.4) -> dict:
    """Deterministic per-event abnormal-return panel with a plantable post-list drift edge.

    For each synthetic event we draw an estimation window of market + idiosyncratic
    returns and a post-list window. The stock return is ``alpha + beta*mkt + eps``; on
    the post-list window a **planted abnormal drift** of ``drift_bps`` basis points *per
    event* is spread across the window (the believers' "the glow keeps paying" effect we
    want the engine to recover). With ``drift_bps = 0`` there is no planted effect and the
    inference must NOT find significance out of ~two dozen events.

    High ``beta`` and ``sig_daily`` mimic the reality that these are freshly-IPO'd,
    high-volatility growth names — which is precisely why a real post-list edge would need
    to be enormous to clear the noise.

    Returns a dict with:
      ``post_car``  — post-list-window CAR per event (market-model abnormal)
      ``post_win``  — same, sign only (for the win-rate)
      ``base_car``  — abnormal CAR on random non-event windows (base rate)
      ``truth``     — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    per_day = (drift_bps * 1e-4) / max(post_days, 1)

    def one_event() -> tuple[float, float]:
        n = est_days + post_days + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        est = slice(0, est_days)
        b, a = np.polyfit(mkt[est], stock[est], 1)
        ev = slice(est_days, est_days + post_days)
        abn = stock[ev] - (a + b * mkt[ev])
        if per_day != 0.0:
            abn = abn + per_day        # plant the drift across the post window
        car = float(abn.sum())
        return car, float(np.sign(car) > 0)

    events = [one_event() for _ in range(n_events)]
    base = []
    for _ in range(2000):
        n = est_days + post_days + 5
        mkt = rng.normal(0.0003, 0.010, n)
        eps = rng.normal(0.0, sig_daily, n)
        stock = beta * mkt + eps
        b, a = np.polyfit(mkt[:est_days], stock[:est_days], 1)
        ev = slice(est_days, est_days + post_days)
        abn = stock[ev] - (a + b * mkt[ev])
        base.append(float(abn.sum()))

    return {
        "post_car": np.array([c for c, _ in events]),
        "post_win": np.array([w for _, w in events]),
        "base_car": np.array(base),
        "truth": {"n_events": n_events, "drift_bps": drift_bps, "seed": seed,
                  "post_days": post_days},
    }


def fingerprint(events: list[dict]) -> str:
    """Short content fingerprint of the event table (list dates), for as-of stamps."""
    arr = np.array([pd.Timestamp(e["list_date"]).value for e in events], dtype=np.int64)
    tick = "".join(e["ticker"] for e in events).encode()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes() + tick)
    return h.hexdigest()[:12]
