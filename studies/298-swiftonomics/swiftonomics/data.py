"""Data layer for Study 298 (Swiftonomics).

"Swiftonomics" is the popular claim that Taylor Swift's *Eras Tour* (2023-2024)
was an economic force large enough to move things: local hotel revenue, GDP
prints, and -- the claim we can actually test on a clean tape -- the share price
of **Live Nation Entertainment (LYV)**, the concert promoter and Ticketmaster
parent that sold and operated the tour. The folklore version: "when Taylor
announces a tour leg, buy LYV."

This is an **event study**. Three components, all offline for the core:

- ``ERAS_TOUR_EVENTS`` -- a hardcoded table of the canonical Eras Tour news
  events (tour announcement, the Ticketmaster pre-sale meltdown, the box-office
  records, international-leg announcements, the concert film, the final show).
  Each event has a calendar date and a short label. Sources: Live Nation press
  releases, Billboard, Pollstar, mainstream financial press. Hardcoded to keep
  the core fully offline and reproducible.

- ``synthetic_panel`` -- a deterministic, offline daily-return generator for a
  *stock* and a *market* index with an optional planted "event bump" knob. A
  ``bump_bps = 0`` setting is the null (events carry no abnormal return), so the
  test-suite can confirm the machinery is truthful before looking at real data.

- ``fetch_prices`` -- real daily adjusted closes for LYV and ^GSPC via yfinance,
  cache-only by default (``fetch=False``), so the test-suite and reproducible
  core never touch the network. The cache lives under ``_cache/lyv_gspc.parquet``.

No look-ahead: abnormal returns are measured with a market model whose betas are
estimated on a clean **estimation window** that ends *before* each event window.
An event-day position is entered at the *next* day's open (one trading-day
execution lag), so we never trade on information we could not yet have.

**Total-return caveat:** yfinance ``auto_adjust=True`` closes are dividend- and
split-adjusted (total return). LYV pays no dividend over the sample, so price and
total return coincide for the stock; ^GSPC is price-only (no dividend). This is
labeled on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(DEFAULT_CACHE, "lyv_gspc.parquet")

STOCK = "LYV"
MARKET = "^GSPC"

# ---------------------------------------------------------------------------
# The hardcoded Eras Tour event table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   date  : event date (the day the news hit; weekend news is mapped to the next
#           trading day inside the strategy layer)
#   label : short human-readable description
#   kind  : 'announce' (tour/leg announcements + ticketing), 'milestone'
#           (records, film, finale) -- lets us slice the event set
#
# Sources: Live Nation Entertainment press releases & 10-K/10-Q filings; Billboard;
# Pollstar; Variety; mainstream financial press (Reuters, Bloomberg, WSJ).
# As-of 2026-06-17. The Eras Tour ran 2023-03-17 (Glendale) to 2024-12-08 (Vancouver).
ERAS_TOUR_EVENTS: list[dict] = [
    {"date": "2022-11-01", "label": "Eras Tour (US leg) announced",          "kind": "announce"},
    {"date": "2022-11-15", "label": "Ticketmaster pre-sale meltdown",         "kind": "announce"},
    {"date": "2022-11-18", "label": "Ticketmaster cancels general sale",      "kind": "announce"},
    {"date": "2023-03-17", "label": "Eras Tour opening night (Glendale)",     "kind": "milestone"},
    {"date": "2023-05-01", "label": "International (LatAm/Europe) leg added",  "kind": "announce"},
    {"date": "2023-06-20", "label": "Asia / Australia leg announced",          "kind": "announce"},
    {"date": "2023-07-13", "label": "Highest-grossing tour record reported",  "kind": "milestone"},
    {"date": "2023-08-03", "label": "2024 US/Canada dates announced",          "kind": "announce"},
    {"date": "2023-08-31", "label": "Eras Tour concert film announced",       "kind": "milestone"},
    {"date": "2023-10-13", "label": "Concert film box-office opening",         "kind": "milestone"},
    {"date": "2023-12-06", "label": "$1bn+ gross / Person of the Year",        "kind": "milestone"},
    {"date": "2024-03-07", "label": "Eras Tour 2024 European leg opens (Asia)","kind": "milestone"},
    {"date": "2024-05-09", "label": "European leg opens (Paris)",             "kind": "milestone"},
    {"date": "2024-08-07", "label": "Vienna shows cancelled (security)",       "kind": "milestone"},
    {"date": "2024-10-18", "label": "US final leg begins (Miami)",            "kind": "milestone"},
    {"date": "2024-12-08", "label": "Eras Tour final show (Vancouver)",        "kind": "milestone"},
]

ERAS_TOUR_DF: pd.DataFrame = pd.DataFrame(ERAS_TOUR_EVENTS)
ERAS_TOUR_DF["date"] = pd.to_datetime(ERAS_TOUR_DF["date"])


def eras_events() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Eras Tour event table.

    Columns: ``date`` (datetime64), ``label`` (str), ``kind`` ('announce' or
    'milestone'). Dates are the day the news broke; weekend dates are mapped to
    the next trading day inside the strategy layer.
    """
    return ERAS_TOUR_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 750,
    beta: float = 1.30,
    bump_bps: float = 0.0,
    mkt_mu_bps: float = 3.0,
    mkt_vol_bps: float = 90.0,
    idio_vol_bps: float = 200.0,
    n_events: int = 16,
    seed: int = 298,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible (stock, market) daily-price pair with an optional event bump.

    The market index follows a daily random walk with drift ``mkt_mu_bps`` and
    vol ``mkt_vol_bps``. The stock follows a one-factor market model::

        r_stock_t = beta * r_mkt_t + idio_t (+ bump on event days)

    where ``idio_t ~ N(0, idio_vol_bps)``. On ``n_events`` randomly chosen event
    days an *abnormal* return of ``bump_bps`` is added to the stock only.
    ``bump_bps = 0`` is the null -- events carry no abnormal return and the
    event study should read ~zero.

    Returns ``(prices, events, truth)`` where:
      - ``prices`` is a daily (date x {LYV, ^GSPC}) frame of cumulative prices,
      - ``events`` is a DataFrame with columns ``date`` (the planted event dates)
        and ``label``,
      - ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-06-01", periods=n_days)

    r_mkt = rng.normal(mkt_mu_bps * 1e-4, mkt_vol_bps * 1e-4, n_days)
    idio = rng.normal(0.0, idio_vol_bps * 1e-4, n_days)
    r_stock = beta * r_mkt + idio

    # Plant abnormal returns on n_events dates, drawn from the interior so each
    # event has a clean estimation window before it.
    lo, hi = 80, n_days - 5
    event_pos = np.sort(rng.choice(np.arange(lo, hi), size=n_events, replace=False))
    r_stock[event_pos] += bump_bps * 1e-4

    px_mkt = 4000.0 * np.exp(np.cumsum(np.log1p(r_mkt)))
    px_stock = 90.0 * np.exp(np.cumsum(np.log1p(r_stock)))

    prices = pd.DataFrame({STOCK: px_stock, MARKET: px_mkt}, index=dates)
    prices.index.name = "date"

    events = pd.DataFrame({
        "date": dates[event_pos],
        "label": [f"synthetic event {i + 1}" for i in range(n_events)],
        "kind": ["announce"] * n_events,
    })

    truth = {
        "beta": beta,
        "bump_bps": bump_bps,
        "n_days": n_days,
        "n_events": n_events,
        "mkt_vol_bps": mkt_vol_bps,
        "idio_vol_bps": idio_vol_bps,
        "seed": seed,
    }
    return prices, events, truth


# ---------------------------------------------------------------------------
# Real tape -- yfinance daily closes for LYV and ^GSPC, cache-only by default
# ---------------------------------------------------------------------------
def fetch_prices(
    fetch: bool = False,
    cache_path: str = PRICES_CACHE,
    start: str = "2021-06-01",
    end: str = "2025-06-01",
) -> pd.DataFrame:
    """Daily adjusted closes for LYV and ^GSPC; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import),
    after which the result is cached as a parquet under
    ``_cache/lyv_gspc.parquet``. The frame is a (date x {LYV, ^GSPC}) panel of
    adjusted daily closes.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False`` --
    callers must guard with ``os.path.exists(cache_path)`` (or branch on a
    ``HAVE_REAL`` flag) before calling.

    **Total-return note:** ``auto_adjust=True`` gives dividend/split-adjusted
    closes. LYV pays no dividend over the sample (price == total return); ^GSPC
    is price-only.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached price panel at {cache_path}. "
                "Call fetch_prices(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        [STOCK, MARKET],
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes.index = pd.to_datetime(closes.index)
    closes = closes[[STOCK, MARKET]].sort_index().dropna(how="any")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached {closes.shape[0]} days x {closes.shape[1]} tickers -> {cache_path}")
    return closes


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (LYV column), for the as-of stamp."""
    col = STOCK if STOCK in df.columns else df.columns[0]
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
