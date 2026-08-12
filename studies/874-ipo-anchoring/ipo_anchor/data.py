"""Data layer for Study 874 — IPO-Price Anchoring.

Three ingredients, all offline-friendly once cached.

* **The curated event table (hard-coded public record).** ``IPOS`` below is a fixed list of
  ~45 well-known recent US listings, each with its **ticker**, its headline **offer price**
  (for a traditional underwritten IPO, the final 424B4 prospectus offer price) or its
  exchange **reference price** (for a direct listing — flagged ``kind="direct"``), and its
  **first-trade date**. These are matters of public record (SEC EDGAR 424B4 prospectuses and
  the listing exchange's reference-price notices; widely reported in the financial press at
  the time). They are encoded here directly, per desk convention for published event dates,
  with the source note in this docstring — NOT scraped live. The offer/reference price is the
  behavioural **anchor** under test.

* **The live tape (yfinance daily closes, cached).** Daily adjusted (total-return) closes for
  every ticker in ``IPOS`` plus ``SPY`` (the market leg for the abnormal-return adjustment).
  ``auto_adjust=True``. Cached wide under ``_cache/ipo_anchor_prices.csv``; everything
  downstream is cache-first and offline. Delisted names (e.g. a name later acquired) simply
  contribute a shorter price history — we deliberately KEEP them (the opposite of survivorship
  pruning); the curation bias that remains is named on the Signal axis.

* **The synthetic control.** A deterministic seeded name-month panel (``synthetic_panel``)
  built in the SAME shape the strategy consumes, carrying a TUNABLE planted **anchoring pull**
  ``edge`` (the forward abnormal return is ``-edge * gap_from_offer + noise``; ``edge=0`` is
  the null world where the gap predicts nothing). The machinery proof: the Fama-MacBeth slope
  detector must recover a planted pull and must NOT fire on the null. The month index is a
  ``PeriodIndex`` kept as periods (never ``.to_timestamp()``) — safely below the pandas
  ns-Timestamp horizon.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells; ``load_prices()`` reads the
cached csv directly (no yfinance import).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICE_CACHE = os.path.join(CACHE_DIR, "ipo_anchor_prices.csv")

# The stamped sample end: June 2026 is the last COMPLETE calendar month as of the run date.
AS_OF = "2026-06-30"
BENCH = "SPY"                     # market leg for the abnormal-return adjustment

# --------------------------------------------------------------------------- #
# The curated IPO anchor table (public record)
# --------------------------------------------------------------------------- #
# Source: final IPO offer prices from each issuer's SEC 424B4 prospectus (EDGAR), and
# direct-listing reference prices from the listing exchange's reference-price notice, both
# widely reported in the financial press on the listing date. Encoded here as public dated
# facts per desk convention. `offer` is the behavioural ANCHOR; `first_trade` is the first
# regular-way trading session. `kind`: "ipo" = underwritten offer price; "direct" = exchange
# reference price (no capital raised at that price — a softer anchor, tested separately).
# Fields: (ticker, offer_price_usd, first_trade_date, name, kind)
IPOS = [
    # --- traditional underwritten IPOs (offer price = the anchor) ------------------ #
    ("ABNB", 68.0,  "2020-12-10", "Airbnb",            "ipo"),
    ("DASH", 102.0, "2020-12-09", "DoorDash",          "ipo"),
    ("SNOW", 120.0, "2020-09-16", "Snowflake",         "ipo"),
    ("U",    52.0,  "2020-09-18", "Unity Software",    "ipo"),
    ("AI",   42.0,  "2020-12-09", "C3.ai",             "ipo"),
    ("FROG", 44.0,  "2020-09-16", "JFrog",             "ipo"),
    ("WISH", 24.0,  "2020-12-16", "ContextLogic",      "ipo"),
    ("RIVN", 78.0,  "2021-11-10", "Rivian",            "ipo"),
    ("HOOD", 38.0,  "2021-07-29", "Robinhood",         "ipo"),
    ("AFRM", 49.0,  "2021-01-13", "Affirm",            "ipo"),
    ("BMBL", 43.0,  "2021-02-11", "Bumble",            "ipo"),
    ("COUR", 33.0,  "2021-03-31", "Coursera",          "ipo"),
    ("ONON", 24.0,  "2021-09-15", "On Holding",        "ipo"),
    ("PATH", 56.0,  "2021-04-21", "UiPath",            "ipo"),
    ("GTLB", 77.0,  "2021-10-14", "GitLab",            "ipo"),
    ("CPNG", 35.0,  "2021-03-11", "Coupang",           "ipo"),
    ("NU",   9.0,   "2021-12-09", "Nu Holdings",       "ipo"),
    ("OLO",  25.0,  "2021-03-17", "Olo",               "ipo"),
    ("MNDY", 155.0, "2021-06-10", "monday.com",        "ipo"),
    ("CFLT", 36.0,  "2021-06-24", "Confluent",         "ipo"),
    ("DUOL", 102.0, "2021-07-28", "Duolingo",          "ipo"),
    ("S",    35.0,  "2021-06-30", "SentinelOne",       "ipo"),
    ("TOST", 40.0,  "2021-09-22", "Toast",             "ipo"),
    ("ARM",  51.0,  "2023-09-14", "Arm Holdings",      "ipo"),
    ("RDDT", 34.0,  "2024-03-21", "Reddit",            "ipo"),
    ("BIRK", 46.0,  "2023-10-11", "Birkenstock",       "ipo"),
    ("CART", 30.0,  "2023-09-19", "Instacart",         "ipo"),
    ("KVYO", 30.0,  "2023-09-20", "Klaviyo",           "ipo"),
    ("CAVA", 22.0,  "2023-06-15", "Cava Group",        "ipo"),
    ("DDOG", 27.0,  "2019-09-19", "Datadog",           "ipo"),
    ("CRWD", 34.0,  "2019-06-12", "CrowdStrike",       "ipo"),
    ("ZM",   36.0,  "2019-04-18", "Zoom Video",        "ipo"),
    ("PINS", 19.0,  "2019-04-18", "Pinterest",         "ipo"),
    ("UBER", 45.0,  "2019-05-10", "Uber",              "ipo"),
    ("LYFT", 72.0,  "2019-03-29", "Lyft",              "ipo"),
    ("NET",  15.0,  "2019-09-13", "Cloudflare",        "ipo"),
    ("ESTC", 36.0,  "2018-10-05", "Elastic",           "ipo"),
    ("BYND", 25.0,  "2019-05-02", "Beyond Meat",       "ipo"),
    ("CVNA", 15.0,  "2017-04-28", "Carvana",           "ipo"),
    ("W",    29.0,  "2014-10-02", "Wayfair",           "ipo"),
    # --- direct listings (exchange reference price = a softer anchor) --------------- #
    ("COIN", 250.0, "2021-04-14", "Coinbase",          "direct"),
    ("RBLX", 45.0,  "2021-03-10", "Roblox",            "direct"),
    ("PLTR", 7.25,  "2020-09-30", "Palantir",          "direct"),
    ("SPOT", 132.0, "2018-04-03", "Spotify",           "direct"),
]

__all__ = [
    "IPOS", "AS_OF", "BENCH", "CACHE_DIR", "PRICE_CACHE",
    "ipo_table", "tickers", "fetch", "have_real", "load_prices", "synthetic_panel",
]


def ipo_table(include_direct: bool = True) -> pd.DataFrame:
    """The curated anchor table as a DataFrame indexed by ticker.

    ``include_direct=False`` drops the four direct listings (whose anchor is a reference
    price, not an underwritten offer), leaving the traditional-IPO subset.
    """
    df = pd.DataFrame(IPOS, columns=["ticker", "offer", "first_trade", "name", "kind"])
    df["first_trade"] = pd.to_datetime(df["first_trade"])
    if not include_direct:
        df = df[df["kind"] == "ipo"]
    return df.set_index("ticker")


def tickers(include_bench: bool = True) -> list[str]:
    """All curated tickers (plus the benchmark by default)."""
    ts = [row[0] for row in IPOS]
    if include_bench and BENCH not in ts:
        ts = ts + [BENCH]
    return ts


# --------------------------------------------------------------------------- #
# The live tape (yfinance daily closes -> csv cache)
# --------------------------------------------------------------------------- #
def fetch(start: str = "2014-01-01", end: str | None = None,
          path: str = PRICE_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the daily-close tape once and cache it (network-only; never offline path).

    Adjusted (total-return) closes for every curated ticker plus ``SPY``. Retries up to
    ``retries`` times on transient Yahoo failures. Cached wide (index=date, columns=ticker).
    """
    import yfinance as yf

    syms = tickers(include_bench=True)
    raw = None
    for _ in range(retries):
        try:
            dl = yf.download(syms, start=start, end=end, auto_adjust=True,
                             progress=False)
            raw = dl["Close"] if isinstance(dl.columns, pd.MultiIndex) else dl
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the IPO-anchor tape")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICE_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICE_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide daily-close frame (columns=ticker incl. SPY), sliced to ``asof``.

    OFFLINE — reads the csv directly, no yfinance import.
    """
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Synthetic control — a world with a KNOWN planted anchoring pull
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 874,
    n_names: int = 45,
    n_months: int = 130,
    idio_sig: float = 0.11,
    gap0_sig: float = 0.30,
    start: str = "2015-01",
) -> dict:
    """Deterministic seeded name-month panel in the strategy's ``Panel`` shape.

    Each synthetic name lists at its offer (gap 0) at a random month and thereafter follows
    a monthly log-gap random walk. The planted mechanism is an **anchoring pull**: the
    forward monthly abnormal return is ``-edge * gap[t] + noise`` — with ``edge > 0`` a name
    stretched above its offer is pulled back down and one below is pulled back up (reversion
    toward the anchor). ``edge = 0`` is the null: the gap wanders but predicts nothing.

    Returns a dict ``{months, names, gap, fwd_abn, below}`` where ``gap``/``fwd_abn``/``below``
    are ``(n_months, n_names)`` float/bool arrays (``fwd_abn``'s last row and pre-listing
    cells are ``nan``), matching :func:`ipo_anchor.strategy.build_panel`'s output. The month
    index is a ``PeriodIndex`` kept as periods (never ``.to_timestamp()``).
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, periods=n_months, freq="M")
    N, T = n_names, n_months

    # each name lists somewhere in the first ~60% of the window
    list_month = rng.integers(0, max(1, int(T * 0.6)), size=N)

    gap = np.full((T, N), np.nan)
    fwd = np.full((T, N), np.nan)
    # per-name noise draws for every month
    eps = rng.normal(0.0, idio_sig, size=(T, N))
    for j in range(N):
        t0 = int(list_month[j])
        g = 0.0
        # small idiosyncratic first-month move off the anchor so gaps disperse
        for t in range(t0, T):
            gap[t, j] = g
            # abnormal return realised NEXT month = pull toward anchor + noise
            if t + 1 < T:
                r_next = -edge * g + eps[t + 1, j]
                fwd[t, j] = r_next
                g = g + r_next
    below = gap < 0.0
    return {"months": months, "names": [f"SYN{j:02d}" for j in range(N)],
            "gap": gap, "fwd_abn": fwd, "below": below}
