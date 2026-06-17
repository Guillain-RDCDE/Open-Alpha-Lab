"""Data layer for Study 299 (Keynote-Drift).

Three components; the first two are fully offline:

- ``KEYNOTE_EVENTS`` -- a hardcoded table of major Apple keynotes (WWDC, the
  September iPhone event, and the Spring/March "Peek Performance"-style events)
  from 2008 to 2025. Each row records the event date, the event type, and a
  short label. The "buy the rumor, sell the news" folklore says AAPL drifts UP
  into a keynote (anticipation) and sells off afterwards. We test that as a
  classic event study on AAPL daily returns. Dates are well-known and
  deterministic -- hardcoded here to keep the core fully offline and
  reproducible. Source: Apple Newsroom / press archives / Wikipedia
  "Apple Worldwide Developers Conference" and "Apple special event" pages.

- ``synthetic_prices`` -- a deterministic, offline daily-price generator with an
  optional planted "keynote drift" knob. ``drift_bps = 0`` is the null (no
  effect), so tests can confirm the machinery is truthful before looking at real
  data. The planted effect adds a small positive drift in the pre-event window
  and a small negative drift in the post-event window around each event date.

- ``fetch_prices`` -- real AAPL daily adjusted closes via yfinance, CACHE-ONLY by
  default (``fetch=False``); the network is touched only on an explicit
  ``fetch=True`` (lazy yfinance import). The cache lives at
  ``_cache/aapl_daily.parquet`` (a date-indexed frame with an ``adj_close``
  column).

No look-ahead: the event study aligns returns to trading days *relative to* each
event date. The tradable "buy the rumor" strategy enters ``pre_window`` trading
days before the event at that day's close and exits at the event-eve close --
returns are price-only (no dividends), gross and net are both reported, and one
trading-day execution lag is applied to the entry. Survivorship is not a concern
for a single ticker, but the keynote calendar is curated ex-post and only covers
the modern (post-2008) iPhone era -- that selection is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")
PRICE_CACHE = os.path.join(DEFAULT_CACHE, "aapl_daily.parquet")

# ---------------------------------------------------------------------------
# The hardcoded Apple keynote table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   date       : the keynote date (ISO string 'YYYY-MM-DD')
#   event_type : 'WWDC' (June developer conf), 'September' (the iPhone event),
#                or 'Spring' (March/Spring special event)
#   label      : a short human label
#
# The "buy the rumor, sell the news" claim: AAPL drifts up into the keynote on
# anticipation, then sells off once the products are revealed. The September
# iPhone event is the canonical case retail traders cite.
#
# Sources: Apple Newsroom press archive; Wikipedia "Apple Worldwide Developers
# Conference" and "Apple special event" pages. As-of 2026-06-17. Dates are the
# first/main day of each event.

KEYNOTE_EVENTS: list[dict] = [
    # 2008
    {"date": "2008-06-09", "event_type": "WWDC",      "label": "WWDC 2008 (iPhone 3G)"},
    {"date": "2008-09-09", "event_type": "September", "label": "Let's Rock (iPod)"},
    # 2009
    {"date": "2009-06-08", "event_type": "WWDC",      "label": "WWDC 2009 (iPhone 3GS)"},
    {"date": "2009-09-09", "event_type": "September", "label": "It's Only Rock and Roll"},
    # 2010
    {"date": "2010-06-07", "event_type": "WWDC",      "label": "WWDC 2010 (iPhone 4)"},
    {"date": "2010-09-01", "event_type": "September", "label": "Fall 2010 music event"},
    # 2011
    {"date": "2011-06-06", "event_type": "WWDC",      "label": "WWDC 2011 (iCloud)"},
    {"date": "2011-10-04", "event_type": "September", "label": "Let's Talk iPhone (4S)"},
    # 2012
    {"date": "2012-06-11", "event_type": "WWDC",      "label": "WWDC 2012"},
    {"date": "2012-09-12", "event_type": "September", "label": "iPhone 5"},
    # 2013
    {"date": "2013-06-10", "event_type": "WWDC",      "label": "WWDC 2013 (iOS 7)"},
    {"date": "2013-09-10", "event_type": "September", "label": "iPhone 5s / 5c"},
    # 2014
    {"date": "2014-06-02", "event_type": "WWDC",      "label": "WWDC 2014 (Swift)"},
    {"date": "2014-09-09", "event_type": "September", "label": "iPhone 6 / Apple Watch"},
    # 2015
    {"date": "2015-03-09", "event_type": "Spring",    "label": "Spring Forward (Watch)"},
    {"date": "2015-06-08", "event_type": "WWDC",      "label": "WWDC 2015"},
    {"date": "2015-09-09", "event_type": "September", "label": "iPhone 6s"},
    # 2016
    {"date": "2016-03-21", "event_type": "Spring",    "label": "Let Us Loop You In (SE)"},
    {"date": "2016-06-13", "event_type": "WWDC",      "label": "WWDC 2016"},
    {"date": "2016-09-07", "event_type": "September", "label": "iPhone 7"},
    # 2017
    {"date": "2017-06-05", "event_type": "WWDC",      "label": "WWDC 2017 (HomePod)"},
    {"date": "2017-09-12", "event_type": "September", "label": "iPhone X / 8"},
    # 2018
    {"date": "2018-03-27", "event_type": "Spring",    "label": "Education event (Chicago)"},
    {"date": "2018-06-04", "event_type": "WWDC",      "label": "WWDC 2018"},
    {"date": "2018-09-12", "event_type": "September", "label": "iPhone XS / XR"},
    # 2019
    {"date": "2019-03-25", "event_type": "Spring",    "label": "Apple TV+ / News+ event"},
    {"date": "2019-06-03", "event_type": "WWDC",      "label": "WWDC 2019 (Mac Pro)"},
    {"date": "2019-09-10", "event_type": "September", "label": "iPhone 11"},
    # 2020
    {"date": "2020-06-22", "event_type": "WWDC",      "label": "WWDC 2020 (Apple Silicon)"},
    {"date": "2020-09-15", "event_type": "September", "label": "Time Flies (Watch / iPad)"},
    {"date": "2020-10-13", "event_type": "September", "label": "Hi, Speed (iPhone 12)"},
    {"date": "2020-11-10", "event_type": "Spring",    "label": "One More Thing (M1)"},
    # 2021
    {"date": "2021-04-20", "event_type": "Spring",    "label": "Spring Loaded (AirTag)"},
    {"date": "2021-06-07", "event_type": "WWDC",      "label": "WWDC 2021"},
    {"date": "2021-09-14", "event_type": "September", "label": "California Streaming (13)"},
    {"date": "2021-10-18", "event_type": "Spring",    "label": "Unleashed (M1 Pro/Max)"},
    # 2022
    {"date": "2022-03-08", "event_type": "Spring",    "label": "Peek Performance (SE 3)"},
    {"date": "2022-06-06", "event_type": "WWDC",      "label": "WWDC 2022"},
    {"date": "2022-09-07", "event_type": "September", "label": "Far Out (iPhone 14)"},
    {"date": "2022-10-18", "event_type": "Spring",    "label": "iPad / M2 release"},
    # 2023
    {"date": "2023-06-05", "event_type": "WWDC",      "label": "WWDC 2023 (Vision Pro)"},
    {"date": "2023-09-12", "event_type": "September", "label": "Wonderlust (iPhone 15)"},
    {"date": "2023-10-30", "event_type": "Spring",    "label": "Scary Fast (M3)"},
    # 2024
    {"date": "2024-05-07", "event_type": "Spring",    "label": "Let Loose (iPad Pro M4)"},
    {"date": "2024-06-10", "event_type": "WWDC",      "label": "WWDC 2024 (Apple Intelligence)"},
    {"date": "2024-09-09", "event_type": "September", "label": "It's Glowtime (iPhone 16)"},
    # 2025
    {"date": "2025-06-09", "event_type": "WWDC",      "label": "WWDC 2025"},
    {"date": "2025-09-09", "event_type": "September", "label": "Awe Dropping (iPhone 17)"},
]

KEYNOTE_DF: pd.DataFrame = pd.DataFrame(KEYNOTE_EVENTS)


def keynote_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Apple keynote table.

    Columns: ``date`` (datetime64), ``event_type`` ('WWDC'/'September'/'Spring'),
    ``label`` (str). Sorted by date.
    """
    df = KEYNOTE_DF.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic daily-price generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_prices(
    drift_bps: float = 0.0,
    pre_window: int = 5,
    post_window: int = 5,
    base_ann_ret: float = 0.20,
    vol_ann: float = 0.30,
    start: str = "2008-01-01",
    end: str = "2025-12-31",
    seed: int = 299,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible AAPL-like daily price series with an optional planted keynote drift.

    Daily log-returns are drawn i.i.d. from N(mu_d, sigma_d^2) calibrated from the
    annualized inputs. For each keynote in ``KEYNOTE_EVENTS``, an extra drift of
    ``+drift_bps`` bps/day is added on each of the ``pre_window`` trading days
    *before* the event (the "buy the rumor" leg) and ``-drift_bps`` bps/day on each
    of the ``post_window`` trading days *after* the event (the "sell the news" leg).

    ``drift_bps = 0`` is the null -- the keynote calendar carries no information.

    Returns ``(price_df, truth)`` where ``price_df`` is a business-day-indexed frame
    with an ``adj_close`` column, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)

    mu_d = base_ann_ret / 252.0
    sigma_d = vol_ann / np.sqrt(252.0)
    log_ret = rng.normal(mu_d, sigma_d, n)

    # Plant the keynote drift on the trading days around each event.
    pos = pd.Series(np.arange(n), index=idx)
    for ev in KEYNOTE_EVENTS:
        ev_date = pd.Timestamp(ev["date"])
        # Nearest trading day at or after the event date.
        future = pos.index[pos.index >= ev_date]
        if len(future) == 0:
            continue
        anchor = pos[future[0]]
        for k in range(1, pre_window + 1):
            j = anchor - k
            if 0 <= j < n:
                log_ret[j] += drift_bps * 1e-4
        for k in range(0, post_window):
            j = anchor + k
            if 0 <= j < n:
                log_ret[j] -= drift_bps * 1e-4

    price = 100.0 * np.exp(np.cumsum(log_ret))
    price_df = pd.DataFrame({"adj_close": price}, index=idx)
    price_df.index.name = "date"

    truth = {
        "drift_bps": drift_bps,
        "pre_window": pre_window,
        "post_window": post_window,
        "base_ann_ret": base_ann_ret,
        "vol_ann": vol_ann,
        "seed": seed,
        "n_days": n,
    }
    return price_df, truth


# ---------------------------------------------------------------------------
# Real tape -- AAPL daily adjusted closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_prices(
    fetch: bool = False,
    cache_path: str = PRICE_CACHE,
    start: str = "2007-06-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Daily AAPL adjusted-close frame; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/aapl_daily.parquet``). The returned frame is
    date-indexed with a single ``adj_close`` column (price-only / split- and
    dividend-adjusted via yfinance ``auto_adjust=True``).

    Raises ``FileNotFoundError`` when the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached AAPL price frame at {cache_path}. "
                "Call fetch_prices(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    # ------------------------------------------------------------------
    # Live fetch: yfinance daily closes for AAPL
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "AAPL",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data for AAPL")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = raw["Close"]

    out = pd.DataFrame({"adj_close": close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index().dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached AAPL daily: {len(out)} rows -> {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the adj_close column, for the as-of stamp."""
    vals = df["adj_close"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
