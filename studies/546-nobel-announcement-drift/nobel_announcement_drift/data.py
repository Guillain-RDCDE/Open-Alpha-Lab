"""Data layer for Study 546 — Nobel-Announcement-Drift.

The claim is news-attention folklore: when the Nobel Prizes are announced each October, the
sectors *thematically* tied to the science prizes should catch a bid and **drift** in the days
that follow — pharma/biotech after **Medicine**, semiconductors/tech after **Physics** and
**Chemistry**. We test it as a clean event study: sector-ETF **cumulative abnormal returns**
(CARs, market-model residuals vs SPY) in windows around each announcement date, against a
placebo of random October dates and a deterministic synthetic positive control.

Two tapes, one schema:

* **Real tape.** Daily adjusted closes for the sector ETFs (XLV health/pharma, XLK tech, SMH
  semiconductors, IBB biotech) plus SPY (the market-model benchmark), yfinance, no key,
  cached under ``_cache/nad_prices.parquet``. The **Nobel announcement calendar** is hardcoded
  below (``NOBEL_DATES``): the actual public announcement dates of the three *science* prizes
  (Physiology/Medicine, Physics, Chemistry), 2001 → 2024 — all inside our as-of window with a
  full post-event drift window available.

* **Synthetic.** A deterministic, fixed-seed daily world for each sector with a *planted*
  post-event drift starting on each (synthetic) announcement day. The positive control: with
  ``drift = 0`` the event-vs-placebo test must NOT manufacture significance; with a planted
  drift it must light up. Faithful-engine / power check only — never cites a real-tape stamp.

Survivorship note (Signal axis): sector ETFs are indices, not a hand-picked survivor basket, so
single-name survivorship bias is minimal; the honest limitations here are the **small event
count** (3 prizes x ~24 years) and the **thematic-mapping assumption** (that a Medicine prize
"should" move XLV at all). Both are stated openly on the Signal axis.

No look-ahead: the announcement date is public the moment it is read out (~11:30 CET), and it is
scheduled weeks ahead, so tagging the first trading session on/after the announcement and
measuring the *forward* drift window uses no future information. The one documented execution
lag: we enter at the **close of the first trading session on/after the announcement** (t0) and
measure the drift over (t0, t0+H]. A one-bar-later entry is reported in the robustness sweep.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Sector ETFs and the science prize each is "supposed" to react to.
#   XLV  — Health Care Select Sector SPDR (pharma-heavy)      <- Medicine
#   IBB  — iShares Biotechnology                              <- Medicine
#   XLK  — Technology Select Sector SPDR                      <- Physics / Chemistry
#   SMH  — VanEck Semiconductors                              <- Physics / Chemistry
SECTOR_ETFS = ["XLV", "IBB", "XLK", "SMH"]
BENCHMARK = "SPY"

# Which sector "should" drift after which prize (the folklore mapping we test).
PRIZE_SECTOR = {
    "medicine": ["XLV", "IBB"],
    "physics": ["XLK", "SMH"],
    "chemistry": ["XLK", "SMH"],
}

TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Hardcoded Nobel science-prize announcement calendar, 2001 -> 2024.
# Source: nobelprize.org press-release dates. The three science prizes are announced on the
# Monday (Physiology/Medicine), Tuesday (Physics) and Wednesday (Chemistry) of the first full
# week of October each year. (Peace/Literature/Economics are excluded — no clean sector map.)
# Dates below are the actual announcement dates.
# --------------------------------------------------------------------------- #
NOBEL_DATES: dict[str, dict[str, str]] = {
    "2001": {"medicine": "2001-10-08", "physics": "2001-10-09", "chemistry": "2001-10-10"},
    "2002": {"medicine": "2002-10-07", "physics": "2002-10-08", "chemistry": "2002-10-09"},
    "2003": {"medicine": "2003-10-06", "physics": "2003-10-07", "chemistry": "2003-10-08"},
    "2004": {"medicine": "2004-10-04", "physics": "2004-10-05", "chemistry": "2004-10-06"},
    "2005": {"medicine": "2005-10-03", "physics": "2005-10-04", "chemistry": "2005-10-05"},
    "2006": {"medicine": "2006-10-02", "physics": "2006-10-03", "chemistry": "2006-10-04"},
    "2007": {"medicine": "2007-10-08", "physics": "2007-10-09", "chemistry": "2007-10-10"},
    "2008": {"medicine": "2008-10-06", "physics": "2008-10-07", "chemistry": "2008-10-08"},
    "2009": {"medicine": "2009-10-05", "physics": "2009-10-06", "chemistry": "2009-10-07"},
    "2010": {"medicine": "2010-10-04", "physics": "2010-10-05", "chemistry": "2010-10-06"},
    "2011": {"medicine": "2011-10-03", "physics": "2011-10-04", "chemistry": "2011-10-05"},
    "2012": {"medicine": "2012-10-08", "physics": "2012-10-09", "chemistry": "2012-10-10"},
    "2013": {"medicine": "2013-10-07", "physics": "2013-10-08", "chemistry": "2013-10-09"},
    "2014": {"medicine": "2014-10-06", "physics": "2014-10-07", "chemistry": "2014-10-08"},
    "2015": {"medicine": "2015-10-05", "physics": "2015-10-06", "chemistry": "2015-10-07"},
    "2016": {"medicine": "2016-10-03", "physics": "2016-10-04", "chemistry": "2016-10-05"},
    "2017": {"medicine": "2017-10-02", "physics": "2017-10-03", "chemistry": "2017-10-04"},
    "2018": {"medicine": "2018-10-01", "physics": "2018-10-02", "chemistry": "2018-10-03"},
    "2019": {"medicine": "2019-10-07", "physics": "2019-10-08", "chemistry": "2019-10-09"},
    "2020": {"medicine": "2020-10-05", "physics": "2020-10-06", "chemistry": "2020-10-07"},
    "2021": {"medicine": "2021-10-04", "physics": "2021-10-05", "chemistry": "2021-10-06"},
    "2022": {"medicine": "2022-10-03", "physics": "2022-10-04", "chemistry": "2022-10-05"},
    "2023": {"medicine": "2023-10-02", "physics": "2023-10-03", "chemistry": "2023-10-04"},
    "2024": {"medicine": "2024-10-07", "physics": "2024-10-08", "chemistry": "2024-10-09"},
}


def nobel_events(prizes: tuple[str, ...] = ("medicine", "physics", "chemistry")) -> pd.DataFrame:
    """The Nobel science-prize announcement calendar as a long frame.

    Columns: ``date`` (Timestamp), ``prize`` (medicine/physics/chemistry), ``year``. One row per
    (year, prize). Sorted by date.
    """
    rows = []
    for yr, d in NOBEL_DATES.items():
        for prize in prizes:
            if prize in d:
                rows.append({"date": pd.Timestamp(d[prize]), "prize": prize, "year": int(yr)})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Real tape — cache-first
# --------------------------------------------------------------------------- #
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "nad_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.5):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001  (network flakiness)
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2000-01-01",
    end: str = "2025-06-30",
) -> pd.DataFrame:
    """Daily adjusted-close prices for the sector ETFs + SPY, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    tickers = list(SECTOR_ETFS) + [BENCHMARK]
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_price_cache())


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic world."""

    drift: float  # per-day extra sector return over the post-event window (0 = null)

    @property
    def has_drift(self) -> bool:
        return self.drift != 0.0


def synthetic_world(
    n_years: int = 24,
    drift: float = 0.0,
    horizon: int = 10,
    seed: int = 546,
    sig_daily: float = 0.011,
    mkt_daily: float = 0.009,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A deterministic sector + market daily world with a *planted* post-event drift.

    Builds a business-day price panel with two columns: ``SECTOR`` and ``SPY`` (the benchmark).
    The sector return = beta * market return + idio; on each (synthetic) October announcement
    day the *next* ``horizon`` sessions of the sector get an extra ``drift`` per day. With
    ``drift = 0`` the post-event window is statistically identical to any other window; with a
    planted drift the event-CAR test must light up.

    Returns ``(prices, events, truth)`` — ``events`` is a one-column frame of announcement dates,
    matching the schema ``nobel_events`` produces (date + a single 'sector' prize).
    """
    rng = np.random.default_rng(seed)
    # one "October" per synthetic year, ~252 sessions apart, offset ~190 into each year block
    idx = pd.bdate_range("2001-01-02", periods=n_years * TRADING_DAYS)
    n = len(idx)
    mkt = rng.normal(0.0003, mkt_daily, size=n)
    beta = 1.05
    idio = rng.normal(0.0, sig_daily, size=n)
    sector = beta * mkt + idio

    event_pos = []
    for y in range(n_years):
        pos = y * TRADING_DAYS + 190  # a fixed "early October" offset in each year block
        if pos + horizon < n:
            event_pos.append(pos)
            sector[pos + 1 : pos + 1 + horizon] += drift  # drift over (t0, t0+H]

    px = pd.DataFrame(
        {
            "SECTOR": 100.0 * np.cumprod(1.0 + sector),
            "SPY": 100.0 * np.cumprod(1.0 + mkt),
        },
        index=idx,
    )
    events = pd.DataFrame(
        {"date": [idx[p] for p in event_pos], "prize": "sector", "year": range(len(event_pos))}
    )
    return px, events, WorldTruth(drift=drift)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
