"""Data layer for Study 289 (Diwali-Muhurat).

The Muhurat trading session is a special ~1-hour evening session held on the day
of Diwali (Laxmi Pujan) on India's NSE and BSE. Folklore says buying during the
auspicious Muhurat window — or simply holding Indian equities across Diwali —
brings prosperity for the year ahead. This study tests whether the days around
the Muhurat session carry any abnormal return for a tradeable India proxy.

Three components, the core fully offline:

- ``DIWALI_DATES`` — a hardcoded table of the Diwali / Laxmi-Pujan date (the day
  the Muhurat session is held) for every year 2003–2025. Diwali follows the
  Hindu lunisolar calendar (Amanta month of Kartik, new-moon / Amavasya), so the
  Gregorian date drifts ~11 days earlier each year before leaping forward. These
  dates are deterministic, small, and well documented — hardcoded here to keep
  the core fully offline and reproducible. Sources: NSE/BSE Muhurat trading
  notices; drikpanchang.com Diwali calendar; Wikipedia "Diwali".

- ``synthetic_panel`` — a deterministic, offline daily-return generator with an
  optional planted "Diwali premium" knob. ``premium_bps = 0`` is the null (no
  effect), so tests can confirm the machinery is truthful before looking at real
  data.

- ``fetch_inda`` — real iShares MSCI India ETF (INDA) daily total-return prices
  via yfinance, **cache-only by default** (``fetch=False``); the network is only
  touched on an explicit ``fetch=True`` (lazy yfinance import). Cache lives under
  ``_cache/inda_daily.parquet``.

**Proxy caveat.** INDA is a USD-denominated US-listed ETF that tracks the MSCI
India index. The Muhurat session itself trades on Indian exchanges in INR;
INDA does *not* trade in that evening window. We therefore measure the return
INDA earns on the US trading day(s) *following* each Diwali — the closest
tradeable instrument a non-Indian investor could actually hold. This is an
honest proxy, not the literal Muhurat session, and it is named as such.

No look-ahead: the Diwali date is known years in advance (it is a calendar
event), but the *return* is measured strictly after it. The headline spec buys
INDA at the close of the first INDA trading day on-or-after Diwali and holds it
for ``hold_days`` sessions (a one-session execution lag from the event).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
INDA_CACHE = os.path.join(DEFAULT_CACHE, "inda_daily.parquet")

# ---------------------------------------------------------------------------
# Hardcoded Diwali / Muhurat-trading dates — the deterministic offline core
# ---------------------------------------------------------------------------
# Each row records the Gregorian date of Diwali (Laxmi Pujan), which is the day
# the NSE/BSE hold their special Muhurat trading session. Diwali is set by the
# Hindu lunisolar calendar (new moon of Kartik), so the date moves each year.
#
# Sources:
#   - NSE / BSE Muhurat trading circulars (annual).
#   - drikpanchang.com Diwali / Lakshmi Puja calendar.
#   - Wikipedia: "Diwali" (list of dates).
#
# As-of: 2026-06-17. The 2025 Diwali (Laxmi Pujan) fell on 2025-10-21; the
# Muhurat session was held that evening.
DIWALI_DATES: list[dict] = [
    # year, diwali date (Laxmi Pujan / Muhurat session day, YYYY-MM-DD)
    {"year": 2003, "diwali": "2003-10-25"},
    {"year": 2004, "diwali": "2004-11-12"},
    {"year": 2005, "diwali": "2005-11-01"},
    {"year": 2006, "diwali": "2006-10-21"},
    {"year": 2007, "diwali": "2007-11-09"},
    {"year": 2008, "diwali": "2008-10-28"},
    {"year": 2009, "diwali": "2009-10-17"},
    {"year": 2010, "diwali": "2010-11-05"},
    {"year": 2011, "diwali": "2011-10-26"},
    {"year": 2012, "diwali": "2012-11-13"},
    {"year": 2013, "diwali": "2013-11-03"},
    {"year": 2014, "diwali": "2014-10-23"},
    {"year": 2015, "diwali": "2015-11-11"},
    {"year": 2016, "diwali": "2016-10-30"},
    {"year": 2017, "diwali": "2017-10-19"},
    {"year": 2018, "diwali": "2018-11-07"},
    {"year": 2019, "diwali": "2019-10-27"},
    {"year": 2020, "diwali": "2020-11-14"},
    {"year": 2021, "diwali": "2021-11-04"},
    {"year": 2022, "diwali": "2022-10-24"},
    {"year": 2023, "diwali": "2023-11-12"},
    {"year": 2024, "diwali": "2024-11-01"},
    {"year": 2025, "diwali": "2025-10-21"},
]

DIWALI_DF: pd.DataFrame = pd.DataFrame(DIWALI_DATES)
DIWALI_DF["diwali"] = pd.to_datetime(DIWALI_DF["diwali"])


def diwali_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Diwali / Muhurat-session date table.

    Columns: ``year`` (int), ``diwali`` (datetime64, the Laxmi-Pujan / Muhurat
    session day). 23 rows, 2003–2025.
    """
    return DIWALI_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily-return panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    start: str = "2003-01-01",
    end: str = "2025-12-31",
    premium_bps: float = 0.0,
    base_bps: float = 4.0,
    vol_bps: float = 100.0,
    hold_days: int = 5,
    seed: int = 289,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily price series with an optional planted Diwali premium.

    Business-day returns are drawn i.i.d. from N(base_bps, vol_bps^2) in bps. On
    the ``hold_days`` sessions immediately following each Diwali date (from the
    hardcoded table), an extra ``premium_bps`` of daily drift is added. With
    ``premium_bps = 0`` there is no planted effect — the Diwali window is
    indistinguishable from any other window. This is the study's null in a bottle.

    Returns ``(price_df, truth)`` where ``price_df`` has a DatetimeIndex of
    business days and a single ``INDA`` column of synthetic prices (base 100),
    and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)

    rets = rng.normal(base_bps * 1e-4, vol_bps * 1e-4, n)

    # Mark the hold_days sessions on-or-after each Diwali.
    diwali = diwali_table()["diwali"].to_numpy()
    day_arr = days.to_numpy()
    in_window = np.zeros(n, dtype=bool)
    for d in diwali:
        # first session index on-or-after the event
        idx = np.searchsorted(day_arr, d, side="left")
        for k in range(idx, min(idx + hold_days, n)):
            in_window[k] = True
    rets = rets + np.where(in_window, premium_bps * 1e-4, 0.0)

    prices = 100.0 * np.exp(np.cumsum(np.log1p(rets)))
    price_df = pd.DataFrame({"INDA": prices}, index=days)

    truth = {
        "premium_bps": premium_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "hold_days": hold_days,
        "n_days": n,
        "seed": seed,
        "start": start,
        "end": end,
    }
    return price_df, truth


# ---------------------------------------------------------------------------
# Real tape — iShares MSCI India ETF (INDA) daily closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_inda(
    fetch: bool = False,
    cache_path: str = INDA_CACHE,
) -> pd.DataFrame:
    """INDA daily total-return prices; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/inda_daily.parquet``). Returns a frame
    with a DatetimeIndex and a single ``INDA`` column of adjusted closes
    (auto_adjust=True, i.e. total-return, dividends reinvested).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.

    **Proxy caveat.** INDA is a USD US-listed ETF tracking MSCI India; it does
    not trade in the Indian evening Muhurat window. We use the US trading day(s)
    following Diwali as the tradeable approximation.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached INDA panel at {cache_path}. "
                "Call fetch_inda(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "INDA",
        start="2012-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no INDA data")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
    else:
        closes = raw["Close"]

    out = pd.DataFrame({"INDA": closes})
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached INDA daily: {out.shape[0]} rows -> {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the INDA column (last row), for the as-of stamp."""
    arr = df["INDA"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
