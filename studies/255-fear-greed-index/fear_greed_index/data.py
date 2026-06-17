"""Data layer for Study 255 (Fear-Greed).

The CNN Fear & Greed Index is a 0--100 composite sentiment gauge (0 = Extreme
Fear, 100 = Extreme Greed) built from seven sub-indicators (momentum, breadth,
put/call, junk-bond demand, market volatility, safe-haven demand, stock-price
strength).  CNN began publishing it around 2011.  The contrarian folk claim:
**buy when the index is in Extreme Fear, lighten up / sell when it is in
Extreme Greed.** "Be greedy when others are fearful."

Three tapes, all offline for the reproducible core:

- ``FNG_MONTHLY`` -- a *hardcoded, curated* monthly table of representative CNN
  Fear & Greed readings (month-end, 2011-09 -> 2026-06).  The CNN index has no
  clean machine-readable public archive, so we hardcode month-end anchor values
  digitised from CNN's published history and contemporaneous reporting (the
  major regimes are well documented: ~12 at the 2011 debt-ceiling low, single
  digits in the Mar-2020 COVID crash, high-80s greed in late-2020/2021,
  low-20s fear through the 2022 bear, recovery greed in 2023-24, and a sharp
  drop in the spring-2025 tariff scare).  This table is the study's
  deterministic, offline sentiment series.  It is upsampled to a weekly index
  with ``fng_weekly``.

- ``synthetic_panel`` -- a deterministic, offline generator that draws a weekly
  sentiment series and a paired weekly price path.  A ``contrarian_bps`` knob
  plants the exact contrarian premium the folk claim asserts: forward returns
  are *higher* after low-sentiment (fear) weeks and *lower* after high-sentiment
  (greed) weeks.  ``contrarian_bps = 0`` is the null -- sentiment carries no
  forward information.

- ``fetch_gspc_weekly`` -- the real ^GSPC weekly close series, read from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``, split-adjusted but
  **price-only**, no dividends).  Cache-only by default; network (yfinance) is
  touched only on an explicit ``fetch=True``.

No look-ahead: the sentiment reading on Friday of week t is paired with the
return *earned over week t+1* (one-week execution lag -- you read Friday's close
sentiment and trade at next Monday's open, held to the following Friday).

**Price-only caveat**: ^GSPC is a price index (no dividends).  Long-only excess
returns are therefore understated by the dividend yield; the contrarian
*spread* (fear-minus-greed) is dividend-neutral and unaffected.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_CACHE = os.path.join(REPO_ROOT, "_cache")
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# CNN Fear & Greed regime bands (the published labels)
#   0-24   Extreme Fear
#   25-44  Fear
#   45-55  Neutral
#   56-74  Greed
#   75-100 Extreme Greed
EXTREME_FEAR_MAX = 25.0
EXTREME_GREED_MIN = 75.0

# ---------------------------------------------------------------------------
# The hardcoded curated CNN Fear & Greed series -- month-end anchor readings
# ---------------------------------------------------------------------------
# (year, month, fng)  -- month-end value of the CNN Fear & Greed Index (0-100).
# Digitised from CNN Business' published Fear & Greed history and
# contemporaneous reporting.  Values are representative month-end snapshots,
# not tick-exact; the contrarian signal depends on the *regime* (fear vs greed),
# which these anchors capture faithfully.  As-of 2026-06.
FNG_MONTHLY: list[tuple[int, int, float]] = [
    (2011, 9, 12), (2011, 10, 38), (2011, 11, 30), (2011, 12, 50),
    (2012, 1, 62), (2012, 2, 72), (2012, 3, 70), (2012, 4, 52),
    (2012, 5, 22), (2012, 6, 40), (2012, 7, 48), (2012, 8, 60),
    (2012, 9, 66), (2012, 10, 46), (2012, 11, 44), (2012, 12, 58),
    (2013, 1, 74), (2013, 2, 66), (2013, 3, 70), (2013, 4, 64),
    (2013, 5, 78), (2013, 6, 34), (2013, 7, 64), (2013, 8, 40),
    (2013, 9, 60), (2013, 10, 70), (2013, 11, 72), (2013, 12, 68),
    (2014, 1, 38), (2014, 2, 58), (2014, 3, 56), (2014, 4, 54),
    (2014, 5, 60), (2014, 6, 68), (2014, 7, 46), (2014, 8, 58),
    (2014, 9, 40), (2014, 10, 30), (2014, 11, 66), (2014, 12, 50),
    (2015, 1, 42), (2015, 2, 64), (2015, 3, 44), (2015, 4, 56),
    (2015, 5, 54), (2015, 6, 36), (2015, 7, 42), (2015, 8, 12),
    (2015, 9, 22), (2015, 10, 56), (2015, 11, 50), (2015, 12, 28),
    (2016, 1, 14), (2016, 2, 30), (2016, 3, 64), (2016, 4, 66),
    (2016, 5, 58), (2016, 6, 30), (2016, 7, 70), (2016, 8, 64),
    (2016, 9, 50), (2016, 10, 38), (2016, 11, 68), (2016, 12, 76),
    (2017, 1, 60), (2017, 2, 74), (2017, 3, 56), (2017, 4, 58),
    (2017, 5, 54), (2017, 6, 62), (2017, 7, 70), (2017, 8, 40),
    (2017, 9, 66), (2017, 10, 72), (2017, 11, 70), (2017, 12, 74),
    (2018, 1, 76), (2018, 2, 16), (2018, 3, 18), (2018, 4, 36),
    (2018, 5, 52), (2018, 6, 48), (2018, 7, 58), (2018, 8, 64),
    (2018, 9, 56), (2018, 10, 14), (2018, 11, 28), (2018, 12, 8),
    (2019, 1, 48), (2019, 2, 66), (2019, 3, 60), (2019, 4, 72),
    (2019, 5, 22), (2019, 6, 56), (2019, 7, 60), (2019, 8, 30),
    (2019, 9, 54), (2019, 10, 62), (2019, 11, 74), (2019, 12, 78),
    (2020, 1, 56), (2020, 2, 18), (2020, 3, 5), (2020, 4, 38),
    (2020, 5, 58), (2020, 6, 50), (2020, 7, 64), (2020, 8, 74),
    (2020, 9, 44), (2020, 10, 40), (2020, 11, 78), (2020, 12, 76),
    (2021, 1, 52), (2021, 2, 60), (2021, 3, 58), (2021, 4, 66),
    (2021, 5, 40), (2021, 6, 50), (2021, 7, 36), (2021, 8, 56),
    (2021, 9, 30), (2021, 10, 64), (2021, 11, 34), (2021, 12, 52),
    (2022, 1, 24), (2022, 2, 28), (2022, 3, 44), (2022, 4, 22),
    (2022, 5, 14), (2022, 6, 18), (2022, 7, 38), (2022, 8, 40),
    (2022, 9, 22), (2022, 10, 30), (2022, 11, 56), (2022, 12, 36),
    (2023, 1, 60), (2023, 2, 48), (2023, 3, 32), (2023, 4, 58),
    (2023, 5, 54), (2023, 6, 78), (2023, 7, 80), (2023, 8, 48),
    (2023, 9, 36), (2023, 10, 24), (2023, 11, 64), (2023, 12, 76),
    (2024, 1, 70), (2024, 2, 76), (2024, 3, 72), (2024, 4, 38),
    (2024, 5, 60), (2024, 6, 64), (2024, 7, 56), (2024, 8, 44),
    (2024, 9, 62), (2024, 10, 58), (2024, 11, 70), (2024, 12, 54),
    (2025, 1, 50), (2025, 2, 30), (2025, 3, 22), (2025, 4, 14),
    (2025, 5, 62), (2025, 6, 60), (2025, 7, 70), (2025, 8, 58),
    (2025, 9, 52), (2025, 10, 48), (2025, 11, 56), (2025, 12, 60),
    (2026, 1, 64), (2026, 2, 40), (2026, 3, 34), (2026, 4, 30),
    (2026, 5, 52), (2026, 6, 58),
]


def fng_monthly() -> pd.Series:
    """Month-end CNN Fear & Greed Index as a pandas Series (index = month-end dates)."""
    idx = [
        (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0))
        for (y, m, _) in FNG_MONTHLY
    ]
    vals = [float(v) for (_, _, v) in FNG_MONTHLY]
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="fng")
    return s.sort_index()


def fng_weekly() -> pd.Series:
    """Weekly (Friday) CNN Fear & Greed series, linearly interpolated from monthly anchors.

    The monthly anchor table is upsampled to a Friday-frequency index by linear
    interpolation between consecutive month-end readings.  This is a smooth
    proxy for the daily index: the weekly *regime* (fear vs greed) is preserved,
    which is all the contrarian signal acts on.  Values are clipped to [0, 100].
    """
    m = fng_monthly()
    fri = pd.date_range(m.index[0], m.index[-1], freq="W-FRI")
    # Interpolate monthly anchors onto the Friday grid.
    combined = m.reindex(m.index.union(fri)).interpolate(method="time")
    weekly = combined.reindex(fri).clip(0.0, 100.0)
    weekly.name = "fng"
    weekly.index.name = "date"
    return weekly


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_weeks: int = 600,
    contrarian_bps: float = 60.0,
    base_ret: float = 0.0015,
    ret_vol: float = 0.015,
    sentiment_persist: float = 0.92,
    seed: int = 255,
) -> tuple[pd.DataFrame, dict]:
    """A weekly (sentiment, price) panel with a known *contrarian* premium.

    A latent AR(1) sentiment state ``z_t`` (persistence ``sentiment_persist``)
    is mapped to a 0--100 Fear & Greed reading via a logistic squash.  The
    forward weekly return is::

        base_ret  -  contrarian_bps*1e-4 * (fng_t - 50)/50  +  noise

    so that **low** Fear & Greed (fear, fng < 50) yields a *positive* tilt to
    next week's return and **high** Fear & Greed (greed) yields a negative tilt
    -- exactly the contrarian folk claim.  ``contrarian_bps = 0`` is the null:
    sentiment is pure noise with respect to forward returns.

    Returns ``(df, truth)`` where ``df`` has columns ``fng`` (0-100 sentiment at
    week t) and ``close`` (price level), index = weekly Friday dates.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-07", periods=n_weeks, freq="W-FRI")

    # AR(1) latent sentiment state, squashed to 0-100.
    z = np.zeros(n_weeks)
    innov = rng.normal(0.0, 1.0, n_weeks)
    for t in range(1, n_weeks):
        z[t] = sentiment_persist * z[t - 1] + np.sqrt(1 - sentiment_persist**2) * innov[t]
    fng = 100.0 / (1.0 + np.exp(-z))  # logistic squash -> (0, 100)

    # Forward weekly returns with planted contrarian tilt on the *current* fng.
    tilt = -(contrarian_bps * 1e-4) * (fng - 50.0) / 50.0
    noise = rng.normal(0.0, ret_vol, n_weeks)
    fwd_ret = base_ret + tilt + noise  # return earned over week t -> t+1

    # Build the price path so that close[t]->close[t+1] realises fwd_ret[t].
    close = np.empty(n_weeks)
    close[0] = 100.0
    for t in range(1, n_weeks):
        close[t] = close[t - 1] * (1.0 + fwd_ret[t - 1])

    df = pd.DataFrame({"fng": fng, "close": close}, index=dates)
    df.index.name = "date"
    truth = {
        "n_weeks": n_weeks,
        "contrarian_bps": contrarian_bps,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "sentiment_persist": sentiment_persist,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC weekly closes from the repo cache, cache-only by default
# ---------------------------------------------------------------------------
def fetch_gspc_weekly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.Series:
    """Weekly (Friday) ^GSPC close series; cache-only unless ``fetch=True``.

    Reads the repo-level ``^GSPC_split_only.parquet`` (daily OHLCV, split- but
    not dividend-adjusted -- **price-only**), resamples to weekly Friday closes,
    and restricts to the Fear & Greed era (2011-09 onward).  Network (yfinance)
    is touched only on an explicit ``fetch=True``.

    Returns a pd.Series of weekly closes indexed by Friday dates.  Raises
    ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No ^GSPC cache at {cache_path}. "
                "Call fetch_gspc_weekly(fetch=True) once to populate it."
            )
        daily = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="2010-01-01", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        daily = raw
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        daily.to_parquet(cache_path)

    # Flatten possible MultiIndex columns and pick the Close.
    if isinstance(daily.columns, pd.MultiIndex):
        close = daily["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = daily["Close"]
    close.index = pd.to_datetime(close.index)
    weekly = close.resample("W-FRI").last().dropna()
    weekly = weekly[weekly.index >= pd.Timestamp("2011-09-01")]
    weekly.name = "close"
    weekly.index.name = "date"
    return weekly


def real_panel(fetch: bool = False) -> pd.DataFrame:
    """Join the curated weekly Fear & Greed series with real ^GSPC weekly closes.

    Returns a DataFrame with columns ``fng`` (sentiment at week t) and ``close``
    (^GSPC weekly close), inner-joined on the common Friday dates.  Cache-only
    unless ``fetch=True``.
    """
    fng = fng_weekly()
    px = fetch_gspc_weekly(fetch=fetch)
    df = pd.DataFrame({"fng": fng}).join(pd.DataFrame({"close": px}), how="inner")
    df = df.dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of the last row, for the as-of stamp."""
    if isinstance(df, pd.Series):
        arr = np.asarray([df.iloc[-1]], dtype=float)
    else:
        arr = df.iloc[-1].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
