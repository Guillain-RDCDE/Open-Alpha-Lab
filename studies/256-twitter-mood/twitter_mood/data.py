"""Data layer for Study 256 (Twitter-Mood).

The claim (Bollen, Mao & Zeng 2011, *Twitter mood predicts the stock market*):
a daily aggregate "Calm" mood index extracted from ~10M tweets Granger-causes
the Dow 3-4 days later, lifting a directional next-day model to ~87% accuracy.

This study mirrors that *daily time-series* set-up offline and deterministically.

Three components, all offline for the reproducible core:

- ``CALM_ANCHORS`` -- a hardcoded table of curated daily "Calm" mood readings
  for the original Bollen window (late Feb -> mid-Dec 2008).  The original
  OpinionFinder / GPOMS feed was never released publicly, so this is a
  *stylized reconstruction*: a handful of well-documented public-mood anchor
  points (the calm before the September crash, the panic of the Lehman week,
  the pre-election dip, the Thanksgiving / Obama-election calm spike Bollen
  himself highlighted) deterministically interpolated to business days.  It is
  a curated proxy, NOT the proprietary feed -- this is named on the Signal axis.

- ``synthetic_series`` -- a deterministic, offline generator of a (mood, return)
  pair with an optional planted ``predictive`` knob.  ``predictive = 0`` is the
  null (mood carries no forward information); a positive knob makes lagged Calm
  predict the next day's return, so tests can confirm the engine is truthful.

- ``fetch_index`` -- the real ^GSPC (S&P 500) daily total-return-free price
  series via yfinance, cache-only by default (``_cache/gspc_daily.parquet``).
  The Dow was Bollen's tape; the S&P is the cleaner, more liquid proxy and the
  desk standard.  No look-ahead: mood on day t predicts the return *earned* on
  day t+1 (positions formed at the t close, held one day -- a one-day
  execution lag).

**Honesty caveats** named on the Signal axis:
- The mood series is a curated reconstruction, not the proprietary GPOMS feed.
- The original window is ~10 trading months (n < 200 business days): far too
  small to detect a 1-3 day lead-lag effect with any power.
- Price-only S&P (no dividends); no survivorship issue (single index).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

INDEX_CACHE = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")
INDEX_TICKER = "^GSPC"

# The Bollen et al. (2011) study window: the GPOMS/OpinionFinder feed ran from
# 2008-02-28 through 2008-12-19 (the famous ~87% directional-accuracy window).
WINDOW_START = "2008-02-28"
WINDOW_END = "2008-12-19"

# ---------------------------------------------------------------------------
# Curated "Calm" mood anchors -- the study's hardcoded, offline core
# ---------------------------------------------------------------------------
# Each anchor is (date, calm_z) where calm_z is a standardized "Calm" mood
# reading (higher = calmer / more positive public mood, ~N(0,1) scaled).
# These are CURATED stylized values pinned to well-documented 2008 public-mood
# moments -- NOT the proprietary GPOMS feed (which was never released).  The
# anchors are deterministically interpolated to all business days in-window.
#
# Sources / rationale for each anchor:
#   - Bollen, J., Mao, H. & Zeng, X. (2011). "Twitter mood predicts the stock
#     market." Journal of Computational Science 2(1), 1-8.  Figure 2 of that
#     paper plots Calm dipping before the bank bailout / election and spiking
#     around Thanksgiving and the Obama election -- the anchors echo that shape.
#   - Public-mood timeline of late 2008: pre-crash calm (spring), the Lehman
#     week panic (mid-Sep), the bailout-vote turmoil (late Sep / early Oct),
#     the election-eve dip, the post-election + Thanksgiving calm spike.
CALM_ANCHORS: list[tuple[str, float]] = [
    ("2008-02-28",  0.30),  # late-winter baseline, pre-crisis calm
    ("2008-03-17", -0.45),  # Bear Stearns collapse weekend
    ("2008-04-15",  0.35),  # spring relief rally calm
    ("2008-05-19",  0.55),  # local market top -- mood calm/complacent
    ("2008-06-16",  0.10),  # oil spike unease creeping in
    ("2008-07-15", -0.30),  # IndyMac failure / financials wobble
    ("2008-08-11",  0.20),  # late-summer lull
    ("2008-09-08", -0.10),  # Fannie/Freddie conservatorship
    ("2008-09-15", -1.40),  # Lehman bankruptcy -- peak anxiety
    ("2008-09-29", -1.10),  # House rejects bailout, DJIA -777
    ("2008-10-10", -1.60),  # crash week trough, VIX ~70
    ("2008-10-27", -0.70),  # tentative stabilization
    ("2008-11-04",  0.40),  # US election day -- mood lifts
    ("2008-11-10",  0.60),  # Bollen's flagged post-election calm spike
    ("2008-11-20", -0.50),  # market new lows, renewed gloom
    ("2008-11-27",  0.75),  # Thanksgiving calm spike (Bollen Fig. 2)
    ("2008-12-05",  0.20),  # December stabilization
    ("2008-12-16",  0.45),  # Fed cuts to ~zero, year-end calm
    ("2008-12-19",  0.30),  # window close
]


def calm_index() -> pd.Series:
    """Curated daily 'Calm' mood index over the Bollen 2008 window.

    Hardcoded anchor points (see ``CALM_ANCHORS``) are linearly interpolated to
    every business day in ``[WINDOW_START, WINDOW_END]`` and lightly smoothed
    (the original feed was a moving-average of daily tweet sentiment).  The
    result is a deterministic, offline pd.Series indexed by business day, in
    standardized 'Calm' units (higher = calmer public mood).

    This is a CURATED PROXY of the GPOMS 'Calm' dimension, not the proprietary
    feed -- the reconstruction is named on the Signal axis and exists only to
    keep the daily lead-lag machinery fully reproducible offline.
    """
    anchor_dates = pd.to_datetime([d for d, _ in CALM_ANCHORS])
    anchor_vals = np.array([v for _, v in CALM_ANCHORS], dtype=float)

    bdays = pd.bdate_range(WINDOW_START, WINDOW_END)
    x_all = bdays.view("int64").astype(float)
    x_anchor = anchor_dates.view("int64").astype(float)
    interp = np.interp(x_all, x_anchor, anchor_vals)

    s = pd.Series(interp, index=bdays, name="calm")
    # Light 3-day smoothing to echo the original moving-average feed; the
    # interpolation already removes look-ahead concerns (no return data used).
    s = s.rolling(3, min_periods=1, center=True).mean()
    return s


# ---------------------------------------------------------------------------
# Synthetic (mood, return) series -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 200,
    predictive: float = 0.0,
    base_vol: float = 0.013,
    mood_persist: float = 0.85,
    seed: int = 256,
) -> tuple[pd.DataFrame, dict]:
    """A daily (mood, return) panel with an optional planted lead-lag effect.

    The mood process is a standardized AR(1) (``mood_persist``).  The next-day
    return is::

        ret_{t+1} = predictive * calm_t + base_vol * noise_{t+1}

    so ``predictive = 0`` is the null -- yesterday's mood carries no information
    about today's return.  A positive ``predictive`` plants exactly the Bollen
    lead-lag the strategy is built to detect.

    Returns ``(df, truth)`` where ``df`` has a business-day index and columns
    ``calm`` (standardized mood) and ``ret`` (daily simple return), and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    # AR(1) mood, standardized
    calm = np.zeros(n_days)
    eps = rng.normal(0.0, 1.0, n_days)
    for t in range(1, n_days):
        calm[t] = mood_persist * calm[t - 1] + np.sqrt(1 - mood_persist**2) * eps[t]
    calm = (calm - calm.mean()) / (calm.std(ddof=0) + 1e-12)

    noise = rng.normal(0.0, base_vol, n_days)
    ret = np.zeros(n_days)
    # ret_t driven by calm_{t-1}
    ret[1:] = predictive * calm[:-1] + noise[1:]
    ret[0] = noise[0]

    df = pd.DataFrame({"calm": calm, "ret": ret}, index=dates)
    truth = {
        "n_days": n_days,
        "predictive": predictive,
        "base_vol": base_vol,
        "mood_persist": mood_persist,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC daily closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_index(
    fetch: bool = False,
    cache_path: str = INDEX_CACHE,
    start: str = "2007-06-01",
    end: str = "2009-06-30",
) -> pd.Series:
    """Daily ^GSPC close series; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/gspc_daily.parquet``).  Returns a Series
    of daily closes indexed by trading day.  Price-only (no dividends) -- this
    is labelled on the Signal axis but is immaterial at a 1-day horizon.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``;
    callers must guard with ``os.path.exists(cache_path)``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached index series at {cache_path}. "
                "Call fetch_index(fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(cache_path)["close"]
        s.index = pd.to_datetime(s.index)
        return s.rename("close")

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        INDEX_TICKER,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data for ^GSPC")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index().rename("close")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    print(f"Cached {len(close)} daily ^GSPC closes -> {cache_path}")
    return close


def build_real_panel(
    fetch: bool = False,
    cache_path: str = INDEX_CACHE,
) -> pd.DataFrame:
    """Join the curated Calm index with real ^GSPC next-day returns.

    Returns a DataFrame indexed by trading day with columns ``calm`` (curated
    mood, standardized) and ``ret`` (the *same-day* simple return).  The
    strategy lags ``calm`` by one day, so mood on day t predicts ret on t+1.
    The frame is restricted to the Bollen window.
    """
    close = fetch_index(fetch=fetch, cache_path=cache_path)
    ret = close.pct_change()
    calm = calm_index()

    df = pd.DataFrame({"calm": calm}).join(pd.DataFrame({"ret": ret}), how="inner")
    df = df.loc[WINDOW_START:WINDOW_END].dropna()
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(s: pd.Series | pd.DataFrame) -> str:
    """A short content fingerprint of a series/frame, for the as-of stamp."""
    if isinstance(s, pd.DataFrame):
        arr = s.to_numpy(dtype=float).ravel()
    else:
        arr = s.dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
