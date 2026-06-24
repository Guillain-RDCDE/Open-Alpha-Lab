"""Data layer for Study 429 (Mass Index).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` - a *deterministic, offline* generator with a planted-edge
  knob.  The Mass Index is a *reversal* detector: it sums the ratio of a 9-day EMA
  of the daily high-low range to a 9-day double-EMA of the same, over a 25-day
  window.  A "range bulge" (the index above 27) is meant to flag that a quiet
  trend is about to flip.  To give that claim something to harvest we plant a knob
  ``edge`` that injects a *post-bulge reversal*: after a wide-range stretch the
  drift sign flips for a few days.  At ``edge = 0`` the tape is a pure random walk
  with stochastic-volatility clustering (so the Mass Index still bulges, but tells
  you nothing) - the null.  At ``edge > 0`` a real post-bulge reversal exists, so
  the positive control proves the harness *can* bank the effect when it is there.

- ``load_real`` - the real Yahoo! daily tape (``yfinance``), cache-first so the
  test-suite and the reproducible core never touch the network unless the cache
  misses.  Daily history is long (20+ years) and free of the 60-day cap on
  sub-hourly bars.

No look-ahead is baked in here - that discipline lives in ``strategy.py``: the
index is formed on closes/ranges up to *t*, the position for *t+1* is taken from
the *t* signal (one ``shift``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Headline instrument + a small robustness panel (liquid, long-listed survivors).
HEADLINE = "SPY"
PANEL = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# ---------------------------------------------------------------------------
# Synthetic tape - the deterministic offline core (positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2005-01-03",
    seed: int = 429,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *known* post-bulge reversal edge.

    The price path is a random walk whose instantaneous volatility clusters
    (an AR(1) log-vol process), so the daily high-low range widens and narrows
    in regimes - which is exactly what makes the Mass Index bulge.  On top of
    that walk we plant, *only when* ``edge > 0``, a reversal: in the days right
    after a wide-range stretch, the local drift is set to the *negative* of the
    trailing drift, scaled by ``edge``.  That is the "range bulge calls a
    reversal" story, made literally true in the data generator.

    - ``edge = 0`` -> a martingale with vol clustering; the Mass Index bulges but
      carries no forecast.  This is the null the placebo must read as flat.
    - ``edge > 0`` -> a genuine post-bulge reversal the rule can harvest.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # --- 1. Schedule "bulge episodes": ~every 60-100 days, a stretch of WIDENING
    # intraday range that pushes the single/double-EMA ratio (hence the Mass Index)
    # up through 27 and back below 26.5.  vol_mult is the per-day range multiplier;
    # it is clipped so the price can never blow up (the old AR(1) overflowed).
    vol_mult = np.ones(n_days)
    bulge_starts = []                      # day each engineered bulge widening begins
    i = int(rng.integers(120, 180))
    while i < n_days - 120:
        widen = int(rng.integers(18, 26))  # ramp the range up over ~3-4 weeks
        narrow = int(rng.integers(10, 16))  # then let it collapse (MI falls back)
        ramp = np.linspace(1.0, 3.2, widen)
        fade = np.linspace(3.2, 1.0, narrow)
        seg = np.concatenate([ramp, fade])
        end = min(i + len(seg), n_days)
        vol_mult[i:end] = seg[: end - i]
        bulge_starts.append(i)
        i = end + int(rng.integers(45, 90))
    # mild everyday clustering noise on top, clipped to keep prices bounded
    vol_mult = np.clip(vol_mult * rng.lognormal(0.0, 0.10, n_days), 0.4, 4.0)

    # --- 2. Returns.  Base random walk.  Each engineered episode is given a clear
    # *pre-bulge trend* (an up- or down-leg over the widening ramp); when edge>0 the
    # trend then REVERSES for the ~25 days *after* the range collapses (i.e. after the
    # Mass Index has bulged and fallen back, which is exactly when the detector fires).
    # edge=0 -> the post-bulge window is pure noise (the null).
    log_ret = rng.normal(0.0, daily_vol, n_days) * vol_mult
    bulge_len = 40                                    # ramp(~22)+narrow(~13) span
    rev = 25
    for k, s in enumerate(bulge_starts):
        pre_sign = 1.0 if (k % 2 == 0) else -1.0     # alternate up-leg / down-leg
        ramp_end = min(s + 24, n_days)
        # a deterministic trend going INTO the bulge (visible to the SMA at detection)
        log_ret[s:ramp_end] += pre_sign * 1.2 * daily_vol
        if edge > 0.0:
            rstart = min(s + bulge_len, n_days)      # post-collapse = detection zone
            rend = min(rstart + rev, n_days)
            if rend > rstart:
                # reverse the prevailing trend: a true "range bulge calls a reversal"
                log_ret[rstart:rend] += -pre_sign * edge * daily_vol

    log_ret = np.clip(log_ret, -0.25, 0.25)          # hard safety clip
    close = 100.0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intraday wick scales with the (engineered) range multiplier -> the bulges.
    wick = np.abs(rng.normal(0.0, 1.0, n_days)) * daily_vol * vol_mult * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = (rng.integers(50_000, 500_000, n_days)).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "seed": seed,
        "n_planted_bulges": len(bulge_starts),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape - Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first**.

    If a cached parquet exists it is used (offline).  Otherwise - or with
    ``fetch=True`` - we go to ``yfinance`` (with a small back-off on rate limits)
    and cache the result so every re-run is offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
            except Exception:  # noqa: BLE001 - network flakiness
                raw = None
            if raw is not None and not raw.empty:
                break
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fetch_panel(
    ticker: str = HEADLINE,
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first (alias of :func:`load_real`).

    Provided so the verify script / panel loop reads like the sibling studies.
    """
    return load_real(ticker, start=start, end=end, fetch=fetch, cache_dir=cache_dir)


def have_real(ticker: str = HEADLINE, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True when a cached daily tape for ``ticker`` exists (offline-safe check)."""
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
