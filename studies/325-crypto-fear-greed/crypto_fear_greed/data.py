"""Data layer for Study 325 (Crypto-Fear-Greed).

The public crypto **Fear & Greed Index** (alternative.me, 0 = Extreme Fear,
100 = Extreme Greed) is a 0--100 composite built from volatility, market
momentum/volume, social media, dominance and a Google-trends term.  The
contrarian folk claim, lifted straight from equities: **buy BTC when the gauge
is in Extreme Fear, lighten / sell in Extreme Greed.**  "Be greedy when others
are fearful."

The live API is network-blocked in this environment, and it has no long, clean,
free historical archive anyway.  So this study does **not** hardcode a curated
sentiment table (that is what the *equities* study 255-fear-greed-index does).
Instead it rebuilds a transparent **proxy gauge** from BTC's *own* observables --
exactly the inputs the published index weights most heavily:

  * **realised volatility** (high vol -> fear),
  * **drawdown** from the trailing 90-day high (deep drawdown -> fear),
  * **trailing momentum** (strong up-move -> greed).

Each component is mapped to a 0--100 percentile-style score over a rolling
window and blended; the result is a self-contained, reproducible Fear & Greed
proxy that needs nothing but the BTC price path.  This is honest about what it
is: a *price-derived* gauge, not the social-media-and-search index.  The
contrarian claim is really a claim about vol/drawdown/momentum mean-reversion,
and that is exactly what the proxy lets us test.

Three tapes, all offline for the reproducible core:

- ``synthetic_btc`` -- a deterministic, seeded daily BTC-like price path.  A
  ``reversion`` knob plants the *one* thing a contrarian fear gauge could
  harvest: short-horizon mean reversion (drops over-shoot and snap back).
  ``reversion = 0`` is a pure random walk -- the null.

- ``proxy_gauge`` -- the price-derived 0--100 Fear & Greed proxy (works on any
  price series, synthetic or real).

- ``load_real`` -- the real BTC-USD daily tape via the shared cache / yfinance,
  cache-first with graceful offline fallback to the synthetic tape.

No look-ahead is baked here -- that discipline lives in ``strategy.py`` (the
gauge known at the close of day *t* earns the return of day *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_CACHE = os.path.join(REPO_ROOT, "_cache")
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The shared repo-cache parquet for BTC, if a previous study populated it.
BTC_CACHE = os.path.join(REPO_CACHE, "BTC-USD_total_return.parquet")

TRADING_DAYS_PER_YEAR = 365  # crypto trades every day

# Published-style band edges for the crypto Fear & Greed Index (alternative.me).
#   0-24   Extreme Fear
#   25-44  Fear
#   45-55  Neutral
#   56-74  Greed
#   75-100 Extreme Greed
EXTREME_FEAR_MAX = 25.0
EXTREME_GREED_MIN = 75.0


# ---------------------------------------------------------------------------
# Synthetic BTC tape -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_btc(
    n_days: int = 2500,
    reversion: float = 0.0,
    daily_vol: float = 0.035,
    drift: float = 0.0008,
    vol_persist: float = 0.94,
    start: str = "2015-01-01",
    seed: int = 325,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily BTC-like close path with a known amount of reversion.

    Log returns follow ``r_t = -reversion * r_{t-1} + sigma_t * eps_t`` where
    ``sigma_t`` is a slowly varying (AR(1)) stochastic volatility around
    ``daily_vol`` -- crypto's defining feature, vol clustering, which makes the
    realised-vol leg of the gauge non-trivial.  The **negative** sign means a
    drop over-shoots and tends to snap back: this is the only structure a
    contrarian fear gauge can possibly harvest.

    - ``reversion = 0`` -> a martingale with vol clustering; the gauge is a fair coin.
    - ``reversion > 0`` -> mean reversion the contrarian rule can exploit.
    - ``reversion < 0`` -> momentum, which the contrarian rule fights.

    Returns ``(df, truth)`` where ``df`` has a single ``close`` column on a daily
    index, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    # Stochastic-volatility AR(1) in log-vol space (vol clustering).
    log_sig = np.empty(n_days)
    log_sig[0] = np.log(daily_vol)
    sig_innov = rng.normal(0.0, 0.15, n_days)
    for t in range(1, n_days):
        log_sig[t] = (
            np.log(daily_vol)
            + vol_persist * (log_sig[t - 1] - np.log(daily_vol))
            + np.sqrt(1 - vol_persist**2) * sig_innov[t]
        )
    sigma = np.exp(log_sig)

    log_ret = np.empty(n_days)
    prev = 0.0
    for t in range(n_days):
        eps = rng.normal(0.0, 1.0)
        r = drift - reversion * prev + sigma[t] * eps
        log_ret[t] = r
        prev = r

    close = 1000.0 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({"close": close}, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "reversion": reversion,
        "daily_vol": daily_vol,
        "drift": drift,
        "vol_persist": vol_persist,
        "n_days": n_days,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# The price-derived Fear & Greed proxy gauge
# ---------------------------------------------------------------------------
def _rolling_rank(x: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the last value within its trailing ``window`` (0..100).

    Uses only past+present data (the window ends at the current bar), so there is
    no look-ahead in the gauge itself.
    """
    def _rank(a: np.ndarray) -> float:
        last = a[-1]
        return 100.0 * (a <= last).mean()

    return x.rolling(window, min_periods=max(10, window // 3)).apply(_rank, raw=True)


def proxy_gauge(
    close: pd.Series,
    vol_window: int = 30,
    dd_window: int = 90,
    mom_window: int = 30,
    rank_window: int = 365,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pd.Series:
    """A 0--100 Fear & Greed proxy from BTC's own price observables.

    Three legs, each a rolling percentile score in [0, 100] where **higher means
    greedier**:

      * **vol leg** -- 100 minus the percentile rank of trailing realised vol
        (high vol -> low score -> fear).
      * **drawdown leg** -- 100 plus-rank of (price / trailing-high), so a deep
        drawdown -> low score -> fear; near a new high -> high score -> greed.
      * **momentum leg** -- percentile rank of trailing ``mom_window`` return
        (strong up-move -> high score -> greed).

    The three legs are blended with ``weights`` (default equal) into a single
    0--100 gauge.  All inputs use only past+present bars (no look-ahead in the
    gauge).  Returns a Series aligned to ``close`` (leading NaNs while the
    windows fill).

    This is a *price-derived* proxy for the alternative.me crypto Fear & Greed
    Index, not the social-media/search index itself -- but the published index
    is dominated by these same volatility/momentum ingredients, and the
    contrarian claim is fundamentally a claim about them.
    """
    close = close.astype(float)
    ret = close.pct_change()

    # Leg 1: realised volatility (annualised), inverted so high vol = fear.
    rv = ret.rolling(vol_window, min_periods=vol_window // 2).std()
    vol_score = 100.0 - _rolling_rank(rv, rank_window)

    # Leg 2: drawdown from trailing high; near high = greed, deep DD = fear.
    roll_high = close.rolling(dd_window, min_periods=dd_window // 3).max()
    dd = close / roll_high  # in (0, 1], 1 == at a new high
    dd_score = _rolling_rank(dd, rank_window)

    # Leg 3: trailing momentum; strong up-move = greed.
    mom = close / close.shift(mom_window) - 1.0
    mom_score = _rolling_rank(mom, rank_window)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    gauge = w[0] * vol_score + w[1] * dd_score + w[2] * mom_score
    gauge = gauge.clip(0.0, 100.0)
    gauge.name = "fng"
    gauge.index.name = "date"
    return gauge


def panel_from_close(close: pd.Series, **gauge_kw) -> pd.DataFrame:
    """Build a (close, fng) panel from a price series via the proxy gauge.

    Drops the leading rows where the gauge has not yet warmed up.  Returns a
    DataFrame with columns ``close`` and ``fng`` on a daily index.
    """
    close = close.astype(float)
    fng = proxy_gauge(close, **gauge_kw)
    df = pd.DataFrame({"close": close, "fng": fng}).dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Real BTC tape -- cache-first, graceful offline fallback
# ---------------------------------------------------------------------------
def _local_cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(cache_dir, "BTC-USD_daily.parquet")


def _read_close(frame: pd.DataFrame) -> pd.Series:
    """Pull a tz-naive daily close Series out of a (possibly MultiIndex) frame."""
    cols = frame.columns
    if isinstance(cols, pd.MultiIndex):
        close = frame["Close"] if "Close" in cols.get_level_values(0) else frame["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        name = "Close" if "Close" in cols else "close"
        close = frame[name]
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.name = "close"
    close.index.name = "date"
    return close.dropna()


def load_real(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series | None:
    """Real BTC-USD daily close, cache-first; returns ``None`` when unavailable.

    Resolution order, all offline unless ``fetch=True``:

      1. the shared repo cache ``_cache/BTC-USD_total_return.parquet`` if present;
      2. the study-local cache ``_cache/BTC-USD_daily.parquet`` if present;
      3. only if ``fetch=True``: yfinance ``BTC-USD`` (then write the local cache).

    Returns ``None`` (never raises) when no cache exists and ``fetch`` is False,
    so callers can fall back to the synthetic tape and stay fully offline.  This
    is the cache-first / graceful-fallback contract the desk requires.
    """
    for path in (BTC_CACHE, _local_cache_path(cache_dir)):
        if os.path.exists(path):
            try:
                return _read_close(pd.read_parquet(path))
            except Exception:
                continue

    if not fetch:
        return None

    try:
        import yfinance as yf  # lazy: only when explicitly fetching

        raw = yf.download(
            "BTC-USD", start="2014-09-17", interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw is None or raw.empty:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        raw.to_parquet(_local_cache_path(cache_dir))
        return _read_close(raw)
    except Exception:
        return None


def real_panel(fetch: bool = False, **gauge_kw) -> pd.DataFrame | None:
    """Real BTC (close, fng-proxy) panel, or ``None`` if no real tape is available."""
    close = load_real(fetch=fetch)
    if close is None:
        return None
    return panel_from_close(close, **gauge_kw)


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of the close column (or series), for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj["close"].to_numpy(dtype=float))
    else:
        arr = np.ascontiguousarray(obj.to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
