"""Data layer for Study 343 (Data-Mining-Roulette).

The p-hacking machine needs only *one price series* and a basket of *features* to mine.
The whole study is about what happens when you search a large rule-space against a tape,
so the data layer supplies two kinds of tape with the *same shape* (a price series plus
a feature matrix), and one knob that decides whether there is anything real to find.

Two tapes, one shape:

- ``synthetic_tape`` — a *deterministic, offline* generator. One knob, ``planted_edge``,
  decides the experiment:
    * ``planted_edge = 0`` → the price is a pure random walk and every feature is pure
      noise, *uncorrelated with tomorrow's return*. This is the **null**: there is no
      rule that beats the market except by luck. Any "winning" backtest here is a false
      positive, and counting them is the whole point.
    * ``planted_edge > 0`` → one designated feature (the "signal feature", index 0) genuinely
      predicts next-day returns with that strength. This is the **positive control**: a
      faithful miner must rediscover that rule and rank it at the top. A pipeline that
      can't bank a planted edge proves nothing by finding nothing in the null.

  The returns are i.i.d. by default (a clean null for inference); a ``vol_cluster`` knob
  adds GARCH-like volatility clustering so the block-bootstrap and HAC machinery have
  something to respect.

- ``load_real`` — a real daily price series (default SPY total-return) via the shared
  ``quantlab.data`` loader, cache-first parquet with graceful offline fallback. The same
  feature builder runs on it; the same roulette is spun. The real tape is *not* a clean
  null — there may be a faint real effect buried in it — which is exactly why the
  Reality Check matters: it tells you whether the best rule clears the bar set by luck.

Features are built only from information available **at the close of day t** (lagged
returns, moving-average gaps, rolling z-scores, calendar dummies), so a rule formed at
the close of t and earning the return of t+1 carries exactly one execution lag — applied
once, in ``strategy.py``. No look-ahead is baked into the features here.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Names of the engineered features. Index 0 is the designated "signal feature" the
# positive-control planted edge loads onto; the rest are decoys that are pure noise
# under the null. These are the columns the random rules draw from.
FEATURE_NAMES = [
    "f0_signal",       # the planted-edge carrier (noise when planted_edge=0)
    "mom_5",           # 5-day momentum (sum of lagged returns)
    "mom_20",          # 20-day momentum
    "ma_gap_10",       # price vs 10-day moving average (gap, lagged)
    "ma_gap_50",       # price vs 50-day moving average
    "zscore_20",       # rolling 20-day z-score of return
    "rsi_proxy_14",    # a bounded oscillator proxy
    "vol_20",          # 20-day realised vol (lagged)
    "dow",             # day-of-week dummy (0..4) scaled
    "noise_a",         # pure decoy
    "noise_b",         # pure decoy
    "noise_c",         # pure decoy
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_tape(
    n_days: int = 2520,
    planted_edge: float = 0.0,
    daily_vol: float = 0.01,
    drift: float = 0.0002,
    vol_cluster: float = 0.0,
    seed: int = 343,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible daily tape: a price series + a feature matrix, with a known edge.

    Returns ``(features, fwd_ret, truth)`` where:

    - ``features`` is a (n_days x n_features) frame of values **known at the close of t**
      (every column is built from information up to and including day t — already lagged
      where it would otherwise peek). These are the variables the random rules condition on.
    - ``fwd_ret`` is the **next-day** simple return, aligned so ``fwd_ret[t]`` is the return
      a rule formed at the close of t would earn (the one and only execution lag is encoded
      here once, and ``strategy.py`` never shifts again).
    - ``truth`` records the planted parameters (``planted_edge``, the signal-feature index, …).

    Mechanism (``planted_edge`` is the only edge-bearing knob):

        eps_t      ~ N(0, sigma_t)                          # idiosyncratic noise
        f0_t       ~ N(0, 1)                                # the signal feature (std normal)
        fwd_ret_t  = drift + planted_edge * f0_t * sigma_t + eps_{t+1}

    At ``planted_edge = 0`` the forward return is independent of *every* feature — a pure
    null. At ``planted_edge > 0`` the sign and size of ``f0`` genuinely predict tomorrow,
    so a rule like "go long when f0 > 0" has a real, recoverable edge. All other features
    are noise in both regimes (decoys), so any rule built on them that "works" is luck.

    ``vol_cluster > 0`` turns sigma_t into a slow AR(1) process (GARCH-ish), giving the
    autocorrelation-robust machinery something to handle without changing the *mean* edge.
    """
    rng = np.random.default_rng(seed)

    # Time-varying vol (optional clustering). Kept strictly positive and mean ~ daily_vol.
    if vol_cluster > 0:
        log_sig = np.zeros(n_days)
        shock = rng.normal(0.0, 1.0, n_days)
        for t in range(1, n_days):
            log_sig[t] = vol_cluster * log_sig[t - 1] + (1.0 - vol_cluster) * 0.0 + 0.25 * shock[t]
        sigma = daily_vol * np.exp(log_sig - 0.5 * np.var(log_sig))
    else:
        sigma = np.full(n_days, daily_vol)

    eps = rng.normal(0.0, 1.0, n_days) * sigma          # idiosyncratic return noise
    f0 = rng.normal(0.0, 1.0, n_days)                   # the signal feature

    # Forward (next-day) return: edge loads on f0 (scaled by that day's vol), plus drift+noise.
    fwd_ret = np.full(n_days, np.nan)
    fwd_ret[:-1] = drift + planted_edge * f0[:-1] * sigma[:-1] + eps[1:]
    fwd_ret[-1] = drift + eps[-1]                       # last day has no realised t+1; set drift+noise

    # Build a price path from the *realised* returns (so MA/momentum features are coherent).
    # The realised return on day t is fwd_ret[t-1] (what was earned overnight into t).
    realised = np.empty(n_days)
    realised[0] = 0.0
    realised[1:] = fwd_ret[:-1]
    price = 100.0 * np.cumprod(1.0 + realised)

    # ---- Feature matrix, every column known at the CLOSE of t (lagged where needed) ----
    px = pd.Series(price)
    ret = pd.Series(realised)

    feats = pd.DataFrame(index=range(n_days))
    feats["f0_signal"] = f0                                              # std-normal carrier
    feats["mom_5"] = ret.rolling(5).sum().shift(0)                      # uses returns up to t
    feats["mom_20"] = ret.rolling(20).sum()
    feats["ma_gap_10"] = (px / px.rolling(10).mean() - 1.0)
    feats["ma_gap_50"] = (px / px.rolling(50).mean() - 1.0)
    feats["zscore_20"] = ((ret - ret.rolling(20).mean()) / (ret.rolling(20).std(ddof=0) + 1e-12))
    # A bounded RSI-like proxy in [-1, 1].
    up = ret.clip(lower=0).rolling(14).mean()
    dn = (-ret.clip(upper=0)).rolling(14).mean()
    feats["rsi_proxy_14"] = (up - dn) / (up + dn + 1e-12)
    feats["vol_20"] = ret.rolling(20).std(ddof=0)
    feats["dow"] = (np.arange(n_days) % 5 - 2.0) / 2.0                  # deterministic calendar decoy
    feats["noise_a"] = rng.normal(0.0, 1.0, n_days)
    feats["noise_b"] = rng.normal(0.0, 1.0, n_days)
    feats["noise_c"] = rng.normal(0.0, 1.0, n_days)
    feats = feats[FEATURE_NAMES]

    # Decorative daily date labels via a SAFE range (no overflow; well under ~290y).
    idx = pd.date_range("2005-01-03", periods=n_days, freq="B")
    feats.index = pd.DatetimeIndex(idx, name="date")
    fwd = pd.Series(fwd_ret, index=feats.index, name="fwd_ret")

    truth = {
        "n_days": n_days,
        "planted_edge": planted_edge,
        "signal_feature": "f0_signal",
        "signal_feature_idx": 0,
        "daily_vol": daily_vol,
        "drift": drift,
        "vol_cluster": vol_cluster,
        "seed": seed,
        "n_features": len(FEATURE_NAMES),
    }
    return feats, fwd, truth


# ---------------------------------------------------------------------------
# Real tape — daily prices via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str = "SPY") -> str:
    return os.path.join(cache_dir, f"roulette_{ticker}.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1995-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real daily tape (default SPY total-return) → the SAME ``(features, fwd_ret)`` shape.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and tests stay deterministic.

    The real tape is *not* a clean null — a faint real effect may exist — which is exactly
    why the Reality Check is the point: it tells you whether the *best of N* random rules
    clears the bar that luck alone sets. No survivorship issue here (a single liquid ETF),
    so no opt-in guard is needed; the honesty axis is multiple-testing, not survivorship.

    Returns ``(features, fwd_ret)`` with one execution lag already encoded in ``fwd_ret``
    (``fwd_ret[t]`` is the t→t+1 return of a rule formed at the close of t).
    """
    cpath = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached tape at {cpath}. "
                f"Call load_real(fetch=True) once to populate (needs network)."
            )
        px = pd.read_parquet(cpath)["close"]
    else:
        px = _fetch_prices(ticker, start)
        os.makedirs(cache_dir, exist_ok=True)
        px.to_frame("close").to_parquet(cpath)

    px = px.dropna()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    # Drop a partial final day already implicit (daily closes); compute returns.
    ret = px.pct_change().fillna(0.0)
    feats = _features_from_price(px, ret)
    # Forward (next-day) return aligned to the close-of-t signal: fwd_ret[t] = ret[t+1].
    fwd = ret.shift(-1).rename("fwd_ret")
    # Drop the last row (no realised t+1) and warm-up rows with NaN features.
    valid = feats.dropna().index.intersection(fwd.dropna().index)
    feats = feats.loc[valid]
    fwd = fwd.loc[valid]
    return feats, fwd


def _features_from_price(px: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """The SAME feature builder as the synthetic tape, sans the planted f0 (uses momentum)."""
    feats = pd.DataFrame(index=px.index)
    # On the real tape there is no planted carrier; f0_signal becomes 1-day lagged return
    # (a plausible decoy) so the column set matches the synthetic shape exactly.
    feats["f0_signal"] = ret.shift(1)
    feats["mom_5"] = ret.rolling(5).sum()
    feats["mom_20"] = ret.rolling(20).sum()
    feats["ma_gap_10"] = (px / px.rolling(10).mean() - 1.0)
    feats["ma_gap_50"] = (px / px.rolling(50).mean() - 1.0)
    feats["zscore_20"] = ((ret - ret.rolling(20).mean()) / (ret.rolling(20).std(ddof=0) + 1e-12))
    up = ret.clip(lower=0).rolling(14).mean()
    dn = (-ret.clip(upper=0)).rolling(14).mean()
    feats["rsi_proxy_14"] = (up - dn) / (up + dn + 1e-12)
    feats["vol_20"] = ret.rolling(20).std(ddof=0)
    feats["dow"] = (px.index.dayofweek.to_numpy() - 2.0) / 2.0
    # Deterministic decoys seeded off the index length (reproducible across reloads).
    rng = np.random.default_rng(343)
    feats["noise_a"] = rng.normal(0.0, 1.0, len(px))
    feats["noise_b"] = rng.normal(0.0, 1.0, len(px))
    feats["noise_c"] = rng.normal(0.0, 1.0, len(px))
    return feats[FEATURE_NAMES]


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily total-return closes via the shared loader, falling back to yfinance."""
    try:
        from quantlab import data as qld

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def fingerprint(features: pd.DataFrame, fwd: pd.Series) -> str:
    """A short content fingerprint of the tape, for the as-of stamp."""
    a = np.ascontiguousarray(features.to_numpy().ravel())
    b = np.ascontiguousarray(fwd.to_numpy().ravel())
    h = hashlib.sha1()
    h.update(a.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()[:12]
