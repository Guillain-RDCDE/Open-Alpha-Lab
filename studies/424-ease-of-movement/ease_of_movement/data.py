"""Data layer for Study 424 (Ease of Movement).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  The Ease-of-Movement (EOM) indicator is a *volume-weighted momentum* oscillator: it
  is positive when price drifts up on light volume ("effortless" up move) and negative
  when it drifts down on light volume. The only structure such a rule can harvest is a
  return that *persists in the same direction as the recent low-effort drift* — i.e.
  short-horizon trend/persistence. ``edge = 0`` is a pure random walk (a fair coin for
  any timing rule); ``edge > 0`` plants exactly that persistence (an AR(1) momentum
  term coupled to *inverse* volume, so the move that EOM sees as "easy" really does
  continue). This is the study's null in a bottle plus its positive control.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first by default so
  the test-suite and the reproducible core never touch the network. Daily bars go back
  decades, giving high statistical power compared with intraday studies.

Richard Arms' Ease of Movement (``EMV``, 1977) is defined per bar as::

    midpoint_move = (high_t + low_t)/2 - (high_{t-1} + low_{t-1})/2
    box_ratio     = (volume / scale) / (high_t - low_t)
    EMV_1(t)      = midpoint_move / box_ratio
    EOM(t)        = SMA_n( EMV_1 )            # n typically 14

The folk rule: EOM > 0 means price is rising with little volume resistance (an
"effortless" advance) — go long; EOM < 0 means an effortless decline — go flat or
short. ``scale`` is a free constant (Arms used 100,000,000 for stocks) that only
rescales EMV and never changes its sign, so the *sign-based* timing rule is invariant
to it.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal
is formed on closes up to *t*, the position is held over *t+1*'s return: one shift).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core with a planted-edge knob
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 2500,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    start: str = "2014-01-02",
    seed: int = 424,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable EOM-harvestable edge.

    The generator builds a return series with *volume-coupled persistence* — exactly
    the structure Ease of Movement is designed to detect:

        r_t = edge * sign(low_effort_drift_{t-1}) * |r_{t-1}| + eps_t

    where ``low_effort_drift`` is the prior bar's mid-price move scaled by inverse
    volume (the EOM kernel) and ``eps_t`` is i.i.d. normal with std ``daily_vol``.

    - ``edge = 0``  → a martingale; EOM carries no information (fair coin for any rule).
    - ``edge > 0``  → an "effortless" up (down) move on light volume tends to *continue*,
                      so EOM > 0 genuinely precedes positive returns — the planted edge.
    - ``edge < 0``  → low-effort moves reverse; EOM points the wrong way.

    Volume is deliberately *anti-correlated* with the absolute move (big moves on heavy
    volume, drifts on light volume) so the box-ratio denominator behaves like Arms
    intended. Bars are stamped on consecutive business days. Returns ``(data, truth)``
    where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    # Volume: a noisy base, later boosted on big-move bars (so big moves cost effort).
    base_vol = rng.lognormal(mean=16.0, sigma=0.45, size=n_days)

    prev_ret = 0.0
    prev_vol = base_vol[0]
    for i in range(n_days):
        # "low-effort drift": prior return divided by prior (relative) volume. A move on
        # light volume scores large; the same move on heavy volume scores small.
        vol_norm = prev_vol / base_vol.mean()
        low_effort = prev_ret / max(vol_norm, 1e-6)
        r = edge * low_effort + eps[i]
        log_ret[i] = r
        prev_ret = r
        # bigger absolute moves draw heavier volume (anti-effort coupling)
        prev_vol = base_vol[i] * (1.0 + 6.0 * abs(r))

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.6, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    # rebuild the realised heavy-on-big-move volume for the frame
    vol = base_vol * (1.0 + 6.0 * np.abs(log_ret))

    data = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"eom_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "20y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on a miss or ``fetch``.

    If a cached parquet exists it is read offline. Otherwise (or with ``fetch=True``) we
    pull from Yahoo, retrying a couple of times with a small backoff for rate limits, and
    cache the result. Yahoo's daily history goes back decades (~5,000 bars for a 20-year
    window), giving the timing race real statistical power.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        last_err = None
        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # noqa: BLE001 — retry any transient failure
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no daily bars for {ticker} ({last_err})"
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
