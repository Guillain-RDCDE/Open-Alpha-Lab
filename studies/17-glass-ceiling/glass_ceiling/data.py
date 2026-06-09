"""The tape the breakout runs on — a synthetic intraday OHLCV generator with a *baked-in* answer,
plus a cache-only reader for real intraday bars.

The whole study turns on one quantity: **after price breaks a prior high, does it keep going?**
So the synthetic generator's load-bearing knob is ``cont_drift`` — a small extra drift injected
into the bars *right after a fresh-high breakout*:

    * ``cont_drift == 0``  → the **null**: a pure random walk. A fresh high carries no information;
      the post-breakout path is a martingale. A symmetric ±1R bracket placed on it is a coin flip
      by construction (gambler's-ruin on a driftless walk), so any apparent edge a backtest reports
      here is selection or luck — the baseline every real result is measured against.
    * ``cont_drift > 0``   → **genuine continuation** (momentum): breakouts really do follow
      through, so the bracket's win rate must rise above 0.5. This is the *steelman* tape — proof
      the method has power and is not rigged to always cry "mirage".
    * ``cont_drift < 0``   → **exhaustion** (the "buy-the-high" trap): breakouts fade, win rate
      drops below 0.5.

One more knob makes the *filters* testable on their own terms. With ``cont_requires_grind=True``
the continuation drift is injected **only** when the approach into the high was a low-volatility
*grind* (Koroush AK's first filter), and never after a vertical *spike*. On that tape a staircase
filter has real predictive content to recover; on the default tape (drift after *every* breakout)
it has nothing to add beyond the unconditional edge — which is the honest way to ask "do the
filters earn their keep, or do they just shrink the sample?".

Data choices, named up front per the house rule. Bars are **arithmetic** (close-to-close simple
returns, not log) because intraday moves are tiny and the bracket is defined in price distance, so
arithmetic keeps the ±1R barriers exactly symmetric and the null exactly a coin flip. Volume is
generated **correlated with bar range** (the real stylized fact: big bars trade more), so the
volume filter reads a realistic channel rather than pure noise. Wicks are drawn so each bar has a
plausible High/Low *around* its open→close body — load-bearing, because the bracket can be touched
intrabar, and a body-only tape would never hit a stop inside a bar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

# A liquid US-session minute count is ~390/day; we index synthetic bars on a plain RangeIndex of
# "bars" and don't pretend a calendar — the study's unit is the *trade*, not the clock.
BARS_PER_DAY = 390


# --------------------------------------------------------------------------- #
# Ground truth — what the generator baked in, so a test can check the backtest
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """The truth the simulator must recover from OHLCV alone."""
    cont_drift: float          # extra per-bar drift injected after a fresh-high breakout
    cont_window: int           # number of bars the extra drift persists after a breakout
    sigma: float               # per-bar return volatility of the base walk
    lookback: int              # bars defining "a prior high" for the generator's breakout events
    requires_grind: bool       # was continuation conditioned on a low-vol approach?
    n_bars: int

    @property
    def is_null(self) -> bool:
        """True when no breakout carries information — the coin-flip tape."""
        return self.cont_drift == 0.0

    @property
    def edge_sign(self) -> int:
        """Sign the post-breakout drift should imprint on a long bracket's win rate vs 0.5."""
        return int(np.sign(self.cont_drift))

    @property
    def fair_win_rate(self) -> float:
        """The win rate a symmetric ±1R bracket earns on the *null* tape: a coin flip."""
        return 0.5


def _grind(close: np.ndarray) -> float:
    """Staircase-vs-spike score in [0,1] of a close path: ``1 − biggest_up_step / total_up``.

    Mirrors :func:`glass_ceiling.filters.grind_score` exactly (kept local so the data layer doesn't
    depend on the strategy layer), so the continuation the generator can gate on is the *same*
    structure the filter later measures. Near 1 = gains spread over many bars (a grind); near 0 = one
    big bar did the work (a spike).
    """
    steps = np.diff(close)
    up = steps[steps > 0].sum()
    if up <= 0:
        return 0.0
    return float(np.clip(1.0 - steps.max() / up, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Synthetic tape
# --------------------------------------------------------------------------- #

def synthetic_intraday(
    n_bars: int = 30_000,
    sigma: float = 0.0010,
    base_drift: float = 0.0,
    cont_drift: float = 0.0,
    cont_window: int = 20,
    lookback: int = 30,
    cont_requires_grind: bool = False,
    grind_vol_quantile: float = 0.5,
    wick_frac: float = 0.5,
    vol_base: float = 1_000.0,
    vol_range_k: float = 40.0,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A toy minute tape whose post-breakout behaviour is *known*.

    Returns ``(bars, truth)``. ``bars`` is an OHLCV ``DataFrame`` on a ``RangeIndex`` named ``bar``
    with float columns ``Open, High, Low, Close, Volume`` (Title-case to match the daily layer).

    The return process is built in two passes so a breakout can feed drift forward without
    look-ahead in the *price* (the drift is applied to bars strictly *after* the breakout bar):

    1. Walk close-to-close: ``r[t] = base_drift + extra[t] + sigma * eps[t]``. ``extra[t]`` is
       ``cont_drift`` while ``t`` is inside the ``cont_window`` bars following a fresh-high
       breakout, else 0. A *breakout* is ``Close[t]`` exceeding the max ``Close`` over the prior
       ``lookback`` bars. The extra drift is decided bar-by-bar from history only.
    2. Dress each close into an OHLC bar: ``Open`` = prior close; the body spans open→close; upper
       and lower wicks are half-normal draws scaled by ``sigma`` (so High/Low straddle the body),
       which is what lets the bracket be touched *inside* a bar.

    ``cont_requires_grind`` gates the continuation drift on a calm approach: the extra drift fires
    only if realized vol over the prior ``lookback`` bars sits below its rolling ``grind_vol_quantile``
    — i.e. the high was reached by a slow staircase, not a spike. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_bars)

    close = np.empty(n_bars)
    close[0] = start_price
    logret = np.empty(n_bars)       # realized simple returns, for vol/grind bookkeeping
    logret[0] = 0.0

    cont_left = 0                   # bars of continuation drift still owed
    recent_grind: list[float] = []  # rolling buffer of approach grind scores, to set the gate

    for t in range(1, n_bars):
        extra = cont_drift if cont_left > 0 else 0.0
        r = base_drift + extra + sigma * eps[t]
        logret[t] = r
        close[t] = close[t - 1] * (1.0 + r)
        if cont_left > 0:
            cont_left -= 1

        # Did THIS bar break the prior `lookback` highs? (history only — no look-ahead)
        lo = max(0, t - lookback)
        if t > lookback and close[t] > close[lo:t].max():
            fire = True
            if cont_requires_grind:
                # grind = the approach is NOT dominated by one big bar — the *same* concentration
                # metric the staircase filter reads (filters.grind_score), so the filter has a real,
                # recoverable signal. The drift fires only for the grindier half of breakouts.
                g = _grind(close[lo: t + 1])
                recent_grind.append(g)
                if len(recent_grind) > 500:
                    recent_grind.pop(0)
                gate = float(np.quantile(recent_grind, grind_vol_quantile)) if len(recent_grind) > 20 else g
                fire = g >= gate
            if fire and cont_drift != 0.0:
                cont_left = cont_window

    # Dress closes into OHLC bars with wicks.
    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    up_wick = np.abs(rng.standard_normal(n_bars)) * wick_frac * sigma * close
    dn_wick = np.abs(rng.standard_normal(n_bars)) * wick_frac * sigma * close
    high = body_hi + up_wick
    low = body_lo - dn_wick

    # Volume rises with bar range (big bars trade more) plus lognormal noise.
    rng_frac = (high - low) / close
    volume = vol_base * (1.0 + vol_range_k * rng_frac) * np.exp(0.3 * rng.standard_normal(n_bars))

    bars = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.RangeIndex(n_bars, name="bar"),
    )
    truth = GroundTruth(
        cont_drift=cont_drift, cont_window=cont_window, sigma=sigma,
        lookback=lookback, requires_grind=cont_requires_grind, n_bars=n_bars,
    )
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — cache-only intraday bars
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, interval: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
    return os.path.join(cache_dir, f"bars_{safe}_{interval}.parquet")


def fetch_bars(
    ticker: str,
    interval: str = "5m",
    period: str = "60d",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> pd.DataFrame:
    """Return (and cache) real intraday OHLCV for ``ticker`` — **cache-only by default**.

    If a parquet exists it is returned; otherwise an empty frame is returned *unless* ``fetch=True``,
    in which case Yahoo! is hit once and the result cached. The network import is lazy so the offline
    core (and CI) never imports ``yfinance``.

    **A named limitation, not a detail:** Yahoo only serves intraday history in a short trailing
    window — roughly the last 7 days at 1-minute and ~60 days at 5-minute — so the real leg of this
    study is a *small-sample sanity check*, never the load-bearing evidence. The synthetic core,
    where the answer is baked in, carries the verdict; the real bars only confirm the coin flip is
    visible in the wild. A deeper history would need a paid intraday feed or an MT5 export
    (``quantlab/brokers/mt5_connector.py``).
    """
    path = _cache_path(ticker, interval, cache_dir)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)

    if not fetch:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    import yfinance as yf  # lazy: offline core never imports it

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw[["Open", "High", "Low", "Close", "Volume"]].astype("float64").dropna()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out.index.name = "timestamp"

    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(path)
    return out
