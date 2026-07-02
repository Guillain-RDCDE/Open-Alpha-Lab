"""Data layer for Study 589 (Genetic-Algo-Overfit).

This is a **research-method demo**, so the tape is *deliberately* synthetic: to prove that a
genetic algorithm manufactures a gorgeous backtest out of nothing, you have to run it on a world
where you *know* the ground truth. Real free data can never give you that certainty — you can't
prove a real market has *zero* timing edge — so the honest thing is to build the world yourself
and state the data-availability limitation openly on the SIGNAL axis. (A synthetic-only study can
never earn `REAL`; that needs a robust ``t >= 2`` on a real tape. This one is capped at `NONE`.)

One generator, one knob:

- ``synthetic_series(signal_strength=0.0, ...)`` — a daily price tape plus a panel of technical
  **features** (past returns over several lookbacks, a moving-average gap, an RSI-like oscillator,
  a volatility ratio). With ``signal_strength = 0`` the next-day return is pure noise, unrelated to
  every feature: a **random walk**, the *null*, where by construction no rule can work. With
  ``signal_strength > 0`` one hidden linear combination of the features genuinely predicts the next
  return: the *positive control*, where a real edge exists for the GA to find and carry OOS.

The whole point of the study: on the **null**, the GA still finds a rule with a beautiful
in-sample Sharpe — and it dies out-of-sample. On the **control**, the same GA finds a rule that
*keeps working* OOS. Same optimiser, opposite fates — that is how you tell searching from having
an edge.

Everything is deterministic (seeded on the study number, 589) and fully offline. There is no real
fetch, by design.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

STUDY_SEED = 589
TRADING_DAYS = 252

# The feature menu the GA gets to evolve rules over. These are ordinary, publicly computable
# technical features — the kind a naive "let the machine find a strategy" search would hand it.
FEATURE_NAMES = [
    "mom_1",    # yesterday's return
    "mom_5",    # trailing 5-day return
    "mom_21",   # trailing 21-day return
    "mom_63",   # trailing 63-day return
    "ma_gap",   # (price - 50d MA) / 50d MA
    "rsi",      # RSI-like oscillator, centred at 0
    "vol_ratio",  # short vol / long vol - 1
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class WorldTruth:
    """Ground truth for the synthetic tape."""

    signal_strength: float  # 0 = random walk (null); > 0 = a genuinely predictable tape

    @property
    def has_edge(self) -> bool:
        return self.signal_strength > 0.0


def _rsi_like(ret: np.ndarray, window: int = 14) -> np.ndarray:
    """A cheap, vectorised RSI-style oscillator centred at 0 in [-1, 1].

    Rolling (mean gain - mean loss) / (mean gain + mean loss) over ``window`` days.
    """
    n = len(ret)
    gain = np.where(ret > 0, ret, 0.0)
    loss = np.where(ret < 0, -ret, 0.0)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        g = gain[lo : i + 1].mean()
        l = loss[lo : i + 1].mean()
        denom = g + l
        out[i] = (g - l) / denom if denom > 1e-12 else 0.0
    return out


def _build_features(price: np.ndarray) -> pd.DataFrame:
    """Compute the feature panel from a price path (each feature standardised, no look-ahead).

    Every feature at row ``t`` uses only prices up to and including ``t`` — it is public at the
    close of day ``t`` and is used to predict the return of ``t+1`` (the one execution lag lives in
    ``strategy.py``, applied once).
    """
    price = np.asarray(price, dtype=float)
    ret = np.zeros_like(price)
    ret[1:] = price[1:] / price[:-1] - 1.0

    def trailing_ret(k):
        out = np.zeros_like(price)
        out[k:] = price[k:] / price[:-k] - 1.0
        return out

    ma50 = pd.Series(price).rolling(50, min_periods=1).mean().to_numpy()
    ma_gap = price / ma50 - 1.0

    short_vol = pd.Series(ret).rolling(10, min_periods=2).std(ddof=1).fillna(0.0).to_numpy()
    long_vol = pd.Series(ret).rolling(60, min_periods=2).std(ddof=1).fillna(0.0).to_numpy()
    safe_long = np.where(long_vol > 1e-9, long_vol, 1.0)
    vol_ratio = np.where(long_vol > 1e-9, short_vol / safe_long - 1.0, 0.0)

    raw = {
        "mom_1": ret,
        "mom_5": trailing_ret(5),
        "mom_21": trailing_ret(21),
        "mom_63": trailing_ret(63),
        "ma_gap": ma_gap,
        "rsi": _rsi_like(ret, 14),
        "vol_ratio": vol_ratio,
    }
    df = pd.DataFrame(raw)
    # Standardise each feature (z-score) so the GA's linear weights are on a common scale.
    for c in df.columns:
        s = df[c]
        sd = s.std(ddof=0)
        df[c] = (s - s.mean()) / sd if sd > 1e-12 else 0.0
    return df


def synthetic_series(
    n_days: int = 3000,
    signal_strength: float = 0.0,
    daily_vol: float = 0.011,
    drift: float = 0.0,
    seed: int = STUDY_SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A deterministic daily tape + feature panel with a tunable *true* predictability.

    Returns ``(frame, truth)`` where ``frame`` has columns:

    - ``price`` — the level, ``ret`` — the daily simple return,
    - the seven standardised features in ``FEATURE_NAMES``,
    - ``fwd_ret`` — the *next-day* return (the target the GA is implicitly trying to time), shifted
      so ``fwd_ret[t]`` is the return earned by a position taken at the close of ``t``.

    Construction:

    - Draw an i.i.d. Gaussian innovation each day.
    - With ``signal_strength > 0``, add a small *predictable* component to the next return that is a
      fixed hidden linear combination of the (lagged) features — a genuine, harvestable edge. With
      ``signal_strength = 0`` there is **no** predictable component: the tape is a pure random walk
      and every feature is, in truth, useless.

    The hidden combination is fixed by ``seed`` so the control is reproducible; the GA never sees it
    and must *discover* it (or, on the null, hallucinate one).
    """
    rng = np.random.default_rng(seed)

    # Hidden "true" predictive weights over the features (used only when signal_strength > 0).
    true_w = rng.standard_normal(N_FEATURES)
    true_w /= np.linalg.norm(true_w)

    innov = rng.standard_normal(n_days) * daily_vol

    price = np.empty(n_days)
    ret = np.empty(n_days)
    price[0] = 100.0
    ret[0] = 0.0

    # We build price and features jointly: the predictable part of tomorrow's return depends on
    # today's features, which depend on the price path so far — so we roll forward day by day, but
    # recompute features on the running path. To keep it O(n) and deterministic we bootstrap with a
    # first pass to get a stable feature scale, then inject the signal.
    #
    # Pass 1: a plain random walk to define the feature panel scale.
    for t in range(1, n_days):
        price[t] = price[t - 1] * (1.0 + drift + innov[t])
        ret[t] = price[t] / price[t - 1] - 1.0
    feats = _build_features(price)

    # Pass 2: if there is an edge, rebuild the path so that the *predictable* component of the next
    # return is signal_strength * (features . true_w). We reuse the same innovations for
    # determinism; the predictable part is added on top and is genuinely forecastable from the
    # public features of the prior day.
    if signal_strength > 0.0:
        fmat = feats[FEATURE_NAMES].to_numpy()
        pred = fmat @ true_w                      # predictable score per day (using day-t features)
        pred = pred / (pred.std(ddof=0) + 1e-12)  # unit scale
        for t in range(1, n_days):
            extra = signal_strength * daily_vol * pred[t - 1]  # day t-1 features predict day-t ret
            price[t] = price[t - 1] * (1.0 + drift + innov[t] + extra)
            ret[t] = price[t] / price[t - 1] - 1.0
        feats = _build_features(price)            # refresh features on the final path

    df = feats.copy()
    df.insert(0, "price", price)
    df.insert(1, "ret", ret)
    df["fwd_ret"] = np.append(ret[1:], np.nan)    # fwd_ret[t] = ret[t+1] (position held from t)
    df = df.iloc[:-1].reset_index(drop=True)      # drop the last row (no forward return)
    return df, WorldTruth(signal_strength=signal_strength)


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
