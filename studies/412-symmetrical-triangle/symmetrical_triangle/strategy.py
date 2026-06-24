"""Detector + inference for Study 412 — Symmetrical Triangle.

The claim (technician folklore): a **symmetrical triangle** — a series of *lower highs*
and *higher lows* that converge toward an apex — is a *continuation* figure. When price
finally breaks out of the converging range, it keeps going **in the breakout direction**,
and the move "runs about the height of the triangle." Trade the confirmed breakout.

Chart figures are partly subjective; we test the **closest mechanical definition** and
say so. The objective detector (:func:`detect_triangles`):

  1. On a trailing ``lookback`` window, find swing pivots: local highs and local lows
     (``scipy.signal.find_peaks`` with a min separation ``min_bars`` and a prominence
     floor tied to ATR).
  2. Fit a least-squares line through the last ``>=2`` swing **highs** (the upper
     trendline) and through the last ``>=2`` swing **lows** (the lower trendline).
  3. Call it a **symmetrical triangle** when the upper line slopes **down**, the lower
     line slopes **up** (they converge), the slopes are roughly symmetric in magnitude,
     and the range has genuinely **contracted** (recent range < ``contract`` x early
     range) — i.e. the two lines are marching toward an apex.
  4. **Confirmed breakout** at bar *t* when the close pierces the projected upper line
     (an **up** breakout, claimed direction +1) or the projected lower line (a **down**
     breakout, −1). One signal per figure; a fresh figure must re-form before the next.

The honest arbiters mirror the desk's pattern-study idiom (see study 189):

  * **Forward returns** at H ∈ {1, 5, 20, 60} days, entered **one day after** the
    confirmed breakout (no look-ahead), signed by the breakout direction.
  * **Random-day placebo** — for each signal draw a random eligible bar from the same
    pooled tape, assign the same direction, measure the same forward return. The figure
    adds value only if its mean **exceeds** this unconditional base rate.
  * **HAC (Newey-West) t** on the *excess* (signal − random) — the inference-bar number.
  * **A label-shuffle permutation** placebo on the pooled signed returns.
  * **Costs** one-way x NAV per round trip; the direction split (up vs down breakout)
    as a myth check on "continuation."

No look-ahead: the figure + breakout are read on the close of *t*; entry is at *t+1*'s
open; forward returns start there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HORIZONS = (1, 5, 20, 60)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _atr(bars: pd.DataFrame, n: int = 20) -> np.ndarray:
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - prev).abs(),
                    (bars["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().to_numpy(dtype=float)


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope, intercept through points (x, y)."""
    if len(x) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def detect_triangles(
    bars: pd.DataFrame,
    lookback: int = 90,
    min_bars: int = 6,
    atr_prominence_mult: float = 0.5,
    slope_symmetry: float = 3.0,
    contract: float = 0.7,
    min_pivots: int = 2,
) -> pd.DataFrame:
    """Detect confirmed symmetrical-triangle breakouts on daily bars.

    For each bar *t* (after a warm-up) we scan the trailing ``lookback`` window for
    swing highs/lows, fit the upper/lower trendlines, and require:

    * upper slope **< 0** and lower slope **> 0** (converging);
    * slope magnitudes roughly symmetric: ``1/slope_symmetry <= |up|/|dn| <= slope_symmetry``;
    * the range has **contracted**: the high-minus-low over the last third of the window
      is below ``contract`` x the range over the first third (a real squeeze).

    A breakout is **confirmed** at the close of *t* when it pierces the *projected*
    upper line (``dir = +1``) or lower line (``dir = -1``). One signal per figure;
    after a signal we wait ``lookback // 2`` bars before allowing the next.

    Returns a DataFrame indexed by date with boolean ``signal`` and integer ``dir``
    (+1 up-breakout / -1 down-breakout / 0 none). ``True`` at *t* means confirmed at the
    close of *t*; forward returns begin at *t+1*.
    """
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    atr = _atr(bars)
    n = len(close)
    sig = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=int)
    cooldown = lookback // 2
    last = -cooldown

    for t in range(lookback, n):
        if t - last < cooldown:
            continue
        a = atr[t] if np.isfinite(atr[t]) else 0.0
        if a <= 0:
            continue
        prom = a * atr_prominence_mult
        ws, we = t - lookback, t  # window [ws, t)
        hwin = high[ws:we]
        lwin = low[ws:we]

        ph, _ = find_peaks(hwin, distance=min_bars, prominence=prom)
        pl, _ = find_peaks(-lwin, distance=min_bars, prominence=prom)
        if len(ph) < min_pivots or len(pl) < min_pivots:
            continue

        # fit the two trendlines through the swing pivots (x = position in window)
        up_slope, up_int = _line(ph.astype(float), hwin[ph])
        dn_slope, dn_int = _line(pl.astype(float), lwin[pl])

        # converging: highs slope down, lows slope up
        if not (up_slope < 0 and dn_slope > 0):
            continue
        # roughly symmetric slope magnitudes
        ratio = abs(up_slope) / (abs(dn_slope) + 1e-12)
        if not (1.0 / slope_symmetry <= ratio <= slope_symmetry):
            continue
        # genuine contraction: late range < contract * early range
        third = lookback // 3
        early_rng = float(np.max(hwin[:third]) - np.min(lwin[:third]))
        late_rng = float(np.max(hwin[-third:]) - np.min(lwin[-third:]))
        if early_rng <= 0 or late_rng >= contract * early_rng:
            continue

        # projected trendlines at the current bar position (x = lookback-1, end of window)
        x_now = float(lookback - 1)
        upper_proj = up_slope * x_now + up_int
        lower_proj = dn_slope * x_now + dn_int

        c = close[t]
        if c > upper_proj:
            sig[t] = True
            direction[t] = +1
            last = t
        elif c < lower_proj:
            sig[t] = True
            direction[t] = -1
            last = t

    return pd.DataFrame({"signal": sig, "dir": direction}, index=bars.index)


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Simple close-to-close forward returns at each horizon (entered t+1 in practice)."""
    close = bars["close"]
    return pd.DataFrame({f"fwd_{h}": close.shift(-h) / close - 1.0 for h in horizons},
                        index=bars.index)


# --------------------------------------------------------------------------- #
# Pooled study — signal vs random-day placebo across the basket
# --------------------------------------------------------------------------- #
def run_pooled(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
               seed: int = 412, lag: int = 1, **detect_kwargs) -> pd.DataFrame:
    """Pool confirmed-breakout events across the basket into one signed-return ledger.

    For every name we detect confirmed triangles, take the forward H-day return entered
    ``lag`` days after the confirmation close (no look-ahead), sign it by the breakout
    direction, and draw a matched random-day baseline from the same name's eligible bars
    (same direction). Returns one row per event with ``ret_{h}d`` (signal, signed) and
    ``base_{h}d`` (random-day, signed), plus ``dir``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    maxh = max(horizons)
    for tk, bars in panel.items():
        if len(bars) < 200:
            continue
        det = detect_triangles(bars, **detect_kwargs)
        fwd = forward_returns(bars, horizons)
        idx = bars.index
        pos = {d: i for i, d in enumerate(idx)}
        n = len(bars)
        eligible = np.arange(maxh + lag, n - maxh - lag)
        if len(eligible) == 0:
            continue
        sig_dates = idx[det["signal"].to_numpy()]
        for d in sig_dates:
            i = pos[d]
            entry = i + lag
            if entry + maxh >= n:
                continue
            dirn = int(det["dir"].iat[i])
            row = {"ticker": tk, "entry_date": idx[entry], "dir": dirn}
            # random matched baseline
            ri = int(rng.choice(eligible))
            for h in horizons:
                rsig = dirn * (bars["close"].iat[entry + h] / bars["close"].iat[entry] - 1.0)
                rbase = dirn * (bars["close"].iat[ri + h] / bars["close"].iat[ri] - 1.0)
                row[f"ret_{h}d"] = float(rsig)
                row[f"base_{h}d"] = float(rbase)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of mean(x) vs 0 (auto lag ~ 4*(n/100)^(2/9))."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def permutation_p(signal: np.ndarray, base: np.ndarray,
                  n_draws: int = 20_000, seed: int = 412) -> float:
    """Permutation placebo: pool signal & base, reshuffle the 'is-signal' labels, and
    ask how often the shuffled signal-minus-base excess beats the observed one.

    ``p = P[shuffled excess >= observed excess]`` — the honest small-effect null.
    """
    s = np.asarray(signal, float); b = np.asarray(base, float)
    m = np.isfinite(s) & np.isfinite(b)
    s, b = s[m], b[m]
    obs = s.mean() - b.mean()
    pool = np.concatenate([s, b])
    k = len(s)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_draws):
        perm = rng.permutation(pool)
        if (perm[:k].mean() - perm[k:].mean()) >= obs:
            cnt += 1
    return float(cnt / n_draws)


def summarize(ledger: pd.DataFrame, horizon: int = 20, cost_bps: float = 2.0,
              placebo: bool = True, n_draws: int = 20_000) -> dict:
    """Headline stats for one horizon: counts, win-rate, mean signal/base, excess,
    HAC t on the excess, permutation p, and net-of-cost mean.

    ``cost_bps`` is one-way x NAV; a confirmed-breakout trade is a round trip (in + out),
    so we charge ``2 * cost_bps`` against the signed return.
    """
    sc, bc = f"ret_{horizon}d", f"base_{horizon}d"
    if sc not in ledger.columns or len(ledger) == 0:
        return {"horizon": horizon, "n": 0}
    s = ledger[sc].to_numpy(float)
    b = ledger[bc].to_numpy(float)
    m = np.isfinite(s) & np.isfinite(b)
    s, b = s[m], b[m]
    excess = s - b
    rt = 2.0 * cost_bps / 1e4
    p = permutation_p(s, b, n_draws=n_draws) if placebo else float("nan")
    return {
        "horizon": horizon, "n": int(len(s)),
        "win_rate": float((s > 0).mean()),
        "mean_sig": float(s.mean()), "mean_base": float(b.mean()),
        "excess": float(excess.mean()),
        "t_excess": hac_t(excess), "t_sig": hac_t(s),
        "p_perm": p, "net": float(s.mean() - rt),
        "cost_bps": cost_bps,
    }


def direction_split(ledger: pd.DataFrame, horizon: int = 20) -> dict:
    """Up-breakout vs down-breakout means + win-rates (the 'continuation' myth check)."""
    sc = f"ret_{horizon}d"
    out = {}
    for name, d in (("up", 1), ("down", -1)):
        sub = ledger[ledger["dir"] == d]
        s = sub[sc].to_numpy(float)
        s = s[np.isfinite(s)]
        out[name] = {
            "n": int(len(s)),
            "mean": float(s.mean()) if len(s) else float("nan"),
            "win": float((s > 0).mean()) if len(s) else float("nan"),
            "t": hac_t(s),
        }
    return out


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   cost_bps: float = 2.0, n_draws: int = 20_000,
                   **detect_kwargs) -> dict:
    """Orchestrator: build the pooled ledger, summarise every horizon, add the split."""
    ledger = run_pooled(panel, horizons=horizons, **detect_kwargs)
    per_h = {h: summarize(ledger, h, cost_bps=cost_bps, n_draws=n_draws) for h in horizons}
    split = {h: direction_split(ledger, h) for h in horizons}
    return {"ledger": ledger, "per_horizon": per_h, "split": split,
            "n_events": int(len(ledger))}
