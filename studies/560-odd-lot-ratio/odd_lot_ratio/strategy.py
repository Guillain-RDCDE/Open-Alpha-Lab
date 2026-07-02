"""The engine and its honest controls — Study 560 (Odd-Lot-Ratio).

The claim, at full strength (Garfield A. Drew, *New Methods for Profit in the Stock Market*, 1950s;
the classic **odd-lot theory**): the small retail investor who trades in odd lots (< 100 shares) is
chronically on the *wrong* side — buying at tops, selling at bottoms. So a spike in odd-lot
**buying** (a high odd-lot ratio) is a contrarian *sell*, and a spike in odd-lot **selling** a
contrarian *buy*. **Fading** the odd-lot crowd should pay.

This module measures whether that fade pays, honestly, on a synthetic weekly tape:

1. **The contrarian signal.** Standardise the *prior*-week odd-lot ratio and take its *negative*:
   position = ``-z(odd_lot_ratio_{t-1})``. High odd-lot buying → short/underweight; heavy odd-lot
   selling → long/overweight. This is the textbook fade.

2. **The headline statistic.** Regress forward return on the standardised prior-week ratio across
   all weeks. The **fade pays** iff the slope is *negative* (more odd-lot buying → less return) at a
   robust *t*. Because the ratio is highly autocorrelated, we report a **Newey–West (HAC) t-stat**,
   not the naive OLS t — autocorrelation-robustness is the house inference bar.

3. **A regime sort.** Split weeks by prior-ratio tercile; compare next-week returns of the
   'euphoric' (high-ratio) tail vs the 'panic' (low-ratio) tail. The fade predicts panic > euphoria.

4. **A label-shuffle placebo null.** Shuffle the ratio labels against forward returns; the p-value
   is the tail probability of the observed |slope|. A real fade sits in the tail; noise does not.

5. **Costs / turnover honesty.** A weekly contrarian overlay rebalances often; charge a one-way cost
   per rebalance on the NAV traded, and — because the fade goes *short* the euphoric names — the
   short leg pays a borrow. Gross AND net are both reported.

6. **A multi-window robustness sweep** and a **seed-robust synthetic positive control** (≥ 20 seeds)
   proving the engine banks a planted fade edge and stays flat at the modern null.

Execution convention: the odd-lot ratio is observed at week *t*'s close (fully public); the position
is entered at that close and held over week *t+1* — one documented execution lag. No future data
enters the signal (the synthetic generator builds forward return from the *prior* week's ratio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52


# ---------------------------------------------------------------------------
# The contrarian signal
# ---------------------------------------------------------------------------
def contrarian_position(frame: pd.DataFrame) -> pd.Series:
    """The textbook fade: position = -z(prior-week odd-lot ratio).

    High odd-lot buying (high ratio) → negative (short/underweight); heavy odd-lot selling (low
    ratio) → positive (long/overweight). Uses only the prior week's ratio (no look-ahead).
    """
    return (-frame["z_ratio_prior"]).rename("position")


# ---------------------------------------------------------------------------
# The headline statistic — HAC slope of forward return on the ratio
# ---------------------------------------------------------------------------
def _newey_west_slope(x: np.ndarray, y: np.ndarray, lags: int = 4) -> dict:
    """OLS slope of y on x with a Newey–West (HAC) standard error.

    The odd-lot ratio is autocorrelated, so a naive OLS t overstates significance; the HAC t at a
    few weekly lags is the honest statistic. Returns slope, HAC t, and n.
    """
    n = len(x)
    if n < 8:
        return {"slope": float("nan"), "t": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    # Newey-West meat with Bartlett weights.
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
    slope = float(beta[1])
    return {"slope": slope, "t": slope / se_slope if se_slope > 0 else float("nan"), "n": n}


def fade_slope(frame: pd.DataFrame, lags: int = 4) -> dict:
    """HAC slope of forward return on the standardised prior-week odd-lot ratio.

    The **fade pays** iff this slope is *negative* at t ≤ −2 (more odd-lot buying → weaker next
    week). Returns slope (per z-unit), HAC t, correlation, and n.
    """
    d = frame.dropna(subset=["z_ratio_prior", "forward_ret"])
    x = d["z_ratio_prior"].to_numpy(dtype=float)
    y = d["forward_ret"].to_numpy(dtype=float)
    out = _newey_west_slope(x, y, lags=lags)
    out["corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# The regime sort — panic vs euphoria
# ---------------------------------------------------------------------------
def regime_sort(frame: pd.DataFrame, frac: float = 0.333) -> dict:
    """Sort weeks by prior-ratio tercile; compare next-week returns of panic vs euphoria tails.

    The fade predicts the **panic** tail (lowest odd-lot buying) beats the **euphoria** tail
    (highest odd-lot buying). Returns tail means (annualised %), the spread, a two-sample t, and n.
    """
    d = frame.dropna(subset=["z_ratio_prior", "forward_ret"]).sort_values("z_ratio_prior")
    k = max(2, int(round(len(d) * frac)))
    panic = d.head(k)["forward_ret"].to_numpy(dtype=float)     # lowest ratio = panic selling
    euph = d.tail(k)["forward_ret"].to_numpy(dtype=float)      # highest ratio = euphoric buying
    va, vb = panic.var(ddof=1), euph.var(ddof=1)
    se = np.sqrt(va / len(panic) + vb / len(euph))
    t = (panic.mean() - euph.mean()) / se if se > 0 else float("nan")
    return {
        "panic_ann": float(panic.mean() * TRADING_WEEKS),
        "euphoria_ann": float(euph.mean() * TRADING_WEEKS),
        "spread_ann": float((panic.mean() - euph.mean()) * TRADING_WEEKS),
        "t": float(t),
        "k": int(k),
        "n": int(len(d)),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(frame: pd.DataFrame, n_perm: int = 2000, seed: int = 560) -> float:
    """Label-shuffle placebo p-value for the fade slope.

    Shuffle the odd-lot-ratio labels against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |slope| ≥ the observed |slope|. A real fade sits in the tail.
    """
    d = frame.dropna(subset=["z_ratio_prior", "forward_ret"])
    if len(d) < 12:
        return float("nan")
    x = d["z_ratio_prior"].to_numpy(dtype=float)
    y = d["forward_ret"].to_numpy(dtype=float)
    obs = abs(_newey_west_slope(x, y)["slope"])
    rng = np.random.default_rng(seed)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        ys = y[rng.permutation(n)]
        s = abs(_newey_west_slope(x, ys)["slope"])
        if s >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The tradable overlay — gross AND net
# ---------------------------------------------------------------------------
def overlay_performance(
    frame: pd.DataFrame,
    cost_bps: float = 2.0,
    borrow_ann_bps: float = 50.0,
) -> dict:
    """A weekly contrarian overlay: hold ``position = -z(prior ratio)``, scaled to unit gross.

    Charge a one-way cost per rebalance on the NAV traded (|Δposition|), and a short borrow on the
    average short exposure (the fade is short the euphoric names). Returns gross and net annualised
    return and Sharpe, plus annual turnover.
    """
    d = frame.dropna(subset=["z_ratio_prior", "forward_ret"]).copy()
    pos = -d["z_ratio_prior"].to_numpy(dtype=float)
    # Scale to unit average gross exposure so costs/borrow are comparable across regimes.
    scale = 1.0 / (np.mean(np.abs(pos)) + 1e-12)
    pos = pos * scale
    r = d["forward_ret"].to_numpy(dtype=float)
    gross = pos * r
    # Turnover: |change in position| each week, one-way cost on it.
    dpos = np.abs(np.diff(pos, prepend=pos[0]))
    cost = dpos * (cost_bps * 1e-4)
    # Borrow on the short leg (negative positions), pro-rated weekly.
    short_exposure = np.clip(-pos, 0.0, None)
    borrow = short_exposure * (borrow_ann_bps * 1e-4) / TRADING_WEEKS
    net = gross - cost - borrow

    def ann_ret(v):
        return float(np.mean(v) * TRADING_WEEKS)

    def sharpe(v):
        sd = np.std(v, ddof=1)
        return float(np.mean(v) / sd * np.sqrt(TRADING_WEEKS)) if sd > 0 else float("nan")

    return {
        "gross_ann": ann_ret(gross),
        "net_ann": ann_ret(net),
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "ann_turnover": float(np.mean(dpos) * TRADING_WEEKS),
    }


# ---------------------------------------------------------------------------
# Robustness — multi-window sweep
# ---------------------------------------------------------------------------
def window_sweep(frame: pd.DataFrame, n_windows: int = 4) -> list[dict]:
    """Split the tape into ``n_windows`` equal spans; report the fade slope & HAC t in each.

    A signal whose *sign* depends on the window is not bankable. Returns one dict per window.
    """
    d = frame.dropna(subset=["z_ratio_prior", "forward_ret"])
    bounds = np.linspace(0, len(d), n_windows + 1).astype(int)
    rows = []
    for i in range(n_windows):
        seg = d.iloc[bounds[i]:bounds[i + 1]]
        r = fade_slope(seg)
        lab = f"{seg.index[0].date()} -> {seg.index[-1].date()}" if len(seg) else "empty"
        rows.append({"window": lab, "slope": r["slope"], "t": r["t"], "n": r["n"]})
    return rows


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, fade_alpha: float, n_seeds: int = 25, base_seed: int = 560) -> float:
    """Average the fade-slope HAC t over ``n_seeds`` synthetic worlds for a planted ``fade_alpha``.

    House rule: any synthetic-dependent claim averages a robust stat over ≥ 20 seeds so no single
    lucky RNG seed can manufacture significance. A genuine fade edge (``fade_alpha > 0``) should
    drive the slope *negative* and past −2; the null (``fade_alpha = 0``) should sit near 0.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(fade_alpha=fade_alpha, seed=s)
        ts.append(fade_slope(frame)["t"])
    return float(np.nanmean(ts))
