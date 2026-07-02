"""The engine and its honest controls — Study 555 (OpenTable-Reservations).

The claim, at full strength: a dining-reservation index (OpenTable-style *seated-diners* YoY) is an
alt-data **nowcast** — this week's reservations *surprise* predicts next week's return of a
restaurant basket (and, more loosely, consumer-discretionary / XLY). The tradable expression is a
timing overlay: go long the basket when the reservations surprise is positive, flat/short when it
is negative, and see whether that beats buy-and-hold after costs.

This module measures that, honestly:

1. **The signal.** ``resv_surprise`` at week *t* — reservations YoY with its own short trend and
   the contemporaneous market removed (built in ``data.py``), so we are not just re-discovering
   market beta. Standardised across the sample.

2. **The predictive regression.** OLS of next-week basket return on this week's surprise,
   *controlling for the contemporaneous market factor* (so the nowcast must add information beyond
   "the market moved"). The slope's Newey-West (HAC) *t* is the headline statistic. A one-week
   execution lag is baked in: signal at close of *t*, return over *t+1*.

3. **The placebo / label-shuffle null.** Shuffle the surprise against the forward returns; a real
   nowcast sits in the tail, noise does not.

4. **The timing overlay + costs.** A long/flat (or long/short) overlay driven by the sign of the
   surprise, with a one-way cost charged on every position change (turnover × cost). Gross AND net,
   both labelled. When shorting is allowed the short leg pays a borrow.

5. **A multi-window robustness sweep** — split the sample into sub-periods and re-estimate the
   slope-*t*; a real nowcast keeps its sign.

6. **A synthetic positive control** — a deterministic world where the nowcast is planted
   (``nowcast_beta > 0``); the engine must recover a positive HAC-*t* and stay flat at the null,
   averaged over ≥ 20 seeds so no single lucky seed manufactures significance.

There is **no real tape** (see ``data.py``): the free seated-diners history does not exist, so this
study proves the *machinery* on synthetic data and is capped below `REAL` on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aligning signal (week t) with forward return (week t+1) — the one execution lag
# ---------------------------------------------------------------------------
def aligned_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of (signal_t, mkt_{t+1}, fwd_ret_{t+1}) with the one-week lag applied.

    ``signal`` is this week's reservations surprise; ``fwd_ret`` is *next* week's basket return;
    ``mkt_fwd`` is *next* week's market factor (the contemporaneous control for the forward return).
    Rows with any NaN (the first year of YoY, the last forward week) are dropped.
    """
    if panel.empty:
        return pd.DataFrame(columns=["signal", "mkt_fwd", "fwd_ret"])
    sig = panel["resv_surprise"]
    fwd = panel["basket_ret"].shift(-1)
    mkt_fwd = panel["mkt_ret"].shift(-1)
    out = pd.DataFrame({"signal": sig, "mkt_fwd": mkt_fwd, "fwd_ret": fwd}).dropna()
    return out


# ---------------------------------------------------------------------------
# Newey-West (HAC) OLS
# ---------------------------------------------------------------------------
def _nw_ols(y: np.ndarray, X: np.ndarray, lags: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Newey-West HAC standard errors. Returns (coef, se) arrays."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ (X.T @ y)
    resid = y - X @ coef
    # HAC meat: S = sum_t x_t x_t' e_t^2 + weighted cross-lags
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        g = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    return coef, se


def predictive_regression(panel: pd.DataFrame, hac_lags: int = 4) -> dict:
    """OLS of next-week basket return on this week's surprise, controlling for next-week market.

    Model: ``fwd_ret = a + b1*signal + b2*mkt_fwd + e``. The nowcast coefficient is ``b1`` — the
    part of the forward return explained by the reservations surprise *beyond* the market. Returns
    the slope ``b1``, its Newey-West HAC *t*, R², and n.
    """
    af = aligned_frame(panel)
    if len(af) < 12:
        return {"slope": float("nan"), "slope_t": float("nan"), "r2": float("nan"), "n": len(af)}
    y = af["fwd_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(af)), af["signal"].to_numpy(float), af["mkt_fwd"].to_numpy(float)])
    coef, se = _nw_ols(y, X, lags=hac_lags)
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se[1]) if se[1] > 0 else float("nan"),
        "r2": r2,
        "n": int(len(af)),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 555) -> float:
    """Label-shuffle placebo p-value for the nowcast slope *t*.

    Shuffle the surprise against the forward returns ``n_perm`` times; the p-value is the fraction
    of shuffles whose |slope-t| ≥ the observed |slope-t|. A real nowcast sits in the tail.
    """
    af = aligned_frame(panel)
    if len(af) < 12:
        return float("nan")
    obs = abs(predictive_regression(panel)["slope_t"])
    y = af["fwd_ret"].to_numpy(dtype=float)
    mkt = af["mkt_fwd"].to_numpy(dtype=float)
    sig = af["signal"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(af)
    count = 0
    for _ in range(n_perm):
        s = sig[rng.permutation(n)]
        X = np.column_stack([np.ones(n), s, mkt])
        coef, se = _nw_ols(y, X, lags=4)
        t = abs(coef[1] / se[1]) if se[1] > 0 else 0.0
        if t >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The timing overlay + costs
# ---------------------------------------------------------------------------
def overlay_backtest(
    panel: pd.DataFrame,
    allow_short: bool = False,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 100.0,
) -> dict:
    """A sign-of-surprise timing overlay vs buy-and-hold the basket. Gross AND net.

    Position at week *t* is +1 if the surprise is positive, else 0 (long/flat) or −1 (if
    ``allow_short``); it earns week *t+1*'s basket return. A one-way ``cost_bps`` is charged on the
    change in position each week (turnover). If shorting, the short weeks pay a pro-rated annual
    ``borrow_ann_bps``. Returns overlay gross/net mean weekly return + annualised, buy-and-hold
    annualised, the information ratio of net excess over buy-and-hold, turnover, and n.
    """
    af = aligned_frame(panel)
    if len(af) < 12:
        return {}
    sig = af["signal"].to_numpy(dtype=float)
    fwd = af["fwd_ret"].to_numpy(dtype=float)
    pos = np.where(sig > 0, 1.0, (-1.0 if allow_short else 0.0))
    gross = pos * fwd
    # turnover cost: |pos_t - pos_{t-1}| * cost, first trade from flat
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * cost_bps * 1e-4
    borrow = np.where(pos < 0, borrow_ann_bps * 1e-4 / 52.0, 0.0)
    net = gross - cost - borrow
    bh = fwd  # buy-and-hold the basket
    ann = 52.0
    net_excess = net - bh
    ir = (net_excess.mean() / net_excess.std(ddof=1) * np.sqrt(ann)) if net_excess.std(ddof=1) > 0 else float("nan")
    return {
        "overlay_gross_ann": float(gross.mean() * ann),
        "overlay_net_ann": float(net.mean() * ann),
        "buyhold_ann": float(bh.mean() * ann),
        "net_excess_ann": float(net_excess.mean() * ann),
        "net_excess_ir": float(ir),
        "turnover_per_yr": float(dpos.mean() * ann),
        "n": int(len(af)),
    }


# ---------------------------------------------------------------------------
# Robustness — sub-period sweep
# ---------------------------------------------------------------------------
def window_sweep(panel: pd.DataFrame, n_windows: int = 4) -> list[dict]:
    """Split the sample into ``n_windows`` contiguous blocks; re-estimate the slope-*t* in each.

    A real nowcast keeps its sign across sub-periods. Returns a list of dicts with the block label
    (start/end), slope, slope-t and n.
    """
    if panel.empty:
        return []
    idx = panel.index
    bounds = np.linspace(0, len(panel), n_windows + 1).astype(int)
    out = []
    for i in range(n_windows):
        sub = panel.iloc[bounds[i]:bounds[i + 1]]
        reg = predictive_regression(sub)
        lab = f"{idx[bounds[i]].date()} -> {idx[min(bounds[i+1], len(panel)) - 1].date()}"
        out.append({"window": lab, "slope": reg["slope"], "slope_t": reg["slope_t"], "n": reg["n"]})
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, nowcast_beta: float, n_seeds: int = 25, base_seed: int = 555) -> float:
    """Average the predictive-regression HAC slope-*t* over ``n_seeds`` synthetic worlds.

    Any synthetic-dependent claim averages the statistic over ≥ 20 seeds so no single lucky RNG seed
    can manufacture significance. Returns the mean slope-*t* across seeds for a planted
    ``nowcast_beta``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_series(nowcast_beta=nowcast_beta, seed=s)
        ts.append(predictive_regression(panel)["slope_t"])
    return float(np.nanmean(ts))
