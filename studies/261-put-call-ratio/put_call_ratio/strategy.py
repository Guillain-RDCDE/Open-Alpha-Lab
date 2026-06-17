"""Strategy and honest controls -- Study 261 (Put-Call-Ratio).

The contrarian put/call claim: when the CBOE total put/call ratio reaches an
*extreme high* (the crowd is fearful, buying puts), the market is supposedly near
a bottom -- a contrarian *buy*. The mirror claim says an extreme *low* (greed)
precedes weakness. We test both as next-month timing rules on ^GSPC.

Three honest comparisons settle the matter:

1. **Extreme-fear long-only** -- go long ^GSPC for the month *after* the put/call
   ratio prints in its top ``q`` quantile of the trailing history; flat (in cash)
   otherwise. Compared against always-invested buy-and-hold.

2. **Contrarian z-spread** -- a continuous regression / quantile sort of forward
   return on the (standardised) put/call ratio: a positive slope means high fear
   precedes higher returns (the folklore), a flat slope means no information.

3. **Random-timing control** -- pick the same number of "in-market" months at
   random; the extreme-fear rule must beat this to claim genuine timing skill.

**No look-ahead**: the ratio is observed at month-end t and used to decide the
position held *through* month t+1. The trailing quantile threshold at t is built
only from observations up to and including t (expanding window with a burn-in).

**Execution lag**: one full month -- the signal at the close of month t sets the
position for the entirety of month t+1. There is no intramonth peeking.

**Costs**: the timing rule trades only when it switches between in-market and
cash; one-way costs are charged on NAV at each switch. Long-only here, so no
borrow; we still net costs honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def extreme_signal(
    pcr: pd.Series,
    q: float = 0.80,
    min_history: int = 36,
) -> pd.Series:
    """+1 when the put/call ratio is in an *extreme-high* (fear) state, else 0.

    At each month-end t, the threshold is the ``q`` quantile of the put/call
    ratio over the *expanding* trailing window (observations up to and including
    t). A value >= that threshold flags extreme fear -> contrarian long signal
    for the next month.

    No look-ahead: the threshold at t uses only data through t. The first
    ``min_history`` months emit 0 (burn-in) so the quantile is meaningfully
    estimated.

    Returns a Series aligned to ``pcr`` of {0, 1}.
    """
    pcr = pd.Series(pcr).astype(float)
    sig = pd.Series(0, index=pcr.index, dtype=int)
    for i, t in enumerate(pcr.index):
        if i < min_history:
            continue
        window = pcr.iloc[: i + 1]
        thresh = window.quantile(q)
        sig.loc[t] = int(pcr.loc[t] >= thresh)
    sig.name = "extreme_signal"
    return sig


def zscore_signal(pcr: pd.Series, min_history: int = 36) -> pd.Series:
    """Expanding-window z-score of the put/call ratio (no look-ahead).

    z_t = (pcr_t - mean(pcr_<=t)) / std(pcr_<=t). High positive z = extreme
    fear. NaN during the burn-in. Used for the continuous regression view.
    """
    pcr = pd.Series(pcr).astype(float)
    out = pd.Series(np.nan, index=pcr.index, dtype=float)
    for i, t in enumerate(pcr.index):
        if i < min_history:
            continue
        window = pcr.iloc[: i + 1]
        mu = window.mean()
        sd = window.std(ddof=1)
        out.loc[t] = (pcr.loc[t] - mu) / sd if sd > 0 else 0.0
    out.name = "pcr_z"
    return out


# ---------------------------------------------------------------------------
# Strategy returns
# ---------------------------------------------------------------------------
def timing_returns(
    df: pd.DataFrame,
    q: float = 0.80,
    min_history: int = 36,
    one_way_bps: float = 5.0,
) -> pd.DataFrame:
    """Contrarian extreme-fear timing returns vs buy-and-hold.

    ``df`` must have columns ``pcr`` (month-end ratio) and ``fwd_ret`` (the
    return earned in the *following* month). The rule: hold ^GSPC during month
    t+1 iff the put/call ratio at t is extreme-high; otherwise hold cash (0%).

    Costs: when the position switches (in<->out), charge ``one_way_bps`` on NAV.

    Returns a DataFrame indexed like ``df`` (rows where a signal exists) with:
      ``signal``   : 0/1 in-market flag for the coming month
      ``bh``       : buy-and-hold next-month return (always invested)
      ``strat``    : gross timing return (signal * fwd_ret)
      ``strat_net``: timing return after switch costs
      ``in_mkt``   : fraction-of-time invested indicator (==signal)
    """
    sig = extreme_signal(df["pcr"], q=q, min_history=min_history)
    out = pd.DataFrame(index=df.index)
    out["signal"] = sig
    out["bh"] = df["fwd_ret"].astype(float)
    out["strat"] = out["signal"] * out["bh"]

    # Switch costs: position changes between consecutive months.
    pos = out["signal"].astype(float)
    switch = pos.diff().abs().fillna(pos.abs())  # first month: cost if entering
    cost = switch * one_way_bps * 1e-4
    out["strat_net"] = out["strat"] - cost
    out["in_mkt"] = out["signal"]
    # Keep only rows past burn-in where the rule is active or explicitly flat
    out = out.iloc[min_history:].copy()
    return out


def random_timing_returns(
    df: pd.DataFrame,
    in_mkt_frac: float,
    n_draws: int = 500,
    min_history: int = 36,
    seed: int = 261,
) -> pd.Series:
    """Null distribution of mean monthly return for *random* in-market timing.

    For each draw, choose ``in_mkt_frac`` of the post-burn-in months at random to
    be invested (rest in cash), and record the mean monthly return. Returns a
    Series of length ``n_draws`` -- the null against which the extreme-fear rule's
    mean return is judged.
    """
    rng = np.random.default_rng(seed)
    fwd = df["fwd_ret"].astype(float).iloc[min_history:].to_numpy()
    n = len(fwd)
    k = max(1, int(round(in_mkt_frac * n)))
    means = np.empty(n_draws, dtype=float)
    idx = np.arange(n)
    for d in range(n_draws):
        pick = rng.choice(idx, size=k, replace=False)
        means[d] = fwd[pick].mean()
    return pd.Series(means, name="random_timing_mean")


# ---------------------------------------------------------------------------
# Continuous predictive regression
# ---------------------------------------------------------------------------
def predictive_regression(df: pd.DataFrame, min_history: int = 36) -> dict:
    """OLS of next-month return on the (expanding) put/call z-score.

    Positive slope => high fear precedes higher returns (the contrarian
    folklore). Reports slope (per 1-sigma of put/call), its HAC t-stat, R^2,
    and n.
    """
    z = zscore_signal(df["pcr"], min_history=min_history)
    y = df["fwd_ret"].astype(float)
    m = z.notna() & y.notna()
    x = z[m].to_numpy()
    yy = y[m].to_numpy()
    n = len(x)
    if n < 10:
        return {"slope": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    # HAC (Newey-West) covariance for the slope
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
    slope = float(beta[1])
    tstat = slope / se_slope if se_slope > 0 else float("nan")
    ss_res = float(resid @ resid)
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "tstat": float(tstat), "r2": float(r2), "n": int(n)}


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat) -- mirrors the desk convention
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return series (Newey-West HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(tstat),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out
