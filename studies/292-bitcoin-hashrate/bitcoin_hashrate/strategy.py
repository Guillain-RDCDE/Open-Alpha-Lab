"""Strategy and honest controls -- Study 292 (Bitcoin-Hashrate).

The folk thesis: a rising hash rate is bullish ("price follows hashrate").  We
formalise it three ways and pit each against an honest benchmark.

1. **Predictive regression.** Regress next-month BTC log-return on this month's
   hash-rate log-growth (``hash_mom``).  A genuine leading indicator needs a
   positive slope with a HAC *t* >= 2.  We also run a *horse race* that adds
   price's own one-month momentum as a control -- if the hashrate slope dies
   once price-momentum is in the regression, the "signal" was just price trend
   wearing a hashrate costume.

2. **Long/flat timing rule.** Go long BTC for month t+1 when this month's
   hash-rate growth (optionally smoothed over ``lookback`` months) is positive,
   else hold cash.  Compare net-of-cost equity to **buy-and-hold BTC** -- the
   only benchmark that matters for a single trending asset.

3. **Hash-Ribbons style cross-over** (``hash_ribbon_signal``): long when a fast
   moving average of hashrate is above a slow one (the popular "miner
   capitulation is over" trigger).

**No look-ahead:** the signal at month-end t uses only data through t and is
applied to the return *earned during t+1* (one-month execution lag, baked into
``forward_returns``).

**Single-asset trend caveat (named on the Signal axis):** BTC went up ~1000x
over the sample and hashrate trends up mechanically; any long-biased rule looks
great in absolute terms.  The benchmark is therefore *buy-and-hold*, and the
decisive test is whether the timing rule beats it net of costs (it does not).

**Costs:** one-way ``cost_bps`` charged on every change in position (a flip from
flat->long or long->flat costs one one-way trade on the NAV).  Long-only here,
so no borrow.  Returns are price-only (BTC pays no yield), labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def hashrate_momentum(
    df: pd.DataFrame,
    lookback: int = 1,
) -> pd.Series:
    """Log-growth of the hash rate over ``lookback`` months, known at month-end t.

    ``lookback=1`` is the raw monthly log-change; larger values smooth out the
    (noisy, China-ban-driven) monthly wiggles.  NaN for the first ``lookback``
    rows.
    """
    log_h = np.log(df["hashrate"].astype(float))
    mom = log_h.diff(lookback)
    mom.name = "hash_mom"
    return mom


def price_momentum(df: pd.DataFrame, lookback: int = 1) -> pd.Series:
    """Log-growth of BTC price over ``lookback`` months (the control regressor)."""
    log_p = np.log(df["price"].astype(float))
    mom = log_p.diff(lookback)
    mom.name = "price_mom"
    return mom


def forward_returns(df: pd.DataFrame) -> pd.Series:
    """Simple BTC return earned *during month t+1*, aligned to the signal at t.

    ``forward_returns.loc[t]`` is the return from close(t) to close(t+1).  This
    is what a position taken at month-end t (one-month execution lag) earns.
    """
    ret = df["price"].astype(float).pct_change().shift(-1)
    ret.name = "fwd_ret"
    return ret


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def predictive_regression(
    df: pd.DataFrame,
    lookback: int = 1,
    add_price_control: bool = False,
) -> dict:
    """Regress next-month BTC log-return on hashrate momentum (HAC t-stats).

    Model::

        r_{t+1} = a + b * hash_mom_t  [+ c * price_mom_t] + e_{t+1}

    Returns a dict with the slope(s), HAC *t*-stats, R^2 and n.  When
    ``add_price_control`` is True, BTC's own one-month momentum is added so we
    can see whether hashrate adds anything beyond price trend.
    """
    hm = hashrate_momentum(df, lookback=lookback)
    fwd = forward_returns(df)
    fwd_log = np.log1p(fwd)

    cols = {"hash_mom": hm}
    if add_price_control:
        cols["price_mom"] = price_momentum(df, lookback=1)
    X = pd.DataFrame(cols)
    data = pd.concat([fwd_log.rename("y"), X], axis=1).dropna()
    if len(data) < 12:
        return {"n": int(len(data)), "slope_hash": np.nan, "t_hash": np.nan,
                "slope_price": np.nan, "t_price": np.nan, "r2": np.nan}

    y = data["y"].to_numpy()
    reg_cols = list(X.columns)
    Xmat = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in reg_cols])
    n, k = Xmat.shape

    xtx_inv = np.linalg.pinv(Xmat.T @ Xmat)
    beta = xtx_inv @ Xmat.T @ y
    resid = y - Xmat @ beta

    # Newey-West HAC covariance
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (Xmat * resid[:, None]).T @ (Xmat * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = Xmat * resid[:, None]
        gamma = u[L:].T @ u[:-L]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = {"n": int(n), "r2": float(r2),
           "slope_hash": float(beta[1]),
           "t_hash": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
           "slope_price": np.nan, "t_price": np.nan}
    if add_price_control:
        out["slope_price"] = float(beta[2])
        out["t_price"] = float(beta[2] / se[2]) if se[2] > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Timing rules
# ---------------------------------------------------------------------------
def timing_signal(
    df: pd.DataFrame,
    lookback: int = 1,
) -> pd.Series:
    """Long (1) when hashrate momentum > 0, else flat (0).  Known at month-end t.

    The position is held for month t+1 (one-month execution lag is applied by
    ``forward_returns``).
    """
    hm = hashrate_momentum(df, lookback=lookback)
    pos = (hm > 0).astype(float)
    pos[hm.isna()] = np.nan
    pos.name = "position"
    return pos


def hash_ribbon_signal(
    df: pd.DataFrame,
    fast: int = 3,
    slow: int = 6,
) -> pd.Series:
    """Hash-Ribbons cross-over: long when fast hashrate MA > slow hashrate MA.

    The popular "miner capitulation is over" trigger.  Known at month-end t,
    applied to month t+1.
    """
    h = df["hashrate"].astype(float)
    ma_fast = h.rolling(fast).mean()
    ma_slow = h.rolling(slow).mean()
    pos = (ma_fast > ma_slow).astype(float)
    pos[ma_slow.isna()] = np.nan
    pos.name = "position"
    return pos


def backtest_timing(
    df: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 30.0,
) -> pd.DataFrame:
    """Net-of-cost monthly returns of a long/flat timing rule on BTC.

    ``position.loc[t]`` is the exposure (0 or 1) held during month t+1.  The
    gross return is ``position * forward_return``.  A flip in position costs
    ``cost_bps`` one-way on the traded notional (|delta position|).  BTC is
    long-only here -- no borrow.  Price-only returns (BTC pays no yield).

    Returns a DataFrame with ``["pos", "fwd_ret", "gross", "cost", "net",
    "bh"]`` where ``bh`` is the buy-and-hold BTC return earned in the same month.
    """
    fwd = forward_returns(df)
    pos = position.reindex(df.index)
    idx = pos.dropna().index.intersection(fwd.dropna().index)
    pos = pos.loc[idx]
    fwd = fwd.loc[idx]

    gross = pos * fwd
    dpos = pos.diff().abs().fillna(pos.abs())   # first trade = entering position
    cost = dpos * (cost_bps * 1e-4)
    net = gross - cost

    out = pd.DataFrame({
        "pos": pos,
        "fwd_ret": fwd,
        "gross": gross,
        "cost": cost,
        "net": net,
        "bh": fwd,                # buy-and-hold benchmark earns the raw BTC return
    })
    return out


def time_in_market(position: pd.Series) -> float:
    """Fraction of months the timing rule is long (exposure share)."""
    p = position.dropna()
    return float((p > 0).mean()) if len(p) else float("nan")


def turnover(position: pd.Series) -> float:
    """Average per-month one-way turnover (|delta position|)."""
    p = position.dropna()
    if len(p) < 2:
        return float("nan")
    return float(p.diff().abs().dropna().mean())


# ---------------------------------------------------------------------------
# Summary statistics
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
