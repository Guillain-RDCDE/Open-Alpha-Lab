"""Strategy and honest controls — Study 332 (Downside-Beta).

The Ang, Chen & Xing (2006) thesis: investors fear losses *especially* when the whole
market is falling, so a stock whose beta is higher on **down-market days** ("downside
beta", β⁻) is a worse hedge and should command a **higher** average return. The
falsifiable claim is a cross-sectional pricing one: sort stocks each month by their
trailing downside-beta; the high-β⁻ portfolio should out-earn the low-β⁻ portfolio, and
that spread should survive controlling for plain (full-sample) beta — otherwise it is
just the ordinary beta premium wearing a costume.

Downside beta (Bawa-Lindenberg / Ang-Chen-Xing convention)::

    beta_dn = cov(r_i, r_m | r_m < mu_m) / var(r_m | r_m < mu_m)

estimated over a trailing window. We default ``mu_m = 0`` (down = a negative market
day); a market-mean threshold is offered as a robustness knob. The companion upside beta
uses ``r_m > mu_m``. The *relative* downside beta ``beta_dn - beta`` (Ang-Chen-Xing's
own preferred control variable) isolates the part of downside risk not explained by
ordinary beta.

Three honest comparisons settle it:

1. **High-β⁻ minus Low-β⁻ spread** — the quintile long-short, the headline premium.
2. **Beta-matched control** — the same sort on plain (symmetric) beta. If the β⁻ spread
   is no bigger than the β spread, the "downside" story adds nothing.
3. **Random-portfolio control** — randomly drawn portfolios of the same size, seeded
   reproducibly, as a concentration null.

**One execution lag, documented exactly**: the signal (trailing downside beta) for month
``t`` is formed on the window ending at the last day of month ``t-1``; the portfolio
holds month ``t``'s realised returns. Signal known at end of ``t-1`` earns the return of
``t`` — one lag, applied once, in ``monthly_quantile_returns``.

**Survivorship bias** is named on the Signal axis (see ``data.load_real``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------
def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x (with intercept). NaN if too few points or x is flat."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 20:
        return np.nan
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan
    return float((xc @ (y - y.mean())) / denom)


def estimate_betas(
    stock_ret: np.ndarray,
    market_ret: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """Plain, downside and upside beta of one stock over one window.

    Returns ``{"beta", "beta_dn", "beta_up"}``. Downside beta is the OLS slope of the
    stock on the market restricted to days where ``market_ret < threshold`` (default 0 =
    a negative market day); upside beta restricts to ``market_ret > threshold``. The
    *full-sample* beta uses every day.
    """
    market_ret = np.asarray(market_ret, dtype=float)
    stock_ret = np.asarray(stock_ret, dtype=float)
    dn = market_ret < threshold
    up = market_ret > threshold
    return {
        "beta": _slope(stock_ret, market_ret),
        "beta_dn": _slope(stock_ret[dn], market_ret[dn]),
        "beta_up": _slope(stock_ret[up], market_ret[up]),
    }


def beta_panel(
    returns_df: pd.DataFrame,
    market: pd.Series,
    window_end: pd.Timestamp,
    lookback_days: int = 252,
    threshold: float = 0.0,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Cross-section of (beta, beta_dn, beta_up) estimated on the window *ending at*
    ``window_end`` (inclusive), looking back ``lookback_days`` trading days.

    Returns a DataFrame indexed by ticker with columns ``beta, beta_dn, beta_up,
    rel_dbeta`` where ``rel_dbeta = beta_dn - beta`` (the Ang-Chen-Xing control). Names
    with fewer than ``min_obs`` valid down-day observations get NaN downside beta and are
    dropped by the caller.
    """
    win = returns_df.loc[:window_end].tail(lookback_days)
    mkt = market.reindex(win.index)
    rows = {}
    mkt_arr = mkt.to_numpy(dtype=float)
    dn_count = int((mkt_arr < threshold).sum())
    for tic in win.columns:
        s = win[tic].to_numpy(dtype=float)
        if np.isfinite(s).sum() < min_obs or dn_count < min_obs:
            b = {"beta": np.nan, "beta_dn": np.nan, "beta_up": np.nan}
        else:
            b = estimate_betas(s, mkt_arr, threshold=threshold)
        rows[tic] = b
    out = pd.DataFrame(rows).T
    out["rel_dbeta"] = out["beta_dn"] - out["beta"]
    return out


# ---------------------------------------------------------------------------
# Month-ends helper
# ---------------------------------------------------------------------------
def month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading day of each month in ``index``."""
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).last())


# ---------------------------------------------------------------------------
# Cross-sectional quintile engine (the one execution lag lives here)
# ---------------------------------------------------------------------------
def monthly_quantile_returns(
    returns_df: pd.DataFrame,
    market: pd.Series,
    signal_col: str = "beta_dn",
    lookback_days: int = 252,
    threshold: float = 0.0,
    q: float = 0.20,
    min_stocks: int = 20,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Monthly high-minus-low quintile spread on ``signal_col`` (default downside beta).

    Each month-end ``t-1`` we estimate the beta panel on the trailing ``lookback_days``,
    sort the cross-section by ``signal_col``, form an equal-weight **high** quintile (top
    ``q``) and **low** quintile (bottom ``q``), and earn each name's realised *simple*
    return over the *next* calendar month (``t``). Signal at end of ``t-1`` → return of
    ``t``: one execution lag, applied once here.

    Returns a monthly DataFrame with columns ``high, low, market, spread, n_total,
    n_high, n_low``. ``spread = high - low`` (the long-short premium). ``market`` is the
    equal-weight mean of all ranked names that month (the long-only benchmark).
    """
    me = month_ends(returns_df.index)
    rows: list[dict] = []
    for k in range(len(me) - 1):
        form_end = me[k]
        hold_start = me[k]      # exclusive below
        hold_end = me[k + 1]
        panel = beta_panel(
            returns_df, market, window_end=form_end,
            lookback_days=lookback_days, threshold=threshold, min_obs=min_obs,
        )
        sig = panel[signal_col].dropna()
        if len(sig) < min_stocks:
            continue
        # Next-month realised simple return per name (compounded daily).
        nxt = returns_df.loc[(returns_df.index > hold_start) & (returns_df.index <= hold_end)]
        if nxt.empty:
            continue
        fwd = (1.0 + nxt).prod() - 1.0
        common = sig.index.intersection(fwd.dropna().index)
        if len(common) < min_stocks:
            continue
        sig = sig.loc[common]
        fwd = fwd.loc[common]
        lo_thr = sig.quantile(q)
        hi_thr = sig.quantile(1.0 - q)
        low_mask = sig <= lo_thr
        high_mask = sig >= hi_thr
        rows.append({
            "date": hold_end,
            "high": float(fwd[high_mask].mean()),
            "low": float(fwd[low_mask].mean()),
            "market": float(fwd.mean()),
            "spread": float(fwd[high_mask].mean() - fwd[low_mask].mean()),
            "n_total": int(len(common)),
            "n_high": int(high_mask.sum()),
            "n_low": int(low_mask.sum()),
        })
    if not rows:
        return pd.DataFrame(
            columns=["high", "low", "market", "spread", "n_total", "n_high", "n_low"]
        )
    res = pd.DataFrame(rows).set_index("date")
    res.index = pd.DatetimeIndex(res.index)
    return res


def random_spread_null(
    returns_df: pd.DataFrame,
    market: pd.Series,
    lookback_days: int = 252,
    q: float = 0.20,
    n_draws: int = 200,
    seed: int = 332,
    min_stocks: int = 20,
    min_obs: int = 60,
) -> pd.Series:
    """Null distribution of *random* high-minus-low spreads (same leg sizes).

    Each month we draw ``n_draws`` random partitions of the ranked names into two
    same-sized groups and record the difference in their forward means — the
    concentration null the real downside-beta spread must beat.
    """
    rng = np.random.default_rng(seed)
    me = month_ends(returns_df.index)
    out: list[float] = []
    for k in range(len(me) - 1):
        form_end, hold_start, hold_end = me[k], me[k], me[k + 1]
        panel = beta_panel(returns_df, market, window_end=form_end,
                           lookback_days=lookback_days, min_obs=min_obs)
        sig = panel["beta_dn"].dropna()
        if len(sig) < min_stocks:
            continue
        nxt = returns_df.loc[(returns_df.index > hold_start) & (returns_df.index <= hold_end)]
        if nxt.empty:
            continue
        fwd = (1.0 + nxt).prod() - 1.0
        common = sig.index.intersection(fwd.dropna().index)
        if len(common) < min_stocks:
            continue
        f = fwd.loc[common].to_numpy(dtype=float)
        n_leg = max(1, int(round(len(f) * q)))
        idx = np.arange(len(f))
        for _ in range(n_draws):
            perm = rng.permutation(idx)
            hi = f[perm[:n_leg]].mean()
            lo = f[perm[n_leg:2 * n_leg]].mean()
            out.append(float(hi - lo))
    return pd.Series(out, name="random_spread")


# ---------------------------------------------------------------------------
# Costs (one-way x NAV; longs and shorts both turn over)
# ---------------------------------------------------------------------------
def net_spread(
    res: pd.DataFrame,
    one_way_bps: float = 10.0,
    monthly_turnover: float = 1.0,
) -> pd.Series:
    """Cost-charged monthly spread.

    The long-short rebalances both legs each month. A round-trip on each leg at
    ``one_way_bps`` one-way, scaled by ``monthly_turnover`` (1.0 = full rebalance),
    costs ``2 legs × 2 (round-trip) × one_way_bps`` per month on the spread. Costs are
    one-way × NAV; shorts pay the same execution here (borrow is layered in beat 6).
    """
    drag = monthly_turnover * 2.0 * 2.0 * one_way_bps * 1e-4
    return (res["spread"] - drag).rename("net_spread")


# ---------------------------------------------------------------------------
# Summary statistics — HAC t and block bootstrap
# ---------------------------------------------------------------------------
def summarize(monthly_ret: pd.Series, periods: int = 12) -> dict:
    """Headline statistics for a monthly return series (HAC t-stat, annualised)."""
    r = pd.Series(monthly_ret).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "mean_ann", "sharpe_ann", "n")}
    mu = float(r.mean())
    std = float(r.std(ddof=1))
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")
    mean_ann = mu * periods
    vol_ann = std * np.sqrt(periods)
    return {
        "mean": mu,
        "vol": std,
        "sharpe": (mu / std) if std > 0 else float("nan"),
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "mean_ann": float(mean_ann),
        "sharpe_ann": float(mean_ann / vol_ann) if vol_ann > 0 else float("nan"),
        "n": n,
    }


def block_bootstrap_ci(
    monthly_ret: pd.Series,
    block: int = 6,
    n_boot: int = 2000,
    seed: int = 332,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a monthly return series.

    Resamples overlapping blocks of length ``block`` to respect autocorrelation, then
    returns the ``(alpha/2, 1-alpha/2)`` quantiles of the bootstrap mean.
    """
    r = pd.Series(monthly_ret).astype(float).dropna().to_numpy()
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[b] = r[idx.ravel()[:n]].mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (lo, hi)
