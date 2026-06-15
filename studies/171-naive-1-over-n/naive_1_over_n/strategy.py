"""The 1/N strategy and its optimiser competitors — Study 171 (Naive-1-Over-N).

The DeMiguel-Garlappi-Uppal (2009, RFS) result, one honest tournament:

    *1/N equal-weight, rebalanced monthly, beats sample mean-variance, minimum-
    variance, and max-Sharpe (Tangency) portfolios out of sample on a rolling
    60-month estimation window — because estimation error swamps the optimiser's
    theoretical advantage over finite samples.*

We implement four strategies, all sharing the same universe, cost model, and
rebalancing schedule:

- **1/N (equal-weight)**     — fixed 1/N weight, no estimation. The null.
- **Min-Variance (MV)**      — minimise portfolio variance with 0 <= w_i <= 1,
                               sum = 1.  Uses rolling sample covariance.
- **Max-Sharpe (MxS)**       — tangency portfolio; sample mean and covariance
                               with short-selling disallowed (clipped to [0,1]).
- **Mean-Variance (MVO)**    — full mean-variance efficient frontier (risk-
                               aversion lambda = 3); uses sample mean + covariance.

Each strategy is simulated *out of sample* only: on the first trading day of
each month, weights are estimated from the prior ``lookback`` months of daily
returns and applied for the coming month. No data from the out-of-sample period
enters any estimation.

Costs: one-way turnover is charged at ``cost_bps`` on each rebalance.
Inference: HAC t-stat (Newey-West) on the annual return differential 1/N − optimiser,
to test whether 1/N significantly *beats* or *trails* each competitor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Weight solvers — the optimiser toolbox
# ---------------------------------------------------------------------------
def _equal_weights(n: int) -> np.ndarray:
    """The 1/N vector — no maths required."""
    return np.full(n, 1.0 / n)


def _min_variance(cov: np.ndarray) -> np.ndarray:
    """Minimum-variance portfolio weights (long-only, fully invested).

    Uses the closed-form solution for the unconstrained case and then
    clips to [0, 1] + renormalises — a simple long-only approximation.
    Degenerate covariance matrices fall back to 1/N.
    """
    n = cov.shape[0]
    try:
        cov_inv = np.linalg.pinv(cov)
        ones = np.ones(n)
        raw = cov_inv @ ones
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total <= 0:
            return _equal_weights(n)
        return raw / total
    except np.linalg.LinAlgError:
        return _equal_weights(n)


def _max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Tangency (max-Sharpe) portfolio weights (long-only, fully invested).

    Closed-form: w ∝ Sigma^{-1} * mu, clipped to [0, 1].
    Falls back to 1/N when mu is not all positive or matrix is degenerate.
    """
    n = cov.shape[0]
    try:
        cov_inv = np.linalg.pinv(cov)
        raw = cov_inv @ mu
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total <= 0:
            return _equal_weights(n)
        return raw / total
    except np.linalg.LinAlgError:
        return _equal_weights(n)


def _mean_variance(mu: np.ndarray, cov: np.ndarray, lam: float = 3.0) -> np.ndarray:
    """Mean-variance optimal weights for risk aversion lambda (long-only).

    Unconstrained: w = (1/lambda) * Sigma^{-1} * mu, then clip + renormalise.
    Falls back to 1/N when solution is degenerate.
    """
    n = cov.shape[0]
    try:
        cov_inv = np.linalg.pinv(cov)
        raw = (1.0 / lam) * (cov_inv @ mu)
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total <= 0:
            return _equal_weights(n)
        # Scale so weights sum to 1 (fully invested)
        return raw / total
    except np.linalg.LinAlgError:
        return _equal_weights(n)


# ---------------------------------------------------------------------------
# Out-of-sample backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    returns: pd.DataFrame,
    lookback: int = 60,
    strategy: str = "1n",
    cost_bps: float = 10.0,
    lam: float = 3.0,
) -> pd.Series:
    """Out-of-sample portfolio return series for one strategy.

    On the first trading day of each month, weights are estimated from the prior
    ``lookback`` months of daily returns (converted to monthly first for MV-type
    solvers, then applied to daily returns for that coming month).  The one-way
    rebalance turnover is charged at ``cost_bps`` per leg per period.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple return matrix (n_days x n_assets), date-indexed.
    lookback : int
        Number of months of in-sample history required before trading starts.
    strategy : str
        One of '1n', 'minvar', 'maxsharpe', 'mvo'.
    cost_bps : float
        One-way rebalance cost in basis points (applied to the sum of absolute
        weight changes at each rebalance).
    lam : float
        Risk aversion parameter for 'mvo'.

    Returns
    -------
    net_returns : pd.Series
        Daily net portfolio return, NaN before the warm-up period.
    """
    strat = strategy.lower().replace("-", "").replace("_", "").replace("/", "")
    n = returns.shape[1]
    daily_idx = returns.index
    cols = returns.columns

    # Group daily returns into months: key = (year, month)
    month_keys = list(zip(daily_idx.year, daily_idx.month))
    unique_months = sorted(set(month_keys), key=lambda x: x)

    # Pre-compute monthly returns for the estimation window
    monthly_ret = (1 + returns).resample("ME").prod() - 1  # monthly total return
    monthly_dates = monthly_ret.index

    # Build weight schedule: for each rebalance month, estimate weights
    # from the prior `lookback` months of MONTHLY returns
    weights_schedule: list[tuple[pd.Timestamp, np.ndarray]] = []
    for i, m_date in enumerate(monthly_dates):
        if i < lookback:
            continue
        in_sample = monthly_ret.iloc[i - lookback : i]
        mu_m = in_sample.mean().to_numpy()
        cov_m = in_sample.cov().to_numpy()

        if strat in ("1n", "ew", "equal", "equalweight"):
            w = _equal_weights(n)
        elif strat in ("minvar", "mv", "minvariance"):
            w = _min_variance(cov_m)
        elif strat in ("maxsharpe", "mxs", "tangency"):
            w = _max_sharpe(mu_m, cov_m)
        elif strat in ("mvo", "meanvariance"):
            w = _mean_variance(mu_m, cov_m, lam=lam)
        else:
            raise ValueError(f"Unknown strategy {strategy!r}")

        weights_schedule.append((m_date, w))

    if not weights_schedule:
        return pd.Series(np.nan, index=daily_idx, name=strategy)

    # Apply weights day-by-day, rebalancing at month boundaries
    port_returns = np.full(len(daily_idx), np.nan)
    current_w = None
    prev_w = None
    schedule_iter = iter(weights_schedule)
    next_rebal_date, next_w = next(schedule_iter)
    exhausted = False

    for i, (dt, row) in enumerate(zip(daily_idx, returns.itertuples(index=False))):
        # Rebalance when we enter a new month for which we have a schedule
        if not exhausted and dt.to_period("M") > next_rebal_date.to_period("M"):
            prev_w = current_w
            current_w = next_w
            try:
                next_rebal_date, next_w = next(schedule_iter)
            except StopIteration:
                exhausted = True

        if current_w is None:
            continue

        # Gross return
        asset_r = np.array(list(row))
        gross = float(current_w @ asset_r)

        # One-way rebalance cost (charged once, at the first day of the new month)
        cost = 0.0
        if prev_w is not None:
            turnover = float(np.abs(current_w - prev_w).sum())
            cost = turnover * cost_bps * 1e-4
            prev_w = None  # only charge once at the rebalance point

        port_returns[i] = gross - cost

    return pd.Series(port_returns, index=daily_idx, name=strategy)


def run_all_strategies(
    returns: pd.DataFrame,
    lookback: int = 60,
    cost_bps: float = 10.0,
    lam: float = 3.0,
) -> pd.DataFrame:
    """Run all four strategies on the same return series and return a DataFrame.

    Columns: '1n', 'minvar', 'maxsharpe', 'mvo'.
    """
    results = {}
    for strat in ("1n", "minvar", "maxsharpe", "mvo"):
        results[strat] = run_backtest(
            returns, lookback=lookback, strategy=strat, cost_bps=cost_bps, lam=lam
        )
    return pd.DataFrame(results).dropna(how="all")


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def annual_returns(daily_ret: pd.Series) -> pd.Series:
    """Compound annual returns from a daily series (drop NaN first)."""
    r = daily_ret.dropna()
    return (1.0 + r).resample("YE").prod() - 1.0


def summarize(daily_ret: pd.Series) -> dict:
    """Per-strategy headline metrics and HAC t-stat on mean annual return.

    Returns n (annual observations), CAGR, Sharpe (annualised), max drawdown,
    and HAC t-stat on the mean annual return. The annual granularity avoids the
    daily-autocorrelation problem and matches the reporting convention of
    DeMiguel, Garlappi & Uppal (2009).
    """
    r = daily_ret.dropna()
    if len(r) < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "tstat_annual")}

    # CAGR — guard against negative total (fully-ruined portfolio edge case)
    total = float((1.0 + r).prod())
    n_years = len(r) / TRADING_DAYS
    if n_years > 0 and total > 0:
        cagr = total ** (1.0 / n_years) - 1.0
    else:
        cagr = float("nan")

    # Annualised Sharpe on daily returns
    ann_mean = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")

    # Max drawdown
    equity = (1.0 + r).cumprod().to_numpy()
    mdd = _max_drawdown(equity)

    # HAC t-stat on ANNUAL returns (Newey-West, using annual observations)
    ann_r = annual_returns(r).to_numpy(dtype=float)
    ann_r = ann_r[np.isfinite(ann_r)]
    n_ann = ann_r.size
    tstat = float("nan")
    if n_ann > 2:
        mu = ann_r.mean()
        e = ann_r - mu
        lags = max(1, int(np.floor(4.0 * (n_ann / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n_ann
        for k in range(1, lags + 1):
            w_k = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w_k * float(e[k:] @ e[:-k]) / n_ann
        se = np.sqrt(max(lrv, 0.0) / n_ann)
        tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n": n_ann,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "tstat_annual": float(tstat),
    }


def compare_to_1n(
    returns: pd.DataFrame,
    lookback: int = 60,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Run all strategies and produce a comparison table vs 1/N.

    Returns a DataFrame with one row per strategy and columns:
    cagr, sharpe, max_dd, tstat_annual, delta_cagr (vs 1/N), delta_sharpe.
    """
    port = run_all_strategies(returns, lookback=lookback, cost_bps=cost_bps)
    rows = {}
    for col in port.columns:
        rows[col] = summarize(port[col])
    df = pd.DataFrame(rows).T
    ref_cagr = float(df.loc["1n", "cagr"]) if "1n" in df.index else float("nan")
    ref_sharpe = float(df.loc["1n", "sharpe"]) if "1n" in df.index else float("nan")
    df["delta_cagr"] = df["cagr"] - ref_cagr
    df["delta_sharpe"] = df["sharpe"] - ref_sharpe
    return df


def hac_tstat_diff(r1: pd.Series, r2: pd.Series) -> dict:
    """HAC t-stat on the per-period difference r1 - r2 (Newey-West on annual diffs).

    Used to test whether strategy A significantly outperforms strategy B, using
    annual return differences to avoid daily-autocorrelation noise.
    """
    ann1 = annual_returns(r1.dropna())
    ann2 = annual_returns(r2.dropna())
    aligned = ann1.align(ann2, join="inner")
    diff = (aligned[0] - aligned[1]).dropna().to_numpy(dtype=float)
    n = diff.size
    if n < 3:
        return {"mean_diff": float("nan"), "tstat": float("nan"), "n": n}
    mu = diff.mean()
    e = diff - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w_k = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w_k * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_diff": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
    }
