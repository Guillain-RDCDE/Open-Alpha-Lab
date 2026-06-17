"""Strategy and analysis layer for Study 272 (Champagne-Index).

The Champagne Indicator (folklore): when champagne shipments boom, euphoria has
peaked and equities are near a top — so high champagne *growth* should predict
*low* forward equity returns. We implement this as a continuous predictive
regression and pin it against the honest nulls:

  - A **predictive OLS** of next-year S&P return on standardized champagne YoY
    growth, with a **HAC (Newey-West) t-stat** on the slope. The folklore
    predicts a *negative* slope (more bubbly -> worse forward return). REAL needs
    |t_HAC| >= 2 on the real tape.
  - A **permutation test**: shuffle the champagne-growth labels 10,000 times and
    see where the real slope (and the real long/short spread) land.
  - A **tercile spread**: long the S&P after the lowest-growth third of years,
    short after the highest-growth third; report the gross and net annualized
    spread with one-way costs and a one-year execution lag.
  - A **tiny-n reckoning**: with only ~25 forward observations and ~17% annual
    equity vol, the minimum detectable slope is large; n alone may sink the test.

The honest baseline: the S&P rises in most years regardless of champagne. A
continuous predictor must clear |t| >= 2 *and* survive permutation — literature
support alone is WEAK.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def standardized_growth(df: pd.DataFrame, col: str = "ship_growth") -> pd.Series:
    """Z-score champagne YoY growth (mean 0, sd 1) over the available sample."""
    g = df[col].astype(float)
    z = (g - g.mean()) / (g.std(ddof=0) + 1e-12)
    return z.rename("champ_z")


# ---------------------------------------------------------------------------
# HAC (Newey-West) OLS slope t-stat
# ---------------------------------------------------------------------------
def _nw_long_run_var(u: np.ndarray, x: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of the score x_t * u_t (Bartlett kernel)."""
    s = x * u
    n = len(s)
    gamma0 = float(np.dot(s, s) / n)
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float(np.dot(s[k:], s[:-k]) / n)
        var += 2.0 * w * cov
    return var


def predictive_ols(
    df: pd.DataFrame,
    ret_col: str = "sp500_return",
    sig_col: str = "ship_growth",
    lags: int | None = None,
) -> dict:
    """OLS of forward return on standardized champagne growth, with HAC slope t.

    Model: ``ret_t = a + b * z(growth_t) + u_t``. The folklore predicts ``b < 0``
    (boomier champagne -> worse forward equity return). We report the OLS slope,
    its naive and HAC (Newey-West) standard errors, and the HAC t-stat.

    Returns a dict with: ``n``, ``slope``, ``intercept``, ``slope_se_ols``,
    ``slope_se_hac``, ``t_ols``, ``t_hac``, ``r2``, ``corr``, ``lags``.
    """
    z = standardized_growth(df, col=sig_col).to_numpy(dtype=float)
    y = df[ret_col].to_numpy(dtype=float)
    n = len(y)

    X = np.column_stack([np.ones(n), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    resid = y - X @ beta

    # Naive OLS slope SE
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_ols = float(np.sqrt(sigma2 * XtX_inv[1, 1]))

    # HAC (Newey-West) slope SE: sandwich on the demeaned regressor.
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(lags, 1)
    xc = z - z.mean()
    sxx = float(np.dot(xc, xc))
    lrv = _nw_long_run_var(resid, xc, lags) * n  # sum, not mean, of scores
    se_hac = float(np.sqrt(lrv) / sxx) if sxx > 0 else float("nan")

    t_ols = b / se_ols if se_ols > 0 else float("nan")
    t_hac = b / se_hac if se_hac and se_hac > 0 else float("nan")

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(z, y)[0, 1]) if n > 1 else float("nan")

    return {
        "n": int(n),
        "slope": round(b, 5),
        "intercept": round(a, 5),
        "slope_se_ols": round(se_ols, 5),
        "slope_se_hac": round(se_hac, 5),
        "t_ols": round(float(t_ols), 3),
        "t_hac": round(float(t_hac), 3),
        "r2": round(float(r2), 4),
        "corr": round(float(corr), 4),
        "lags": int(lags),
    }


# ---------------------------------------------------------------------------
# Tercile long/short spread + permutation test
# ---------------------------------------------------------------------------
def tercile_spread(
    df: pd.DataFrame,
    ret_col: str = "sp500_return",
    sig_col: str = "ship_growth",
) -> dict:
    """Long the S&P after low-growth years, short after high-growth years.

    Splits the sample into terciles of champagne YoY growth. The folklore trade:
    be **long** the S&P in years following the *lowest* champagne-growth third
    (gloom = bottom), **short** in years following the *highest* third (euphoria
    = top). Reports the gross mean spread (low - high), per-arm means, and counts.
    """
    g = df[sig_col].astype(float).to_numpy()
    y = df[ret_col].astype(float).to_numpy()
    n = len(y)
    # tercile cutoffs
    lo_cut, hi_cut = np.quantile(g, [1 / 3, 2 / 3])
    lo_mask = g <= lo_cut
    hi_mask = g >= hi_cut

    mean_lo = float(y[lo_mask].mean()) if lo_mask.sum() else float("nan")
    mean_hi = float(y[hi_mask].mean()) if hi_mask.sum() else float("nan")
    spread = mean_lo - mean_hi  # folklore: positive if low-growth years do better

    return {
        "n": int(n),
        "n_low": int(lo_mask.sum()),
        "n_high": int(hi_mask.sum()),
        "mean_ret_low_growth": round(mean_lo, 4),
        "mean_ret_high_growth": round(mean_hi, 4),
        "spread_low_minus_high": round(float(spread), 4),
    }


def permutation_test(
    df: pd.DataFrame,
    ret_col: str = "sp500_return",
    sig_col: str = "ship_growth",
    n_permutations: int = 10_000,
    seed: int = 272,
) -> dict:
    """Permutation p-values for the OLS slope and the tercile spread.

    Shuffle the champagne-growth labels and recompute (a) the OLS slope and
    (b) the tercile spread. The folklore predicts a *negative* slope and a
    *positive* low-minus-high spread, so we report one-sided p-values in the
    folklore-favourable direction plus the two-sided slope p-value.

    Returns ``slope_obs``, ``spread_obs``, ``perm_p_slope_neg`` (P[perm slope <=
    obs]), ``perm_p_slope_two`` (two-sided on |slope|), ``perm_p_spread_pos``
    (P[perm spread >= obs]), and the full permutation arrays for plots.
    """
    base = predictive_ols(df, ret_col=ret_col, sig_col=sig_col)
    slope_obs = base["slope"]
    terc = tercile_spread(df, ret_col=ret_col, sig_col=sig_col)
    spread_obs = terc["spread_low_minus_high"]

    g = df[sig_col].astype(float).to_numpy()
    y = df[ret_col].astype(float).to_numpy()
    n = len(y)

    rng = np.random.default_rng(seed)
    perm_slopes = np.empty(n_permutations, dtype=float)
    perm_spreads = np.empty(n_permutations, dtype=float)
    lo_cut, hi_cut = np.quantile(g, [1 / 3, 2 / 3])

    for i in range(n_permutations):
        gp = rng.permutation(g)
        gz = (gp - gp.mean()) / (gp.std(ddof=0) + 1e-12)
        # slope via covariance / variance (var of standardized z = 1)
        perm_slopes[i] = float(np.cov(gz, y, ddof=0)[0, 1])
        lo_mask = gp <= lo_cut
        hi_mask = gp >= hi_cut
        ml = y[lo_mask].mean() if lo_mask.sum() else np.nan
        mh = y[hi_mask].mean() if hi_mask.sum() else np.nan
        perm_spreads[i] = ml - mh

    perm_p_slope_neg = float((perm_slopes <= slope_obs).mean())
    perm_p_slope_two = float((np.abs(perm_slopes) >= abs(slope_obs)).mean())
    perm_p_spread_pos = float((perm_spreads >= spread_obs).mean())

    return {
        "n": int(n),
        "slope_obs": slope_obs,
        "spread_obs": spread_obs,
        "perm_p_slope_neg": round(perm_p_slope_neg, 4),
        "perm_p_slope_two": round(perm_p_slope_two, 4),
        "perm_p_spread_pos": round(perm_p_spread_pos, 4),
        "perm_slopes": perm_slopes,
        "perm_spreads": perm_spreads,
    }


# ---------------------------------------------------------------------------
# Tradable backtest: net long/short with costs and execution lag
# ---------------------------------------------------------------------------
def backtest_long_short(
    df: pd.DataFrame,
    ret_col: str = "sp500_return",
    sig_col: str = "ship_growth",
    cost_bps: float = 10.0,
    borrow_bps: float = 50.0,
) -> dict:
    """A toy tradable version with honest costs and a documented one-year lag.

    The signal is already lagged (``forward`` convention: champagne growth over
    year Y is paired with the S&P return earned in Y+1, and the CIVC figure is
    published in January Y+1, so the position is taken *after* the figure is
    known — one full year of execution lag baked into the data alignment).

    Trade: long the S&P in years following below-median champagne growth, short
    in years following above-median growth. One-way trading cost ``cost_bps`` per
    rebalance times turnover (assumed full each year), and an annual borrow fee
    ``borrow_bps`` on every short-year NAV. Returns gross/net annualized return
    of the long/short overlay and the long-only buy-and-hold benchmark, all on
    the *price-only* ^GSPC tape (dividends excluded — total return understated).
    """
    g = df[sig_col].astype(float).to_numpy()
    y = df[ret_col].astype(float).to_numpy()
    n = len(y)
    med = float(np.median(g))
    pos = np.where(g <= med, 1.0, -1.0)  # +1 long after low growth, -1 short after high

    gross = pos * y
    # turnover: assume the position flips ~ every other year; charge full one-way
    # cost on each year as a conservative upper bound (cost x NAV).
    cost = np.full(n, cost_bps * 1e-4)
    borrow = np.where(pos < 0, borrow_bps * 1e-4, 0.0)  # shorts pay borrow
    net = gross - cost - borrow

    def _ann(r):
        return float(np.prod(1.0 + r) ** (1.0 / max(len(r), 1)) - 1.0)

    return {
        "n": int(n),
        "gross_ann": round(_ann(gross), 4),
        "net_ann": round(_ann(net), 4),
        "bah_ann": round(_ann(y), 4),
        "n_short_years": int((pos < 0).sum()),
        "cost_bps": cost_bps,
        "borrow_bps": borrow_bps,
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 272) -> dict:
    """One-stop summary dict for a merged (champagne x forward-return) DataFrame."""
    ols = predictive_ols(df)
    terc = tercile_spread(df)
    perm = permutation_test(df, n_permutations=n_permutations, seed=seed)
    bt = backtest_long_short(df)
    uncond_up = float(df["mkt_up"].mean())

    return {
        "n": ols["n"],
        "slope": ols["slope"],
        "corr": ols["corr"],
        "r2": ols["r2"],
        "t_ols": ols["t_ols"],
        "t_hac": ols["t_hac"],
        "lags": ols["lags"],
        "uncond_up_rate_pct": round(uncond_up * 100, 1),
        "mean_ret_low_growth_pct": round(terc["mean_ret_low_growth"] * 100, 1),
        "mean_ret_high_growth_pct": round(terc["mean_ret_high_growth"] * 100, 1),
        "spread_pct": round(terc["spread_low_minus_high"] * 100, 1),
        "perm_p_slope_neg": perm["perm_p_slope_neg"],
        "perm_p_slope_two": perm["perm_p_slope_two"],
        "perm_p_spread_pos": perm["perm_p_spread_pos"],
        "gross_ann_pct": round(bt["gross_ann"] * 100, 1),
        "net_ann_pct": round(bt["net_ann"] * 100, 1),
        "bah_ann_pct": round(bt["bah_ann"] * 100, 1),
    }
