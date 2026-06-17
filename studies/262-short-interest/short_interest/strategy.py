"""Strategy and honest controls -- Study 262 (Short-Interest).

The recipe is a single cross-sectional sort: rank the basket by short interest
(short percent of float), form a *high-SI* portfolio (the most heavily shorted
quantile) and a *low-SI* portfolio (the least shorted quantile), and compare
their forward returns.  Two opposing hypotheses:

- **Informed shorts (Boehmer-Jones-Zhang 2008)**: short sellers are informed,
  so high-SI stocks should *bleed* -- the high-minus-low spread should be
  NEGATIVE (high-SI underperforms).
- **Crowded-short reversal / squeeze**: heavily shorted stocks are oversold and
  prone to squeezes, so high-SI stocks should *bounce* -- the spread should be
  POSITIVE.

Because this is a SINGLE cross-section (one snapshot date, forward returns over
one horizon), the "spread" is a single number.  Honest inference therefore uses
the *cross-sectional* dispersion of individual-stock forward returns: the spread
of two quantile means with a two-sample t-stat, plus a permutation/bootstrap
null that shuffles the SI labels.  This is deliberately weaker than a long
monthly time series -- and we say so on the Signal axis.

Three honest comparisons:

1. **High-SI minus Low-SI** -- the classic long-short quantile spread.
2. **Cross-sectional OLS slope** -- regress forward return on standardized short
   interest; the slope's sign and t-stat summarise the monotone relationship.
3. **Label-shuffle null** -- randomly permute the SI labels across stocks many
   times; the spread under shuffling is the null distribution.  The real spread
   must clear this to be called a signal.

**Survivorship + staleness**: the basket is fixed survivors as of the snapshot,
and short interest is a single static reading.  All real-tape results are upper
bounds; the bias is named on the Signal axis.

**Execution lag**: short interest is reported with a settlement lag (FINRA
bi-monthly), so a tradable rule would form positions only after the figure is
public -- roughly one settlement period (~2 weeks) after the snapshot.  The
forward returns here are measured from the snapshot close, which is a mild
optimism; a realistic implementation enters ~2 weeks later (documented).

**Costs**: high-SI names are exactly the hard-to-borrow, high-fee, illiquid
shorts.  The short leg pays borrow (often 5-50%+/yr for the crowded names) plus
wide spreads; ``cost_adjusted_spread`` charges one-way costs x NAV with a borrow
add-on on the short leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional quantile sort
# ---------------------------------------------------------------------------
def quantile_spread(
    df: pd.DataFrame,
    si_col: str = "short_pct_float",
    ret_col: str = "fwd_ret",
    q: float = 0.20,
) -> dict:
    """High-SI minus low-SI forward-return spread on a single cross-section.

    Sorts ``df`` by ``si_col``; the top ``q`` fraction (most heavily shorted) is
    the HIGH portfolio, the bottom ``q`` fraction is the LOW portfolio.  Returns
    a dict with each leg's equal-weight mean forward return, the spread
    (HIGH - LOW), and a two-sample (Welch) t-stat on that spread, plus counts.

    Sign convention: a POSITIVE spread means heavily shorted stocks *bounce*
    (bounce); a NEGATIVE spread means they *bleed*.
    """
    d = df[[si_col, ret_col]].dropna()
    n = len(d)
    if n < 6:
        return {k: np.nan for k in ("high", "low", "spread", "tstat", "n_high", "n_low", "n")}

    hi_thresh = d[si_col].quantile(1.0 - q)
    lo_thresh = d[si_col].quantile(q)
    hi = d.loc[d[si_col] >= hi_thresh, ret_col]
    lo = d.loc[d[si_col] <= lo_thresh, ret_col]

    high = float(hi.mean())
    low = float(lo.mean())
    spread = high - low

    # Welch two-sample t-stat on the difference of means (cross-sectional).
    vh, vl = float(hi.var(ddof=1)), float(lo.var(ddof=1))
    nh, nl = len(hi), len(lo)
    se = np.sqrt(vh / nh + vl / nl) if (nh > 1 and nl > 1) else np.nan
    tstat = float(spread / se) if (se and se > 0) else np.nan

    return {
        "high": high,
        "low": low,
        "spread": float(spread),
        "tstat": tstat,
        "n_high": int(nh),
        "n_low": int(nl),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Cross-sectional OLS slope (standardized short interest -> forward return)
# ---------------------------------------------------------------------------
def cross_sectional_slope(
    df: pd.DataFrame,
    si_col: str = "short_pct_float",
    ret_col: str = "fwd_ret",
) -> dict:
    """OLS of forward return on standardized short interest.

    Returns ``slope`` (return per +1 SD of short interest), its OLS ``tstat``,
    the ``r2``, and ``n``.  A negative slope = high SI predicts low returns
    (informed shorts bleed); positive = bounce.
    """
    d = df[[si_col, ret_col]].dropna()
    n = len(d)
    if n < 6:
        return {k: np.nan for k in ("slope", "tstat", "r2", "n")}

    x = d[si_col].to_numpy(dtype=float)
    y = d[ret_col].to_numpy(dtype=float)
    xs = (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)

    X = np.column_stack([np.ones(n), xs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = np.sqrt(sigma2 * xtx_inv[1, 1]) if np.isfinite(sigma2) else np.nan
    tstat = float(beta[1] / se_slope) if (se_slope and se_slope > 0) else np.nan

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"slope": float(beta[1]), "tstat": tstat, "r2": float(r2), "n": int(n)}


# ---------------------------------------------------------------------------
# Label-shuffle null
# ---------------------------------------------------------------------------
def shuffle_null(
    df: pd.DataFrame,
    si_col: str = "short_pct_float",
    ret_col: str = "fwd_ret",
    q: float = 0.20,
    n_shuffles: int = 2000,
    seed: int = 262,
) -> dict:
    """Permutation null: shuffle SI labels, recompute the quantile spread.

    Returns the empirical two-sided p-value of the observed spread against the
    label-shuffle null, plus the null mean/std.  A genuine signal yields a small
    p-value; pure luck yields p ~ 0.5-1.0.
    """
    d = df[[si_col, ret_col]].dropna()
    obs = quantile_spread(d, si_col=si_col, ret_col=ret_col, q=q)["spread"]
    if not np.isfinite(obs):
        return {"obs": np.nan, "p_value": np.nan, "null_mean": np.nan, "null_std": np.nan}

    rng = np.random.default_rng(seed)
    si = d[si_col].to_numpy(dtype=float)
    ret = d[ret_col].to_numpy(dtype=float)
    n = len(d)
    k = max(1, int(round(n * q)))

    null_spreads = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(n)
        si_p = si[perm]
        order = np.argsort(si_p)
        lo_idx = order[:k]
        hi_idx = order[-k:]
        null_spreads[i] = ret[hi_idx].mean() - ret[lo_idx].mean()

    p_value = float((np.abs(null_spreads) >= abs(obs)).mean())
    return {
        "obs": float(obs),
        "p_value": p_value,
        "null_mean": float(null_spreads.mean()),
        "null_std": float(null_spreads.std(ddof=1)),
    }


def bootstrap_spread_ci(
    df: pd.DataFrame,
    si_col: str = "short_pct_float",
    ret_col: str = "fwd_ret",
    q: float = 0.20,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 262,
) -> dict:
    """Bootstrap CI for the quantile spread (resample stocks with replacement).

    Returns the point spread, the (1-alpha) percentile CI, and the fraction of
    resamples with the *opposite* sign of the point estimate (a blunt
    "could-this-be-zero" read).
    """
    d = df[[si_col, ret_col]].dropna()
    point = quantile_spread(d, si_col=si_col, ret_col=ret_col, q=q)["spread"]
    if not np.isfinite(point):
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "frac_opposite": np.nan}

    rng = np.random.default_rng(seed)
    n = len(d)
    si = d[si_col].to_numpy(dtype=float)
    ret = d[ret_col].to_numpy(dtype=float)
    k = max(1, int(round(n * q)))

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        si_b = si[idx]
        ret_b = ret[idx]
        order = np.argsort(si_b)
        lo_idx = order[:k]
        hi_idx = order[-k:]
        boots.append(ret_b[hi_idx].mean() - ret_b[lo_idx].mean())
    boots = np.asarray(boots)

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    sign = np.sign(point) if point != 0 else 1.0
    frac_opposite = float((np.sign(boots) != sign).mean())
    return {"point": float(point), "ci_low": lo, "ci_high": hi, "frac_opposite": frac_opposite}


# ---------------------------------------------------------------------------
# Honest costs -- borrow + spread on the short leg
# ---------------------------------------------------------------------------
def cost_adjusted_spread(
    df: pd.DataFrame,
    si_col: str = "short_pct_float",
    ret_col: str = "fwd_ret",
    q: float = 0.20,
    one_way_bps: float = 25.0,
    borrow_ann_pct: float = 8.0,
    horizon_months: int = 3,
) -> dict:
    """The quantile spread net of trading costs and short-leg borrow.

    The strategy that exploits a POSITIVE spread is *long high-SI, short low-SI*;
    a NEGATIVE spread is exploited by *short high-SI, long low-SI* (where the
    expensive borrow lands on the high-SI leg either way).  We charge:

    - round-trip trading cost = ``2 * one_way_bps`` on both legs (entry+exit), and
    - borrow on whichever leg is SHORT for ``horizon_months``.  When the high-SI
      leg is the short (negative-spread / informed-shorts trade), borrow is the
      crowded-short rate; when the low-SI leg is short, borrow is trivial.

    Returns gross spread, the cost charged, and the net spread (all over the
    holding horizon, not annualized).
    """
    qs = quantile_spread(df, si_col=si_col, ret_col=ret_col, q=q)
    gross = qs["spread"]
    if not np.isfinite(gross):
        return {"gross": np.nan, "cost": np.nan, "net": np.nan}

    horizon_frac = horizon_months / 12.0
    trade_cost = 2.0 * (2.0 * one_way_bps * 1e-4)  # both legs, round trip
    # Borrow lands on the short leg. If gross > 0 we are long high-SI (cheap to
    # borrow the low-SI short); if gross < 0 we short the high-SI leg (expensive).
    if gross >= 0:
        borrow = 0.5 * 1e-2 * horizon_frac  # ~0.5%/yr generic short on low-SI leg
    else:
        borrow = borrow_ann_pct * 1e-2 * horizon_frac  # crowded-short borrow
    cost = trade_cost + borrow
    net_mag = abs(gross) - cost
    net = np.sign(gross) * max(net_mag, 0.0) if abs(gross) > cost else np.sign(gross) * (abs(gross) - cost)
    return {"gross": float(gross), "cost": float(cost), "net": float(net)}


# ---------------------------------------------------------------------------
# Summary helper (mirrors the desk convention)
# ---------------------------------------------------------------------------
def summarize_returns(series: pd.Series) -> dict:
    """Mean / vol / two-sided t-stat / hit-rate for a vector of stock returns.

    For a single cross-section this is the cross-sectional t-stat on the mean
    (i.i.d. across stocks); it is NOT a time-series HAC t-stat and is reported
    as such.  ``n`` is the number of stocks.
    """
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "tstat", "hit_rate", "n")}
    mu = float(r.mean())
    std = float(r.std(ddof=1))
    se = std / np.sqrt(n) if std > 0 else np.nan
    tstat = float(mu / se) if (se and se > 0) else np.nan
    return {
        "mean": mu,
        "vol": std,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "n": n,
    }
