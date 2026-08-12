"""Strategy + inference for Study 886 — Agency MBS Carry.

The mechanical claim: agency MBS pay a spread over duration-matched Treasuries as
compensation for prepayment / negative-convexity risk. We isolate that spread as a
**duration-neutral, cash-neutral** monthly carry

    carry_t = (MBS_t - cash_t) - beta * (IEF_t - cash_t)

where both legs are excess of the BIL cash return and ``beta`` duration-matches the
Treasury leg to the MBS leg. Two hedges are reported:

  * **empirical** ``beta`` = OLS slope of the MBS excess on the Treasury excess (the
    *realized* rate sensitivity — it already prices in the negative-convexity drag, so
    the MBS realized beta sits well below its static OAD ratio);
  * **static** ``beta = OAD_MBS / OAD_IEF ~ 0.80`` (what a manager duration-matching off
    the published effective durations would short).

Inference: Newey-West (HAC) *t* on the carry mean (monthly bond returns are serially
correlated), a block-bootstrap CI on the annualized carry mean, an excess-vs-excess
Sharpe race (MBS-minus-cash vs Treasury-minus-cash), an era cut, and a costed version
that charges ETF spreads on both legs plus borrow on the short Treasury leg. All means
are quoted excess of cash so the ``(1 - beta)`` residual is genuine carry, not duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 220)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """Plain one-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) t of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    mu = x.mean()
    e = x - mu
    g0 = float(e @ e) / n
    var = g0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * float(e[k:] @ e[:-k]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def annualized_sharpe(x: np.ndarray, ppy: int = MONTHS_PER_YEAR) -> float:
    """Annualized Sharpe of an excess-return series (mean/std * sqrt(ppy))."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int = NW_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West HAC standard errors.

    Used both to fit the duration hedge (regress MBS excess on Treasury excess: the
    slope is the realized rate beta) and to report its HAC t's and R^2.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    Z = X * e[:, None]
    S = Z.T @ Z
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        G = Z[k:].T @ Z[:-k]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {"alpha": float(b[0]), "beta": float(b[1]),
            "t_alpha": float(b[0] / se[0]), "t_beta": float(b[1] / se[1]),
            "r2": r2, "n": n}


def mean_ci_bootstrap(x: np.ndarray, n_boot: int = 4000, alpha: float = 0.05,
                      ppy: int = MONTHS_PER_YEAR, seed: int = 886) -> dict:
    """Circular block-bootstrap CI for the ANNUALIZED mean of ``x`` (in %/yr).

    Monthly carry returns are mildly autocorrelated, so an i.i.d. bootstrap understates
    the interval; blocks of length ~ n^(1/3) preserve the local dependence.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return {"mean_ann_pct": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_negative": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    blk = max(1, round(n ** (1.0 / 3.0)))
    n_blocks = int(np.ceil(n / blk))
    offs = np.arange(blk)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[b] = x[idx].mean() * ppy * 100.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_ann_pct": float(x.mean() * ppy * 100.0),
            "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()),
            "n": n, "block_size": blk}


# --------------------------------------------------------------------------- #
# The duration-neutral carry
# --------------------------------------------------------------------------- #
def empirical_beta(ef: pd.DataFrame) -> float:
    """Realized rate sensitivity of the MBS leg on the Treasury leg (OLS slope)."""
    return float(np.polyfit(ef["ief"].values, ef["mbs"].values, 1)[0])


def carry_series(ef: pd.DataFrame, beta: float | None = None) -> tuple[pd.Series, float]:
    """Duration-neutral, cash-neutral monthly carry ``mbs - beta*ief`` (excess-of-cash).

    ``beta=None`` fits the empirical (realized-duration) hedge; pass a float for the
    static OAD-ratio hedge. Returns the carry series and the beta used.
    """
    if beta is None:
        beta = empirical_beta(ef)
    return ef["mbs"] - beta * ef["ief"], beta


def carry_stats(ef: pd.DataFrame, beta: float | None = None, lags: int = NW_LAGS) -> dict:
    """Headline stats for the duration-neutral carry over one MBS/Treasury window."""
    carry, beta_used = carry_series(ef, beta)
    reg = hac_ols(ef["mbs"].values, ef["ief"].values, lags)
    c = carry.values
    return {
        "n": len(ef),
        "start": str(ef.index.min().date()), "end": str(ef.index.max().date()),
        "beta": beta_used, "reg_beta": reg["beta"], "t_reg_beta": reg["t_beta"],
        "r2": reg["r2"],
        "carry_ann_pct": float(c.mean()) * MONTHS_PER_YEAR * 100.0,
        "carry_bps_mo": float(c.mean()) * 1e4,
        "t_hac": newey_west_t(c, lags),
        "t_1s": one_sample_t(c),
        "sharpe": annualized_sharpe(c),
        "max_dd_pct": max_drawdown(carry) * 100.0,
    }


def sharpe_race(ef: pd.DataFrame) -> dict:
    """Excess-vs-excess Sharpe race: MBS-minus-cash vs Treasury-minus-cash.

    Both legs are already excess of BIL. Reports each leg's annualized excess mean,
    volatility and Sharpe, and the Welch t on the raw (unhedged) MBS-minus-Treasury
    return difference.
    """
    mbs, ief = ef["mbs"].values, ef["ief"].values
    return {
        "n": len(ef),
        "mbs_sharpe": annualized_sharpe(mbs), "ief_sharpe": annualized_sharpe(ief),
        "mbs_ann_pct": float(np.nanmean(mbs)) * MONTHS_PER_YEAR * 100.0,
        "ief_ann_pct": float(np.nanmean(ief)) * MONTHS_PER_YEAR * 100.0,
        "mbs_vol_pct": float(np.nanstd(mbs, ddof=1)) * np.sqrt(MONTHS_PER_YEAR) * 100.0,
        "ief_vol_pct": float(np.nanstd(ief, ddof=1)) * np.sqrt(MONTHS_PER_YEAR) * 100.0,
        "sharpe_adv": annualized_sharpe(mbs) - annualized_sharpe(ief),
        "welch_t_raw": welch_t(mbs, ief),
    }


def era_cut(ef: pd.DataFrame, splits: list[str], beta: float | None = None,
            lags: int = NW_LAGS) -> list[dict]:
    """Duration-neutral carry stats over consecutive eras defined by ``splits`` dates."""
    edges = [ef.index.min()] + [pd.Timestamp(s) for s in splits] + [ef.index.max() + pd.Timedelta(days=1)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = ef[(ef.index >= lo) & (ef.index < hi)]
        if len(seg) < 8:
            continue
        carry, _ = carry_series(seg, beta)
        c = carry.values
        out.append({
            "start": str(seg.index.min().date()), "end": str(seg.index.max().date()),
            "n": len(seg), "carry_ann_pct": float(c.mean()) * MONTHS_PER_YEAR * 100.0,
            "t_hac": newey_west_t(c, lags),
        })
    return out


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the compounded carry series (fraction, <= 0)."""
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def calendar_year_table(returns: pd.Series) -> pd.Series:
    """Compounded carry return per calendar year (fraction)."""
    return (1.0 + returns).groupby(returns.index.year).prod() - 1.0


# --------------------------------------------------------------------------- #
# Tradability
# --------------------------------------------------------------------------- #
def costed_carry(ef: pd.DataFrame, beta: float | None = None,
                 mbs_spread_bps: float = 1.0, ief_spread_bps: float = 2.0,
                 borrow_annual_bps: float = 40.0, rebalances_per_year: float = 12.0,
                 lags: int = NW_LAGS) -> dict:
    """Net duration-neutral carry after realistic ETF frictions.

    Charges, per month: one-way ETF spreads on both legs prorated over
    ``rebalances_per_year`` hedge rebalances (each touches ~half the book turning over,
    charged conservatively as a full one-way on each leg per rebalance), plus borrow on
    the short Treasury leg (``beta`` notional). Reports gross vs net annualized carry and
    the net HAC t.
    """
    carry, beta_used = carry_series(ef, beta)
    per_rebal = (mbs_spread_bps + beta_used * ief_spread_bps) / 1e4
    cost_m = per_rebal * rebalances_per_year / MONTHS_PER_YEAR
    borrow_m = beta_used * borrow_annual_bps / 1e4 / MONTHS_PER_YEAR
    net = carry - cost_m - borrow_m
    return {
        "beta": beta_used,
        "gross_ann_pct": float(carry.mean()) * MONTHS_PER_YEAR * 100.0,
        "charge_ann_pct": (cost_m + borrow_m) * MONTHS_PER_YEAR * 100.0,
        "net_ann_pct": float(net.mean()) * MONTHS_PER_YEAR * 100.0,
        "t_net_hac": newey_west_t(net.values, lags),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(ef: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the duration-neutral carry estimator on a synthetic excess-return frame."""
    s = carry_stats(ef, beta=None, lags=lags)
    return {"beta": s["beta"], "carry_ann_pct": s["carry_ann_pct"],
            "t_hac": s["t_hac"], "sharpe": s["sharpe"], "n": s["n"]}
