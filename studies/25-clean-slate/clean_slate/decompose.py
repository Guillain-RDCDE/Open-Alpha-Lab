"""The teardown that earns the stamps — residual momentum is real, and it sheds the crash.

Four legs:

  1. :func:`capm_alpha` — regress the residual-WML on the market: a positive HAC-significant alpha is the
     `REAL` signal.
  2. :func:`crash_comparison` — the headline of *this* study: put residual-WML next to total-WML on
     skew, worst month and drawdown. Residual momentum is meant to keep the premium while shedding the
     crash (the systematic, beta-driven tail) — this measures whether it does.
  3. :func:`paired_crash_bootstrap` — "cleaner than total" is a claim about a *difference*, so it gets a
     paired test: block-bootstrap the aligned monthly (residual, total) pairs and put an interval on the
     skew gap and the Sharpe gap. Two individually insignificant books can't certify a difference by
     eyeball; this is the leg the third-axis stamp rests on.
  4. :func:`subsample_sharpe` / :func:`sharpe_bootstrap` — decay and an interval on the residual Sharpe.
     :func:`null_alpha_battery` calibrates the harness: across no-momentum panels (fresh seeds) the
     residual-WML alpha must be centred on ≈ 0 gross, so any single-seed reading can be placed inside
     the null spread instead of over-read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import wml_returns, summary

TRADING_DAYS_PER_YEAR = 252


def _ols_nw(y, x, lags=None):
    y = np.asarray(y, float); x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x]); n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ (X.T @ y); resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]; M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); G = u[k:].T @ u[:-k]; M += w * (G + G.T)
    cov = XtX_inv @ M @ XtX_inv; se = np.sqrt(np.diag(cov))
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan}


def capm_alpha(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
               periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """CAPM regression of the residual-WML on the equal-weight market: alpha (ann), HAC t, beta."""
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    mkt = panel.mean(axis=1).reindex(res.index)
    reg = _ols_nw(res.to_numpy(), mkt.to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    s = summary(res, periods_per_year)
    reg.update(sharpe=s["sharpe"], max_drawdown=s["max_drawdown"], skew=s["skew"])
    return reg


def _crash(r, periods_per_year):
    s = summary(r, periods_per_year)
    monthly = ((1.0 + r).resample("ME").prod() - 1.0).dropna()
    return {"sharpe": s["sharpe"], "skew": float(monthly.skew()),
            "worst_month_pct": float(monthly.min() * 100.0), "max_drawdown_pct": float(s["max_drawdown"] * 100.0)}


def crash_comparison(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Residual-WML vs total-WML on Sharpe, skew, worst month and drawdown — the cleaner-cousin test."""
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    tot = wml_returns(panel, market, residual=False, cost_bps=cost_bps, **kw)
    idx = res.index.intersection(tot.index)
    return {"residual": _crash(res.reindex(idx), periods_per_year),
            "total": _crash(tot.reindex(idx), periods_per_year)}


def subsample_sharpe(panel: pd.DataFrame, market=None, cost_bps: float = 5.0, n_chunks: int = 3, **kw) -> pd.DataFrame:
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    bounds = np.linspace(0, len(res), n_chunks + 1).astype(int)
    rows = {i: {"start": res.iloc[bounds[i]:bounds[i+1]].index.min().date(),
                "end": res.iloc[bounds[i]:bounds[i+1]].index.max().date(),
                "sharpe": summary(res.iloc[bounds[i]:bounds[i+1]])["sharpe"]} for i in range(n_chunks)}
    out = pd.DataFrame(rows).T; out.index.name = "chunk"
    return out


def paired_crash_bootstrap(panel: pd.DataFrame, market: pd.Series | None = None, n_boot: int = 2000,
                           alpha: float = 0.05, seed: int = 0, cost_bps: float = 5.0,
                           block_months: int = 6, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                           **kw) -> dict:
    """Paired block-bootstrap intervals for ``skew(residual) − skew(total)`` and the Sharpe gap.

    The two WML books trade the same names on the same days, so the only honest test of "cleaner"
    is *paired*: compound both streams to calendar months on their common window, then resample the
    aligned monthly **pairs** in circular blocks of ``block_months`` (skew and Sharpe are
    moment-based, and momentum-book months are autocorrelated — an i.i.d. resample would understate
    the spread). For each resample we recompute both books' monthly skew and annualised Sharpe and
    keep the differences. Reports the point gaps, the (1−alpha) percentile intervals, and the
    fraction of resamples in which the residual book is *better* (higher skew / higher Sharpe) — the
    paired evidence the third-axis stamp must rest on. An interval that straddles zero means the
    "cleaner" reading is directional, not significant.
    """
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    tot = wml_returns(panel, market, residual=False, cost_bps=cost_bps, **kw)
    idx = res.index.intersection(tot.index)
    m_res = ((1.0 + res.reindex(idx)).resample("ME").prod() - 1.0).dropna()
    m_tot = ((1.0 + tot.reindex(idx)).resample("ME").prod() - 1.0).dropna()
    common = m_res.index.intersection(m_tot.index)
    a_res, a_tot = m_res.reindex(common).to_numpy(), m_tot.reindex(common).to_numpy()
    n = a_res.size
    rng = np.random.default_rng(seed)

    def _skew(x):
        d = x - x.mean(); m2 = float((d**2).mean())
        return float((d**3).mean() / m2**1.5) if m2 > 0 else np.nan

    def _sr(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(12.0)) if sd > 0 else np.nan

    point_skew = _skew(a_res) - _skew(a_tot)
    point_sr = _sr(a_res) - _sr(a_tot)
    blk = max(1, min(int(block_months), n))
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    d_skew = np.full(n_boot, np.nan)
    d_sr = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        ix = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        r, t = a_res[ix], a_tot[ix]
        d_skew[b] = _skew(r) - _skew(t)
        d_sr[b] = _sr(r) - _sr(t)
    d_skew = d_skew[np.isfinite(d_skew)]
    d_sr = d_sr[np.isfinite(d_sr)]
    lo_s, hi_s = np.percentile(d_skew, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lo_r, hi_r = np.percentile(d_sr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_months": int(n), "n_boot": int(n_boot), "block_months": int(blk),
        "skew_diff": float(point_skew), "skew_ci_low": float(lo_s), "skew_ci_high": float(hi_s),
        "skew_frac_residual_better": float((d_skew > 0).mean()),
        "skew_significant": bool(lo_s > 0 or hi_s < 0),
        "sharpe_diff": float(point_sr), "sharpe_ci_low": float(lo_r), "sharpe_ci_high": float(hi_r),
        "sharpe_frac_residual_better": float((d_sr > 0).mean()),
        "sharpe_significant": bool(lo_r > 0 or hi_r < 0),
    }


def null_alpha_battery(seeds=(0, 1, 2, 3, 4), cost_bps: float = 5.0, **panel_kw) -> pd.DataFrame:
    """Residual-WML CAPM alpha across **no-momentum** synthetic panels, one fresh seed each.

    The harness calibration the null check needs: a single 16-year tape is *one draw*, so a single
    null run can land a 2-sigma alpha (≈ ±3-4%/yr at this panel size) without anything being wrong.
    Across seeds the gross alpha must centre on ≈ 0 — the harness already ranks 12-1 (the
    Blitz-Huij-Martens skip month sits between formation and holding), so there is no mechanical
    reversion to absorb — while the *net* mean sits below zero by the cost drag (``cost_bps`` on a
    ~12×/yr-turnover book ≈ −0.7%/yr at 5 bp). Returns one row per seed (alpha, HAC t) so a test or
    notebook can assert on the battery mean instead of over-reading one tape.
    """
    from .data import synthetic_panel
    rows = {}
    for s in seeds:
        p0, m0, _ = synthetic_panel(mom_strength=0.0, seed=int(s), **panel_kw)
        a = capm_alpha(p0, m0, cost_bps=cost_bps)
        rows[int(s)] = {"alpha_ann_pct": a["alpha_ann_pct"], "alpha_t": a["alpha_t"]}
    out = pd.DataFrame(rows).T
    out.index.name = "seed"
    return out


def sharpe_bootstrap(panel: pd.DataFrame, market=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                     cost_bps: float = 5.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw).to_numpy()
    n = res.size; rng = np.random.default_rng(seed)
    def _sr(x):
        sd = x.std(ddof=1); return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    point = _sr(res)
    boots = np.array([_sr(res[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean())}
