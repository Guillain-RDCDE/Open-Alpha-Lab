"""The engine and its honest controls — Study 335 (Buzz-Sentiment-ETF).

The marketed claim is simple: an AI that reads social media can build a 75-stock
basket that *beats the market*. The honest test is therefore an **alpha** test,
not a return test — and specifically alpha **net of the fee you actually pay** and
**after stripping the beta you were always paid for**. A concentrated, high-beta
basket will out-return SPY in a bull tape on beta alone; that is not skill.

The machinery here:

1. ``daily_returns`` — simple close-to-close returns from a level frame.
2. ``capm_alpha`` — regress BUZZ excess return on SPY excess return; the
   **intercept is the alpha**, reported as bps/day, annualised %/yr, with a
   Newey-West (HAC) t-stat. This is the inference-bar number: REAL needs |t| ≥ 2
   on the *real* tape.
3. ``block_bootstrap_alpha_ci`` — a circular block bootstrap CI on the alpha,
   because i.i.d. resampling would destroy the volatility clustering the HAC SE
   is there to respect.
4. ``excess_sharpe`` — Sharpe of each leg's **excess-of-cash** return, so the
   BUZZ-vs-SPY race is excess-vs-excess (never raw-vs-excess), plus the
   information ratio of the BUZZ−SPY active return.
5. ``net_of_fee_summary`` — the headline table: CAGR, vol, Sharpe, max drawdown
   and total return for each leg, gross and net. BUZZ's published expense ratio
   is a real, recurring cost; ``fee_bps_yr`` lets the curious see it bite.

Costs / lag convention. The core BUZZ-vs-SPY comparison is a **buy-and-hold race**
— both legs fully invested, no turnover beyond the fund's own (already inside its
NAV and its fee). So the core needs **no execution lag**. The optional
``sentiment_overlay`` variant (own the higher-momentum leg) is a *timing* overlay:
its signal is known at the close of *t* and earns the return of *t+1* — one
``shift``, applied once — and rebalancing turnover is charged one-way × NAV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def daily_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns for each column of a level frame."""
    return levels.pct_change().dropna()


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat on a mean — no hard quantlab dependency
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat for the hypothesis that mean(x) = 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# CAPM alpha — the inferential heart
# ---------------------------------------------------------------------------
def capm_alpha(
    rets: pd.DataFrame,
    rf_bps_yr: float = 0.0,
    buzz: str = "buzz",
    spy: str = "spy",
) -> dict:
    """Regress BUZZ excess return on SPY excess return; the intercept is alpha.

    ``rf_bps_yr`` is the annual risk-free rate in bps, spread evenly across the
    year and subtracted from *both* legs (excess-of-cash, applied once). The
    alpha is reported in bps/day and annualised %/yr; its t-stat is the HAC
    (Newey-West) t-stat on the regression residual-implied intercept, computed
    via the standard OLS-with-HAC formula so it respects autocorrelation.

    Returns a dict: ``alpha_bps_day, alpha_ann_pct, beta, tstat, r2, n``.
    """
    rf_daily = rf_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR
    y = (rets[buzz] - rf_daily).to_numpy(dtype=float)
    x = (rets[spy] - rf_daily).to_numpy(dtype=float)
    n = y.size
    X = np.column_stack([np.ones(n), x])
    # OLS
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    # HAC (Newey-West) covariance of the coefficient vector.
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag-0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        A = (X[k:] * resid[k:, None])
        B = (X[:-k] * resid[:-k, None])
        Gamma = A.T @ B
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    alpha_daily = float(beta_hat[0])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    return {
        "alpha_bps_day": alpha_daily * 1e4,
        "alpha_ann_pct": alpha_daily * TRADING_DAYS_PER_YEAR * 100.0,
        "beta": float(beta_hat[1]),
        "tstat": float(alpha_daily / se_alpha) if se_alpha > 0 else float("nan"),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Circular block bootstrap CI on the alpha
# ---------------------------------------------------------------------------
def block_bootstrap_alpha_ci(
    rets: pd.DataFrame,
    rf_bps_yr: float = 0.0,
    block: int = 20,
    n_boot: int = 1000,
    seed: int = 335,
    buzz: str = "buzz",
    spy: str = "spy",
) -> dict:
    """Circular block bootstrap CI (95%) on the annualised CAPM alpha.

    Resamples contiguous blocks of the joined (BUZZ, SPY) return rows so the
    contemporaneous beta relationship and any autocorrelation survive the
    resample (i.i.d. row resampling would not). Returns the 2.5/50/97.5
    percentiles of the annualised alpha and the bootstrap P(alpha > 0).
    """
    rng = np.random.default_rng(seed)
    rf_daily = rf_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR
    y_all = (rets[buzz] - rf_daily).to_numpy(dtype=float)
    x_all = (rets[spy] - rf_daily).to_numpy(dtype=float)
    n = y_all.size
    n_blocks = int(np.ceil(n / block))
    alphas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        y = y_all[idx]
        x = x_all[idx]
        X = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alphas[b] = coef[0] * TRADING_DAYS_PER_YEAR * 100.0
    return {
        "lo": float(np.percentile(alphas, 2.5)),
        "med": float(np.percentile(alphas, 50.0)),
        "hi": float(np.percentile(alphas, 97.5)),
        "p_positive": float((alphas > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Excess-vs-excess Sharpe + information ratio
# ---------------------------------------------------------------------------
def _ann_sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd == 0 or r.size < 2:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def excess_sharpe(
    rets: pd.DataFrame,
    rf_bps_yr: float = 0.0,
    buzz: str = "buzz",
    spy: str = "spy",
) -> dict:
    """Excess-of-cash Sharpe for each leg + the information ratio of BUZZ−SPY.

    Both legs are put on the same clock — return *in excess of the same cash
    rate* — so the comparison is excess-vs-excess (never raw-vs-excess, which has
    manufactured verdicts elsewhere on this desk). The information ratio is the
    annualised Sharpe of the active (BUZZ minus SPY) return.
    """
    rf_daily = rf_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR
    rb = (rets[buzz] - rf_daily).to_numpy(dtype=float)
    rs = (rets[spy] - rf_daily).to_numpy(dtype=float)
    active = (rets[buzz] - rets[spy]).to_numpy(dtype=float)
    return {
        "sharpe_buzz": _ann_sharpe(rb),
        "sharpe_spy": _ann_sharpe(rs),
        "info_ratio": _ann_sharpe(active),
        "active_tstat": hac_tstat(active),
        "active_mean_bps_day": float(np.nanmean(active) * 1e4),
    }


# ---------------------------------------------------------------------------
# Net-of-fee headline summary
# ---------------------------------------------------------------------------
def _max_drawdown(level: np.ndarray) -> float:
    peak = np.maximum.accumulate(level)
    return float((level / peak - 1.0).min())


def net_of_fee_summary(
    levels: pd.DataFrame,
    fee_bps_yr: float = 0.0,
    buzz: str = "buzz",
    spy: str = "spy",
) -> pd.DataFrame:
    """Per-leg CAGR / vol / Sharpe / max-drawdown / total return.

    ``fee_bps_yr`` is an *additional* fee drag applied to BUZZ only (use 0 when
    the input levels already embed the fund's NAV-level fee — the real tape does;
    the synthetic tape lets you toggle it). A purely descriptive table; the
    inference lives in ``capm_alpha``.
    """
    rets = daily_returns(levels)
    fee_daily = fee_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR
    rows = []
    for col in (buzz, spy):
        r = rets[col].to_numpy(dtype=float).copy()
        if col == buzz and fee_daily:
            r = r - fee_daily
        lvl = np.concatenate([[1.0], np.cumprod(1.0 + r)])
        years = len(r) / TRADING_DAYS_PER_YEAR
        total = lvl[-1] - 1.0
        cagr = lvl[-1] ** (1.0 / years) - 1.0 if years > 0 else float("nan")
        rows.append({
            "leg": col,
            "total_ret_pct": total * 100.0,
            "cagr_pct": cagr * 100.0,
            "vol_pct": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0),
            "sharpe": _ann_sharpe(r),
            "max_dd_pct": _max_drawdown(lvl) * 100.0,
        })
    return pd.DataFrame(rows).set_index("leg")


# ---------------------------------------------------------------------------
# Optional timing overlay (the only place a lag is needed)
# ---------------------------------------------------------------------------
def sentiment_overlay(
    levels: pd.DataFrame,
    lookback: int = 20,
    cost_bps: float = 5.0,
    buzz: str = "buzz",
    spy: str = "spy",
) -> dict:
    """A momentum-switch overlay: hold whichever leg has the stronger trailing
    return, re-checked daily.

    Look-ahead discipline: the leg choice uses returns through the close of *t*
    and earns the return of *t+1* (one ``shift``, applied once). Switching incurs
    a one-way cost of ``cost_bps`` × NAV on the turnover. Returned net + gross
    annualised Sharpe so the overlay can be raced against plain BUZZ and SPY.
    """
    rets = daily_returns(levels)
    trail = levels.pct_change(lookback)
    pick = (trail[buzz] > trail[spy]).astype(int)  # 1 = hold BUZZ, 0 = hold SPY
    # signal known at close of t earns the return of t+1 (one shift), then align
    # to the returns index (which drops the first level row).
    pos = pick.shift(1).reindex(rets.index).fillna(0).astype(int)
    leg_ret = np.where(pos.to_numpy() == 1, rets[buzz].to_numpy(), rets[spy].to_numpy())
    switches = pos.diff().abs().fillna(0).to_numpy()
    gross = leg_ret
    net = leg_ret - switches * cost_bps * 1e-4
    return {
        "sharpe_gross": _ann_sharpe(gross),
        "sharpe_net": _ann_sharpe(net),
        "n_switches": int(switches.sum()),
        "frac_in_buzz": float(pos.mean()),
    }
