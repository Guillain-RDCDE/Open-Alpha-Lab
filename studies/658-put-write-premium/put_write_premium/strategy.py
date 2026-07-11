"""Strategy + inference for Study 658 — Put-Write-Premium.

The claim: **systematically writing cash-secured S&P 500 puts harvests the variance risk
premium (implied vol is, on average, richer than what subsequently realizes) and beats
buy-and-hold on a risk-adjusted basis.** PUTW rolls a one-month at-the-money S&P 500 put every
month, collateralized in T-bills — CBOE's PUT methodology, in the only liquid ETF wrapper.

Two questions, kept separate on purpose (this is exactly where "harvests a real premium" and
"beats buy&hold risk-adjusted" can come apart):

1. **Does PUTW earn a return excess of cash — real evidence of a harvested premium?** A daily
   Newey-West (HAC) *t* on ``PUTW − BIL``.
2. **Is that excess just captured market beta, or something more?** A CAPM-style regression of
   ``PUTW − BIL`` on ``SPY − BIL`` — the alpha term is the honest answer to "is the premium just
   truncated equity beta?" A regression beta with an insignificant alpha says: no, it isn't
   beating beta, it just *is* beta (a smaller slice of it).
3. **Does the "protection" survive a crash, or does beta itself widen exactly when it hurts?**
   An interaction regression splits beta into a normal-day slope and an extra crash-day slope
   on days SPY falls ≥3% (an ex-ante, non-snooped threshold) — the asymmetric-beta smoking gun.
4. **Did it actually beat SPY, Sharpe for Sharpe, over its own live tape?** A circular block
   bootstrap on the Sharpe *difference* (PUTW excess-of-cash minus SPY excess-of-cash).

Execution convention: **zero look-ahead by construction** — every number here is the realized
daily return of a live, already-listed ETF (PUTW/SPY/BIL); there is no signal-to-trade lag to
document because nothing is timed, only held. Costs are already inside PUTW's own NAV (its
0.44% expense ratio and the fund's own roll execution) — the fairest, most honest cost model
for "can a retail investor actually get this," since no external cost assumption is layered on
top of what the fund itself already pays away.

Pure numpy + pandas + scipy.stats (for the synthetic engine's Black-Scholes pricer only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total returns from a close-price panel."""
    return px.pct_change().dropna(how="any")


def excess(ret: pd.DataFrame, cash_col: str = "BIL") -> pd.DataFrame:
    """Every column minus the cash column, cash column dropped."""
    ex = ret.sub(ret[cash_col], axis=0)
    return ex.drop(columns=[cash_col])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 10
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HAC (Newey-West, Bartlett kernel) OLS: returns (beta, se, t) for y = X @ beta + eps."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    t = np.where(se > 0, beta / np.where(se > 0, se, np.nan), np.nan)
    return beta, se, t


def hac_mean_t(x: np.ndarray, lags: int = 10) -> dict:
    """HAC t of the mean of a series (one-sample, via the newey_west_ols machinery)."""
    x = np.asarray(x, dtype=float)
    X = np.ones((len(x), 1))
    beta, se, t = newey_west_ols(x, X, lags=lags)
    return {"mean": float(beta[0]), "se": float(se[0]), "t": float(t[0]), "n": len(x)}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The headline: excess-of-cash return, CAPM alpha/beta, both HAC
# --------------------------------------------------------------------------- #
def excess_return_stats(ex_col: np.ndarray, lags: int = 10) -> dict:
    """HAC-t'd annualized excess-of-cash return for a single leg (PUTW or SPY)."""
    r = hac_mean_t(ex_col, lags=lags)
    return {"ann_pct": r["mean"] * TRADING_DAYS * 100.0, "t": r["t"], "n": r["n"]}


def capm_alpha_beta(ex_putw: np.ndarray, ex_spy: np.ndarray, lags: int = 10,
                    periods_per_year: int = TRADING_DAYS) -> dict:
    """CAPM regression ex_putw = alpha + beta * ex_spy + eps, Newey-West (HAC) t's.

    Answers "is the premium just truncated equity beta?" directly: a beta well below 1 with a
    statistically insignificant alpha means PUTW behaves like a smaller, unlevered slice of
    SPY's beta and nothing more — no genuine excess return beyond what beta already explains.

    ``periods_per_year`` annualizes ``alpha`` — 252 for the real DAILY tape (the default), 12
    for the synthetic engine's MONTHLY panel (the caller passes it explicitly there).
    """
    y = np.asarray(ex_putw, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(ex_spy, dtype=float)])
    beta, se, t = newey_west_ols(y, X, lags=lags)
    return {
        "alpha_period": float(beta[0]),
        "alpha_ann_pct": float(beta[0] * periods_per_year * 100.0),
        "t_alpha": float(t[0]),
        "beta": float(beta[1]), "t_beta": float(t[1]),
        "n": len(y),
    }


def sharpe_ann(ex: np.ndarray) -> float:
    ex = np.asarray(ex, dtype=float)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe_diff(ex_a: np.ndarray, ex_b: np.ndarray, block: int = 21,
                          n_boot: int = 2000, seed: int = 658) -> dict:
    """Circular block bootstrap CI for the Sharpe DIFFERENCE (a minus b), already excess-of-cash.

    Block = 21 trading days (~1 month) preserves the option-roll-cycle serial correlation.
    """
    a = np.asarray(ex_a, dtype=float)
    b = np.asarray(ex_b, dtype=float)
    n = len(a)
    point = sharpe_ann(a) - sharpe_ann(b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = sharpe_ann(a[sel]), sharpe_ann(b[sel])
        diffs[i] = sa - sb
        if sa > sb:
            wins += 1
    lo, hi = float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "frac_putw_wins": wins / n_boot, "n": n, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Crash-conditional beta — the asymmetric-beta test ("truncated beta" myth check)
# --------------------------------------------------------------------------- #
def crash_beta_interaction(ex_putw: np.ndarray, ex_spy: np.ndarray, spy_ret: np.ndarray,
                           threshold: float = -0.03, lags: int = 10) -> dict:
    """ex_putw = a + b*ex_spy + c*D + d*(D*ex_spy) + eps, D = 1{spy_ret <= threshold}.

    ``b`` is the normal-day beta, ``b + d`` the crash-day beta. A significant, positive ``d``
    means the "lower beta" that PUTW shows on average widens toward full equity beta exactly on
    the days a diversifier is supposed to earn its keep — the opposite of what "truncated beta,
    smaller downside" would predict if beta were genuinely stable.
    """
    y = np.asarray(ex_putw, dtype=float)
    x = np.asarray(ex_spy, dtype=float)
    D = (np.asarray(spy_ret, dtype=float) <= threshold).astype(float)
    X = np.column_stack([np.ones(len(y)), x, D, D * x])
    beta, se, t = newey_west_ols(y, X, lags=lags)
    return {
        "alpha_ann_pct": float(beta[0] * TRADING_DAYS * 100.0), "t_alpha": float(t[0]),
        "beta_normal": float(beta[1]), "t_beta_normal": float(t[1]),
        "crash_dummy_ann_pct": float(beta[2] * TRADING_DAYS * 100.0), "t_crash_dummy": float(t[2]),
        "crash_beta_extra": float(beta[3]), "t_crash_beta_extra": float(t[3]),
        "crash_beta_total": float(beta[1] + beta[3]),
        "n_crash_days": int(D.sum()), "n": len(y),
    }


# --------------------------------------------------------------------------- #
# Drawdown, capture, tails
# --------------------------------------------------------------------------- #
def nav(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).cumprod()


def max_drawdown(ret: pd.Series) -> float:
    n = nav(ret)
    return float((n / n.cummax() - 1.0).min())


def window_drawdown(ret: pd.Series, start: str, end: str) -> float:
    """Max peak-to-trough drawdown of ``ret`` restricted to [start, end] (own-window peak)."""
    seg = ret.loc[(ret.index >= start) & (ret.index <= end)]
    if len(seg) == 0:
        return float("nan")
    n = nav(seg)
    return float((n / n.cummax() - 1.0).min())


def monthly_capture(px: pd.DataFrame, putw_col: str = "PUTW", spy_col: str = "SPY") -> dict:
    """Up-month / down-month capture ratios of PUTW vs SPY (Morningstar-style mean-return
    ratio), month-end resample, trailing partial month dropped."""
    m = px.resample("ME").last()
    if px.index.max() < m.index.max():
        m = m.iloc[:-1]
    mret = m.pct_change().dropna(how="any")
    up = mret[spy_col] > 0
    dn = mret[spy_col] < 0
    up_cap = float(mret.loc[up, putw_col].mean() / mret.loc[up, spy_col].mean())
    dn_cap = float(mret.loc[dn, putw_col].mean() / mret.loc[dn, spy_col].mean())
    worst_m_putw = float(mret[putw_col].min())
    worst_m_spy = float(mret[spy_col].min())
    return {"n_months": len(mret), "n_up": int(up.sum()), "n_dn": int(dn.sum()),
            "up_capture": up_cap, "dn_capture": dn_cap,
            "worst_month_putw": worst_m_putw, "worst_month_spy": worst_m_spy,
            "worst_month_date": str(mret[putw_col].idxmin().date())}


def tail_stats(ret: pd.Series) -> dict:
    r = ret.dropna().to_numpy(dtype=float)
    mu, sd = r.mean(), r.std(ddof=1)
    skew = float(np.mean(((r - mu) / sd) ** 3)) if sd > 0 else float("nan")
    return {"vol_ann_pct": float(sd * np.sqrt(TRADING_DAYS) * 100.0), "skew": skew,
            "worst_day_pct": float(r.min() * 100.0)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — CAPM alpha, never real-tape evidence)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = 3) -> dict:
    """Run the CAPM alpha/beta regression on a synthetic (spy_ret, putw_ret, cash_ret) MONTHLY
    panel — ``periods_per_year=12`` so the annualized alpha is on the same footing as the
    monthly-generated engine (the real-tape headline instead annualizes daily x 252)."""
    ex_putw = (world["putw_ret"] - world["cash_ret"]).to_numpy()
    ex_spy = (world["spy_ret"] - world["cash_ret"]).to_numpy()
    return capm_alpha_beta(ex_putw, ex_spy, lags=lags, periods_per_year=12)
