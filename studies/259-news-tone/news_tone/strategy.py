"""Strategy and honest controls -- Study 259 (News-Tone).

The claim under test: aggregate daily *news tone* (a single sentiment score for
the macro/market headline flow) predicts the *next* day's S&P 500 return.

Two readings of the data settle it:

1. **Same-day correlation** (the seductive but useless one): does today's tone
   line up with *today's* return?  Of course it does -- bad-news days are
   down days *by construction*.  This is mechanical contemporaneous comovement,
   not a tradable edge, and we report it only to show how it flatters the story.

2. **Next-day predictability** (the honest one): does today's tone forecast
   *tomorrow's* return?  This is what a trader could actually act on (form the
   position at today's close, hold through tomorrow -- a one-day execution lag).
   If markets price news within the day, this should be ~zero.

The strategy: go long ^GSPC tomorrow when today's tone is positive (good-news
flow), short when today's tone is negative.  We report the long-short timing
return, a sign-only variant, and a continuous-tilt variant, each net of a
documented one-way cost on the implied turnover (shorts pay borrow).

**No look-ahead**: the signal at the close of day t uses only tone available at
day t; the position earns the t->t+1 return.

**Proxy caveat**: the tone proxy is a hand-curated, in-sample read of known
macro mood swings -- it can only *over*-state predictability.  Any positive
result is an upper bound, named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def tone_sign_signal(df: pd.DataFrame, tone_col: str = "tone") -> pd.Series:
    """+1 when today's tone is positive (good news), -1 when negative.

    The classic "trade the mood" rule.  Position formed at today's close, held
    through tomorrow.
    """
    s = np.sign(df[tone_col]).replace(0, 0.0)
    return s.rename("sig_sign").astype(float)


def tone_tilt_signal(
    df: pd.DataFrame, tone_col: str = "tone", clip: float = 2.0
) -> pd.Series:
    """Continuous tilt = clipped tone z-score (scaled to unit max gross).

    A proportional version: lean harder into stronger tone readings, capped at
    ``clip`` so a single extreme month does not dominate the book.
    """
    s = df[tone_col].clip(-clip, clip) / clip
    return s.rename("sig_tilt").astype(float)


# ---------------------------------------------------------------------------
# Strategy returns
# ---------------------------------------------------------------------------
def timing_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    fwd_col: str = "fwd_ret",
) -> pd.Series:
    """Daily strategy return = signal_t * fwd_ret_{t+1}.

    The signal (formed at the close of day t) earns the t->t+1 return.  This is
    the one-day execution-lag convention.
    """
    common = signal.index.intersection(df.index)
    r = signal.loc[common].astype(float) * df.loc[common, fwd_col].astype(float)
    return r.dropna().rename("strat_ret")


def net_of_costs(
    signal: pd.Series,
    gross: pd.Series,
    one_way_bps: float = 1.0,
    borrow_bps_per_day: float = 0.04,
) -> pd.Series:
    """Subtract turnover cost (one-way x |position change| x NAV) and short borrow.

    The implied position is ``signal``; turnover is ``|signal_t - signal_{t-1}|``.
    Each unit of turnover costs ``one_way_bps``.  Days held short additionally
    pay a daily borrow of ``borrow_bps_per_day`` on the short notional (|neg part|).

    Costs are charged on NAV (the position is in units of NAV), matching the
    one-way x NAV convention with shorts paying borrow.
    """
    sig = signal.reindex(gross.index).fillna(0.0).astype(float)
    turnover = sig.diff().abs().fillna(sig.abs())
    trade_cost = turnover * one_way_bps * 1e-4
    short_notional = (-sig).clip(lower=0.0)
    borrow_cost = short_notional * borrow_bps_per_day * 1e-4
    return (gross - trade_cost - borrow_cost).rename("net_ret")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods: int = 252) -> dict:
    """Headline statistics for a daily return series (Newey-West HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat
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
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "sharpe_ann": sr * np.sqrt(periods),
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "mean_ann": mu * periods,
        "vol_ann": std * np.sqrt(periods),
        "n": n,
    }


def predictive_regression(
    df: pd.DataFrame,
    tone_col: str = "tone",
    target_col: str = "fwd_ret",
) -> dict:
    """OLS of the target return on the tone z-score, with a HAC-style t-stat.

    Regresses ``target`` (e.g. next-day return) on a constant and ``tone``.
    Returns slope ``beta``, its Newey-West t-stat, the R-squared, and n.

    Run with ``target_col='fwd_ret'`` for the honest *next-day* test, or
    ``target_col='ret'`` for the (mechanical) *same-day* fit -- the gap between
    them is the whole story.
    """
    sub = df[[tone_col, target_col]].dropna()
    x = sub[tone_col].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"beta": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef

    # Newey-West HAC covariance for the slope
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(cov[1, 1]))
    beta = float(coef[1])
    tstat = beta / se_beta if se_beta > 0 else float("nan")

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"beta": beta, "tstat": float(tstat), "r2": float(r2), "n": int(n)}
