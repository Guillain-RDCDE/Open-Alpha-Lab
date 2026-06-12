"""The Fed Model timing rule, and the test that exposes it.

The rule: hold equities when the earnings yield E/P exceeds the 10-year Treasury yield, else hold
bonds. The honest tests: (1) does the timing beat buy-and-hold, and (2) does the *bond-yield* term add
anything — i.e. does the Fed signal (E/P − yield) forecast returns better than **E/P alone**? If E/P by
itself does as well, the model's defining ingredient is decoration (Asness 2003: it confuses a nominal
bond yield with a real earnings yield).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def fed_signal(ep: pd.Series, y10: pd.Series) -> pd.Series:
    """The Fed-Model signal: earnings yield minus the 10-year yield (>0 ⇒ 'stocks cheap vs bonds')."""
    return (pd.Series(ep).astype(float) - pd.Series(y10).astype(float)).rename("fed_signal")


def fed_timing(tr: pd.Series, bond: pd.Series, signal: pd.Series) -> pd.Series:
    """Hold equities next month when this month's signal > 0, else bonds. Lagged — no look-ahead."""
    pos = (pd.Series(signal).astype(float).shift(1) > 0).astype(float)
    out = pos * pd.Series(tr).astype(float) + (1.0 - pos) * pd.Series(bond).astype(float).shift(1)
    return out.dropna().rename("fed_timing")


def time_in_stocks(signal: pd.Series) -> float:
    """Fraction of months the rule is in equities (signal > 0) — high when E/P usually beats yields."""
    return float((pd.Series(signal).astype(float).shift(1) > 0).mean())


def buy_hold(tr: pd.Series) -> pd.Series:
    return pd.Series(tr).astype(float).dropna().rename("buy_hold")


def forward_return(tr: pd.Series, horizon: int = 12) -> pd.Series:
    """Cumulative equity return over the next ``horizon`` months (for the predictive-corr test)."""
    r = pd.Series(tr).astype(float)
    fwd = (1.0 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    return fwd.rename("fwd")


def predictive_corr(predictor: pd.Series, tr: pd.Series, horizon: int = 12) -> float:
    """Correlation of a predictor with the next-``horizon``-month equity return (point estimate only —
    see :func:`predictive_corr_race` for the overlap-corrected uncertainty)."""
    fwd = forward_return(tr, horizon)
    df = pd.concat([pd.Series(predictor).astype(float).rename("p"), fwd], axis=1).dropna()
    return float(df["p"].corr(df["fwd"])) if len(df) > 2 else np.nan


def predictive_corr_race(
    p1: pd.Series, p2: pd.Series, tr: pd.Series, horizon: int = 12,
    block: int = 12, n_boot: int = 2000, seed: int = 47,
) -> dict:
    """The horse race done honestly: both predictive correlations *and their difference*, with
    overlap-corrected standard errors.

    The forward-return target overlaps massively at ``horizon=12`` — consecutive months share 11 of
    their 12 forward months, so ~1,500 monthly rows carry only ~125 independent year-length
    observations and naive SEs are ~3.5× too small. A **circular moving-block bootstrap** (blocks of
    ``block`` months, seeded — deterministic) resamples year-sized chunks of the aligned panel, which
    preserves the overlap inside each block. Returns the point correlations, each bootstrap SE, and
    the difference ``corr1 − corr2`` with its SE and 95% percentile CI — the only number that can
    license a "predictor A beats predictor B" sentence.
    """
    fwd = forward_return(tr, horizon)
    df = pd.concat([pd.Series(p1).astype(float).rename("p1"),
                    pd.Series(p2).astype(float).rename("p2"), fwd], axis=1).dropna()
    n = len(df)
    if n < 3 * block:
        return {k: np.nan for k in ("corr1", "corr2", "diff", "se1", "se2", "se_diff",
                                    "diff_lo", "diff_hi", "n", "n_indep")}
    x1, x2, y = df["p1"].to_numpy(), df["p2"].to_numpy(), df["fwd"].to_numpy()
    c1, c2 = float(df["p1"].corr(df["fwd"])), float(df["p2"].corr(df["fwd"]))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n  # circular blocks
        bx1, bx2, by = x1[idx], x2[idx], y[idx]
        draws[b, 0] = np.corrcoef(bx1, by)[0, 1]
        draws[b, 1] = np.corrcoef(bx2, by)[0, 1]
    diffs = draws[:, 0] - draws[:, 1]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "corr1": c1, "corr2": c2, "diff": c1 - c2,
        "se1": float(draws[:, 0].std(ddof=1)), "se2": float(draws[:, 1].std(ddof=1)),
        "se_diff": float(diffs.std(ddof=1)), "diff_lo": float(lo), "diff_hi": float(hi),
        "n": int(n), "n_indep": int(n // horizon),
    }


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
