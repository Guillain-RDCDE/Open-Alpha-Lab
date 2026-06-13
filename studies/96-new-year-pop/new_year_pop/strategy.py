"""The analysis and its honest controls — Study 96 (New-Year-Pop).

The folk claim: *"small-cap stocks systematically beat large caps in January — the
'January effect' — so tilt to small caps for the turn of the year."* (Keim 1983;
Reinganum 1983.) We work on the **monthly small-minus-large return spread** and ask the
sharp question:

- **Is January's spread significantly higher than the rest of the year?** — a HAC *t*
  on the contrast (January small-minus-large mean minus the other-eleven-months mean),
  plus a Wilson interval on the hit-rate that January's small leg beats the large leg.
- **Did it decay after publication?** — a pre/post split with a test of the *difference*
  in January's spread between the two eras (Keim/Reinganum publicised the effect in
  1983; the famous decay is post-publication). The split point is justified, not snooped.
- **Could you trade it?** — a seasonal timer that holds small caps in January and large
  caps the other eleven months, against buy-and-hold of each leg, net of a switch cost.

Execution: a calendar month is a *calendar-known* rule — you know it's January before it
starts — so no execution lag is applied (the desk's convention; lags are for signals read
off the tape). The seasonal timer rebalances at most twice a year (into small for Jan,
back to large for Feb), and we charge a switch cost on each of those two turns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# The spread and its January-vs-rest contrast
# ---------------------------------------------------------------------------
def spread(frame: pd.DataFrame) -> pd.Series:
    """Monthly small-minus-large return spread (the object the whole study studies)."""
    return (frame["small"] - frame["large"]).rename("spread")


def january_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True on January bars."""
    return np.asarray(index.month == 1)


def contrast(frame: pd.DataFrame) -> dict:
    """January's small-minus-large spread vs the other eleven months.

    Returns the January mean spread, the rest-of-year mean spread, their difference, the
    HAC *t* on a regression of the spread on a January dummy (i.e. on the difference), and
    the Wilson interval on the hit-rate that the small leg beats the large leg in January.
    """
    sp = spread(frame)
    jan = january_mask(frame.index)
    jan_sp = sp[jan]
    rest_sp = sp[~jan]

    diff = float(jan_sp.mean() - rest_sp.mean())
    # HAC t on the contrast: regress spread on a January dummy, HAC SE on the slope.
    t_contrast = _hac_dummy_tstat(sp.to_numpy(), jan)

    wins = int((jan_sp > 0).sum())
    n_jan = int(jan_sp.size)
    lo, hi = wilson_interval(wins, n_jan)

    return {
        "jan_mean": float(jan_sp.mean()),
        "rest_mean": float(rest_sp.mean()),
        "diff": diff,
        "t_contrast": t_contrast,
        "jan_wins": wins,
        "n_jan": n_jan,
        "jan_hit_rate": wins / n_jan if n_jan else float("nan"),
        "wilson_lo": lo,
        "wilson_hi": hi,
    }


def monthly_means(frame: pd.DataFrame) -> pd.Series:
    """Mean small-minus-large spread by calendar month (Jan..Dec), for the bar chart."""
    sp = spread(frame)
    g = sp.groupby(frame.index.month).mean()
    g.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in g.index]
    return g.rename("mean_spread")


# ---------------------------------------------------------------------------
# Sub-period decay: a test of the DIFFERENCE in January's spread across eras
# ---------------------------------------------------------------------------
def subperiod_january(frame: pd.DataFrame, split_year: int = 2000) -> dict:
    """January small-minus-large spread, early vs late, with a test of the difference.

    Splits Januaries into pre-``split_year`` and from-``split_year`` cohorts and runs a
    Welch two-sample *t*-test on the **difference** in mean January spread between the two
    eras (the famous post-publication decay). ``split_year`` is a justified break — Keim
    and Reinganum publicised the effect in 1983, and the literature dates the strong decay
    to the late 1980s/1990s — not a snooped optimum.
    """
    sp = spread(frame)
    jan = january_mask(frame.index)
    jan_sp = sp[jan]
    years = jan_sp.index.year
    early = jan_sp[years < split_year].to_numpy()
    late = jan_sp[years >= split_year].to_numpy()

    t_diff, dof = _welch_t(early, late)
    return {
        "split_year": split_year,
        "early_mean": float(np.mean(early)) if early.size else float("nan"),
        "late_mean": float(np.mean(late)) if late.size else float("nan"),
        "early_n": int(early.size),
        "late_n": int(late.size),
        "diff": float(np.mean(early) - np.mean(late)) if early.size and late.size else float("nan"),
        "t_diff": t_diff,
        "dof": dof,
    }


# ---------------------------------------------------------------------------
# The seasonal timer (tradability) vs buy-and-hold of each leg
# ---------------------------------------------------------------------------
def seasonal_timer(frame: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    """Hold small in January, large the other eleven months — net of a switch cost.

    Two turns a year (into small for Jan, back to large for Feb), each charged ``cost_bps``
    one-way on NAV. Compared against buy-and-hold of the small leg and of the large leg.
    Returns net stats for all three arms and the daily/monthly net series of the timer.
    """
    s = frame["small"].to_numpy()
    l = frame["large"].to_numpy()
    jan = january_mask(frame.index)

    # Position: 1.0 = in small, 0.0 = in large. Calendar-known, so no lag.
    in_small = jan.astype(float)
    gross = np.where(jan, s, l)
    # A switch happens whenever the leg changes month to month.
    switch = np.abs(np.diff(in_small, prepend=in_small[0]))
    switch[0] = 0.0
    net = gross - switch * cost_bps * 1e-4

    timer = _stats(pd.Series(net, index=frame.index))
    small_bh = _stats(frame["small"])
    large_bh = _stats(frame["large"])
    return {
        "timer": timer,
        "small_bh": small_bh,
        "large_bh": large_bh,
        "net": pd.Series(net, index=frame.index),
    }


def _stats(monthly_ret: pd.Series) -> dict:
    r = monthly_ret.astype(float)
    equity = (1.0 + r).cumprod()
    n = len(r)
    years = n / MONTHS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if r.std() > 0 else float("nan")
    eq = equity.to_numpy()
    max_dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "final": float(equity.iloc[-1]), "equity": equity, "n": n,
    }


# ---------------------------------------------------------------------------
# Inference helpers — HAC t, Welch t, Wilson interval (local, no quantlab dep)
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _hac_dummy_tstat(y: np.ndarray, dummy: np.ndarray, lags: int | None = None) -> float:
    """HAC t-stat on the slope of ``y = a + b * dummy`` — i.e. on the group-mean difference.

    Equivalent to a HAC test that the January mean minus the rest-of-year mean is zero.
    Uses Newey-West weights on the OLS residuals.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(dummy, dtype=float)
    n = y.size
    if n <= 5:
        return float("nan")
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    # Newey-West meat matrix S = sum_k w_k (Gamma_k + Gamma_k').
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        u = X * resid[:, None]
        G = u[k:].T @ u[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = np.sqrt(cov[1, 1])
    return float(beta[1] / se_b) if se_b > 0 else float("nan")


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch two-sample t-stat and dof for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return float(t), float(dof)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(centre - half), float(centre + half)
