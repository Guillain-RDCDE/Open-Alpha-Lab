"""Strategy + inference for Study 895 — Defensive Momentum.

The claim: momentum (MTUM) crashes in violent reversals; min-vol (USMV) softens them, so
a blend — "momentum without the crashes" — should deliver a **higher excess-of-cash
Sharpe** and **shallower crash drawdowns** than MTUM alone, net of costs.

What we build and test, on monthly total returns (excess = minus the BIL cash leg):

* **Two blends.** A fixed **50/50** monthly-rebalanced blend, and an **inverse-vol**
  blend whose weights come from each sleeve's trailing 12-month realized vol, formed on
  information known at the close of month *t−1* (one ``lag``) and applied to month *t* —
  no look-ahead.
* **The excess-vs-excess Sharpe race.** Both legs excess-of-cash; the Sharpe **advantage**
  of a blend over a sleeve, the **Newey-West (HAC) *t*** on the monthly return difference,
  and a **block-bootstrap CI** on the Sharpe advantage.
* **Crash geometry.** Max drawdown, the drawdown curve, a calendar-year table and the
  drawdown suffered in named crash windows (2020 COVID, the 2022 bear).
* **Era cut.** The Sharpe advantage in each half of the sample (has it held up?).
* **Costs.** A costed net series: one-way cost × the turnover the rebalance actually
  trades (the blend is long-only, so there is no borrow leg).

Everything is deterministic (fixed seeds); pure numpy + pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained; mirror the desk's quantlab helpers)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """Plain one-sample t of the mean vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) t of mean(a) − mean(b)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    var = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * float(u[k:] @ u[:-k]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Excess-of-cash returns
# --------------------------------------------------------------------------- #
def excess(monthly: pd.DataFrame, col: str | pd.Series, cash: str = "BIL") -> pd.Series:
    """Monthly return minus the cash leg (excess-of-cash). ``col`` is a column name or
    a return series to net; ``cash`` is the cash column in ``monthly``."""
    r = monthly[col] if isinstance(col, str) else col
    df = pd.concat([r.rename("r"), monthly[cash].rename("cash")], axis=1).dropna()
    return (df["r"] - df["cash"]).rename(
        col if isinstance(col, str) else getattr(col, "name", "series"))


# --------------------------------------------------------------------------- #
# The blends
# --------------------------------------------------------------------------- #
def inv_vol_weights(sleeves: pd.DataFrame, lookback: int = 12, lag: int = 1) -> pd.DataFrame:
    """Inverse-trailing-vol target weights, shifted by ``lag`` so month-*t* weights use
    only returns known through month *t−1* (no look-ahead)."""
    vol = sleeves.rolling(lookback, min_periods=lookback).std()
    iv = 1.0 / vol
    w = iv.div(iv.sum(axis=1), axis=0)
    return w.shift(lag)


def blend_returns(sleeves: pd.DataFrame, target_w: pd.DataFrame) -> pd.DataFrame:
    """Gross blend return + realized turnover, given per-month TARGET weights.

    ``target_w`` (rows sum to 1, already lagged so they are known at the start of month
    *t*) is applied to month-*t* sleeve returns. Turnover on month *t* is the L1 distance
    between the target weights and the weights that drifted in from the previous month's
    target after realizing returns — i.e. exactly what the rebalance must trade. The first
    live month establishes the book (turnover 1.0).
    """
    df = pd.concat([sleeves, target_w.add_suffix("_w")], axis=1).dropna()
    cols = list(sleeves.columns)
    r = df[cols]
    w = df[[c + "_w" for c in cols]]
    w.columns = cols
    gross = (r * w).sum(axis=1)
    drift = w.shift(1) * (1.0 + r)
    drift = drift.div(drift.sum(axis=1), axis=0)
    turnover = (w - drift).abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = 1.0
    return pd.DataFrame({"gross": gross, "turnover": turnover})


def fixed_blend(sleeves: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
    """Fixed monthly-rebalanced blend: ``w`` in the first sleeve, ``1−w`` in the second."""
    cols = list(sleeves.columns)
    tw = pd.DataFrame({cols[0]: w, cols[1]: 1.0 - w}, index=sleeves.index)
    return blend_returns(sleeves, tw)


def vol_weighted_blend(sleeves: pd.DataFrame, lookback: int = 12, lag: int = 1) -> pd.DataFrame:
    """Inverse-vol blend (weights formed at *t−1*, applied at *t*)."""
    return blend_returns(sleeves, inv_vol_weights(sleeves, lookback, lag))


def apply_costs(blend: pd.DataFrame, cost_bps_oneway: float = 3.0) -> pd.Series:
    """Net monthly blend return = gross − one-way cost × turnover (long-only, no borrow)."""
    return blend["gross"] - (cost_bps_oneway / 1e4) * blend["turnover"]


# --------------------------------------------------------------------------- #
# Performance / crash geometry
# --------------------------------------------------------------------------- #
def ann_stats(total_ret: pd.Series, cash: pd.Series) -> dict:
    """CAGR, annualised vol, excess-of-cash Sharpe and max drawdown on monthly returns."""
    df = pd.concat([total_ret.rename("r"), cash.rename("c")], axis=1).dropna()
    r, c = df["r"], df["c"]
    n = len(r)
    wealth = float((1.0 + r).prod())
    cagr = wealth ** (MONTHS / n) - 1.0 if n else float("nan")
    vol = float(r.std() * np.sqrt(MONTHS))
    ex = r - c
    sharpe = float(ex.mean() / ex.std() * np.sqrt(MONTHS)) if ex.std() > 0 else float("nan")
    curve = (1.0 + r).cumprod()
    dd = float((curve / curve.cummax() - 1.0).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd,
            "wealth": wealth, "n_months": n}


def drawdown_curve(total_ret: pd.Series) -> pd.Series:
    """Running peak-to-trough drawdown of the cumulative total-return curve."""
    curve = (1.0 + total_ret.dropna()).cumprod()
    return curve / curve.cummax() - 1.0


def window_drawdown(total_ret: pd.Series, start: str, end: str) -> float:
    """Worst peak-to-trough drawdown inside a named calendar window (e.g. a crash)."""
    r = total_ret.dropna()
    r = r[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    if len(r) < 2:
        return float("nan")
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def calendar_year_returns(total_ret: pd.Series) -> pd.Series:
    """Calendar-year total returns from monthly returns."""
    r = total_ret.dropna()
    return r.groupby(r.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))


# --------------------------------------------------------------------------- #
# The excess-vs-excess Sharpe race (the headline test)
# --------------------------------------------------------------------------- #
def _sharpe(ex: np.ndarray) -> float:
    ex = ex[~np.isnan(ex)]
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def sharpe_advantage(a_ex: pd.Series, b_ex: pd.Series, nw_lags: int = 6) -> dict:
    """Excess-Sharpe advantage of series ``a`` over series ``b`` (both already excess-of-
    cash), plus the Newey-West *t* on the monthly return DIFFERENCE ``a − b``.

    The Sharpe advantage answers "is the blend a better risk-adjusted deal?"; the HAC *t*
    on the mean difference answers "is that gap distinguishable from zero?".
    """
    df = pd.concat([a_ex.rename("a"), b_ex.rename("b")], axis=1).dropna()
    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    diff = a - b
    return {"sharpe_a": _sharpe(a), "sharpe_b": _sharpe(b),
            "sharpe_adv": _sharpe(a) - _sharpe(b),
            "diff_bps": float(diff.mean() * 1e4),
            "t_nw": newey_west_t(diff, nw_lags),
            "t_1s": one_sample_t(diff), "n": len(diff)}


def bootstrap_sharpe_adv(a_ex: pd.Series, b_ex: pd.Series, n_draws: int = 2000,
                         block: int = 6, seed: int = 895) -> dict:
    """Moving-block bootstrap CI for the excess-Sharpe advantage (a − b).

    Joint monthly rows are resampled in blocks so the serial and cross correlation
    survive; the Sharpe advantage is recomputed per draw. Deterministic (fixed seed).
    """
    df = pd.concat([a_ex.rename("a"), b_ex.rename("b")], axis=1).dropna()
    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    n = len(a)
    obs = _sharpe(a) - _sharpe(b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[i] = _sharpe(a[idx]) - _sharpe(b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"obs": float(obs), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((draws > 0).mean()), "n_draws": n_draws, "block": block}


def era_split(a_ex: pd.Series, b_ex: pd.Series, cut: str | None = None) -> dict:
    """Sharpe advantage (a − b) in each half of the sample.

    ``cut`` is a date; ``None`` splits at the median month. Returns the advantage, the
    return-difference mean and its NW *t* for the early and late eras.
    """
    df = pd.concat([a_ex.rename("a"), b_ex.rename("b")], axis=1).dropna()
    if cut is None:
        mid = df.index[len(df) // 2]
    else:
        mid = pd.Timestamp(cut)
    out = {}
    for name, sub in (("early", df[df.index < mid]), ("late", df[df.index >= mid])):
        a = sub["a"].to_numpy(dtype=float); b = sub["b"].to_numpy(dtype=float)
        out[name] = {"n": len(sub), "start": str(sub.index.min().date()) if len(sub) else "",
                     "end": str(sub.index.max().date()) if len(sub) else "",
                     "sharpe_adv": _sharpe(a) - _sharpe(b),
                     "diff_bps": float((a - b).mean() * 1e4) if len(sub) else float("nan"),
                     "t_nw": newey_west_t(a - b, 6)}
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, w: float = 0.5) -> dict:
    """Run the headline blend-vs-MTUM race on a synthetic sleeves frame.

    ``world`` has MTUM / USMV / SPY / BIL columns (``data.synthetic_sleeves``). Reports the
    fixed-blend excess-Sharpe advantage over MTUM, the NW *t* on the return difference and
    the two max drawdowns — must be ~0 / equal on the null, positive / shallower on a
    planted edge. Machinery proof only — never cited for a stamp.
    """
    sleeves = world[["MTUM", "USMV"]]
    cash = world["BIL"]
    blend = fixed_blend(sleeves, w)
    b_ex = excess(world, "MTUM", "BIL")
    bl_ex = (blend["gross"] - cash).reindex(b_ex.index).dropna()
    b_ex = b_ex.reindex(bl_ex.index)
    race = sharpe_advantage(bl_ex, b_ex)
    return {"sharpe_adv": race["sharpe_adv"], "t_nw": race["t_nw"],
            "diff_bps": race["diff_bps"],
            "blend_maxdd": ann_stats(blend["gross"], cash)["maxdd"],
            "mtum_maxdd": ann_stats(world["MTUM"], cash)["maxdd"],
            "n": race["n"]}
