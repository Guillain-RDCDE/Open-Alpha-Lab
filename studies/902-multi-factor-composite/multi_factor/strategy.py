"""Strategy + inference for Study 902 — Multi-Factor Composite.

The tested object is a **live equal-weight sleeve** of the five iShares single-factor ETFs
(VLUE, QUAL, MTUM, USMV, SIZE), rebalanced monthly back to equal weight, versus **SPY**.
The practitioner's pitch is diversification — single factors take turns working, so a blend
carries less factor-timing risk than any one sleeve. We ask the harder question: does the
blend also buy a **risk-adjusted advantage over the market**, on the excess-of-cash (minus
BIL) Sharpe, net of the rebalancing turnover the blend actually pays?

Everything is deterministic (fixed seeds); pure numpy + pandas + statsmodels-free HAC.
Reuses ``quantlab.stats.sharpe_ci_bootstrap`` where a single-series Sharpe CI is wanted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# HAC / inference primitives
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 on a monthly series."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 6:
        return float("nan")
    e = v - v.mean()
    s0 = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s0 += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s0 / n)
    return float(v.mean() / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def excess(monthly: pd.DataFrame, col: pd.Series | str, rf: str = "BIL") -> pd.Series:
    """Monthly return in excess of the tradable cash ETF (BIL)."""
    s = monthly[col] if isinstance(col, str) else col
    return (s - monthly[rf]).dropna()


def ann_return(monthly_ret: pd.Series) -> float:
    r = monthly_ret.dropna()
    if len(r) == 0:
        return float("nan")
    return ((1.0 + r).prod() ** (MONTHS / len(r)) - 1.0) * 100.0


def sharpe_excess(ret: pd.Series, cash: pd.Series) -> float:
    """Annualised Sharpe of monthly EXCESS-of-cash returns."""
    df = pd.concat([ret, cash], axis=1, sort=False).dropna()
    e = df.iloc[:, 0] - df.iloc[:, 1]
    return float(e.mean() / e.std(ddof=1) * np.sqrt(MONTHS)) if e.std(ddof=1) > 0 else float("nan")


def max_drawdown(monthly_ret: pd.Series) -> float:
    r = monthly_ret.dropna()
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def ann_stats(ret: pd.Series, cash: pd.Series) -> dict:
    """CAGR, annualised vol, excess-of-cash Sharpe, max drawdown on monthly returns."""
    df = pd.concat([ret, cash], axis=1, sort=False).dropna()
    r, c = df.iloc[:, 0], df.iloc[:, 1]
    n = len(r)
    cagr = float((1.0 + r).prod() ** (MONTHS / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(MONTHS))
    e = r - c
    sharpe = float(e.mean() / e.std(ddof=1) * np.sqrt(MONTHS)) if e.std(ddof=1) > 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": max_drawdown(r),
            "n_months": n}


# --------------------------------------------------------------------------- #
# The composite sleeve
# --------------------------------------------------------------------------- #
def _weighted_composite(R: pd.DataFrame, W: np.ndarray, cost_bps: float) -> pd.DataFrame:
    """Core engine: monthly composite return with a target-weight matrix ``W``.

    ``R`` is the (n x K) member-return frame on the common window; ``W`` is the (n x K)
    matrix of target weights for each month (rows sum to 1). The month-``t`` gross return
    is ``sum_i W[t,i]·R[t,i]``. Between months the held weights **drift** with returns;
    the rebalance at the START of month ``t`` trades from month ``t-1``'s drifted weights
    to ``W[t]``, so the month-``t`` one-sided-sum turnover is ``sum_i |W[t,i] − drift[t-1,i]|``
    and the cost charged to month ``t`` is ``turnover · cost_bps`` (one-way, long-only, no
    borrow). Month 0 carries no rebalancing cost (the initial buy is common to any
    buy-and-hold benchmark and is not the incremental drag under test).
    """
    r = R.to_numpy(dtype=float)
    n, K = r.shape
    gross = (W * r).sum(axis=1)
    grow = 1.0 + r
    held = W * grow
    drift = held / held.sum(axis=1, keepdims=True)  # post-return weights, end of month t
    turnover = np.zeros(n)
    turnover[1:] = np.abs(W[1:] - drift[:-1]).sum(axis=1)  # trade at start of month t
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover, "cost": cost},
                        index=R.index)


def equal_weight_composite(monthly: pd.DataFrame, tickers: list[str],
                           cost_bps: float = 2.0) -> pd.DataFrame:
    """Equal-weight (1/K), monthly-rebalanced composite over the members' common window."""
    R = monthly[tickers].dropna()
    K = R.shape[1]
    W = np.full((len(R), K), 1.0 / K)
    return _weighted_composite(R, W, cost_bps)


def inverse_vol_composite(monthly: pd.DataFrame, tickers: list[str],
                          lookback: int = 12, cost_bps: float = 2.0) -> pd.DataFrame:
    """Inverse-trailing-vol weighted composite (robustness alt to equal weight).

    Weight_i for month ``t`` ∝ 1/σ_i, where σ_i is the trailing ``lookback``-month
    return volatility known at the close of ``t-1`` (one ``shift`` — point-in-time, no
    look-ahead). Rows before a full lookback fall back to equal weight.
    """
    R = monthly[tickers].dropna()
    vol = R.rolling(lookback, min_periods=lookback).std().shift(1)
    inv = 1.0 / vol
    W = inv.div(inv.sum(axis=1), axis=0)
    eq = np.full(R.shape[1], 1.0 / R.shape[1])
    W = np.array(W.to_numpy(dtype=float), copy=True)
    bad = ~np.isfinite(W).all(axis=1)
    W[bad] = eq
    return _weighted_composite(R, W, cost_bps)


# --------------------------------------------------------------------------- #
# The race: composite vs SPY, excess-of-cash
# --------------------------------------------------------------------------- #
def sharpe_race(comp: pd.Series, spy: pd.Series, cash: pd.Series) -> dict:
    """Excess-of-cash Sharpe race: composite vs SPY, both minus BIL.

    Reports each leg's excess Sharpe, the advantage (comp − SPY), the annualised vols,
    and the HAC *t* on the monthly **active** return (composite − SPY; cash cancels).
    """
    df = pd.concat([comp, spy, cash], axis=1, sort=False).dropna()
    c, s, k = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    ce, se = c - k, s - k
    sh_c = float(ce.mean() / ce.std(ddof=1) * np.sqrt(MONTHS))
    sh_s = float(se.mean() / se.std(ddof=1) * np.sqrt(MONTHS))
    active = (c - s).to_numpy(dtype=float)
    return {
        "n_months": len(df),
        "sharpe_comp": sh_c, "sharpe_spy": sh_s, "sharpe_adv": sh_c - sh_s,
        "vol_comp": float(c.std(ddof=1) * np.sqrt(MONTHS)),
        "vol_spy": float(s.std(ddof=1) * np.sqrt(MONTHS)),
        "active_bps": float(np.mean(active) * 1e4),
        "t_active_nw": newey_west_t(active, lags=6),
        "t_active_1s": one_sample_t(active),
        "win_rate": float((active > 0).mean()),
    }


def adv_bootstrap_ci(comp: pd.Series, spy: pd.Series, cash: pd.Series,
                     n_boot: int = 4000, block: int = 6, seed: int = 902) -> dict:
    """Paired moving-block bootstrap 95% CI on the Sharpe ADVANTAGE (comp − SPY).

    Joint monthly rows (comp, spy, cash) are resampled in blocks so the serial and the
    cross correlation survive; the excess-Sharpe advantage is recomputed per draw. The
    share of draws with a *negative* advantage is the blunt "could the advantage be ≤ 0?"
    read.
    """
    df = pd.concat([comp, spy, cash], axis=1, sort=False).dropna()
    c = df.iloc[:, 0].to_numpy(dtype=float)
    s = df.iloc[:, 1].to_numpy(dtype=float)
    k = df.iloc[:, 2].to_numpy(dtype=float)
    n = len(df)
    ann = np.sqrt(MONTHS)

    def adv(ci, si, ki):
        ce, se = ci - ki, si - ki
        sc = ce.mean() / ce.std(ddof=1) if ce.std(ddof=1) > 0 else np.nan
        ss = se.mean() / se.std(ddof=1) if se.std(ddof=1) > 0 else np.nan
        return (sc - ss) * ann

    obs = adv(c, s, k)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[i] = adv(c[idx], s[idx], k[idx])
    v = draws[np.isfinite(draws)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"obs": float(obs), "lo": float(lo), "hi": float(hi),
            "frac_negative": float((v < 0).mean()), "n_boot": int(v.size), "block": block}


def era_split(comp: pd.Series, spy: pd.Series, cash: pd.Series) -> dict:
    """Split the common window in half; report the Sharpe advantage in each era."""
    df = pd.concat([comp, spy, cash], axis=1, sort=False).dropna()
    mid = len(df) // 2
    out = {}
    for name, seg in (("early", df.iloc[:mid]), ("late", df.iloc[mid:])):
        r = sharpe_race(seg.iloc[:, 0], seg.iloc[:, 1], seg.iloc[:, 2])
        out[name] = {"sharpe_adv": r["sharpe_adv"], "active_bps": r["active_bps"],
                     "t_active_nw": r["t_active_nw"], "n_months": r["n_months"],
                     "start": str(seg.index.min().date()), "end": str(seg.index.max().date())}
    return out


def calendar_year_table(comp: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Per-calendar-year compounded returns (%) for the composite and SPY."""
    df = pd.concat([comp.rename("composite"), spy.rename("SPY")], axis=1, sort=False).dropna()
    def comp_year(g):
        return (1.0 + g).prod() - 1.0
    tbl = df.groupby(df.index.year).apply(lambda g: g.apply(comp_year)) * 100.0
    tbl["diff"] = tbl["composite"] - tbl["SPY"]
    return tbl


# --------------------------------------------------------------------------- #
# Diversification / factor-timing-risk metrics
# --------------------------------------------------------------------------- #
def single_sleeve_stats(monthly: pd.DataFrame, tickers: list[str], cash: pd.Series,
                        spy: pd.Series) -> pd.DataFrame:
    """Per-single-factor ETF stats on the COMMON blend window (CAGR, vol, excess Sharpe,
    active-vs-SPY NW t, max drawdown) — the raw material the blend diversifies over."""
    R = monthly[tickers].dropna()
    rows = {}
    for tk in tickers:
        r = R[tk]
        st = ann_stats(r, cash.reindex(R.index))
        act = (r - spy.reindex(R.index)).dropna().to_numpy()
        rows[tk] = {"cagr_pct": st["cagr"] * 100, "vol_pct": st["vol"] * 100,
                    "sharpe": st["sharpe"], "maxdd_pct": st["maxdd"] * 100,
                    "active_t_nw": newey_west_t(act, lags=6)}
    return pd.DataFrame(rows).T


def factor_timing_risk(monthly: pd.DataFrame, tickers: list[str], comp_gross: pd.Series,
                       cash: pd.Series, spy: pd.Series) -> dict:
    """Quantify the diversification pitch.

    * ``mean_single_vol`` / ``comp_vol`` — the blend's annualised vol vs the average of
      the single sleeves' (the pure diversification of *level* risk).
    * ``mean_single_sharpe`` / ``comp_sharpe`` — mean single excess Sharpe vs the blend's.
    * ``annual_dispersion`` — the average cross-sleeve standard deviation of *calendar-year*
      returns (how far apart the single factors land each year — the timing risk you eat if
      you pick one) vs ``comp_annual_sd`` (the blend's own year-to-year sd).
    """
    R = monthly[tickers].dropna()
    idx = R.index
    cash_a, spy_a = cash.reindex(idx), spy.reindex(idx)
    single_vols = [R[tk].std(ddof=1) * np.sqrt(MONTHS) for tk in tickers]
    single_sh = [sharpe_excess(R[tk], cash_a) for tk in tickers]
    yr = R.groupby(idx.year).apply(lambda g: (1.0 + g).prod() - 1.0) * 100.0
    disp = yr.std(axis=1, ddof=1).mean()  # avg cross-sleeve dispersion per year
    comp_yr = comp_gross.reindex(idx).groupby(idx.year).apply(lambda g: (1.0 + g).prod() - 1.0) * 100.0
    return {
        "comp_vol_pct": float(comp_gross.reindex(idx).std(ddof=1) * np.sqrt(MONTHS) * 100),
        "mean_single_vol_pct": float(np.mean(single_vols) * 100),
        "min_single_vol_pct": float(np.min(single_vols) * 100),
        "comp_sharpe": sharpe_excess(comp_gross.reindex(idx), cash_a),
        "mean_single_sharpe": float(np.mean(single_sh)),
        "best_single_sharpe": float(np.max(single_sh)),
        "spy_sharpe": sharpe_excess(spy_a, cash_a),
        "annual_cross_dispersion_pct": float(disp),
        "comp_annual_sd_pct": float(comp_yr.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof only)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, cost_bps: float = 0.0) -> dict:
    """Run the composite-vs-benchmark race on a synthetic world.

    The sleeve members are the ``F*`` columns; the benchmark is ``SPY``; cash is ``cash``.
    Returns the Sharpe advantage and its active NW *t* — must stay ~0 on the ``edge=0``
    null and light up positive when a per-annum blend edge is planted.
    """
    members = [c for c in world.columns if c.startswith("F")]
    comp = _weighted_composite(world[members], np.full((len(world), len(members)),
                                                       1.0 / len(members)), cost_bps)
    race = sharpe_race(comp["net"], world["SPY"], world["cash"])
    return {"sharpe_adv": race["sharpe_adv"], "t_active_nw": race["t_active_nw"],
            "active_bps": race["active_bps"], "n_months": race["n_months"]}
