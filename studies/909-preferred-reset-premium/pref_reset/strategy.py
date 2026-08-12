"""Strategy + inference for Study 909 — Preferred Reset Premium.

The question: does a **variable-rate** preferred sleeve (VRP, PFFV) deliver a better
*rate-adjusted carry* than a **fixed-rate** preferred sleeve (PFF, PGX, PGF), and does that
edge live only in the **high-rate regime** (2022+) the way the reset story predicts?

Everything is measured excess-of-cash (minus BIL), so a Sharpe *race* compares two
excess-of-cash return streams on the same footing. The core spread is (variable − fixed):
a within-asset-class difference that nets out most of the common credit beta and leaves the
duration/reset difference. Inference is Newey-West (HAC) on the monthly spread — an
overlapping-coupon-reset series is serially correlated, so a plain *t* would overstate — plus
a block-bootstrap CI on the Sharpe advantage and the mean spread. A single documented
execution lag drives the regime-switch tradability test. Costs: one-way bps × NAV × turnover
per rebalance, and borrow on the short leg of the isolation spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 150)


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained; mirror the quantlab HAC/bootstrap)
# --------------------------------------------------------------------------- #
def _label(x) -> str:
    """Date label that works for both a Timestamp and a Period index element."""
    d = getattr(x, "date", None)
    return str(d()) if callable(d) else str(x)


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ann_sharpe(x: np.ndarray) -> float:
    """Annualised Sharpe of a monthly return stream (mean/std × sqrt(12))."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def max_drawdown(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def block_bootstrap_ci(x: np.ndarray, stat, block: int = 6, n_boot: int = 2000,
                       seed: int = 909, alpha: float = 0.05) -> tuple[float, float, float]:
    """Circular block-bootstrap (point, lo, hi) CI for ``stat(sample)`` on a monthly series.

    Blocks preserve the serial correlation of the coupon-reset spread. Returns the point
    estimate on the observed sample and the (alpha/2, 1-alpha/2) percentile CI.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    point = float(stat(x))
    if n < block + 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offs = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offs.reshape(-1)[:n]
        vals[i] = stat(x[sel])
    finite = vals[np.isfinite(vals)]
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# The excess-vs-excess Sharpe race and the (variable − fixed) spread
# --------------------------------------------------------------------------- #
def race_frame(sleeves: pd.DataFrame, var_col: str = "variable",
               fix_col: str = "fixed", cash_col: str = "cash",
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Aligned monthly frame: excess-of-cash variable & fixed returns and their spread.

    ``var_ex`` = variable − cash, ``fix_ex`` = fixed − cash, ``spread`` = variable − fixed
    (cash cancels in the spread). Rows with any NaN member are dropped, so the window begins
    where both sleeves have data (VRP bounds the flagship pair at 2014-06).
    """
    df = pd.DataFrame({
        "var_ex": sleeves[var_col] - sleeves[cash_col],
        "fix_ex": sleeves[fix_col] - sleeves[cash_col],
        "spread": sleeves[var_col] - sleeves[fix_col],
    }).dropna()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def race_stats(rf: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Headline stats for one window: excess Sharpe of each sleeve, the Sharpe advantage,
    and the (variable − fixed) spread with its HAC/one-sample *t* and annualised size.
    """
    ve = rf["var_ex"].to_numpy(float)
    fe = rf["fix_ex"].to_numpy(float)
    sp = rf["spread"].to_numpy(float)
    sh_v, sh_f = ann_sharpe(ve), ann_sharpe(fe)
    return {
        "n": int(len(rf)),
        "start": _label(rf.index.min()) if len(rf) else None,
        "end": _label(rf.index.max()) if len(rf) else None,
        "var_ex_sharpe": sh_v,
        "fix_ex_sharpe": sh_f,
        "sharpe_adv": sh_v - sh_f,
        "spread_bps_mo": float(np.nanmean(sp) * 1e4),
        "spread_ann_pct": float(np.nanmean(sp) * MONTHS * 100),
        "t_nw": newey_west_t(sp, lags),
        "t_1s": one_sample_t(sp),
        "welch_t": welch_t(ve, fe),
        "var_dd": max_drawdown(rf["var_ex"].to_numpy() + 0.0),  # excess-of-cash dd proxy
    }


def sleeve_summary(sleeves: pd.DataFrame, col: str, cash_col: str = "cash",
                   start: str | None = None, end: str | None = None) -> dict:
    """Total-return CAGR / vol / max-drawdown / excess-of-cash Sharpe for one sleeve."""
    s = sleeves[col].dropna()
    cash = sleeves[cash_col].reindex(s.index)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]; cash = cash.reindex(s.index)
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]; cash = cash.reindex(s.index)
    x = s.to_numpy(float)
    ex = (s - cash).to_numpy(float)
    eq = np.cumprod(1.0 + x)
    years = len(x) / MONTHS
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else float("nan")
    return {
        "n": len(x),
        "cagr": cagr,
        "vol_ann": float(np.std(x, ddof=1) * np.sqrt(MONTHS)),
        "max_dd": max_drawdown(x),
        "ex_sharpe": ann_sharpe(ex),
        "ann_ret_pct": float(np.mean(x) * MONTHS * 100),
    }


# --------------------------------------------------------------------------- #
# Era cut and calendar-year table
# --------------------------------------------------------------------------- #
def era_cut(rf: pd.DataFrame, split: str, lags: int = NW_LAGS) -> dict:
    """Split the race frame at ``split`` and report race stats on each half (low/high rate).

    Works for both a DatetimeIndex (real tape) and a PeriodIndex (synthetic world) by casting
    the split point to the index's own type.
    """
    if isinstance(rf.index, pd.PeriodIndex):
        cut = pd.Period(split, freq=rf.index.freq)
    else:
        cut = pd.Timestamp(split)
    early = rf[rf.index < cut]
    late = rf[rf.index >= cut]
    return {"low_rate": race_stats(early, lags), "high_rate": race_stats(late, lags)}


def calendar_year_table(sleeves: pd.DataFrame,
                        cols=("variable", "fixed", "cash")) -> pd.DataFrame:
    """Per-calendar-year total return of each sleeve (compounded monthly)."""
    out = {}
    for c in cols:
        s = sleeves[c].dropna()
        out[c] = s.groupby(s.index.year).apply(lambda g: float(np.prod(1.0 + g) - 1.0))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Tradability
# --------------------------------------------------------------------------- #
def costed_spread(rf: pd.DataFrame, cost_bps_oneway: float = 8.0,
                  borrow_annual_bps: float = 40.0, turnover_per_year: float = 1.0,
                  lags: int = NW_LAGS) -> dict:
    """Long variable / short fixed, dollar-neutral (per $ of long NAV): isolate the spread.

    Gross monthly P&L = ``spread``. Charges: borrow on the short fixed leg (annual bps / 12)
    and one-way cost × NAV on ~``turnover_per_year`` rebalances of BOTH legs per year. Wide
    preferred-ETF spreads justify the 8 bps default one-way. Reports gross and net annualised
    spread with HAC *t*.
    """
    charge_m = (borrow_annual_bps / 1e4 / MONTHS
                + 2 * cost_bps_oneway / 1e4 * turnover_per_year / MONTHS)
    gross = rf["spread"].to_numpy(float)
    net = gross - charge_m
    return {
        "gross_ann_pct": float(np.nanmean(gross) * MONTHS * 100),
        "t_gross": newey_west_t(gross, lags),
        "net_ann_pct": float(np.nanmean(net) * MONTHS * 100),
        "t_net": newey_west_t(net, lags),
        "charge_ann_pct": float(charge_m * MONTHS * 100),
        "net_sharpe": ann_sharpe(net),
    }


def rate_signal(sleeves: pd.DataFrame, lookback: int = 6, cash_col: str = "cash") -> pd.Series:
    """A knowable *rising-rate* signal from the cash leg: the trailing ``lookback``-month
    change in the annualised short yield (12 × cash monthly return). Positive = rates rising.

    Cash is BIL, whose monthly return ≈ short_rate / 12, so 12× it is an observable yield
    proxy; its trailing change is knowable at month-end with no look-ahead.
    """
    yld = sleeves[cash_col] * MONTHS
    return yld - yld.shift(lookback)


def switch_strategy(sleeves: pd.DataFrame, lookback: int = 6,
                    cost_bps_oneway: float = 8.0, cash_col: str = "cash",
                    var_col: str = "variable", fix_col: str = "fixed") -> dict:
    """Hold the VARIABLE sleeve when the PRIOR month-end rising-rate signal > 0, else FIXED.

    ONE execution lag: the signal decided at month-end t−1 selects the sleeve held over
    month t (the cash-yield trend is observable, so this is genuinely knowable). A full
    switch costs 2 × one-way bps (sell one sleeve, buy the other). Compared, net of cost and
    excess-of-cash, against always-fixed and always-variable buy-and-hold.
    """
    df = pd.DataFrame({
        "var": sleeves[var_col], "fix": sleeves[fix_col], "cash": sleeves[cash_col],
        "sig": rate_signal(sleeves, lookback, cash_col).shift(1),
    }).dropna()
    hold_var = (df["sig"] > 0).to_numpy()
    strat = np.where(hold_var, df["var"].to_numpy(), df["fix"].to_numpy())
    switches = int(np.sum(hold_var[1:] != hold_var[:-1]))
    n = len(df)
    cost_total = switches * 2 * cost_bps_oneway / 1e4
    cash = df["cash"].to_numpy()
    strat_net = strat - cost_total / n  # amortise switch cost across the held months
    return {
        "n": n,
        "switches": switches,
        "share_variable": float(hold_var.mean()),
        "switch_ex_sharpe": ann_sharpe(strat_net - cash),
        "switch_net_ann_pct": float(np.mean(strat_net) * MONTHS * 100),
        "always_fixed_ex_sharpe": ann_sharpe(df["fix"].to_numpy() - cash),
        "always_var_ex_sharpe": ann_sharpe(df["var"].to_numpy() - cash),
        "always_fixed_ann_pct": float(np.mean(df["fix"].to_numpy()) * MONTHS * 100),
        "always_var_ann_pct": float(np.mean(df["var"].to_numpy()) * MONTHS * 100),
    }


# --------------------------------------------------------------------------- #
# Synthetic control detector
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the (variable − fixed) spread detector on a synthetic world.

    Returns the spread NW *t*, the regime-conditional spread means, and the excess-of-cash
    Sharpe advantage — the same machinery the real tape uses, so the planted edge must be
    recovered and the null must stay silent.
    """
    spread = (world["variable"] - world["fixed"]).to_numpy(float)
    ve = (world["variable"] - world["cash"]).to_numpy(float)
    fe = (world["fixed"] - world["cash"]).to_numpy(float)
    reg = world["regime"].to_numpy(float).astype(bool)
    return {
        "t_nw": newey_west_t(spread, lags),
        "spread_ann_pct": float(np.nanmean(spread) * MONTHS * 100),
        "spread_high_ann_pct": float(np.nanmean(spread[reg]) * MONTHS * 100) if reg.any() else float("nan"),
        "spread_low_ann_pct": float(np.nanmean(spread[~reg]) * MONTHS * 100) if (~reg).any() else float("nan"),
        "sharpe_adv": ann_sharpe(ve) - ann_sharpe(fe),
    }
