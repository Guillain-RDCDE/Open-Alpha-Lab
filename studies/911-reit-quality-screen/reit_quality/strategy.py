"""Strategy & inference for Study 911 — REIT Quality Screen.

The question: does a **quality REIT screen** — hold the durable-income equity sleeve
(residential / broad property), screen *out* the leveraged mortgage-REIT carry — beat the
**broad REIT index** on a *risk-adjusted, net-of-cost* basis? Two legs to separate:

  1. **The durable-income tilt.** Quality equity sleeve (REZ, or an equal-weight
     VNQ/REZ/RWR "quality book") vs the broad index (VNQ). The honest test is an
     **excess-vs-excess Sharpe race** (both minus the T-bill leg) plus a HAC *t* on the
     monthly return spread and a bootstrap Sharpe-advantage CI.
  2. **The leveraged-carry trap.** Mortgage REITs (REM) vs the equity sleeve — does the fat
     mortgage-REIT dividend survive on a *total-return* basis, or is it a yield trap with a
     structurally worse Sharpe and deeper drawdowns?

Everything on the real-tape path is a deterministic function of the cached total-return
tape — no RNG. The only randomness lives in the synthetic control (fixed-seed) and in the
block-bootstrap CI (fixed default seed). One documented rebalance lag on the costed book.

Primitives (self-contained so the tests run offline & engine-independent): a Newey-West
HAC *t* on the monthly spread, an annualised excess Sharpe, a paired block-bootstrap
Sharpe-advantage CI, an era cut, a calendar-year table, a daily max-drawdown, and a costed
quality-book spread. (``examples/verify.py`` and the notebooks additionally cross-check
against ``quantlab.stats`` / ``quantlab.analytics``.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CASH

MONTHS = 12
NW_LAGS = 6   # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 230)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple total returns (month-end to month-end), partial month dropped."""
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


def excess(monthly: pd.DataFrame, col: str, rf: str = CASH) -> pd.Series:
    """Monthly excess return of ``col`` over the T-bill ETF proxy."""
    return (monthly[col] - monthly[rf]).dropna()


def common_sample(monthly: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows where every column in ``cols`` has a return (the aligned common window).

    Duplicate names (e.g. a book leg that is also the benchmark) are collapsed to one
    column, so overlapping role lists never trigger a duplicate-axis reindex.
    """
    uniq = list(dict.fromkeys(cols))
    return monthly[uniq].dropna()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def nw_mean_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 — monthly spreads are serially
    correlated, so a plain t overstates significance."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def ann_return(monthly_ret: pd.Series) -> float:
    """Geometric annualised return (%) from monthly simple returns."""
    r = pd.Series(monthly_ret).dropna()
    if len(r) == 0:
        return float("nan")
    return float(((1.0 + r).prod() ** (MONTHS / len(r)) - 1.0) * 100.0)


def ann_vol(monthly_ret: pd.Series) -> float:
    """Annualised volatility (%) from monthly simple returns."""
    return float(pd.Series(monthly_ret).dropna().std(ddof=1) * np.sqrt(MONTHS) * 100.0)


def excess_sharpe(monthly: pd.DataFrame, col: str, rf: str = CASH) -> float:
    """Annualised Sharpe of monthly EXCESS returns (excess-vs-excess race only)."""
    e = excess(monthly, col, rf)
    sd = e.std(ddof=1)
    return float(e.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def sharpe_table(monthly: pd.DataFrame, cols: list[str], rf: str = CASH) -> pd.DataFrame:
    """Per-sleeve ann return / ann vol / excess Sharpe over the common window of ``cols``."""
    sub = common_sample(monthly, list(cols) + [rf])
    rows = {}
    for c in cols:
        rows[c] = {
            "ann_ret_pct": ann_return(sub[c]),
            "ann_vol_pct": ann_vol(sub[c]),
            "excess_sharpe": excess_sharpe(sub, c, rf),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# The durable-income tilt: quality sleeve vs broad index
# --------------------------------------------------------------------------- #
def quality_book(monthly: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Equal-weight monthly return of the quality equity-REIT book (rebalanced monthly)."""
    return monthly[cols].dropna().mean(axis=1)


def spread_stats(monthly: pd.DataFrame, a: str, b: str, lags: int = NW_LAGS) -> dict:
    """Monthly return spread ``a - b``: mean (bps/mo), HAC t, n, over the common window."""
    d = (monthly[a] - monthly[b]).dropna()
    return {"a": a, "b": b, "n": int(len(d)),
            "start": str(d.index.min().date()), "end": str(d.index.max().date()),
            "mean_bps": float(d.mean() * 1e4), "t_nw": nw_mean_t(d.values, lags)}


def book_spread_stats(monthly: pd.DataFrame, book_cols: list[str], bench: str,
                      lags: int = NW_LAGS) -> dict:
    """Monthly spread of the equal-weight quality BOOK minus the benchmark index."""
    sub = common_sample(monthly, list(book_cols) + [bench])
    book = sub[book_cols].mean(axis=1)
    d = (book - sub[bench]).dropna()
    return {"n": int(len(d)), "start": str(d.index.min().date()),
            "end": str(d.index.max().date()),
            "mean_bps": float(d.mean() * 1e4), "t_nw": nw_mean_t(d.values, lags)}


def sharpe_advantage(monthly: pd.DataFrame, a: str, b: str, rf: str = CASH,
                     n_boot: int = 2000, block: int = 6, seed: int = 911) -> dict:
    """Excess-Sharpe advantage of ``a`` over ``b`` with a **paired circular-block
    bootstrap** CI (both legs excess of ``rf``; blocks preserve serial dependence and the
    pairing preserves the cross-sectional correlation). ``frac_neg`` = share of resamples
    where the advantage flips negative — a blunt "could this be zero?" read.
    """
    sub = common_sample(monthly, [a, b, rf])
    ea = (sub[a] - sub[rf]).to_numpy(dtype=float)
    eb = (sub[b] - sub[rf]).to_numpy(dtype=float)
    n = len(ea)
    ann = np.sqrt(MONTHS)

    def _sh(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * ann if sd > 0 else np.nan

    point = float(_sh(ea) - _sh(eb))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        d = _sh(ea[idx]) - _sh(eb[idx])
        if np.isfinite(d):
            diffs[i] = d
    valid = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(valid, [2.5, 97.5]) if valid.size else (np.nan, np.nan)
    return {"a": a, "b": b, "n": n, "sharpe_a": float(_sh(ea)), "sharpe_b": float(_sh(eb)),
            "advantage": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_neg": float((valid < 0).mean()) if valid.size else float("nan"),
            "n_boot_valid": int(valid.size)}


# --------------------------------------------------------------------------- #
# Robustness — era cut
# --------------------------------------------------------------------------- #
def era_spreads(monthly: pd.DataFrame, a: str, b: str, cut: str = "2017-01-01",
                lags: int = NW_LAGS) -> list[dict]:
    """Monthly ``a - b`` spread in the two eras split at ``cut``."""
    d = (monthly[a] - monthly[b]).dropna()
    out = []
    for lo, hi, lbl in [(d.index.min(), pd.Timestamp(cut), f"pre {cut[:7]}"),
                        (pd.Timestamp(cut), d.index.max() + pd.Timedelta(days=1),
                         f"{cut[:7]}+")]:
        seg = d[(d.index >= lo) & (d.index < hi)]
        out.append({"era": lbl, "n": int(len(seg)),
                    "mean_bps": float(seg.mean() * 1e4) if len(seg) else float("nan"),
                    "t_nw": nw_mean_t(seg.values, lags)})
    return out


def calendar_year_table(monthly: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calendar-year total returns (%) per column (compounded within each year)."""
    sub = monthly[cols].dropna()
    yr = (1.0 + sub).groupby(sub.index.year).prod() - 1.0
    return (yr * 100.0).round(2)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def max_drawdown(prices: pd.DataFrame, col: str) -> dict:
    """Max drawdown of a daily total-return price series: depth (%), peak/trough dates."""
    px = prices[col].dropna()
    peak = px.cummax()
    dd = px / peak - 1.0
    trough = dd.idxmin()
    peak_date = px.loc[:trough].idxmax()
    return {"depth_pct": float(dd.min() * 100.0),
            "peak": str(peak_date.date()), "trough": str(trough.date())}


# --------------------------------------------------------------------------- #
# Tradability — cost the quality book
# --------------------------------------------------------------------------- #
def costed_book(monthly: pd.DataFrame, book_cols: list[str], bench: str,
                cost_bps_oneway: float = 5.0, rebal_per_year: float = 12.0,
                lags: int = NW_LAGS) -> dict:
    """Net spread of the equal-weight quality BOOK over the buy-and-hold benchmark index.

    The book is rebalanced ``rebal_per_year`` times a year back to equal weights; each
    rebalance turns over a slice of NAV and pays one-way cost x NAV. We charge a
    conservative flat drag of ``rebal_per_year * turnover_frac * cost`` where the turnover
    fraction per rebalance is bounded by the average absolute weight drift; for an N-name
    equal-weight book of correlated REIT sleeves this is small, so we use a deliberately
    generous ``turnover_frac = 0.10`` (10% of NAV traded each rebalance). The benchmark is
    a single buy-and-hold ETF (one entry, ~0 ongoing turnover). One documented lag: weights
    are set at the prior month-end close and held the next month.
    """
    sub = common_sample(monthly, list(book_cols) + [bench])
    book = sub[book_cols].mean(axis=1)
    n = len(sub)
    years = n / MONTHS
    turnover_frac = 0.10
    drag_annual = rebal_per_year * turnover_frac * cost_bps_oneway / 1e4
    drag_monthly = drag_annual / MONTHS
    gross = (book - sub[bench])
    net = gross - drag_monthly
    return {"n": n, "years": years,
            "gross_bps_mo": float(gross.mean() * 1e4),
            "net_bps_mo": float(net.mean() * 1e4),
            "drag_bps_yr": float(drag_annual * 1e4),
            "t_net": nw_mean_t(net.values, lags),
            "gross_ann_pct": float(gross.mean() * MONTHS * 100),
            "net_ann_pct": float(net.mean() * MONTHS * 100)}


def trap_gap(monthly: pd.DataFrame, trap: str, quality: str, broad: str,
             rf: str = CASH) -> dict:
    """The leveraged-carry trap, quantified: mortgage-REIT (``trap``) vs the quality sleeve
    and the broad index on total return and excess Sharpe, over the common window."""
    sub = common_sample(monthly, [trap, quality, broad, rf])
    return {
        "n": int(len(sub)),
        "start": str(sub.index.min().date()), "end": str(sub.index.max().date()),
        "trap_ann_pct": ann_return(sub[trap]), "trap_sharpe": excess_sharpe(sub, trap, rf),
        "quality_ann_pct": ann_return(sub[quality]),
        "quality_sharpe": excess_sharpe(sub, quality, rf),
        "broad_ann_pct": ann_return(sub[broad]), "broad_sharpe": excess_sharpe(sub, broad, rf),
        "quality_minus_trap_bps": float((sub[quality] - sub[trap]).mean() * 1e4),
        "t_quality_minus_trap": nw_mean_t((sub[quality] - sub[trap]).values),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detectors (machinery proof — never market evidence)
# --------------------------------------------------------------------------- #
def synth_detect(world: pd.DataFrame) -> dict:
    """Run the headline estimators on a synthetic world (BROAD/QUAL/TRAP/CASH).

    Returns the quality-vs-broad Sharpe advantage, the HAC t on the QUAL-BROAD spread, and
    the trap flag (is TRAP's excess Sharpe below BROAD's?)."""
    adv = sharpe_advantage(world, "QUAL", "BROAD", rf="CASH", n_boot=800)
    sp = spread_stats(world, "QUAL", "BROAD")
    sh_broad = excess_sharpe(world, "BROAD", "CASH")
    sh_trap = excess_sharpe(world, "TRAP", "CASH")
    return {"adv": adv["advantage"], "adv_ci_low": adv["ci_low"], "adv_frac_neg": adv["frac_neg"],
            "spread_bps": sp["mean_bps"], "spread_t": sp["t_nw"],
            "sharpe_broad": sh_broad, "sharpe_trap": sh_trap,
            "trap_flagged": bool(sh_trap < sh_broad)}
