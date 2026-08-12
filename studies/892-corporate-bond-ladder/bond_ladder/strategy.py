"""Strategy + inference for Study 892 — Corporate-Bond Ladder.

The race: a duration-staggered Treasury **ladder** (a fixed-weight SHY/IEI/IEF/TLT
basket, annually rolled) vs a constant-maturity **fund** (AGG/BND), both on monthly
total returns. We ask three honest questions:

  1. **Excess-vs-excess Sharpe.** Both legs minus the T-bill cash leg (BIL); is the
     ladder's risk-adjusted return really higher? A block-bootstrap CI on each Sharpe and
     on the *difference* says whether any gap is distinguishable from zero.
  2. **Is the ladder-minus-fund premium real?** Newey-West (HAC) *t* on the monthly
     return difference, full-sample and across eras (2007-2015 / 2016-2021 / 2022+). The
     mechanical claim ("HTM ladder pockets a premium the fund can't") predicts a positive,
     era-stable diff. For default-free bonds it should be ~0 once duration is matched.
  3. **Does it survive costs?** The ETF ladder must be rebalanced/rolled annually (it pays
     spreads on SHY/IEI/IEF/TLT); the one-ticker fund is buy-and-hold. A costed net diff.

Inference primitives (newey_west_t / one_sample_t / welch_t / wilson_interval / a block
bootstrap) are local and unit-tested; ``quantlab.stats`` / ``quantlab.analytics`` helpers
are reused where they fit. One documented convention: weights are applied to the SAME
month's returns (a static buy-and-hold basket has no look-ahead); the annual roll is a
turnover charge, not a timing signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DURATION

MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~1.5 * T^(1/3) at T ~ 200)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
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
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_sharpe_ci(x: np.ndarray, n_boot: int = 2000, block: int = 6,
                              seed: int = 892, alpha: float = 0.05,
                              periods: int = MONTHS) -> dict:
    """Circular-block-bootstrap CI for the annualized Sharpe of ``x`` (monthly)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    ann = np.sqrt(periods)
    point = float(x.mean() / x.std(ddof=1) * ann) if x.std(ddof=1) > 0 else float("nan")
    rng = np.random.default_rng(seed)
    nbk = int(np.ceil(n / block))
    off = np.arange(block)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nbk)
        idx = ((starts[:, None] + off[None, :]) % n).ravel()[:n]
        s = x[idx]
        sd = s.std(ddof=1)
        if sd > 0:
            boots.append(s.mean() / sd * ann)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n": n}


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def portfolio_returns(ret: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Fixed-weight (annually-rebalanced) basket return series.

    A static buy-and-hold basket carries no look-ahead: the weight on ETF ``k`` multiplies
    the SAME month's return. (The annual roll is priced as turnover in :func:`costed_race`,
    not as a signal.)
    """
    w = pd.Series(weights, dtype=float)
    cols = list(weights.keys())
    return (ret[cols] * w).sum(axis=1)


def max_drawdown(port: pd.Series) -> float:
    cum = (1.0 + port).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def ann_return(port: pd.Series, periods: int = MONTHS) -> float:
    port = port.dropna()
    if len(port) == 0:
        return float("nan")
    return float((1.0 + port).prod() ** (periods / len(port)) - 1.0)


def excess_sharpe(port: pd.Series, cash: pd.Series, periods: int = MONTHS) -> float:
    ex = (port - cash).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def _blend_duration(weights: dict[str, float]) -> float:
    return float(sum(DURATION[k] * w for k, w in weights.items()))


# --------------------------------------------------------------------------- #
# The head-to-head race
# --------------------------------------------------------------------------- #
def race(ret: pd.DataFrame, ladder_weights: dict[str, float], fund: str = "AGG",
         cash: str = "BIL", nw_lags: int = NW_LAGS, seed: int = 892) -> dict:
    """Ladder vs fund on the joint monthly window: annualized returns, excess-of-cash
    Sharpes (each with a block-bootstrap CI), max drawdowns, and the ladder-minus-fund
    monthly diff with its HAC *t*, one-sample *t*, and diff-Sharpe bootstrap CI."""
    lad = portfolio_returns(ret, ladder_weights)
    fnd = ret[fund]
    csh = ret[cash]
    diff = (lad - fnd).dropna()

    lad_ci = block_bootstrap_sharpe_ci((lad - csh).dropna().values, seed=seed)
    fnd_ci = block_bootstrap_sharpe_ci((fnd - csh).dropna().values, seed=seed)
    diff_ci = block_bootstrap_sharpe_ci(diff.values, seed=seed)

    return {
        "n_months": int(len(diff)),
        "start": str(diff.index.min().date()), "end": str(diff.index.max().date()),
        "ladder_dur": _blend_duration(ladder_weights),
        "fund_dur": DURATION.get(fund, float("nan")),
        "ladder_ann_pct": ann_return(lad) * 100,
        "fund_ann_pct": ann_return(fnd) * 100,
        "ladder_ex_sharpe": excess_sharpe(lad, csh),
        "fund_ex_sharpe": excess_sharpe(fnd, csh),
        "ladder_sharpe_ci": (lad_ci["ci_low"], lad_ci["ci_high"]),
        "fund_sharpe_ci": (fnd_ci["ci_low"], fnd_ci["ci_high"]),
        "ladder_maxdd_pct": max_drawdown(lad) * 100,
        "fund_maxdd_pct": max_drawdown(fnd) * 100,
        "diff_bps_mo": float(diff.mean() * 1e4),
        "diff_ann_pct": float(diff.mean() * MONTHS * 100),
        "t_hac": newey_west_t(diff.values, nw_lags),
        "t_1s": one_sample_t(diff.values),
        "diff_sharpe": diff_ci["sharpe"],
        "diff_sharpe_ci": (diff_ci["ci_low"], diff_ci["ci_high"]),
    }


def era_table(ret: pd.DataFrame, ladder_weights: dict[str, float], fund: str = "AGG",
              cuts=(("2007-06", "2016-01", "2007-2015"),
                    ("2016-01", "2022-01", "2016-2021"),
                    ("2022-01", "2027-01", "2022-2026")),
              nw_lags: int = NW_LAGS) -> pd.DataFrame:
    """Ladder-minus-fund premium (ann %/yr) and HAC *t* across sub-eras. A REAL structural
    premium is positive and stable; a duration/credit artefact flips sign era to era."""
    lad = portfolio_returns(ret, ladder_weights)
    fnd = ret[fund]
    rows = []
    for lo, hi, lbl in cuts:
        mask = (ret.index >= pd.Timestamp(lo)) & (ret.index < pd.Timestamp(hi))
        d = (lad - fnd)[mask].dropna()
        if len(d) < 6:
            continue
        rows.append({
            "era": lbl, "n": int(len(d)),
            "ladder_ann_pct": ann_return(lad[mask]) * 100,
            "fund_ann_pct": ann_return(fnd[mask]) * 100,
            "diff_ann_pct": float(d.mean() * MONTHS * 100),
            "t_hac": newey_west_t(d.values, nw_lags),
        })
    return pd.DataFrame(rows).set_index("era")


def calendar_year_table(ret: pd.DataFrame, ladder_weights: dict[str, float],
                        fund: str = "AGG", cash: str = "BIL") -> pd.DataFrame:
    """Calendar-year total returns (%) for ladder, fund, cash and the ladder-minus-fund
    gap — the 2022 rate-shock row is the headline stress test."""
    lad = portfolio_returns(ret, ladder_weights)
    fnd = ret[fund]
    csh = ret[cash]

    def _yr(s):
        return s.groupby(s.index.year).apply(lambda x: (1.0 + x).prod() - 1.0) * 100

    out = pd.DataFrame({"ladder": _yr(lad), "fund": _yr(fnd), "cash": _yr(csh)})
    out["ladder_minus_fund"] = out["ladder"] - out["fund"]
    return out


# --------------------------------------------------------------------------- #
# Tradability — the costed net edge
# --------------------------------------------------------------------------- #
def costed_race(ret: pd.DataFrame, ladder_weights: dict[str, float], fund: str = "AGG",
                cash: str = "BIL", spread_bps_oneway: float = 3.0,
                annual_turnover: float = 0.15, nw_lags: int = NW_LAGS) -> dict:
    """Charge the ladder its real implementation cost, leave the buy-and-hold fund free.

    The ETF ladder is rebalanced/rolled once a year: ~``annual_turnover`` of NAV trades
    across SHY/IEI/IEF/TLT at ``spread_bps_oneway`` each way (both sides), spread evenly
    over 12 months. The one-ticker fund is bought once and held (no ongoing trading). The
    net ladder-minus-fund diff and its HAC *t* say whether any gross gap survives friction.
    """
    lad = portfolio_returns(ret, ladder_weights)
    fnd = ret[fund]
    monthly_cost = annual_turnover * 2.0 * spread_bps_oneway / 1e4 / MONTHS
    net_diff = (lad - monthly_cost - fnd).dropna()
    return {
        "spread_bps_oneway": spread_bps_oneway,
        "annual_turnover": annual_turnover,
        "ladder_cost_bps_yr": monthly_cost * MONTHS * 1e4,
        "gross_diff_ann_pct": float((lad - fnd).dropna().mean() * MONTHS * 100),
        "net_diff_ann_pct": float(net_diff.mean() * MONTHS * 100),
        "net_diff_bps_mo": float(net_diff.mean() * 1e4),
        "t_hac_net": newey_west_t(net_diff.values, nw_lags),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof only — never market evidence)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, nw_lags: int = NW_LAGS) -> dict:
    """Run the ladder-minus-fund premium test on a synthetic world (planted or null)."""
    diff = (world["ladder"] - world["fund"]).values
    return {
        "diff_ann_pct": float(np.nanmean(diff) * MONTHS * 100),
        "t_hac": newey_west_t(diff, nw_lags),
        "t_1s": one_sample_t(diff),
        "ladder_ex_sharpe": excess_sharpe(world["ladder"], world["cash"]),
        "fund_ex_sharpe": excess_sharpe(world["fund"], world["cash"]),
        "n": int(len(world)),
    }
