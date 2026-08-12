"""Strategy + inference for Study 888 — CLO AAA Carry.

The question: is the **AAA-CLO carry** — the spread a senior CLO tranche pays over cash
and over same-rated corporates — a *real, mechanical, risk-adjusted* pickup, and does it
survive costs? Everything is measured **excess-of-cash** (minus BIL), so the comparison is
an excess-vs-excess Sharpe race, not a raw total-return beauty contest.

Method, per leg (JAAA / ICLO / LQD / IEF / BKLN):

* **Excess-of-cash daily return.** ``leg_return - BIL_return`` — the carry over the
  risk-free short rate. The AAA-CLO story lives here: JAAA's excess is a small, *steady*
  spread; LQD/IEF's excess is a large, *volatile* duration bet.
* **Carry stats.** Annualised excess mean, annualised vol, the excess **Sharpe**, its
  block-bootstrap CI (reused from ``quantlab.stats.sharpe_ci_bootstrap``), a Newey-West
  (HAC) *t* on the daily excess mean (reused from ``quantlab.analytics.mean_tstat_hac``),
  and the max drawdown of the leg's own NAV.
* **Head-to-head.** JAAA minus each benchmark's *excess* return, HAC *t* — the
  risk-adjusted spread. (The raw JAAA-minus-LQD return is confounded by duration; the
  excess-Sharpe race is the honest read.)
* **Era cut.** A ZIRP era (rates ~0, thin AAA-CLO spread in *dollars*) vs a high-rate era
  (2023+, fat carry) — the effect must not be one regime's artefact.
* **Costed version.** ETF total-return NAV is already net of the expense ratio, so the
  only extra frictions are the bid-ask **spread on rebalances** (one-way bps x NAV x
  turnover/yr) and, for the relative long-JAAA / short-LQD isolation trade, **borrow** on
  the short leg. A buy-and-hold carry harvest turns over ~nothing, so the net barely moves;
  the relative trade pays for its short.

Inference primitives are reused from ``quantlab`` where they fit; a local Newey-West and a
one-sample *t* keep the module importable stand-alone.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make quantlab importable when this package is used directly (verify.py / notebooks add
# the repo root to sys.path; this fallback keeps `python -c "import clo_aaa"` working too).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import annualized_sharpe, sharpe_ci_bootstrap  # noqa: E402

TRADING_DAYS = 252
NW_LAGS = 10  # HAC lag window for daily series


# --------------------------------------------------------------------------- #
# Excess-of-cash returns
# --------------------------------------------------------------------------- #
def excess_returns(ret: pd.DataFrame, cash: str = "BIL") -> pd.DataFrame:
    """Every column minus the cash column (excess-of-cash). Drops the cash column."""
    out = ret.sub(ret[cash], axis=0)
    return out.drop(columns=[cash])


def _drawdown(r: pd.Series) -> float:
    """Max drawdown (a negative number) of the compounded NAV of a return series."""
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# Inference primitives (local; HAC/bootstrap reuse quantlab)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC (Newey-West, Bartlett) t of mean(x) vs 0 — a stand-alone twin of
    ``quantlab.analytics.mean_tstat_hac`` (kept so the module needs no quantlab to run)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Per-leg carry stats (the excess-vs-excess race)
# --------------------------------------------------------------------------- #
def carry_stats(ret: pd.DataFrame, leg: str, cash: str = "BIL",
                lags: int = NW_LAGS, n_boot: int = 2000, seed: int = 888) -> dict:
    """Excess-of-cash carry stats for one leg over its own valid history.

    Aligns ``leg`` and ``cash`` on their common non-NaN dates (so a young ETF is graded on
    its own live window, not padded with zeros). Returns annualised excess mean/vol, the
    excess Sharpe, its block-bootstrap CI, the HAC *t* on the daily excess, and the leg's
    own NAV max drawdown.
    """
    pair = ret[[leg, cash]].dropna()
    ex = (pair[leg] - pair[cash])
    ex = ex.dropna()
    n = len(ex)
    if n < 20:
        return {"leg": leg, "n": n}
    hac = mean_tstat_hac(ex, lags=lags)
    boot = sharpe_ci_bootstrap(ex, n_boot=n_boot, seed=seed, method="cbb")
    sharpe = annualized_sharpe(ex)
    return {
        "leg": leg,
        "n": n,
        "start": str(ex.index.min().date()),
        "end": str(ex.index.max().date()),
        "excess_ann_pct": float(ex.mean()) * TRADING_DAYS * 100,
        "vol_ann_pct": float(ex.std(ddof=1)) * np.sqrt(TRADING_DAYS) * 100,
        "sharpe": float(sharpe),
        "sharpe_lo": float(boot["ci_low"]),
        "sharpe_hi": float(boot["ci_high"]),
        "frac_neg": float(boot["frac_negative"]),
        "excess_bps_day": float(ex.mean()) * 1e4,
        "t_hac": float(hac["tstat"]),
        "t_1s": one_sample_t(ex.to_numpy()),
        "maxdd_pct": _drawdown(pair[leg]) * 100,
    }


def race(ret: pd.DataFrame, legs: list[str], cash: str = "BIL",
         start: str | None = None, end: str | None = None, **kw) -> pd.DataFrame:
    """Excess-of-cash carry stats for every leg, one row each, sorted by Sharpe."""
    r = ret
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    rows = [carry_stats(r, leg, cash, **kw) for leg in legs if leg in r.columns]
    rows = [x for x in rows if x.get("n", 0) >= 20]
    df = pd.DataFrame(rows).set_index("leg")
    return df.sort_values("sharpe", ascending=False)


# --------------------------------------------------------------------------- #
# Head-to-head: JAAA excess minus a benchmark excess (the risk-adjusted spread)
# --------------------------------------------------------------------------- #
def head_to_head(ret: pd.DataFrame, leg: str, bench: str, cash: str = "BIL",
                 lags: int = NW_LAGS) -> dict:
    """Daily (leg_excess - bench_excess) over their common window: mean, HAC *t*, Sharpe.

    This is ``(leg - cash) - (bench - cash) = leg - bench`` on common dates, but framing it
    as an excess-vs-excess difference keeps the accounting honest. Positive+significant =
    the AAA-CLO carry beat the benchmark on a like-for-like (per-dollar) basis.
    """
    sub = ret[[leg, bench, cash]].dropna()
    diff = (sub[leg] - sub[cash]) - (sub[bench] - sub[cash])  # == leg - bench
    diff = diff.dropna()
    n = len(diff)
    if n < 20:
        return {"leg": leg, "bench": bench, "n": n}
    hac = mean_tstat_hac(diff, lags=lags)
    sd = diff.std(ddof=1)
    return {
        "leg": leg, "bench": bench, "n": n,
        "start": str(diff.index.min().date()), "end": str(diff.index.max().date()),
        "diff_ann_pct": float(diff.mean()) * TRADING_DAYS * 100,
        "t_hac": float(hac["tstat"]),
        "sharpe": float(diff.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The costed carry harvest + the relative isolation trade
# --------------------------------------------------------------------------- #
def costed_carry(ret: pd.DataFrame, leg: str = "JAAA", cash: str = "BIL",
                 spread_bps_oneway: float = 3.0, rebalances_per_year: float = 12.0,
                 lags: int = NW_LAGS) -> dict:
    """Buy-and-hold the carry leg funded by cash; charge the ETF bid-ask on rebalances.

    ETF total-return NAV is already net of the fund expense ratio, so the extra friction is
    only the bid-ask spread paid when you (re)establish the position. We charge
    ``spread_bps_oneway`` x NAV x ``rebalances_per_year`` per year against the excess-of-cash
    carry. A true buy-and-hold turns over ~once at entry (rebalances_per_year -> ~0); we keep
    a conservative monthly-rebalance default to show the carry survives even needless churn.
    """
    pair = ret[[leg, cash]].dropna()
    ex = (pair[leg] - pair[cash]).dropna()
    n = len(ex)
    charge_annual = spread_bps_oneway / 1e4 * rebalances_per_year
    charge_daily = charge_annual / TRADING_DAYS
    net = ex - charge_daily
    hac = mean_tstat_hac(net, lags=lags)
    sd = net.std(ddof=1)
    return {
        "leg": leg, "n": n,
        "gross_ann_pct": float(ex.mean()) * TRADING_DAYS * 100,
        "charge_ann_pct": charge_annual * 100,
        "net_ann_pct": float(net.mean()) * TRADING_DAYS * 100,
        "net_sharpe": float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "t_net_hac": float(hac["tstat"]),
    }


def relative_trade(ret: pd.DataFrame, leg: str = "JAAA", short: str = "LQD",
                   borrow_annual_bps: float = 40.0, spread_bps_oneway: float = 3.0,
                   rebalances_per_year: float = 12.0, lags: int = NW_LAGS) -> dict:
    """Long the carry leg / short a benchmark (dollar-neutral): isolates the spread but
    pays borrow on the short + bid-ask on BOTH legs per rebalance. Gross P&L = leg - short
    on common dates (this is NOT excess-of-cash — the two cash legs cancel)."""
    sub = ret[[leg, short]].dropna()
    gross = (sub[leg] - sub[short]).dropna()
    n = len(gross)
    charge_annual = (borrow_annual_bps / 1e4
                     + 2.0 * spread_bps_oneway / 1e4 * rebalances_per_year)
    net = gross - charge_annual / TRADING_DAYS
    hac = mean_tstat_hac(net, lags=lags)
    sd = net.std(ddof=1)
    return {
        "leg": leg, "short": short, "n": n,
        "gross_ann_pct": float(gross.mean()) * TRADING_DAYS * 100,
        "charge_ann_pct": charge_annual * 100,
        "net_ann_pct": float(net.mean()) * TRADING_DAYS * 100,
        "net_sharpe": float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "t_net_hac": float(hac["tstat"]),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, carry: str = "carry", cash: str = "cash",
                     lags: int = NW_LAGS) -> dict:
    """Run the excess-of-cash carry detector on a synthetic world's carry leg."""
    ret = world[[carry, cash]].rename(columns={carry: "CARRY", cash: "BIL"})
    return carry_stats(ret, "CARRY", cash="BIL", lags=lags, n_boot=800, seed=888)
