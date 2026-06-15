"""The retirement engine and its honest controls — Study 173 (Four-Percent-Rule).

The famous recipe: Bengen (1994) showed that a US retiree withdrawing 4% of the
*initial* portfolio value in real terms from a 60% stock / 40% bond mix never ran
out of money over any 30-year window in Shiller history back to 1926.  We implement
the engine, run every historical rolling cohort, and compare it against two controls:

  1. **The unconstrained baseline** — what is the *actual* safe withdrawal rate (SWR)
     for a 95% historical success rate?  Is 4% conservative, exact, or optimistic?
  2. **Sequence-of-returns stress** — the first few years of a retirement dominate
     survival: a retiree with a bad opening decade almost never recovers, even if
     the same 30-year mean return later obtains.

The verdict is honest: 4% worked historically (Signal REAL), but it is fragile to
today's starting conditions — high valuations (PE10 ~34 in 2024) and low real bond
yields imply a mechanically lower SWR going forward.

No look-ahead: the initial portfolio is fixed on the retirement date; withdrawals use
the *real* dollar amount set at t=0 adjusted for inflation already embedded in the
real-return data (we work in real terms throughout, so a constant real withdrawal
IS the inflation-adjusted recipe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core retirement simulation
# ---------------------------------------------------------------------------
def simulate_cohort(
    returns: pd.DataFrame,
    start_year: int,
    horizon: int = 30,
    withdrawal_rate: float = 0.04,
    equity_weight: float = 0.60,
    eq_col: str = "eq_real_tr",
    bond_col: str = "bond_real_tr",
    initial_wealth: float = 1.0,
    rebalance: bool = True,
) -> dict:
    """Simulate one retirement cohort starting at ``start_year``.

    The retiree starts with ``initial_wealth = 1.0`` (normalised), withdraws
    ``withdrawal_rate * initial_wealth`` *at the start of each year* (beginning-of-
    year, conservative vs. end-of-year), invests the remainder in the 60/40 mix,
    then lets it compound.  In real-return terms this is exactly Bengen's recipe.

    Returns a dict with:
      - ``survived``: bool — portfolio > 0 after all ``horizon`` years.
      - ``terminal_wealth``: float — portfolio value at end of horizon (can be 0).
      - ``years_solvent``: int — how many years the portfolio stayed above 0.
      - ``annual_wealth``: list[float] — wealth at start of each year after withdrawal.
      - ``drawdown_peak``: float — worst peak-to-trough drawdown over the horizon.
    """
    # Clip to available data.
    avail = returns.loc[
        (returns.index >= start_year) & (returns.index < start_year + horizon)
    ]
    if len(avail) < horizon:
        return {
            "survived": False,
            "terminal_wealth": 0.0,
            "years_solvent": 0,
            "annual_wealth": [],
            "drawdown_peak": float("nan"),
        }

    eq_rets = avail[eq_col].to_numpy()
    bond_rets = avail[bond_col].to_numpy()

    w = float(initial_wealth)
    annual_w = []
    peak = w
    worst_dd = 0.0
    annual_real_withdrawal = withdrawal_rate * initial_wealth

    for t in range(horizon):
        # Withdraw at start of year.
        w -= annual_real_withdrawal
        if w <= 0:
            annual_w.append(0.0)
            return {
                "survived": False,
                "terminal_wealth": 0.0,
                "years_solvent": t,
                "annual_wealth": annual_w,
                "drawdown_peak": worst_dd,
            }
        annual_w.append(w)
        # Invest remainder.
        if rebalance:
            w_eq = w * equity_weight
            w_bond = w * (1.0 - equity_weight)
        else:
            # Drift weights (no rebalancing).
            if t == 0:
                w_eq = w * equity_weight
                w_bond = w * (1.0 - equity_weight)
            # else keep from prior year (already updated below)
        w_eq *= 1.0 + eq_rets[t]
        w_bond *= 1.0 + bond_rets[t]
        w = w_eq + w_bond
        peak = max(peak, w)
        worst_dd = max(worst_dd, (peak - w) / peak if peak > 0 else 0.0)

    return {
        "survived": w > 0,
        "terminal_wealth": float(w),
        "years_solvent": horizon,
        "annual_wealth": annual_w,
        "drawdown_peak": float(worst_dd),
    }


# ---------------------------------------------------------------------------
# Rolling-cohort analysis across all start years
# ---------------------------------------------------------------------------
def run_all_cohorts(
    returns: pd.DataFrame,
    horizon: int = 30,
    withdrawal_rate: float = 0.04,
    equity_weight: float = 0.60,
    eq_col: str = "eq_real_tr",
    bond_col: str = "bond_real_tr",
) -> pd.DataFrame:
    """Run every available rolling 30-year cohort and return a per-cohort ledger.

    A cohort starting at year ``y`` uses returns ``y`` through ``y + horizon - 1``.
    The last valid start year is therefore ``max_year - horizon + 1``.

    Columns: ``start_year, survived, terminal_wealth, years_solvent, drawdown_peak``.
    """
    last_year = int(returns.index.max())
    first_year = int(returns.index.min())
    last_start = last_year - horizon + 1

    rows = []
    for y in range(first_year, last_start + 1):
        res = simulate_cohort(
            returns, y, horizon, withdrawal_rate, equity_weight, eq_col, bond_col
        )
        rows.append({
            "start_year": y,
            "survived": res["survived"],
            "terminal_wealth": res["terminal_wealth"],
            "years_solvent": res["years_solvent"],
            "drawdown_peak": res["drawdown_peak"],
        })
    return pd.DataFrame(rows).set_index("start_year")


# ---------------------------------------------------------------------------
# Safe withdrawal rate finder
# ---------------------------------------------------------------------------
def find_swr(
    returns: pd.DataFrame,
    horizon: int = 30,
    equity_weight: float = 0.60,
    target_success_rate: float = 0.95,
    eq_col: str = "eq_real_tr",
    bond_col: str = "bond_real_tr",
    lo: float = 0.01,
    hi: float = 0.15,
    tol: float = 1e-4,
) -> float:
    """Binary-search the highest withdrawal rate that achieves ``target_success_rate``
    across all historical rolling ``horizon``-year cohorts.

    This is the *historical* safe withdrawal rate — what Bengen (1994) called the
    SAFEMAX: the highest rate that would have survived every single 30-year window.
    Pfau (2010) introduces the ``target_success_rate < 1.0`` framing we use here.
    """
    def success_rate(wr: float) -> float:
        cohorts = run_all_cohorts(returns, horizon, wr, equity_weight, eq_col, bond_col)
        return float(cohorts["survived"].mean())

    # Binary search for the SWR at the target success rate.
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if success_rate(mid) >= target_success_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(lo)


# ---------------------------------------------------------------------------
# Sequence-of-returns analysis
# ---------------------------------------------------------------------------
def sequence_risk_stats(
    returns: pd.DataFrame,
    horizon: int = 30,
    withdrawal_rate: float = 0.04,
    equity_weight: float = 0.60,
    first_n_years: int = 5,
    eq_col: str = "eq_real_tr",
    bond_col: str = "bond_real_tr",
) -> pd.DataFrame:
    """Quantify sequence-of-returns risk.

    For each cohort, compute the mean equity return in the first ``first_n_years``
    and regress survival / terminal wealth on that.  Bad early returns are the
    primary driver of ruin — this is the "sequence risk" Kitces (2008) and Pfau
    (2011) document.

    Returns the cohort ledger augmented with ``early_eq_return`` and ``bad_sequence``
    (below-median first-5-year return).
    """
    cohorts = run_all_cohorts(
        returns, horizon, withdrawal_rate, equity_weight, eq_col, bond_col
    )
    last_year = int(returns.index.max())
    last_start = last_year - horizon + 1

    early = {}
    for y in cohorts.index:
        if y > last_start:
            continue
        slice_ = returns.loc[
            (returns.index >= y) & (returns.index < y + first_n_years), eq_col
        ]
        early[y] = float(slice_.mean()) if len(slice_) == first_n_years else float("nan")

    cohorts["early_eq_return"] = pd.Series(early)
    cohorts["bad_sequence"] = cohorts["early_eq_return"] < cohorts["early_eq_return"].median()
    return cohorts


# ---------------------------------------------------------------------------
# Valuation-conditional SWR (the forward-looking fragility)
# ---------------------------------------------------------------------------
def swr_by_pe10_quartile(
    returns: pd.DataFrame,
    pe10: pd.Series,
    horizon: int = 30,
    withdrawal_rate: float = 0.04,
    equity_weight: float = 0.60,
    eq_col: str = "eq_real_tr",
    bond_col: str = "bond_real_tr",
) -> pd.DataFrame:
    """Survival rate by CAPE (PE10) quartile at retirement date.

    Pfau (2011) and Kitces (2008) show that starting valuations predict the safe
    withdrawal rate better than historical averages.  This function slices the cohort
    ledger by PE10 quartile at the start year.

    ``pe10`` must be an annual Series indexed by year.
    """
    cohorts = run_all_cohorts(
        returns, horizon, withdrawal_rate, equity_weight, eq_col, bond_col
    )
    # Align PE10 to start years.
    cohorts["pe10"] = pe10.reindex(cohorts.index)
    cohorts = cohorts.dropna(subset=["pe10"])
    cohorts["pe10_quartile"] = pd.qcut(cohorts["pe10"], q=4, labels=["Q1 (cheap)", "Q2", "Q3", "Q4 (dear)"])
    summary = cohorts.groupby("pe10_quartile", observed=True).agg(
        n=("survived", "count"),
        survival_rate=("survived", "mean"),
        median_terminal_wealth=("terminal_wealth", "median"),
    )
    return summary


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(cohorts: pd.DataFrame) -> dict:
    """Headline statistics for a cohort ledger.

    Returns n_cohorts, survival_rate, median_terminal_wealth, worst_terminal_wealth,
    pct_fail, years_solvent_if_fail (median years before ruin among failures).
    """
    n = len(cohorts)
    survived = cohorts["survived"]
    terminal = cohorts["terminal_wealth"]
    fails = cohorts[~survived]
    return {
        "n_cohorts": int(n),
        "survival_rate": float(survived.mean()),
        "median_terminal_wealth": float(terminal.median()),
        "worst_terminal_wealth": float(terminal.min()),
        "pct_fail": float((~survived).mean() * 100),
        "years_solvent_if_fail": (
            float(fails["years_solvent"].median()) if len(fails) > 0 else float("nan")
        ),
        "max_drawdown_median": float(cohorts["drawdown_peak"].median()),
    }
