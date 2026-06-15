"""The lifecycle glidepath engine and its honest controls — Study 172 (Hundred-Minus-Age).

The folk recipe: invest (100 − age)% in stocks and the rest in bonds, gliding the
equity weight down by one percentage point every year as you age.  A 30-year-old
holds 70% stocks; a 60-year-old holds 40%; a 70-year-old holds 30%.

We compare four strategies over a 40-year investor lifecycle (age 25 to 65) on a
fixed stream of monthly contributions followed by a fixed-period withdrawal phase:

1. **HMA** — Hundred-Minus-Age: equity weight = max(10%, (100 − age) / 100).
2. **HMA_rising** — Rising-equity glidepath (Pfau-Kitces 2014): start at 30%
   equity and glide *up* to 70%, then hold.  Tests the counterintuitive "bond tent"
   / rising-equity variant from the academic literature.
3. **Sixty_Forty** — Classic 60/40 constant-mix, annually rebalanced.  The
   benchmark every financial planner knows.
4. **All_Equity** — 100% equities throughout.  The return ceiling; the glidepath
   should beat it on risk even as it trails on raw return.

Three output metrics frame the honest verdict:

- **Terminal real wealth** — final portfolio value in real terms (normalised so
  contributions sum to 1).  Signal of return.
- **Sequence risk** — standard deviation of terminal wealth across many historical
  cohorts (rolling 40-year windows on the Shiller tape).  The glidepath's
  *raison d'être*: it should reduce the variance of outcomes, not just the mean.
- **Worst-cohort outcome** — the 5th-percentile terminal wealth across cohorts.
  The planning floor — what happens to the unlucky investor who retires into a
  bear market?

No look-ahead: each month's equity weight is computed from the investor's age at
the *start* of that month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


# ---------------------------------------------------------------------------
# Glidepath weight schedules
# ---------------------------------------------------------------------------

def hma_weight(age: float, floor: float = 0.10) -> float:
    """Hundred-Minus-Age equity weight: (100 − age) / 100, floored at ``floor``."""
    return max(floor, (100.0 - age) / 100.0)


def hma_rising_weight(age: float, start_age: float = 25.0, end_age: float = 65.0,
                       start_w: float = 0.30, end_w: float = 0.70) -> float:
    """Rising-equity glidepath (Pfau-Kitces 2014).

    Linear interpolation from ``start_w`` at ``start_age`` to ``end_w`` at
    ``end_age``; clamped at the endpoints.  The idea: sequence risk is highest in
    the early years after retirement, not at entry; so *increasing* equities
    during the accumulation phase and holding at the peak reduces the probability
    of a catastrophic sequence near the finish line.
    """
    t = (age - start_age) / max(end_age - start_age, 1.0)
    t = max(0.0, min(1.0, t))
    return start_w + t * (end_w - start_w)


def fixed_weight(w_equity: float = 0.60) -> float:
    """A constant equity weight (e.g. 60/40)."""
    return w_equity


# ---------------------------------------------------------------------------
# Lifecycle simulator
# ---------------------------------------------------------------------------

def simulate_lifecycle(
    returns: pd.DataFrame,
    start_age: float = 25.0,
    retire_age: float = 65.0,
    strategy: str = "hma",
    contribution: float = 1.0 / (40.0 * 12.0),  # normalised: 1 unit spread over 40yr
    rebalance: str = "annual",
    cost_bps: float = 5.0,
) -> pd.Series:
    """Simulate one lifecycle on a monthly return series and return an equity curve.

    ``returns`` must have columns ``['EQ', 'BD']`` as monthly simple returns.

    The lifecycle has two phases:
    - **Accumulation** (``start_age`` to ``retire_age``): equal monthly
      contributions of ``contribution`` units are added *at the start* of each
      month before the return is applied.
    - **Withdrawal**: omitted here — we focus on *terminal wealth* at retirement
      as our metric (the retirement-income phase is analysed via the sequence-risk
      distribution, not simulated explicitly).

    Weight schedule (``strategy``):
    - ``"hma"`` — 100-minus-age.
    - ``"hma_rising"`` — rising-equity (Pfau-Kitces 2014).
    - ``"sixty_forty"`` — constant 60/40.
    - ``"all_equity"`` — 100% equities throughout.

    Rebalancing (``rebalance``):
    - ``"annual"`` — reset weights at the first month of each calendar year,
      charging one-way turnover at ``cost_bps``.
    - ``"monthly"`` — reset each month (more costly; a sensitivity check).
    - ``"none"`` — buy-and-hold drift (approximates the low-rebalance case).

    Returns a monthly equity-curve Series (portfolio value in real return units,
    starting from 0 and growing via contributions and returns).
    """
    n_months_acc = int(round((retire_age - start_age) * 12))
    months = returns.index[:n_months_acc]
    if len(months) < n_months_acc:
        raise ValueError(
            f"Not enough return data for {n_months_acc} months. "
            f"Have {len(returns)} rows."
        )

    ret_eq = returns["EQ"].values[:n_months_acc]
    ret_bd = returns["BD"].values[:n_months_acc]

    val = 0.0
    val_eq = 0.0
    val_bd = 0.0
    equity_curve = np.empty(n_months_acc)
    cost = cost_bps * 1e-4
    prev_w_eq = None

    for m in range(n_months_acc):
        age = start_age + m / 12.0

        # Compute target weight
        if strategy == "hma":
            w_eq = hma_weight(age)
        elif strategy == "hma_rising":
            w_eq = hma_rising_weight(age, start_age=start_age, end_age=retire_age)
        elif strategy == "sixty_forty":
            w_eq = 0.60
        elif strategy == "all_equity":
            w_eq = 1.00
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        w_bd = 1.0 - w_eq

        # Add monthly contribution (at start of month, before returns)
        val += contribution
        val_eq += contribution * w_eq
        val_bd += contribution * w_bd

        # Apply returns
        val_eq *= (1.0 + ret_eq[m])
        val_bd *= (1.0 + ret_bd[m])
        val = val_eq + val_bd

        # Rebalancing
        rebal_now = False
        if rebalance == "annual" and m % 12 == 0 and m > 0:
            rebal_now = True
        elif rebalance == "monthly" and m > 0:
            rebal_now = True
        elif rebalance == "none":
            rebal_now = False

        if rebal_now and val > 0:
            cur_w_eq = val_eq / val
            turnover = abs(w_eq - cur_w_eq)
            val_eq = val * w_eq - val * turnover * cost
            val_bd = val * w_bd
            val = val_eq + val_bd

        equity_curve[m] = val

    return pd.Series(equity_curve, index=months, name=strategy)


# ---------------------------------------------------------------------------
# Rolling cohort analysis — the sequence-risk measurement
# ---------------------------------------------------------------------------

def rolling_cohorts(
    returns: pd.DataFrame,
    n_years: int = 40,
    start_age: float = 25.0,
    strategies: tuple[str, ...] = ("hma", "sixty_forty", "all_equity", "hma_rising"),
    contribution: float | None = None,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Terminal wealth distribution across all rolling historical cohorts.

    For each monthly start date in ``returns`` that has at least ``n_years * 12``
    months of data ahead, simulate each strategy and record the **terminal wealth**
    (portfolio value at the end of the accumulation phase).  The contribution is
    normalised so the sum of contributions (before any return) equals 1 unit,
    making terminal wealth a pure *multiplier* on lifetime savings.

    Returns a DataFrame indexed by cohort start date, one column per strategy.
    """
    retire_age = start_age + n_years
    n_months = n_years * 12
    # Normalise contribution so total contributions sum to 1 unit
    _contribution = contribution if contribution is not None else 1.0 / n_months
    rows = []
    dates = returns.index
    n = len(dates)
    for i in range(n - n_months + 1):
        ret_slice = returns.iloc[i: i + n_months]
        row = {"start": dates[i]}
        for strat in strategies:
            try:
                curve = simulate_lifecycle(
                    ret_slice, strategy=strat,
                    start_age=start_age,
                    retire_age=retire_age,
                    contribution=_contribution,
                    cost_bps=cost_bps,
                )
                row[strat] = float(curve.iloc[-1])
            except Exception:
                row[strat] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("start")
    return df


# ---------------------------------------------------------------------------
# Summary statistics across cohorts
# ---------------------------------------------------------------------------

def cohort_stats(cohorts: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy summary: mean, median, 5th-pct, 95th-pct terminal wealth and vol."""
    stats = {}
    for col in cohorts.columns:
        s = cohorts[col].dropna()
        stats[col] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "vol": float(s.std(ddof=1)),
            "n_cohorts": int(s.notna().sum()),
        }
    return pd.DataFrame(stats).T


# ---------------------------------------------------------------------------
# HAC t-stat on terminal-wealth differences (HMA − benchmark)
# ---------------------------------------------------------------------------

def hac_tstat(a: np.ndarray, b: np.ndarray) -> float:
    """Newey-West HAC t-stat for mean(a − b).

    Cohorts overlap by up to (n_years * 12 - 1) months, so the serial
    correlation can be massive — HAC is essential.  We use the standard
    Newey-West bandwidth selector: L = 4 * (n/100)^(2/9).
    """
    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 4:
        return float("nan")
    mu = diff.mean()
    e = diff - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Single-path stats (for the aggregate historical single-cohort headline)
# ---------------------------------------------------------------------------

def path_stats(curve: pd.Series) -> dict:
    """Return headline stats for a single portfolio equity curve."""
    n = len(curve)
    if n == 0:
        return {}
    terminal = float(curve.iloc[-1])
    peak = curve.cummax()
    drawdown = (curve / peak - 1.0).min()
    return {
        "terminal_wealth": terminal,
        "max_drawdown": float(drawdown),
        "n_months": n,
    }
