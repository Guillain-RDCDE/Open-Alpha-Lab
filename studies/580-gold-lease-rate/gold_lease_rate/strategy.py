"""The engine and its honest controls — Study 580 (Gold-Lease-Rate).

The claim, at full strength: the **gold lease rate** (the cost to borrow bullion, classically
``LIBOR − GOFO``) is a *leading indicator* for gold. A high or rising lease rate — a physical
scramble, backwardation — is folklore-said to foreshadow a gold rally; a lease rate collapsing
toward zero (ample metal, contango) to foreshadow weakness.

This module tests that lead-lag honestly:

1. **The predictive regression.** OLS of the *forward* gold return on the *lagged* standardised
   lease rate::

       gold_ret[t] = a + b * lease_z[t-lag] + e[t]

   The sign and *t*-stat of ``b`` are the claim (folklore says ``b > 0`` at *t* ≥ 2).

2. **The tradable long-flat rule.** Go long gold when the lagged lease-rate z is above a threshold,
   else hold cash — the folklore's operational form. We report gross and net (one-way cost × NAV
   on every switch), plus a buy-&-hold benchmark, so a "signal" that only matches beta is exposed.

3. **A label-shuffle placebo null.** Shuffle the lease-rate series against forward gold returns and
   re-fit; the p-value is the tail probability of the observed |t|. A real lead-lag sits in the
   tail; noise does not.

4. **A multi-lag / multi-window robustness sweep.** Re-fit at several lags and over sub-samples;
   a genuine microstructure lead-lag should not depend on picking one lucky lag.

5. **A seed-robust synthetic positive control (≥ 20 seeds).** Plant a lead-lag (``lead_beta > 0``)
   and confirm the engine recovers a positive *t*; set ``lead_beta = 0`` and confirm it stays flat.

Execution convention (the single documented lag): the predictor is the lease-rate z observed at the
close of month ``t-lag`` — *already public* — and the position is entered at that close and held
one month. The signal never uses contemporaneous or future data. The default ``lag = 1`` is the
folklore's "this month's borrow cost predicts next month's price"; the sweep varies it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The predictive regression — the sign & t of b ARE the claim
# ---------------------------------------------------------------------------
def _align(frame: pd.DataFrame, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x = lease_z[t-lag], y = gold_ret[t]) aligned, dropping the warm-up rows."""
    z = frame["lease_z"].to_numpy(dtype=float)
    r = frame["gold_ret"].to_numpy(dtype=float)
    if lag <= 0:
        return z, r
    return z[:-lag], r[lag:]


def predictive_regression(frame: pd.DataFrame, lag: int = 1) -> dict:
    """OLS of forward gold return on the lagged standardised lease rate.

    Returns the slope ``b`` (forward return per 1-sigma of lagged lease z), its *t*-stat, the
    Pearson correlation, R², and n. ``b > 0`` at *t* ≥ 2 is the folklore claim (high borrow cost
    now → higher gold next period).
    """
    x, y = _align(frame, lag)
    n = len(y)
    if n < 8 or np.std(x) == 0:
        return {"b": float("nan"), "t": float("nan"), "corr": float("nan"),
                "r2": float("nan"), "n": int(n)}
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {
        "b": float(coef[1]),
        "t": float(coef[1] / se_b) if se_b > 0 else float("nan"),
        "corr": corr,
        "r2": r2,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The tradable long-flat rule — with costs and a buy-&-hold benchmark
# ---------------------------------------------------------------------------
def long_flat_backtest(
    frame: pd.DataFrame,
    lag: int = 1,
    z_threshold: float = 0.0,
    cost_bps: float = 5.0,
) -> dict:
    """Long-gold-when-lease-high, else-flat, entered on the lagged (public) signal.

    Position at month ``t`` is 1 (long gold) if ``lease_z[t-lag] > z_threshold`` else 0 (cash).
    A one-way cost of ``cost_bps`` is charged on every position change (× NAV). Returns annualised
    gross and net strategy returns, the buy-&-hold gold return, the strategy Sharpe (net), the
    number of switches, and the hit rate of long months.
    """
    z = frame["lease_z"].to_numpy(dtype=float)
    r = frame["gold_ret"].to_numpy(dtype=float)
    n = len(r)
    if n <= lag + 6:
        return {"ann_gross": float("nan"), "ann_net": float("nan"),
                "ann_bh": float("nan"), "sharpe_net": float("nan"),
                "switches": 0, "n": int(n), "frac_long": float("nan")}
    pos = np.zeros(n)
    for t in range(lag, n):
        pos[t] = 1.0 if z[t - lag] > z_threshold else 0.0
    switches = int(np.sum(np.abs(np.diff(pos)) > 0))
    cost = np.zeros(n)
    cost[1:] = np.abs(np.diff(pos)) * cost_bps * 1e-4
    strat_gross = pos * r
    strat_net = pos * r - cost
    eff = strat_net[lag:]                                   # drop warm-up
    ann = 12.0
    ann_gross = float(np.mean(strat_gross[lag:]) * ann)
    ann_net = float(np.mean(eff) * ann)
    ann_bh = float(np.mean(r[lag:]) * ann)
    sd = float(np.std(eff, ddof=1))
    sharpe_net = float(np.mean(eff) / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    return {
        "ann_gross": ann_gross,
        "ann_net": ann_net,
        "ann_bh": ann_bh,
        "sharpe_net": sharpe_net,
        "switches": switches,
        "n": int(n - lag),
        "frac_long": float(np.mean(pos[lag:])),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(frame: pd.DataFrame, lag: int = 1, n_perm: int = 2000,
                   seed: int = 580) -> float:
    """Label-shuffle placebo p-value for the predictive-regression |t|.

    Shuffle the lease-rate predictor against forward gold returns ``n_perm`` times and re-fit; the
    p-value is the fraction of shuffles whose |t| ≥ the observed |t|. A real lead-lag sits in the
    tail; noise scatters uniformly.
    """
    x, y = _align(frame, lag)
    n = len(y)
    if n < 12 or np.std(x) == 0:
        return float("nan")
    obs = abs(predictive_regression(frame, lag)["t"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        xs = rng.permutation(x)
        X = np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sigma2 = float(resid @ resid) / max(n - 2, 1)
        se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
        t = abs(coef[1] / se) if se > 0 else 0.0
        if t >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Robustness — a lag sweep
# ---------------------------------------------------------------------------
def lag_sweep(frame: pd.DataFrame, lags=(0, 1, 2, 3, 6, 12)) -> pd.DataFrame:
    """Re-fit the predictive regression at several lags; return a tidy table of (lag, b, t, n)."""
    rows = []
    for lg in lags:
        reg = predictive_regression(frame, lag=lg)
        rows.append({"lag": lg, "b": reg["b"], "t": reg["t"], "n": reg["n"]})
    return pd.DataFrame(rows).set_index("lag")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, lead_beta: float, lag: int = 1, n_seeds: int = 25,
                     base_seed: int = 580) -> float:
    """Average the predictive-regression *t* over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the test stat over ≥ 20 seeds so no
    single lucky RNG seed can manufacture significance. Returns the mean *t* across seeds for a
    planted ``lead_beta`` (0 = null → mean t ≈ 0; positive → mean t rises).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(lead_beta=lead_beta, lag=lag, seed=s)
        ts.append(predictive_regression(frame, lag=lag)["t"])
    return float(np.nanmean(ts))
