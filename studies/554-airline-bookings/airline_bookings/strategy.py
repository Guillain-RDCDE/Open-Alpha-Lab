"""The engine and its honest controls — Study 554 (Airline-Bookings).

The claim, at full strength: a flight-**bookings momentum** signal turns up *before* airline
earnings and therefore *before* airline stock returns, so buying the airline basket when booking
momentum is high (and standing aside / shorting when it is low) harvests a predictable edge.

This module measures that, honestly, on the study's synthetic tape (no free real booking index
exists — see ``data.py``):

1. **The predictive regression.** OLS of the airline basket's *next*-month return on *this*
   month's booking momentum. The slope's *sign and t-stat* are the pitch: a positive, significant
   slope means bookings lead returns. We report a **Newey-West / HAC** t-stat (autocorrelation-
   robust, the desk's inference bar) alongside the OLS t.

2. **The contemporaneous check.** The *same*-month regression (return on same-month bookings). A
   big contemporaneous slope with a ~0 *forward* slope is the efficient-market story: the signal is
   already in the price, nothing left to predict.

3. **The trading rule + costs.** A long/flat (and a long/short) timing rule on the sign of booking
   momentum, held one month, with a one-way cost per rebalance and a borrow charge on the short
   leg. Gross *and* net, both labelled.

4. **The placebo null.** Circular block-shuffle of the signal against forward returns; the
   fraction of shuffles whose |t| >= the observed |t| is the placebo p-value.

5. **A robustness sweep.** The predictive t across sub-samples (halves + rolling windows) — a real
   lead keeps its sign.

6. **A synthetic positive control.** A deterministic world where the lead is planted
   (``lead_beta > 0``); the engine must recover a positive predictive t and stay flat at the null.

Execution convention: the signal at month *t* is built from data through the close of month *t*;
the traded return is month *t+1*'s. That one-month lag between a public signal and the traded
return is the single documented execution convention (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


# ---------------------------------------------------------------------------
# HAC (Newey-West) t for a single-regressor OLS slope
# ---------------------------------------------------------------------------
def _ols_hac(x: np.ndarray, y: np.ndarray, lags: int = 3) -> dict:
    """OLS of y on [1, x]; return slope, plain-OLS t, and Newey-West HAC t."""
    n = len(x)
    if n < 6:
        return {"slope": float("nan"), "t_ols": float("nan"), "t_hac": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    # plain OLS SE
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    se_ols = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    # Newey-West HAC (Bartlett kernel)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov_hac = xtx_inv @ S @ xtx_inv
    se_hac = float(np.sqrt(max(cov_hac[1, 1], 0.0)))
    return {
        "slope": float(coef[1]),
        "t_ols": float(coef[1] / se_ols) if se_ols > 0 else float("nan"),
        "t_hac": float(coef[1] / se_hac) if se_hac > 0 else float("nan"),
        "n": n,
    }


# ---------------------------------------------------------------------------
# The predictive regression — the sign IS the pitch
# ---------------------------------------------------------------------------
def predictive_regression(frame: pd.DataFrame, lags: int = 3) -> dict:
    """OLS of next-month airline return on this-month booking momentum.

    Slope > 0 with |t_hac| >= 2 = bookings genuinely *lead* returns. Returns slope, OLS t, HAC t,
    Pearson correlation and n.
    """
    df = frame.dropna(subset=["bookings_mom", "fwd_ret"])
    if len(df) < 6:
        return {"slope": float("nan"), "t_ols": float("nan"), "t_hac": float("nan"),
                "corr": float("nan"), "n": len(df)}
    x = df["bookings_mom"].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    out = _ols_hac(x, y, lags=lags)
    out["corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return out


def contemporaneous_regression(frame: pd.DataFrame, lags: int = 3) -> dict:
    """OLS of *same*-month airline return on same-month booking momentum.

    A large contemporaneous slope with a ~0 predictive slope is the efficient-market story: the
    booking signal is already reflected in the price this month.
    """
    df = frame.dropna(subset=["bookings_mom", "ret"])
    if len(df) < 6:
        return {"slope": float("nan"), "t_ols": float("nan"), "t_hac": float("nan"),
                "corr": float("nan"), "n": len(df)}
    x = df["bookings_mom"].to_numpy(dtype=float)
    y = df["ret"].to_numpy(dtype=float)
    out = _ols_hac(x, y, lags=lags)
    out["corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# The trading rule — gross and net
# ---------------------------------------------------------------------------
def timing_rule(frame: pd.DataFrame, mode: str = "long_flat",
                cost_bps: float = 10.0, borrow_ann_bps: float = 300.0) -> dict:
    """Sign-of-momentum timing on the airline basket, held one month.

    - ``long_flat``: hold the basket next month iff booking momentum > 0, else flat.
    - ``long_short``: long iff momentum > 0, else short (short pays borrow, pro-rated monthly).

    Charges a one-way ``cost_bps`` each time the position changes, plus (for the short leg) a
    monthly slice of ``borrow_ann_bps``. Returns gross and net mean monthly return, annualised
    gross/net, and the number of trades. The *buy-and-hold* basket is the honest benchmark.
    """
    df = frame.dropna(subset=["bookings_mom", "fwd_ret"])
    if len(df) < 6:
        return {}
    sig = df["bookings_mom"].to_numpy(dtype=float)
    r = df["fwd_ret"].to_numpy(dtype=float)         # return earned NEXT month from a signal now
    if mode == "long_flat":
        pos = (sig > 0).astype(float)
    elif mode == "long_short":
        pos = np.where(sig > 0, 1.0, -1.0)
    else:
        raise ValueError(mode)

    gross = pos * r
    # costs: one-way cost each time position changes (entry included)
    prev = np.concatenate([[0.0], pos[:-1]])
    turns = np.abs(pos - prev)
    cost = turns * cost_bps * 1e-4
    borrow = np.where(pos < 0, borrow_ann_bps * 1e-4 / 12.0, 0.0)
    net = gross - cost - borrow

    bh = r  # buy-and-hold the basket
    return {
        "mode": mode,
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "gross_ann": float(gross.mean() * 12.0),
        "net_ann": float(net.mean() * 12.0),
        "bh_ann": float(bh.mean() * 12.0),
        "net_minus_bh_ann": float((net.mean() - bh.mean()) * 12.0),
        "n_trades": int(turns.sum()),
        "n": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Placebo / block-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(frame: pd.DataFrame, n_perm: int = 2000, block: int = 6,
                   lags: int = 3, seed: int = 554) -> float:
    """Circular block-bootstrap placebo p-value for the predictive HAC t.

    Break the signal->forward-return link by circularly rotating the signal in blocks (preserving
    the signal's own autocorrelation), re-fit, and count the fraction of shuffles with
    |t_hac| >= |observed t_hac|. A genuine lead sits in the tail; a spurious one does not.
    """
    df = frame.dropna(subset=["bookings_mom", "fwd_ret"])
    if len(df) < 12:
        return float("nan")
    x = df["bookings_mom"].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    obs = abs(_ols_hac(x, y, lags=lags)["t_hac"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block, n - block)) if n > 2 * block else int(rng.integers(1, n))
        xs = np.roll(x, shift)                     # circular rotation preserves x's autocorr
        t = abs(_ols_hac(xs, y, lags=lags)["t_hac"])
        if np.isfinite(t) and t >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Robustness — the predictive HAC t across sub-samples
# ---------------------------------------------------------------------------
def robustness_windows(frame: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Predictive HAC t over the full sample, each half, and three thirds."""
    df = frame.dropna(subset=["bookings_mom", "fwd_ret"])
    n = len(df)
    spans = {
        "full": df,
        "first half": df.iloc[: n // 2],
        "second half": df.iloc[n // 2 :],
        "third 1": df.iloc[: n // 3],
        "third 2": df.iloc[n // 3 : 2 * n // 3],
        "third 3": df.iloc[2 * n // 3 :],
    }
    rows = []
    for lab, sub in spans.items():
        reg = predictive_regression(sub, lags=lags)
        rows.append((lab, reg["slope"], reg["t_hac"], reg["n"]))
    return pd.DataFrame(rows, columns=["window", "slope", "t_hac", "n"]).set_index("window")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, lead_beta: float, contemp_beta: float = 0.9,
                     n_seeds: int = 25, base_seed: int = 554, lags: int = 3) -> float:
    """Average the predictive HAC t over ``n_seeds`` synthetic worlds for a planted ``lead_beta``.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean predictive HAC t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(lead_beta=lead_beta, contemp_beta=contemp_beta, seed=s)
        ts.append(predictive_regression(frame, lags=lags)["t_hac"])
    return float(np.nanmean(ts))
