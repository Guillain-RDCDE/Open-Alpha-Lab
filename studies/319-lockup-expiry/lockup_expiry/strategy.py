"""Event-study engine and honest controls — Study 319 (Lockup-Expiry).

The believers' claim: when an IPO's 180-day lock-up expires, insiders and early investors
are suddenly free to sell. That supply shock should push the price down — a predictable,
calendar-known sag a trader could short into.

We test it as a textbook **event study** on *abnormal returns* (stock − market benchmark):

1. ``aar``  — the average abnormal return at each event-time bar ``tau`` (relative to the
   expiry bar at ``tau = 0``), across all IPOs in the basket.
2. ``car``  — the cumulative abnormal return over a chosen window (e.g. ``[-5, +5]`` or
   ``[0, +5]``), with a cross-sectional HAC/Newey-West *t*-stat and a block-bootstrap CI.
3. ``tradable_short`` — the believers' trade made real: short each IPO over a post-expiry
   window, market-hedged, charged one-way costs, with one documented execution lag.

Inference bar (METHODOLOGY.md):
- REAL requires a robust |t| ≥ 2 on the **real** tape. A synthetic control only proves the
  harness *can* detect a planted sag — never market evidence.
- The event date is **calendar-known in advance** (it is in the S-1), so no execution lag
  is needed to *learn* the signal; the lag we document is purely the fill convention
  (signal/intent at the close of ``tau-1``, position carried over ``tau`` onward).
- Costs are one-way × NAV; a market-hedged short pays the leg twice (stock + hedge) and,
  in reality, borrow — flagged where it bites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newey-West HAC t-stat on a mean
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of ``x`` being zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Average abnormal return per event-time bar
# ---------------------------------------------------------------------------
def aar(panel: pd.DataFrame) -> pd.DataFrame:
    """Average Abnormal Return at each ``tau`` across events.

    Returns a DataFrame indexed by ``tau`` with columns:
    ``aar`` (mean abnormal return, fractional), ``n`` (events at that tau),
    ``tstat`` (cross-sectional t of the mean at that bar).
    """
    rows = []
    for tau, g in panel.groupby("tau"):
        x = g["abn_ret"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 2:
            rows.append({"tau": int(tau), "aar": float("nan"), "n": n, "tstat": float("nan")})
            continue
        se = x.std(ddof=1) / np.sqrt(n)
        rows.append({
            "tau": int(tau),
            "aar": float(x.mean()),
            "n": int(n),
            "tstat": float(x.mean() / se) if se > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cumulative abnormal return over a window
# ---------------------------------------------------------------------------
def per_event_car(panel: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Per-event cumulative abnormal return over the inclusive window ``[lo, hi]`` (taus)."""
    win = panel[(panel["tau"] >= lo) & (panel["tau"] <= hi)]
    car = win.groupby("event")["abn_ret"].sum()
    return car.to_numpy(dtype=float)


def car(panel: pd.DataFrame, lo: int = -5, hi: int = 5, n_boot: int = 2000, seed: int = 319) -> dict:
    """Cumulative abnormal return statistics over the window ``[lo, hi]``.

    Sums each event's abnormal returns across the window (a simple, additive CAR — the
    standard event-study aggregate), then tests the cross-section of per-event CARs against
    zero with a HAC *t* and a circular block-bootstrap CI on the mean.

    Returns a dict: ``window, n, mean_car (fractional), mean_car_bps, median_car_bps,
    win_rate (fraction CAR < 0, i.e. a "sag"), tstat, ci_lo_bps, ci_hi_bps``.
    """
    car_i = per_event_car(panel, lo, hi)
    car_i = car_i[np.isfinite(car_i)]
    n = car_i.size
    if n < 2:
        return {
            "window": f"[{lo:+d},{hi:+d}]", "n": int(n), "mean_car": float("nan"),
            "mean_car_bps": float("nan"), "median_car_bps": float("nan"),
            "win_rate": float("nan"), "tstat": float("nan"),
            "ci_lo_bps": float("nan"), "ci_hi_bps": float("nan"),
        }
    mean = float(car_i.mean())
    lo_b, hi_b = block_bootstrap_ci(car_i, n_boot=n_boot, seed=seed)
    return {
        "window": f"[{lo:+d},{hi:+d}]",
        "n": int(n),
        "mean_car": mean,
        "mean_car_bps": mean * 1e4,
        "median_car_bps": float(np.median(car_i)) * 1e4,
        # "sag" = a negative CAR; report the fraction of events that sagged.
        "win_rate": float((car_i < 0).mean()),
        "tstat": hac_tstat(car_i),
        "ci_lo_bps": lo_b * 1e4,
        "ci_hi_bps": hi_b * 1e4,
    }


# ---------------------------------------------------------------------------
# Circular block bootstrap CI on a mean
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 4,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 319,
) -> tuple[float, float]:
    """Circular block-bootstrap (1−alpha) CI for the mean of ``x``.

    Blocks preserve any residual cross-event dependence (e.g. cohort/vintage clustering),
    which an i.i.d. resample would destroy.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The believers' trade made real
# ---------------------------------------------------------------------------
def tradable_short(
    panel: pd.DataFrame,
    lo: int = 0,
    hi: int = 5,
    cost_bps: float = 10.0,
    borrow_bps_per_day: float = 0.0,
) -> dict:
    """Net result of shorting each IPO over the post-expiry window ``[lo, hi]``, hedged.

    Because abnormal returns are already market-relative (stock − benchmark), shorting the
    abnormal path *is* the market-hedged short: a profit when the IPO sags relative to the
    market. The per-event PnL is the *negative* of the window's CAR (you are short).

    Costs (house style): one-way ``cost_bps`` × NAV is charged on **each leg** at entry and
    exit (stock short + the implicit hedge), i.e. ``4 × cost_bps`` round-trip across the two
    legs, plus optional ``borrow_bps_per_day`` financing over the holding days. The
    execution lag is the fill convention only — the expiry date is calendar-known, so no lag
    is needed to learn the signal.

    Returns a dict mirroring ``car`` but on the *short* PnL (gross & net), so a positive
    mean = the believers' trade actually pays.
    """
    car_i = per_event_car(panel, lo, hi)
    car_i = car_i[np.isfinite(car_i)]
    n = car_i.size
    if n < 2:
        return {k: float("nan") for k in (
            "n", "gross_bps", "net_bps", "tstat_gross", "tstat_net", "win_rate")}
    pnl_gross = -car_i  # short the abnormal path
    hold = max(hi - lo, 1)
    cost = 4.0 * cost_bps * 1e-4 + borrow_bps_per_day * 1e-4 * hold
    pnl_net = pnl_gross - cost
    return {
        "n": int(n),
        "gross_bps": float(pnl_gross.mean()) * 1e4,
        "net_bps": float(pnl_net.mean()) * 1e4,
        "tstat_gross": hac_tstat(pnl_gross),
        "tstat_net": hac_tstat(pnl_net),
        "win_rate": float((pnl_net > 0).mean()),
        "cost_bps_total": float(cost * 1e4),
    }


# ---------------------------------------------------------------------------
# Synthetic positive-control helper
# ---------------------------------------------------------------------------
def detects_sag(panel: pd.DataFrame, lo: int = -1, hi: int = 1) -> bool:
    """True if the harness finds a *significant negative* CAR over ``[lo, hi]``.

    The machinery proof: on a panel with a planted sag this should be True; on a null panel
    it should (almost always) be False. Never used to back a Signal stamp.
    """
    s = car(panel, lo=lo, hi=hi, n_boot=500)
    return bool(s["mean_car"] < 0 and s["tstat"] <= -2.0)
