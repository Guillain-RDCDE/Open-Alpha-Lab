"""Strategy and honest controls — Study 116 (Power-Hour).

The folk claim: the last hour (15:00–16:00 ET) shows *directional momentum* — the day's
trend (or a reliable reversal of it) continues into the close. The recipe: observe the
morning/midday return (09:30→15:00 ET) and bet the **same direction** in the last hour.

We test two sub-claims:

1. **Continuation** (trending close): morning return predicts same-direction last-hour return.
   A positive ``corr(morning_ret, last_ret)`` would support this; zero means no signal.

2. **Last-hour momentum trade** (enter at 15:00, hold to 16:00, direction = sign of
   morning_ret): does following the day beat a **random-direction control**?
   The random-direction arm enters at the same sessions with coin-flip ±1 directions, giving
   us a direct "signal vs noise" comparison.

Three exit conventions for the momentum trade:

- **follow** — go in the direction of the morning return (the main claim).
- **fade** — go *against* the morning return (a contrarian check).
- **random** — i.i.d. ±1 coin on each session (the always-present control arm).

No look-ahead: morning_ret is computed from bars that close *before* 15:00; the trade enters
at 15:00's open and exits at 16:00's close — a gap unavoidable in a 1-hour bar study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prediction / correlation diagnostic
# ---------------------------------------------------------------------------
def morning_last_corr(panel: pd.DataFrame) -> dict:
    """Pearson correlation between the morning return and the last-hour return.

    Under the continuation hypothesis, ``corr > 0``. Under reversal, ``corr < 0``.
    Under the null (no predictability), ``corr ≈ 0``. Returns the correlation, its
    t-stat (test vs zero, n-2 dof), and the number of sessions.
    """
    m = panel["morning_ret"].to_numpy(dtype=float)
    l = panel["last_ret"].to_numpy(dtype=float)
    valid = np.isfinite(m) & np.isfinite(l)
    m, l = m[valid], l[valid]
    n = len(m)
    if n < 5:
        return {"n": n, "corr": float("nan"), "tstat": float("nan")}
    r = float(np.corrcoef(m, l)[0, 1])
    # Fisher z-transform t-stat: t = r * sqrt(n-2) / sqrt(1-r^2)
    if abs(r) < 1.0:
        t = r * np.sqrt(n - 2) / np.sqrt(1.0 - r**2)
    else:
        t = float("inf") * np.sign(r)
    return {"n": n, "corr": r, "tstat": float(t)}


# ---------------------------------------------------------------------------
# Per-session trade ledger
# ---------------------------------------------------------------------------
def build_ledger(
    panel: pd.DataFrame,
    arm: str = "follow",
    cost_bps: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per session: enter 15:00 in ``arm`` direction, exit at 16:00 close.

    Arms:
    - ``'follow'``  — go long (short) if morning was positive (negative). The main claim.
    - ``'fade'``    — the opposite: bet against the morning direction. Contrarian check.
    - ``'random'``  — i.i.d. ±1 coin per session (reproducible via ``seed``). The control.

    ``cost_bps`` is a one-way round-trip (enter + exit) cost applied to the net return.
    The gross return of each session's last-hour trade is just the signed last-hour return
    in the chosen direction.

    Columns: ``date, morning_ret, last_ret, dir, ret_gross, ret_net``.
    """
    panel = panel.dropna(subset=["morning_ret", "last_ret"])
    n = len(panel)
    if n == 0:
        return pd.DataFrame(columns=["date", "morning_ret", "last_ret",
                                     "dir", "ret_gross", "ret_net"])

    if arm == "follow":
        dirs = np.sign(panel["morning_ret"].to_numpy(dtype=float)).astype(int)
        dirs[dirs == 0] = 1  # flat morning → default long
    elif arm == "fade":
        dirs = -np.sign(panel["morning_ret"].to_numpy(dtype=float)).astype(int)
        dirs[dirs == 0] = -1
    elif arm == "random":
        rng = np.random.default_rng(seed)
        dirs = rng.choice([-1, 1], size=n)
    else:
        raise ValueError(f"Unknown arm {arm!r}; choose 'follow', 'fade', or 'random'.")

    last = panel["last_ret"].to_numpy(dtype=float)
    ret_gross = dirs * last
    ret_net = ret_gross - cost_bps * 1e-4

    return pd.DataFrame(
        {
            "date": panel.index,
            "morning_ret": panel["morning_ret"].to_numpy(),
            "last_ret": last,
            "dir": dirs,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ledger summary with HAC t-stat
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-session statistics for one ledger.

    Returns session count, win-rate, mean return (bps/trade), P&L skew, and a HAC
    (Newey-West) t-stat on the mean — the inference-bar number that decides whether
    the last-hour momentum effect is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC — shared idiom with quantlab.analytics.mean_tstat_hac.
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)
