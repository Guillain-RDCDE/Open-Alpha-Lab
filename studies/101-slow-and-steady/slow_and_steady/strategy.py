"""The race and its honest scoring — Study 101 (Slow-and-Steady).

The folk claim: *"dollar-cost averaging is safer AND smarter than investing a lump sum —
spreading your money in lowers risk and beats going all-in."* We implement both arms over
**every** rolling start date and let the tape decide.

The two arms, given a fixed budget of \\$1 and a deployment horizon of ``deploy_months``
(default 12) starting at some date *t0*:

- **Lump-sum (LS):** invest the whole \\$1 at *t0*'s close. Terminal wealth at the end of
  the deployment window is ``close[t_end] / close[t0]``.
- **Dollar-cost averaging (DCA):** split the \\$1 into ``deploy_months`` equal monthly
  tranches, one bought at each month-boundary close across the window. Cash waiting to be
  deployed earns **0%** — a stated, conservative-to-DCA choice (a T-bill would lift DCA
  only marginally and is noted in the write-up, not modelled here). Terminal wealth at the
  end of the window is ``sum_i (tranche / close[t_i]) * close[t_end]``.

Both arms are scored at the **same** end date (the close of the deployment window), so the
race is purely "how you got in", not "how long you stayed". A window contributes only if
the full horizon fits inside the sample — no partial windows.

The honest reading: on an up-drifting tape (equities' norm) LS deploys sooner and
compounds longer, so it wins most windows on *return*; DCA's only genuine benefit is a
**tighter dispersion** of outcomes — a behavioural/risk hedge, not higher expected wealth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# A single window: lump-sum vs DCA terminal wealth
# ---------------------------------------------------------------------------
def _month_boundary_locs(index: pd.DatetimeIndex, start_loc: int, n_months: int) -> list[int]:
    """Integer positions of the ``n_months`` monthly buy dates, starting at ``start_loc``.

    Tranche *i* (i = 0..n_months-1) is bought on the first trading day on or after
    ``t0 + i`` calendar months. Returns the integer locations into ``index``; the list is
    shorter than ``n_months`` only if the window runs off the end of the sample (the
    caller treats that as "window does not fit").
    """
    t0 = index[start_loc]
    locs: list[int] = []
    for i in range(n_months):
        target = t0 + pd.DateOffset(months=i)
        loc = index.searchsorted(target, side="left")
        if loc >= len(index):
            break
        locs.append(int(loc))
    return locs


def window_wealth(
    close: pd.Series, start_loc: int, deploy_months: int = 12
) -> tuple[float, float] | None:
    """Terminal wealth of (lump_sum, dca) for a \\$1 budget over one deployment window.

    The window starts at integer position ``start_loc`` and runs ``deploy_months`` calendar
    months. Both arms are valued at the **same** end date — the close of the last monthly
    buy date (i.e. the end of the deployment period). Returns ``None`` if the full horizon
    does not fit inside the sample (no partial windows).
    """
    index = close.index
    buy_locs = _month_boundary_locs(index, start_loc, deploy_months)
    if len(buy_locs) < deploy_months:
        return None
    end_loc = buy_locs[-1]
    p = close.to_numpy()
    p_end = p[end_loc]

    # Lump-sum: whole $1 at t0, valued at end.
    ls = p_end / p[start_loc]

    # DCA: equal tranches at each monthly close, cash earns 0% while waiting.
    tranche = 1.0 / deploy_months
    shares = sum(tranche / p[loc] for loc in buy_locs)
    dca = shares * p_end
    return float(ls), float(dca)


# ---------------------------------------------------------------------------
# All rolling windows
# ---------------------------------------------------------------------------
def rolling_race(
    close: pd.Series, deploy_months: int = 12, step: int = 1
) -> pd.DataFrame:
    """Run the lump-sum vs DCA race over every rolling start date.

    For each start position (every ``step`` trading days) where the full ``deploy_months``
    horizon fits, compute terminal wealth of both arms. Returns a tidy frame indexed by
    the window start date with columns ``lump_sum``, ``dca``, and ``ls_minus_dca`` (the
    terminal-wealth gap, lump-sum minus DCA, on the \\$1 budget).
    """
    rows = []
    idx = close.index
    for loc in range(0, len(close), step):
        out = window_wealth(close, loc, deploy_months)
        if out is None:
            continue
        ls, dca = out
        rows.append((idx[loc], ls, dca, ls - dca))
    df = pd.DataFrame(rows, columns=["start", "lump_sum", "dca", "ls_minus_dca"])
    return df.set_index("start")


# ---------------------------------------------------------------------------
# Win-rate with a Wilson confidence interval
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion ``k/n``.

    Returns ``(p_hat, lo, hi)``. Used to put an honest interval on the lump-sum win-rate
    (the windows are heavily overlapping, so this is an optimistic lower bound on the true
    uncertainty — the HAC/block treatment of the *gap* in ``hac_tstat`` is the rigorous
    companion).
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(p), float(centre - half), float(centre + half)


def win_stats(race: pd.DataFrame) -> dict:
    """Headline win-rate and dispersion stats for a rolling-race frame.

    Reports the share of windows lump-sum wins (with a Wilson CI), the mean and median
    terminal-wealth gap, and — the risk angle — the **dispersion** (std and worst case) of
    each arm's terminal wealth. Lower DCA dispersion is its one real benefit.
    """
    n = len(race)
    ls_wins = int((race["ls_minus_dca"] > 0).sum())
    p, lo, hi = wilson_ci(ls_wins, n)
    gap = race["ls_minus_dca"].to_numpy()
    ls, dca = race["lump_sum"].to_numpy(), race["dca"].to_numpy()
    return {
        "n_windows": n,
        "ls_wins": ls_wins,
        "ls_win_rate": p,
        "ls_win_ci": (lo, hi),
        "mean_gap": float(np.mean(gap)),
        "median_gap": float(np.median(gap)),
        "mean_gap_pct": float(np.mean(gap) * 100.0),     # cents per $1 of budget
        "median_gap_pct": float(np.median(gap) * 100.0),
        "ls_mean": float(np.mean(ls)),
        "dca_mean": float(np.mean(dca)),
        "ls_std": float(np.std(ls, ddof=1)),
        "dca_std": float(np.std(dca, ddof=1)),
        "ls_worst": float(np.min(ls)),
        "dca_worst": float(np.min(dca)),
        "dispersion_ratio": float(np.std(dca, ddof=1) / np.std(ls, ddof=1))
        if np.std(ls, ddof=1) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on the terminal-wealth gap series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep).

    The rolling windows overlap massively (a one-day step means adjacent windows share
    ~all of their path), so the per-window gaps are strongly autocorrelated. A naive t-stat
    would be wildly overstated; the HAC correction with a long lag is the honest read on
    "is the mean gap distinguishable from zero?".
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")
