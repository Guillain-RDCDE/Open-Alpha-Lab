"""The freeze trade and its honest controls — Study 309 (OJ-Frost).

The folklore (the *Trading Places* trade): a hard freeze in the Florida citrus belt
destroys part of the orange crop, so frozen-concentrate OJ futures (OJ=F) spike. If you
could *anticipate* — or even just *react fast to* — a freeze, you'd ride the spike. And
even absent a freeze, the story goes, OJ has a tradable **winter seasonality** (cold
risk is priced in Dec–Feb).

We test three things, each against an honest baseline:

1. **The freeze event study.** Around each hardcoded freeze date, the cumulative
   abnormal return over a forward window (the "did OJ pop after the freeze?" question).
   The control is the *same-length window starting on random non-freeze dates* — so we
   measure the freeze excess, not just OJ's ambient drift/vol.
2. **The reactive freeze trade.** You can't trade a freeze you don't know about, so the
   tradable version enters on the **first session after** the freeze date and holds the
   window. One execution lag, documented: a freeze known on (cold) night *d* is acted on
   at the close of the first session at-or-after *d* and the trade earns the *following*
   window — never the freeze-day move itself (which you couldn't have caught).
3. **Winter seasonality.** Mean daily return in Dec–Feb vs the rest of the year, with a
   HAC *t* on the difference. A calendar rule — no lag needed.

Inference: Newey-West (HAC) *t* on the per-event / per-day series and a **circular block
bootstrap** CI that respects OJ's volatility clustering. Costs are charged one-way ×
NAV (a round trip = 2× the one-way bps), deducted from the gross window return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def daily_log_returns(frame: pd.DataFrame) -> pd.Series:
    """Daily log returns of the close, NaN-dropped."""
    return np.log(frame["close"]).diff().dropna()


# ---------------------------------------------------------------------------
# Event-window helper
# ---------------------------------------------------------------------------
def _first_loc_at_or_after(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int | None:
    """Position of the first tape session at or after ``ts`` (None if past the end)."""
    pos = index.searchsorted(ts, side="left")
    return int(pos) if pos < len(index) else None


def window_returns(
    frame: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    window: int = 5,
    lag: int = 1,
    cost_bps_one_way: float = 0.0,
) -> pd.DataFrame:
    """Per-event forward window log-return around each event date.

    For each event date the engine finds the first tape session at-or-after the event,
    then *waits ``lag`` sessions* (the documented execution lag: you learn of the freeze
    on the cold night and can act no sooner than the next session), and accumulates the
    log-return over the following ``window`` sessions.

    With ``lag=1`` (canonical reactive trade) the trade **never** earns the freeze-day
    move itself — only the forward window you could actually have captured. With
    ``lag=0`` the window starts on the event session (the "perfect foresight" event-study
    variant, look-ahead, used only to size the un-tradable ceiling).

    Costs: ``cost_bps_one_way`` charged once on entry and once on exit (a round trip is
    ``2 × cost_bps_one_way``), deducted from the gross window log-return.

    Columns: ``event_date, entry_idx, ret_gross, ret_net, n_obs``.
    """
    idx = frame.index
    close = frame["close"]
    rows = []
    for ev in pd.DatetimeIndex(event_dates):
        loc = _first_loc_at_or_after(idx, ev)
        if loc is None:
            continue
        entry = loc + lag
        exit_ = entry + window
        if entry >= len(idx) or exit_ >= len(idx):
            continue
        # entry at the close of bar `entry`, exit at the close of bar `exit_`.
        ret_gross = float(np.log(close.iat[exit_] / close.iat[entry]))
        rt_cost = 2.0 * cost_bps_one_way * 1e-4
        rows.append(
            {
                "event_date": ev,
                "entry_idx": entry,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - rt_cost,
                "n_obs": window,
            }
        )
    return pd.DataFrame(rows)


def random_control_windows(
    frame: pd.DataFrame,
    n_events: int,
    window: int = 5,
    n_draws: int = 2000,
    seed: int = 309,
) -> np.ndarray:
    """Mean window log-return for ``n_draws`` random placebo event sets.

    Each placebo set is ``n_events`` random entry sessions (uniform over the tape, leaving
    room for the window). Returns the array of placebo *mean* window returns — the null
    distribution the freeze events' mean is read against. This is the "same window on
    random dates" control: it nets out OJ's ambient drift and the window length.
    """
    close = frame["close"].to_numpy()
    n = close.size
    rng = np.random.default_rng(seed)
    hi = n - window - 1
    means = np.empty(n_draws)
    for d in range(n_draws):
        entries = rng.integers(1, hi, size=n_events)
        rets = np.log(close[entries + window] / close[entries])
        means[d] = float(rets.mean())
    return means


# ---------------------------------------------------------------------------
# Winter seasonality
# ---------------------------------------------------------------------------
def winter_seasonality(frame: pd.DataFrame) -> dict:
    """Mean daily log-return in Dec–Feb vs the rest of the year, with a HAC *t* on the
    difference. A calendar rule — no execution lag needed."""
    r = daily_log_returns(frame)
    m = r.index.month
    winter = (m == 12) | (m == 1) | (m == 2)
    rw = r[winter].to_numpy()
    ro = r[~winter].to_numpy()
    # HAC t on the difference of means via a regression of r on a winter dummy.
    y = r.to_numpy()
    x = winter.astype(float)
    diff = float(rw.mean() - ro.mean())
    t = _hac_t_slope(y, x)
    return {
        "winter_mean_bps": float(rw.mean() * 1e4),
        "other_mean_bps": float(ro.mean() * 1e4),
        "diff_bps": float(diff * 1e4),
        "tstat": t,
        "n_winter": int(rw.size),
        "n_other": int(ro.size),
    }


# ---------------------------------------------------------------------------
# Inference — HAC t-stat and circular block bootstrap
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC *t*-stat of the mean of ``x`` (H0: mean = 0)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
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


def _hac_t_slope(y: np.ndarray, x: np.ndarray) -> float:
    """HAC *t* on the slope of an OLS of ``y`` on a constant + ``x``."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 5:
        return float("nan")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        g = (X[k:] * resid[k:, None]).T @ (X[:-k] * resid[:-k, None]) / n
        S += w * (g + g.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(cov[1, 1], 0.0))
    return float(beta[1] / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 4,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 309,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (preserves serial dependence)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Event-study summary
# ---------------------------------------------------------------------------
def summarize_events(
    ledger: pd.DataFrame,
    col: str = "ret_net",
    control_means: np.ndarray | None = None,
) -> dict:
    """Headline statistics for an event ledger.

    Returns event count, mean window return (bps), win-rate, HAC *t*, block-bootstrap CI,
    and — if a ``control_means`` placebo distribution is supplied — the empirical
    percentile of the events' mean within it (the "how unusual is the freeze window vs
    random dates" *p*-value) and the excess over the placebo mean.
    """
    if ledger.empty:
        return {"n_events": 0, "mean_bps": float("nan"), "win_rate": float("nan"),
                "tstat": float("nan"), "ci_lo_bps": float("nan"), "ci_hi_bps": float("nan"),
                "placebo_pct": float("nan"), "excess_bps": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    mean = float(r.mean())
    lo, hi = block_bootstrap_ci(r)
    out = {
        "n_events": int(r.size),
        "mean_bps": float(mean * 1e4),
        "win_rate": float((r > 0).mean()),
        "tstat": hac_tstat(r),
        "ci_lo_bps": float(lo * 1e4),
        "ci_hi_bps": float(hi * 1e4),
        "placebo_pct": float("nan"),
        "excess_bps": float("nan"),
    }
    if control_means is not None and control_means.size:
        out["placebo_pct"] = float((control_means < mean).mean())
        out["excess_bps"] = float((mean - control_means.mean()) * 1e4)
    return out
