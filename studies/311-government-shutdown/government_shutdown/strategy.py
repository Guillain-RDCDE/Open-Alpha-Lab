"""The event-study engine and its honest controls — Study 311 (Government-Shutdown).

The folklore: *"buy the dip every time the US government shuts down — stocks always
bounce back."* We test it as a clean event study around the funding-gap start dates in
``data.SHUTDOWNS``:

1. Snap each shutdown start to the first trading session on or after it (the event bar
   ``t=0``). The start date is calendar-known, so the canonical window needs no extra
   execution lag; a ``+1``-session entry lag is offered as a robustness check.
2. Measure the **forward** total return from the event over a horizon (e.g. enter at the
   event bar's close, hold ``H`` sessions) — this is the "buy the dip" trade.
3. Race that against a **synthetic event-null**: the *same* horizon measured around many
   random dates. If shutdowns carry no special signal, the shutdown trades look exactly
   like random-date trades.

Inference is honest: a HAC (Newey-West) t-stat on the per-event forward returns, a
circular block-bootstrap CI, and a permutation/placebo test of the shutdown mean vs the
random-date distribution. Costs are charged one-way × NAV (round-trip on entry+exit).

With only a handful of shutdowns in the modern SPY era, this is a low-n event study by
construction — we say so loudly and let the (wide) confidence interval carry the verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Snap event dates onto the trading calendar
# ---------------------------------------------------------------------------
def snap_to_sessions(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    lag: int = 0,
) -> np.ndarray:
    """Integer positions in ``bars.index`` of the first session on/after each event.

    ``lag`` shifts the entry bar that many sessions later (the look-ahead robustness
    knob; the canonical event window uses ``lag=0`` because the shutdown date is known
    the moment funding lapses). Events outside the tape are dropped.
    """
    idx = bars.index
    locs = []
    for e in pd.DatetimeIndex(events):
        pos = idx.searchsorted(e, side="left")  # first session >= e
        pos += lag
        if 0 <= pos < len(idx):
            locs.append(int(pos))
    return np.array(sorted(set(locs)), dtype=int)


# ---------------------------------------------------------------------------
# The buy-the-dip forward-return trade
# ---------------------------------------------------------------------------
def event_forward_returns(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 20,
    lag: int = 0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Per-event forward total return of buying at the event close and holding ``horizon``.

    For each event the trade enters at the event bar's close (``lag`` sessions after the
    snap) and exits ``horizon`` sessions later at the close. ``cost_bps`` is a round-trip
    cost deducted once from the gross return (one-way × NAV, both legs).

    Columns: ``event_date, entry, exit, horizon, ret_gross, ret_net``.
    """
    close = bars["close"].to_numpy(dtype=float)
    locs = snap_to_sessions(bars, events, lag=lag)
    rows = []
    n = len(close)
    for e in locs:
        x = e + horizon
        if x >= n:
            continue
        entry, exit_ = close[e], close[x]
        ret_gross = exit_ / entry - 1.0
        rows.append(
            {
                "event_date": bars.index[e],
                "entry": entry,
                "exit": exit_,
                "horizon": horizon,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


def all_horizon_returns(
    bars: pd.DataFrame,
    locs: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forward ``horizon``-session total returns at a set of integer positions ``locs``."""
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    out = []
    for e in locs:
        x = e + horizon
        if x < n:
            out.append(close[x] / close[e] - 1.0)
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Honest inference
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``x`` (no quantlab hard dependency)."""
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


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 3,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 311,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (respects local dependence)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def permutation_pvalue(
    event_ret: np.ndarray,
    null_ret: np.ndarray,
    n_perm: int = 20000,
    seed: int = 311,
) -> tuple[float, float]:
    """Two-sided permutation p-value: is the shutdown mean unusual vs random-date returns?

    Pools the ``event_ret`` (shutdown trades) and ``null_ret`` (random-date trades),
    repeatedly re-draws a group of the event size, and asks how often a random draw's mean
    is at least as far from the pooled mean as the observed shutdown mean. Returns
    ``(observed_mean_minus_null_mean, p_value)``.
    """
    event_ret = np.asarray(event_ret, dtype=float)
    null_ret = np.asarray(null_ret, dtype=float)
    event_ret = event_ret[np.isfinite(event_ret)]
    null_ret = null_ret[np.isfinite(null_ret)]
    k = event_ret.size
    if k == 0 or null_ret.size == 0:
        return (float("nan"), float("nan"))
    obs = event_ret.mean() - null_ret.mean()
    pool = null_ret  # the random-date distribution is the reference
    rng = np.random.default_rng(seed)
    null_means = np.array([rng.choice(pool, size=k, replace=True).mean()
                           for _ in range(n_perm)])
    centre = pool.mean()
    p = float((np.abs(null_means - centre) >= abs(event_ret.mean() - centre)).mean())
    return float(obs), p


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(ret: np.ndarray) -> dict:
    """Headline statistics for a set of per-event forward returns."""
    ret = np.asarray(ret, dtype=float)
    ret = ret[np.isfinite(ret)]
    n = ret.size
    if n == 0:
        return dict(n=0, win_rate=float("nan"), mean=float("nan"),
                    median=float("nan"), std=float("nan"), tstat=float("nan"))
    return dict(
        n=int(n),
        win_rate=float((ret > 0).mean()),
        mean=float(ret.mean()),
        median=float(np.median(ret)),
        std=float(ret.std(ddof=1)) if n > 1 else float("nan"),
        tstat=hac_tstat(ret),
    )


def car_curve(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 10,
    post: int = 40,
    lag: int = 0,
) -> pd.DataFrame:
    """Average cumulative return path around the events, aligned on the event bar (t=0).

    Returns a frame indexed by relative session ``[-pre, +post]`` with the cross-event
    *mean* cumulative total return (rebased to 0 at t=0) and its standard error.
    """
    close = bars["close"].to_numpy(dtype=float)
    locs = snap_to_sessions(bars, events, lag=lag)
    rel = np.arange(-pre, post + 1)
    paths = []
    n = len(close)
    for e in locs:
        if e - pre < 0 or e + post >= n:
            continue
        seg = close[e - pre : e + post + 1]
        paths.append(seg / close[e] - 1.0)  # rebased to event bar
    if not paths:
        return pd.DataFrame({"rel": rel, "mean": np.nan, "se": np.nan}).set_index("rel")
    P = np.vstack(paths)
    mean = P.mean(axis=0)
    se = P.std(axis=0, ddof=1) / np.sqrt(P.shape[0]) if P.shape[0] > 1 else np.zeros_like(mean)
    return pd.DataFrame({"rel": rel, "mean": mean, "se": se}).set_index("rel")
