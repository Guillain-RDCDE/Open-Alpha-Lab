"""The engine and its honest controls — Study 586 (Liquidation-Cascade).

The folklore, at full strength: a large **forced-liquidation** cascade is a *capitulation bottom*.
The exchange dumps margin-called longs mechanically, the price overshoots down, and then it
**bounces** — so the tradable read is "buy after the liquidation spike, hold H days."

This module tests that as an **event study**:

1. **Event flag.** A day is a *liquidation event* if its liquidation flow ``liq`` sits in the top
   ``q`` quantile of the trailing distribution (a large cascade). Using a trailing quantile keeps
   the flag point-in-time (no look-ahead in the threshold).

2. **Forward return.** For each day, the cumulative return over the *next* ``horizon`` days
   (strictly after the event day — the mechanical same-day crash is excluded by construction).

3. **The event-study statistic.** Mean forward return on event days minus the unconditional
   baseline (all days), with a two-sample (Welch) *t*. The folklore predicts event forward
   returns > baseline (a bounce).

4. **A label-shuffle placebo.** Shuffle the event labels against the forward returns many times;
   the p-value is the tail probability of the observed gap. A real bounce sits in the tail.

5. **Costs.** The trade is a discrete entry+exit round trip per event; net subtracts a one-way
   cost on each side (× NAV). (It is a *long* — no borrow — but we keep the cost honest.)

6. **Robustness sweep.** Multiple horizons × multiple event thresholds — the sign and size of the
   post-event bounce should be stable if it is a signal.

7. **A synthetic positive control.** A deterministic world where the bounce is planted
   (``bounce_alpha > 0``); the engine must recover a positive event-minus-baseline *t* and stay
   flat at the null — averaged over ≥ 20 seeds (house rule) so no single lucky seed can fake it.

Execution convention (the one documented lag): the event is known at the event day's **close**
(``liq`` and ``ret`` for that day are complete), so we enter at that close and hold the *next*
``horizon`` days. The forward window is strictly ``t+1 .. t+horizon`` — the same-day cascade never
enters the tradable return, so there is no look-ahead. The sweep and the control use the identical
convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event flag and forward returns
# ---------------------------------------------------------------------------
def forward_returns(ret: pd.Series, horizon: int) -> pd.Series:
    """Cumulative simple return over the next ``horizon`` days, strictly after each day.

    ``fwd[t] = prod(1 + ret[t+1 .. t+horizon]) - 1``. The last ``horizon`` days are NaN (no full
    forward window). The same-day return is *excluded* — the event day's own crash never enters the
    tradable return, which is the whole point of an event study on the *bounce*.
    """
    r = ret.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    logr = np.log1p(r)
    csum = np.concatenate([[0.0], np.cumsum(logr)])
    for t in range(n - horizon):
        out[t] = np.expm1(csum[t + 1 + horizon] - csum[t + 1])
    return pd.Series(out, index=ret.index, name=f"fwd{horizon}")


def event_flag(liq: pd.Series, q: float = 0.95, lookback: int = 252) -> pd.Series:
    """Boolean flag: liquidation flow in the top ``1-q`` tail of the *trailing* window.

    A day is a liquidation *event* if ``liq[t]`` exceeds the ``q`` quantile of the trailing
    ``lookback`` days (inclusive). The trailing quantile keeps the threshold point-in-time — no
    future liquidation levels inform whether today is "large". Days before a full trailing window
    are ``False`` (not enough history to call a cascade).
    """
    x = liq.to_numpy(dtype=float)
    n = len(x)
    flag = np.zeros(n, dtype=bool)
    for t in range(lookback, n):
        window = x[t - lookback + 1 : t + 1]
        thr = np.quantile(window, q)
        flag[t] = x[t] >= thr and x[t] > 0
    return pd.Series(flag, index=liq.index, name="event")


# ---------------------------------------------------------------------------
# The event-study statistic
# ---------------------------------------------------------------------------
def event_study(
    frame: pd.DataFrame, horizon: int = 5, q: float = 0.95, lookback: int = 252
) -> dict:
    """Mean forward return on liquidation-event days vs the unconditional baseline.

    Returns a dict with the event mean, the baseline (all-day) mean, the gap (event − baseline),
    a two-sample Welch *t* on event forward returns vs non-event forward returns, and the counts.
    The folklore predicts ``gap > 0`` (a bounce after the cascade).
    """
    ret, liq = frame["ret"], frame["liq"]
    fwd = forward_returns(ret, horizon)
    ev = event_flag(liq, q=q, lookback=lookback)

    valid = fwd.notna()
    fwd_v = fwd[valid].to_numpy(dtype=float)
    ev_v = ev[valid].to_numpy(dtype=bool)

    a = fwd_v[ev_v]          # event forward returns
    b = fwd_v[~ev_v]         # non-event forward returns
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {
            "event_mean": float("nan"), "baseline_mean": float(np.nanmean(fwd_v)) if len(fwd_v) else float("nan"),
            "gap": float("nan"), "t": float("nan"), "n_event": na, "n_base": nb, "horizon": horizon, "q": q,
        }
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    baseline = fwd_v.mean()
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {
        "event_mean": float(a.mean()),
        "baseline_mean": float(baseline),
        "gap": float(a.mean() - b.mean()),
        "t": float(t),
        "n_event": int(na),
        "n_base": int(nb),
        "horizon": int(horizon),
        "q": float(q),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    frame: pd.DataFrame, horizon: int = 5, q: float = 0.95, lookback: int = 252,
    n_perm: int = 2000, seed: int = 586,
) -> float:
    """Label-shuffle placebo p-value for the event-minus-baseline gap.

    Shuffle the event labels against the (fixed) forward returns ``n_perm`` times; the p-value is
    the fraction of shuffles whose |gap| ≥ the observed |gap|. A real post-liquidation bounce sits
    in the tail; noise does not.
    """
    ret, liq = frame["ret"], frame["liq"]
    fwd = forward_returns(ret, horizon)
    ev = event_flag(liq, q=q, lookback=lookback)
    valid = fwd.notna()
    fwd_v = fwd[valid].to_numpy(dtype=float)
    ev_v = ev[valid].to_numpy(dtype=bool)
    ne = int(ev_v.sum())
    if ne < 2 or (len(ev_v) - ne) < 2:
        return float("nan")
    obs = abs(fwd_v[ev_v].mean() - fwd_v.mean())
    rng = np.random.default_rng(seed)
    n = len(fwd_v)
    count = 0
    for _ in range(n_perm):
        idx = rng.choice(n, size=ne, replace=False)
        gap = fwd_v[idx].mean() - fwd_v.mean()
        if abs(gap) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def net_gap(gap: float, cost_bps: float = 10.0) -> float:
    """Event-minus-baseline gap net of a round-trip cost.

    The trade is a discrete entry at the event close and an exit ``horizon`` days later — one
    entry + one exit, each crossing ``cost_bps`` (× NAV). It is a *long* (buy the bounce), so no
    borrow. Crypto perp/spot round trips are wider than equities, so 10 bps/side is generous-tight.
    """
    if not np.isfinite(gap):
        return float("nan")
    cost = 2.0 * cost_bps * 1e-4
    return float(gap - cost)


# ---------------------------------------------------------------------------
# Robustness sweep — horizons x thresholds
# ---------------------------------------------------------------------------
def robustness_sweep(
    frame: pd.DataFrame, horizons=(1, 3, 5, 10, 20), quantiles=(0.90, 0.95, 0.99),
    lookback: int = 252,
) -> pd.DataFrame:
    """Event-minus-baseline gap and its *t* across horizons × event thresholds.

    A real capitulation-bounce should keep its sign (and roughly its size) across sensible horizons
    and cutoffs. Returns a tidy frame (one row per (horizon, q)) with gap%, t, and event count.
    """
    rows = []
    for h in horizons:
        for q in quantiles:
            es = event_study(frame, horizon=h, q=q, lookback=lookback)
            rows.append(
                {"horizon": h, "q": q, "gap_pct": es["gap"] * 100.0,
                 "t": es["t"], "n_event": es["n_event"]}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(
    data_mod, bounce_alpha: float, horizon: int = 5, q: float = 0.95,
    lookback: int = 252, n_seeds: int = 25, base_seed: int = 586,
) -> float:
    """Average the event-study *t* over ``n_seeds`` synthetic worlds for a planted ``bounce_alpha``.

    The house rule: any synthetic-dependent claim averages the stat over ≥ 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean event-minus-baseline *t* across
    seeds. At ``bounce_alpha = 0`` (null) this must sit ≈ 0; a genuine planted bounce must drive it
    positive and past +2 as it grows.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(bounce_alpha=bounce_alpha, horizon=horizon, seed=s)
        ts.append(event_study(frame, horizon=horizon, q=q, lookback=lookback)["t"])
    return float(np.nanmean(ts))
