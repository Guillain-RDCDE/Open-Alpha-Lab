"""Strategy + inference for Study 394 — Defense-Basket (event study around shocks).

The claim under test:

    "Defense stocks reliably rally on war and rearmament headlines."

We operationalise it as an **event study**. For each hardcoded geopolitical-shock date we
measure the defense basket's cumulative return *in excess of SPY* over the ``window`` trading
days **after** the shock, entering one day after the event (no look-ahead). The believers'
version says this excess is reliably positive and big; we test it with:

  * a **t-stat** of the event-window excess returns against zero (the "does defense beat the
    market after a shock" question), with both a plain t and a HAC/Newey-West variant on the
    daily excess series for the time-series view;
  * a **placebo / bootstrap** null — draw the same number of *random* non-event dates many
    times and ask how often pure chance gives an event-window excess as large as the shocks;
  * a **win-rate** (P[event-window excess > 0]) vs the unconditional base rate (the share of
    *all* windows whose excess is positive — defense's natural drift relative to the market);
  * **one-day execution lag** and **one-way costs** on a "buy the basket the day after a
    shock, hold ``window`` days, short SPY against it" market-neutral trade.

The decisive number is not the point estimate but its standard error: with ~20 lifetime
shocks, even a few-point average excess is statistically indistinguishable from luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 21          # ~1 trading month after the shock
TRADING_YEAR = 252


# --------------------------------------------------------------------------- #
# Event-window measurement
# --------------------------------------------------------------------------- #
def _nearest_pos(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    """Position of the first trading day on/after ``date`` (None if past the tape)."""
    pos = index.searchsorted(date, side="left")
    if pos >= len(index):
        return None
    return int(pos)


def event_excess(frame: pd.DataFrame, events: pd.DatetimeIndex,
                 window: int = DEFAULT_WINDOW, lag: int = 1,
                 col: str = "excess") -> np.ndarray:
    """Cumulative ``window``-day excess return of the basket over each shock, 1-day entry lag.

    Entry is the close ``lag`` days after the first trading day on/after the shock (no
    look-ahead); the window holds ``window`` trading days. Compounds daily excess returns.
    Events whose window overruns the tape are dropped.
    """
    idx = frame.index
    r = frame[col].values
    n = len(r)
    out = []
    for d in events:
        p = _nearest_pos(idx, d)
        if p is None:
            continue
        start = p + lag
        end = start + window
        if end > n:
            continue
        seg = r[start:end]
        if np.isnan(seg).any():
            continue
        out.append(float(np.prod(1.0 + seg) - 1.0))
    return np.asarray(out, dtype=float)


def unconditional_excess(frame: pd.DataFrame, window: int = DEFAULT_WINDOW,
                         lag: int = 1, col: str = "excess") -> np.ndarray:
    """All overlapping ``window``-day cumulative excess returns (the base-rate distribution)."""
    r = frame[col].values
    n = len(r)
    one = 1.0 + r
    out = []
    for start in range(lag, n - window):
        seg = one[start:start + window]
        if np.isnan(seg).any():
            continue
        out.append(float(np.prod(seg) - 1.0))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(sample: np.ndarray) -> float:
    """t-stat of mean(sample) against zero. NaN if fewer than 2 points."""
    if len(sample) < 2:
        return float("nan")
    sd = sample.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(sample.mean() / (sd / np.sqrt(len(sample))))


def hac_t_daily(frame: pd.DataFrame, events: pd.DatetimeIndex,
                window: int = DEFAULT_WINDOW, lag: int = 1,
                col: str = "excess") -> float:
    """Newey-West (HAC) t of the *daily* excess return inside the event windows vs zero.

    Pools every in-window daily excess observation and tests its mean against zero with a HAC
    standard error (Bartlett kernel, lag = window-1) so overlapping/auto-correlated daily
    observations don't inflate the t. This is the autocorrelation-robust view of "does the
    basket out-earn the market on the days right after a shock?".
    """
    idx = frame.index
    r = frame[col].values
    n = len(r)
    pooled = []
    for d in events:
        p = _nearest_pos(idx, d)
        if p is None:
            continue
        start = p + lag
        end = start + window
        if end > n:
            continue
        seg = r[start:end]
        if np.isnan(seg).any():
            continue
        pooled.extend(seg.tolist())
    x = np.asarray(pooled, dtype=float)
    if len(x) < 3:
        return float("nan")
    x = x - 0.0
    m = x.mean()
    e = x - m
    T = len(x)
    L = max(1, window - 1)
    gamma0 = (e @ e) / T
    s = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1)
        cov = (e[k:] @ e[:-k]) / T
        s += 2.0 * w * cov
    se = np.sqrt(s / T)
    if se == 0:
        return float("nan")
    return float(m / se)


def placebo_pvalue(frame: pd.DataFrame, events: pd.DatetimeIndex,
                   window: int = DEFAULT_WINDOW, lag: int = 1,
                   n_draws: int = 20_000, seed: int = 394,
                   col: str = "excess") -> dict:
    """Small-sample placebo null: draw ``len(events)`` random valid entry dates many times and
    ask how often a random draw's mean event-window excess matches/beats the shock set.

    Returns the shock mean, the placebo mean, and the empirical p-value
    P[random-draw mean >= shock mean] — the honest answer to "could ~20 random dates have
    looked this good?"
    """
    obs = event_excess(frame, events, window=window, lag=lag, col=col)
    k = len(obs)
    if k == 0:
        return {"k": 0, "shock_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    r = frame[col].values
    n = len(r)
    one = 1.0 + r
    # precompute all valid overlapping window excesses (skip any with NaN)
    valid = []
    for start in range(lag, n - window):
        seg = one[start:start + window]
        if np.isnan(seg).any():
            continue
        valid.append(np.prod(seg) - 1.0)
    valid = np.asarray(valid, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = valid[rng.integers(0, len(valid), size=k)].mean()
    obs_mean = float(obs.mean())
    p = float((means >= obs_mean).mean())
    return {"k": k, "shock_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, events: pd.DatetimeIndex,
              window: int = DEFAULT_WINDOW, lag: int = 1, col: str = "excess") -> dict:
    """Headline stats for one window: n, conditional mean/win-rate, unconditional base-rate,
    plain t, HAC t, and the placebo p-value."""
    cond = event_excess(frame, events, window=window, lag=lag, col=col)
    base = unconditional_excess(frame, window=window, lag=lag, col=col)
    pl = placebo_pvalue(frame, events, window=window, lag=lag, col=col)
    return {
        "window": window,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "t": one_sample_t(cond),
        "t_hac": hac_t_daily(frame, events, window=window, lag=lag, col=col),
        "p_placebo": pl["p_value"],
    }


def net_of_costs(frame: pd.DataFrame, events: pd.DatetimeIndex,
                 window: int = DEFAULT_WINDOW, cost_bps: float = 10.0,
                 lag: int = 1, col: str = "excess") -> dict:
    """Buy-the-basket / short-SPY market-neutral trade after each shock, hold ``window`` days.

    One round-trip cost in bps applied per trade. Because the trade is *two-sided* (long the
    basket, short SPY) the round-trip touches four legs; we charge ``cost_bps`` once as the
    all-in round-trip on the pair, the conservative-friendly reading. The point is that the
    per-trade number is intact net of costs — frequency (~20 trades ever), not cost, is the
    binding constraint.
    """
    gross = event_excess(frame, events, window=window, lag=lag, col=col)
    c = cost_bps / 1e4
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }
