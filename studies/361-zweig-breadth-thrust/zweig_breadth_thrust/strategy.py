"""Strategy + inference for Study 361 — Zweig Breadth Thrust.

Zweig's rule (on a breadth proxy here, since true NYSE breadth is not free):

    Let ``e`` = 10-day EMA of the advance ratio (advancers / (adv+dec)).
    A **Breadth Thrust** fires on the day ``e`` first rises above 0.615 having been
    below 0.40 within the prior ``window`` (~10) trading days.

The pitch: such thrusts are "never wrong" — a powerful bull move always follows. We test it
by measuring forward SPY returns at 3/6/12 months after each thrust vs the *unconditional*
forward-return distribution, with:

  * a **t-stat** of the conditional mean against the unconditional mean (Welch);
  * a **bootstrap / placebo** null — draw the same number of random dates and ask how often
    a random draw beats the thrust set (the honest small-sample test);
  * a **win-rate** (P[forward return > 0]) vs the unconditional base rate;
  * **one-day execution lag** and **one-way costs** applied to a simple "buy on the thrust,
    hold H months" rule.

The decisive number is not the point estimate (it is positive, as the folklore says) but
the sample size: with ~a dozen lifetime events, the conditional mean is statistically
indistinguishable from luck — exactly the Study 167 (Hindenburg) pathology, inverted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {3: 63, 6: 126, 12: 252}      # months -> trading days


# --------------------------------------------------------------------------- #
# Thrust detection
# --------------------------------------------------------------------------- #
def breadth_ema(adv_ratio: pd.Series, span: int = 10) -> pd.Series:
    """Zweig's 10-day EMA of the advance ratio."""
    return adv_ratio.ewm(span=span, adjust=False).mean()


def detect_thrusts(frame: pd.DataFrame, lo: float = 0.40, hi: float = 0.615,
                   window: int = 10, span: int = 10,
                   cooldown: int = 252) -> pd.DatetimeIndex:
    """Dates on which a Zweig Breadth Thrust completes.

    A thrust completes on the first day the EMA crosses **above** ``hi`` provided the EMA was
    **below** ``lo`` at some point within the prior ``window`` trading days. ``cooldown``
    trading days suppress repeat fires from the same up-leg (one event per thrust).
    """
    e = breadth_ema(frame["adv_ratio"], span=span).values
    n = len(e)
    below_lo = e < lo
    # rolling "was below lo within prior `window` days" (inclusive of today's lookback)
    was_low = np.zeros(n, dtype=bool)
    for t in range(n):
        a = max(0, t - window)
        was_low[t] = below_lo[a:t + 1].any()
    cross_up = np.zeros(n, dtype=bool)
    cross_up[1:] = (e[1:] > hi) & (e[:-1] <= hi)
    fire = cross_up & was_low
    idx = frame.index
    events = []
    last = -10 ** 9
    for t in np.where(fire)[0]:
        if t - last < cooldown:
            continue
        events.append(idx[t])
        last = t
    return pd.DatetimeIndex(events)


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(spy: pd.Series, horizon_days: int) -> pd.Series:
    """Simple forward return over ``horizon_days`` for every date (NaN near the tail)."""
    fwd = spy.shift(-horizon_days) / spy - 1.0
    return fwd


def event_returns(frame: pd.DataFrame, events: pd.DatetimeIndex,
                  months: int, lag: int = 1) -> np.ndarray:
    """Forward ``months``-month SPY returns following each thrust, with a 1-day entry lag.

    Entry is the close ``lag`` days after the signal (no look-ahead); exit ``horizon`` days
    later. Events whose horizon runs past the tape are dropped.
    """
    h = TRADING_DAYS[months]
    spy = frame["spy"]
    pos = {d: i for i, d in enumerate(frame.index)}
    out = []
    n = len(frame)
    for d in events:
        i = pos.get(d)
        if i is None:
            continue
        entry = i + lag
        exit_ = entry + h
        if exit_ >= n:
            continue
        out.append(spy.iloc[exit_] / spy.iloc[entry] - 1.0)
    return np.asarray(out, dtype=float)


def unconditional_returns(frame: pd.DataFrame, months: int) -> np.ndarray:
    """All overlapping ``months``-month forward SPY returns (the base-rate distribution)."""
    h = TRADING_DAYS[months]
    fwd = forward_returns(frame["spy"], h).dropna().values
    return np.asarray(fwd, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    if len(sample) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(frame: pd.DataFrame, events: pd.DatetimeIndex, months: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 361) -> dict:
    """Small-sample placebo null: draw ``len(events)`` random valid entry dates many times
    and ask how often a random draw's mean forward return matches/beats the thrust set.

    Returns the thrust conditional mean, the placebo mean, and the empirical p-value
    P[random-draw mean >= thrust mean] — the honest answer to "could a dozen random dates
    have looked this good?"
    """
    h = TRADING_DAYS[months]
    spy = frame["spy"].values
    n = len(spy)
    obs = event_returns(frame, events, months, lag=lag)
    k = len(obs)
    if k == 0:
        return {"k": 0, "thrust_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    valid_hi = n - h - lag - 1
    rng = np.random.default_rng(seed)
    fwd = spy[lag + h:lag + h + valid_hi] / spy[lag:lag + valid_hi] - 1.0
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, valid_hi, size=k)
        means[i] = fwd[pick].mean()
    obs_mean = float(obs.mean())
    p = float((means >= obs_mean).mean())
    return {"k": k, "thrust_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, events: pd.DatetimeIndex, months: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, conditional mean/win-rate, unconditional
    base-rate, Welch t, and the placebo p-value."""
    cond = event_returns(frame, events, months, lag=lag)
    base = unconditional_returns(frame, months)
    pl = placebo_pvalue(frame, events, months, lag=lag)
    return {
        "months": months,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


def net_of_costs(frame: pd.DataFrame, events: pd.DatetimeIndex, months: int,
                 cost_bps: float = 5.0, lag: int = 1) -> dict:
    """Buy-on-thrust / hold-``months`` rule, one round-trip cost in bps applied per trade.

    Returns the average gross and net trade return and the implied annualised contribution
    (purely illustrative — the point is that ~a dozen trades a *century* is not a NAV-scale
    strategy regardless of the per-trade number)."""
    gross = event_returns(frame, events, months, lag=lag)
    c = cost_bps / 1e4
    net = gross - c                      # one round-trip per trade
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }
