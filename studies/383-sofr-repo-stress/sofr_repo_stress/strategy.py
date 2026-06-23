"""Strategy + inference for Study 383 — SOFR-Repo-Stress.

The claim (steelmanned): a spike in overnight repo / SOFR — the plumbing seizing — is a
forward *warning* for risk assets. If true, the forward return of equities (SPY) and
credit (HYG / LQD) after a repo-stress episode should be **meaningfully worse than the
unconditional base rate** — negative enough, and reliable enough, that you'd de-risk on
the signal.

We test it by measuring forward returns at 5 / 20 / 60 trading days after each named
repo-stress episode vs the *unconditional* forward-return distribution, with:

  * a **Welch t** of the conditional mean against the unconditional mean (a *negative* t
    is the believer's hoped-for sign — risk assets do worse after a spike);
  * a **placebo / randomization** null — draw the same number of random dates and ask how
    often a random draw is at least as *bearish* as the episode set (the honest
    small-sample test, sized to the event count);
  * a **win-rate** (here the "warning hit-rate": P[forward return < 0]) vs the
    unconditional base rate;
  * **one-day execution lag** and **one-way costs** applied to a simple "de-risk on the
    spike, stay out H days" rule.

The decisive number is not the point estimate but the sample size and the confounding:
with ~a dozen named lifetime episodes — several of which were *not* bearish, and the one
true crash (March 2020) is COVID, not repo — the conditional mean is statistically
indistinguishable from luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# horizon label -> trading days
HORIZONS = {5: 5, 20: 20, 60: 60}


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(price: pd.Series, horizon_days: int) -> pd.Series:
    """Simple forward return over ``horizon_days`` for every date (NaN near the tail)."""
    return price.shift(-horizon_days) / price - 1.0


def event_returns(price: pd.Series, events: pd.DatetimeIndex,
                  horizon_days: int, lag: int = 1) -> np.ndarray:
    """Forward ``horizon_days`` returns of ``price`` following each episode, 1-day lag.

    Entry is the close ``lag`` days after the episode date (no look-ahead); exit
    ``horizon_days`` later. Episodes whose date isn't in the index snap to the next
    available trading day; episodes whose horizon runs past the tape are dropped.
    """
    idx = price.index
    vals = price.values
    n = len(idx)
    out = []
    for d in events:
        # snap to the first trading day on/after the episode date
        i = idx.searchsorted(pd.Timestamp(d), side="left")
        if i >= n:
            continue
        entry = i + lag
        exit_ = entry + horizon_days
        if exit_ >= n:
            continue
        out.append(vals[exit_] / vals[entry] - 1.0)
    return np.asarray(out, dtype=float)


def unconditional_returns(price: pd.Series, horizon_days: int) -> np.ndarray:
    """All overlapping ``horizon_days`` forward returns (the base-rate distribution)."""
    fwd = forward_returns(price, horizon_days).dropna().values
    return np.asarray(fwd, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2.

    A *negative* t is the believer's wanted direction here: risk assets do *worse*
    after a repo spike than on a random day.
    """
    if len(sample) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(price: pd.Series, events: pd.DatetimeIndex, horizon_days: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 383) -> dict:
    """Small-sample placebo null: draw ``len(events)`` random valid entry dates many
    times and ask how often a random draw is at least as *bearish* as the episode set.

    Returns the episode conditional mean, the placebo mean, and the empirical p-value
    P[random-draw mean <= episode mean] — the honest answer to "could a dozen random
    dates have looked this scary?" (Left tail, because the claim is a *bearish* warning.)
    """
    h = horizon_days
    p = price.values
    n = len(p)
    obs = event_returns(price, events, horizon_days, lag=lag)
    k = len(obs)
    if k == 0:
        return {"k": 0, "event_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    valid_hi = n - h - lag - 1
    rng = np.random.default_rng(seed)
    fwd = p[lag + h:lag + h + valid_hi] / p[lag:lag + valid_hi] - 1.0
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, valid_hi, size=k)
        means[i] = fwd[pick].mean()
    obs_mean = float(obs.mean())
    # left-tail: how often is a random draw at least as bearish (<=) as the episodes?
    pv = float((means <= obs_mean).mean())
    return {"k": k, "event_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": pv}


def summarize(price: pd.Series, events: pd.DatetimeIndex, horizon_days: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon on one asset: n, conditional mean, the 'warning
    hit-rate' P[ret<0], the unconditional base rate, Welch t, and the placebo p-value."""
    cond = event_returns(price, events, horizon_days, lag=lag)
    base = unconditional_returns(price, horizon_days)
    pl = placebo_pvalue(price, events, horizon_days, lag=lag)
    return {
        "h": horizon_days,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_down": float((cond < 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_down": float((base < 0).mean()),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


def net_of_costs(price: pd.Series, events: pd.DatetimeIndex, horizon_days: int,
                 cost_bps: float = 5.0, lag: int = 1) -> dict:
    """"De-risk on the spike" rule: the avoided loss is the *negative* of the forward
    return (you sidestep it), minus a one round-trip cost in bps per de-risking trade.

    Returns the average gross and net avoided-loss per trade. The point isn't the
    per-trade number — it is that ~a dozen de-riskings in 15 years can't be a NAV-scale
    strategy whatever the per-trade figure."""
    fwd = event_returns(price, events, horizon_days, lag=lag)
    avoided = -fwd                              # de-risking captures minus the move
    c = cost_bps / 1e4
    net = avoided - c
    return {
        "n_trades": int(len(fwd)),
        "gross_mean": float(avoided.mean()) if len(fwd) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }
