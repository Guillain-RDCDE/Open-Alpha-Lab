"""Strategy + inference for Study 379 — ETF Lead-Lag.

The believers' rule (on daily bars here, since intraday tick data is not free):

    The leader ETF (SPY) digests information first; its smaller, slower members catch up a
    day later. So *yesterday's* leader return should predict *today's* member return — and a
    tradable rule is: when the leader pops up (top-decile day), go long the laggard basket on
    the *next* bar.

We test it three ways:

  * a **lead-lag cross-correlation profile** — corr(leader_{t-k}, members_t) for k = -5..+5;
    a genuine lead shows a positive bump at k = +1 (leader leads members by a day);
  * a **next-bar rule** — enter the laggard basket the day after a big-leader day, hold one
    day, net of **one-way costs**, with a **Welch t** of the conditional next-day member
    return vs the unconditional mean and a **placebo / bootstrap null** sized to the trade
    count, plus a **win-rate** vs the unconditional base rate;
  * a deterministic **synthetic control** with a *planted* one-day lead (``data.synthetic_leadlag``)
    that the profile and the rule must recover — and must NOT find when the lead is zero.

The decisive finding is not the contemporaneous co-movement (members move *with* the leader
today — large, real, and **untradable**, because you can't act on information you don't yet
have) but the *next-day* predictability net of costs. On daily SPY-vs-members bars that
next-day signal is tiny and dies under realistic one-way costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Lead-lag cross-correlation profile
# --------------------------------------------------------------------------- #
def crosscorr_profile(frame: pd.DataFrame, max_lag: int = 5) -> dict:
    """corr(leader_{t-k}, members_t) for k in [-max_lag, +max_lag].

    k > 0 means the *leader leads*: leader at t-k vs members at t. A genuine one-day lead
    (leader predicts next-day members) shows up as a positive correlation at **k = +1**.
    k = 0 is the (large, untradable) contemporaneous co-movement; k < 0 means members lead.

    Returns ``{lag: correlation}`` with integer keys from -max_lag..+max_lag.
    """
    lead = frame["leader"]
    mem = frame["members"]
    out = {}
    for k in range(-max_lag, max_lag + 1):
        # members_t vs leader_{t-k}  ==  mem vs lead.shift(k)
        a = mem
        b = lead.shift(k)
        pair = pd.concat([a, b], axis=1).dropna()
        if len(pair) < 3:
            out[k] = float("nan")
        else:
            out[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return out


# --------------------------------------------------------------------------- #
# Next-bar rule events + forward member return
# --------------------------------------------------------------------------- #
def big_leader_days(frame: pd.DataFrame, q: float = 0.90) -> pd.DatetimeIndex:
    """Dates on which the leader return is in the top ``q`` quantile (a 'big up' leader day).

    These are the signal days; the rule enters the laggard basket on the *next* bar.
    """
    lead = frame["leader"]
    thr = lead.quantile(q)
    return frame.index[lead >= thr]


def next_day_member_returns(frame: pd.DataFrame, events: pd.DatetimeIndex) -> np.ndarray:
    """Member-basket return on the bar **after** each signal day (1-day execution lag).

    Signal known at the close of the big-leader day t; we earn the member return of t+1.
    Events at the very end of the tape (no t+1) are dropped.
    """
    mem = frame["members"]
    pos = {d: i for i, d in enumerate(frame.index)}
    n = len(frame)
    out = []
    for d in events:
        i = pos.get(d)
        if i is None or i + 1 >= n:
            continue
        out.append(mem.iloc[i + 1])
    return np.asarray(out, dtype=float)


def unconditional_member_returns(frame: pd.DataFrame) -> np.ndarray:
    """All daily member-basket returns (the base-rate distribution for the next-day rule)."""
    return np.asarray(frame["members"].values, dtype=float)


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


def placebo_pvalue(frame: pd.DataFrame, events: pd.DatetimeIndex,
                   n_draws: int = 20_000, seed: int = 379) -> dict:
    """Small-sample placebo null: draw ``len(events)`` random next-day member returns many
    times and ask how often a random draw's mean matches/beats the conditional set.

    Returns the conditional mean, the placebo mean, and the empirical p-value
    P[random-draw mean >= conditional mean] — the honest "could random next days have looked
    this good?" test for a next-bar rule.
    """
    obs = next_day_member_returns(frame, events)
    k = len(obs)
    mem = frame["members"].values
    n = len(mem)
    if k == 0 or n < 2:
        return {"k": k, "cond_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, n, size=k)
        means[i] = mem[pick].mean()
    obs_mean = float(obs.mean())
    p = float((means >= obs_mean).mean())
    return {"k": k, "cond_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, events: pd.DatetimeIndex) -> dict:
    """Headline stats for the next-bar rule: n trades, conditional next-day member mean and
    win-rate, the unconditional base-rate mean/win-rate, Welch t, and the placebo p-value."""
    cond = next_day_member_returns(frame, events)
    base = unconditional_member_returns(frame)
    pl = placebo_pvalue(frame, events)
    return {
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


def net_of_costs(frame: pd.DataFrame, events: pd.DatetimeIndex,
                 cost_bps: float = 5.0) -> dict:
    """Next-bar long-the-laggards rule, one-way cost in bps per entry (round-turn = 2x).

    The rule opens a one-day position the bar after each big-leader day, so each trade pays
    one entry + one exit = ``2 * cost_bps``. Returns the average gross and net trade return.
    (One-way cost × turnover; this is a single-day hold so turnover is one round-turn/trade.)
    """
    gross = next_day_member_returns(frame, events)
    c = 2.0 * cost_bps / 1e4                  # entry + exit, one-way each
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }


def lag1_beta(frame: pd.DataFrame) -> dict:
    """OLS of members_t on leader_{t-1} (the one-day lead-lag slope) with a Newey-West-style
    HAC t. The slope is the planted-lead analogue; a real next-day lead gives slope > 0 at t≥2.

    Uses a simple HAC (Newey-West, 5 lags) standard error so daily autocorrelation/overlap
    doesn't inflate the t. Returns slope, HAC t, and n.
    """
    mem = frame["members"]
    lead1 = frame["leader"].shift(1)
    pair = pd.concat([mem, lead1], axis=1).dropna()
    y = pair.iloc[:, 0].values
    x = pair.iloc[:, 1].values
    n = len(y)
    if n < 10:
        return {"slope": float("nan"), "t_hac": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    # Newey-West HAC with L lags
    L = 5
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    se_slope = np.sqrt(cov[1, 1])
    t = float(beta[1] / se_slope) if se_slope > 0 else float("nan")
    return {"slope": float(beta[1]), "t_hac": t, "n": n}
