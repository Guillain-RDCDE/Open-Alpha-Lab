"""Strategy + inference for Study 386 — NFCI-Conditions.

The folklore rule (on our conditions proxy here, since the true NFCI is not reachable):

    Let ``fci`` be the weekly financial-conditions index (high = tight). When conditions are
    **tight** (``fci`` above a threshold, default its top tercile / +0.5 z), step OUT of
    stocks (hold cash); otherwise hold the S&P 500. The pitch: tightness is a regime switch —
    it tells you to de-risk before equities fall.

We separate two things the pitch fuses:

  * **Contemporaneous** link — do tight *weeks* coincide with down *weeks*? (Mechanical: the
    index is partly built from equity vol, so this is near-tautological and predicts nothing.)
  * **Forward / predictive** link — does tightness measured this week beat the base rate for
    *next* week / next month's return? That is the only thing a trader can act on.

Inference on the forward link:
  * a **Welch t** of the conditional (tight-week) forward mean vs the unconditional mean;
  * a **placebo / randomization null** — draw the same number of random weeks and ask how
    often a random draw is at least as bearish as the tight set (the honest small-sample test);
  * a **win-rate** vs the unconditional base rate;
  * a **timing backtest**: step out when tight, with a **one-week execution lag** (signal known
    at week ``t``'s close earns week ``t+1``'s return) and **one-way costs** charged on each
    switch, raced excess-of-cash vs the buy-and-hold benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS = {1: 1, 4: 4, 13: 13}      # forward horizons in weeks (1w, ~1m, ~1q)


# --------------------------------------------------------------------------- #
# Regime split
# --------------------------------------------------------------------------- #
def tight_mask(fci: pd.Series, thresh: float = 0.5) -> pd.Series:
    """Boolean Series: True on weeks whose conditions index is at/above ``thresh`` (tight).

    Default +0.5 z is roughly the top-tercile tightness cut on a standardised index.
    """
    return fci >= thresh


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(spy: pd.Series, horizon_w: int, lag: int = 1) -> pd.Series:
    """Forward ``horizon_w``-week return entered ``lag`` weeks after the signal week.

    Signal known at the close of week ``t`` earns the return from ``t+lag`` to ``t+lag+h``
    (one execution lag, applied once — no look-ahead).
    """
    entry = spy.shift(-lag)
    exit_ = spy.shift(-lag - horizon_w)
    return exit_ / entry - 1.0


def conditional_forward(frame: pd.DataFrame, horizon_w: int, thresh: float = 0.5,
                        lag: int = 1) -> np.ndarray:
    """Forward returns following *tight* weeks (1-week execution lag)."""
    fwd = forward_returns(frame["spy"], horizon_w, lag=lag)
    m = tight_mask(frame["fci"], thresh=thresh)
    return fwd[m].dropna().values


def unconditional_forward(frame: pd.DataFrame, horizon_w: int, lag: int = 1) -> np.ndarray:
    """All overlapping forward returns (the base-rate distribution)."""
    return forward_returns(frame["spy"], horizon_w, lag=lag).dropna().values


def contemporaneous_corr(frame: pd.DataFrame) -> float:
    """Correlation between this-week's FCI *change* and this-week's S&P return.

    This is the mechanical, NON-predictive link (the index embeds equity vol). A strongly
    negative number here is expected and proves nothing about timing."""
    r = np.log(frame["spy"]).diff()
    df = frame["fci"].diff()
    both = pd.concat([r, df], axis=1).dropna()
    return float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])


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


def placebo_pvalue(frame: pd.DataFrame, horizon_w: int, thresh: float = 0.5,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 386) -> dict:
    """Small-sample placebo null for the *bearish* claim.

    Draw ``k`` random weeks (k = number of tight weeks) many times and ask how often a random
    draw's mean forward return is **at or below** the tight set's mean (i.e. how often chance
    looks as bearish). ``p`` small => tight weeks really are unusually bearish forward.
    """
    fwd_all = forward_returns(frame["spy"], horizon_w, lag=lag).dropna()
    cond = conditional_forward(frame, horizon_w, thresh=thresh, lag=lag)
    k = len(cond)
    if k == 0 or len(fwd_all) == 0:
        return {"k": 0, "tight_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    vals = fwd_all.values
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = vals[rng.integers(0, len(vals), size=k)].mean()
    obs = float(cond.mean())
    p = float((means <= obs).mean())          # share of random draws at least as bearish
    return {"k": k, "tight_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def summarize(frame: pd.DataFrame, horizon_w: int, thresh: float = 0.5,
              lag: int = 1) -> dict:
    """Headline stats for one forward horizon: n tight weeks, conditional vs base mean and
    win-rate, Welch t, placebo p (bearish-tail)."""
    cond = conditional_forward(frame, horizon_w, thresh=thresh, lag=lag)
    base = unconditional_forward(frame, horizon_w, lag=lag)
    pl = placebo_pvalue(frame, horizon_w, thresh=thresh, lag=lag)
    return {
        "horizon_w": horizon_w,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Timing backtest
# --------------------------------------------------------------------------- #
def backtest_timing(frame: pd.DataFrame, thresh: float = 0.5, cost_bps: float = 5.0,
                    lag: int = 1) -> dict:
    """Step-out-when-tight timing rule vs buy-and-hold, weekly, one execution lag, one-way
    costs charged on each switch.

    Position is 1 (in stocks) when last week's FCI is loose, 0 (cash) when tight; the signal
    at week ``t`` sets the position for week ``t+lag`` (one lag). Cost ``cost_bps`` one-way is
    charged whenever the position changes. Returns annualised mean, vol, Sharpe and the net
    edge over buy-and-hold (all on weekly log-ish simple returns; cash earns 0 — a
    conservative excess-of-cash race since the rule sits in cash part-time)."""
    spy = frame["spy"]
    wk_ret = spy.pct_change().fillna(0.0)
    sig = tight_mask(frame["fci"], thresh=thresh).astype(float)
    pos = (1.0 - sig).shift(lag).fillna(1.0)           # in stocks unless tight (lagged)
    switches = pos.diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    strat_ret = pos * wk_ret - switches * c
    bh_ret = wk_ret

    def _ann(r):
        mu = r.mean() * 52
        sd = r.std(ddof=1) * np.sqrt(52)
        return mu, sd, (mu / sd if sd else float("nan"))

    s_mu, s_sd, s_sh = _ann(strat_ret)
    b_mu, b_sd, b_sh = _ann(bh_ret)
    n_switch = int(switches.sum())
    return {
        "weeks": int(len(frame)),
        "n_switches": n_switch,
        "pct_in_cash": float((pos == 0).mean()),
        "strat_cagr": float((1 + strat_ret).prod() ** (52 / len(frame)) - 1),
        "bh_cagr": float((1 + bh_ret).prod() ** (52 / len(frame)) - 1),
        "strat_ann_ret": float(s_mu), "strat_ann_vol": float(s_sd), "strat_sharpe": float(s_sh),
        "bh_ann_ret": float(b_mu), "bh_ann_vol": float(b_sd), "bh_sharpe": float(b_sh),
        "sharpe_gap": float(s_sh - b_sh),
        "cost_bps": cost_bps,
    }
