"""Strategy + inference for Study 387 — Economic-Surprise-Index.

The claim, operationalised on a monthly frame of ``esi`` (the constructed surprise index) and
``spy`` (month-end close):

    When the data is beating expectations (ESI high / positive), be long stocks; the surprise
    is supposed to predict a higher forward return.

We test it three ways:

  * **Conditional vs unconditional forward returns.** Mean H-month forward SPY return when ESI
    is positive (a 'beat' month) vs the unconditional base rate, with a Welch *t*.
  * **A placebo / randomization null.** Resample the same number of random entry dates many
    times and ask how often chance matches the beat-month set — the honest small-sample test.
  * **A timing backtest, net of costs.** A long/flat (or long/short) rule that holds SPY when
    ESI > 0, with a one-month execution lag and a one-way cost per turn, raced against
    buy-and-hold on a Sharpe basis.

The decisive object is the confound: a surprise index built against a *trailing-average*
consensus is a slow autocorrelation signal on macro momentum, and most of its apparent edge
is the market's own post-recession recovery drift rather than a distinct nowcast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = {1: 1, 3: 3, 6: 6, 12: 12}
ANN = 12  # months per year


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(spy: pd.Series, horizon: int) -> pd.Series:
    """``horizon``-month forward simple return for every month-end (NaN near the tail)."""
    return spy.shift(-horizon) / spy - 1.0


def beat_months(frame: pd.DataFrame, thr: float = 0.0, lag: int = 1) -> pd.Series:
    """Boolean Series: was the ESI 'beating' (> ``thr``) as known ``lag`` months earlier?

    A signal at the end of month ``t`` is acted on from month ``t+lag`` (one-month execution
    lag by default — no look-ahead).
    """
    return (frame["esi"].shift(lag) > thr)


def conditional_returns(frame: pd.DataFrame, horizon: int, thr: float = 0.0,
                        lag: int = 1) -> np.ndarray:
    """Forward ``horizon``-month SPY returns over months entered after a 'beat' signal."""
    fwd = forward_returns(frame["spy"], horizon)
    sig = beat_months(frame, thr=thr, lag=lag)
    sel = fwd[sig].dropna()
    return np.asarray(sel.values, dtype=float)


def unconditional_returns(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """All overlapping ``horizon``-month forward SPY returns (the base-rate distribution)."""
    return np.asarray(forward_returns(frame["spy"], horizon).dropna().values, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    if len(sample) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(frame: pd.DataFrame, horizon: int, thr: float = 0.0, lag: int = 1,
                   n_draws: int = 20_000, seed: int = 387) -> dict:
    """Small-sample placebo null: draw ``k`` random entry months (k = number of beat months)
    many times and ask how often a random draw's mean forward return matches/beats the
    conditional set. Returns the conditional mean, placebo mean, and p = P[random >= obs]."""
    obs = conditional_returns(frame, horizon, thr=thr, lag=lag)
    k = len(obs)
    fwd = forward_returns(frame["spy"], horizon).dropna().values
    n = len(fwd)
    if k == 0 or n == 0:
        return {"k": k, "cond_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = fwd[rng.integers(0, n, size=k)].mean()
    obs_mean = float(obs.mean())
    return {"k": k, "cond_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs_mean).mean())}


def summarize(frame: pd.DataFrame, horizon: int, thr: float = 0.0, lag: int = 1) -> dict:
    """Headline stats for one horizon: n beat months, conditional mean/win-rate, the
    unconditional base rate, Welch t, and the placebo p-value."""
    cond = conditional_returns(frame, horizon, thr=thr, lag=lag)
    base = unconditional_returns(frame, horizon)
    pl = placebo_pvalue(frame, horizon, thr=thr, lag=lag)
    return {
        "horizon": horizon,
        "n": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Timing backtest, net of costs (the Tradability axis)
# --------------------------------------------------------------------------- #
def monthly_returns(frame: pd.DataFrame) -> pd.Series:
    """One-month simple SPY returns on the month-end tape."""
    return (frame["spy"] / frame["spy"].shift(1) - 1.0).dropna()


def timing_backtest(frame: pd.DataFrame, thr: float = 0.0, lag: int = 1,
                    cost_bps: float = 10.0, allow_short: bool = False) -> dict:
    """Long/flat (or long/short) SPY-timing rule driven by the ESI sign.

    Position for month ``m`` is decided by the ESI known ``lag`` months earlier:
    +1 when ESI > ``thr``; 0 (long/flat) or -1 (long/short) otherwise. A one-way cost of
    ``cost_bps`` is charged on each change of position (turnover, both legs counted by the
    diff). Returns gross/net annualized return, vol, and Sharpe for the rule and for
    buy-and-hold (both excess of zero — a price-only series carries no cash leg here, so the
    race is rule-vs-hold on the same price tape; labelled price-only)."""
    ret = monthly_returns(frame)
    pos_raw = np.where(frame["esi"].shift(lag) > thr, 1.0,
                       (-1.0 if allow_short else 0.0))
    pos = pd.Series(pos_raw, index=frame.index).reindex(ret.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    c = cost_bps / 1e4
    gross = pos * ret
    net = gross - turn * c

    def _stats(r: pd.Series) -> dict:
        mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
        return {"ann_ret": float(mu), "ann_vol": float(sd),
                "sharpe": float(mu / sd) if sd > 0 else float("nan")}

    return {
        "n_months": int(len(ret)),
        "n_turns": float(turn.sum()),
        "exposure": float((pos != 0).mean()),
        "gross": _stats(gross),
        "net": _stats(net),
        "buy_hold": _stats(ret),
        "cost_bps": cost_bps,
        "allow_short": allow_short,
    }
