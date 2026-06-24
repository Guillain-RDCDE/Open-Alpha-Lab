"""Strategy + honest inference for Study 444 — Dow Theory confirmation.

The folklore (Charles Dow, via Hamilton & Rhea): a primary trend is only **valid**
when the **Industrials** and the **Transports** *confirm* each other. A bull market
is in force when **both** averages make new **higher highs**; the trend is suspect
(a non-confirmation / sell signal) when one makes a new high the other fails to
match, and is broken when both make **lower lows**.

We encode the tightest mechanical rule a proponent would accept:

    * **Confirmation regime.** Both the Industrials and the Transports close at or
      above their own ``lookback``-day trailing high → the primary uptrend is
      *confirmed* (the "higher-high in both averages" rule). This is a latched
      regime: once confirmed it stays on until a **non-confirmation breakdown** —
      both averages closing below their ``lookback``-day trailing **low** — flips it
      off. (Latching is what Dow Theorists mean by "the primary trend persists until
      reversed"; we also test the un-latched instantaneous flag.)

    * **The trade.** Be **long the Industrials** (the thing you hold) only while the
      regime is confirmed; otherwise sit in **cash** (the defensive reading of "stay
      out when the averages disagree"). Signal known at close *t*, position earns the
      Industrials return of *t+1* — one documented execution lag.

Inference, the desk's shared spirit:

  * the strategy's **daily excess return** (long-when-confirmed minus the
    risk-free-equivalent of cash = 0) vs **buy-and-hold** the Industrials, both
    excess-of-cash, raced on Sharpe;
  * a **HAC (Newey-West) one-sample t** on the *active-vs-benchmark daily spread*
    (does timing the trend with the Transports add anything to just holding?);
  * a **placebo** that replaces the real confirmation regime with a random regime of
    the *same on-fraction and same average run length* — the honest "is it the
    Transports, or just being in cash 40% of the time?" null;
  * **one-way costs × turnover** charged on every regime flip.

The decisive number is the HAC t on the active-minus-benchmark spread: if Dow Theory
adds real timing value it must clear t ≥ 2 there. Most market-timing folklore does
not — the headline is whether this one does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 63            # ~one quarter of trading days defining "the recent high/low"
ANN = 252.0


# --------------------------------------------------------------------------- #
# The confirmation regime
# --------------------------------------------------------------------------- #
def trailing_high(s: pd.Series, lookback: int) -> pd.Series:
    """Trailing ``lookback``-day high *as of the prior close* (shifted, no look-ahead)."""
    return s.rolling(lookback, min_periods=lookback).max().shift(1)


def trailing_low(s: pd.Series, lookback: int) -> pd.Series:
    return s.rolling(lookback, min_periods=lookback).min().shift(1)


def confirmation_flag(prices: pd.DataFrame, lookback: int = LOOKBACK,
                      tol: float = 0.0) -> pd.Series:
    """Instantaneous (un-latched) confirmation: BOTH averages at/above their N-day high.

    Day ``t`` is confirmed if ``close_t >= (1 - tol) * trailing_high_{t}`` for *both*
    the Industrials and the Transports, where the trailing high is computed through
    ``t-1`` (so the flag uses only information available at the close of ``t``).
    """
    ind, tra = prices["indu"], prices["tran"]
    hi_i = trailing_high(ind, lookback)
    hi_t = trailing_high(tra, lookback)
    conf = (ind >= (1 - tol) * hi_i) & (tra >= (1 - tol) * hi_t)
    return conf.fillna(False)


def latched_regime(prices: pd.DataFrame, lookback: int = LOOKBACK,
                   tol: float = 0.0) -> pd.Series:
    """Latched Dow-Theory regime: ON at a both-averages new-high *confirmation*, OFF at a
    both-averages new-low *breakdown*; persists between signals (the "primary trend"
    idea). Returns a 0/1 series aligned to ``prices`` (state known at each close).
    """
    ind, tra = prices["indu"], prices["tran"]
    hi_i = trailing_high(ind, lookback)
    hi_t = trailing_high(tra, lookback)
    lo_i = trailing_low(ind, lookback)
    lo_t = trailing_low(tra, lookback)
    confirm = (ind >= (1 - tol) * hi_i) & (tra >= (1 - tol) * hi_t)
    breakdown = (ind <= (1 + tol) * lo_i) & (tra <= (1 + tol) * lo_t)

    state = np.zeros(len(prices), dtype=float)
    cur = 0.0
    cf = confirm.to_numpy()
    bd = breakdown.to_numpy()
    for i in range(len(prices)):
        if cf[i]:
            cur = 1.0
        elif bd[i]:
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=prices.index)


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
def backtest(prices: pd.DataFrame, regime: pd.Series, cost_bps: float = 2.0,
             lag: int = 1) -> pd.DataFrame:
    """Long-Industrials-when-regime-ON, cash otherwise; one-day execution lag.

    Returns a daily frame with: ``r_indu`` (the Industrials daily return, the
    benchmark = buy-and-hold), ``pos`` (lagged regime weight in {0,1}),
    ``r_strat_gross`` / ``r_strat_net`` (the timed book), and ``r_active`` (net
    strategy minus benchmark — the value-add of timing). Cash earns 0 (so all
    Sharpes below are excess-of-cash on both sides — a fair excess-vs-excess race).
    """
    r_indu = prices["indu"].pct_change().fillna(0.0)
    pos = regime.shift(lag).fillna(0.0).clip(0, 1)
    turn = pos.diff().abs().fillna(pos.abs())          # one-way turnover per day
    cost = turn * (cost_bps / 1e4)
    r_gross = pos * r_indu
    r_net = r_gross - cost
    out = pd.DataFrame({
        "r_indu": r_indu,
        "pos": pos,
        "r_strat_gross": r_gross,
        "r_strat_net": r_net,
        "r_active": r_net - r_indu,                     # timing value-add (net)
    }, index=prices.index)
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def sharpe(r: np.ndarray, ann: float = ANN) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(ann))


def cagr(r: np.ndarray, ann: float = ANN) -> float:
    r = np.asarray(r, dtype=float)
    g = np.prod(1.0 + r)
    yrs = len(r) / ann
    if yrs <= 0 or g <= 0:
        return float("nan")
    return float(g ** (1.0 / yrs) - 1.0)


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def hac_t(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC one-sample t of ``x`` against 0 (autocorrelation-robust)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _run_lengths(flag: np.ndarray):
    """Lengths of consecutive ON runs and the on-fraction of a 0/1 array."""
    flag = np.asarray(flag, dtype=float)
    on = flag > 0.5
    runs = []
    c = 0
    for v in on:
        if v:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return runs, float(on.mean())


def placebo_pvalue(prices: pd.DataFrame, regime: pd.Series, cost_bps: float = 2.0,
                   n_draws: int = 2000, seed: int = 444, lag: int = 1) -> dict:
    """Random-regime placebo: does the *Transports confirmation* matter, or just being
    in cash part of the time with similar persistence?

    Build random 0/1 regimes that match the real regime's **on-fraction** and average
    **run length** (a two-state Markov chain with the same stay-probabilities), run the
    same backtest, and ask how often a random regime's net Sharpe **beats** the real
    one. ``p`` = P[random Sharpe >= observed].
    """
    r_indu = prices["indu"].pct_change().fillna(0.0).to_numpy()
    n = len(prices)
    runs, on_frac = _run_lengths(regime.to_numpy())
    mean_on = np.mean(runs) if runs else 5.0
    off_frac = max(1.0 - on_frac, 1e-6)
    mean_off = mean_on * off_frac / max(on_frac, 1e-6)
    p_stay_on = max(1.0 - 1.0 / max(mean_on, 1.0), 0.0)
    p_stay_off = max(1.0 - 1.0 / max(mean_off, 1.0), 0.0)

    obs = backtest(prices, regime, cost_bps=cost_bps, lag=lag)
    obs_sharpe = sharpe(obs["r_strat_net"].to_numpy())

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_draws)
    cost = cost_bps / 1e4
    for d in range(n_draws):
        flag = np.zeros(n)
        state = 0
        for i in range(n):
            if state == 1:
                state = 1 if rng.random() < p_stay_on else 0
            else:
                state = 0 if rng.random() < p_stay_off else 1
            flag[i] = state
        pos = np.concatenate([[0.0], flag[:-1]])       # one-day lag
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        r_net = pos * r_indu - turn * cost
        sharpes[d] = sharpe(r_net)
    p = float((sharpes >= obs_sharpe).mean())
    return {"obs_sharpe": float(obs_sharpe), "placebo_mean": float(np.nanmean(sharpes)),
            "p_value": p, "on_frac": on_frac, "draws": sharpes}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance). NaN if either has < 2 points."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def confirmed_vs_not(prices: pd.DataFrame, lookback: int = LOOKBACK,
                     latched: bool = True, tol: float = 0.0, lag: int = 1) -> dict:
    """Direct content test: is the Industrials' *next-day* return higher on confirmed
    days than on non-confirmed days?

    This isolates the confirmation flag's predictive value from the cash-drag of the
    timing book (which just dodges volatility). The signal is known at close ``t``; we
    compare the return earned at ``t + lag``. Returns both means, the spread, the
    Welch t, and the share of confirmed days. The clean number the synthetic control
    targets: with a planted edge it must be strongly positive; with none, ~0.
    """
    regime = (latched_regime(prices, lookback, tol) if latched
              else confirmation_flag(prices, lookback, tol).astype(float))
    r_next = prices["indu"].pct_change().shift(-lag)        # return earned after the signal
    on = regime.to_numpy() > 0.5
    rn = r_next.to_numpy()
    valid = np.isfinite(rn)
    on_r = rn[valid & on]
    off_r = rn[valid & ~on]
    return {
        "n_on": int(on_r.size), "n_off": int(off_r.size),
        "mean_on_ann": float(np.nanmean(on_r) * ANN) if on_r.size else float("nan"),
        "mean_off_ann": float(np.nanmean(off_r) * ANN) if off_r.size else float("nan"),
        "spread_ann": float((np.nanmean(on_r) - np.nanmean(off_r)) * ANN)
        if (on_r.size and off_r.size) else float("nan"),
        "welch_t": welch_t(on_r, off_r),
        "conf_frac": float(on.mean()),
    }


def summarize(prices: pd.DataFrame, lookback: int = LOOKBACK, cost_bps: float = 2.0,
              latched: bool = True, lag: int = 1, tol: float = 0.0) -> dict:
    """Headline stats: buy-and-hold vs Dow-Theory-timed Industrials.

    Reports CAGR / Sharpe / max-drawdown for both legs (net), the active-spread HAC t
    (the decisive significance number), turnover, and the on-fraction.
    """
    regime = (latched_regime(prices, lookback, tol) if latched
              else confirmation_flag(prices, lookback, tol).astype(float))
    bt = backtest(prices, regime, cost_bps=cost_bps, lag=lag)
    bh = bt["r_indu"].to_numpy()
    strat = bt["r_strat_net"].to_numpy()
    active = bt["r_active"].to_numpy()
    flips = float(regime.shift(lag).fillna(0).diff().abs().sum())
    yrs = len(prices) / ANN
    return {
        "n_days": int(len(prices)), "years": float(yrs),
        "on_frac": float(bt["pos"].mean()),
        "flips": flips, "flips_per_yr": float(flips / yrs) if yrs else float("nan"),
        "bh_cagr": cagr(bh), "bh_sharpe": sharpe(bh), "bh_mdd": max_drawdown(bh),
        "st_cagr": cagr(strat), "st_sharpe": sharpe(strat), "st_mdd": max_drawdown(strat),
        "active_mean_ann": float(np.nanmean(active) * ANN),
        "active_t": hac_t(active),
        "lookback": lookback, "cost_bps": cost_bps, "latched": latched,
    }
