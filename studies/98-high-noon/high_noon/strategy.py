"""The analysis and its honest controls — Study 98 (High-Noon).

The folk claim: *"Never buy at an all-time high — it's the riskiest moment, a classic
sell signal; wait for a pullback."* We take it literally and ask whether buying SPY
*at/near an all-time high* leads to **worse** forward returns than buying when **not**
near a high. Two engines:

- **Conditional forward-return engine** — split every day into *at-ATH* (close within
  ``near_pct`` of the running all-time high) vs *not-at-ATH*, then compare the forward
  1/3/12-month total returns: means with HAC *t*-stats, win-rates with Wilson intervals,
  and a HAC test of the **difference** (at minus not-at). The honest, well-known finding
  is that ATHs are *not* bearish — they cluster in uptrends.
- **Avoid-the-highs backtest** — be invested only when **not** at an ATH, in cash
  (earning 0%, a stated conservative choice) otherwise, vs buy-and-hold. The folk rule
  predicts this *helps*; if ATHs are benign-or-bullish it should *hurt* (you sit out the
  best trends).

Execution convention: the at-ATH state is known at the close of day *t*; the forward
return it is matched to is the return *after* *t* (no peeking). For the backtest the
position is the at/not-at state shifted forward one day — one ``shift``, applied once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH = 21  # trading days per month, the desk convention


# ---------------------------------------------------------------------------
# The all-time-high flag
# ---------------------------------------------------------------------------
def at_ath(close: pd.Series, near_pct: float = 0.01) -> pd.Series:
    """Boolean: is the close within ``near_pct`` of the *running* all-time high?

    The running max is the cumulative max up to and including day *t* — so a fresh peak
    counts as at-ATH, and ``near_pct=0.0`` means *exactly* a new high. Known at close
    *t*; no look-ahead (the cummax never reaches into the future).
    """
    running_max = close.cummax()
    return close >= running_max * (1.0 - near_pct)


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Forward total return over ``horizon`` trading days: close[t+h]/close[t] - 1.

    The value at *t* is the return earned *after* *t*; the last ``horizon`` rows are NaN
    (no future to look at). This is the return an investor who buys at the close of *t*
    realises ``horizon`` days later.
    """
    return close.shift(-horizon) / close - 1.0


# ---------------------------------------------------------------------------
# Inference helpers (local, no quantlab dep)
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x``.

    Overlapping forward returns are strongly autocorrelated, so the HAC lag defaults to
    a generous Newey-West rule; pass an explicit ``lags`` (e.g. the horizon) to be safe.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def hac_diff_tstat(a: np.ndarray, b: np.ndarray, lags: int | None = None) -> float:
    """HAC t-stat for the difference of two means (mean(a) - mean(b)).

    The two arms are disjoint day-sets, so we compare their means via the pooled HAC SEs
    of each group's centred series. Conservative and dependency-aware.
    """
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if a.size <= 5 or b.size <= 5:
        return float("nan")

    def _lrv(x: np.ndarray, k: int) -> float:
        e = x - x.mean()
        n = x.size
        v = float(e @ e) / n
        for j in range(1, k + 1):
            w = 1.0 - j / (k + 1.0)
            v += 2.0 * w * float(e[j:] @ e[:-j]) / n
        return max(v, 0.0)

    la = lags if lags is not None else int(np.floor(4.0 * (a.size / 100.0) ** (2.0 / 9.0)))
    lb = lags if lags is not None else int(np.floor(4.0 * (b.size / 100.0) ** (2.0 / 9.0)))
    se2 = _lrv(a, la) / a.size + _lrv(b, lb) / b.size
    se = np.sqrt(se2)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (default 95%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------------------
# Conditional forward-return engine
# ---------------------------------------------------------------------------
def conditional_forward_stats(
    close: pd.Series, horizon: int, near_pct: float = 0.01
) -> dict:
    """Forward-return stats split by at-ATH vs not-at-ATH for one horizon.

    Returns means, win-rates (Wilson CIs), counts, HAC *t* on each arm's mean and on the
    **difference** (at minus not-at). HAC lags are set to the horizon (overlap length).
    A negative difference with a HAC \\|t\\| >= 2 would support "ATH is bearish".
    """
    flag = at_ath(close, near_pct)
    fwd = forward_return(close, horizon)
    df = pd.DataFrame({"flag": flag, "fwd": fwd}).dropna()
    at = df.loc[df["flag"], "fwd"].to_numpy()
    no = df.loc[~df["flag"], "fwd"].to_numpy()

    def _arm(x: np.ndarray) -> dict:
        n = x.size
        wins = int((x > 0).sum())
        lo, hi = wilson_interval(wins, n)
        return {
            "n": n,
            "mean": float(x.mean()) if n else float("nan"),
            "win_rate": wins / n if n else float("nan"),
            "win_lo": lo,
            "win_hi": hi,
            "hac_t": hac_tstat(x, lags=horizon),
        }

    return {
        "horizon": horizon,
        "near_pct": near_pct,
        "at": _arm(at),
        "not_at": _arm(no),
        "diff_mean": float(at.mean() - no.mean()) if at.size and no.size else float("nan"),
        "diff_hac_t": hac_diff_tstat(at, no, lags=horizon),
        "frac_at_ath": float(df["flag"].mean()),
    }


# ---------------------------------------------------------------------------
# Avoid-the-highs backtest vs buy-and-hold
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def avoid_highs_position(close: pd.Series, near_pct: float = 0.01, lag: int = 1) -> pd.Series:
    """Long-or-flat: invested (1) only when NOT at an ATH, in cash (0) when at an ATH.

    The at-ATH state is known at close *t*; the position is shifted forward by ``lag``
    days so it is acted on at *t+1*'s return — one lag, applied once. This is the folk
    rule taken literally: avoid the highs.
    """
    flag = at_ath(close, near_pct)
    raw = (~flag).astype(float)
    return raw.shift(lag).fillna(0.0)


def backtest(close: pd.Series, position: pd.Series, cost_bps: float = 5.0) -> dict:
    """Run a long-or-flat position over ``close``; net of ``cost_bps`` one-way per switch.

    Cash earns 0% (a stated, conservative choice). Returns the net daily series plus
    headline stats (CAGR, vol, Sharpe, max drawdown, switches, time-in-market).
    """
    ret = close.pct_change().fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    return _stats(net, pos, turn)


def buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold the tape: position 1 every day, no costs after entry."""
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def _stats(net: pd.Series, pos: pd.Series, turn: pd.Series) -> dict:
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if net.std() > 0 else float("nan")
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos > 0).mean()),
        "final": float(equity.iloc[-1]),
    }
