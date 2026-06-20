"""The look-ahead engine and its honest controls — Study 347 (Look-Ahead-Bias).

ONE signal, three execution conventions, on the *same* tape. The signal is a
plain causal z-score mean-reversion rule: position on day ``t`` is
``-tanh(z_t)``, long when the price is below its rolling mean, short when above,
where ``z_t`` is the rolling z-score of the close using closes up to and
including ``t`` (causal — no future data in the *signal* itself).

The bias lives in *which return the position earns*:

- ``peek`` (look-ahead / same-bar fill) — the position formed from the close of
  ``t`` earns the return *of* ``t`` (``pos_t * ret_t``). But ``ret_t`` is the
  close-to-close move that *produced* ``z_t``; you could only have known ``z_t``
  *after* that move completed. This is the canonical look-ahead bug.
- ``lag1`` (the desk rule) — the position formed at the close of ``t`` earns the
  return of ``t+1`` (``pos_t * ret_{t+1}``): one ``shift``, applied once. This is
  the only convention an actual trader can realise.

The headline metric is the **inflation**: Sharpe(peek) − Sharpe(lag1), and the
ratio of net edges. On the null tape (no real edge) the lagged Sharpe is ~0 and
the peek Sharpe is large and HAC-significant — the gap is *pure bias*.

Costs: one-way turnover × NAV at ``cost_bps`` per side (``|Δpos| * cost``). The
look-ahead Sharpe survives costs precisely because the bias is not a real edge —
it is correlation with the *contemporaneous* return, which costs cannot touch.

One execution lag, documented: ``lag1`` is the correct convention (effective lag
= 1 bar). ``peek`` is the *bug we are measuring* (effective lag = 0 against the
signal-generating bar) — it is here on purpose, labelled, never sold as tradable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def zscore_signal(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Causal rolling z-score of the close over ``lookback`` days.

    Uses closes up to and including day ``t`` (``min_periods=lookback``), so the
    *signal value* contains no future information. The look-ahead bug studied here
    is purely about *when the position earns its return*, not the signal itself.
    """
    logp = np.log(prices)
    mu = logp.rolling(lookback).mean()
    sd = logp.rolling(lookback).std(ddof=1)
    z = (logp - mu) / sd
    return z.rename("z")


def position(prices: pd.Series, lookback: int = 20, signal: str = "momentum") -> pd.Series:
    """Position ∈ (−1, 1) from the causal z-score, capped by ``tanh``.

    ``signal``:
      - ``"momentum"`` (default, the headline demo): ``+tanh(z_t)`` — long when the
        price is above its rolling mean. On a *random walk* the lagged version
        earns ~0; the **peek** version picks up positive *contemporaneous*
        correlation, manufacturing free *positive* alpha out of nothing. This is
        the clean "free alpha from peeking" demonstration.
      - ``"reversion"`` (the positive control): ``-tanh(z_t)`` — long cheap, short
        dear. Matches the planted mean-reversion edge in ``data.synthetic_prices``,
        so the *lagged* book earns a genuine positive Sharpe a trader could bank.

    ``tanh`` caps leverage at ±1 and is smooth (finite turnover). The position on
    day ``t`` is known *at the close of t* (it uses ``z_t``).
    """
    z = zscore_signal(prices, lookback=lookback)
    if signal == "momentum":
        pos = np.tanh(z)
    elif signal == "reversion":
        pos = -np.tanh(z)
    else:
        raise ValueError(f"unknown signal {signal!r} (use 'momentum' or 'reversion')")
    return pos.rename("pos")


# ---------------------------------------------------------------------------
# The three execution conventions
# ---------------------------------------------------------------------------
def book_returns(
    prices: pd.Series,
    lookback: int = 20,
    convention: str = "lag1",
    cost_bps: float = 0.0,
    signal: str = "momentum",
) -> pd.Series:
    """Net daily strategy returns under one execution convention.

    ``convention``:
      - ``"lag1"``  : position at close of ``t`` earns return of ``t+1`` (correct).
      - ``"peek"``  : position at close of ``t`` earns return of ``t`` (look-ahead).

    ``signal`` is passed through to :func:`position` (``"momentum"`` for the
    free-alpha headline, ``"reversion"`` for the planted-edge positive control).

    Costs are one-way turnover × NAV: ``|pos_t - pos_{t-1}| * cost_bps * 1e-4``,
    charged on the bar the position is *changed*, identically across conventions
    so the only difference between books is the look-ahead lag.
    """
    if convention not in ("lag1", "peek"):
        raise ValueError(f"unknown convention {convention!r} (use 'lag1' or 'peek')")

    ret = prices.pct_change()                                     # return of bar t
    pos = position(prices, lookback=lookback, signal=signal)      # known at close of t

    if convention == "lag1":
        # Position from close of t earns the return of t+1: one shift, once.
        gross = pos.shift(1) * ret
    else:  # peek
        # Position from close of t earns the return *of* t — the bug.
        gross = pos * ret

    # Turnover cost: charged when the position is rebalanced (close of t),
    # identical accounting for both conventions.
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * cost_bps * 1e-4
    if convention == "lag1":
        cost = cost.shift(1)

    net = (gross - cost).rename(f"{convention}_net")
    return net.dropna()


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(daily_ret: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline metrics for one daily return series (Sharpe is excess-of-zero).

    Returns n (days), CAGR, annualised Sharpe, max drawdown, annualised mean.
    """
    r = daily_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod().to_numpy()
    return {
        "n": int(n),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": _max_drawdown(equity),
        "mean_ann": float(ann_mean),
    }


def hac_tstat(returns: pd.Series) -> dict:
    """Newey-West (HAC) t-stat on the mean of a daily return series.

    Tests whether the strategy's mean daily return is distinguishable from zero,
    with a lag-truncation bandwidth ``floor(4*(n/100)^(2/9))`` (Newey-West 1987).
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
        "sharpe_ann": float(mu / x.std(ddof=1) * np.sqrt(TRADING_DAYS)) if x.std(ddof=1) > 0 else float("nan"),
    }


def block_bootstrap_ci(
    returns: pd.Series, block: int = 10, n_boot: int = 2000, seed: int = 347, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the *mean* daily return.

    Block resampling preserves the autocorrelation an i.i.d. bootstrap destroys.
    Returns the (lower, upper) percentile interval on the mean daily return.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def compare(
    prices: pd.Series,
    lookback: int = 20,
    cost_bps: float = 0.0,
    signal: str = "momentum",
    seed: int = 347,
) -> dict:
    """Run the SAME signal under both conventions and quantify the look-ahead tax.

    Returns a bundle with the peek/lag1 net series, their summaries, HAC tests,
    a block-bootstrap CI on each, and the headline ``inflation`` numbers:

      - ``sharpe_inflation``   = Sharpe(peek) − Sharpe(lag1)
      - ``mean_ratio``         = mean(peek) / mean(lag1)  (∞-ish on a null tape)
      - ``free_sharpe``        = Sharpe(peek) when lag1 ≈ 0 (the manufactured edge)
    """
    peek = book_returns(prices, lookback=lookback, convention="peek",
                        cost_bps=cost_bps, signal=signal)
    lag1 = book_returns(prices, lookback=lookback, convention="lag1",
                        cost_bps=cost_bps, signal=signal)

    s_peek = summarize(peek)
    s_lag1 = summarize(lag1)
    t_peek = hac_tstat(peek)
    t_lag1 = hac_tstat(lag1)

    mean_ratio = (s_peek["mean_ann"] / s_lag1["mean_ann"]
                  if s_lag1["mean_ann"] not in (0.0,) and np.isfinite(s_lag1["mean_ann"])
                  else float("inf"))

    return {
        "peek": peek,
        "lag1": lag1,
        "summary_peek": s_peek,
        "summary_lag1": s_lag1,
        "test_peek": t_peek,
        "test_lag1": t_lag1,
        "ci_peek": block_bootstrap_ci(peek, seed=seed),
        "ci_lag1": block_bootstrap_ci(lag1, seed=seed),
        "sharpe_inflation": float(s_peek["sharpe"] - s_lag1["sharpe"]),
        "mean_ratio": float(mean_ratio),
        "free_sharpe": float(s_peek["sharpe"]),
        "lag1_sharpe": float(s_lag1["sharpe"]),
    }
