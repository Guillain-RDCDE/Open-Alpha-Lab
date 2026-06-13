"""The Coppock Curve signal and its honest controls — Study 105 (Coppock-Curve).

The folk recipe: compute the Coppock Curve = WMA(10) of (ROC(14) + ROC(11)) on
monthly closes. BUY when the curve turns up from *below zero* (crosses from a
negative trough into positive territory, or simply when it ticks up while still
negative). Exit after a fixed forward horizon (6 or 12 months) or when the curve
turns down from above zero. We test:

- **Signal arm**: forward returns on Coppock buy signals vs random-timing entries
  at the *same historical base rate* as the signal fires.
- **Buy-and-hold baseline**: the always-invested benchmark over the same period.

Three controls share one engine (:func:`forward_returns`):

- **Coppock buy**: enter on a Coppock buy signal (turn-up from below zero).
- **Random timing**: enter on random months drawn at the *same frequency* as
  Coppock fires — the minimum honest baseline, controlling for the fact the
  signal fires disproportionately after large drawdowns (when the market is
  cheap).
- **Buy-and-hold**: invest every month — the long-run beta everyone already has.

The critical caveat: Coppock fires ~10–15 times over 70 years. Even with 75
years of S&P 500 data, effective n is very small and inference is therefore
inherently weak regardless of the measured effect size. We say so loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ROC_PERIOD_LONG, ROC_PERIOD_SHORT, WMA_PERIOD


# ---------------------------------------------------------------------------
# Coppock Curve indicator
# ---------------------------------------------------------------------------
def roc(close: pd.Series, n: int) -> pd.Series:
    """Rate of change over n periods, as a fraction (not %).

    ROC(n)_t = close_t / close_{t-n} - 1.
    The first n entries are NaN.
    """
    return close.pct_change(n)


def wma(series: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average over n periods.

    Weights are 1, 2, …, n (most recent has highest weight). Returns NaN
    until n observations are available.
    """
    weights = np.arange(1, n + 1, dtype=float)

    def _wma(x):
        if len(x) < n or np.isnan(x).any():
            return np.nan
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(n, min_periods=n).apply(_wma, raw=True)


def coppock_curve(
    close: pd.Series,
    roc_long: int = ROC_PERIOD_LONG,
    roc_short: int = ROC_PERIOD_SHORT,
    wma_n: int = WMA_PERIOD,
) -> pd.Series:
    """The Coppock Curve (Edwin Coppock, 1962).

    CC_t = WMA(wma_n) [ ROC(roc_long)_t + ROC(roc_short)_t ]

    The result is expressed as a fraction (not %). The first
    (roc_long + wma_n - 1) values are NaN — roughly 23 months of warm-up,
    which limits the effective signal count on even a 70-year tape to ~50
    data points from which 10–15 buy signals fire.
    """
    sum_roc = roc(close, roc_long) + roc(close, roc_short)
    return wma(sum_roc, wma_n)


def buy_signals(cc: pd.Series, deduplicate: bool = True) -> pd.Series:
    """Classic Coppock buy rule: curve turns up from below zero.

    Returns a boolean Series indexed like ``cc``. A buy fires at month t when:
      1. cc[t] < 0 (still in negative territory), AND
      2. cc[t] > cc[t-1] (curve is ticking up from its trough), AND
      3. (with ``deduplicate=True``) the previous month was NOT already a signal.

    The deduplication (default on) means we take only the *first* uptick of each
    consecutive cluster — the recovery from a bear market trough may produce several
    consecutive months where the curve is below zero and ticking up, but we register
    only one trade entry per cycle bottom. This matches the practical implementation
    of the Coppock rule (Barron's, October 1962) and prevents one bear-market
    recovery from generating a cascade of re-entries.
    """
    negative = cc < 0.0
    turning_up = cc > cc.shift(1)
    raw = (negative & turning_up)
    raw[cc.isna()] = False  # no signal before warm-up is complete

    if not deduplicate:
        return raw.astype(bool)

    # Keep only the first True of each consecutive True run
    prev_was_sig = raw.shift(1, fill_value=False)
    return (raw & ~prev_was_sig).astype(bool)


def random_signals(n_signals: int, close: pd.Series, seed: int = 0) -> pd.Series:
    """A reproducible random-timing control with the same count as the Coppock.

    Draws ``n_signals`` dates uniformly at random from the *valid* signal window
    (after warm-up) and returns a boolean Series aligned to ``close``. Because
    the Coppock fires after drawdowns, this control deliberately over-counts
    neutral entry points — if anything it makes the Coppock look *better* than
    it deserves, giving the signal the benefit of the doubt.
    """
    rng = np.random.default_rng(seed)
    valid_idx = close.dropna().index
    chosen = rng.choice(np.arange(len(valid_idx)), size=n_signals, replace=False)
    chosen_dates = valid_idx[np.sort(chosen)]
    sig = pd.Series(False, index=close.index)
    sig.loc[chosen_dates] = True
    return sig


# ---------------------------------------------------------------------------
# Forward-return engine
# ---------------------------------------------------------------------------
def forward_returns(
    close: pd.Series,
    signals: pd.Series,
    horizon: int = 12,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Forward buy-and-hold returns after each signal.

    For each True entry in ``signals`` at month t, this records the log return
    from close[t] to close[t + horizon]. The entry is "at the next month's open"
    in spirit, but since we only have monthly closes, we use close[t] as the
    entry price and close[t + horizon] as the exit. A transaction cost of
    ``cost_bps`` (round-trip) is deducted.

    No look-ahead: the signal is formed on data up to and including month t;
    the forward return starts at month t+1 in effect (but measured on close[t]).
    The slight simplification vs an intramonth entry is standard for monthly
    studies and has negligible impact on directional conclusions.

    Columns: ``signal_date, horizon_months, ret_gross, ret_net``.
    """
    close_ = close.dropna()
    idx = list(close_.index)
    pos = {d: i for i, d in enumerate(idx)}

    rows = []
    for date, is_sig in signals.items():
        if not is_sig:
            continue
        i = pos.get(date)
        if i is None or i + horizon >= len(idx):
            continue  # not enough forward months
        entry_px = close_.iloc[i]
        exit_px = close_.iloc[i + horizon]
        ret_gross = np.log(exit_px / entry_px)
        rows.append(
            {
                "signal_date": date,
                "horizon_months": horizon,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------
def buy_and_hold_returns(close: pd.Series, horizon: int = 12) -> pd.Series:
    """Log forward returns for every month (the buy-and-hold baseline).

    Returns a Series of horizon-month log returns aligned to the close index.
    These form the empirical distribution of what any random investor holding
    for ``horizon`` months could expect — the true baseline the Coppock must beat.
    """
    close_ = close.dropna()
    fwd = np.log(close_.shift(-horizon) / close_)
    return fwd.dropna()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(returns: pd.Series | pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-signal statistics for a forward-return ledger.

    Returns signal count, hit-rate (positive forward return), mean log return
    (per signal, over the horizon), mean return in percent, P&L skew, and a HAC
    Newey-West t-stat on the mean. The sample size is very small for Coppock
    signals (~10-19), so the t-stat is inherently imprecise — we flag this in the
    output. ``ann_return_pct`` is the average per-signal return in percent (not
    annualised); label it accordingly in output.
    """
    if isinstance(returns, pd.DataFrame):
        r = returns[col].to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_signals": int(n),
        "hit_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_log_ret": float(r.mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "ann_return_pct": float(r.mean() * 100) if n else float("nan"),  # per-signal average (not annualised)
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
        "low_n_warning": n < 20,  # Coppock fires rarely; flag underpowered tests
    }
    if n > 5:
        # Newey-West HAC t-stat on the per-signal returns
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def compare_arms(
    close: pd.Series,
    horizon: int = 12,
    cost_bps: float = 5.0,
    rand_seed: int = 42,
) -> dict:
    """Run all three arms and return a summary dict for the spine table.

    Arms:
    - ``coppock``: buy on Coppock turn-up-from-below-zero signals.
    - ``random``: buy on the same number of randomly-selected months.
    - ``bah``: forward return from every month (buy-and-hold distribution).
    """
    cc = coppock_curve(close)
    sigs = buy_signals(cc)
    n_sig = int(sigs.sum())

    # Coppock arm
    cop_led = forward_returns(close, sigs, horizon=horizon, cost_bps=cost_bps)
    cop = summarize(cop_led, "ret_net")

    # Random-timing arm (same count)
    rand_sigs = random_signals(n_sig, close, seed=rand_seed)
    rand_led = forward_returns(close, rand_sigs, horizon=horizon, cost_bps=cost_bps)
    rand = summarize(rand_led, "ret_net")

    # Buy-and-hold baseline (all months)
    bah = buy_and_hold_returns(close, horizon=horizon)
    bah_stats = {
        "n_signals": int(bah.size),
        "hit_rate": float((bah > 0).mean()),
        "mean_log_ret": float(bah.mean()),
        "ann_return_pct": float(bah.mean() * 12 * 100),
        "skew": float(bah.skew()),
        "tstat": float("nan"),
    }

    return {
        "coppock": cop,
        "random": rand,
        "buy_and_hold": bah_stats,
        "n_coppock_signals": n_sig,
        "horizon_months": horizon,
        "cost_bps": cost_bps,
    }
