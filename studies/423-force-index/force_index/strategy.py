"""The strategy and its honest controls — Study 423 (Force Index).

Alexander Elder's Force Index (*Trading for a Living*, 1993) combines three pieces of
information into one number: the direction of the price move, its size, and the volume
behind it::

    raw_FI(t) = (Close(t) - Close(t-1)) * Volume(t)
    FI_n(t)   = EMA_n( raw_FI )

Elder smooths the raw series with an exponential moving average — **FI(2)** for short-term
entries, **FI(13)** for the intermediate trend.  The folk rule under test: *the Force Index
flags reversals.*  Concretely:

- **Zero-cross timing.** When FI(13) crosses **up** through zero, buying pressure has
  returned → go **long**.  When it crosses **down** through zero, selling pressure
  dominates → go **flat** (long/flat) or **short** (long/short).
- We hold the implied position until the next zero-cross, charging one-way costs on each
  flip.

We turn this into a daily long/flat (and long/short) timing rule and race its **NET**
Sharpe against **buy-and-hold**, on an **excess-of-cash to excess-of-cash** basis, with a
Newey-West HAC *t* on the daily return difference, a sign-permutation placebo, a one-way
cost sweep, and a comparison against the obvious simpler timing benchmarks (a 50/200 SMA
filter and an RSI(14) filter) so the "Force Index is better" claim is actually tested.

No look-ahead: the FI signal is computed on closes/volume up to bar *t*; the position it
implies is applied with **one** ``shift`` so it earns the return of bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def force_index(bars: pd.DataFrame, period: int = 13) -> pd.Series:
    """Elder's Force Index: ``EMA_period( (close - close.shift) * volume )``.

    Returns a Series indexed identically to ``bars``.  The raw force is the signed
    one-day price change times that day's volume; the EMA smooths it.  ``period = 13`` is
    Elder's trend filter, ``period = 2`` his short-term trigger.  The first bar is NaN
    (no prior close).
    """
    raw = (bars["close"] - bars["close"].shift(1)) * bars["volume"]
    return raw.ewm(span=period, adjust=False).mean().rename(f"fi{period}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on the close — a benchmark oscillator timing filter."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).rename("rsi")


# ---------------------------------------------------------------------------
# Positions — the timing rules (signal known at close t, applied with one shift)
# ---------------------------------------------------------------------------
def fi_positions(bars: pd.DataFrame, period: int = 13, long_short: bool = False) -> pd.Series:
    """Force-Index zero-cross position series in {0,1} (long/flat) or {-1,1} (long/short).

    Long while FI(period) > 0 (buying pressure), flat (or short) while FI(period) <= 0.
    The position is the *state* implied by the close-of-*t* signal; the caller applies the
    one-day execution lag (``book_returns``) so it earns *t+1*'s return.
    """
    fi = force_index(bars, period=period)
    pos = (fi > 0).astype(float)
    if long_short:
        pos = pos * 2.0 - 1.0  # {0,1} -> {-1,1}
    return pos.rename("pos")


def sma_positions(bars: pd.DataFrame, fast: int = 50, slow: int = 200,
                  long_short: bool = False) -> pd.Series:
    """50/200 SMA-crossover position — the obvious simpler trend benchmark."""
    c = bars["close"]
    pos = (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)
    if long_short:
        pos = pos * 2.0 - 1.0
    return pos.rename("pos")


def rsi_positions(bars: pd.DataFrame, period: int = 14, level: float = 50.0,
                  long_short: bool = False) -> pd.Series:
    """RSI(14)>50 momentum filter — a benchmark oscillator timing rule."""
    pos = (rsi(bars["close"], period=period) > level).astype(float)
    if long_short:
        pos = pos * 2.0 - 1.0
    return pos.rename("pos")


# ---------------------------------------------------------------------------
# Booking returns — one execution lag, one-way costs x NAV, shorts pay borrow
# ---------------------------------------------------------------------------
def book_returns(
    bars: pd.DataFrame,
    pos: pd.Series,
    cost_bps: float = 1.0,
    borrow_bps_yr: float = 50.0,
) -> pd.Series:
    """Daily strategy return for a position series, net of costs.

    The position known at the close of *t* earns the close-to-close return of *t+1*
    (one ``shift``).  Turnover is ``|pos_t - pos_{t-1}|`` (one-way x NAV); each unit of
    turnover costs ``cost_bps``.  A short position (pos < 0) pays daily borrow at
    ``borrow_bps_yr`` annualised on the absolute short exposure.  Returns a daily net
    return Series aligned to ``bars`` (NaN dropped).
    """
    ret = bars["close"].pct_change()
    held = pos.shift(1)  # one execution lag
    gross = held * ret

    turnover = held.diff().abs().fillna(0.0)
    cost = turnover * cost_bps * 1e-4
    borrow = held.clip(upper=0.0).abs() * (borrow_bps_yr * 1e-4 / TRADING_DAYS)

    net = (gross - cost - borrow).dropna()
    return net.rename("ret")


def buy_hold_returns(bars: pd.DataFrame) -> pd.Series:
    """Daily buy-and-hold return (always long, no costs after entry)."""
    return bars["close"].pct_change().dropna().rename("bh")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def sharpe(ret: pd.Series, rf_daily: float = 0.0) -> float:
    """Annualised Sharpe of a daily return series (excess of ``rf_daily``)."""
    r = (ret - rf_daily).to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat that the mean of ``x`` differs from zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def excess_vs_excess(strat: pd.Series, bench: pd.Series,
                     rf_daily: float = 0.0) -> dict:
    """Race a strategy's NET return against a benchmark, excess-of-cash to excess-of-cash.

    Aligns the two daily series, subtracts the (constant) daily risk-free from each, and
    reports both annualised Sharpes and a HAC t-stat on the *daily return difference*
    (strat - bench).  When the strategy sits in cash part-time, subtracting the same rf
    from both sides keeps the race fair.
    """
    a, b = strat.align(bench, join="inner")
    a = (a - rf_daily)
    b = (b - rf_daily)
    diff = (a - b).to_numpy(dtype=float)
    return {
        "sharpe_strat": sharpe(a),
        "sharpe_bench": sharpe(b),
        "ann_ret_strat": float(a.mean() * TRADING_DAYS),
        "ann_ret_bench": float(b.mean() * TRADING_DAYS),
        "mean_diff_bps": float(np.nanmean(diff) * 1e4),
        "hac_t_diff": hac_tstat(diff),
        "n_days": int(np.isfinite(diff).sum()),
    }


def permutation_pvalue(bars: pd.DataFrame, period: int, cost_bps: float,
                       n_perm: int = 1000, long_short: bool = False,
                       seed: int = 423) -> dict:
    """Sign-shuffle placebo: how often does a randomly-flipped FI position series beat
    buy-and-hold by as much (Sharpe) as the real one?

    The real FI position series is the timing skeleton; the placebo *keeps the same set of
    positions* but reassigns their daily order by a random permutation of the position
    vector, breaking any genuine timing alignment with returns while preserving the
    exposure profile.  The one-sided p-value is the fraction of placebos whose
    strat-minus-bench Sharpe difference >= the real one.
    """
    rng = np.random.default_rng(seed)
    pos = fi_positions(bars, period=period, long_short=long_short)
    bh = buy_hold_returns(bars)

    real_net = book_returns(bars, pos, cost_bps=cost_bps)
    real = excess_vs_excess(real_net, bh)
    real_gap = real["sharpe_strat"] - real["sharpe_bench"]

    pos_vals = pos.to_numpy(dtype=float)
    idx = pos.index
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pos_vals)
        pp = pd.Series(perm, index=idx, name="pos")
        net = book_returns(bars, pp, cost_bps=cost_bps)
        gap = sharpe(net) - real["sharpe_bench"]
        if np.isfinite(gap) and gap >= real_gap:
            count += 1
    return {
        "real_gap": float(real_gap),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    period: int = 13,
    cost_bps: float = 1.0,
    long_short: bool = False,
    rf_daily: float = 0.0,
) -> dict:
    """Full single-tape teardown: FI timing vs buy-and-hold, plus the simpler benchmarks.

    Returns a dict with the FI race (Sharpe strat/bench, HAC t on the diff, turnover),
    the SMA(50/200) and RSI(14) benchmark races, and headline counts.  All NET of one-way
    costs x NAV with one-day execution lag; shorts (long_short=True) pay borrow.
    """
    bh = buy_hold_returns(bars)

    fi_pos = fi_positions(bars, period=period, long_short=long_short)
    fi_net = book_returns(bars, fi_pos, cost_bps=cost_bps)
    fi_race = excess_vs_excess(fi_net, bh, rf_daily=rf_daily)
    fi_turn = float(fi_pos.shift(1).diff().abs().sum())

    sma_pos = sma_positions(bars, long_short=long_short)
    sma_net = book_returns(bars, sma_pos, cost_bps=cost_bps)
    sma_race = excess_vs_excess(sma_net, bh, rf_daily=rf_daily)

    rsi_pos = rsi_positions(bars, long_short=long_short)
    rsi_net = book_returns(bars, rsi_pos, cost_bps=cost_bps)
    rsi_race = excess_vs_excess(rsi_net, bh, rf_daily=rf_daily)

    n_years = max((bars.index[-1] - bars.index[0]).days / 365.25, 1e-9)
    return {
        "period": period,
        "long_short": long_short,
        "cost_bps": cost_bps,
        "fi": fi_race,
        "sma": sma_race,
        "rsi": rsi_race,
        "fi_flips_per_yr": float(fi_turn / 2.0 / n_years) if not long_short
        else float(fi_turn / 2.0 / n_years),
        "fi_sharpe": fi_race["sharpe_strat"],
        "bh_sharpe": fi_race["sharpe_bench"],
        "fi_hac_t": fi_race["hac_t_diff"],
    }
