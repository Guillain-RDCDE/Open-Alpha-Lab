"""The Schaff Trend Cycle, its timing rule, and the honest arbiters — Study 431.

Doug Schaff's Schaff Trend Cycle (STC, 2008) is marketed as a "faster MACD". The recipe:

    1. macd_line = EMA(close, fast) - EMA(close, slow)          (the MACD line, fast=23 slow=50)
    2. run a `cycle`-period stochastic over the MACD line:
         %K_macd = 100 * (macd - min(macd, cycle)) / (max(macd, cycle) - min(macd, cycle))
    3. smooth it: %D_macd = EMA(%K_macd, smooth=0.5 factor)     (Schaff uses a 0.5 EMA weight)
    4. run a SECOND `cycle`-period stochastic over %D_macd, smooth again → STC in 0..100.

The double-stochastic-of-MACD is what makes STC whip from 0 to 100 quickly — the source of
the "faster" claim. The folk trading rule:

- STC crosses **up through 25** (rising out of oversold) → go long.
- STC crosses **down through 75** (falling out of overbought) → exit / go short.

We turn that into a daily **long/flat** position (the natural, no-leverage form) and a
**long/short** variant, race the NET Sharpe against buy-and-hold **excess-vs-excess**
(both sides over the same risk-free), run a HAC t-stat on the strategy's daily returns, a
block-permutation placebo on the sign series, a cost sweep, and a head-to-head against the
plain **MACD** crossover it claims to beat.

Execution lag (documented exactly): the STC state is known at the close of bar *t*; the
position decided at *t* earns the return of bar *t+1*. One `shift(1)`, applied once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def macd_line(close: pd.Series, fast: int = 23, slow: int = 50) -> pd.Series:
    """The MACD line = EMA(fast) - EMA(slow). Schaff's STC defaults are fast=23, slow=50."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _stoch(x: pd.Series, cycle: int) -> pd.Series:
    """A `cycle`-period stochastic %K of a series, in 0..100; flat 50 where range is zero."""
    lo = x.rolling(cycle, min_periods=cycle).min()
    hi = x.rolling(cycle, min_periods=cycle).max()
    rng = hi - lo
    k = 100.0 * (x - lo) / rng.where(rng != 0.0, np.nan)
    return k.fillna(50.0)


def schaff_trend_cycle(
    close: pd.Series,
    fast: int = 23,
    slow: int = 50,
    cycle: int = 10,
    factor: float = 0.5,
) -> pd.Series:
    """Doug Schaff's Schaff Trend Cycle, bounded to 0..100.

    Double-stochastic-of-MACD with EMA-style smoothing weight ``factor`` (Schaff's 0.5),
    matching the canonical TradingView/MetaStock implementation. Returns a Series indexed
    identically to ``close`` (early bars before the windows fill are smoothed-but-warming).
    """
    m = macd_line(close, fast, slow)

    # First stochastic over the MACD line, then EMA-smooth with weight `factor`.
    k1 = _stoch(m, cycle)
    d1 = k1.ewm(alpha=factor, adjust=False).mean()

    # Second stochastic over the smoothed series, then EMA-smooth again.
    k2 = _stoch(d1, cycle)
    stc = k2.ewm(alpha=factor, adjust=False).mean()
    stc.name = "stc"
    return stc


# ---------------------------------------------------------------------------
# Position rules (signal known at close of t, applied to t+1 return)
# ---------------------------------------------------------------------------
def stc_positions(
    stc: pd.Series,
    lower: float = 25.0,
    upper: float = 75.0,
    long_short: bool = False,
) -> pd.Series:
    """Daily position from the STC folk rule, as a step series in {0/+1} or {-1/+1}.

    Go long when STC crosses **up through `lower`** (rising out of oversold); flat (or
    short, if ``long_short``) when STC crosses **down through `upper`** (falling out of
    overbought). The position is held between signals (a step function), exactly the way a
    trend-timing rule is used. NaN STC bars carry position 0.
    """
    s = stc.to_numpy(dtype=float)
    n = len(s)
    pos = np.zeros(n, dtype=float)
    cur = 0.0
    short_val = -1.0 if long_short else 0.0
    for i in range(1, n):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            pos[i] = cur
            continue
        if s[i - 1] <= lower < s[i]:        # cross up through lower → long
            cur = 1.0
        elif s[i - 1] >= upper > s[i]:      # cross down through upper → exit/short
            cur = short_val
        pos[i] = cur
    return pd.Series(pos, index=stc.index, name="pos")


def macd_positions(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    long_short: bool = False,
) -> pd.Series:
    """The plain MACD crossover the STC claims to beat — the benchmark timing rule.

    Long when the MACD line is above its signal EMA, flat (or short) otherwise. Classic
    12/26/9 settings. Same step-series convention as :func:`stc_positions`.
    """
    m = macd_line(close, fast, slow)
    sig = m.ewm(span=signal, adjust=False).mean()
    raw = np.where(m > sig, 1.0, (-1.0 if long_short else 0.0))
    return pd.Series(raw, index=close.index, name="pos")


# ---------------------------------------------------------------------------
# Backtest: position -> net daily returns with one execution lag + costs
# ---------------------------------------------------------------------------
def book_returns(
    bars: pd.DataFrame,
    pos: pd.Series,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Net daily returns for a position series, with ONE documented execution lag.

    The position decided at the close of bar *t* (``pos`` at *t*) earns bar *t+1*'s
    close-to-close return — applied as a single ``shift(1)`` here, the only lag in the
    pipeline. Costs are charged one-way × NAV on each change in position
    (``cost_bps`` per unit turnover), and a short leg pays ``borrow_bps_ann`` financing
    per year while held short. Everything is computed as **excess of the risk-free rate**
    so the Sharpe race against buy-and-hold is excess-vs-excess.

    Columns: ``ret`` (asset daily return), ``pos`` (lagged position actually held),
    ``strat_excess`` (net strategy excess return), ``bh_excess`` (buy-and-hold excess).
    """
    ret = bars["close"].pct_change().fillna(0.0)
    rf_daily = rf_annual / TRADING_DAYS

    held = pos.shift(1).fillna(0.0)                  # one execution lag, applied once
    turnover = held.diff().abs().fillna(held.abs())  # one-way change in exposure
    cost = turnover * (cost_bps * 1e-4)
    borrow = (held < 0).astype(float) * (borrow_bps_ann * 1e-4 / TRADING_DAYS)

    strat_gross_excess = held * (ret - rf_daily)
    strat_excess = strat_gross_excess - cost - borrow
    bh_excess = ret - rf_daily

    return pd.DataFrame(
        {
            "ret": ret,
            "pos": held,
            "strat_excess": strat_excess,
            "bh_excess": bh_excess,
        }
    )


# ---------------------------------------------------------------------------
# Honest arbiters
# ---------------------------------------------------------------------------
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) t-stat of the mean of a daily return series vs zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
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


def sharpe(x: np.ndarray) -> float:
    """Annualised Sharpe of a daily (excess) return series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily_ret: np.ndarray) -> float:
    """Compound annual growth rate from a daily (total) return series."""
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    years = r.size / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")


def max_drawdown(daily_ret: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the equity curve (a negative number)."""
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def block_permutation_p(
    bars: pd.DataFrame,
    pos: pd.Series,
    n_draws: int = 2000,
    block: int = 21,
    cost_bps: float = 1.0,
    seed: int = 431,
) -> dict:
    """Block-permutation placebo: does the STC *position* beat a shuffled position?

    We keep the asset returns fixed and circularly **block-shuffle the position series**
    (blocks of ``block`` days to preserve the holding-period structure). For each shuffle
    we recompute the net excess-return mean. The p-value is the fraction of shuffles whose
    mean meets or beats the observed one. A real timing edge must place the observed mean
    in the right tail; a coin lands in the middle.
    """
    obs = book_returns(bars, pos, cost_bps=cost_bps)["strat_excess"]
    obs_mean = float(obs.mean())

    held = pos.shift(1).fillna(0.0).to_numpy(dtype=float)
    n = held.size
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    base = book_returns(bars, pos, cost_bps=cost_bps)  # for ret/rf alignment
    ret_x = base["ret"].to_numpy(dtype=float)           # asset return (rf=0 default)

    ge = 0
    draws = np.empty(n_draws)
    blocks = [held[i * block:(i + 1) * block] for i in range(nb)]
    for d in range(n_draws):
        order = rng.permutation(nb)
        shuffled = np.concatenate([blocks[i] for i in order])[:n]
        turn = np.abs(np.diff(shuffled, prepend=0.0))
        cost = turn * (cost_bps * 1e-4)
        m = float((shuffled * ret_x - cost).mean())
        draws[d] = m
        if m >= obs_mean:
            ge += 1
    return {
        "obs_mean": obs_mean,
        "p_value": (ge + 1) / (n_draws + 1),
        "draws": draws,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
    fast: int = 23,
    slow: int = 50,
    cycle: int = 10,
    lower: float = 25.0,
    upper: float = 75.0,
) -> dict:
    """One end-to-end run: STC timing rule vs buy-and-hold vs the MACD benchmark.

    Returns a dict of headline numbers — net annualised Sharpe for the STC rule,
    buy-and-hold, and the MACD benchmark (excess-vs-excess), plus CAGR / max-drawdown /
    HAC t / per-year turnover for the STC rule. All net of ``cost_bps`` one-way × NAV with
    one execution lag.
    """
    close = bars["close"]
    stc = schaff_trend_cycle(close, fast=fast, slow=slow, cycle=cycle)
    pos_stc = stc_positions(stc, lower=lower, upper=upper, long_short=long_short)
    pos_macd = macd_positions(close, long_short=long_short)

    bk = book_returns(bars, pos_stc, cost_bps=cost_bps, rf_annual=rf_annual)
    bk_macd = book_returns(bars, pos_macd, cost_bps=cost_bps, rf_annual=rf_annual)

    held = bk["pos"]
    years = len(bars) / TRADING_DAYS

    return {
        "n_days": int(len(bars)),
        "years": float(years),
        "stc_sharpe": sharpe(bk["strat_excess"].to_numpy()),
        "bh_sharpe": sharpe(bk["bh_excess"].to_numpy()),
        "macd_sharpe": sharpe(bk_macd["strat_excess"].to_numpy()),
        "stc_minus_bh": (
            sharpe(bk["strat_excess"].to_numpy()) - sharpe(bk["bh_excess"].to_numpy())
        ),
        "stc_cagr": cagr((bk["strat_excess"] + rf_annual / TRADING_DAYS).to_numpy()),
        "bh_cagr": cagr(bk["ret"].to_numpy()),
        "stc_maxdd": max_drawdown((bk["strat_excess"] + rf_annual / TRADING_DAYS).to_numpy()),
        "bh_maxdd": max_drawdown(bk["ret"].to_numpy()),
        "stc_hac_t": hac_t(bk["strat_excess"].to_numpy()),
        "turnover_per_yr": float(held.diff().abs().sum() / years),
        "exposure": float((held != 0).mean()),
    }
