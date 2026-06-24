"""Strategy + honest arbiters for Study 430 — Klinger Volume Oscillator.

Stephen Klinger's oscillator (1977, popularised in *Technical Analysis of Stocks &
Commodities*, 1997) is built to make **long-term volume "lead" price**. The recipe:

    TP(t)  = high + low + close                       (Klinger's "typical price" sum)
    trend  = +1 if TP(t) > TP(t-1) else -1            (accumulation vs distribution)
    dm     = high - low                               (the day's range)
    cm     = running sum of dm while trend unchanged, reset+carry on a flip
    VF(t)  = volume * |2*(dm/cm) - 1| * trend * 100   (the "Volume Force")
    KVO(t) = EMA(VF, 34) - EMA(VF, 55)                (the oscillator)
    signal = EMA(KVO, 13)                             (the trigger line)

The folk timing rules on an index:

- **KVO > 0** (or KVO above its signal line) → accumulation → *be long*.
- **KVO < 0** (or KVO below its signal line) → distribution → *be flat / short*.

We turn this into a daily **long/flat** (headline) and **long/short** (variant) timing
rule on SPY, then race it honestly:

  * **NET Sharpe vs buy-and-hold**, excess-of-cash vs excess-of-cash (apples to apples);
  * a **HAC (Newey-West) t** on the per-day strategy-minus-benchmark return difference;
  * a **permutation / block-shuffle placebo** — shuffle the *position blocks* and ask how
    often a random schedule of the same long/flat exposure does as well (the honest null
    for a timing rule that is in-market part-time);
  * **one-day execution lag** (signal known at the close of *t*, position held over *t+1* —
    one shift, applied once);
  * **one-way costs × NAV turnover** (and borrow on any short leg);
  * a head-to-head against the **obvious simpler benchmarks** (SMA cross, MACD) so the
    "Klinger is better" claim is actually tested.

No look-ahead: every indicator is computed on closes/volumes up to bar *t*; the position
is shifted one day before it earns a return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


# --------------------------------------------------------------------------- #
# The Klinger Volume Oscillator
# --------------------------------------------------------------------------- #
def klinger(bars: pd.DataFrame, fast: int = 34, slow: int = 55,
            signal: int = 13) -> pd.DataFrame:
    """Klinger Volume Oscillator (KVO) + its trigger line.

    Implements the standard Volume-Force formulation. Returns a DataFrame with columns
    ``vf`` (Volume Force), ``kvo`` (EMA_fast(vf) − EMA_slow(vf)) and ``signal``
    (EMA_signal(kvo)), indexed identically to ``bars``.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float)
    n = len(close)

    tp = high + low + close
    dm = high - low

    trend = np.ones(n)
    cm = np.zeros(n)
    vf = np.zeros(n)
    for i in range(1, n):
        trend[i] = 1.0 if tp[i] > tp[i - 1] else -1.0
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i]
        if cm[i] == 0.0:
            vf[i] = 0.0
        else:
            vf[i] = vol[i] * abs(2.0 * (dm[i] / cm[i]) - 1.0) * trend[i] * 100.0

    vf_s = pd.Series(vf, index=bars.index)
    kvo = vf_s.ewm(span=fast, adjust=False).mean() - vf_s.ewm(span=slow, adjust=False).mean()
    sig = kvo.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"vf": vf_s, "kvo": kvo, "signal": sig}, index=bars.index)


# --------------------------------------------------------------------------- #
# Position rules (all return a 0/1 [long/flat] or -1/0/1 [long/short] series)
# --------------------------------------------------------------------------- #
def klinger_position(bars: pd.DataFrame, mode: str = "zero", long_short: bool = False,
                     fast: int = 34, slow: int = 55, signal: int = 13) -> pd.Series:
    """Klinger long/flat (or long/short) position, formed on closes up to *t*.

    ``mode="zero"`` → long when KVO > 0; ``mode="signal"`` → long when KVO > its trigger.
    With ``long_short=True`` the flat leg becomes a short (-1).

    The returned series is the **desired exposure as known at the close of bar t**. The
    one-day execution lag is applied later in :func:`book_returns` (a single shift).
    """
    k = klinger(bars, fast=fast, slow=slow, signal=signal)
    if mode == "zero":
        bull = k["kvo"] > 0
    elif mode == "signal":
        bull = k["kvo"] > k["signal"]
    else:
        raise ValueError(f"mode must be 'zero' or 'signal', got {mode!r}")
    low_state = -1 if long_short else 0
    pos = pd.Series(np.where(bull, 1, low_state), index=bars.index, dtype=float)
    pos[k["kvo"].isna()] = 0.0
    return pos


def sma_position(bars: pd.DataFrame, window: int = 200, long_short: bool = False) -> pd.Series:
    """Benchmark: long when close > SMA(window), else flat/short (the classic trend filter)."""
    sma = bars["close"].rolling(window, min_periods=window).mean()
    bull = bars["close"] > sma
    low_state = -1 if long_short else 0
    pos = pd.Series(np.where(bull, 1, low_state), index=bars.index, dtype=float)
    pos[sma.isna()] = 0.0
    return pos


def macd_position(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
                  long_short: bool = False) -> pd.Series:
    """Benchmark: long when MACD line > its signal line (the textbook MACD timing rule)."""
    c = bars["close"]
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    bull = macd > sig
    low_state = -1 if long_short else 0
    # warm-up: require the slow EMA to have seen at least `slow` bars
    pos = pd.Series(np.where(bull, 1, low_state), index=bars.index, dtype=float)
    pos.iloc[:slow] = 0.0
    return pos


# --------------------------------------------------------------------------- #
# Backtest engine — net of costs, excess-of-cash, one-day lag
# --------------------------------------------------------------------------- #
def book_returns(bars: pd.DataFrame, position: pd.Series, cost_bps: float = 1.0,
                 borrow_bps_ann: float = 50.0, rf_ann: float = 0.0) -> pd.DataFrame:
    """Daily P&L of a timed book, net of costs, **excess of cash**, with a one-day lag.

    Convention (one execution lag, applied exactly once): the position known at the close
    of *t* (``position`` row *t*) earns the asset return of *t+1* — a single ``shift(1)``.

    Costs: ``cost_bps`` one-way × |Δposition| (turnover as one-way × NAV) charged on the
    day the position changes. Any *short* exposure pays ``borrow_bps_ann`` annualised.
    Returns are **excess of the risk-free rate** so the Sharpe race is excess-vs-excess.

    Columns: ``asset_excess`` (buy-and-hold excess), ``strat_excess`` (timed book excess,
    net), ``pos`` (the lagged exposure actually held).
    """
    asset_ret = bars["close"].pct_change().fillna(0.0)
    rf_daily = rf_ann / ANN
    asset_excess = asset_ret - rf_daily

    held = position.shift(1).fillna(0.0)             # one-day execution lag
    turnover = held.diff().abs().fillna(held.abs())  # one-way turnover × NAV
    cost = turnover * (cost_bps / 1e4)
    borrow = np.clip(-held, 0.0, None) * (borrow_bps_ann / 1e4) / ANN

    # A flat book earns 0 excess (it sits in cash = the risk-free leg).
    strat_excess = held * asset_excess - cost - borrow
    return pd.DataFrame(
        {"asset_excess": asset_excess, "strat_excess": strat_excess, "pos": held},
        index=bars.index,
    )


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def sharpe(excess: np.ndarray) -> float:
    """Annualised Sharpe of a daily *excess* return stream."""
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))


def hac_t(diff: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``diff`` (per-day strat−benchmark) vs 0."""
    r = np.asarray(diff, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def cagr(excess: np.ndarray) -> float:
    """Annualised compound growth of a daily *excess* return stream (illustrative)."""
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / ANN
    return float(growth ** (1.0 / yrs) - 1.0) if growth > 0 and yrs > 0 else float("nan")


def max_drawdown(excess: np.ndarray) -> float:
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def block_permutation_p(book: pd.DataFrame, n_draws: int = 5000, block: int = 21,
                        seed: int = 430) -> dict:
    """Block-shuffle placebo for a timing rule's edge over buy-and-hold.

    The strategy is in-market part-time; the honest question is "does *when* Klinger is
    long beat a random schedule with the same average exposure and the same block
    persistence?". We shuffle the held-position series in ``block``-day chunks (preserving
    autocorrelation / time-in-market), re-book each shuffle net of the same costs, and
    measure how often the random schedule's Sharpe beats the observed one.

    Returns ``{obs_sharpe, p_value, draws}``.
    """
    asset_excess = book["asset_excess"].to_numpy(dtype=float)
    obs = sharpe(book["strat_excess"].to_numpy(dtype=float))
    pos = book["pos"].to_numpy(dtype=float)
    n = len(pos)
    # chop into blocks
    n_blocks = int(np.ceil(n / block))
    blocks = [pos[i * block:(i + 1) * block] for i in range(n_blocks)]
    rng = np.random.default_rng(seed)
    cost = 1.0 / 1e4  # match the 1bp one-way used in the headline race
    draws = np.empty(n_draws)
    for j in range(n_draws):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate([blocks[k] for k in order])[:n]
        turn = np.abs(np.diff(shuffled, prepend=0.0))
        borrow = np.clip(-shuffled, 0.0, None) * (50.0 / 1e4) / ANN
        se = shuffled * asset_excess - turn * cost - borrow
        draws[j] = sharpe(se)
    p = float((draws >= obs).mean())
    return {"obs_sharpe": obs, "p_value": p, "draws": draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, mode: str = "zero", long_short: bool = False,
                   cost_bps: float = 1.0, with_placebo: bool = True,
                   n_draws: int = 5000) -> dict:
    """Full race for one tape: Klinger long/flat vs buy-and-hold vs SMA & MACD benchmarks.

    Returns a dict of headline numbers: each rule's net annualised Sharpe (excess-of-cash),
    CAGR, max drawdown, time-in-market, annual turnover, the HAC t of (Klinger − buy-hold)
    daily return difference, and (optionally) the block-permutation p-value for Klinger.
    """
    kpos = klinger_position(bars, mode=mode, long_short=long_short)
    spos = sma_position(bars, long_short=long_short)
    mpos = macd_position(bars, long_short=long_short)

    kb = book_returns(bars, kpos, cost_bps=cost_bps)
    sb = book_returns(bars, spos, cost_bps=cost_bps)
    mb = book_returns(bars, mpos, cost_bps=cost_bps)

    asset = kb["asset_excess"].to_numpy(dtype=float)
    ks = kb["strat_excess"].to_numpy(dtype=float)
    ss = sb["strat_excess"].to_numpy(dtype=float)
    ms = mb["strat_excess"].to_numpy(dtype=float)

    def _stats(book):
        s = book["strat_excess"].to_numpy(dtype=float)
        held = book["pos"].to_numpy(dtype=float)
        turn = np.abs(np.diff(held, prepend=0.0)).sum()
        yrs = len(s) / ANN
        return dict(
            sharpe=sharpe(s), cagr=cagr(s), mdd=max_drawdown(s),
            tim=float((held != 0).mean()),
            turnover_yr=float(turn / yrs) if yrs > 0 else float("nan"),
        )

    out = {
        "n_days": int(len(bars)),
        "asset_sharpe": sharpe(asset), "asset_cagr": cagr(asset),
        "asset_mdd": max_drawdown(asset),
        "klinger": _stats(kb), "sma": _stats(sb), "macd": _stats(mb),
        # HAC t of Klinger minus buy-and-hold daily excess (the "is it better?" test)
        "t_vs_hold": hac_t(ks - asset),
        "t_vs_sma": hac_t(ks - ss),
        "t_vs_macd": hac_t(ks - ms),
    }
    if with_placebo:
        pl = block_permutation_p(kb, n_draws=n_draws)
        out["placebo_p"] = pl["p_value"]
        out["placebo_obs"] = pl["obs_sharpe"]
    return out
