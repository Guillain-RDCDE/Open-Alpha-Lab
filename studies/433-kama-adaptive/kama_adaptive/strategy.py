"""The strategy and its honest controls — Study 433 (KAMA).

Perry Kaufman's Adaptive Moving Average (KAMA, *Smarter Trading*, 1995) is an EMA whose
smoothing constant adapts to the **efficiency** of the price move:

    ER(t)   = |C(t) - C(t-n)| / sum_{i} |C(t-i+1) - C(t-i)|        (Efficiency Ratio, window n)
    SC(t)   = [ ER(t) * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1) ]^2 (adaptive smoothing const)
    KAMA(t) = KAMA(t-1) + SC(t) * (C(t) - KAMA(t-1))

When the path is a clean trend (ER ~ 1) KAMA tracks like a *fast* EMA; when it is chop
(ER ~ 0) it freezes like a *slow* EMA, dodging the whipsaws that hurt a fixed-window MA.

We turn each moving average into the simplest, fairest **long/flat timing rule**:

    long when close > MA(t), flat otherwise.

(A long/short variant flips to short instead of flat.)  Signal is known at the close of
*t* and earns the return of *t+1* — **one** shift, applied once.  We then race the KAMA
book against (a) buy-and-hold and (b) the *same rule on a fixed SMA of matched length* —
because the only claim worth testing is "the adaptive filter beats a fixed one", not "a
trend filter beats holding".  Everything is **excess-of-cash** (a part-time-in-cash book
vs buy-and-hold must be compared excess-vs-excess), NET of one-way costs × NAV turnover,
with shorts paying borrow.

Arbiters: a one-sample / HAC *t* on the daily out-performance (KAMA − SMA) book, a
**permutation placebo** (shuffle the daily position vector and re-book to get the null of
"the timing carries no information"), a cost sweep, and a parameter robustness grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def kama(
    price: pd.Series,
    er_period: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman's Adaptive Moving Average on a price series.

    ``er_period`` is the Efficiency Ratio lookback; ``fast``/``slow`` are the EMA spans the
    adaptive smoothing constant interpolates between.  Returns a Series aligned to ``price``;
    the first ``er_period`` bars are NaN (no ER yet), and KAMA is seeded at the first valid
    bar with the price itself.
    """
    p = price.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    if n <= er_period:
        return pd.Series(out, index=price.index, name="kama")

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    # Seed KAMA at bar er_period with the close.
    out[er_period] = p[er_period]
    for i in range(er_period + 1, n):
        change = abs(p[i] - p[i - er_period])
        volatility = np.abs(np.diff(p[i - er_period : i + 1])).sum()
        er = change / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (p[i] - out[i - 1])
    return pd.Series(out, index=price.index, name="kama")


def sma(price: pd.Series, window: int = 30) -> pd.Series:
    """Plain simple moving average — the fixed-window benchmark."""
    return price.rolling(window).mean().rename("sma")


def efficiency_ratio(price: pd.Series, er_period: int = 10) -> pd.Series:
    """Kaufman's Efficiency Ratio (0 = pure chop, 1 = clean straight-line move)."""
    p = price.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    for i in range(er_period, n):
        change = abs(p[i] - p[i - er_period])
        volatility = np.abs(np.diff(p[i - er_period : i + 1])).sum()
        out[i] = change / volatility if volatility > 0 else 0.0
    return pd.Series(out, index=price.index, name="er")


# ---------------------------------------------------------------------------
# Timing rule -> daily position -> NET excess book
# ---------------------------------------------------------------------------
def ma_positions(price: pd.Series, ma: pd.Series, long_short: bool = False) -> pd.Series:
    """Long/flat (or long/short) position from a price-vs-MA cross.

    Position is +1 when ``close > MA``, else 0 (long/flat) or −1 (long/short).  The signal
    is computed on closes up to *t*; the one-bar execution lag is applied by the caller's
    book function (so the position formed at *t* earns *t+1*'s return).
    """
    raw = (price > ma).astype(float)
    if long_short:
        pos = raw * 2.0 - 1.0  # +1 / -1
    else:
        pos = raw  # +1 / 0
    pos[ma.isna()] = 0.0
    return pos.rename("pos")


def book_excess(
    price: pd.Series,
    position: pd.Series,
    rf_annual: float = 0.0,
    cost_bps: float = 1.0,
    borrow_bps_annual: float = 50.0,
) -> pd.Series:
    """Daily NET **excess-of-cash** return of a timing book.

    - ``position`` known at close *t* earns the simple return of *t+1* — one ``shift(1)``.
    - **Excess of cash:** the daily risk-free is subtracted from the *invested* fraction
      only (cash earns rf, which nets out in excess space), so a part-time-in-cash book is
      compared apples-to-apples against buy-and-hold's excess return.
    - **Costs:** one-way ``cost_bps`` × |Δposition| (turnover as one-way × NAV).
    - **Borrow:** a short leg (negative position) pays ``borrow_bps_annual`` pro-rata.

    Returns a daily excess-return Series (NaN-free, aligned to dates with a valid lagged
    position and a defined next-day return).
    """
    ret = price.pct_change()
    pos = position.shift(1)  # the single, documented execution lag
    rf_daily = rf_annual / TRADING_DAYS
    borrow_daily = borrow_bps_annual * 1e-4 / TRADING_DAYS

    gross_excess = pos * (ret - rf_daily)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps * 1e-4
    borrow = np.where(pos < 0, -pos, 0.0) * borrow_daily
    net = gross_excess - cost - pd.Series(borrow, index=pos.index)
    return net.dropna().rename("net_excess")


def buy_hold_excess(price: pd.Series, rf_annual: float = 0.0) -> pd.Series:
    """Buy-and-hold daily excess-of-cash return (the race's benchmark)."""
    ret = price.pct_change()
    rf_daily = rf_annual / TRADING_DAYS
    return (ret - rf_daily).dropna().rename("bh_excess")


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------
def sharpe(daily_excess: pd.Series) -> float:
    r = daily_excess.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily_excess: pd.Series) -> float:
    r = daily_excess.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / TRADING_DAYS
    return float(growth ** (1.0 / yrs) - 1.0) if yrs > 0 and growth > 0 else float("nan")


def max_drawdown(daily_excess: pd.Series) -> float:
    r = daily_excess.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC *t*-stat of the mean of a daily series (H0: mean = 0)."""
    r = daily.to_numpy(dtype=float)
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


# ---------------------------------------------------------------------------
# Permutation placebo
# ---------------------------------------------------------------------------
def permutation_pvalue(
    price: pd.Series,
    position: pd.Series,
    n_perm: int = 2000,
    seed: int = 433,
    cost_bps: float = 1.0,
) -> float:
    """Two-sided permutation *p* for "the KAMA timing carries information".

    The observed statistic is the net-excess **mean** of the real KAMA book.  Under the
    null, the position vector carries no timing information, so we shuffle the daily
    positions (destroying their alignment to returns) and re-book.  *p* is the fraction of
    shuffles whose |mean| ≥ the observed |mean|.
    """
    obs = book_excess(price, position, cost_bps=cost_bps).mean()
    rng = np.random.default_rng(seed)
    pos_vals = position.to_numpy(dtype=float)
    count = 0
    for _ in range(n_perm):
        shuffled = pd.Series(rng.permutation(pos_vals), index=position.index)
        m = book_excess(price, shuffled, cost_bps=cost_bps).mean()
        if abs(m) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    price: pd.Series,
    er_period: int = 10,
    fast: int = 2,
    slow: int = 30,
    sma_window: int = 30,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
    n_perm: int = 0,
    perm_seed: int = 433,
) -> dict:
    """End-to-end: build KAMA & SMA timing books, race them, run the arbiters.

    Returns a dict of headline numbers: net Sharpe / CAGR / maxDD for KAMA, SMA and
    buy-and-hold (all excess-of-cash, NET of costs); the KAMA−SMA daily out-performance
    HAC *t*; KAMA−buyhold HAC *t*; and (when ``n_perm > 0``) the permutation *p* for the
    KAMA book itself.
    """
    k = kama(price, er_period=er_period, fast=fast, slow=slow)
    s = sma(price, window=sma_window)
    pos_k = ma_positions(price, k, long_short=long_short)
    pos_s = ma_positions(price, s, long_short=long_short)

    bk = book_excess(price, pos_k, rf_annual=rf_annual, cost_bps=cost_bps)
    bs = book_excess(price, pos_s, rf_annual=rf_annual, cost_bps=cost_bps)
    bh = buy_hold_excess(price, rf_annual=rf_annual)

    # Align KAMA & SMA books on common dates for the difference test.
    common = bk.index.intersection(bs.index)
    diff_ks = (bk.reindex(common) - bs.reindex(common)).dropna()
    common_bh = bk.index.intersection(bh.index)
    diff_kbh = (bk.reindex(common_bh) - bh.reindex(common_bh)).dropna()

    out = {
        "n_days": int(len(bk)),
        "kama_sharpe": sharpe(bk),
        "sma_sharpe": sharpe(bs),
        "bh_sharpe": sharpe(bh),
        "kama_cagr": cagr(bk),
        "sma_cagr": cagr(bs),
        "bh_cagr": cagr(bh),
        "kama_mdd": max_drawdown(bk),
        "sma_mdd": max_drawdown(bs),
        "bh_mdd": max_drawdown(bh),
        "diff_ks_mean_bps": float(diff_ks.mean() * 1e4),
        "diff_ks_t": hac_tstat(diff_ks),
        "diff_kbh_mean_bps": float(diff_kbh.mean() * 1e4),
        "diff_kbh_t": hac_tstat(diff_kbh),
        "kama_turnover": float(pos_k.diff().abs().sum()),
        "sma_turnover": float(pos_s.diff().abs().sum()),
    }
    if n_perm > 0:
        out["perm_p"] = permutation_pvalue(
            price, pos_k, n_perm=n_perm, seed=perm_seed, cost_bps=cost_bps
        )
    return out
