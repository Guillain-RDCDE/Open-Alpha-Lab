"""The ROC timing rule and its honest controls — Study 427 (Rate of Change).

The folk recipe: the **Rate of Change** oscillator,

    ROC_t(N) = 100 * (P_t / P_{t-N} - 1) ,

is "the oldest momentum tool." The textbook timing rule: **go long when ROC is
above zero, step aside (to cash) when it is below** — riding the trend and
sidestopping the drawdowns. We implement this as a daily long/flat timing rule
(and, where natural, long/short) and pin it against the questions the folklore
skips:

1. **Costs.** Turnover × one-way bps × NAV, charged on every position change.
   A second sweep finds the break-even cost.
2. **Excess-vs-excess Sharpe.** The rule sits in cash part of the time, so its
   Sharpe is computed on returns **in excess of the daily risk-free rate**, and
   raced against buy-and-hold's *excess* Sharpe — never raw-vs-excess.
3. **HAC inference.** A Newey-West *t* on the strategy's mean excess return, and
   a *t* on the **difference** (ROC minus buy-and-hold) — the number that decides
   whether ROC actually *adds* anything.
4. **A permutation placebo.** Sign-shuffle the daily position vector and re-run:
   if a random long/flat schedule with the same exposure does as well, the timing
   carries no information.
5. **The simpler-benchmark race.** A 200-day SMA cross, MACD, and RSI(14) — so the
   implicit "ROC is better" claim is actually tested against the obvious
   alternatives, not asserted.

Execution lag (documented exactly): the signal is known at the **close of day t**;
the position earns the return of **day t+1**. One ``shift(1)``, applied once, in
:func:`book_returns`. No second silent shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RF_ANNUAL = 0.02  # flat risk-free for the excess-vs-excess race (annual)


# ---------------------------------------------------------------------------
# The indicator + the timing rules
# ---------------------------------------------------------------------------
def roc(close: pd.Series, window: int = 126) -> pd.Series:
    """Rate of Change over ``window`` trading days, in percent.

    ROC_t = 100 * (P_t / P_{t-N} - 1). Valid from bar ``window`` onward.
    """
    return 100.0 * (close / close.shift(window) - 1.0)


def roc_signal(close: pd.Series, window: int = 126, mode: str = "long_flat") -> pd.Series:
    """Target position from the ROC rule, known at the **close of day t**.

    - ``mode="long_flat"`` → +1 when ROC > 0 else 0 (the textbook rule).
    - ``mode="long_short"`` → +1 when ROC > 0 else -1 (the aggressive variant).

    The returned series is the *desired* exposure for the **next** day; the lag is
    applied once in :func:`book_returns`.
    """
    r = roc(close, window)
    if mode == "long_flat":
        pos = (r > 0).astype(float)
    elif mode == "long_short":
        pos = np.sign(r).replace(0.0, 0.0).astype(float)
    else:
        raise ValueError(f"mode must be 'long_flat' or 'long_short', got {mode!r}")
    pos[r.isna()] = np.nan
    return pos


def sma_cross_signal(close: pd.Series, window: int = 200, mode: str = "long_flat") -> pd.Series:
    """Benchmark: long when close > N-day SMA, else flat (or short)."""
    sma = close.rolling(window, min_periods=window).mean()
    long = (close > sma).astype(float)
    if mode == "long_short":
        pos = long * 2.0 - 1.0
    else:
        pos = long
    pos[sma.isna()] = np.nan
    return pos


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9,
                mode: str = "long_flat") -> pd.Series:
    """Benchmark: long when MACD line > signal line, else flat (or short)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    long = (macd > signal).astype(float)
    pos = (long * 2.0 - 1.0) if mode == "long_short" else long
    pos.iloc[: slow + sig] = np.nan
    return pos


def rsi_signal(close: pd.Series, window: int = 14, lo: float = 50.0,
               mode: str = "long_flat") -> pd.Series:
    """Benchmark: long when RSI(14) > 50, else flat (or short) — the momentum cut."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    long = (rsi > lo).astype(float)
    pos = (long * 2.0 - 1.0) if mode == "long_short" else long
    pos.iloc[:window] = np.nan
    return pos


# ---------------------------------------------------------------------------
# Booking returns — the one documented execution lag + costs
# ---------------------------------------------------------------------------
def book_returns(
    close: pd.Series,
    target_pos: pd.Series,
    cost_bps: float = 1.0,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> pd.DataFrame:
    """Daily P&L of a timing rule, with the one execution lag and costs.

    The signal ``target_pos`` is known at the close of *t*; we ``shift(1)`` it so
    the held position earns the return of *t+1* (one lag, applied **once**). Costs
    are charged as ``one-way bps × |Δposition| × NAV`` on each day the position
    changes (turnover = one-way × NAV). Returns a frame with:

    - ``mkt``   : the asset's daily simple return (total return if auto-adjusted).
    - ``pos``   : the lagged, actually-held position.
    - ``gross`` : pos × mkt (no costs).
    - ``cost``  : cost_bps×1e-4 × |Δpos|.
    - ``net``   : gross − cost.
    - ``exc``   : net − daily risk-free (the excess series for the Sharpe race).
    """
    mkt = close.pct_change()
    held = target_pos.shift(1)
    dpos = held.diff().abs().fillna(held.abs())
    cost = cost_bps * 1e-4 * dpos
    gross = held * mkt
    net = gross - cost
    rf_daily = rf_annual / TRADING_DAYS
    df = pd.DataFrame(
        {"mkt": mkt, "pos": held, "gross": gross, "cost": cost, "net": net},
        index=close.index,
    )
    df["exc"] = df["net"] - rf_daily
    return df.dropna(subset=["pos", "mkt"])


# ---------------------------------------------------------------------------
# Performance stats — annualised, excess-vs-excess
# ---------------------------------------------------------------------------
def _ann_sharpe(daily_excess: np.ndarray) -> float:
    daily_excess = daily_excess[np.isfinite(daily_excess)]
    if daily_excess.size < 2 or daily_excess.std(ddof=1) == 0:
        return float("nan")
    return float(daily_excess.mean() / daily_excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _ann_return(daily_net: np.ndarray) -> float:
    daily_net = daily_net[np.isfinite(daily_net)]
    if daily_net.size == 0:
        return float("nan")
    return float((1.0 + daily_net).prod() ** (TRADING_DAYS / daily_net.size) - 1.0)


def _max_drawdown(daily_net: np.ndarray) -> float:
    daily_net = daily_net[np.isfinite(daily_net)]
    eq = np.cumprod(1.0 + daily_net)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else float("nan")


def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat of the mean of ``x`` against zero."""
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


def summarize(book: pd.DataFrame) -> dict:
    """Headline annualised stats for one booked strategy frame.

    The Sharpe is the **excess** Sharpe (net of costs, in excess of the risk-free)
    so the buy-and-hold race is excess-vs-excess. ``t_exc`` is a HAC t on the daily
    excess return; ``cagr``/``maxdd`` are net of costs.
    """
    exc = book["exc"].to_numpy(dtype=float)
    net = book["net"].to_numpy(dtype=float)
    n = int(np.isfinite(net).sum())
    return {
        "n_days": n,
        "years": n / TRADING_DAYS,
        "sharpe": _ann_sharpe(exc),
        "cagr": _ann_return(net),
        "maxdd": _max_drawdown(net),
        "t_exc": hac_tstat(exc),
        "ann_turnover": float(book["cost"].gt(0).mean() * TRADING_DAYS),
        "exposure": float(np.nanmean(book["pos"].to_numpy())),
        "ann_cost": float(np.nansum(book["cost"]) / n * TRADING_DAYS) if n else float("nan"),
    }


def buy_and_hold(close: pd.Series, rf_annual: float = DEFAULT_RF_ANNUAL) -> pd.DataFrame:
    """The benchmark: fully invested every day (pos=1), no turnover after entry."""
    pos = pd.Series(1.0, index=close.index)
    return book_returns(close, pos, cost_bps=0.0, rf_annual=rf_annual)


# ---------------------------------------------------------------------------
# The decisive test — does ROC ADD anything? (difference HAC t)
# ---------------------------------------------------------------------------
def diff_vs_hold(
    close: pd.Series,
    window: int = 126,
    mode: str = "long_flat",
    cost_bps: float = 1.0,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> dict:
    """ROC timing minus buy-and-hold: the incremental edge and its HAC t.

    Both legs are booked on the same calendar; the difference of the daily *net*
    returns is the value ROC adds over simply holding. A HAC t on that difference
    is the inference-bar number for the Signal axis.
    """
    sig = roc_signal(close, window, mode)
    roc_book = book_returns(close, sig, cost_bps=cost_bps, rf_annual=rf_annual)
    bh_book = buy_and_hold(close, rf_annual=rf_annual)
    j = roc_book.join(bh_book[["net"]], how="inner", rsuffix="_bh")
    d = (j["net"] - j["net_bh"]).to_numpy(dtype=float)
    return {
        "delta_ann": float(np.nanmean(d) * TRADING_DAYS),
        "t_diff": hac_tstat(d),
        "n_days": int(np.isfinite(d).sum()),
    }


# ---------------------------------------------------------------------------
# Permutation placebo — does the *timing* carry information?
# ---------------------------------------------------------------------------
def permutation_pvalue(
    close: pd.Series,
    window: int = 126,
    mode: str = "long_flat",
    cost_bps: float = 1.0,
    n_draws: int = 2000,
    seed: int = 427,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> dict:
    """Block-permutation null on the *position vector*.

    We keep the realised position series (so exposure/turnover are preserved) but
    randomly **circular-shift** it relative to the returns ``n_draws`` times, and
    measure how often a shuffled schedule's excess Sharpe beats the observed one.
    If the timing carries no information, the observed Sharpe sits in the cloud.
    """
    sig = roc_signal(close, window, mode)
    book = book_returns(close, sig, cost_bps=cost_bps, rf_annual=rf_annual)
    obs = _ann_sharpe(book["exc"].to_numpy(dtype=float))

    mkt = book["mkt"].to_numpy(dtype=float)
    pos = book["pos"].to_numpy(dtype=float)
    rf_daily = rf_annual / TRADING_DAYS
    n = pos.size
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        k = int(rng.integers(window, n - window)) if n > 2 * window else int(rng.integers(1, n))
        rolled = np.roll(pos, k)
        dpos = np.abs(np.diff(rolled, prepend=rolled[0]))
        net = rolled * mkt - cost_bps * 1e-4 * dpos
        draws[i] = _ann_sharpe(net - rf_daily)
    draws = draws[np.isfinite(draws)]
    p = float((draws >= obs).mean()) if draws.size else float("nan")
    return {"obs": obs, "p_value": p, "draws": draws}


# ---------------------------------------------------------------------------
# Cost sweep + break-even
# ---------------------------------------------------------------------------
def cost_sweep(
    close: pd.Series,
    window: int = 126,
    mode: str = "long_flat",
    costs=(0.0, 1.0, 2.0, 5.0, 10.0),
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> pd.DataFrame:
    """Net excess Sharpe + CAGR of the ROC rule across one-way cost levels (bps)."""
    sig = roc_signal(close, window, mode)
    rows = []
    for c in costs:
        b = book_returns(close, sig, cost_bps=c, rf_annual=rf_annual)
        s = summarize(b)
        rows.append({"cost_bps": c, "sharpe": s["sharpe"], "cagr": s["cagr"],
                     "ann_turnover": s["ann_turnover"]})
    return pd.DataFrame(rows)


def breakeven_cost(close: pd.Series, window: int = 126, mode: str = "long_flat",
                   rf_annual: float = DEFAULT_RF_ANNUAL, hi: float = 50.0) -> float:
    """One-way bps at which the ROC rule's net CAGR equals buy-and-hold's CAGR."""
    bh = summarize(buy_and_hold(close, rf_annual=rf_annual))["cagr"]
    sig = roc_signal(close, window, mode)
    lo, hi_ = 0.0, hi
    # If even zero-cost ROC trails buy-and-hold, the break-even is negative -> 0.
    if summarize(book_returns(close, sig, cost_bps=0.0, rf_annual=rf_annual))["cagr"] < bh:
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi_)
        c = summarize(book_returns(close, sig, cost_bps=mid, rf_annual=rf_annual))["cagr"]
        if c > bh:
            lo = mid
        else:
            hi_ = mid
    return 0.5 * (lo + hi_)


# ---------------------------------------------------------------------------
# The simpler-benchmark race
# ---------------------------------------------------------------------------
def benchmark_race(
    close: pd.Series,
    roc_window: int = 126,
    mode: str = "long_flat",
    cost_bps: float = 1.0,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> pd.DataFrame:
    """Race ROC against the obvious simpler timing rules + buy-and-hold.

    Each rule is booked with the same one execution lag, same costs, same
    risk-free, and summarised on the **excess** Sharpe — so "ROC is better" is
    actually tested, not asserted.
    """
    rules = {
        "Buy & hold": pd.Series(1.0, index=close.index),
        f"ROC({roc_window})": roc_signal(close, roc_window, mode),
        "SMA(200)": sma_cross_signal(close, 200, mode),
        "MACD(12,26,9)": macd_signal(close, mode=mode),
        "RSI(14)>50": rsi_signal(close, 14, mode=mode),
    }
    rows = []
    for name, sig in rules.items():
        cost = 0.0 if name == "Buy & hold" else cost_bps
        b = book_returns(close, sig, cost_bps=cost, rf_annual=rf_annual)
        s = summarize(b)
        rows.append({"rule": name, "sharpe": s["sharpe"], "cagr": s["cagr"],
                     "maxdd": s["maxdd"], "t_exc": s["t_exc"],
                     "ann_turnover": s["ann_turnover"], "exposure": s["exposure"]})
    return pd.DataFrame(rows).set_index("rule")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    close: pd.Series,
    window: int = 126,
    mode: str = "long_flat",
    cost_bps: float = 1.0,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    n_draws: int = 2000,
    seed: int = 427,
) -> dict:
    """End-to-end: ROC book, buy-and-hold, difference t, placebo, costs, race."""
    sig = roc_signal(close, window, mode)
    roc_book = book_returns(close, sig, cost_bps=cost_bps, rf_annual=rf_annual)
    bh_book = buy_and_hold(close, rf_annual=rf_annual)
    return {
        "roc": summarize(roc_book),
        "hold": summarize(bh_book),
        "diff": diff_vs_hold(close, window, mode, cost_bps, rf_annual),
        "placebo": permutation_pvalue(close, window, mode, cost_bps, n_draws, seed, rf_annual),
        "breakeven_bps": breakeven_cost(close, window, mode, rf_annual),
        "race": benchmark_race(close, window, mode, cost_bps, rf_annual),
    }
