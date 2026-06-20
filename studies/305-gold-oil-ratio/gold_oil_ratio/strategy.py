"""The strategy and its honest controls — Study 305 (Gold-Oil-Ratio).

The folk claim: *the gold/oil ratio is a macro regime gauge with a PhD in economics, just
like Dr. Copper* — when gold is expensive relative to oil (ratio **high**), the market is in
a risk-off / contractionary regime, so you should be **defensive** (sit in cash); when oil is
firm relative to gold (ratio **low**), growth is on and you should be **long equities**. The
testable, tradable version is a binary **market-timing switch**:

    position_{t+1} = 1 (hold SPY)  if z_t <= enter_z   (ratio not stretched-high → risk-on)
                   = 0 (hold cash)  if z_t >  enter_z   (ratio stretched-high  → risk-off)

where ``z_t`` is the standardised deviation of the log gold/oil ratio from its trailing mean,
known at the close of *t*. The switch earns SPY's return of *t+1* when long and the risk-free
rate when in cash — **one** execution lag, applied once.

The honest yardstick is **not** raw return but a race, on an **excess-of-cash** basis, against:

1. **Buy-and-hold SPY** — always long, the thing you'd do instead. We compare
   *excess-of-cash Sharpe* to *excess-of-cash Sharpe* (the switch sits in cash part-time, so
   a raw-vs-excess Sharpe comparison would be rigged — see METHODOLOGY house rules).
2. **A random-timing control** — the *same* fraction of days in cash, but on **random** days
   (seeded). If the gold/oil ratio carries regime information, the real switch beats random
   timing; if not, it is just a lower-beta version of buy-and-hold.

Costs are charged one-way × NAV on every switch (each entry/exit is one one-way trade of the
whole book). No look-ahead: the signal is the *lagged* z-score, the return is the *forward*
day's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Signal — the standardised gold/oil ratio deviation
# ---------------------------------------------------------------------------
def gold_oil_ratio(daily: pd.DataFrame) -> pd.Series:
    """The gold/oil price ratio from the ``gold`` and ``oil`` close columns."""
    r = (daily["gold"] / daily["oil"]).rename("ratio")
    return r


def ratio_zscore(daily: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Standardised deviation of the *log* gold/oil ratio from its trailing mean.

    ``z_t = (log_ratio_t - rolling_mean) / rolling_std`` over a ``lookback``-day window. The
    first ``lookback-1`` values are NaN. This is the signal, stamped on the close of day *t*
    and acted on at *t+1* (the one execution lag lives in ``run_switch``).
    """
    lr = np.log(gold_oil_ratio(daily))
    mu = lr.rolling(lookback, min_periods=lookback).mean()
    sd = lr.rolling(lookback, min_periods=lookback).std(ddof=1)
    return ((lr - mu) / sd).rename("z")


# ---------------------------------------------------------------------------
# The timing switch
# ---------------------------------------------------------------------------
def switch_positions(daily: pd.DataFrame, lookback: int = 60, enter_z: float = 1.0) -> pd.Series:
    """Binary risk-on/risk-off target weight from the gold/oil z-score.

    Returns a Series of *target* SPY weights (1.0 long / 0.0 cash) **already shifted by one
    day**: the weight on day *t* is set from the z-score known at the close of *t-1*, so it is
    the weight you actually hold (and earn) on *t*. A high ratio (z > ``enter_z``) → cash.

    Before the lookback warms up (z is NaN) the position defaults to long (1.0), the
    buy-and-hold prior; this is conservative (the switch can only *subtract* exposure).
    """
    z = ratio_zscore(daily, lookback=lookback)
    raw = (z <= enter_z).astype(float)  # 1 = risk-on/long, 0 = risk-off/cash
    raw = raw.where(z.notna(), 1.0)      # default long while warming up
    pos = raw.shift(1).fillna(1.0)       # ONE execution lag, applied once
    return pos.rename("position")


def random_positions(pos: pd.Series, seed: int = 0) -> pd.Series:
    """Random-timing control: the *same* number of cash days, on random days (seeded).

    Preserves the switch's cash fraction (so beta exposure matches) but destroys the timing
    information. If the gold/oil ratio carries regime signal, the real switch beats this.
    """
    rng = np.random.default_rng(seed)
    n = len(pos)
    n_cash = int((pos.to_numpy() == 0.0).sum())
    out = np.ones(n)
    if n_cash > 0:
        cash_idx = rng.choice(n, size=n_cash, replace=False)
        out[cash_idx] = 0.0
    return pd.Series(out, index=pos.index, name="position")


# ---------------------------------------------------------------------------
# Returns engine
# ---------------------------------------------------------------------------
def book_returns(
    daily: pd.DataFrame,
    pos: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Daily gross/net returns of the timing book and its building blocks.

    The book earns the SPY simple return when ``pos == 1`` and the daily risk-free rate when
    ``pos == 0``. Costs: each *change* in ``pos`` is one one-way trade of the whole book, so a
    full round-trip (out then back in) costs ``2 * cost_bps``. Cost is one-way × NAV.

    ``pos`` is assumed already lagged (the weight you hold on day *t*; see ``switch_positions``).
    Returns a frame with columns:
    - ``r_eq``    — SPY simple return that day
    - ``rf``      — daily risk-free rate that day
    - ``pos``     — the held weight (1 long / 0 cash)
    - ``r_gross`` — book gross return (no costs)
    - ``r_net``   — book net return (after switch costs)
    - ``r_bh``    — buy-and-hold SPY return (always long) — the race benchmark
    """
    eq = daily["equity"].astype(float)
    r_eq = eq.pct_change().fillna(0.0)
    rf = daily["rf"].astype(float).reindex(eq.index).fillna(0.0)
    p = pos.reindex(eq.index).fillna(1.0)

    r_gross = p * r_eq + (1.0 - p) * rf
    turnover = p.diff().abs().fillna(0.0)  # one-way fraction of NAV traded
    cost = turnover * (cost_bps * 1e-4)
    r_net = r_gross - cost

    return pd.DataFrame(
        {
            "r_eq": r_eq,
            "rf": rf,
            "pos": p,
            "r_gross": r_gross,
            "r_net": r_net,
            "r_bh": r_eq,
        }
    )


# ---------------------------------------------------------------------------
# Performance statistics — excess-of-cash, HAC, bootstrap
# ---------------------------------------------------------------------------
def _ann_excess_sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    """Annualised Sharpe of the *excess-of-cash* daily return stream."""
    ex = r - rf
    ex = ex[np.isfinite(ex)]
    if ex.size < 2 or ex.std(ddof=1) == 0:
        return float("nan")
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _hac_t_mean(x: np.ndarray) -> float:
    """Newey-West HAC t-stat that the mean of ``x`` differs from zero."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
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


def ann_return(r: np.ndarray) -> float:
    """Annualised geometric return of a daily simple-return stream."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    yrs = r.size / TRADING_DAYS_PER_YEAR
    if growth <= 0 or yrs <= 0:
        return float("nan")
    return growth ** (1.0 / yrs) - 1.0


def max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown (a negative number) of a daily simple-return stream."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def block_bootstrap_ci(
    diff: np.ndarray,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 305,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a daily difference series.

    Used for the *excess daily return of the switch over buy-and-hold* (and over the random
    control). Circular blocks preserve the autocorrelation i.i.d. resampling would destroy.
    """
    d = diff[np.isfinite(diff)]
    n = d.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = d[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def summarize_book(book: pd.DataFrame, ret_col: str = "r_net") -> dict:
    """Headline statistics for one return stream, on an excess-of-cash basis.

    Returns annualised return, annualised *excess-of-cash* Sharpe, max drawdown, the fraction
    of days long, and the HAC t-stat that the *excess-of-cash* daily mean differs from zero.
    """
    r = book[ret_col].to_numpy(dtype=float)
    rf = book["rf"].to_numpy(dtype=float)
    return {
        "ann_return": ann_return(r),
        "sharpe_excess": _ann_excess_sharpe(r, rf),
        "max_dd": max_drawdown(r),
        "pct_long": float(book["pos"].mean()),
        "t_excess": _hac_t_mean(r - rf),
        "n_days": int(np.isfinite(r).sum()),
    }


def race(
    daily: pd.DataFrame,
    lookback: int = 60,
    enter_z: float = 1.0,
    cost_bps: float = 5.0,
    rand_seed: int = 305,
) -> dict:
    """The full race: the gold/oil switch vs buy-and-hold SPY vs a random-timing control.

    The decisive inference number is the HAC t-stat that the switch's **net excess-of-cash
    daily return exceeds buy-and-hold's** (``t_vs_bh``) — *not* the switch's own Sharpe, which
    a low-beta book can inflate. We also bootstrap the mean daily excess of switch-minus-BH and
    switch-minus-random.

    Returns a nested dict with the three books' summaries and the head-to-head inference.
    """
    pos = switch_positions(daily, lookback=lookback, enter_z=enter_z)
    rnd = random_positions(pos, seed=rand_seed)

    switch = book_returns(daily, pos, cost_bps=cost_bps)
    bh = book_returns(daily, pd.Series(1.0, index=daily.index), cost_bps=cost_bps)
    rand = book_returns(daily, rnd, cost_bps=cost_bps)

    sw_sum = summarize_book(switch, "r_net")
    bh_sum = summarize_book(bh, "r_net")
    rand_sum = summarize_book(rand, "r_net")

    # Head-to-head: net excess-of-cash daily return, switch minus benchmark.
    sw_ex = (switch["r_net"] - switch["rf"]).to_numpy(dtype=float)
    bh_ex = (bh["r_net"] - bh["rf"]).to_numpy(dtype=float)
    rand_ex = (rand["r_net"] - rand["rf"]).to_numpy(dtype=float)

    d_bh = sw_ex - bh_ex
    d_rand = sw_ex - rand_ex
    ci_bh = block_bootstrap_ci(d_bh, seed=rand_seed)
    ci_rand = block_bootstrap_ci(d_rand, seed=rand_seed + 1)

    return {
        "switch": sw_sum,
        "buy_hold": bh_sum,
        "random": rand_sum,
        "t_vs_bh": _hac_t_mean(d_bh),
        "t_vs_random": _hac_t_mean(d_rand),
        "ci_vs_bh": ci_bh,
        "ci_vs_random": ci_rand,
        "n_switches": int(pos.diff().abs().sum()),
        "lookback": lookback,
        "enter_z": enter_z,
        "cost_bps": cost_bps,
    }
