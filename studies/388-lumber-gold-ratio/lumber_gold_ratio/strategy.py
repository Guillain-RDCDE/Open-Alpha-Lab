"""The strategy and its honest controls — Study 388 (Lumber-Gold-Ratio).

The folk claim (popularised by Gayed & Bilello, *An Intermarket Approach to Beta Rotation*,
2015): *the lumber/gold ratio is a clean risk-on/risk-off switch.* Lumber is cyclical (housing,
construction, growth); gold is defensive (fear, store of value). So when **lumber outruns gold**
(ratio rising) growth/risk-on is in, and you should be **in stocks (SPY)**; when **gold outruns
lumber** (ratio falling) fear/risk-off is in, and you should be **in long bonds (TLT)**. The
testable, tradable version is a binary **stock/bond rotation switch**:

    position_{t+1} = 1 (hold SPY)  if z_t >  enter_z   (ratio stretched-high → risk-on  → stocks)
                   = 0 (hold TLT)  if z_t <= enter_z   (ratio not high      → risk-off → bonds)

where ``z_t`` is the standardised deviation of the log lumber/gold ratio from its trailing mean,
known at the close of *t*. The switch earns SPY's return of *t+1* when risk-on and TLT's return
when risk-off — **one** execution lag, applied once.

The honest yardstick is **not** raw return but a race, on an **excess-of-cash** basis, against:

1. **Buy-and-hold SPY** — always in stocks, the simplest thing you'd do instead.
2. **A static 60/40 SPY/TLT blend** — the obvious passive stock/bond mix the switch must beat to
   justify the timing.
3. **A random-timing control** — the *same* fraction of days in bonds, but on **random** days
   (seeded). If the lumber/gold ratio carries regime information, the real switch beats random
   rotation; if not, it is just a random mix of the two sleeves.

Sharpe races are excess-of-cash vs excess-of-cash (both sleeves are risky, so we still compare
excess-of-cash to keep it on one clock). Costs are charged one-way × NAV on every switch (each
rotation is one one-way trade of the whole book). No look-ahead: the signal is the *lagged*
z-score, the return is the *forward* day's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Signal — the standardised lumber/gold ratio deviation
# ---------------------------------------------------------------------------
def lumber_gold_ratio(daily: pd.DataFrame) -> pd.Series:
    """The lumber/gold price ratio from the ``lumber`` and ``gold`` close columns."""
    return (daily["lumber"] / daily["gold"]).rename("ratio")


def ratio_zscore(daily: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Standardised deviation of the *log* lumber/gold ratio from its trailing mean.

    ``z_t = (log_ratio_t - rolling_mean) / rolling_std`` over a ``lookback``-day window. The
    first ``lookback-1`` values are NaN. This is the signal, stamped on the close of day *t*
    and acted on at *t+1* (the one execution lag lives in ``switch_positions``).
    """
    lr = np.log(lumber_gold_ratio(daily))
    mu = lr.rolling(lookback, min_periods=lookback).mean()
    sd = lr.rolling(lookback, min_periods=lookback).std(ddof=1)
    return ((lr - mu) / sd).rename("z")


# ---------------------------------------------------------------------------
# The rotation switch
# ---------------------------------------------------------------------------
def switch_positions(daily: pd.DataFrame, lookback: int = 60, enter_z: float = 0.0) -> pd.Series:
    """Binary risk-on/risk-off target weight from the lumber/gold z-score.

    Returns a Series of *target* SPY weights (1.0 = stocks / 0.0 = bonds) **already shifted by
    one day**: the weight on day *t* is set from the z-score known at the close of *t-1*, so it
    is the weight you actually hold (and earn) on *t*. A high ratio (z > ``enter_z``) → stocks;
    otherwise → bonds.

    Before the lookback warms up (z is NaN) the position defaults to a neutral 0.5 (a 50/50
    stock/bond blend), so the switch neither sneaks in look-ahead nor takes a directional bet
    while it has no signal.
    """
    z = ratio_zscore(daily, lookback=lookback)
    raw = (z > enter_z).astype(float)    # 1 = risk-on/stocks, 0 = risk-off/bonds
    raw = raw.where(z.notna(), 0.5)       # neutral 50/50 while warming up
    pos = raw.shift(1).fillna(0.5)        # ONE execution lag, applied once
    return pos.rename("position")


def random_positions(pos: pd.Series, seed: int = 0) -> pd.Series:
    """Random-timing control: the *same* number of stock/bond days, on random days (seeded).

    Preserves the switch's stock fraction (so average beta exposure matches) but destroys the
    timing information. If the lumber/gold ratio carries regime signal, the real switch beats
    this. Warm-up 0.5 days are preserved in place; the remaining 0/1 days are reshuffled.
    """
    rng = np.random.default_rng(seed)
    p = pos.to_numpy().copy()
    fixed = (p == 0.5)
    movable = np.where(~fixed)[0]
    vals = p[movable]
    rng.shuffle(vals)
    p[movable] = vals
    return pd.Series(p, index=pos.index, name="position")


# ---------------------------------------------------------------------------
# Returns engine
# ---------------------------------------------------------------------------
def book_returns(
    daily: pd.DataFrame,
    pos: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Daily gross/net returns of the rotation book and its building blocks.

    The book earns ``pos`` × SPY return + ``(1-pos)`` × TLT return that day. Costs: each
    *change* in ``pos`` is one one-way trade of the whole book (turnover = |Δpos|), priced at
    ``cost_bps``. Cost is one-way × NAV. ``pos`` is assumed already lagged (the weight you hold
    on day *t*; see ``switch_positions``).

    Returns a frame with columns:
    - ``r_eq``    — SPY simple return that day
    - ``r_bd``    — TLT simple return that day
    - ``rf``      — daily risk-free rate that day
    - ``pos``     — the held SPY weight (1 stocks / 0 bonds / 0.5 neutral)
    - ``r_gross`` — book gross return (no costs)
    - ``r_net``   — book net return (after switch costs)
    - ``r_bh``    — buy-and-hold SPY return (always long) — a race benchmark
    - ``r_6040``  — static 60/40 SPY/TLT return — a race benchmark
    """
    eq = daily["equity"].astype(float)
    bd = daily["bond"].astype(float)
    r_eq = eq.pct_change().fillna(0.0)
    r_bd = bd.pct_change().fillna(0.0)
    rf = daily["rf"].astype(float).reindex(eq.index).fillna(0.0)
    p = pos.reindex(eq.index).fillna(0.5)

    r_gross = p * r_eq + (1.0 - p) * r_bd
    turnover = p.diff().abs().fillna(0.0)  # one-way fraction of NAV traded
    cost = turnover * (cost_bps * 1e-4)
    r_net = r_gross - cost

    return pd.DataFrame(
        {
            "r_eq": r_eq,
            "r_bd": r_bd,
            "rf": rf,
            "pos": p,
            "r_gross": r_gross,
            "r_net": r_net,
            "r_bh": r_eq,
            "r_6040": 0.6 * r_eq + 0.4 * r_bd,
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
    seed: int = 388,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a daily difference series.

    Used for the *excess daily return of the switch over a benchmark*. Circular blocks
    preserve the autocorrelation i.i.d. resampling would destroy.
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
    of days in stocks, and the HAC t-stat that the *excess-of-cash* daily mean differs from
    zero.
    """
    r = book[ret_col].to_numpy(dtype=float)
    rf = book["rf"].to_numpy(dtype=float)
    return {
        "ann_return": ann_return(r),
        "sharpe_excess": _ann_excess_sharpe(r, rf),
        "max_dd": max_drawdown(r),
        "pct_stocks": float(book["pos"].mean()),
        "t_excess": _hac_t_mean(r - rf),
        "n_days": int(np.isfinite(r).sum()),
    }


def race(
    daily: pd.DataFrame,
    lookback: int = 60,
    enter_z: float = 0.0,
    cost_bps: float = 5.0,
    rand_seed: int = 388,
) -> dict:
    """The full race: the lumber/gold switch vs buy-and-hold SPY, a static 60/40, and a random
    rotation control.

    The decisive inference number is the HAC t-stat that the switch's **net excess-of-cash
    daily return exceeds the 60/40 blend's** (``t_vs_6040``) — *not* the switch's own Sharpe,
    which the bond sleeve can flatter in a falling-rate regime. We also bootstrap the mean daily
    excess of switch-minus-benchmark for each race.

    Returns a nested dict with the books' summaries and the head-to-head inference.
    """
    pos = switch_positions(daily, lookback=lookback, enter_z=enter_z)
    rnd = random_positions(pos, seed=rand_seed)

    switch = book_returns(daily, pos, cost_bps=cost_bps)
    bh = book_returns(daily, pd.Series(1.0, index=daily.index), cost_bps=cost_bps)
    blend = book_returns(daily, pd.Series(0.6, index=daily.index), cost_bps=cost_bps)
    rand = book_returns(daily, rnd, cost_bps=cost_bps)

    sw_sum = summarize_book(switch, "r_net")
    bh_sum = summarize_book(bh, "r_net")
    blend_sum = summarize_book(blend, "r_net")
    rand_sum = summarize_book(rand, "r_net")

    # Head-to-head: net excess-of-cash daily return, switch minus benchmark.
    sw_ex = (switch["r_net"] - switch["rf"]).to_numpy(dtype=float)
    bh_ex = (bh["r_net"] - bh["rf"]).to_numpy(dtype=float)
    blend_ex = (blend["r_net"] - blend["rf"]).to_numpy(dtype=float)
    rand_ex = (rand["r_net"] - rand["rf"]).to_numpy(dtype=float)

    d_bh = sw_ex - bh_ex
    d_6040 = sw_ex - blend_ex
    d_rand = sw_ex - rand_ex

    return {
        "switch": sw_sum,
        "buy_hold": bh_sum,
        "blend_6040": blend_sum,
        "random": rand_sum,
        "t_vs_bh": _hac_t_mean(d_bh),
        "t_vs_6040": _hac_t_mean(d_6040),
        "t_vs_random": _hac_t_mean(d_rand),
        "ci_vs_6040": block_bootstrap_ci(d_6040, seed=rand_seed),
        "ci_vs_random": block_bootstrap_ci(d_rand, seed=rand_seed + 1),
        "n_switches": int(pos.diff().abs().sum() * 2),  # |Δpos| sums halves; ×2 ≈ # rotations
        "lookback": lookback,
        "enter_z": enter_z,
        "cost_bps": cost_bps,
    }
