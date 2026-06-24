"""The strategy and its honest controls — Study 449 (Renko-Charts).

The folk claim: a **Renko chart** strips out time and price-noise — it prints a new
"brick" only when price moves a fixed amount (the *brick size*) in the same or opposite
direction.  Proponents say the resulting smoothed brick series makes trend signals
*cleaner*, so a classic **moving-average crossover** run on the Renko brick series should
beat the same crossover run on raw daily closes.

We encode that as the tightest falsifiable mechanical rule a Renko proponent would accept:

1. Build the Renko brick series from daily closes with an **ATR-based brick size**
   (the most common modern recipe — "ATR Renko").
2. Map each brick back to the calendar bar on which it printed, giving a step-function
   "Renko close" aligned to the daily index.
3. Run a fast/slow **simple-MA crossover** on the Renko close → long when fast > slow,
   flat otherwise (long/flat; the long-only version retail actually trades).
4. Run the **identical** crossover on the raw daily close — the head-to-head control.

The decisive question for the Signal axis is the **incremental** Sharpe of the Renko-filtered
crossover *over* the raw-close crossover (and over buy-and-hold), tested with a HAC *t* on
the daily strategy returns and a **return-permutation placebo** (shuffle the daily returns,
destroying serial structure, and re-measure — the brick filter cannot manufacture an edge
out of i.i.d. noise).

Controls:

- **Raw-close crossover** — same MA windows, no Renko filtering.  Does the brick step add
  anything beyond the plain crossover?
- **Buy-and-hold** — the trend follower must beat just owning the tape.
- **Return-permutation placebo** — shuffle returns; any positive Renko Sharpe here is a
  data-mining artefact, not signal.

No look-ahead: the crossover state is known at the close of day *t*; the position earns the
return of *t+1* — one ``shift``, applied once, in :func:`crossover_returns`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Renko brick construction
# ---------------------------------------------------------------------------
def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-style Average True Range (simple rolling mean of true range)."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def renko_series(
    close: pd.Series,
    brick: float,
) -> pd.Series:
    """Map daily closes to a step-function Renko "close" aligned to the daily index.

    A new brick prints when price moves at least ``brick`` from the last brick level in
    either direction (possibly several bricks at once on a big move).  The Renko close on
    day *t* is the level of the most recent brick *as of the close of t* — so it is a
    causal, no-look-ahead transform of the price path.  Days before the first brick carry
    the seed level (the first close).

    ``brick`` must be > 0.  Returns a float Series on ``close.index``.
    """
    if brick <= 0:
        raise ValueError(f"brick size must be positive, got {brick!r}")
    c = close.to_numpy(dtype=float)
    level = np.empty(c.size, dtype=float)
    last = c[0]              # current brick level
    for i in range(c.size):
        move = c[i] - last
        if abs(move) >= brick:
            steps = int(abs(move) // brick)
            last = last + np.sign(move) * steps * brick
        level[i] = last
    return pd.Series(level, index=close.index, name="renko")


def renko_from_atr(
    bars: pd.DataFrame,
    atr_window: int = 14,
    atr_mult: float = 1.0,
) -> pd.Series:
    """ATR-Renko: brick size = ``atr_mult`` × ATR(atr_window), measured once on the full
    sample (a single fixed brick for the whole tape — the standard daily-Renko recipe).

    Using the median ATR over the sample keeps the brick a single fixed number (true
    Renko), not a time-varying brick, while still scaling to the instrument's volatility.
    """
    a = atr(bars, atr_window)
    brick = float(a.median() * atr_mult)
    if not np.isfinite(brick) or brick <= 0:
        brick = float(bars["close"].median() * 0.01)  # 1% fallback
    return renko_series(bars["close"], brick)


# ---------------------------------------------------------------------------
# Moving-average crossover (long/flat)
# ---------------------------------------------------------------------------
def crossover_position(price: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """Long/flat crossover state: +1 when fast SMA > slow SMA, else 0.

    State is computed on the price series itself (raw close or Renko close); it is the
    *known-at-close-of-t* signal — the one-day execution lag is applied by
    :func:`crossover_returns`, not here.
    """
    f = price.rolling(fast, min_periods=fast).mean()
    s = price.rolling(slow, min_periods=slow).mean()
    return (f > s).astype(float)


def crossover_returns(
    bars: pd.DataFrame,
    signal_price: pd.Series,
    fast: int = 10,
    slow: int = 30,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily strategy returns for a long/flat crossover driven by ``signal_price``.

    The crossover state known at the close of *t* earns the **raw close-to-close return
    of t+1** (one ``shift``, applied once — the documented execution lag).  Turnover is
    one-way × NAV: each change in position pays ``cost_bps`` once.  Returns a daily
    arithmetic-return Series (NaN-free, aligned to ``bars.index``).
    """
    pos = crossover_position(signal_price, fast, slow)
    pos = pos.reindex(bars.index).fillna(0.0)
    # underlying close-to-close returns of the actual tape
    ret = bars["close"].pct_change().fillna(0.0)
    # one execution lag: position set at t earns t+1's return
    strat = pos.shift(1).fillna(0.0) * ret
    # one-way turnover cost on every position change
    turn = pos.diff().abs().fillna(0.0)
    strat = strat - turn * cost_bps * 1e-4
    return strat


def buy_and_hold(bars: pd.DataFrame) -> pd.Series:
    """Daily buy-and-hold returns of the underlying tape."""
    return bars["close"].pct_change().fillna(0.0)


# ---------------------------------------------------------------------------
# Performance + HAC inference
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West (HAC) t-stat of the mean of a return series vs zero."""
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


def perf(returns: pd.Series) -> dict:
    """Headline stats for a daily-return strategy: annualised return, vol, Sharpe,
    HAC *t* (on daily returns), max drawdown, and trade/exposure proxies."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = r.size
    if n == 0:
        return dict(n=0, ann_ret=float("nan"), ann_vol=float("nan"),
                    sharpe=float("nan"), tstat=float("nan"), max_dd=float("nan"),
                    exposure=float("nan"))
    mu_d, sd_d = r.mean(), r.std(ddof=1)
    ann_ret = (1.0 + mu_d) ** TRADING_DAYS - 1.0
    ann_vol = sd_d * np.sqrt(TRADING_DAYS)
    sharpe = (mu_d / sd_d * np.sqrt(TRADING_DAYS)) if sd_d > 0 else float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    max_dd = float((curve / peak - 1.0).min())
    return dict(
        n=int(n), ann_ret=float(ann_ret), ann_vol=float(ann_vol),
        sharpe=float(sharpe), tstat=hac_tstat(r), max_dd=max_dd,
        exposure=float((np.abs(r) > 0).mean()),
    )


def sharpe(returns: pd.Series) -> float:
    s = pd.Series(returns).dropna().to_numpy(dtype=float)
    if s.size < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Placebo: return-permutation null
# ---------------------------------------------------------------------------
def permutation_pvalue(
    bars: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    atr_window: int = 14,
    atr_mult: float = 1.0,
    n_perm: int = 1000,
    seed: int = 449,
) -> dict:
    """Return-permutation placebo for the *incremental* Renko-over-raw Sharpe.

    Shuffle the daily log-returns (destroying all serial structure), rebuild a price
    path, recompute the Renko-crossover Sharpe minus the raw-crossover Sharpe, and count
    how often the shuffled delta matches or beats the observed delta.  Under the null
    (no exploitable serial structure) the brick filter adds nothing, so a small *p* would
    mean the Renko advantage is not a permutation artefact.
    """
    rng = np.random.default_rng(seed)
    log_ret = np.log(bars["close"] / bars["close"].shift(1)).dropna().to_numpy()
    base_close = float(bars["close"].iloc[0])

    def _delta(prices: pd.Series, frame: pd.DataFrame) -> float:
        renko = renko_from_atr(frame, atr_window, atr_mult)
        s_renko = sharpe(crossover_returns(frame, renko, fast, slow))
        s_raw = sharpe(crossover_returns(frame, prices, fast, slow))
        return s_renko - s_raw

    obs = _delta(bars["close"], bars)

    count = 0
    valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(log_ret)
        close = base_close * np.exp(np.cumsum(np.concatenate([[0.0], perm])))
        idx = bars.index[: close.size]
        frame = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999,
             "close": close, "volume": 1.0},
            index=idx,
        )
        d = _delta(frame["close"], frame)
        if np.isfinite(d):
            valid += 1
            if d >= obs:
                count += 1
    pval = (count + 1) / (valid + 1) if valid else float("nan")
    return dict(observed_delta=float(obs), p_value=float(pval), n_valid=int(valid))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    atr_window: int = 14,
    atr_mult: float = 1.0,
    cost_bps: float = 0.0,
) -> dict:
    """Full head-to-head on one tape: Renko-crossover vs raw-crossover vs buy-and-hold.

    Returns a dict of ``perf`` dicts plus the brick size and the incremental Sharpe
    (Renko minus raw).  Costs (one-way × NAV) are charged on both crossover arms.
    """
    renko = renko_from_atr(bars, atr_window, atr_mult)
    a = atr(bars, atr_window)
    brick = float(a.median() * atr_mult)

    r_renko = crossover_returns(bars, renko, fast, slow, cost_bps)
    r_raw = crossover_returns(bars, bars["close"], fast, slow, cost_bps)
    r_bh = buy_and_hold(bars)

    p_renko = perf(r_renko)
    p_raw = perf(r_raw)
    p_bh = perf(r_bh)
    return dict(
        renko=p_renko, raw=p_raw, bh=p_bh,
        brick=brick,
        delta_sharpe=p_renko["sharpe"] - p_raw["sharpe"],
        delta_vs_bh=p_renko["sharpe"] - p_bh["sharpe"],
    )
