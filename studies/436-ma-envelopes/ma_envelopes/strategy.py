"""The envelope timing rule and its honest controls — Study 436 (MA Envelopes).

A **Moving-Average Envelope** is a simple moving average flanked by two parallel lines a
fixed *percent* above and below it::

    mid(t)   = SMA(close, n)
    upper(t) = mid(t) * (1 + k)
    lower(t) = mid(t) * (1 - k)

The half-width ``k`` is a constant fraction of the average — it does *not* breathe with
volatility, which is the one structural difference from Bollinger Bands (whose half-width is
``m * rolling_std``).  This study's headline race: **does the percent envelope beat the
Bollinger Band as a timing rule?**

Two folk reads, each turned into a daily **long/flat** position (and, where natural, a
long/short variant):

- **Mean-reversion** — long while ``close < lower`` ("oversold"), flat once back inside.
- **Breakout** — long while ``close > upper`` ("trend confirmed"), flat once back inside.

We also build the two obvious simpler benchmarks so the "it's better" claim is actually
tested:

- **Bollinger(n, m)** — the same two rules with a volatility-scaled band.
- **SMA(n)** — the bare "price above its average" filter (no band at all).

The whole thing is raced, **net of one-way costs × NAV**, against **buy-and-hold**, on an
**excess-of-cash vs excess-of-cash** Sharpe (the timing rule sits in cash part-time, so an
excess-vs-excess race is the only fair one).  Inference is a Newey-West HAC *t* on the daily
strategy returns plus a **block-permutation placebo** that scrambles the *position* in
blocks (destroying the band's timing while preserving the marginal exposure).

**One execution lag, documented exactly:** the band and the rule are evaluated on the close
of bar *t*; the resulting position is ``shift(1)`` so it earns the return of bar *t+1*.  One
shift, applied once, in :func:`book_returns`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------
def envelope(close: pd.Series, n: int = 20, k: float = 0.05) -> pd.DataFrame:
    """Percent moving-average envelope: SMA(n) and fixed ±k% bands.

    Returns a frame with columns ``mid, upper, lower`` indexed like ``close``; the first
    ``n - 1`` bars are NaN (window not yet full).
    """
    mid = close.rolling(n).mean()
    return pd.DataFrame(
        {"mid": mid, "upper": mid * (1 + k), "lower": mid * (1 - k)}, index=close.index
    )


def bollinger(close: pd.Series, n: int = 20, m: float = 2.0) -> pd.DataFrame:
    """Bollinger Band: SMA(n) and ±m·rolling-std bands (volatility-scaled half-width)."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + m * sd, "lower": mid - m * sd}, index=close.index
    )


# ---------------------------------------------------------------------------
# Position generators (state = the daily target weight, 0/1 long-flat or -1/+1 L/S)
# ---------------------------------------------------------------------------
def _stateful_band_position(
    close: pd.Series,
    band: pd.DataFrame,
    mode: str = "reversion",
    long_short: bool = False,
) -> pd.Series:
    """Turn a band into a daily target position, as a *stateful* long/flat rule.

    - ``mode="reversion"``: enter long when ``close < lower``; hold until ``close`` crosses
      back above ``mid``; then flat.  (Buy oversold, exit at the mean — the recipe's own
      exit.)  With ``long_short=True`` the symmetric short leg fires on ``close > upper``,
      covered when ``close`` falls back below ``mid``.
    - ``mode="breakout"``: enter long when ``close > upper``; hold until ``close`` crosses
      back below ``mid``; then flat.  With ``long_short=True`` short on ``close < lower``.

    The position is the *state at the close of bar t* (no shift here — the one-day lag is
    applied in :func:`book_returns`).
    """
    c = close.to_numpy(dtype=float)
    lo = band["lower"].to_numpy(dtype=float)
    hi = band["upper"].to_numpy(dtype=float)
    mid = band["mid"].to_numpy(dtype=float)
    n = len(c)
    pos = np.zeros(n)
    state = 0
    for i in range(n):
        if np.isnan(mid[i]):
            pos[i] = 0
            continue
        if mode == "reversion":
            if state == 0:
                if c[i] < lo[i]:
                    state = 1
                elif long_short and c[i] > hi[i]:
                    state = -1
            elif state == 1 and c[i] >= mid[i]:
                state = 0
            elif state == -1 and c[i] <= mid[i]:
                state = 0
        else:  # breakout
            if state == 0:
                if c[i] > hi[i]:
                    state = 1
                elif long_short and c[i] < lo[i]:
                    state = -1
            elif state == 1 and c[i] <= mid[i]:
                state = 0
            elif state == -1 and c[i] >= mid[i]:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=close.index, name="position")


def envelope_position(
    close: pd.Series, n: int = 20, k: float = 0.05, mode: str = "reversion",
    long_short: bool = False,
) -> pd.Series:
    """Daily target position from a percent envelope (see :func:`_stateful_band_position`)."""
    return _stateful_band_position(close, envelope(close, n, k), mode, long_short)


def bollinger_position(
    close: pd.Series, n: int = 20, m: float = 2.0, mode: str = "reversion",
    long_short: bool = False,
) -> pd.Series:
    """Daily target position from a Bollinger Band (the volatility-scaled benchmark)."""
    return _stateful_band_position(close, bollinger(close, n, m), mode, long_short)


def sma_position(close: pd.Series, n: int = 200) -> pd.Series:
    """Daily target position from the bare SMA filter: long while close > SMA(n), else flat.

    The simplest possible "trend" benchmark with no band at all.
    """
    mid = close.rolling(n).mean()
    pos = (close > mid).astype(float)
    pos[mid.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=close.index, name="position")


# ---------------------------------------------------------------------------
# Book — one execution lag, costs one-way × NAV, shorts pay borrow
# ---------------------------------------------------------------------------
def book_returns(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    borrow_bps_yr: float = 50.0,
    rf_bps_yr: float = 0.0,
) -> pd.DataFrame:
    """Daily *excess-of-cash* net returns for a long/flat (or long/short) timing rule.

    Convention (the desk's one execution lag, applied once here): the position known at the
    close of bar *t* — ``position`` — is shifted forward one bar so it earns the simple
    return of bar *t+1*.  ``cost_bps`` is a one-way cost charged on |Δposition| × NAV at
    each rebalance (a flat→long→flat round-trip pays it twice).  A short leg pays
    ``borrow_bps_yr`` daily while held.  Returns are **excess of the cash rate**
    ``rf_bps_yr`` (subtracted on the *unexposed* fraction so a flat day earns zero excess,
    matching the buy-and-hold excess benchmark).

    Returns a frame: ``ret`` (asset simple return), ``pos`` (lagged position),
    ``gross`` (pos·ret), ``net`` (gross − costs − borrow, all excess-of-cash).
    """
    ret = close.pct_change().fillna(0.0)
    pos = position.shift(1).fillna(0.0)

    gross = pos * ret
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (cost_bps * 1e-4)
    borrow = (pos < 0).astype(float) * (-pos) * (borrow_bps_yr * 1e-4 / TRADING_DAYS)
    # excess-of-cash: the asset return already embeds the risk-free drift; subtract rf on
    # the *exposed notional* so long-flat is compared excess-vs-excess with buy-and-hold.
    rf = (rf_bps_yr * 1e-4 / TRADING_DAYS)
    gross_excess = gross - pos.abs() * rf
    net = gross_excess - cost - borrow

    return pd.DataFrame(
        {"ret": ret, "pos": pos, "gross": gross_excess, "net": net}, index=close.index
    )


def buy_and_hold_excess(close: pd.Series, rf_bps_yr: float = 0.0) -> pd.Series:
    """Daily excess-of-cash return of a fully-invested buy-and-hold position."""
    ret = close.pct_change().fillna(0.0)
    return ret - (rf_bps_yr * 1e-4 / TRADING_DAYS)


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def annualised_sharpe(daily: pd.Series) -> float:
    r = daily.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC *t*-stat on the mean of a daily return series (H0: mean = 0)."""
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


def cagr(daily: pd.Series) -> float:
    r = daily.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    years = r.size / TRADING_DAYS
    return float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    r = daily.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else float("nan")


def summarize(book: pd.DataFrame, col: str = "net") -> dict:
    """Headline stats for one daily return column: Sharpe, HAC *t*, CAGR, maxDD, exposure."""
    daily = book[col]
    exposure = float((book["pos"].abs() > 0).mean()) if "pos" in book else float("nan")
    turn_yr = (
        float(book["pos"].diff().abs().fillna(book["pos"].abs()).sum() / (len(book) / TRADING_DAYS))
        if "pos" in book else float("nan")
    )
    return {
        "sharpe": annualised_sharpe(daily),
        "tstat": hac_tstat(daily),
        "cagr": cagr(daily),
        "maxdd": max_drawdown(daily),
        "exposure": exposure,
        "turnover_yr": turn_yr,
        "n_days": int(len(daily)),
    }


# ---------------------------------------------------------------------------
# Block-permutation placebo — scramble the position in blocks, keep exposure
# ---------------------------------------------------------------------------
def block_permutation_pvalue(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    n_perm: int = 2000,
    block: int = 21,
    seed: int = 436,
    stat: str = "sharpe",
) -> dict:
    """Two-sided placebo: is the rule's Sharpe better than the *same exposure*, mis-timed?

    We chop the realised position path into ``block``-day blocks and shuffle the block order
    (circular block bootstrap of the *position*), recompute the net Sharpe, and ask how
    often a scrambled timing matches or beats the real one.  This preserves the marginal
    long-fraction (and roughly the autocorrelation of being in the market) while destroying
    the band's specific timing — the sharpest test of "is it the band, or just being long?".

    Returns ``{stat_real, p_value, perm_mean}``.
    """
    rng = np.random.default_rng(seed)
    real = summarize(book_returns(close, position, cost_bps=cost_bps), "net")[stat]

    pos = position.to_numpy(dtype=float)
    n = len(pos)
    n_blocks = int(np.ceil(n / block))
    perms = np.empty(n_perm)
    for j in range(n_perm):
        starts = rng.integers(0, n, size=n_blocks)
        chunks = [np.take(pos, range(s, s + block), mode="wrap") for s in starts]
        scrambled = np.concatenate(chunks)[:n]
        sp = pd.Series(scrambled, index=position.index)
        perms[j] = summarize(book_returns(close, sp, cost_bps=cost_bps), "net")[stat]
    perms = perms[np.isfinite(perms)]
    p = float((perms >= real).mean()) if perms.size else float("nan")
    return {"stat_real": float(real), "p_value": p, "perm_mean": float(np.nanmean(perms))}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    close: pd.Series,
    n: int = 20,
    k: float = 0.05,
    mode: str = "reversion",
    cost_bps: float = 1.0,
    boll_m: float = 2.0,
    sma_n: int = 200,
    n_perm: int = 2000,
    seed: int = 436,
) -> dict:
    """One full teardown on a single tape: envelope vs Bollinger vs SMA vs buy-and-hold.

    Returns a dict of summaries (each a :func:`summarize` dict) plus the permutation
    p-value for the envelope rule.  All numbers are excess-of-cash; ``cost_bps`` is the
    one-way per-rebalance cost.
    """
    env_pos = envelope_position(close, n, k, mode)
    boll_pos = bollinger_position(close, n, boll_m, mode)
    sma_pos = sma_position(close, sma_n)

    env_book = book_returns(close, env_pos, cost_bps=cost_bps)
    boll_book = book_returns(close, boll_pos, cost_bps=cost_bps)
    sma_book = book_returns(close, sma_pos, cost_bps=cost_bps)

    bh = buy_and_hold_excess(close)
    bh_book = pd.DataFrame({"net": bh, "gross": bh, "pos": pd.Series(1.0, index=close.index)})

    perm = block_permutation_pvalue(close, env_pos, cost_bps=cost_bps, n_perm=n_perm, seed=seed)

    return {
        "envelope": summarize(env_book, "net"),
        "envelope_gross": summarize(env_book, "gross"),
        "bollinger": summarize(boll_book, "net"),
        "sma": summarize(sma_book, "net"),
        "buyhold": summarize(bh_book, "net"),
        "perm": perm,
    }
