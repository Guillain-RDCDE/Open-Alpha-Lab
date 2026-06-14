"""The strategy and its honest controls — Study 137 (Mansfield-RS).

Stan Weinstein's Stage Analysis, from *Secrets for Profiting in Bull and Bear
Markets* (1988): buy stocks in "Stage 2" — price above a rising 30-week SMA AND
Mansfield Relative Strength vs the market turning positive.  The recipe is a joint
trend + relative-strength momentum filter; Weinstein framed it as identifying stocks
that are *ready to outperform*.

We test whether Stage-2 membership predicts forward outperformance vs the benchmark
(SPY), using two honest baselines:

1. **Equal-weight buy-and-hold** (the market).  After all, any positive-momentum
   filter will over-weight stocks that have been rising — the question is whether it
   does better than simply holding *everything* equally.
2. **Random-entry control**: the same holding-period return on randomly selected
   (non-Stage-2) weeks.  This pins down whether the timing of Stage-2 entry adds
   anything, or whether we're just selecting high-beta stocks that always outperform.

The inference number is a HAC t-stat on the per-entry excess return (stock minus
SPY) for Stage-2 entries, vs the same for a random-entry control.

No look-ahead: Stage-2 is confirmed at the close of week *t*; the hypothetical
position is entered at week *t+1*'s open.  For the Mansfield RS line, we use the
close at week *t* for both the stock and the benchmark — the signal is knowable at
the Friday close before the following Monday's open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mansfield Relative Strength (MRS) and Stage-2 signal
# ---------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average with minimum-periods guard."""
    return s.rolling(n, min_periods=n).mean()


def mansfield_rs(
    stock_close: pd.Series,
    bm_close: pd.Series,
    sma_n: int = 30,
) -> pd.Series:
    """Mansfield Relative Strength vs the benchmark.

    MRS_t = (stock_t / sma30_stock_t) / (benchmark_t / sma30_benchmark_t) - 1

    A positive MRS means the stock is trading above its own 30-week baseline by
    *more* than the benchmark is — i.e. the stock is outperforming the market
    on a trend-adjusted basis.  This is the RS component of Weinstein's Stage 2.

    Both series are aligned on their common index before computation.
    """
    common = stock_close.index.intersection(bm_close.index)
    sc = stock_close.reindex(common)
    bc = bm_close.reindex(common)
    sc_sma = sma(sc, sma_n)
    bc_sma = sma(bc, sma_n)
    # Avoid division-by-zero; any NaN propagates naturally
    mrs = (sc / sc_sma) / (bc / bc_sma) - 1.0
    return mrs


def stage2_signal(
    stock_bars: pd.DataFrame,
    bm_bars: pd.DataFrame,
    sma_n: int = 30,
    min_slope_weeks: int = 4,
) -> pd.Series:
    """Boolean Stage-2 flag for each week.

    A week is Stage 2 when ALL of:
    1. stock close > 30-week SMA (price in the upper zone).
    2. 30-week SMA is rising (slope over ``min_slope_weeks`` is positive).
    3. Mansfield RS > 0 (stock outperforms the benchmark on a 30w baseline).

    Returns a boolean Series indexed on the stock's weekly dates.  The first
    ``sma_n`` bars are NaN/False due to warmup.  Computed entirely on data up to and
    including week *t* — no look-ahead.
    """
    sc = stock_bars["close"]
    bc = bm_bars["close"].reindex(sc.index, method="ffill")

    sc_sma = sma(sc, sma_n)
    sma_slope = sc_sma - sc_sma.shift(min_slope_weeks)

    mrs = mansfield_rs(sc, bc, sma_n=sma_n)

    above_sma = sc > sc_sma               # condition 1
    sma_rising = sma_slope > 0            # condition 2
    rs_positive = mrs > 0                  # condition 3

    return above_sma & sma_rising & rs_positive


# ---------------------------------------------------------------------------
# Stage-entry signal table
# ---------------------------------------------------------------------------
def stage2_entries(
    stock_bars: pd.DataFrame,
    bm_bars: pd.DataFrame,
    sma_n: int = 30,
    min_slope_weeks: int = 4,
) -> pd.DataFrame:
    """Weeks where the stock *enters* Stage 2 (transitions from not-Stage-2 to Stage-2).

    Returns a DataFrame with columns:
    - ``in_stage2``: the week where Stage 2 began (bool transition True).

    The entry signal is formed at the end of week *t* (Friday close); the
    position is taken at week *t+1*'s open.
    """
    flag = stage2_signal(stock_bars, bm_bars, sma_n=sma_n, min_slope_weeks=min_slope_weeks)
    # Entry = first week where flag turns True
    entered = flag & (~flag.shift(1, fill_value=False))
    idx = entered[entered].index
    return pd.DataFrame({"dir": 1}, index=idx)


# ---------------------------------------------------------------------------
# Forward-return ledger (hold for N weeks, compute excess return vs SPY)
# ---------------------------------------------------------------------------
def run_forward_returns(
    stock_bars: pd.DataFrame,
    bm_bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_weeks: int = 13,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hold for ``hold_weeks`` weeks after entry; record the stock's excess return vs SPY.

    Entry is at the week *t+1* open (next bar after the signal week).  Exit is at
    the close of week *t+1+hold_weeks*.  Excess return = stock gross return minus
    benchmark gross return over the same window.

    ``directions`` overrides entry direction (for the random-control arm: all +1 since
    Stage 2 is always long).  ``cost_bps`` is a round-trip cost on the gross return.

    Columns: ``entry_ts, dir, stock_ret, bm_ret, excess_ret, ret_gross, ret_net``.
    """
    sc = stock_bars["close"]
    so = stock_bars["open"]
    bc = bm_bars["close"].reindex(sc.index, method="ffill")
    bo = bm_bars["open"].reindex(so.index, method="ffill")

    pos = {ts: i for i, ts in enumerate(sc.index)}
    n = len(sc)

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n:
            continue
        e = i + 1  # enter at next week's open
        exit_i = min(e + hold_weeks - 1, n - 1)
        if exit_i <= e:
            continue

        entry_s = so.iat[e]
        exit_s = sc.iat[exit_i]
        entry_b = bo.iat[e]
        exit_b = bc.iat[exit_i]

        if entry_s <= 0 or entry_b <= 0:
            continue

        stock_ret = d * (exit_s - entry_s) / entry_s
        bm_ret = (exit_b - entry_b) / entry_b  # always long the benchmark
        excess = stock_ret - bm_ret

        rows.append(
            {
                "entry_ts": sc.index[e],
                "dir": int(d),
                "stock_ret": stock_ret,
                "bm_ret": bm_ret,
                "excess_ret": excess,
                "ret_gross": excess,
                "ret_net": excess - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Random-entry control
# ---------------------------------------------------------------------------
def random_entries(
    stock_bars: pd.DataFrame,
    n_entries: int,
    seed: int = 0,
    sma_n: int = 30,
) -> pd.DataFrame:
    """``n_entries`` random Stage-agnostic weeks as control entries.

    We sample from weeks after the SMA warmup period, so we are comparing
    Stage-2 entries against *all* other entry timing, not just non-Stage-2.
    This is the honest question: does Stage-2 *timing* add value, or do the
    selected stocks just always outperform?
    """
    sc = stock_bars["close"]
    valid = sc.index[sma_n:]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=min(n_entries, len(valid)), replace=False)
    chosen = pd.DatetimeIndex(sorted(chosen))
    return pd.DataFrame({"dir": 1}, index=chosen)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — not used for long-only, kept for API symmetry."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Trade-ledger summary (HAC t-stat on excess returns)
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate (excess > 0), mean excess return (bps/trade),
    the P&L skew, and a HAC Newey-West t-stat on the mean — the inference-bar
    number that decides whether the edge over the benchmark is distinguishable
    from zero.
    """
    if ledger.empty:
        return {k: float("nan") for k in
                ["n_trades", "win_rate", "mean_bps", "skew", "tstat"]}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC t-stat — same kernel as other studies in this family
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Equal-weight benchmark return (naive buy-and-hold, same hold period)
# ---------------------------------------------------------------------------
def basket_bm_return(
    tapes: dict[str, pd.DataFrame],
    hold_weeks: int = 13,
    start_date: str | None = None,
) -> pd.Series:
    """Equal-weight buy-and-hold return across the stock basket for comparison.

    At each week *t* (after warmup), assume you buy an equal share of every stock
    in ``tapes`` and hold for ``hold_weeks``.  This is the naïve baseline that the
    Stage-2 screen must beat to add value.

    Returns a Series of per-entry excess returns (nan where there is insufficient
    forward data).
    """
    tickers = [t for t in tapes if t != BENCHMARK]
    bm = tapes.get(BENCHMARK) or next(iter(tapes.values()))
    all_rets = []
    for tk in tickers:
        sc = tapes[tk]["close"]
        if start_date:
            sc = sc[sc.index >= start_date]
        for i in range(len(sc) - hold_weeks):
            entry_px = sc.iat[i]
            exit_px = sc.iat[i + hold_weeks]
            bm_reindexed = bm["close"].reindex(sc.index, method="ffill")
            bm_e = bm_reindexed.iat[i]
            bm_x = bm_reindexed.iat[i + hold_weeks]
            if entry_px > 0 and bm_e > 0:
                all_rets.append((exit_px - entry_px) / entry_px - (bm_x - bm_e) / bm_e)
    return pd.Series(all_rets, dtype=float)


# Expose the benchmark ticker name for cross-module use
BENCHMARK = "SPY"
