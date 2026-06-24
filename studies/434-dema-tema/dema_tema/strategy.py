"""Indicators, timing rule and honest arbiters — Study 434 (DEMA & TEMA).

The claim: ordinary moving averages **lag** price, so trend signals built on them are late.
The fix sold in every charting package is **DEMA** (Mulloy 1994) and **TEMA** — double- and
triple-smoothed EMAs that algebraically cancel most of the lag while staying smooth. The
promise: *"same trend signal, but earlier — so you make more money."* We test whether that
extra responsiveness turns into a **better net, excess-of-cash Sharpe** than the obvious
simpler benchmarks (a plain SMA and a plain EMA), or whether it just trades more for nothing.

The indicators
--------------
* ``ema(x, n)``  — standard exponential MA, span ``n`` (pandas ``ewm``, ``adjust=False``).
* ``dema(x, n)`` = ``2*EMA(x) - EMA(EMA(x))`` — Mulloy's double EMA, lag roughly halved.
* ``tema(x, n)`` = ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`` — triple EMA, lag cut further.
* ``sma(x, n)``  — the laggy simple MA, the honest benchmark.

The timing rule (long / flat)
-----------------------------
A textbook trend filter: **be long when the close is above its moving average, flat
otherwise** (cash earns 0 in the gross book; excess-of-cash handled in the Sharpe race).
The signal is read at the close of day *t*; the position earns the return of day *t+1*
(one documented ``shift``). Turnover is one-way × NAV; costs are charged on |Δposition|.

Inference, the desk's shared spirit
-----------------------------------
* a **HAC (Newey-West) t** on the strategy's *excess-over-buy-and-hold* daily return — the
  honest "did the rule beat just holding the thing?" test;
* a **block / sign-flip permutation placebo** on the timing signal: shuffle the position
  series in circular blocks and ask how often a random schedule of the same long/flat days
  matches the real net Sharpe (the right null for an autocorrelated timing rule);
* **net Sharpe** annualised, **excess-vs-excess** (strategy excess-of-cash vs buy-and-hold
  excess-of-cash), gross *and* net of one-way costs × turnover;
* the same battery run on **SMA and EMA** so the "DEMA/TEMA is better" claim is actually
  raced against the simpler thing it claims to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def dema(x: pd.Series, n: int) -> pd.Series:
    """Double EMA (Mulloy 1994): 2*EMA - EMA(EMA). Lag roughly halved vs EMA."""
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 2.0 * e1 - e2


def tema(x: pd.Series, n: int) -> pd.Series:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)). Lag cut further than DEMA."""
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3.0 * e1 - 3.0 * e2 + e3


MA_FUNCS = {"sma": sma, "ema": ema, "dema": dema, "tema": tema}


def moving_average(x: pd.Series, kind: str, n: int) -> pd.Series:
    try:
        return MA_FUNCS[kind](x, n)
    except KeyError as exc:
        raise ValueError(f"unknown MA kind {kind!r}; pick {list(MA_FUNCS)}") from exc


# --------------------------------------------------------------------------- #
# Timing rule -> position (long / flat), one documented lag
# --------------------------------------------------------------------------- #
def trend_position(close: pd.Series, kind: str, n: int) -> pd.Series:
    """Long (1) when close > its ``kind`` moving average, flat (0) otherwise.

    The signal uses information through the close of day *t*. The caller pairs it with the
    *t+1* return via one ``shift`` in :func:`backtest`. Bars before the MA is defined are
    flat. Returns a 0/1 position series aligned to ``close``.
    """
    ma = moving_average(close, kind, n)
    pos = (close > ma).astype(float)
    pos[ma.isna()] = 0.0
    return pos


# --------------------------------------------------------------------------- #
# Backtest — gross & net, excess-of-cash, one-day lag, one-way costs × turnover
# --------------------------------------------------------------------------- #
def backtest(bars: pd.DataFrame, kind: str, n: int, cost_bps: float = 2.0,
             rf_annual: float = 0.0) -> dict:
    """Run the long/flat trend rule and return daily books + headline stats.

    * **Lag.** Position known at close of *t* earns the return of *t+1*: ``pos.shift(1)``.
    * **Costs.** One-way ``cost_bps`` × |Δposition| (turnover = one-way × NAV) charged the
      day the position changes — entries and exits both pay.
    * **Excess-of-cash.** When flat the strategy earns the risk-free rate; the Sharpe race
      below is run on returns **in excess of cash** for both the rule and buy-and-hold, so a
      part-time-in-cash rule is compared fairly to an always-invested benchmark.

    Returns a dict with the gross/net strategy return series, the buy-and-hold series,
    turnover, and the annualised net excess Sharpe + its components.
    """
    close = bars["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)
    rf_daily = rf_annual / TRADING_DAYS

    pos = trend_position(close, kind, n)
    held = pos.shift(1).fillna(0.0)                  # one documented lag
    dpos = held.diff().abs().fillna(held.abs())      # turnover proxy (one-way × NAV)
    cost = dpos * (cost_bps / 1e4)

    # Gross book: invested -> asset return; flat -> cash (rf).
    strat_gross = held * asset_ret + (1.0 - held) * rf_daily
    strat_net = strat_gross - cost

    bh = asset_ret.copy()                            # buy-and-hold (always invested)

    # Excess-of-cash for the Sharpe race (both legs).
    strat_excess = strat_net - rf_daily
    bh_excess = bh - rf_daily

    out = {
        "kind": kind, "n": n, "cost_bps": cost_bps,
        "pos": pos, "held": held, "asset_ret": asset_ret,
        "strat_gross": strat_gross, "strat_net": strat_net, "bh": bh,
        "strat_excess": strat_excess, "bh_excess": bh_excess,
        "cost": cost,
        "turnover_per_yr": float(dpos.sum() / (len(dpos) / TRADING_DAYS)),
        "time_in_market": float(held.mean()),
        "sharpe_net": ann_sharpe(strat_excess),
        "sharpe_gross": ann_sharpe(strat_gross - rf_daily),
        "sharpe_bh": ann_sharpe(bh_excess),
        "cagr_net": cagr(strat_net),
        "cagr_bh": cagr(bh),
    }
    out["sharpe_minus_bh"] = out["sharpe_net"] - out["sharpe_bh"]
    return out


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def ann_sharpe(daily_excess: pd.Series) -> float:
    r = np.asarray(daily_excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily_ret: pd.Series) -> float:
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    yrs = r.size / TRADING_DAYS
    if growth <= 0 or yrs <= 0:
        return float("nan")
    return growth ** (1.0 / yrs) - 1.0


def hac_t(daily: pd.Series, lags: int | None = None) -> float:
    """Newey-West HAC t-stat of mean(daily) vs 0 (no hard quantlab dependency).

    Used on the strategy's daily **excess-over-buy-and-hold** return: a HAC *t* >= 2 there
    is what the desk requires to stamp the timing rule's outperformance ``REAL``.
    """
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Permutation placebo — circular block shuffle of the position schedule
# --------------------------------------------------------------------------- #
def block_permutation_pvalue(bars: pd.DataFrame, kind: str, n: int,
                             cost_bps: float = 2.0, block: int = 21,
                             n_draws: int = 2000, seed: int = 434) -> dict:
    """Null: a random *schedule* of the same long/flat days, in circular blocks.

    We hold the realised position series fixed in shape (same fraction of days long, same
    autocorrelation via block size ``block``) but **rotate / re-tile** it at random against
    the return tape, recomputing the net excess Sharpe each draw. ``p`` = P[shuffled Sharpe
    >= observed]. This is the honest answer to "could a random timing schedule of the same
    activity level have looked this good on this tape?".
    """
    bt = backtest(bars, kind, n, cost_bps=cost_bps)
    obs = bt["sharpe_net"]
    asset_ret = bt["asset_ret"].to_numpy()
    held = bt["held"].to_numpy()
    N = len(held)
    rng = np.random.default_rng(seed)

    # split the held series into contiguous blocks, then permute block order + circular shift
    n_blocks = int(np.ceil(N / block))
    blocks = [held[i * block:(i + 1) * block] for i in range(n_blocks)]
    sharpes = np.empty(n_draws)
    for d in range(n_draws):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate([blocks[i] for i in order])[:N]
        shift = int(rng.integers(0, N))
        shuffled = np.roll(shuffled, shift)
        held_s = np.empty(N)
        held_s[0] = 0.0
        held_s[1:] = shuffled[:-1]                    # same one-day lag
        dpos = np.abs(np.diff(np.concatenate([[0.0], held_s])))
        cost = dpos * (cost_bps / 1e4)
        strat = held_s * asset_ret - cost             # rf=0 in placebo (cash term cancels)
        sd = strat.std(ddof=1)
        sharpes[d] = strat.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    p = float((sharpes >= obs).mean())
    return {"obs_sharpe": float(obs), "placebo_mean": float(np.nanmean(sharpes)),
            "p_value": p, "draws": sharpes}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, n: int = 50, cost_bps: float = 2.0,
                   kinds: tuple[str, ...] = ("sma", "ema", "dema", "tema"),
                   placebo_for: str = "tema", n_draws: int = 2000,
                   seed: int = 434) -> dict:
    """Race all MA kinds at window ``n`` and run the placebo on ``placebo_for``.

    Returns a dict: ``table`` (a DataFrame, one row per MA kind with Sharpe gross/net,
    Sharpe vs buy-and-hold, HAC t on excess-over-BH, turnover, time-in-market), the
    buy-and-hold Sharpe, and the placebo result on the headline (DEMA/TEMA) rule.
    """
    rows = []
    books = {}
    for k in kinds:
        bt = backtest(bars, k, n, cost_bps=cost_bps)
        books[k] = bt
        excess_over_bh = bt["strat_net"] - bt["bh"]
        rows.append({
            "kind": k,
            "sharpe_gross": bt["sharpe_gross"],
            "sharpe_net": bt["sharpe_net"],
            "sharpe_minus_bh": bt["sharpe_minus_bh"],
            "hac_t_vs_bh": hac_t(excess_over_bh),
            "turnover_per_yr": bt["turnover_per_yr"],
            "time_in_market": bt["time_in_market"],
            "cagr_net": bt["cagr_net"],
        })
    table = pd.DataFrame(rows).set_index("kind")
    placebo = block_permutation_pvalue(bars, placebo_for, n, cost_bps=cost_bps,
                                       n_draws=n_draws, seed=seed)
    return {
        "table": table,
        "sharpe_bh": books[kinds[0]]["sharpe_bh"],
        "books": books,
        "placebo": placebo,
        "n": n, "cost_bps": cost_bps,
    }
