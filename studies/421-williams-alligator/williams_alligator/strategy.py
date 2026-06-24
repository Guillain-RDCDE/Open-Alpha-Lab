"""The strategy and its honest controls — Study 421 (Williams Alligator).

Bill Williams' Alligator (in *Trading Chaos*, 1995) is built from three smoothed moving
averages (SMMA / Wilder RMA) of the **median price** (high+low)/2, each *displaced
forward* in time:

    Jaw   = SMMA(median, 13) shifted +8 bars   (the "blue" line)
    Teeth = SMMA(median,  8) shifted +5 bars   (the "red" line)
    Lips  = SMMA(median,  5) shifted +3 bars   (the "green" line)

The folk recipe:

- **Alligator sleeping** — the three lines are intertwined (close together). No trend;
  stay flat / out of the market.
- **Alligator waking & eating** — the lines fan out *in order*: Lips > Teeth > Jaw
  (uptrend → go long), or Lips < Teeth < Jaw (downtrend → go short). Ride the trend
  until the lines converge again.

We turn this into two daily timing rules and race them against buy-and-hold and against
the obvious simpler trend benchmark, **SMA(200)**:

1. **long/flat** — long SPY when the alligator is bullishly fanned, else in cash
   (T-bill). The fair, no-short version most retail traders would actually run.
2. **long/short** — long on a bullish fan, short on a bearish fan, flat when sleeping.

Honest arbiters, the desk's shared spirit:

- a **HAC (Newey-West) t-stat** on the strategy's mean daily return and on the
  return *difference* vs buy-and-hold (the Jobson-Korkie style Sharpe-difference test);
- a **permutation / block-shuffle placebo** — shuffle the signal in circular blocks and
  re-race, asking how often a trend-agnostic reshuffle of the *same exposure* beats B&H;
- **one documented execution lag** (signal known at the close of *t*, position held over
  the return of *t+1*) and **one-way costs × NAV** charged on every position change, with
  the short leg paying **borrow**;
- the **SMA(200) benchmark** run through the identical engine, so "the Alligator is
  better" is an actual head-to-head, not an assertion.

The decisive number is the NET Sharpe of the Alligator timing rule vs buy-and-hold
(excess-of-cash vs excess-of-cash), with its HAC difference t-stat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Canonical Bill Williams parameters: (period, forward shift)
JAW = (13, 8)
TEETH = (8, 5)
LIPS = (5, 3)


# ---------------------------------------------------------------------------
# Alligator indicator
# ---------------------------------------------------------------------------
def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed moving average (Wilder's RMA): EWM with alpha = 1/period.

    This is the smoothing Bill Williams' Alligator uses (equivalent to ``ewm(com=period-1,
    adjust=False)``). The first ``period`` bars warm up to NaN.
    """
    return series.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def alligator(bars: pd.DataFrame) -> pd.DataFrame:
    """The three Alligator lines (Jaw / Teeth / Lips), forward-shifted as Williams defines.

    Median price = (high + low) / 2. Each line is an SMMA of the median price, then
    *displaced forward* by its shift (``shift(+k)``), exactly as plotted on a chart — the
    line you see at bar *t* was computed from data up to bar *t - k*. The forward shift is
    therefore *conservative* (it only ever uses older data), so the lines at *t* are fully
    known at the close of *t*. Returns columns ``jaw``, ``teeth``, ``lips`` aligned to
    ``bars.index``.
    """
    median = (bars["high"] + bars["low"]) / 2.0
    jaw = smma(median, JAW[0]).shift(JAW[1])
    teeth = smma(median, TEETH[0]).shift(TEETH[1])
    lips = smma(median, LIPS[0]).shift(LIPS[1])
    return pd.DataFrame({"jaw": jaw, "teeth": teeth, "lips": lips}, index=bars.index)


# ---------------------------------------------------------------------------
# Signal generation — the "wake and eat" timing rule
# ---------------------------------------------------------------------------
def alligator_signal(bars: pd.DataFrame, mode: str = "long_flat") -> pd.Series:
    """Daily position signal from the Alligator fan, lagged one day (no look-ahead).

    The alligator is "fanned bullish" when ``lips > teeth > jaw`` (the green line leads,
    the blue lags — an uptrend), "fanned bearish" when ``lips < teeth < jaw`` (a
    downtrend), and "sleeping" otherwise (lines intertwined).

    - ``mode = "long_flat"`` → +1 when bullishly fanned, 0 otherwise (long or cash).
    - ``mode = "long_short"`` → +1 bullish fan, -1 bearish fan, 0 sleeping.

    The fan state is known at the close of bar *t*; the position is held over the return
    of bar *t+1*. We encode that with a single ``shift(1)`` here (one documented lag).
    Returns a signal Series in {-1, 0, +1} aligned to ``bars.index``.
    """
    a = alligator(bars)
    bull = (a["lips"] > a["teeth"]) & (a["teeth"] > a["jaw"])
    bear = (a["lips"] < a["teeth"]) & (a["teeth"] < a["jaw"])

    raw = pd.Series(0, index=bars.index, dtype=float)
    raw[bull] = 1.0
    if mode == "long_short":
        raw[bear] = -1.0
    elif mode != "long_flat":
        raise ValueError("mode must be 'long_flat' or 'long_short'")
    # any line still NaN (warmup) → flat
    raw[a.isna().any(axis=1)] = 0.0
    # one documented execution lag
    return raw.shift(1).rename("signal")


def sma_signal(bars: pd.DataFrame, n: int = 200, mode: str = "long_flat") -> pd.Series:
    """The benchmark trend rule: SMA(n) on the close, lagged one day.

    +1 when close > SMA(n); long/flat returns 0 below, long/short returns -1. Same lag
    convention as :func:`alligator_signal` so the race is apples-to-apples.
    """
    sma = bars["close"].rolling(n, min_periods=n).mean()
    above = (bars["close"] > sma)
    raw = pd.Series(0, index=bars.index, dtype=float)
    raw[above] = 1.0
    if mode == "long_short":
        raw[~above & sma.notna()] = -1.0
    raw[sma.isna()] = 0.0
    return raw.shift(1).rename("signal")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Apply a {-1,0,+1} position signal to daily close-to-close returns.

    - **Long (signal=+1)** earns the equity return; **flat (0)** earns the T-bill;
      **short (-1)** earns minus the equity return *plus* the T-bill (short rebate proxy)
      *minus* borrow.
    - ``cost_bps`` (one-way) is charged on **NAV** every time the position *changes*
      (``|Δsignal|`` units of turnover, so a +1 → -1 flip pays two one-way costs).
    - ``borrow_bps_ann`` is charged daily on the short leg only.

    The signal must already be lagged (signal at *t* acts on the return *t* → *t+1*,
    handled in the ``*_signal`` builders). Returns a frame with ``r_equity``, ``r_tbill``,
    ``signal``, ``r_strategy`` (net) and ``r_bh`` (buy-and-hold equity).
    """
    r_eq = np.log(bars["close"] / bars["close"].shift(1)).rename("r_equity")
    if isinstance(tbill_daily, pd.Series):
        r_tb = tbill_daily.reindex(bars.index).fillna(0.0)
    else:
        r_tb = pd.Series(float(tbill_daily), index=bars.index)
    r_tb.name = "r_tbill"

    sig = signal.reindex(bars.index).fillna(0.0)
    turnover = sig.diff().abs().fillna(sig.abs())  # first bar: entering from flat
    cost = turnover * (cost_bps * 1e-4)
    borrow = (sig < 0).astype(float) * (borrow_bps_ann * 1e-4 / TRADING_DAYS_PER_YEAR)

    # position return: long=+eq, flat=tbill, short=-eq + tbill
    long_leg = (sig > 0).astype(float) * r_eq
    flat_leg = (sig == 0).astype(float) * r_tb
    short_leg = (sig < 0).astype(float) * (-r_eq + r_tb)
    r_strat = (long_leg + flat_leg + short_leg - cost - borrow).rename("r_strategy")

    out = pd.concat([r_eq, r_tb, sig.rename("signal"), r_strat,
                     r_eq.rename("r_bh")], axis=1)
    return out.dropna(subset=["r_equity"])


# ---------------------------------------------------------------------------
# Performance + inference
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``r`` against 0."""
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


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised statistics for a daily return series.

    Returns Sharpe, annualised vol, CAGR (geometric), max drawdown, mean daily return in
    bps, and the HAC (Newey-West) t-stat on the mean — the inference-bar number.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ["n_days", "cagr", "sharpe", "vol_ann", "max_drawdown",
                 "mean_daily_bps", "tstat"]}
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))
    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": _hac_tstat(r.to_numpy()),
    }


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """HAC t-stat on the daily return *difference* r1 − r2 (Jobson-Korkie style).

    If the mean of (r1 − r2) is significantly positive at |t| ≥ 2, arm 1 has a reliably
    higher excess-return-per-unit-risk than arm 2 *on this tape*. This is the decisive
    "is the Alligator better than buy-and-hold?" number (excess-vs-excess: both arms here
    credit the cash leg, so the difference is excess-of-cash vs excess-of-cash).
    """
    diff = (r1 - r2).dropna()
    return _hac_tstat(diff.to_numpy())


# ---------------------------------------------------------------------------
# Permutation placebo — circular block shuffle of the signal
# ---------------------------------------------------------------------------
def block_permutation_pvalue(
    bars: pd.DataFrame,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
    n_draws: int = 2000,
    block: int = 21,
    seed: int = 421,
) -> dict:
    """Circular-block-shuffle placebo for the Alligator's Sharpe advantage over B&H.

    We keep the *exposure profile* of the real signal but destroy its *timing*: rotate the
    signal series by a random offset (circular block shuffle preserving autocorrelation
    structure) ``n_draws`` times, re-run the net backtest, and record each placebo's
    Sharpe-difference vs buy-and-hold. ``p`` = P[placebo Sharpe-diff ≥ observed] — the
    honest answer to "could a trend-agnostic reshuffle of the same in/out pattern have
    beaten B&H this much by luck?".
    """
    real = run_backtest(bars, signal, tbill_daily=tbill_daily,
                        cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
    obs = summary(real["r_strategy"])["sharpe"] - summary(real["r_bh"])["sharpe"]

    sig = signal.reindex(bars.index).fillna(0.0).to_numpy()
    n = len(sig)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_draws)
    n_blocks = max(1, n // block)
    for i in range(n_draws):
        # rotate by whole blocks to preserve block-level autocorrelation
        off = int(rng.integers(0, n_blocks)) * block
        shuffled = np.roll(sig, off)
        ssig = pd.Series(shuffled, index=bars.index, name="signal")
        bt = run_backtest(bars, ssig, tbill_daily=tbill_daily,
                          cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
        diffs[i] = summary(bt["r_strategy"])["sharpe"] - summary(bt["r_bh"])["sharpe"]
    p = float((diffs >= obs).mean())
    return {"obs_sharpe_diff": float(obs), "placebo_mean": float(diffs.mean()),
            "p_value": p, "draws": diffs}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
    mode: str = "long_flat",
    sma_n: int = 200,
    placebo: bool = False,
    n_draws: int = 2000,
    seed: int = 421,
) -> dict:
    """Run the full Alligator teardown: B&H vs Alligator vs SMA(200) benchmark.

    Returns a dict with ``bh``, ``alligator``, ``sma`` :func:`summary` dicts, the
    Sharpe-difference t-stats (Alligator vs B&H, Alligator vs SMA), the in-market /
    in-position fraction, the number of position changes, and (if ``placebo``) the
    block-permutation p-value of the Alligator's advantage over B&H.
    """
    sig_a = alligator_signal(bars, mode=mode)
    sig_s = sma_signal(bars, n=sma_n, mode=mode)

    bt_a = run_backtest(bars, sig_a, tbill_daily=tbill_daily,
                        cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
    bt_s = run_backtest(bars, sig_s, tbill_daily=tbill_daily,
                        cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)

    s_bh = summary(bt_a["r_bh"])
    s_a = summary(bt_a["r_strategy"])
    s_s = summary(bt_s["r_strategy"])

    out = {
        "bh": s_bh,
        "alligator": s_a,
        "sma": s_s,
        "t_allig_vs_bh": sharpe_diff_tstat(bt_a["r_strategy"], bt_a["r_bh"]),
        "t_allig_vs_sma": sharpe_diff_tstat(bt_a["r_strategy"], bt_s["r_strategy"]),
        "in_pos_frac": float((sig_a.abs() > 0).mean()),
        "n_changes": int(sig_a.diff().abs().fillna(sig_a.abs()).gt(0).sum()),
        "mode": mode,
    }
    if placebo:
        pl = block_permutation_pvalue(
            bars, sig_a, tbill_daily=tbill_daily, cost_bps=cost_bps,
            borrow_bps_ann=borrow_bps_ann, n_draws=n_draws, seed=seed)
        out["placebo_p"] = pl["p_value"]
        out["placebo_obs"] = pl["obs_sharpe_diff"]
    return out
