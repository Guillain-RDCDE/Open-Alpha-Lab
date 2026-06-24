"""The Awesome Oscillator timing rule, its MACD benchmark, and the honest arbiters.

Bill Williams' **Awesome Oscillator** (AO) is the spread between a fast and a slow simple
moving average of the bar *midpoint* (HL2 = (high + low) / 2):

    AO_t = SMA_5(HL2)_t - SMA_34(HL2)_t

It is pitched as a momentum gauge: AO > 0 (and rising) = bullish; AO < 0 = bearish.
The folk recipe we test is the cleanest long/flat reading: **hold the index when AO > 0,
move to cash (T-bill) when AO <= 0.** Re-evaluated daily. The hook: *does AO beat a plain
MACD?* — so we run the **exact same long/flat machine** on the classic MACD line
(EMA_12 - EMA_26 of the close) as the obvious simpler benchmark.

Three comparison arms share one backtest engine (:func:`run_backtest`):

1. **buy_and_hold** — always invested; the baseline everyone races against.
2. **ao_timing** — long when AO > 0, cash otherwise.
3. **macd_timing** — long when MACD line > 0, cash otherwise. The "is AO *better*?" test.

Plus a **random-timing** control (same in-market fraction on random days) to separate
genuine timing from mere exposure reduction.

No look-ahead: the AO/MACD value on day *t* uses only data up to *t*; the position is
entered at *t+1* (we proxy the next open by the next close — the standard daily-data
approximation for a slow timing rule). One `shift`, applied once, in :func:`*_signal`.

The inference bar: a REAL Signal stamp needs a HAC (Newey-West) *t* >= 2 on the **excess**
daily return *difference* (AO timing minus buy-and-hold, both excess-of-cash) on the real
tape — and, for the hook, AO must beat MACD on the same difference test. A label-shuffle
permutation placebo guards against the rule reading noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def hl2(bars: pd.DataFrame) -> pd.Series:
    """Bar midpoint price (high + low) / 2 — the Awesome Oscillator's input."""
    return ((bars["high"] + bars["low"]) / 2.0).rename("hl2")


def awesome_oscillator(bars: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Bill Williams' Awesome Oscillator: SMA_fast(HL2) - SMA_slow(HL2).

    Valid from bar ``slow`` onward (the first ``slow-1`` rows are NaN). The canonical
    Williams parameters are fast=5, slow=34 simple moving averages of the midpoint.
    """
    mid = hl2(bars)
    fast_ma = mid.rolling(fast, min_periods=fast).mean()
    slow_ma = mid.rolling(slow, min_periods=slow).mean()
    return (fast_ma - slow_ma).rename("ao")


def macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Classic MACD line: EMA_fast(close) - EMA_slow(close).

    The obvious simpler momentum benchmark. We use the MACD *line* crossing zero
    (the standard "MACD above/below zero = trend" reading), matching the AO's
    long/flat zero-line rule for an apples-to-apples race.
    """
    ema_f = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return (ema_f - ema_s).rename("macd")


# ---------------------------------------------------------------------------
# Signals — binary long/flat, lagged exactly once (no look-ahead)
# ---------------------------------------------------------------------------
def ao_signal(bars: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Daily {0,1} long/flat signal: 1 when AO > 0 [long], 0 otherwise [cash].

    The AO at the close of day *t* is acted on at *t+1* — one `shift`, applied here.
    """
    ao = awesome_oscillator(bars, fast=fast, slow=slow)
    return (ao > 0).astype(float).shift(1).rename("ao_signal")


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Daily {0,1} long/flat signal: 1 when MACD line > 0 [long], 0 otherwise [cash]."""
    m = macd_line(close, fast=fast, slow=slow)
    return (m > 0).astype(float).shift(1).rename("macd_signal")


def random_timing_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """Null control: same in-market *fraction* as ``signal`` but on random days.

    If the timing rule's edge were purely from reduced exposure, this matched-frequency
    coin would show the same advantage. If the rule beats it, the *timing* adds value.
    """
    frac = float(signal.dropna().mean())
    rng = np.random.default_rng(seed)
    n = len(signal)
    rand = pd.Series((rng.random(n) < frac).astype(float), index=signal.index,
                     name="random_signal")
    rand[signal.isna()] = np.nan
    return rand


# ---------------------------------------------------------------------------
# Backtest engine — apply a binary in/out signal, credit cash, charge switches
# ---------------------------------------------------------------------------
def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Apply a {0,1} long/flat signal to daily equity returns; credit cash when flat.

    Parameters
    ----------
    close : adjusted daily close (total-return when ``auto_adjust=True`` upstream).
    signal : {0,1} series aligned to ``close``, **already lagged** (signal at *t* acts on
        the return from *t* to *t+1*).
    tbill_daily : daily cash return (fraction) credited during flat periods, scalar or series.
    cost_bps : **one-way** transaction cost in bps, charged on each side every time the
        signal *changes* (one-way x NAV). Long-only, so no borrow.

    Returns a frame with ``r_equity, r_tbill, signal, r_strategy, r_bh`` (log returns),
    plus ``r_strat_excess`` and ``r_bh_excess`` (excess-of-cash, for fair Sharpe races).
    """
    r_eq = np.log(close / close.shift(1)).rename("r_equity")
    if isinstance(tbill_daily, pd.Series):
        r_tb = tbill_daily.reindex(close.index).fillna(0.0)
    else:
        r_tb = pd.Series(float(tbill_daily), index=close.index)
    r_tb.name = "r_tbill"

    sig = signal.reindex(close.index).fillna(0.0)
    switches = sig.diff().abs().fillna(0.0)
    cost = cost_bps * 1e-4

    r_strat = (sig * r_eq + (1.0 - sig) * r_tb - switches * cost).rename("r_strategy")
    out = pd.concat(
        [r_eq, r_tb, sig.rename("signal"), r_strat, r_eq.rename("r_bh")], axis=1
    ).dropna()
    out["r_strat_excess"] = out["r_strategy"] - out["r_tbill"]
    out["r_bh_excess"] = out["r_bh"] - out["r_tbill"]
    return out


# ---------------------------------------------------------------------------
# Performance summary + HAC inference
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``r`` (the inference-bar number)."""
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
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
    """Headline annualised statistics for a daily log-return series.

    Sharpe (of the supplied series — pass an *excess* series for an excess Sharpe),
    annualised vol, CAGR, max drawdown, mean daily bps, and the HAC t-stat on the mean.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))
    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = (float(cum.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and cum.iloc[-1] > 0 else float("nan"))
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


def diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC t-stat on the daily return *difference* r1 - r2 (the Sharpe-race test).

    A Jobson-Korkie style Newey-West test: if mean(r1 - r2) clears |t| >= 2, arm 1 has a
    reliably higher risk-adjusted return than arm 2 *on this tape*. Pass excess series.
    """
    return _hac_tstat((r1 - r2).dropna().to_numpy())


# ---------------------------------------------------------------------------
# Permutation placebo — shuffle the signal labels, keep the return sequence
# ---------------------------------------------------------------------------
def permutation_pvalue(
    close: pd.Series,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 0.0,
    n_draws: int = 2000,
    seed: int = 420,
) -> dict:
    """Label-shuffle placebo for the AO timing edge over buy-and-hold (excess).

    The observed statistic is the mean *excess* daily return difference
    (AO timing minus buy-and-hold). We then circularly *rotate* the signal a random
    number of days ``n_draws`` times (preserving the signal's own autocorrelation and
    in-market fraction while breaking its alignment to returns) and recompute the same
    statistic. The p-value is the fraction of rotations whose edge is >= the observed one.

    A rotation placebo (not an i.i.d. shuffle) respects the persistence of a slow timing
    signal, so it does not manufacture significance from the destruction of structure.
    """
    bt = run_backtest(close, signal, tbill_daily=tbill_daily, cost_bps=cost_bps)
    obs = float((bt["r_strat_excess"] - bt["r_bh_excess"]).mean())

    sig = signal.reindex(close.index).fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(close)
    null = np.empty(n_draws)
    for i in range(n_draws):
        k = int(rng.integers(1, n - 1))
        rolled = pd.Series(np.roll(sig, k), index=close.index)
        bt_i = run_backtest(close, rolled, tbill_daily=tbill_daily, cost_bps=cost_bps)
        null[i] = float((bt_i["r_strat_excess"] - bt_i["r_bh_excess"]).mean())
    pval = float((null >= obs).mean())
    return {"obs_edge_bps": obs * 1e4, "p_value": pval,
            "null_mean_bps": float(null.mean()) * 1e4, "n_draws": n_draws}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 2.0,
    ao_fast: int = 5,
    ao_slow: int = 34,
    macd_fast: int = 12,
    macd_slow: int = 26,
    random_seed: int = 42,
) -> dict:
    """Run the full race: buy-and-hold vs AO timing vs MACD timing vs random timing.

    Returns a dict of per-arm summaries (on **excess-of-cash** returns for fair Sharpe
    races), the in-market fractions, switch counts, and the decisive difference t-stats:
    AO vs BH, MACD vs BH, and AO vs MACD (the hook).
    """
    close = bars["close"]
    ao_sig = ao_signal(bars, fast=ao_fast, slow=ao_slow)
    macd_sig = macd_signal(close, fast=macd_fast, slow=macd_slow)
    rand_sig = random_timing_signal(ao_sig, seed=random_seed)

    bt_ao = run_backtest(close, ao_sig, tbill_daily=tbill_daily, cost_bps=cost_bps)
    bt_macd = run_backtest(close, macd_sig, tbill_daily=tbill_daily, cost_bps=cost_bps)
    bt_rand = run_backtest(close, rand_sig, tbill_daily=tbill_daily, cost_bps=cost_bps)

    # Align all arms to the AO backtest index for fair difference tests.
    idx = bt_ao.index
    bh_ex = bt_ao["r_bh_excess"]
    ao_ex = bt_ao["r_strat_excess"]
    macd_ex = bt_macd["r_strat_excess"].reindex(idx)
    rand_ex = bt_rand["r_strat_excess"].reindex(idx)

    out = {
        "bh": summary(bt_ao["r_bh"]),                # gross BH (price+div)
        "bh_excess": summary(bh_ex),                 # excess-of-cash BH
        "ao": summary(bt_ao["r_strategy"]),
        "ao_excess": summary(ao_ex),
        "macd": summary(bt_macd["r_strategy"]),
        "macd_excess": summary(macd_ex),
        "random_excess": summary(rand_ex),
        "ao_in_mkt": float(ao_sig.dropna().mean()),
        "macd_in_mkt": float(macd_sig.dropna().mean()),
        "ao_switches": int(ao_sig.diff().abs().fillna(0).sum()),
        "macd_switches": int(macd_sig.diff().abs().fillna(0).sum()),
        "t_ao_vs_bh": diff_tstat(ao_ex, bh_ex),
        "t_macd_vs_bh": diff_tstat(macd_ex, bh_ex),
        "t_ao_vs_macd": diff_tstat(ao_ex, macd_ex),
        "t_ao_vs_random": diff_tstat(ao_ex, rand_ex),
    }
    return out
