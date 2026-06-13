"""The Faber (2007) 10-month / 200-day SMA timing rule — and its honest controls.

The folk recipe: hold the equity index (SPY / ^GSPC) when its price is **above** its
10-month (200-day) SMA; move entirely to cash (earning the T-bill rate) when price falls
**below** the SMA. Re-evaluated once per month or day. The rule is binary (fully in or
fully out) — no short selling.

This module runs three comparison arms:

1. **buy_and_hold** — always invested; the baseline everyone compares to.
2. **faber_timing** — the SMA rule: invested when close > SMA(n), in T-bills otherwise.
3. **random_timing** — the null control: a coin that goes to cash with the same *frequency*
   as the SMA rule but on random days (matched in-market fraction, different days). If the
   SMA rule beats this, the *timing* (not just the reduced-exposure) is working.

No look-ahead: the signal on day *t* (or end-of-month *m*) uses only data up to *t* (or
*m*); positions are entered at the *next* bar's open (approximated by the next close in
our daily data, which is the standard approximation for long-horizon tactial studies).

The key test: does the SMA rule improve **Sharpe ratio** vs buy-and-hold on the real tape,
and does it beat the random-timing control? Separately: does it reduce max drawdown
significantly? These are two different things — a rule can dramatically reduce drawdown
(by missing the crash) while having the same or lower Sharpe (by also missing the
recovery), and the brief specifically asks us to call this out honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average, minimum ``n`` observations required (no partial windows)."""
    return close.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Signal generation — daily (200-day SMA)
# ---------------------------------------------------------------------------
def timing_signal(close: pd.Series, sma_n: int = 200) -> pd.Series:
    """Daily binary signal: +1 when close > SMA(n) [stay invested], 0 when below [go to cash].

    The signal on day *t* is formed using only data up to *t*; positions are then
    entered at the *next* day's open. We proxy the next open by the next close (the
    standard daily-data approximation for tactical rules).

    Returns a ``pd.Series`` of {0, 1} aligned to ``close.index``, lagged one day.
    """
    above = (close > sma(close, sma_n)).astype(int)
    # Lag one day: the signal formed at day t is acted on at day t+1.
    return above.shift(1).rename("signal")


def timing_signal_monthly(close: pd.Series, sma_n: int = 10) -> pd.Series:
    """Monthly binary signal: +1 when end-of-month close > SMA(n months) [original Faber].

    Uses a 10-month SMA on end-of-month closes, the original Faber (2007) specification.
    Returns a monthly signal lagged one month. Caller should resample daily close to monthly
    end-of-month first (use ``data.resample_monthly``).
    """
    above = (close > sma(close, sma_n)).astype(int)
    return above.shift(1).rename("signal_monthly")


def random_timing_signal(
    signal: pd.Series,
    seed: int = 0,
) -> pd.Series:
    """A null-control signal: same in-market *frequency* as ``signal``, but random days.

    This is the key control: if the SMA rule's Sharpe advantage over buy-and-hold
    were purely from *reduced exposure* (running fewer days invested), the random
    control — with the same reduced exposure on random days — would show the same
    Sharpe improvement. If the SMA rule beats the random control, the *timing* (which
    days to be in) adds value on top of the simple exposure reduction.
    """
    in_market_frac = float(signal.dropna().mean())
    rng = np.random.default_rng(seed)
    n = len(signal)
    rand = pd.Series(
        rng.random(n) < in_market_frac,
        index=signal.index,
        name="random_signal",
        dtype=int,
    )
    # Preserve the NaN mask of the original signal (warmup period)
    rand[signal.isna()] = np.nan
    return rand


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Apply a binary in/out signal to daily equity returns, crediting cash during out periods.

    Parameters
    ----------
    close:
        Adjusted daily close series (e.g. SPY total-return).
    signal:
        A {0, 1} series aligned to ``close``. 1 = invested, 0 = in cash.
        Must already be lagged (signal at t acts on the return from t to t+1).
    tbill_daily:
        Daily T-bill return (in fraction, not bps), or a scalar flat rate
        (e.g. 0.04/252 for a 4% annual rate). This is credited during cash periods.
    cost_bps:
        One-way transaction cost in bps, charged each time the signal *changes*
        (a switch from equity to cash or back). Default 5 bps one-way (10 bps
        round-trip) — on the generous side for ETF trading but conservative for
        an institutional book with market impact.

    Returns
    -------
    pd.DataFrame with columns:
        ``r_equity``     — daily log-return of the equity leg (close-to-close).
        ``r_tbill``      — daily T-bill log-return.
        ``signal``       — the applied signal (0/1).
        ``r_strategy``   — signal * r_equity + (1-signal) * r_tbill − switch_cost.
        ``r_bh``         — buy-and-hold equity return (for the fair comparison).
    """
    # Daily log returns
    r_eq = np.log(close / close.shift(1)).rename("r_equity")
    # Daily T-bill rate
    if isinstance(tbill_daily, pd.Series):
        r_tb = tbill_daily.reindex(close.index).fillna(0.0)
    else:
        r_tb = pd.Series(float(tbill_daily), index=close.index)
    r_tb.name = "r_tbill"

    sig = signal.reindex(close.index).fillna(0.0)

    # Switching cost: charged on each side when the signal flips.
    switches = sig.diff().abs().fillna(0.0)
    cost_per_switch = cost_bps * 1e-4

    # Strategy return: in equity when signal=1, in cash when signal=0, minus costs
    r_strat = (sig * r_eq + (1.0 - sig) * r_tb - switches * cost_per_switch).rename("r_strategy")

    out = pd.concat([r_eq, r_tb, sig.rename("signal"), r_strat, r_eq.rename("r_bh")], axis=1)
    return out.dropna()


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised statistics for a daily return series.

    Returns Sharpe ratio, annualised vol, CAGR (geometric), maximum drawdown, and
    the HAC t-stat (Newey-West) for the mean return — the inference-bar number.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))

    # CAGR from cumulative log return
    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")

    # Max drawdown (using price level from log returns)
    eq = cum
    dd = (eq / eq.cummax() - 1.0).min()

    # Newey-West HAC t-stat on the mean daily return
    e = r.to_numpy() - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    # In-market fraction (only meaningful for timed strategies)
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": float(dd),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": tstat,
    }


def compare_strategies(
    close: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    sma_n: int = 200,
    cost_bps: float = 5.0,
    random_seed: int = 42,
) -> dict[str, dict]:
    """Run all three arms (buy-and-hold, SMA timing, random timing) and return their summaries.

    The fair comparison adjusts for exposure: the SMA rule spends some time in cash
    (earning T-bills) and some in equity. The random-control arm matches the same
    fraction of time in-market on random days, so the test isolates *timing* skill.

    Returns a dict with keys ``'bh'``, ``'timing'``, ``'random'``, each a ``summary()``
    dict, plus an ``'in_market_frac'`` key.
    """
    sig = timing_signal(close, sma_n=sma_n)
    in_mkt = float(sig.dropna().mean())
    rand_sig = random_timing_signal(sig, seed=random_seed)

    bh_ret = np.log(close / close.shift(1)).dropna()

    bt_timing = run_backtest(close, sig, tbill_daily=tbill_daily, cost_bps=cost_bps)
    bt_random = run_backtest(close, rand_sig, tbill_daily=tbill_daily, cost_bps=cost_bps)

    return {
        "bh": summary(bh_ret),
        "timing": summary(bt_timing["r_strategy"]),
        "random": summary(bt_random["r_strategy"]),
        "in_market_frac": in_mkt,
    }


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """HAC t-stat on the daily return *difference* r1 − r2 (tests whether Sharpes differ).

    This is the Jobson-Korkie / Ledoit-Wolf difference test in its simplest Newey-West
    form: if the mean of (r1 − r2) is significantly positive at |t| >= 2, arm 1 has a
    reliably higher risk-adjusted return than arm 2 *on this tape*.
    """
    diff = (r1 - r2).dropna()
    s = summary(diff, periods_per_year=periods_per_year)
    return s["tstat"]
