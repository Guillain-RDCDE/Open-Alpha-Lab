"""The MA-timing strategy and its honest controls — Study 210 (Crypto-Trend).

The folk recipe: hold Bitcoin when its price is **above** its 200-day simple moving
average; move entirely to cash (earning a T-bill proxy) when price falls **below**
the SMA. Re-evaluated every day. The rule is binary (fully in or fully out) — no
short selling. This is the Faber (2007) rule applied to crypto, and matches the
study design in Study 110 (Faber-Timing on SPY).

Three comparison arms:

1. **buy_and_hold** — always invested in BTC; the baseline.
2. **ma_timing** — the SMA(200) rule: invested when close > SMA(200), in cash otherwise.
3. **random_timing** — the null control: same in-market *frequency* as the SMA rule but on
   random days (matched reduced-exposure fraction). If the SMA rule beats this, the *timing*
   (which days to be out) is adding value beyond mere exposure reduction.

No look-ahead: the signal on day *t* uses only data up to *t*; positions are entered at
the *next* day's open, approximated by the next close (standard daily-data approximation
for long-horizon tactical studies).

The key distinction: the MA rule may cut BTC's brutal drawdowns (−80%+ historically) while
still not beating buy-and-hold on a risk-adjusted basis, or vice versa. These are measured
separately and reported honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365   # crypto: no weekends/holidays


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

    Returns a ``pd.Series`` of {0, 1} aligned to ``close.index``, lagged one day so that
    the first valid signal fires on day ``sma_n + 1`` (the day after the SMA warms up).
    """
    above = (close > sma(close, sma_n)).astype(int)
    # Lag one day: the signal formed at day t is acted on at day t+1.
    return above.shift(1).rename("signal")


def random_timing_signal(
    signal: pd.Series,
    seed: int = 0,
) -> pd.Series:
    """A null-control signal: same in-market *frequency* as ``signal``, but random days.

    This is the key control: if the MA rule's Sharpe advantage over buy-and-hold were
    purely from *reduced exposure* (running fewer days invested), the random control —
    with the same reduced exposure on random days — would show the same result. If the
    MA rule beats the random control, the *timing* (which days to be in) adds value.
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
    # Preserve the NaN mask of the original signal (warmup period).
    rand[signal.isna()] = np.nan
    return rand


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    tbill_daily: float | pd.Series = 0.0,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Apply a binary in/out signal to daily BTC returns, crediting cash during out periods.

    Parameters
    ----------
    close:
        Daily adjusted close series (BTC-USD total-return, i.e. the raw price).
    signal:
        A {0, 1} series aligned to ``close``. 1 = invested, 0 = in cash.
        Must already be lagged (signal at t acts on the return from t to t+1).
    tbill_daily:
        Daily T-bill return (in fraction, not bps), or a scalar flat rate
        (e.g. 0.04/365 for a 4% annual rate). Credited during cash periods.
    cost_bps:
        One-way transaction cost in bps, charged each time the signal *changes*.
        Default 10 bps one-way (20 bps round-trip) — crypto ETF/spot trading
        is cheaper than historical crypto exchange spreads but more expensive than
        equity ETF trading. Sensitivity tested in the cost sweep.

    Returns
    -------
    pd.DataFrame with columns:
        ``r_equity``     — daily log-return of BTC (close-to-close).
        ``r_tbill``      — daily T-bill log-return.
        ``signal``       — the applied signal (0/1).
        ``r_strategy``   — signal * r_equity + (1-signal) * r_tbill − switch_cost.
        ``r_bh``         — buy-and-hold BTC return (for the fair comparison).
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

    # Strategy return: in BTC when signal=1, in cash when signal=0, minus costs.
    r_strat = (sig * r_eq + (1.0 - sig) * r_tb - switches * cost_per_switch).rename("r_strategy")

    out = pd.concat([r_eq, r_tb, sig.rename("signal"), r_strat, r_eq.rename("r_bh")], axis=1)
    return out.dropna()


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def summary(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Headline annualised statistics for a daily return series.

    Returns Sharpe ratio, annualised vol, CAGR (geometric), maximum drawdown, and
    the HAC t-stat (Newey-West) for the mean return — the inference-bar number.
    Uses 365 days/year by default (crypto: no weekends).
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
    cost_bps: float = 10.0,
    random_seed: int = 42,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, dict]:
    """Run all three arms (buy-and-hold, MA timing, random timing) and return their summaries.

    The fair comparison adjusts for exposure: the MA rule spends some time in cash
    (earning T-bills) and some in BTC. The random-control arm matches the same
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
        "bh": summary(bh_ret, periods_per_year=periods_per_year),
        "timing": summary(bt_timing["r_strategy"], periods_per_year=periods_per_year),
        "random": summary(bt_random["r_strategy"], periods_per_year=periods_per_year),
        "in_market_frac": in_mkt,
    }


def sharpe_diff_tstat(
    r1: pd.Series,
    r2: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """HAC t-stat on the daily return *difference* r1 − r2 (tests whether Sharpes differ).

    This is the Jobson-Korkie / Ledoit-Wolf difference test in its simplest Newey-West
    form: if the mean of (r1 − r2) is significantly positive at |t| >= 2, arm 1 has a
    reliably higher risk-adjusted return than arm 2 *on this tape*.
    """
    diff = (r1 - r2).dropna()
    s = summary(diff, periods_per_year=periods_per_year)
    return s["tstat"]
