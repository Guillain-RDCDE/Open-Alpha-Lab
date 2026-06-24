"""The Triple-MA-Crossover rule and its honest controls — Study 438.

The folk recipe (a staple of YouTube / TradingView "this beats everything" videos): use
**three** moving averages — fast, medium, slow (the classic is 5 / 20 / 50, or sometimes
10 / 50 / 200). Go **long** only when they line up bullishly, ``fast > mid > slow``; go to
cash (or short) otherwise. The pitch is that the third MA is a "confirmation filter" that
keeps you out of whipsaws a simple two-MA cross would take, so three MAs *beat* two.

We test that claim head-on as a daily long/flat (and optionally long/short) timing rule and
**race it against the obvious simpler benchmarks**:

1. **buy_and_hold** — always invested; the baseline.
2. **dual_ma** — the two-MA cross (long when ``fast > slow``). The thing the triple stack
   claims to beat.
3. **triple_ma** — the ribbon stack: long when ``fast > mid > slow``.

The decisive questions:
- Does the triple stack's **NET Sharpe** beat buy-and-hold (excess-vs-excess), and does its
  mean return clear the HAC *t* ≥ 2 inference bar on the real tape?
- Does it beat the **dual_ma** benchmark? (If not, the third MA earns nothing — "it's better"
  is false.)
- Does it survive a **permutation/label-shuffle** placebo and realistic **costs**?

No look-ahead: the MA stack is read on the close of day *t*; the position then earns the
return of *t+1* (one ``shift``, applied once — see :func:`positions`). Costs are charged
one-way × NAV on every position change; a short leg pays borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average, requiring ``n`` observations (no partial windows)."""
    return close.rolling(n, min_periods=n).mean()


def ma_stack(close: pd.Series, fast: int, mid: int, slow: int) -> pd.DataFrame:
    """The three SMAs as a frame with columns ``fast``, ``mid``, ``slow``."""
    return pd.DataFrame(
        {"fast": sma(close, fast), "mid": sma(close, mid), "slow": sma(close, slow)},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Raw signals (read on close of t; NOT yet lagged)
# ---------------------------------------------------------------------------
def triple_signal(
    close: pd.Series, fast: int = 5, mid: int = 20, slow: int = 50,
    allow_short: bool = False,
) -> pd.Series:
    """Triple-MA stack target exposure (read on close of t, unlagged).

    +1 when ``fast > mid > slow`` (bullish ribbon). 0 otherwise — or, if ``allow_short``,
    -1 when ``fast < mid < slow`` (bearish ribbon) and 0 in the mush in between.
    """
    s = ma_stack(close, fast, mid, slow)
    bull = (s["fast"] > s["mid"]) & (s["mid"] > s["slow"])
    pos = bull.astype(float)
    if allow_short:
        bear = (s["fast"] < s["mid"]) & (s["mid"] < s["slow"])
        pos = pos - bear.astype(float)
    # Mask the warm-up where the slow MA is undefined.
    pos[s["slow"].isna()] = np.nan
    return pos.rename("triple")


def dual_signal(
    close: pd.Series, fast: int = 5, slow: int = 50, allow_short: bool = False,
) -> pd.Series:
    """The simpler two-MA cross target exposure (read on close of t, unlagged).

    +1 when ``fast > slow``; 0 (or -1 if ``allow_short``) otherwise. This is the benchmark
    the triple stack claims to beat — same fast/slow legs, no middle "confirmation" MA.
    """
    f, sl = sma(close, fast), sma(close, slow)
    pos = (f > sl).astype(float)
    if allow_short:
        pos = pos - (f < sl).astype(float)
    pos[sl.isna()] = np.nan
    return pos.rename("dual")


# ---------------------------------------------------------------------------
# Position lag (the single documented execution lag)
# ---------------------------------------------------------------------------
def positions(signal: pd.Series) -> pd.Series:
    """Lag the unlagged target exposure by one day: signal at close of t -> held over t+1.

    This is the *one* execution lag in the whole study (one ``shift``). A position formed
    on the close of day *t* earns the close-to-close return of day *t+1*.
    """
    return signal.shift(1).rename("pos")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def backtest(
    close: pd.Series,
    pos: pd.Series,
    rf_annual: float = 0.0,
    cost_bps: float = 5.0,
    borrow_annual: float = 0.005,
) -> pd.DataFrame:
    """Apply a lagged exposure series to daily log-returns; credit cash, charge costs/borrow.

    Parameters
    ----------
    close : adjusted (total-return) daily close.
    pos   : a {-1, 0, +1} exposure series, **already lagged** (use :func:`positions`).
    rf_annual : flat risk-free / cash yield credited on the un-invested fraction
        (``1 - |pos|``) when long-only, and the financing reference for the book.
    cost_bps : one-way transaction cost in bps, charged on ``|Δpos| × NAV`` each day the
        exposure changes (one-way × NAV; a flip from +1 to -1 pays two units).
    borrow_annual : borrow fee charged on the short leg (the negative part of ``pos``).

    Returns a frame with ``r_asset`` (the underlying), ``pos``, ``r_strat`` (net), and
    ``r_bh`` (buy-and-hold = ``r_asset``), all as daily log-returns.
    """
    r_asset = np.log(close / close.shift(1)).rename("r_asset")
    p = pos.reindex(close.index)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    borrow_daily = borrow_annual / TRADING_DAYS_PER_YEAR

    # Cash leg: the un-invested fraction earns the risk-free rate (long-only: 1-pos in cash).
    cash_frac = (1.0 - p.clip(lower=0.0)).clip(lower=0.0)
    # Costs: |Δ exposure| × NAV × one-way bps.
    dpos = p.diff().abs().fillna(0.0)
    cost = dpos * cost_bps * 1e-4
    # Borrow on the short leg only.
    short_frac = (-p).clip(lower=0.0)
    borrow = short_frac * borrow_daily

    r_strat = (p * r_asset + cash_frac * rf_daily - cost - borrow).rename("r_strat")
    out = pd.concat(
        [r_asset, p.rename("pos"), r_strat, r_asset.rename("r_bh")], axis=1
    )
    return out.dropna()


# ---------------------------------------------------------------------------
# Performance summary (with HAC t on the EXCESS daily return)
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``r`` (auto lag = floor(4(n/100)^(2/9)))."""
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


def summary(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Annualised stats for a daily *net* return series, plus a HAC t on the EXCESS mean.

    The Sharpe and the HAC *t* are computed on **excess-of-cash** returns (subtract the
    daily risk-free rate) so the timing-vs-buy-and-hold race is excess-vs-excess. The
    *t* is the inference-bar number: it tests whether the strategy's excess mean return
    is distinguishable from zero.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    rf_daily = rf_annual / periods_per_year
    ex = r - rf_daily  # excess of cash

    mu = ex.mean()
    std = ex.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = (
        float(cum.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and cum.iloc[-1] > 0 else float("nan")
    )
    dd = float((cum / cum.cummax() - 1.0).min())

    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,                       # excess-of-cash, annualised
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(r.mean() * 1e4),
        "tstat": _hac_tstat(ex.to_numpy()),     # HAC t on the EXCESS mean
    }


def sharpe_diff_tstat(
    r1: pd.Series, r2: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """HAC *t* on the daily return difference ``r1 - r2`` (Jobson-Korkie, NW form).

    If the mean of ``r1 - r2`` is significantly positive at |t| ≥ 2, arm 1 has a reliably
    higher risk-adjusted return than arm 2 *on this tape*. Cash drops out of the difference,
    so this is already an excess-vs-excess comparison.
    """
    diff = (r1 - r2).dropna()
    return _hac_tstat(diff.to_numpy())


# ---------------------------------------------------------------------------
# Permutation placebo
# ---------------------------------------------------------------------------
def permutation_pvalue(
    close: pd.Series,
    fast: int = 5, mid: int = 20, slow: int = 50,
    rf_annual: float = 0.0, cost_bps: float = 5.0,
    n_draws: int = 2000, seed: int = 438,
) -> dict:
    """Block-permutation null for the triple-MA *net excess* Sharpe.

    The observed Sharpe is compared to a null built by re-running the *same* timing rule on
    **circular-block-shuffled** returns (block ≈ 21 days, to preserve short-horizon
    autocorrelation while destroying the trend persistence the rule needs). If the rule has
    no genuine trend edge, the observed Sharpe sits inside this cloud.

    Returns ``{obs, p_value, draws}`` where ``p`` = Pr[shuffled Sharpe ≥ observed].
    """
    rng = np.random.default_rng(seed)
    r_asset = np.log(close / close.shift(1)).dropna()
    idx = r_asset.index
    arr = r_asset.to_numpy()
    n = arr.size
    block = 21

    # Observed
    pos = positions(triple_signal(close, fast, mid, slow))
    bt = backtest(close, pos, rf_annual=rf_annual, cost_bps=cost_bps)
    obs = summary(bt["r_strat"], rf_annual=rf_annual)["sharpe"]

    draws = np.empty(n_draws)
    base = 100.0
    for d in range(n_draws):
        # circular block bootstrap of the return series, rebuild a price path, re-run rule
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        chunks = [np.take(arr, range(s, s + block), mode="wrap") for s in starts]
        shuffled = np.concatenate(chunks)[:n]
        sp = pd.Series(base * np.exp(np.cumsum(shuffled)), index=idx)
        p2 = positions(triple_signal(sp, fast, mid, slow))
        b2 = backtest(sp, p2, rf_annual=rf_annual, cost_bps=cost_bps)
        draws[d] = summary(b2["r_strat"], rf_annual=rf_annual)["sharpe"]

    draws = draws[np.isfinite(draws)]
    p = float((draws >= obs).mean()) if draws.size else float("nan")
    return {"obs": float(obs), "p_value": p, "draws": draws}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    close: pd.Series,
    fast: int = 5, mid: int = 20, slow: int = 50,
    rf_annual: float = 0.0, cost_bps: float = 5.0,
    borrow_annual: float = 0.005, allow_short: bool = False,
) -> dict:
    """Run the three arms (buy-and-hold, dual-MA, triple-MA) and the head-to-head tests.

    Returns a dict with per-arm ``summary`` dicts plus:
      - ``in_market`` fractions for each timing rule,
      - ``t_triple_vs_bh`` / ``t_triple_vs_dual`` HAC Sharpe-difference t-stats,
      - the net mean and HAC *t* of each timing rule (the inference-bar numbers),
      - switch counts (turnover).
    """
    tri_sig = triple_signal(close, fast, mid, slow, allow_short=allow_short)
    dua_sig = dual_signal(close, fast, slow, allow_short=allow_short)
    tri_pos = positions(tri_sig)
    dua_pos = positions(dua_sig)

    bt_tri = backtest(close, tri_pos, rf_annual=rf_annual, cost_bps=cost_bps,
                      borrow_annual=borrow_annual)
    bt_dua = backtest(close, dua_pos, rf_annual=rf_annual, cost_bps=cost_bps,
                      borrow_annual=borrow_annual)

    bh = summary(bt_tri["r_bh"], rf_annual=rf_annual)
    tri = summary(bt_tri["r_strat"], rf_annual=rf_annual)
    dua = summary(bt_dua["r_strat"], rf_annual=rf_annual)

    return {
        "bh": bh,
        "triple": tri,
        "dual": dua,
        "in_market_triple": float(tri_pos.reindex(close.index).dropna().abs().mean()),
        "in_market_dual": float(dua_pos.reindex(close.index).dropna().abs().mean()),
        "switches_triple": int(tri_pos.diff().abs().fillna(0).sum()),
        "switches_dual": int(dua_pos.diff().abs().fillna(0).sum()),
        "t_triple_vs_bh": sharpe_diff_tstat(bt_tri["r_strat"], bt_tri["r_bh"]),
        "t_triple_vs_dual": sharpe_diff_tstat(bt_tri["r_strat"], bt_dua["r_strat"]),
    }
