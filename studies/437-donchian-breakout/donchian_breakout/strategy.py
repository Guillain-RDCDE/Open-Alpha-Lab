"""The Donchian-Breakout timing rule — and its honest arbiters. Study 437.

The folk recipe (the channel-breakout half of the Turtle system, stripped of the
position-sizing, pyramiding and exit machinery): **buy when today's close makes a new
N-day high; sell / go flat when it makes a new N-day low.** The classic window is
N = 20. Richard Donchian's channel and the Turtles' S1 system made this famous; the
question this study asks is whether the 20-day channel *on its own* — sans the rest of
the Turtle apparatus — holds up as a timing rule on a liquid index.

We implement it as a long/flat (and, where natural, long/short) daily timing rule,
raced against buy-and-hold and two controls:

1. **buy_and_hold** — always invested; the baseline.
2. **random_timing** — a coin that goes flat with the same in-market *frequency* as the
   breakout rule but on random days. Isolates *timing skill* from mere reduced exposure.
3. **sma_cross** — the obvious simpler benchmark (price vs its SMA(N)): is the Donchian
   channel actually *better* than a plain moving average, or is "it's better" untested?

Honest arbiters:

- **HAC (Newey-West) t-stat** on the mean daily return (``summary``) and on the daily
  return *difference* (``sharpe_diff_tstat``) — the inference-bar numbers.
- **A block-permutation placebo** (``permutation_pvalue``): shuffle the *sign* of the
  position in blocks (preserving the autocorrelation of the return tape), re-run, and
  ask how often a random in/out schedule with the same exposure beats the real rule's
  mean return. A breakout edge that is real should sit in the right tail.
- **Costs** charged one-way × NAV on every position *switch* (long/short legs counted
  both); shorts pay borrow.

No look-ahead: the channel is computed on closes up to *t-1* (``shift(1)``) so the
new-high test on day *t* uses only past bars; the signal is then lagged one more day so
the position is *entered at t+1* — one documented execution lag, applied once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Donchian channel + signals
# ---------------------------------------------------------------------------
def donchian_channel(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    """The Donchian channel: rolling N-day high/low of the *prior* N bars.

    Both bands are shifted by one bar so that the channel known at the close of day *t*
    uses only bars up to *t-1* — a new-high breakout on day *t* is then genuinely a
    close exceeding the highest of the previous N closes, with no peeking. Valid from
    bar ``n`` onward (first ``n`` rows are NaN).
    """
    upper = high.rolling(n, min_periods=n).max().shift(1)
    lower = low.rolling(n, min_periods=n).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower}, index=high.index)


def breakout_signal(bars: pd.DataFrame, n: int = 20, allow_short: bool = False) -> pd.Series:
    """Donchian timing signal, lagged one day for next-bar execution.

    A long is opened when the close makes a new N-day high (``close > upper``), and
    held until the close makes a new N-day low (``close < lower``), at which point the
    rule goes flat (``allow_short=False``) or short (``allow_short=True``). Between a
    high break and a low break the position is *carried* (the channel-breakout state
    machine), which is what makes Donchian a position rule rather than a per-bar filter.

    The raw state on day *t* is formed from the channel known at *t* (which itself only
    sees bars up to *t-1*); we then ``shift(1)`` so the position is entered at *t+1* —
    one execution lag, documented and applied once. Returns a {0,1} (or {-1,0,1}) series.
    """
    ch = donchian_channel(bars["high"], bars["low"], n=n)
    close = bars["close"]
    long_break = close > ch["upper"]
    short_break = close < ch["lower"]

    pos = np.zeros(len(close))
    state = 0
    for i in range(len(close)):
        if long_break.iat[i]:
            state = 1
        elif short_break.iat[i]:
            state = -1 if allow_short else 0
        pos[i] = state
    sig = pd.Series(pos, index=close.index, name="signal")
    # one execution lag: position formed at t is entered at t+1
    return sig.shift(1)


def sma_cross_signal(bars: pd.DataFrame, n: int = 20, allow_short: bool = False) -> pd.Series:
    """The obvious simpler benchmark: long when close > SMA(N), flat/short otherwise.

    Same window, same execution lag as :func:`breakout_signal`, so the race is apples
    to apples. This is the "is the channel actually *better* than a plain moving
    average?" control the brief asks for.
    """
    sma = bars["close"].rolling(n, min_periods=n).mean()
    above = bars["close"] > sma
    pos = np.where(above, 1.0, -1.0 if allow_short else 0.0)
    sig = pd.Series(pos, index=bars["close"].index, name="signal")
    return sig.shift(1)


def random_timing_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """Null control: same long/short/flat *frequencies* as ``signal``, random days.

    Reproduces the in-market (and, if shorting, the short) fraction of the real signal
    but scatters it across random days. If the breakout rule beats this, the *timing*
    (which days) adds value beyond the mere exposure profile.
    """
    s = signal.dropna()
    rng = np.random.default_rng(seed)
    # sample with replacement from the realised position distribution
    drawn = rng.choice(s.to_numpy(), size=len(signal), replace=True)
    out = pd.Series(drawn, index=signal.index, name="random_signal")
    out[signal.isna()] = np.nan
    return out


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 2.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Apply a {-1,0,1} position to daily log-returns; charge costs and borrow.

    Parameters
    ----------
    bars : OHLCV frame (uses ``close``).
    signal : a {-1,0,1} series aligned to ``bars``, already lagged for next-bar entry.
    cost_bps : one-way cost in bps, charged on |Δposition| × NAV at each switch (a
        flip 0→1 costs one leg; a flip -1→1 costs two). Default 2 bps (liquid ETF).
    borrow_bps_ann : annual borrow charged on the short leg only (per day it is short).

    Returns a frame with ``r_asset`` (buy-and-hold log-return), ``signal``,
    ``r_strategy`` (net), and ``r_bh`` (alias of ``r_asset`` for the fair race).
    """
    r_asset = np.log(bars["close"] / bars["close"].shift(1)).rename("r_asset")
    sig = signal.reindex(bars.index).fillna(0.0)

    switches = sig.diff().abs().fillna(0.0)
    cost = switches * (cost_bps * 1e-4)
    borrow = (sig < 0).astype(float) * (borrow_bps_ann * 1e-4 / TRADING_DAYS_PER_YEAR)

    r_strat = (sig * r_asset - cost - borrow).rename("r_strategy")
    out = pd.concat([r_asset, sig.rename("signal"), r_strat,
                     r_asset.rename("r_bh")], axis=1)
    return out.dropna()


# ---------------------------------------------------------------------------
# Performance summary + HAC inference
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of a daily return array (no quantlab dep)."""
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
    """Headline annualised stats + the HAC t-stat (the inference-bar number)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n == 0:
        return dict(n_days=0, cagr=float("nan"), sharpe=float("nan"),
                    vol_ann=float("nan"), max_drawdown=float("nan"),
                    mean_daily_bps=float("nan"), tstat=float("nan"))
    mu, std = r.mean(), r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1.0).min())
    return dict(
        n_days=int(n), cagr=cagr, sharpe=sharpe,
        vol_ann=float(std * np.sqrt(periods_per_year)),
        max_drawdown=dd, mean_daily_bps=float(mu * 1e4),
        tstat=_hac_tstat(r.to_numpy()),
    )


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """HAC t-stat on the daily return *difference* r1 − r2 (Jobson-Korkie style).

    If the mean of (r1 − r2) is significantly positive at |t| >= 2, arm 1 has a
    reliably higher mean (hence Sharpe, at equal vol) than arm 2 *on this tape*.
    """
    diff = (r1 - r2).dropna()
    return _hac_tstat(diff.to_numpy())


# ---------------------------------------------------------------------------
# Block-permutation placebo
# ---------------------------------------------------------------------------
def permutation_pvalue(bars: pd.DataFrame, signal: pd.Series, n_perm: int = 2000,
                       block: int = 20, cost_bps: float = 0.0, seed: int = 437) -> dict:
    """Block-shuffle the position schedule; how often does random beat the real rule?

    The real rule's mean daily (net) return is compared against a null built by
    shuffling the position series in contiguous **blocks** (length ``block``), which
    preserves the autocorrelation/clustering of *being positioned* while destroying the
    *alignment* with the return tape. The p-value is the fraction of permutations whose
    mean return is >= the real rule's. A genuine breakout edge sits in the right tail
    (small p); a rule that is just exposure earns a large p.
    """
    bt = run_backtest(bars, signal, cost_bps=cost_bps)
    real_mean = bt["r_strategy"].mean()
    r_asset = bt["r_asset"].to_numpy()
    pos = bt["signal"].to_numpy()
    n = len(pos)
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block))
    ge = 0
    for _ in range(n_perm):
        # break the position series into blocks and reorder them
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate(
            [pos[b * block:(b + 1) * block] for b in order]
        )[:n]
        m = float((shuffled * r_asset).mean())
        if m >= real_mean:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {"real_mean_bps": float(real_mean * 1e4), "p_value": float(p),
            "n_perm": int(n_perm), "block": int(block)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(bars: pd.DataFrame, n: int = 20, cost_bps: float = 2.0,
                   allow_short: bool = False, random_seed: int = 42) -> dict:
    """Run all arms (Donchian / SMA / random / buy-and-hold) and the arbiters.

    Returns a dict of ``summary`` dicts for each arm plus the head-to-head HAC t-stats
    (Donchian vs BH, Donchian vs random, Donchian vs SMA) and the in-market fraction.
    """
    sig = breakout_signal(bars, n=n, allow_short=allow_short)
    sig_sma = sma_cross_signal(bars, n=n, allow_short=allow_short)
    sig_rand = random_timing_signal(sig, seed=random_seed)

    bt = run_backtest(bars, sig, cost_bps=cost_bps)
    bt_sma = run_backtest(bars, sig_sma, cost_bps=cost_bps)
    bt_rand = run_backtest(bars, sig_rand, cost_bps=cost_bps)

    in_mkt = float((sig.dropna() != 0).mean())
    return {
        "bh": summary(bt["r_bh"]),
        "donchian": summary(bt["r_strategy"]),
        "sma": summary(bt_sma["r_strategy"]),
        "random": summary(bt_rand["r_strategy"]),
        "t_vs_bh": sharpe_diff_tstat(bt["r_strategy"], bt["r_bh"]),
        "t_vs_random": sharpe_diff_tstat(bt["r_strategy"], bt_rand["r_strategy"]),
        "t_vs_sma": sharpe_diff_tstat(bt["r_strategy"], bt_sma["r_strategy"]),
        "in_market_frac": in_mkt,
        "n_switches": int(sig.diff().abs().fillna(0).gt(0).sum()),
    }
