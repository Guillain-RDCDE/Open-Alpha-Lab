"""The Stochastic-RSI detector, the timing rule, and the honest arbiters — Study 428.

Tushar Chande & Stanley Kroll introduced the Stochastic RSI in *The New Technical Trader*
(1994). It applies the Stochastic-oscillator transform to the RSI series rather than to
price, so it ranks the current RSI within its own recent range:

    RSI(n)            = Wilder's RSI over n bars (Wilder's smoothing)
    StochRSI(t)       = (RSI_t - min_k RSI) / (max_k RSI - min_k RSI)        in [0, 1]
    %K                = 100 * SMA_smoothK(StochRSI)
    %D                = SMA_smoothD(%K)

The folk rule on equities: %K below 20 = "oversold" → be long; %K above 80 = "overbought"
→ be flat (long-flat) or short (long-short). The pitch is that the double-normalisation
makes it *more sensitive* than plain RSI and therefore *adds* signal. This module turns
that into a daily long/flat (and long/short) timing rule and races it, NET of costs, on an
excess-of-cash basis against (a) buy-and-hold and (b) the obvious simpler benchmarks —
plain RSI(14) and an SMA(50/200) crossover — so the "it is better" claim is actually tested.

No look-ahead: every indicator is computed on closes up to bar *t*; the position taken on
the strength of bar *t*'s close earns the return of bar *t+1* (one ``shift``, applied once
in :func:`positions_to_returns`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index over ``period`` bars.

    Uses Wilder's smoothing (an EMA with alpha = 1/period) of the up- and down-moves, the
    canonical RSI definition. Returns a Series indexed identically to ``close``; the first
    ``period`` bars are NaN.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no down-moves, RS -> inf, RSI -> 100.
    out = out.where(roll_down != 0.0, 100.0)
    out[roll_up.isna() | roll_down.isna()] = np.nan
    return out.rename("rsi")


def stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Chande & Kroll's Stochastic RSI as %K and %D (0..100).

    StochRSI normalises the RSI within its own trailing ``stoch_period``-bar range, then
    %K is an ``smooth_k``-bar SMA of that (scaled to 0..100) and %D an ``smooth_d``-bar SMA
    of %K. Returns a DataFrame with columns ``k`` and ``d`` indexed like ``close``.
    """
    r = rsi(close, period=rsi_period)
    lo = r.rolling(stoch_period, min_periods=stoch_period).min()
    hi = r.rolling(stoch_period, min_periods=stoch_period).max()
    rng = (hi - lo).replace(0.0, np.nan)
    raw = (r - lo) / rng  # 0..1; flat-RSI windows -> NaN (no information)
    k = (100.0 * raw).rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({"k": k, "d": d}, index=close.index)


def sma_cross_position(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long-flat position from an SMA(fast/slow) crossover (1 when fast >= slow, else 0)."""
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    pos = (f >= s).astype(float)
    pos[f.isna() | s.isna()] = np.nan
    return pos.rename("pos")


# ---------------------------------------------------------------------------
# Timing rules -> daily target position (0/1 long-flat, -1/0/1 long-short)
# ---------------------------------------------------------------------------
def stoch_rsi_position(
    close: pd.Series,
    lower: float = 20.0,
    upper: float = 80.0,
    long_short: bool = False,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.Series:
    """Stateful long-flat (or long-short) position from the Stochastic-RSI %K line.

    The folk rule, made stateful so it holds between triggers:

    - Go (or stay) **long** the bar %K crosses *up* through ``lower`` (oversold reset).
    - Go **flat** (long-flat) or **short** (long-short) the bar %K crosses *down*
      through ``upper`` (overbought reset).
    - Hold the last state otherwise.

    The position is the *target weight* known at the close of bar *t*; it is lagged one bar
    in :func:`positions_to_returns`, so it earns bar *t+1*'s return. Returns a float Series
    on the same index (NaN during indicator warm-up).
    """
    sr = stoch_rsi(close, rsi_period, stoch_period, smooth_k, smooth_d)
    k = sr["k"]
    prev = k.shift(1)
    buy = (k > lower) & (prev <= lower) & prev.notna()
    sell = (k < upper) & (prev >= upper) & prev.notna()

    flat_state = 0.0 if not long_short else -1.0
    pos = pd.Series(np.nan, index=close.index, dtype=float)
    state = np.nan
    valid = k.notna().to_numpy()
    buy_a = buy.to_numpy()
    sell_a = sell.to_numpy()
    out = np.full(len(close), np.nan)
    for i in range(len(close)):
        if not valid[i]:
            out[i] = np.nan
            continue
        if buy_a[i]:
            state = 1.0
        elif sell_a[i]:
            state = flat_state
        # before the first trigger, default to flat (cash) — no peeking
        out[i] = 0.0 if (state != state) else state  # NaN check
    pos[:] = out
    return pos.rename("pos")


def rsi_position(
    close: pd.Series,
    lower: float = 30.0,
    upper: float = 70.0,
    long_short: bool = False,
    period: int = 14,
) -> pd.Series:
    """The simpler benchmark: a stateful long-flat rule on plain RSI(14) 30/70 crossings."""
    r = rsi(close, period=period)
    prev = r.shift(1)
    buy = (r > lower) & (prev <= lower) & prev.notna()
    sell = (r < upper) & (prev >= upper) & prev.notna()
    flat_state = 0.0 if not long_short else -1.0
    out = np.full(len(close), np.nan)
    valid = r.notna().to_numpy()
    buy_a, sell_a = buy.to_numpy(), sell.to_numpy()
    state = np.nan
    for i in range(len(close)):
        if not valid[i]:
            continue
        if buy_a[i]:
            state = 1.0
        elif sell_a[i]:
            state = flat_state
        out[i] = 0.0 if (state != state) else state
    return pd.Series(out, index=close.index, name="pos")


# ---------------------------------------------------------------------------
# Position -> net daily returns (one lag, costs one-way x NAV, borrow on shorts)
# ---------------------------------------------------------------------------
def positions_to_returns(
    close: pd.Series,
    pos: pd.Series,
    cost_bps: float = 1.0,
    borrow_bps_yr: float = 50.0,
    rf_bps_yr: float = 0.0,
) -> pd.DataFrame:
    """Turn a daily target position into a net, excess-of-cash daily-return ledger.

    - **One execution lag.** The position known at the close of *t* (``pos``) earns the
      simple return of *t+1* — a single ``shift(1)``, applied here and nowhere else.
    - **Costs.** ``cost_bps`` one-way on the *NAV traded* — turnover = |Δposition|, charged
      once per unit of weight changed (so a 0 -> 1 entry costs ``cost_bps``; a 1 -> -1 flip
      costs ``2 * cost_bps``).
    - **Borrow.** When short, a daily financing cost of ``borrow_bps_yr / 252`` on the
      absolute short weight.
    - **Excess of cash.** Returns are reported excess of the cash rate ``rf_bps_yr`` so a
      part-time-in-cash strategy is compared *excess-vs-excess* against buy-and-hold.

    Returns a DataFrame with columns ``mkt_excess`` (buy-and-hold excess return), ``gross``,
    ``net`` and ``pos`` aligned on the tape index (warm-up rows dropped).
    """
    ret = close.pct_change()
    rf_daily = rf_bps_yr * 1e-4 / TRADING_DAYS
    mkt_excess = ret - rf_daily

    w = pos.shift(1)  # the one and only execution lag
    dpos = pos.fillna(0.0).diff().abs()  # turnover known at signal time
    cost = dpos.shift(1) * cost_bps * 1e-4  # cost hits the bar the new weight is in force
    borrow = (-w.clip(upper=0.0)) * (borrow_bps_yr * 1e-4 / TRADING_DAYS)

    gross = w * mkt_excess
    net = gross - cost - borrow

    out = pd.DataFrame(
        {"mkt_excess": mkt_excess, "gross": gross, "net": net, "pos": w}
    ).dropna(subset=["mkt_excess", "net"])
    return out


# ---------------------------------------------------------------------------
# Performance metrics + inference arbiters
# ---------------------------------------------------------------------------
def sharpe(daily: np.ndarray) -> float:
    """Annualised Sharpe of a daily excess-return series (NaN-safe)."""
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def hac_t(daily: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of a daily excess-return series."""
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wgt * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def max_drawdown(daily: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the cumulative excess curve (a negative fraction)."""
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def summarize(led: pd.DataFrame, col: str = "net", exposure_col: str = "pos") -> dict:
    """Headline annualised statistics for a return ledger column."""
    r = led[col].to_numpy(dtype=float)
    exp = led[exposure_col].to_numpy(dtype=float) if exposure_col in led else None
    n = np.isfinite(r).sum()
    return {
        "n_days": int(n),
        "ann_ret": float(np.nansum(r) / max(n, 1) * TRADING_DAYS),
        "sharpe": sharpe(r),
        "tstat": hac_t(r),
        "maxdd": max_drawdown(r),
        "time_in_mkt": float(np.nanmean(np.abs(exp))) if exp is not None else float("nan"),
    }


def permutation_p(
    daily: np.ndarray,
    n_perm: int = 2000,
    seed: int = 428,
) -> float:
    """Sign-flip permutation p-value for a one-sided positive-Sharpe null.

    Randomly flips the sign of each daily return (the null: the series is symmetric around
    zero, i.e. no directional edge) and counts how often the permuted mean meets or beats
    the observed mean. A small p means the observed positive drift is unlikely under no edge.
    """
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    obs = r.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, r.size))
    perm_means = (signs * r).mean(axis=1)
    return float((perm_means >= obs).mean())


def diff_sharpe_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 428,
) -> dict:
    """Block-bootstrap test of Sharpe(a) - Sharpe(b) on paired daily series.

    Resamples calendar blocks (length ~ sqrt(n)) from the *paired* (a, b) returns and
    recomputes the Sharpe gap, returning the point estimate, a 95% CI and the fraction of
    resamples in which a's Sharpe fails to beat b's. The pairing keeps the same market days
    on both legs, so the test isolates the *added* value of stacking the stochastic on RSI.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 10:
        return {"diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_a_not_better": float("nan")}
    point = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(seed)
    blk = max(int(np.sqrt(n)), 2)
    n_blocks = int(np.ceil(n / blk))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + blk) % n for s in starts])[:n]
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    return {
        "diff": float(point),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "frac_a_not_better": float((diffs <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    cost_bps: float = 1.0,
    long_short: bool = False,
    lower: float = 20.0,
    upper: float = 80.0,
) -> dict:
    """Run the full Stochastic-RSI teardown on one tape and return a results dict.

    Builds the StochRSI long-flat (or long-short) timing rule, the plain-RSI benchmark, the
    SMA(50/200) benchmark and buy-and-hold; nets costs; and computes annualised stats, the
    HAC t, a permutation p on the StochRSI net series, and the bootstrapped Sharpe gap vs
    each benchmark. All on an excess-of-cash basis.
    """
    close = bars["close"]

    srsi_pos = stoch_rsi_position(close, lower=lower, upper=upper, long_short=long_short)
    rsi_pos = rsi_position(close, long_short=long_short)
    sma_pos = sma_cross_position(close)

    srsi = positions_to_returns(close, srsi_pos, cost_bps=cost_bps)
    rsi_l = positions_to_returns(close, rsi_pos, cost_bps=cost_bps)
    sma_l = positions_to_returns(close, sma_pos, cost_bps=cost_bps)

    # Align all legs + buy-and-hold on a common index for fair excess-vs-excess races.
    common = srsi.index.intersection(rsi_l.index).intersection(sma_l.index)
    srsi, rsi_l, sma_l = srsi.loc[common], rsi_l.loc[common], sma_l.loc[common]
    bh = srsi["mkt_excess"]  # buy-and-hold excess of cash on the same days

    s_net = srsi["net"].to_numpy()
    s_gross = srsi["gross"].to_numpy()
    r_net = rsi_l["net"].to_numpy()
    sma_net = sma_l["net"].to_numpy()
    bh_np = bh.to_numpy()

    res = {
        "n_days": int(len(common)),
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "srsi_gross": summarize(srsi, "gross"),
        "srsi_net": summarize(srsi, "net"),
        "rsi_net": summarize(rsi_l, "net"),
        "sma_net": summarize(sma_l, "net"),
        "bh": {
            "ann_ret": float(np.nansum(bh_np) / max(len(bh_np), 1) * TRADING_DAYS),
            "sharpe": sharpe(bh_np),
            "tstat": hac_t(bh_np),
            "maxdd": max_drawdown(bh_np),
        },
        "perm_p_srsi_net": permutation_p(s_net),
        "perm_p_srsi_gross": permutation_p(s_gross),
        "vs_bh": diff_sharpe_bootstrap(s_net, bh_np),
        "vs_rsi": diff_sharpe_bootstrap(s_net, r_net),
        "vs_sma": diff_sharpe_bootstrap(s_net, sma_net),
    }
    return res
