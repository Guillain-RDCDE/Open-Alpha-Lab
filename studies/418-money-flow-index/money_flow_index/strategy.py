"""The indicators, the timing rule, and the honest arbiters — Study 418 (MFI).

The **Money Flow Index (MFI)** is the RSI's volume-weighted cousin.  Where the RSI
sums up- vs down-day *price* changes, the MFI sums up- vs down-day *money flow* =
typical-price x volume.  The folk claim: because it weights moves by volume, the MFI
is a "better RSI" — sharper overbought/oversold signals, fewer fakeouts.

We test that claim by turning *both* indicators into the **same** long/flat timing
rule on SPY and racing them — and racing both against buy-and-hold:

    Indicator <= buy_thr  ->  go long (oversold; the contrarian buy)
    Indicator >= sell_thr ->  go flat (overbought; step aside)
    in between            ->  hold the previous state

Execution discipline (one lag, documented once): the indicator is computed on closes
up to and including day *t*; the position for day *t* is decided at *t*'s close and
earns the return of day *t+1* (a single ``shift(1)``).  When flat, the book earns the
risk-free rate (cash).  Costs are charged one-way x NAV on every change of position.

Arbiters:
- one-sample / HAC (Newey-West) *t* on the strategy's daily excess-of-cash returns,
- a **block permutation placebo** that shuffles the position *blocks* against the
  returns (destroys timing, keeps the marginal exposure) — the honest null for "is the
  timing doing anything?",
- a **cost sweep** (one-way bps x NAV),
- the head-to-head **MFI vs RSI vs buy-and-hold** NET excess Sharpe race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI on closes (0..100), valid from bar ``window`` onward."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder's smoothing == EMA with alpha = 1/window.
    roll_up = up.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    roll_dn = down.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[roll_dn == 0.0] = 100.0
    return out.rename("rsi")


def money_flow_index(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Classic Money Flow Index (0..100) — the volume-weighted RSI.

    Typical price ``TP = (high + low + close) / 3``; raw money flow ``MF = TP * volume``.
    A day is "positive" if TP rose vs the prior day, "negative" if it fell.  The money
    flow ratio over the trailing ``window`` is ``sum(positive MF) / sum(negative MF)``;
    ``MFI = 100 - 100 / (1 + ratio)``.  Valid from bar ``window`` onward.
    """
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    mf = tp * bars["volume"].astype(float)
    chg = tp.diff()
    pos_mf = mf.where(chg > 0, 0.0)
    neg_mf = mf.where(chg < 0, 0.0)
    pos_sum = pos_mf.rolling(window, min_periods=window).sum()
    neg_sum = neg_mf.rolling(window, min_periods=window).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    out[neg_sum == 0.0] = 100.0
    return out.rename("mfi")


# ---------------------------------------------------------------------------
# Timing rule: indicator -> long/flat position (no look-ahead)
# ---------------------------------------------------------------------------
def positions_from_indicator(
    indicator: pd.Series,
    buy_thr: float = 30.0,
    sell_thr: float = 70.0,
) -> pd.Series:
    """Long/flat position series from an oscillator (RSI or MFI).

    State machine on the indicator value:
      <= buy_thr  -> long (1)
      >= sell_thr -> flat (0)
      otherwise   -> carry the previous state.

    The position on day *t* uses only the indicator value at *t* (a close-formed
    decision); ``backtest`` applies the single execution shift so it earns *t+1*'s
    return.  Starts flat until the first signal.
    """
    ind = indicator.to_numpy()
    pos = np.zeros(len(ind))
    state = 0
    for i, v in enumerate(ind):
        if np.isnan(v):
            pos[i] = 0
            state = 0
            continue
        if v <= buy_thr:
            state = 1
        elif v >= sell_thr:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=indicator.index, name="pos")


# ---------------------------------------------------------------------------
# Backtest engine — daily long/flat, excess-of-cash, one lag, one-way costs
# ---------------------------------------------------------------------------
def backtest(
    bars: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 2.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Run a daily long/flat book and return per-day net excess-of-cash returns.

    - ``position`` is the close-of-*t* decision; it earns the asset return of *t+1*
      (a single ``shift(1)``) — the one documented execution lag.
    - When long, the book earns the asset's daily return; when flat it earns the daily
      risk-free rate ``rf_daily`` (cash).  We report **excess-of-cash** returns
      (asset_ret - rf when long, 0 when flat) so the MFI/RSI/buy-hold race is
      excess-vs-excess, as the house rules require.
    - ``cost_bps`` is charged **one-way x NAV** on every change in position
      (|pos_t - pos_{t-1}| * cost), subtracted from that day's return.

    Columns: ``asset_ret`` (asset daily return), ``pos`` (the lagged, effective
    position), ``gross_excess`` (excess return before costs), ``cost`` (cost drag),
    ``net_excess`` (excess return after costs).
    """
    close = bars["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0

    # Effective position earning today's return = yesterday's close-decision.
    eff = position.shift(1).fillna(0.0)
    # Excess-of-cash return of the book: long -> asset_ret - rf; flat -> 0.
    gross_excess = eff * (asset_ret - rf_daily)
    # one-way turnover x NAV, charged on the day the position changes.
    turn = eff.diff().abs().fillna(eff.abs())
    cost = turn * (cost_bps * 1e-4)
    net_excess = gross_excess - cost

    return pd.DataFrame(
        {
            "asset_ret": asset_ret,
            "pos": eff,
            "gross_excess": gross_excess,
            "cost": cost,
            "net_excess": net_excess,
        }
    )


def buy_hold_excess(bars: pd.DataFrame, rf_annual: float = 0.0) -> pd.Series:
    """Daily excess-of-cash return of always-long buy-and-hold."""
    asset_ret = bars["close"].astype(float).pct_change().fillna(0.0)
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    return (asset_ret - rf_daily).rename("bh_excess")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(x: pd.Series | np.ndarray) -> float:
    """Newey-West (HAC) t-stat that the mean of ``x`` differs from zero."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
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


def sharpe(x: pd.Series | np.ndarray, ann: float = TRADING_DAYS) -> float:
    """Annualised Sharpe of a daily excess-return series."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann))


def ann_return(x: pd.Series | np.ndarray, ann: float = TRADING_DAYS) -> float:
    """Annualised mean of a daily return series (arithmetic x trading days)."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    return float(r.mean() * ann)


def max_drawdown(daily_ret: pd.Series | np.ndarray) -> float:
    """Max drawdown of the cumulative (compounded) book, as a negative fraction."""
    r = np.asarray(daily_ret, dtype=float)
    r = np.nan_to_num(r)
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    return float(dd.min()) if dd.size else float("nan")


def summarize_book(net_excess: pd.Series, position: pd.Series | None = None,
                   n_years: float | None = None) -> dict:
    """Headline stats for a daily excess-return book: Sharpe, ann return, HAC t, dd."""
    r = pd.Series(net_excess).dropna()
    out = {
        "sharpe": sharpe(r),
        "ann_return": ann_return(r),
        "hac_t": hac_tstat(r),
        "max_dd": max_drawdown(r),
        "n_days": int(r.size),
    }
    if position is not None:
        out["exposure"] = float(pd.Series(position).mean())
    return out


# ---------------------------------------------------------------------------
# Block permutation placebo — is the *timing* doing anything?
# ---------------------------------------------------------------------------
def block_permutation_pvalue(
    asset_excess: pd.Series,
    position: pd.Series,
    n_iter: int = 2000,
    block: int = 21,
    seed: int = 418,
) -> dict:
    """Placebo: shuffle the position in contiguous blocks vs the realised returns.

    The position series is cut into ``block``-day chunks and the chunks are randomly
    re-ordered, preserving the *fraction of time long* and the autocorrelation within a
    block, but **destroying the alignment** between the signal and the next-day return.
    If the real strategy's mean excess return sits in the bulk of this shuffled
    distribution, the timing carries no information.

    Returns the observed mean, the placebo mean (avg), and a two-sided-ish one-tailed
    p-value = P(placebo mean >= observed mean).
    """
    rng = np.random.default_rng(seed)
    ae = pd.Series(asset_excess).to_numpy(dtype=float)
    pos = pd.Series(position).shift(1).fillna(0.0).to_numpy(dtype=float)
    n = min(len(ae), len(pos))
    ae, pos = ae[:n], pos[:n]
    obs = float(np.nanmean(pos * ae))

    n_blocks = int(np.ceil(n / block))
    ge = 0
    placebo_means = np.empty(n_iter)
    for it in range(n_iter):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate(
            [pos[b * block:(b + 1) * block] for b in order]
        )[:n]
        m = float(np.nanmean(shuffled * ae))
        placebo_means[it] = m
        if m >= obs:
            ge += 1
    pval = (ge + 1) / (n_iter + 1)
    return {
        "observed_mean": obs,
        "placebo_mean": float(placebo_means.mean()),
        "p_value": float(pval),
        "n_iter": int(n_iter),
        "block": int(block),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    window: int = 14,
    buy_thr: float = 30.0,
    sell_thr: float = 70.0,
    cost_bps: float = 2.0,
    rf_annual: float = 0.0,
    perm_iter: int = 2000,
    perm_seed: int = 418,
) -> dict:
    """End-to-end: build MFI & RSI, run both as long/flat books, race vs buy-hold.

    Returns a nested dict with one block per arm (``mfi``, ``rsi``, ``buy_hold``),
    the head-to-head delta (MFI net excess Sharpe minus RSI), and the block-permutation
    placebo p-value for each indicator arm.
    """
    mfi = money_flow_index(bars, window=window)
    rsi_ = rsi(bars["close"], window=window)

    pos_mfi = positions_from_indicator(mfi, buy_thr, sell_thr)
    pos_rsi = positions_from_indicator(rsi_, buy_thr, sell_thr)

    bt_mfi = backtest(bars, pos_mfi, cost_bps=cost_bps, rf_annual=rf_annual)
    bt_rsi = backtest(bars, pos_rsi, cost_bps=cost_bps, rf_annual=rf_annual)
    bh = buy_hold_excess(bars, rf_annual=rf_annual)

    asset_excess = (bars["close"].astype(float).pct_change().fillna(0.0)
                    - ((1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0))

    res = {
        "mfi": {
            "net": summarize_book(bt_mfi["net_excess"], bt_mfi["pos"]),
            "gross": summarize_book(bt_mfi["gross_excess"], bt_mfi["pos"]),
            "perm": block_permutation_pvalue(asset_excess, pos_mfi,
                                             n_iter=perm_iter, seed=perm_seed),
        },
        "rsi": {
            "net": summarize_book(bt_rsi["net_excess"], bt_rsi["pos"]),
            "gross": summarize_book(bt_rsi["gross_excess"], bt_rsi["pos"]),
            "perm": block_permutation_pvalue(asset_excess, pos_rsi,
                                             n_iter=perm_iter, seed=perm_seed + 1),
        },
        "buy_hold": {"net": summarize_book(bh)},
        "params": {
            "window": window, "buy_thr": buy_thr, "sell_thr": sell_thr,
            "cost_bps": cost_bps, "rf_annual": rf_annual,
        },
    }
    res["delta_sharpe_mfi_minus_rsi"] = (
        res["mfi"]["net"]["sharpe"] - res["rsi"]["net"]["sharpe"]
    )
    res["delta_sharpe_mfi_minus_bh"] = (
        res["mfi"]["net"]["sharpe"] - res["buy_hold"]["net"]["sharpe"]
    )
    return res
