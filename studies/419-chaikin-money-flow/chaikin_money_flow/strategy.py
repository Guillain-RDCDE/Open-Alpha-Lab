"""The indicator, the timing rule, and the honest arbiters — Study 419 (CMF).

**Chaikin Money Flow (CMF)** is Marc Chaikin's volume-flow oscillator.  For each bar it
computes a *Money Flow Multiplier* from where the close sits inside the bar's range,
multiplies it by volume to get *Money Flow Volume*, then sums that over a trailing window
and divides by the window's total volume:

    MFM_t  = ((C - L) - (H - C)) / (H - L)            in [-1, +1]
    MFV_t  = MFM_t * Volume_t
    CMF_t  = sum_{w} MFV / sum_{w} Volume             in [-1, +1]

CMF > 0 is "accumulation" (buyers in control), CMF < 0 is "distribution" (sellers).  The
folk claim: **accumulation/distribution leads price** — sustained CMF > 0 precedes a
rise, CMF < 0 precedes a fall.

We test that by turning CMF into a **long/flat timing rule** on SPY (and a small panel),
and racing it — NET of costs, excess-of-cash — against the obvious simpler benchmarks
(a 50/200 SMA trend filter, a MACD trend filter) and against buy-and-hold:

    CMF > 0  ->  go long  (accumulation; the bullish lead)
    CMF <= 0 ->  go flat  (distribution; step aside)

Execution discipline (one lag, documented once): the indicator is computed on bars up to
and including day *t*; the position for day *t* is decided at *t*'s close and earns the
return of day *t+1* (a single ``shift(1)``).  When flat, the book earns the risk-free
rate (cash) and we report **excess-of-cash** so every race is excess-vs-excess.  Costs
are charged one-way x NAV on every change of position.

Arbiters:
- one-sample / HAC (Newey-West) *t* on the strategy's daily NET excess returns,
- a **block permutation placebo** that re-orders the position blocks against the realised
  returns (destroys timing, keeps the marginal exposure) — the honest null for "is the
  CMF timing doing anything?",
- a **cost sweep** (one-way bps x NAV),
- the head-to-head **CMF vs SMA(50/200) vs MACD vs buy-and-hold** NET excess Sharpe race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def chaikin_money_flow(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow over a trailing ``window`` (default 20), values in [-1, +1].

    MFM = ((C-L) - (H-C)) / (H-L); MFV = MFM * Volume; CMF = sum_w MFV / sum_w Volume.
    On a doji bar (H == L) the multiplier is undefined; we set its MFV to 0 (no flow).
    Valid from bar ``window`` onward.
    """
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    vol = bars["volume"].astype(float)

    rng = (high - low)
    mfm = ((close - low) - (high - close)) / rng.replace(0.0, np.nan)
    mfm = mfm.fillna(0.0)                       # doji bar -> no directional flow
    mfv = mfm * vol
    num = mfv.rolling(window, min_periods=window).sum()
    den = vol.rolling(window, min_periods=window).sum()
    cmf = num / den.replace(0.0, np.nan)
    return cmf.rename("cmf")


def sma_trend(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """50/200 SMA trend filter: +1 when fast SMA > slow SMA else 0 (a long/flat signal)."""
    sf = close.rolling(fast, min_periods=fast).mean()
    ss = close.rolling(slow, min_periods=slow).mean()
    sig = (sf > ss).astype(float)
    sig[ss.isna()] = np.nan
    return sig.rename("sma_trend")


def macd_trend(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD trend filter: +1 when the MACD line is above its signal line else 0."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    out = (macd > sig_line).astype(float)
    out.iloc[:slow] = np.nan                     # warm-up
    return out.rename("macd_trend")


# ---------------------------------------------------------------------------
# Timing rule: indicator -> long/flat position (no look-ahead)
# ---------------------------------------------------------------------------
def positions_from_cmf(cmf: pd.Series, thr: float = 0.0) -> pd.Series:
    """Long/flat position from CMF: long (1) when CMF > thr, flat (0) otherwise.

    The position on day *t* uses only the CMF value at *t* (a close-formed decision);
    ``backtest`` applies the single execution shift so it earns *t+1*'s return.  Flat
    (0) while CMF is NaN (warm-up).
    """
    pos = (cmf > thr).astype(float)
    pos[cmf.isna()] = 0.0
    return pos.rename("pos")


def positions_from_filter(sig: pd.Series) -> pd.Series:
    """Turn a +1/0/NaN filter (SMA or MACD) into a clean long/flat position (NaN->0)."""
    return sig.fillna(0.0).rename("pos")


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
      (asset_ret - rf when long, 0 when flat) so the race is excess-vs-excess.
    - ``cost_bps`` is charged **one-way x NAV** on every change in position
      (|pos_t - pos_{t-1}| * cost), subtracted from that day's return.

    Columns: ``asset_ret``, ``pos`` (the lagged effective position), ``gross_excess``,
    ``cost``, ``net_excess``.
    """
    close = bars["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0

    eff = position.shift(1).fillna(0.0)          # yesterday's close-decision earns today
    gross_excess = eff * (asset_ret - rf_daily)
    turn = eff.diff().abs().fillna(eff.abs())    # one-way turnover x NAV
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


def summarize_book(net_excess: pd.Series, position: pd.Series | None = None) -> dict:
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
    seed: int = 419,
) -> dict:
    """Placebo: shuffle the position in contiguous blocks vs the realised returns.

    The position series is cut into ``block``-day chunks and the chunks are randomly
    re-ordered, preserving the *fraction of time long* and the within-block
    autocorrelation, but **destroying the alignment** between the CMF signal and the
    next-day return.  If the real strategy's mean excess return sits in the bulk of this
    shuffled distribution, the timing carries no information.

    Returns the observed mean, the placebo mean (avg), and a one-tailed p-value
    = P(placebo mean >= observed mean).
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
    window: int = 20,
    cmf_thr: float = 0.0,
    cost_bps: float = 2.0,
    rf_annual: float = 0.0,
    perm_iter: int = 2000,
    perm_seed: int = 419,
) -> dict:
    """End-to-end: build CMF, SMA(50/200), MACD; run each as a long/flat book; race them.

    Returns a nested dict with one block per arm (``cmf``, ``sma``, ``macd``,
    ``buy_hold``), the head-to-head deltas (CMF net excess Sharpe minus each benchmark),
    and the block-permutation placebo p-value for the CMF arm.
    """
    cmf = chaikin_money_flow(bars, window=window)
    pos_cmf = positions_from_cmf(cmf, thr=cmf_thr)
    pos_sma = positions_from_filter(sma_trend(bars["close"]))
    pos_macd = positions_from_filter(macd_trend(bars["close"]))

    bt_cmf = backtest(bars, pos_cmf, cost_bps=cost_bps, rf_annual=rf_annual)
    bt_sma = backtest(bars, pos_sma, cost_bps=cost_bps, rf_annual=rf_annual)
    bt_macd = backtest(bars, pos_macd, cost_bps=cost_bps, rf_annual=rf_annual)
    bh = buy_hold_excess(bars, rf_annual=rf_annual)

    asset_excess = (bars["close"].astype(float).pct_change().fillna(0.0)
                    - ((1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0))

    res = {
        "cmf": {
            "net": summarize_book(bt_cmf["net_excess"], bt_cmf["pos"]),
            "gross": summarize_book(bt_cmf["gross_excess"], bt_cmf["pos"]),
            "perm": block_permutation_pvalue(asset_excess, pos_cmf,
                                             n_iter=perm_iter, seed=perm_seed),
        },
        "sma": {
            "net": summarize_book(bt_sma["net_excess"], bt_sma["pos"]),
            "gross": summarize_book(bt_sma["gross_excess"], bt_sma["pos"]),
        },
        "macd": {
            "net": summarize_book(bt_macd["net_excess"], bt_macd["pos"]),
            "gross": summarize_book(bt_macd["gross_excess"], bt_macd["pos"]),
        },
        "buy_hold": {"net": summarize_book(bh)},
        "params": {
            "window": window, "cmf_thr": cmf_thr,
            "cost_bps": cost_bps, "rf_annual": rf_annual,
        },
    }
    res["delta_sharpe_cmf_minus_sma"] = (
        res["cmf"]["net"]["sharpe"] - res["sma"]["net"]["sharpe"]
    )
    res["delta_sharpe_cmf_minus_macd"] = (
        res["cmf"]["net"]["sharpe"] - res["macd"]["net"]["sharpe"]
    )
    res["delta_sharpe_cmf_minus_bh"] = (
        res["cmf"]["net"]["sharpe"] - res["buy_hold"]["net"]["sharpe"]
    )
    return res
