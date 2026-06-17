"""The rotation strategy and honest controls — Study 222 (Altseason-Rotation).

The folk recipe: when BTC dominance is *falling*, rotate from BTC into altcoins — go long
an equal-weighted alt basket and short BTC. Exit the rotation when dominance recovers.
This is the explicit strategy version of the dominance question tested in Study 134:
instead of measuring regression slopes and conditional means, we simulate the actual
long/short PnL track record, apply transaction costs, and compute Sharpe ratios vs
buy-and-hold benchmarks.

Two signal variants define the rotation trigger:
- **Regime-fall signal**: BTC dominance has fallen more than a threshold (e.g. 2 percentage
  points) over the past 20 trading days. Enter the rotation; exit when dominance rises above
  half the threshold over 10 days.
- **Level signal**: BTC dominance is below its rolling 52-week median. Enter the rotation;
  exit when it rises back above.

Two controls prevent this from being a win-by-design exercise:
- **Buy-and-hold BTC**: the simplest crypto benchmark. The rotation must beat BTC's own
  Sharpe ratio to be a genuine timing improvement.
- **Buy-and-hold equal-weight alts**: if alts just always beat BTC (secular beta), the
  strategy's alpha is zero even if it posts positive returns.

Costs are applied on every rotation event (entry and exit). Default: 40 bps round-trip
per event (realistic for liquid perpetuals on large exchanges). A high-cost scenario
(80 bps) is also reported.

No look-ahead: the dominance signal is formed on day t closes; the position is entered
at day t+1 (approximated as the next-day close, the standard 1-day delay convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import compute_dominance, ALT_TICKERS, BTC_TICKER, SUPPLY_M, DEFAULT_COST_BPS


# ---------------------------------------------------------------------------
# Core signal construction
# ---------------------------------------------------------------------------
def dominance_change_signal(
    panel: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.02,
    btc_col: str = BTC_TICKER,
    supply_map: dict | None = None,
) -> pd.Series:
    """Binary rotation signal: 1 = 'alt season' (dominance falling), 0 = 'BTC season'.

    A value of 1 is emitted on day t when:
      dom_t − dom_{t-lookback} < −threshold

    i.e. BTC's basket dominance has fallen by more than ``threshold`` (in decimal, so
    0.02 = 2 percentage points) over the past ``lookback`` trading days.

    Returns a daily Series with values in {0, 1}, indexed by date. NaN for the first
    ``lookback`` rows (insufficient history). Signal is formed on day t closes;
    positions are entered on day t+1.
    """
    if supply_map is None:
        supply_map = SUPPLY_M
    dom = compute_dominance(panel, btc_col=btc_col, supply_map=supply_map)
    dom_change = dom.diff(lookback)
    signal = (dom_change < -threshold).astype(float)
    signal[dom_change.isna()] = float("nan")
    signal.name = "rotation_signal"
    return signal


def dominance_level_signal(
    panel: pd.DataFrame,
    window: int = 252,
    btc_col: str = BTC_TICKER,
    supply_map: dict | None = None,
) -> pd.Series:
    """Rotation signal based on dominance level vs its rolling median.

    Emits 1 when BTC dominance is below its trailing ``window``-day median (alts
    have been structurally gaining market share). Emits 0 when BTC dominance is
    above or equal to its median.

    Returns a daily Series with values in {0, 1}, NaN where insufficient history.
    """
    if supply_map is None:
        supply_map = SUPPLY_M
    dom = compute_dominance(panel, btc_col=btc_col, supply_map=supply_map)
    med = dom.rolling(window, min_periods=window // 2).median()
    signal = (dom < med).astype(float)
    signal[med.isna()] = float("nan")
    signal.name = "level_signal"
    return signal


# ---------------------------------------------------------------------------
# Strategy PnL engine
# ---------------------------------------------------------------------------
def rotation_pnl(
    panel: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
    btc_col: str = BTC_TICKER,
    alt_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Simulate the long-alt / short-BTC rotation strategy.

    Position logic (daily, no leverage):
    - When signal[t] = 1 (alt season): long equal-weighted alts, short BTC (dollar-neutral).
    - When signal[t] = 0 (BTC season): flat (0% exposure).
    - Entry and exit at t+1 close (1-day lag after signal formation on day t).
    - Transaction costs applied on each position change (entry or exit): ``cost_bps`` bps
      on the total notional (round-trip).

    Returns a DataFrame indexed by date with columns:
      ``signal``, ``alt_fwd`` (log), ``btc_fwd`` (log), ``spread_fwd`` (log),
      ``position`` (lagged signal, 0/1), ``gross_ret``, ``cost``, ``net_ret``,
      ``cum_gross``, ``cum_net``.

    All returns are log-returns. The strategy is dollar-neutral (equal long alt/short BTC),
    so benchmark is the alt-minus-BTC spread, not either asset in isolation.
    """
    if alt_cols is None:
        alt_cols = [c for c in panel.columns if c != btc_col]

    log_panel = np.log(panel)
    # Daily log returns
    btc_ret = log_panel[btc_col].diff()
    alt_rets = log_panel[alt_cols].diff()
    alt_ew = alt_rets.mean(axis=1)  # equal-weighted alt daily log-return
    spread = alt_ew - btc_ret  # daily alt-minus-BTC spread

    # Align signal: position on day t is driven by signal on day t-1 (1-day lag)
    sig_aligned = signal.reindex(panel.index).ffill(limit=1)  # no forward fill beyond 1 day
    pos = sig_aligned.shift(1).fillna(0.0)  # 1-day lag entry

    # Detect position changes for cost calculation
    pos_change = pos.diff().abs()
    pos_change.iloc[0] = 0.0
    cost = pos_change * (cost_bps / 10_000.0)  # cost deducted on change days

    gross_ret = pos * spread
    net_ret = gross_ret - cost

    # Drop NaN rows (first few days where signal or returns are NaN)
    mask = pos.notna() & spread.notna() & gross_ret.notna()
    mask &= ~spread.isna()

    out = pd.DataFrame(
        {
            "signal": sig_aligned,
            "alt_fwd": alt_ew,
            "btc_fwd": btc_ret,
            "spread_fwd": spread,
            "position": pos,
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
        },
        index=panel.index,
    )
    out = out[mask].dropna(subset=["spread_fwd", "gross_ret"])

    # Cumulative log returns
    out["cum_gross"] = out["gross_ret"].cumsum()
    out["cum_net"] = out["net_ret"].cumsum()

    return out


# ---------------------------------------------------------------------------
# Benchmark returns
# ---------------------------------------------------------------------------
def benchmark_pnl(
    panel: pd.DataFrame,
    btc_col: str = BTC_TICKER,
    alt_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Buy-and-hold benchmarks: BTC, equal-weight alts, equal-weight full basket.

    Returns a DataFrame indexed by date with columns:
      ``btc_ret``, ``alt_ew_ret``, ``basket_ew_ret``  (daily log-returns),
      ``cum_btc``, ``cum_alt``, ``cum_basket``  (cumulative log-returns).
    """
    if alt_cols is None:
        alt_cols = [c for c in panel.columns if c != btc_col]

    log_panel = np.log(panel)
    btc_ret = log_panel[btc_col].diff().dropna()
    alt_ew_ret = log_panel[alt_cols].diff().mean(axis=1).dropna()

    n_assets = 1 + len(alt_cols)
    basket_ew_ret = (btc_ret + alt_ew_ret * len(alt_cols)) / n_assets

    out = pd.DataFrame(
        {
            "btc_ret": btc_ret,
            "alt_ew_ret": alt_ew_ret,
            "basket_ew_ret": basket_ew_ret,
        }
    ).dropna()

    out["cum_btc"] = out["btc_ret"].cumsum()
    out["cum_alt"] = out["alt_ew_ret"].cumsum()
    out["cum_basket"] = out["basket_ew_ret"].cumsum()
    return out


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def performance_summary(
    returns: pd.Series | np.ndarray,
    label: str = "",
    periods_per_year: int = 252,
) -> dict:
    """Annualised performance statistics for a daily return series.

    Returns: n, ann_return_pct, ann_vol_pct, sharpe, max_dd_pct, calmar, skew,
    and HAC t-stat (for testing whether mean daily return != 0).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
        return {k: float("nan") for k in
                ["label", "n", "ann_return_pct", "ann_vol_pct", "sharpe",
                 "max_dd_pct", "calmar", "skew", "tstat"]}

    ann_ret = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    # Max drawdown on cumulative log-return series
    cum = np.cumsum(r)
    roll_max = np.maximum.accumulate(cum)
    dd = cum - roll_max
    max_dd = float(dd.min())

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")
    skew = float(pd.Series(r).skew()) if n > 2 else float("nan")

    # HAC t-stat
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "label": label,
        "n": int(n),
        "ann_return_pct": float(ann_ret * 100),
        "ann_vol_pct": float(ann_vol * 100),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd * 100),
        "calmar": float(calmar),
        "skew": float(skew),
        "tstat": float(tstat),
    }


# ---------------------------------------------------------------------------
# Rotation event counter
# ---------------------------------------------------------------------------
def count_rotations(pnl: pd.DataFrame) -> dict:
    """Count rotation entry/exit events and average holding period.

    A 'rotation' is a contiguous block of days where position = 1.
    Returns: n_rotations, avg_hold_days, total_days_active, pct_days_active.
    """
    pos = pnl["position"].values
    n = len(pos)
    if n == 0:
        return {"n_rotations": 0, "avg_hold_days": 0, "total_days_active": 0,
                "pct_days_active": 0.0}

    entries = 0
    hold_runs = []
    run = 0
    for i in range(n):
        if pos[i] == 1.0:
            run += 1
            if i == 0 or pos[i - 1] == 0.0:
                entries += 1
        else:
            if run > 0:
                hold_runs.append(run)
            run = 0
    if run > 0:
        hold_runs.append(run)

    total_active = int((pos == 1.0).sum())
    avg_hold = float(np.mean(hold_runs)) if hold_runs else 0.0

    return {
        "n_rotations": entries,
        "avg_hold_days": avg_hold,
        "total_days_active": total_active,
        "pct_days_active": float(100.0 * total_active / n),
    }
