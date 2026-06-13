"""The strategy and its honest controls — Study 115 (Credit-Spreads).

The folk claim: the credit-risk premium leads the equity market.  When high-yield credit
(HYG) underperforms Treasuries (IEF), equity stress follows.  We operationalise this as:

1. **Credit-spread proxy** — 20-day rolling return of HYG minus IEF
   (negative = credit stress; positive = credit tightening / risk-on).
2. **Signal** — when the proxy falls below its rolling median (credit widening regime),
   we expect the next ``forward_days`` SPY return to be negative vs the unconditional.
3. **Baseline** — the unconditional ``forward_days`` SPY return (buy-and-hold).
4. **Alternative proxy** — HYG/LQD relative strength (HY vs IG, strips duration).

We do NOT use a barrier exit here — instead we look at the *distribution* of forward
returns conditioned on the credit signal vs the unconditional.  The HAC t-stat on the
difference (signal mean − unconditional mean) decides the inference.  For a predictive
claim ("widening credit → negative equity") we specifically test:

- Signal periods: forward return (mean, t-stat, win-rate of negative next return).
- Non-signal periods: same stats.
- Difference (signal − non-signal): the key number for the inference bar.

No look-ahead: the spread is computed on prices up to day *t*, the forward return is
from *t* to *t + forward_days*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Credit-spread proxy construction
# ---------------------------------------------------------------------------
def hyg_ief_spread(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling ``window``-day return of HYG minus IEF.

    Negative values indicate HYG underperforming Treasuries — a proxy for widening
    credit spreads (rising risk premium in high-yield).  Positive = credit tightening.
    Uses *total-return* proxying via adjusted-close ETF prices.  The result is
    tz-naive and aligned on trading days.

    Note: this is an ETF-based approximation of the OAS credit spread, not the
    level-spread itself (FRED data is unavailable in this environment).  The proxy
    captures the economic direction but not the exact magnitude.
    """
    hyg_ret = prices["hyg"].pct_change(window)
    ief_ret = prices["ief"].pct_change(window)
    spread = hyg_ret - ief_ret
    spread.name = "hyg_ief_spread"
    return spread


def hyg_lqd_rs(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling ``window``-day relative strength of HYG vs LQD.

    HYG/LQD captures the *credit-risk premium* component more cleanly than HYG/IEF
    because both HYG and LQD carry similar interest-rate duration — subtracting LQD
    removes the rate sensitivity, leaving mainly the high-yield vs investment-grade
    credit differential.

    Negative → HY underperforming IG (stress).  Positive → HY outperforming (risk-on).
    """
    hyg_ret = prices["hyg"].pct_change(window)
    lqd_ret = prices["lqd"].pct_change(window)
    rs = hyg_ret - lqd_ret
    rs.name = "hyg_lqd_rs"
    return rs


# ---------------------------------------------------------------------------
# Forward-return engine
# ---------------------------------------------------------------------------
def forward_returns(prices: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    """SPY ``forward_days``-day forward return, formed without look-ahead.

    At each date *t*, the return is computed from *t*'s close to *t + forward_days*'s
    close.  This is assigned to *t* (the signal date), so it is strictly forward-looking
    from the decision point.
    """
    spy = prices["spy"]
    fwd = spy.shift(-forward_days) / spy - 1.0
    fwd.name = f"spy_fwd_{forward_days}d"
    return fwd


def build_signals(
    prices: pd.DataFrame,
    window: int = 20,
    forward_days: int = 5,
    rolling_median_window: int = 60,
) -> pd.DataFrame:
    """Assemble the signal DataFrame with spread proxies, regime flags, and forward returns.

    Columns:
    - ``hyg_ief_spread``     rolling 20d return HYG − IEF
    - ``hyg_lqd_rs``         rolling 20d relative strength HYG vs LQD
    - ``spy_fwd``            SPY forward return (forward_days)
    - ``spread_wide``        1 if hyg_ief_spread < its rolling median (credit stress)
    - ``rs_wide``            1 if hyg_lqd_rs < its rolling median (HY stress vs IG)
    - ``spy_ret``            SPY daily log-return (for context)

    The regime flags are formed from the rolling median over the past
    ``rolling_median_window`` observations, not in-sample — so there is no look-ahead
    in the classification.
    """
    spread = hyg_ief_spread(prices, window=window)
    rs = hyg_lqd_rs(prices, window=window)
    fwd = forward_returns(prices, forward_days=forward_days)
    spy_ret = np.log(prices["spy"] / prices["spy"].shift(1))

    # Rolling median computed from data up to and including t (no future data)
    roll_med_spread = spread.rolling(rolling_median_window, min_periods=rolling_median_window // 2).median()
    roll_med_rs = rs.rolling(rolling_median_window, min_periods=rolling_median_window // 2).median()

    df = pd.DataFrame(
        {
            "hyg_ief_spread": spread,
            "hyg_lqd_rs": rs,
            "spy_fwd": fwd,
            "spread_wide": (spread < roll_med_spread).astype(int),
            "rs_wide": (rs < roll_med_rs).astype(int),
            "spy_ret": spy_ret,
        }
    )
    return df.dropna(subset=["hyg_ief_spread", "hyg_lqd_rs", "spy_fwd"])


# ---------------------------------------------------------------------------
# Regime-conditional analysis
# ---------------------------------------------------------------------------
def regime_stats(signal_col: pd.Series, fwd_col: pd.Series) -> dict:
    """Return summary stats split by signal (1 = wide/stress) and non-signal (0 = tight).

    The key number is ``diff_tstat`` — the HAC t-stat on the difference of forward
    returns between the two regimes.  For the claim ("widening credit → lower equity")
    to pass the inference bar, ``diff_tstat`` should be < −2 (stress periods have
    significantly lower forward returns than non-stress periods).
    """
    mask_on = signal_col == 1
    mask_off = signal_col == 0

    on = fwd_col[mask_on].dropna().to_numpy(dtype=float)
    off = fwd_col[mask_off].dropna().to_numpy(dtype=float)
    all_ = fwd_col.dropna().to_numpy(dtype=float)

    def _stats(r: np.ndarray, label: str) -> dict:
        n = r.size
        if n < 5:
            return {f"{label}_n": n, f"{label}_mean_bps": float("nan"),
                    f"{label}_win_rate": float("nan"), f"{label}_tstat": float("nan")}
        mu = r.mean()
        # HAC Newey-West
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        tstat = float(mu / se) if se > 0 else float("nan")
        return {
            f"{label}_n": int(n),
            f"{label}_mean_bps": float(mu * 1e4),
            f"{label}_win_rate": float((r > 0).mean()),
            f"{label}_tstat": tstat,
        }

    out = {}
    out.update(_stats(on, "stress"))
    out.update(_stats(off, "calm"))
    out.update(_stats(all_, "unconditional"))

    # Difference: stress minus calm (the key test of the claim)
    diff = on.mean() - off.mean() if (on.size > 0 and off.size > 0) else float("nan")
    # HAC t-stat on pooled difference (approximate: use combined residuals)
    if on.size > 5 and off.size > 5:
        diff_series = np.concatenate([on - on.mean(), -(off - off.mean())])
        n_d = on.size + off.size
        lags_d = max(1, int(np.floor(4.0 * (n_d / 100.0) ** (2.0 / 9.0))))
        lrv_on = _long_run_var(on)
        lrv_off = _long_run_var(off)
        se_diff = np.sqrt(lrv_on / on.size + lrv_off / off.size)
        diff_tstat = float(diff / se_diff) if se_diff > 0 else float("nan")
    else:
        diff_tstat = float("nan")

    out["diff_mean_bps"] = float(diff * 1e4) if np.isfinite(diff) else float("nan")
    out["diff_tstat"] = diff_tstat
    return out


def _long_run_var(r: np.ndarray) -> float:
    """Newey-West long-run variance of the mean of ``r``."""
    n = r.size
    if n < 2:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return max(lrv, 0.0)


# ---------------------------------------------------------------------------
# Full study summary
# ---------------------------------------------------------------------------
def summarize(
    prices: pd.DataFrame,
    window: int = 20,
    forward_days: int = 5,
    rolling_median_window: int = 60,
) -> dict:
    """Compute all headline statistics for the credit-spread study.

    Returns a flat dict with:
    - signal stats for both proxy variants (HYG-IEF and HYG/LQD)
    - regime conditional forward-return stats
    - the HAC t-stat on the stress-vs-calm difference (the inference-bar number)
    - basic sample info (n, date range, fingerprint)
    """
    sigs = build_signals(
        prices,
        window=window,
        forward_days=forward_days,
        rolling_median_window=rolling_median_window,
    )
    n = len(sigs)
    date_start = sigs.index[0].strftime("%Y-%m-%d")
    date_end = sigs.index[-1].strftime("%Y-%m-%d")

    stats_spread = regime_stats(sigs["spread_wide"], sigs["spy_fwd"])
    stats_rs = regime_stats(sigs["rs_wide"], sigs["spy_fwd"])

    # Correlation: does the spread level predict next-period SPY return?
    corr_spread = float(
        np.corrcoef(
            sigs["hyg_ief_spread"].dropna().to_numpy(),
            sigs["spy_fwd"].reindex(sigs["hyg_ief_spread"].dropna().index).to_numpy(),
        )[0, 1]
    ) if n > 10 else float("nan")

    corr_rs = float(
        np.corrcoef(
            sigs["hyg_lqd_rs"].dropna().to_numpy(),
            sigs["spy_fwd"].reindex(sigs["hyg_lqd_rs"].dropna().index).to_numpy(),
        )[0, 1]
    ) if n > 10 else float("nan")

    out = {
        "n": n,
        "date_start": date_start,
        "date_end": date_end,
        "window": window,
        "forward_days": forward_days,
        # HYG-IEF spread proxy
        "spread_corr": corr_spread,
        **{f"spread_{k}": v for k, v in stats_spread.items()},
        # HYG/LQD RS proxy
        "rs_corr": corr_rs,
        **{f"rs_{k}": v for k, v in stats_rs.items()},
    }
    return out
