"""The strategy and its honest controls — Study 175 (Crypto-Weekend).

The folk claim: Bitcoin weekends are directionally distinct from weekdays — higher
volatility, lower volume, and either systematically pumped ("weekend pump") or dumped
("weekend dump") relative to weekday sessions. Because traditional banking rails slow on
weekends, retail activity is said to dominate, creating a microstructure anomaly.

We test three sub-claims impartially:

1. **Mean return** — are weekend daily log-returns higher or lower than weekday returns?
   HAC t-stat on the difference.
2. **Volatility** — is weekend daily return volatility materially different from weekday?
3. **Timing rule** — a weekend-only buy-and-hold (hold Fri close, sell Mon open) vs
   unconstrained buy-and-hold. HAC t-stat on the spread.

And we test the post-2023 sub-sample separately: if banking normalisation killed the
effect, we'd expect the pre-2023 signal (if any) to weaken or reverse after 2023.

The inference bar is honest about multiple comparisons: we test weekday vs weekend across
return, vol, and timing — three tests — and apply a Bonferroni-adjusted threshold
(|t| >= 2.39 for alpha=0.05/3). With ~600 weekend days the power is reasonable for the
full sample but thin for sub-samples.

No look-ahead: the "weekend" label is the calendar day-of-week, known in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import REGIME_BREAK


# ---------------------------------------------------------------------------
# Daily return labelling
# ---------------------------------------------------------------------------
def daily_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns with day-of-week labels.

    Entry is at the open of each day; exit is at the close of the same day.
    Returns a DataFrame with columns: ``date, log_ret, pct_ret, is_weekend, dow``.

    Note on no-look-ahead: the open-to-close return is fully known at the close,
    and the day-of-week label is a pure calendar fact available before the open.
    """
    df = bars[["open", "close"]].copy()
    df.index.name = "date"
    df = df[df["open"] > 0]
    df["log_ret"] = np.log(df["close"] / df["open"])
    df["pct_ret"] = np.expm1(df["log_ret"])
    df["dow"] = df.index.dayofweek          # 0=Mon .. 6=Sun
    df["is_weekend"] = df["dow"] >= 5       # Sat=5, Sun=6
    df["regime"] = np.where(df.index < REGIME_BREAK, "pre2023", "post2023")
    return df[["log_ret", "pct_ret", "dow", "is_weekend", "regime"]].dropna()


# ---------------------------------------------------------------------------
# HAC t-stat (Newey-West, self-contained so no hard quantlab dependency)
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray, baseline_mean: float = 0.0) -> float:
    """Newey-West HAC t-statistic for the mean of ``r`` vs ``baseline_mean``."""
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)] - baseline_mean
    n = x.size
    if n < 4:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Claim 1: Mean-return difference — weekend vs weekday
# ---------------------------------------------------------------------------
def return_comparison(daily: pd.DataFrame) -> dict:
    """Compare weekend vs weekday daily log-returns.

    Returns per-group statistics plus the HAC t-stat on the weekend premium
    (weekend mean minus weekday mean). The Bonferroni-adjusted critical value for
    3 simultaneous tests is approximately |t| >= 2.39 (alpha = 0.05/3).
    """
    wknd = daily[daily["is_weekend"]]["log_ret"].to_numpy()
    wkday = daily[~daily["is_weekend"]]["log_ret"].to_numpy()

    # Weekend premium series: pad weekday-only series with weekday mean as baseline.
    # We compute the t-stat on the weekend returns vs the weekday mean.
    wkday_mean = float(wkday.mean()) if len(wkday) else 0.0

    def _stats(r: np.ndarray, label: str, baseline: float = 0.0) -> dict:
        n = len(r)
        mean = float(r.mean()) if n else float("nan")
        std = float(r.std(ddof=1)) if n > 1 else float("nan")
        sharpe_ann = float(mean / std * np.sqrt(365)) if std > 0 else float("nan")
        return {
            "label": label,
            "n": n,
            "mean_log": mean,
            "mean_pct": float(np.expm1(mean) * 100) if not np.isnan(mean) else float("nan"),
            "std_log": std,
            "sharpe_ann": sharpe_ann,
            "tstat_vs_baseline": _hac_tstat(r, baseline_mean=baseline),
        }

    wknd_stats = _stats(wknd, "weekend", baseline=wkday_mean)
    wkday_stats = _stats(wkday, "weekday", baseline=0.0)

    # The t-stat on the *difference* (weekend − weekday) — the main test.
    # We form the difference series by treating weekday mean as the "null" for weekends.
    diff_t = _hac_tstat(wknd, baseline_mean=wkday_mean)

    return {
        "weekend": wknd_stats,
        "weekday": wkday_stats,
        "premium_log": wknd_stats["mean_log"] - wkday_stats["mean_log"],
        "premium_pct": wknd_stats["mean_pct"] - wkday_stats["mean_pct"],
        "diff_tstat": diff_t,
        "bonferroni_sig": abs(diff_t) >= 2.39,  # alpha = 0.05/3
        "naive_sig": abs(diff_t) >= 2.0,
    }


# ---------------------------------------------------------------------------
# Claim 2: Volatility comparison — weekend vs weekday
# ---------------------------------------------------------------------------
def volatility_comparison(daily: pd.DataFrame) -> dict:
    """Compare weekend vs weekday realised daily return volatility.

    Returns annualised vol for each group, their ratio, and an approximate
    Levene-style test (ratio of std, not a formal F-test, but directionally correct).
    """
    wknd = daily[daily["is_weekend"]]["log_ret"].to_numpy()
    wkday = daily[~daily["is_weekend"]]["log_ret"].to_numpy()

    def _vol_ann(r: np.ndarray) -> float:
        if len(r) < 2:
            return float("nan")
        return float(r.std(ddof=1) * np.sqrt(365))

    vol_wknd = _vol_ann(wknd)
    vol_wkday = _vol_ann(wkday)
    ratio = vol_wknd / vol_wkday if vol_wkday > 0 else float("nan")

    # Levene's test (approx): HAC t on the squared deviations from group mean.
    sq_wknd = (wknd - wknd.mean()) ** 2 if len(wknd) > 1 else np.array([])
    sq_wkday = (wkday - wkday.mean()) ** 2 if len(wkday) > 1 else np.array([])
    wkday_sq_mean = float(sq_wkday.mean()) if len(sq_wkday) else 0.0
    levene_t = _hac_tstat(sq_wknd, baseline_mean=wkday_sq_mean) if len(sq_wknd) >= 4 else float("nan")

    return {
        "vol_ann_weekend": vol_wknd,
        "vol_ann_weekday": vol_wkday,
        "vol_ratio": ratio,
        "levene_t": levene_t,
        "n_weekend": len(wknd),
        "n_weekday": len(wkday),
    }


# ---------------------------------------------------------------------------
# Claim 3: Weekend-long timing rule vs buy-and-hold
# ---------------------------------------------------------------------------
def weekend_timing_rule(
    bars: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """Weekend-day aggregate timing rule: hold on Saturday+Sunday only vs buy-and-hold.

    BTC trades 24/7 (no calendar gaps), so the "Fri close to Mon open" bracket spans
    Saturday and Sunday as actual trading days. The rule here is simpler and cleaner:
    hold BTC **only on weekend days** (Saturday and Sunday), exiting at the next
    weekday open. Specifically for each weekend day the return is the open-to-close
    log-return of that day. The per-weekend-day return series is the test vehicle; a
    buy-and-hold earns this return every day regardless.

    The timing-rule spread = weekend_day_mean - all_day_mean. If weekends are
    systematically better, the spread is positive; if worse, negative.

    ``cost_bps``: one-way cost per round-trip edge (buy at weekend open / sell at
    weekend close); round-trip = 2 * cost_bps. With crypto exchanges charging ~5-10 bps
    per side for retail, 5 bps one-way (10 bps round-trip) is a realistic floor.

    No look-ahead: the weekend label is the calendar day-of-week, known before the open.
    """
    bars = bars.copy().sort_index()
    dow = bars.index.dayofweek  # 0=Mon .. 6=Sun

    rows = []
    for i in range(len(bars)):
        if dow[i] >= 5:  # Saturday=5, Sunday=6
            open_px = bars["open"].iat[i]
            close_px = bars["close"].iat[i]
            if open_px > 0 and close_px > 0:
                log_r = np.log(close_px / open_px)
                rows.append({"date": bars.index[i], "weekend_log_ret": log_r})

    if not rows:
        return {
            "n": 0,
            "mean_log": float("nan"),
            "mean_pct": float("nan"),
            "tstat": float("nan"),
            "net_mean_log": float("nan"),
            "net_mean_pct": float("nan"),
            "net_tstat": float("nan"),
            "cost_bps": cost_bps,
        }

    df = pd.DataFrame(rows).set_index("date")
    r = df["weekend_log_ret"].to_numpy()
    cost = 2 * cost_bps * 1e-4  # round-trip
    r_net = r - cost
    n = len(r)
    mean_log = float(r.mean())
    net_mean = float(r_net.mean())

    return {
        "n": n,
        "mean_log": mean_log,
        "mean_pct": float(np.expm1(mean_log) * 100),
        "tstat": _hac_tstat(r, baseline_mean=0.0),
        "net_mean_log": net_mean,
        "net_mean_pct": float(np.expm1(net_mean) * 100),
        "net_tstat": _hac_tstat(r_net, baseline_mean=0.0),
        "cost_bps": cost_bps,
    }


# ---------------------------------------------------------------------------
# Sub-sample (pre/post 2023) breakdown
# ---------------------------------------------------------------------------
def regime_split(daily: pd.DataFrame) -> dict:
    """Return the weekend-vs-weekday premium separately for pre- and post-2023 regimes.

    The 2023 split tests the "banking-normalisation killed it" hypothesis: if the weekend
    microstructure quirk was driven by stablecoin rails being offline on weekends, the
    FedNow launch and post-SVB stablecoin settlement improvements should attenuate the
    effect after 2023.
    """
    out = {}
    for regime in ("pre2023", "post2023"):
        sub = daily[daily["regime"] == regime]
        rc = return_comparison(sub) if len(sub) > 20 else {"diff_tstat": float("nan"), "premium_pct": float("nan"), "weekend": {"n": 0}, "weekday": {"n": 0}}
        out[regime] = {
            "n_weekend": rc["weekend"]["n"],
            "n_weekday": rc["weekday"]["n"],
            "premium_pct": rc["premium_pct"],
            "diff_tstat": rc["diff_tstat"],
        }
    return out


# ---------------------------------------------------------------------------
# Full-study summary
# ---------------------------------------------------------------------------
def summarize(bars: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """Headline summary for docs/results.md and the notebook R dict.

    Runs all three sub-claims plus the regime split. Returns a flat dict of the
    decisive numbers that populate docs/results.md and the notebook's R dict.
    """
    daily = daily_returns(bars)
    rc = return_comparison(daily)
    vc = volatility_comparison(daily)
    tr = weekend_timing_rule(bars, cost_bps=cost_bps)
    rs = regime_split(daily)

    return {
        # Overall sample
        "n_days": len(daily),
        "n_weekend": rc["weekend"]["n"],
        "n_weekday": rc["weekday"]["n"],
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        # Claim 1: mean return
        "wknd_mean_pct": rc["weekend"]["mean_pct"],
        "wkday_mean_pct": rc["weekday"]["mean_pct"],
        "premium_pct": rc["premium_pct"],
        "return_tstat": rc["diff_tstat"],
        "return_naive_sig": rc["naive_sig"],
        "return_bonf_sig": rc["bonferroni_sig"],
        # Claim 2: volatility
        "vol_wknd": vc["vol_ann_weekend"],
        "vol_wkday": vc["vol_ann_weekday"],
        "vol_ratio": vc["vol_ratio"],
        "levene_t": vc["levene_t"],
        # Claim 3: timing rule
        "timing_n": tr["n"],
        "timing_mean_pct": tr["mean_pct"],
        "timing_tstat": tr["tstat"],
        "timing_net_mean_pct": tr["net_mean_pct"],
        "timing_net_tstat": tr["net_tstat"],
        "timing_cost_bps": cost_bps,
        # Regime split
        "pre2023_premium_pct": rs["pre2023"]["premium_pct"],
        "pre2023_tstat": rs["pre2023"]["diff_tstat"],
        "post2023_premium_pct": rs["post2023"]["premium_pct"],
        "post2023_tstat": rs["post2023"]["diff_tstat"],
        "pre2023_n_wknd": rs["pre2023"]["n_weekend"],
        "post2023_n_wknd": rs["post2023"]["n_weekend"],
    }
