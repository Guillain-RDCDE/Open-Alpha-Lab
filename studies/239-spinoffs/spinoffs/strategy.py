"""Strategy and honest controls — Study 239 (Spinoffs).

The claim: spin-off children systematically outperform the market (SPY) in the
12–24 months following the ex-distribution date (Greenblatt 1997, Cusatis, Miles &
Woolridge 1993, McConnell & Ovtchinnikov 2004).

We test:

**Alpha arm** — forward returns of the child ticker (buy-and-hold from the
ex-distribution date + 1 trading day) vs the contemporaneous SPY return over
the same window, at 6/12/18/24-month horizons.

**Null arm** — same spun-off children but with randomly drawn ex-dates (permuted
baseline), to verify the engine detects signal when it exists.

Honest caveats (stated prominently):
1. The hardcoded table is *curated* — it covers widely-discussed spin-offs, which
   biases toward children that were at least somewhat notable.
2. Spin-off tickers often change (post-merger absorptions, reverse mergers);
   yfinance may have gaps or missing history for small or short-lived children.
3. The children may be small/illiquid in early trading; bid-ask spreads and market
   impact are not modelled.
4. "Beat SPY" is a weak bar vs a size-matched or sector-matched benchmark.

Functions:
- ``summarize_alpha``     — headline HAC stats on alpha (child − SPY) for one horizon.
- ``compare_horizons``    — full comparison table across all four horizons.
- ``hac_tstat``           — Newey-West HAC t-stat.
- ``cost_adjusted_return``— deduct round-trip cost and annualise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``x``."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Summarize one horizon
# ---------------------------------------------------------------------------
def summarize_alpha(
    events: pd.DataFrame,
    col: str = "alpha_12m",
) -> dict:
    """Headline statistics on the alpha (child − SPY) for one horizon.

    Returns n, mean alpha (%), median, std, HAC t-stat, win-rate (alpha > 0).
    Also returns child mean and SPY mean separately.
    """
    alpha_col = col  # e.g. "alpha_12m"
    # Infer the child and spy columns from the alpha column name
    suffix = col.replace("alpha_", "")
    ret_col = f"ret_{suffix}"
    spy_col = f"spy_{suffix}"

    alpha = events[alpha_col].dropna().to_numpy(dtype=float)
    n = alpha.size
    if n < 2:
        return {k: float("nan") for k in ("n", "mean_alpha_pct", "median_alpha_pct",
                                           "std_alpha_pct", "win_rate", "tstat",
                                           "mean_child_pct", "mean_spy_pct")}

    ret_arr = events[ret_col].dropna().to_numpy(dtype=float) if ret_col in events.columns else np.array([])
    spy_arr = events[spy_col].dropna().to_numpy(dtype=float) if spy_col in events.columns else np.array([])

    return {
        "n": int(n),
        "mean_alpha_pct": float(alpha.mean() * 100),
        "median_alpha_pct": float(np.median(alpha) * 100),
        "std_alpha_pct": float(alpha.std(ddof=1) * 100),
        "win_rate": float((alpha > 0).mean()),
        "tstat": hac_tstat(alpha),
        "mean_child_pct": float(ret_arr.mean() * 100) if ret_arr.size else float("nan"),
        "mean_spy_pct": float(spy_arr.mean() * 100) if spy_arr.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Full horizon comparison table
# ---------------------------------------------------------------------------
def compare_horizons(
    events: pd.DataFrame,
    alpha_cols: tuple[str, ...] = ("alpha_6m", "alpha_12m", "alpha_18m", "alpha_24m"),
    labels: tuple[str, ...] = ("6m (~126d)", "12m (~252d)", "18m (~378d)", "24m (~504d)"),
) -> pd.DataFrame:
    """For each horizon return a summary row with alpha stats.

    Returns a DataFrame with columns:
    horizon, n, mean_child_pct, mean_spy_pct, mean_alpha_pct,
    median_alpha_pct, std_alpha_pct, win_rate, tstat.
    """
    rows = []
    for col, lbl in zip(alpha_cols, labels):
        s = summarize_alpha(events, col=col)
        rows.append({"horizon": lbl, **s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Positive-control helper
# ---------------------------------------------------------------------------
def spinoff_vs_null_alpha(
    events_planted: pd.DataFrame,
    events_null: pd.DataFrame,
    col: str = "alpha_12m",
) -> tuple[float, float]:
    """Returns (planted_mean_alpha, null_mean_alpha) for one horizon.

    Used in the synthetic positive-control test: when premium_bps > 0 the planted
    tape's mean alpha should exceed the null tape's mean alpha.
    """
    planted = events_planted[col].dropna().to_numpy(dtype=float)
    null = events_null[col].dropna().to_numpy(dtype=float)
    return float(planted.mean() * 100), float(null.mean() * 100)


# ---------------------------------------------------------------------------
# Tradability: cost-adjusted annualised return
# ---------------------------------------------------------------------------
def cost_adjusted_return(
    mean_alpha: float,
    horizon_days: int,
    cost_bps: float = 10.0,
) -> float:
    """Deduct a round-trip cost from the *alpha* (not gross return) and annualise.

    Spin-off children are often small-cap immediately post-distribution, so we use
    a conservative 10 bps round-trip default (vs 5 bps for large-caps).

    Returns the annualised net alpha (fraction), so 0.05 means +5%/yr alpha.
    """
    net = mean_alpha - cost_bps * 1e-4
    ann_factor = TRADING_DAYS_PER_YEAR / horizon_days
    return float((1.0 + net) ** ann_factor - 1.0)
