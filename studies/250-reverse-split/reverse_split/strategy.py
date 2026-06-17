"""Strategy and honest controls — Study 250 (Reverse-Split).

The claim: stocks underperform (the "kiss of death") in the months following a
reverse split.  Reverse splits are done by distressed companies to avoid delisting,
so negative subsequent performance would be expected — but is it *statistically real*
and *separable from the underlying distress*?

We test:
- **Signal arm** — forward returns (1/3/6/12 months) starting the day AFTER the
  reverse-split effective date.
- **Null hypothesis** — H0: mean post-RS return = 0.  If distress is the real driver,
  returns could be negative but not due to the RS event *per se*.
- **Distress confound** — named prominently: we cannot isolate "RS effect" from "distress
  continuation" without a distress-matched control.  Even a significant negative t-stat
  may reflect ongoing deterioration, not a tradable signal.

Functions:
- ``summarize_horizon``        — HAC t-test on post-RS forward returns.
- ``cost_adjusted_return``     — net annualised return after costs (one-way bps).
- ``rs_vs_random_edge``        — positive / null control comparison.
- ``build_random_baseline``    — sample non-RS forward returns from the same price panel.
- ``compare_horizons``         — per-horizon RS vs random baseline comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# HAC t-test (Newey-West) on forward returns
# ---------------------------------------------------------------------------

def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``x``."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
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


def summarize_horizon(
    events: pd.DataFrame,
    col: str = "ret_3m",
) -> dict:
    """Headline statistics for one horizon's forward return across all RS events.

    Returns n, mean_pct, median_pct, std_pct, win_rate (fraction > 0), HAC t-stat.
    """
    r = events[col].dropna().to_numpy(dtype=float)
    n = r.size
    if n < 2:
        return {k: float("nan") for k in ("n", "mean_pct", "median_pct", "std_pct",
                                           "win_rate", "tstat")}
    return {
        "n": int(n),
        "mean_pct": float(r.mean() * 100),
        "median_pct": float(np.median(r) * 100),
        "std_pct": float(r.std(ddof=1) * 100),
        "win_rate": float((r > 0).mean()),
        "tstat": _hac_tstat(r),
    }


# ---------------------------------------------------------------------------
# Random baseline — non-RS dates from the same price panel
# ---------------------------------------------------------------------------

def build_random_baseline(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: tuple[int, ...] = (21, 63, 126, 252),
    n_control_per_event: int = 5,
    seed: int = 250,
) -> pd.DataFrame:
    """Sample random forward returns from the price panel (non-RS dates).

    This baseline answers: "how does the average stock behave over the same
    look-ahead horizon if entered on a random date?"  It is the unconditional
    market return, not a distress-matched control — so any negative difference
    between RS events and this baseline conflates the RS signal with the
    distress that typically precedes RS events.

    We name this limitation explicitly and do not claim it separates the two.
    """
    rng = np.random.default_rng(seed)
    horizon_cols = ("ret_1m", "ret_3m", "ret_6m", "ret_12m")
    max_h = max(horizon_days)

    # Build exclusion zones around RS event dates (per ticker where we have prices)
    exclude_positions: dict[str, set[int]] = {}
    for _, row in events.iterrows():
        tkr = row["ticker"]
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        pos = int(np.searchsorted(close_s.index, pd.Timestamp(row["event_date"])))
        if tkr not in exclude_positions:
            exclude_positions[tkr] = set()
        for offset in range(-126, 126 + 1):
            p = pos + offset
            if 0 <= p < len(close_s):
                exclude_positions[tkr].add(p)

    rows = []
    # Sample from all available tickers in prices, not just RS tickers
    for tkr in prices.columns:
        close_s = prices[tkr].dropna()
        n = len(close_s)
        excluded = exclude_positions.get(tkr, set())
        eligible = [i for i in range(n - max_h) if i not in excluded]
        if len(eligible) < 3:
            continue
        chosen = rng.choice(
            eligible,
            size=min(n_control_per_event, len(eligible)),
            replace=False,
        )
        for pos in chosen:
            ev_close = close_s.iloc[pos]
            if ev_close <= 0 or not np.isfinite(ev_close):
                continue
            fwd: dict[str, float] = {}
            for h, col in zip(horizon_days, horizon_cols):
                end_pos = pos + h
                if end_pos >= n:
                    fwd[col] = float("nan")
                else:
                    fwd[col] = float(close_s.iloc[end_pos] / ev_close - 1.0)
            rows.append(
                {
                    "ticker": tkr,
                    "event_date": close_s.index[pos],
                    "ratio": float("nan"),
                    "ret_1m": fwd["ret_1m"],
                    "ret_3m": fwd["ret_3m"],
                    "ret_6m": fwd["ret_6m"],
                    "ret_12m": fwd["ret_12m"],
                    "is_rs": False,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def compare_horizons(
    rs_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    cols: tuple[str, ...] = ("ret_1m", "ret_3m", "ret_6m", "ret_12m"),
) -> pd.DataFrame:
    """Per-horizon comparison: RS events vs random baseline.

    Returns a DataFrame with one row per horizon showing n_rs, n_base,
    mean_rs_pct, mean_base_pct, diff_pct (RS − base), tstat_rs, tstat_diff.
    The inference bar for the desk is |tstat| ≥ 2.
    """
    labels = ["1m (~21d)", "3m (~63d)", "6m (~126d)", "12m (~252d)"]
    rows = []
    for col, lbl in zip(cols, labels):
        sp = rs_events[col].dropna().to_numpy(dtype=float)
        bs = baseline_events[col].dropna().to_numpy(dtype=float)

        diff_mean = float(sp.mean() - bs.mean()) if sp.size and bs.size else float("nan")
        diff_series = sp - bs.mean() if sp.size and bs.size else np.array([])
        tstat_diff = _hac_tstat(diff_series)

        rows.append(
            {
                "horizon": lbl,
                "n_rs": int(sp.size),
                "n_base": int(bs.size),
                "mean_rs_pct": float(sp.mean() * 100) if sp.size else float("nan"),
                "mean_base_pct": float(bs.mean() * 100) if bs.size else float("nan"),
                "diff_pct": float(diff_mean * 100) if not np.isnan(diff_mean) else float("nan"),
                "tstat_rs": _hac_tstat(sp),
                "tstat_diff": tstat_diff,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tradability: cost-adjusted annualised return
# ---------------------------------------------------------------------------

def cost_adjusted_return(
    mean_return: float,
    horizon_days: int,
    cost_bps: float = 10.0,
) -> float:
    """Deduct a round-trip transaction cost and annualise.

    For reverse-split names we use 10 bps (higher than large-caps) to reflect
    wider bid-ask spreads typical of distressed small-caps.

    Returns annualised net return as a fraction (0.05 = 5%/yr).
    """
    net = mean_return - cost_bps * 1e-4
    ann_factor = TRADING_DAYS_PER_YEAR / horizon_days
    return float((1.0 + net) ** ann_factor - 1.0)


# ---------------------------------------------------------------------------
# Positive / null control helper
# ---------------------------------------------------------------------------

def rs_vs_random_edge(
    events_planted: pd.DataFrame,
    events_null: pd.DataFrame,
    col: str = "ret_3m",
) -> tuple[float, float]:
    """Returns (planted_mean_pct, null_mean_pct) to verify the engine.

    When drift_bps < 0 the planted tape should show a more negative mean return
    than the null tape — confirming the engine recovers a negative drift.
    """
    planted = events_planted[col].dropna().to_numpy(dtype=float)
    null = events_null[col].dropna().to_numpy(dtype=float)
    return float(planted.mean() * 100), float(null.mean() * 100)
