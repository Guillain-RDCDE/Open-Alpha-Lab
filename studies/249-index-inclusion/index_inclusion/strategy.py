"""Strategy and honest controls — Study 249 (Index-Inclusion).

The claim: a stock added to the S&P 500 pops from announcement to effective date as
passive index funds pre-position.  Buying on announcement and selling on the effective
date captures this pop.  A secondary hypothesis: the pop subsequently reverses as index
buying pressure fades (the 'give-back').

We test both:

**Pop window** — buy-and-hold return from announce_date close to effective_date close.
The signal is the public S&P press release (available same day market opens).  Entry
is at the close on announce_date + 1 in a realistic implementation.  We measure close-
to-close as the gross upper bound, then apply a cost haircut for the tradability verdict.

**Give-back window** — buy-and-hold return from effective_date + 1 at 1m (21 trading
days) and 3m (63 trading days).  A statistically significant *negative* return here
would support the "front-run reversal" narrative: the buying pressure of index funds
is priced in by effective date, leaving a supply overhang as arb books unwind.

No look-ahead bias: announcement is a public, datable corporate event.

Functions:
- ``summarize_window``     — HAC t-stat, mean, win-rate for one return column.
- ``compare_pop_giveback`` — side-by-side pop vs give-back statistics table.
- ``cost_adjusted_pop``    — net return after round-trip costs (bid-ask, market impact).
- ``pop_vs_random_edge``   — positive control check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# HAC t-statistic (Newey-West)
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


# ---------------------------------------------------------------------------
# Per-window statistics
# ---------------------------------------------------------------------------
def summarize_window(
    events: pd.DataFrame,
    col: str = "ret_pop",
) -> dict:
    """Headline statistics for one return window across all inclusion events.

    Returns n, mean (%), median (%), std (%), win-rate (fraction > 0), HAC t-stat.
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
# Pop vs give-back summary table
# ---------------------------------------------------------------------------
def compare_pop_giveback(events: pd.DataFrame) -> pd.DataFrame:
    """Compare the pop window vs the give-back windows.

    Returns a tidy DataFrame with one row per window (pop, 1m give-back, 3m give-back)
    showing mean, median, std, win-rate, and HAC t-stat.
    """
    windows = [
        ("ret_pop",          "Pop (announce→effective)"),
        ("ret_giveback_1m",  "Give-back 1m (~21d after eff.)"),
        ("ret_giveback_3m",  "Give-back 3m (~63d after eff.)"),
    ]
    rows = []
    for col, lbl in windows:
        if col not in events.columns:
            continue
        s = summarize_window(events, col=col)
        rows.append({"window": lbl, **s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time-series evolution: has the pop decayed?
# ---------------------------------------------------------------------------
def pop_by_period(
    events: pd.DataFrame,
    cutoffs: tuple[str, ...] = ("2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"),
) -> pd.DataFrame:
    """Split the event table into sub-periods and compute mean pop per period.

    Helps test whether the pop has decayed over time as passive indexing grew and
    index-arbitrage strategies became crowded.

    Returns one row per period with: period_label, n, mean_pop_pct, tstat_pop.
    """
    df = events.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df = df.sort_values("effective_date")

    cuts = [pd.Timestamp("1990-01-01")] + [pd.Timestamp(c) for c in cutoffs] + [pd.Timestamp("2099-12-31")]
    labels_list = []
    for i in range(len(cuts) - 1):
        lo = cuts[i].strftime("%Y")
        hi = cuts[i + 1].strftime("%Y")
        labels_list.append(f"{lo}–{hi}")

    rows = []
    for i, lbl in enumerate(labels_list):
        mask = (df["effective_date"] >= cuts[i]) & (df["effective_date"] < cuts[i + 1])
        sub = df.loc[mask, "ret_pop"].dropna().to_numpy(dtype=float)
        if sub.size < 3:
            continue
        rows.append(
            {
                "period": lbl,
                "n": int(sub.size),
                "mean_pop_pct": float(sub.mean() * 100),
                "tstat_pop": _hac_tstat(sub),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost-adjusted return (one-way entry + exit)
# ---------------------------------------------------------------------------
def cost_adjusted_pop(
    mean_return: float,
    cost_bps: float = 10.0,
) -> float:
    """Deduct a round-trip transaction cost from the pop and return the net.

    The announce→effective window is typically 3–5 business days.  Entry may face
    slippage if many arbs crowd in on announcement day; borrow is not applicable
    (long-only trade).  A conservative 10 bps round-trip covers institutional
    equity-desk costs; retail would be higher.

    Returns the net return (fraction), e.g. 0.015 → +1.5% net.
    """
    return float(mean_return - cost_bps * 1e-4)


# ---------------------------------------------------------------------------
# Positive control — synthetic check
# ---------------------------------------------------------------------------
def pop_vs_random_edge(
    events_planted: pd.DataFrame,
    events_null: pd.DataFrame,
    col: str = "ret_pop",
) -> tuple[float, float]:
    """Returns (planted_mean_pct, null_mean_pct) for one window.

    The planted tape (pop_bps > 0) should show a higher mean pop than the null
    (pop_bps = 0).  This confirms the engine recovers an effect when one is planted.
    """
    planted = events_planted[col].dropna().to_numpy(dtype=float)
    null = events_null[col].dropna().to_numpy(dtype=float)
    return float(planted.mean() * 100), float(null.mean() * 100)
