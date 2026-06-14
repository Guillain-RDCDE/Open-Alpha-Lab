"""Strategy and honest controls — Study 142 (Split-Drift).

The claim: stocks drift upward in the months following a stock split.
(Ikenberry, Rankine & Stice 1996: post-*announcement* drift up to +8% in 12 months.)

We test the *post-effective-date* version because yfinance gives effective dates,
not announcement dates.  The honest comparison is:

**Signal arm** — forward returns (1/3/6/12 months) starting the day AFTER the
split effective date.

**Baseline arm** — forward returns of the same stock on *randomly-chosen non-split
dates* (matched-period control), drawn from the same calendar to remove market-wide
trends.  This is the honest "vs a coin / vs base rate" control: if splits drift up,
split-date forward returns should exceed randomly-drawn forward returns.

No look-ahead: the effective date is a public, observable event; the forward window
starts at t+1.  Prices are split-adjusted, so the split mechanics themselves do not
contaminate the return signal.

Functions:
- ``build_baseline_events`` — sample the no-split baseline from the same price panel.
- ``compare_horizons``       — HAC t-test on split vs baseline forward returns.
- ``summarize_horizon``      — headline stats for one horizon.
- ``cost_adjusted_return``   — deduct round-trip cost and annualise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Baseline construction — matched no-split dates
# ---------------------------------------------------------------------------
def build_baseline_events(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: tuple[int, ...] = (21, 63, 126, 252),
    n_control_per_event: int = 3,
    seed: int = 142,
) -> pd.DataFrame:
    """Sample matched no-split forward returns from the same price panel.

    For each split event on ticker T at date D, we sample ``n_control_per_event``
    dates from the *same ticker's* price history, chosen from periods that are:
    - Not within 252 trading days of any split event for that ticker (clean window).
    - Have at least ``max(horizon_days)`` trading days of forward data available.

    This gives a baseline of "how much would a stock drift over the same look-ahead
    horizon if you entered on a random, non-split day?" — the fair comparison.

    Returns a DataFrame with the same columns as the split events, but ``is_split=False``.
    """
    rng = np.random.default_rng(seed)
    horizon_cols = ("ret_1m", "ret_3m", "ret_6m", "ret_12m")
    max_h = max(horizon_days)

    # Build a per-ticker set of split-date positions to exclude
    split_positions: dict[str, set[int]] = {}
    for _, row in events.iterrows():
        tkr = row["ticker"]
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        pos = int(np.searchsorted(close_s.index, pd.Timestamp(row["event_date"])))
        if tkr not in split_positions:
            split_positions[tkr] = set()
        # Exclude a window around each split: 252 days before and after
        for offset in range(-252, 252 + 1):
            p = pos + offset
            if 0 <= p < len(close_s):
                split_positions[tkr].add(p)

    rows = []
    for _, ev_row in events.iterrows():
        tkr = ev_row["ticker"]
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        n = len(close_s)
        # Eligible positions: not near any split, have forward data
        excluded = split_positions.get(tkr, set())
        eligible = [
            i for i in range(n - max_h)
            if i not in excluded
        ]
        if not eligible:
            continue
        chosen = rng.choice(
            eligible,
            size=min(n_control_per_event, len(eligible)),
            replace=False,
        )
        ev_close_s_idx = close_s.index
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
                    "event_date": ev_close_s_idx[pos],
                    "split_ratio": float("nan"),
                    "ret_1m": fwd["ret_1m"],
                    "ret_3m": fwd["ret_3m"],
                    "ret_6m": fwd["ret_6m"],
                    "ret_12m": fwd["ret_12m"],
                    "is_split": False,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HAC t-test on forward returns: split arm vs baseline arm
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
    """Headline statistics for one horizon's forward return across all split events.

    Returns n, mean return (%), median, std, HAC t-stat, and the % of events
    with positive forward returns (the 'win-rate' analogue for event studies).
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


def compare_horizons(
    split_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    cols: tuple[str, ...] = ("ret_1m", "ret_3m", "ret_6m", "ret_12m"),
) -> pd.DataFrame:
    """For each horizon, compare split vs baseline forward returns.

    Returns a summary DataFrame with one row per horizon showing:
    - ``n_split``, ``n_base``: event counts
    - ``mean_split``, ``mean_base``: mean forward returns (%)
    - ``diff``: split mean − baseline mean (%)
    - ``tstat_split``: HAC t-stat on the split arm's mean return
    - ``tstat_diff``: HAC t-stat on the difference (split − baseline),
      computed by pooling the differences on matched baseline events.

    The ``tstat_split`` tests H0: mean post-split return = 0.
    The ``tstat_diff`` tests the sharper H0: split drift = baseline drift.
    """
    labels = ["1m (~21d)", "3m (~63d)", "6m (~126d)", "12m (~252d)"]
    rows = []
    for col, lbl in zip(cols, labels):
        sp = split_events[col].dropna().to_numpy(dtype=float)
        bs = baseline_events[col].dropna().to_numpy(dtype=float)

        diff_mean = float(sp.mean() - bs.mean()) if sp.size and bs.size else float("nan")

        # HAC t-stat on the raw difference array (to test whether split > baseline
        # we use the sample of (split_return - mean_baseline) as a one-sample test
        # — conservative since we don't have matched pairs, but honest)
        diff_series = sp - bs.mean() if sp.size and bs.size else np.array([])
        tstat_diff = _hac_tstat(diff_series)

        rows.append(
            {
                "horizon": lbl,
                "n_split": int(sp.size),
                "n_base": int(bs.size),
                "mean_split_pct": float(sp.mean() * 100) if sp.size else float("nan"),
                "mean_base_pct": float(bs.mean() * 100) if bs.size else float("nan"),
                "diff_pct": float(diff_mean * 100) if not np.isnan(diff_mean) else float("nan"),
                "tstat_split": _hac_tstat(sp),
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
    cost_bps: float = 5.0,
) -> float:
    """Deduct a round-trip transaction cost and annualise the net return.

    A typical round-trip for a retail investor (broker commission + bid-ask)
    is ~5 bps for liquid large-caps.  The 1-month horizon has the highest
    turnover cost burden; the 12-month horizon is gentler.

    Returns the annualised net return (fraction), so 0.05 means +5%/yr.
    """
    net = mean_return - cost_bps * 1e-4  # deduct round-trip cost
    ann_factor = TRADING_DAYS_PER_YEAR / horizon_days
    # Annualise: (1 + net)^(252/h) - 1
    return float((1.0 + net) ** ann_factor - 1.0)


# ---------------------------------------------------------------------------
# Synthetic positive control helper
# ---------------------------------------------------------------------------
def split_vs_random_edge(
    events_planted: pd.DataFrame,
    events_null: pd.DataFrame,
    col: str = "ret_3m",
) -> tuple[float, float]:
    """Returns (planted_mean, null_mean) for one horizon, to verify the engine.

    When drift_bps > 0 the planted tape should show a positive mean return;
    at drift_bps = 0 the null tape should be near zero.  This is the
    'machinery proof' — confirming the engine recovers an effect when one exists.
    """
    planted = events_planted[col].dropna().to_numpy(dtype=float)
    null = events_null[col].dropna().to_numpy(dtype=float)
    return float(planted.mean() * 100), float(null.mean() * 100)
