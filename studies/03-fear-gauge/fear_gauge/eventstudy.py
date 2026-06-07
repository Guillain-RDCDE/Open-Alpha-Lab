"""Event study: what does the S&P 500 do, on average, around a VIX event?

The centrepiece, and a deliberate mirror of Study 02 — same machinery, different
trigger. For every day the gauge fires (t=0: VIX crosses 30, or jumps +30%), we
line up the *S&P* price path from a few days before to many days after, normalise
to the entry close, and average across events. The output is the picture the
Altucher chart draws ("after a VIX spike the market tends to bounce") — but here
it comes with dispersion, a t-stat, and the *pre*-event leg that reveals you are
buying into a selloff already in progress.

The signal lives on the gauge frame; the path is measured on the market frame. As
long as the two come from :func:`fear_gauge.data.aligned`, they share an index and
line up 1:1 with no look-ahead.

Whether the conditional path is actually an *edge* is decided in
:mod:`fear_gauge.benchmark`, against the path from a random day — never here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_positions(signal: pd.Series) -> np.ndarray:
    """Integer positions where ``signal`` is True."""
    return np.flatnonzero(signal.fillna(False).to_numpy())


def forward_matrix(
    market: pd.DataFrame,
    positions: np.ndarray,
    horizon: int = 21,
    pre: int = 5,
) -> pd.DataFrame:
    """Per-event cumulative S&P return paths, normalised to the entry close (t=0).

    Returns a DataFrame indexed by event date, columns ``-pre ... +horizon`` (in
    trading days relative to the event), values = ``close[t+k]/close[t] - 1``.
    Events too close to either edge are dropped (no look-ahead, no partial paths).

    The default ``horizon=21`` ≈ one trading month, to reproduce the famous
    "+1 month" column of the spike chart exactly.
    """
    close = market["Close"].to_numpy()
    dates = market.index
    n = len(close)
    offsets = np.arange(-pre, horizon + 1)

    rows = {}
    for p in positions:
        if p - pre < 0 or p + horizon >= n:
            continue
        base = close[p]
        rows[dates[p]] = close[p - pre : p + horizon + 1] / base - 1.0

    mat = pd.DataFrame.from_dict(rows, orient="index", columns=offsets)
    mat.index.name = "event_date"
    mat.columns.name = "rel_day"
    return mat


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    """Collapse a forward matrix to per-horizon statistics.

    Columns: ``mean, median, std, pct_positive, tstat, n``. The t-stat is the
    cross-event mean over its standard error — a rough gauge that ignores event
    clustering (handled honestly in :mod:`fear_gauge.robustness`).
    """
    n = len(matrix)
    mean = matrix.mean()
    std = matrix.std(ddof=1)
    se = std / np.sqrt(max(n, 1))
    out = pd.DataFrame(
        {
            "mean": mean,
            "median": matrix.median(),
            "std": std,
            "pct_positive": (matrix > 0).mean(),
            "tstat": mean / se.replace(0.0, np.nan),
            "n": n,
        }
    )
    out.index.name = "rel_day"
    return out


def event_study(
    market: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 21,
    pre: int = 5,
) -> dict:
    """Convenience wrapper: positions -> matrix -> summary.

    ``signal`` is a boolean gauge series (e.g. from :mod:`fear_gauge.triggers`),
    usually debounced into fresh events via ``triggers.first_crossings`` first, and
    reindexed onto the market index. Returns ``matrix``, ``summary``, ``n_events``.
    """
    sig = signal.reindex(market.index).fillna(False)
    pos = event_positions(sig)
    matrix = forward_matrix(market, pos, horizon=horizon, pre=pre)
    return {"matrix": matrix, "summary": summarize(matrix), "n_events": len(matrix)}


# Horizons that match the chart this study rebuts: +1 day, +1 week, +1 month.
CHART_HORIZONS = {"+1 day": 1, "+1 week": 5, "+1 month": 21}
