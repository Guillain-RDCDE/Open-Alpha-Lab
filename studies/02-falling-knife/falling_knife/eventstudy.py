"""Event study: what does NDX do, on average, around a -3% event?

This is the centrepiece. For every day a trigger fires (t=0), we line up the
price path from a few days before to many days after, normalise to the entry
close, and average across all events. The result is a single curve a beginner
can *see* ("after a -3% day the market tends to bounce / keep falling") and that
a quant respects (mean forward return per horizon, with dispersion and a t-stat).

Crucially this measures the *conditional* path. Whether that path is actually an
edge is decided in :mod:`falling_knife.benchmark`, by comparing it to the path
from a random day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_positions(signal: pd.Series) -> np.ndarray:
    """Integer positions where ``signal`` is True."""
    return np.flatnonzero(signal.fillna(False).to_numpy())


def forward_matrix(
    ohlc: pd.DataFrame,
    positions: np.ndarray,
    horizon: int = 20,
    pre: int = 5,
) -> pd.DataFrame:
    """Per-event cumulative return paths, normalised to the entry close (t=0).

    Returns a DataFrame indexed by event date, columns ``-pre ... +horizon`` (in
    trading days relative to the event), values = ``close[t+k]/close[t] - 1``.
    Events too close to either edge of the sample are dropped (no look-ahead, no
    partial paths).
    """
    close = ohlc["Close"].to_numpy()
    dates = ohlc.index
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
    cross-event mean over its standard error (``mean / (std/sqrt(n))``) — a rough
    significance gauge that ignores the clustering of events (handled honestly in
    :mod:`falling_knife.robustness`).
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
    ohlc: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 20,
    pre: int = 5,
) -> dict:
    """Convenience wrapper: positions -> matrix -> summary.

    Returns a dict with keys ``matrix`` (per-event paths) and ``summary``
    (per-horizon stats). ``signal`` should usually be debounced into fresh events
    via :func:`falling_knife.triggers.first_crossings` first.
    """
    pos = event_positions(signal)
    matrix = forward_matrix(ohlc, pos, horizon=horizon, pre=pre)
    return {"matrix": matrix, "summary": summarize(matrix), "n_events": len(matrix)}
