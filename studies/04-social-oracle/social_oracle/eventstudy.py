"""Event study: what does a name do, on average, around the moment it's mentioned?

The centrepiece. For every event (``t=0`` = the mention session) we line up the
name's **abnormal** return path — the name minus its market, ``r_cc - r_mkt`` — from
a few sessions before to many sessions after, and average across events. The output
is the picture the "signal dashboard" implies ("after she names it, it runs") — but
here it carries the *pre*-event leg that reveals how much of the move already
happened before the tweet, and the *post*-event leg that reveals the **fade**.

We use the textbook event-study convention: the path is the **cumulative abnormal
return** (CAR) — the *sum* of daily abnormal returns relative to ``t=0`` — not a
compounded price ratio. Over the short windows that matter here the two are
numerically close, but CAR is the right object: it's additive, it centres cleanly
at zero, and pop-then-fade reads straight off it as "up into 0, down after".

Whether that path is an *edge* is decided in :mod:`social_oracle.benchmark`, against
a random day in the same universe — never here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def abnormal(frame: pd.DataFrame) -> np.ndarray:
    """Daily abnormal return ``r_cc - r_mkt`` as a NaN-free array (NaN -> 0)."""
    ab = (frame["r_cc"] - frame["r_mkt"]).to_numpy()
    return np.nan_to_num(ab, nan=0.0)


def _prefix(frame: pd.DataFrame) -> np.ndarray:
    """Prefix sums of abnormal return; ``prefix[i] = sum(ab[:i])`` (len n+1)."""
    return np.concatenate([[0.0], np.cumsum(abnormal(frame))])


def forward_matrix(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizon: int = 21,
    pre: int = 5,
) -> pd.DataFrame:
    """Per-event cumulative *abnormal* return paths, centred at the mention (t=0).

    Returns a DataFrame indexed by ``(ticker, entry_date)``, columns
    ``-pre ... +horizon`` (sessions relative to the mention), values = CAR measured
    from ``t=0``: ``sum(abnormal over (t, t+k])`` for ``k>0`` and the negative run-up
    for ``k<0``. Events without a full window are assumed already filtered by
    :func:`social_oracle.mentions.to_events`; any that slip through are skipped.
    """
    offsets = np.arange(-pre, horizon + 1)
    rows = {}
    for t, grp in events.groupby("ticker"):
        frame = panel.get(t)
        if frame is None:
            continue
        prefix = _prefix(frame)
        n = len(frame)
        dates = frame.index
        for p in grp["entry_pos"].to_numpy():
            p = int(p)
            if p - pre < 0 or p + horizon >= n:
                continue
            base = prefix[p + 1]
            rows[(t, dates[p])] = prefix[p + 1 - pre: p + 2 + horizon] - base

    mat = pd.DataFrame.from_dict(rows, orient="index", columns=offsets)
    mat.index = pd.MultiIndex.from_tuples(mat.index, names=["ticker", "event_date"])
    mat.columns.name = "rel_day"
    return mat


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    """Collapse a forward matrix to per-session statistics.

    Columns: ``mean, median, std, pct_positive, tstat, n``. The t-stat is the
    cross-event mean over its standard error — a rough gauge that treats events as
    independent (clustering is handled honestly in
    :mod:`social_oracle.robustness`).
    """
    n = len(matrix)
    mean = matrix.mean()
    std = matrix.std(ddof=1)
    se = std / np.sqrt(max(n, 1))
    out = pd.DataFrame({
        "mean": mean,
        "median": matrix.median(),
        "std": std,
        "pct_positive": (matrix > 0).mean(),
        "tstat": mean / se.replace(0.0, np.nan),
        "n": n,
    })
    out.index.name = "rel_day"
    return out


def event_study(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizon: int = 21,
    pre: int = 5,
) -> dict:
    """Convenience wrapper: events -> abnormal-path matrix -> summary.

    Returns ``matrix``, ``summary``, ``n_events``.
    """
    matrix = forward_matrix(panel, events, horizon=horizon, pre=pre)
    return {"matrix": matrix, "summary": summarize(matrix), "n_events": len(matrix)}


# Horizons that match how social-signal dashboards quote performance: next day, week, month.
CHART_HORIZONS = {"+1 day": 1, "+1 week": 5, "+1 month": 21}
