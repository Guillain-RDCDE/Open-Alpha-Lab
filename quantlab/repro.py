"""Reproducibility stamp — turn silent data drift into a verifiable match.

The headline numbers in each study come from live Yahoo! Finance data, which gets
*revised* and *extended* over time. Two honest re-runs months apart can therefore
disagree — silently — and the reader has no way to tell a real correction from a
data artefact. This module makes that impossible. Every verify-script prints, above
its tables, a one-line **data stamp** carrying:

  * an **as-of** date — the sample is sliced to end here, so it never creeps as new
    sessions arrive (the difference between "1990–2026" meaning the same thing today
    and next year);
  * a **fingerprint** — a short content hash of the exact input series.

A reader who reruns and gets the *same* fingerprint holds, byte for byte, the data
behind the published verdict. A *different* fingerprint flags drift loudly — before
they puzzle over a number that moved on its own.

Pure stdlib + pandas; no network, no other quantlab dependency, so it is safe to
import from any study's example scripts.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

# The frozen as-of for every headline run. Bump this deliberately (and re-quote the
# fingerprints) when you want the published numbers to advance — never let the
# sample grow by accident.
DEFAULT_AS_OF = "2026-06-01"


def as_of(df: pd.DataFrame, asof: str | None = DEFAULT_AS_OF) -> pd.DataFrame:
    """Slice ``df`` to rows on or before ``asof`` (no-op if ``asof`` is None).

    Use this on every fetched frame *before* computing a headline number, so the
    sample is pinned and reruns can't drift just because the calendar moved.
    """
    if asof is None:
        return df
    return df[df.index <= pd.Timestamp(asof)].copy()


def fingerprint(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    round_dp: int = 6,
) -> str:
    """A short, stable content hash of ``df`` over ``cols`` (default: numeric cols).

    Canonical by construction — independent of column order, pandas/NumPy version
    and float formatting — so the same data always yields the same 12-hex digest
    and any change to a value, a date or the row count changes it.
    """
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cols = sorted(cols)

    h = hashlib.sha256()
    # 1) the index, as canonical ISO dates — pins the sample's dates and length.
    for ts in df.index:
        h.update(str(pd.Timestamp(ts).date()).encode())
        h.update(b"\x1f")
    # 2) each column's rounded values, NaN canonicalised — pins the content.
    for c in cols:
        h.update(b"\x1e")
        h.update(str(c).encode())
        for v in df[c].to_numpy():
            token = "nan" if pd.isna(v) else f"{round(float(v), round_dp):.{round_dp}f}"
            h.update(token.encode())
            h.update(b",")
    return h.hexdigest()[:12]


def data_stamp(
    name: str,
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    asof: str | None = DEFAULT_AS_OF,
) -> str:
    """One-line provenance banner: ``[data] <name>: N rows lo->hi as-of A  fp=...``.

    Print this above any table of headline numbers. ``df`` should already be sliced
    with :func:`as_of` (the banner echoes ``asof`` for the record either way).
    """
    fp = fingerprint(df, cols=cols)
    lo, hi = df.index.min().date(), df.index.max().date()
    asof_s = f"  as-of {asof}" if asof else ""
    return f"[data] {name}: {len(df):,} rows  {lo} -> {hi}{asof_s}  fingerprint={fp}"
