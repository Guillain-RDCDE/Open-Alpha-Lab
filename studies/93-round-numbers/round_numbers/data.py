"""Data layer for Study 93 (Round-Numbers).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a *deterministic, offline* generator. A magnetism knob
  (``magnet``) plants the only thing a round-number fade can possibly harvest: a
  **reversal near round levels**. ``magnet=0`` is a constant-drift geometric random
  walk — price drifts through the round numbers with no special behaviour, so a fade
  cannot collect a bounce. ``magnet>0`` makes price *reflect* off the band around each
  round level (when within ``band`` of a multiple of ``grid``, a fraction ``magnet`` of
  days reverse the move away from the level), so a fade genuinely banks the bounce. This
  is the study's null (and its positive control) in a bottle.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet).
  The adjustment mode is a *decision*: clustering of *traded* price levels (Harris 1991)
  is a property of the **as-traded nominal** tape, so individual names are pulled in
  ``raw`` mode (split factors multiplied back up to the price actually quoted that day);
  the spot index ``^GSPC`` and the ETF ``SPY`` carry no split history of consequence, so
  ``split_only`` *is* the traded level for them. Cache-first; the network is touched only
  on a cache miss.

No look-ahead is baked in here — the one-day execution lag lives in ``strategy.py``
(the round-level proximity is known at close *t*, the fade is entered at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The basket: an index milestone (Donaldson & Kim 1993), an index ETF, and two liquid
# single names whose nominal price has roamed across many whole-dollar levels. Each entry
# is (ticker, adjustment_mode, round_grid) — the grid is the spacing of the "round" levels
# the believer would fade (100-pt index milestones; $10 levels for the rest).
BASKET = [
    ("^GSPC", "split_only", 100.0),
    ("SPY", "split_only", 10.0),
    ("AAPL", "raw", 10.0),
    ("MSFT", "raw", 10.0),
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 14000,
    magnet: float = 0.0,
    grid: float = 100.0,
    band: float = 0.006,
    daily_vol: float = 0.008,
    drift: float = 0.0003,
    start: str = "1970-01-02",
    seed: int = 93,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable amount of round-number magnetism.

    - ``magnet = 0``  → a single constant-drift geometric random walk. Price wanders through
      the round levels with no special behaviour near them, so a fade cannot collect a
      bounce — its expected edge is zero. This is the study's null.
    - ``magnet > 0``  → a *reflective barrier* at each round level. When the close is within
      ``band`` (a fraction of price) of a multiple of ``grid``, a fraction ``magnet`` of days
      reverse the day's move so price is pushed back *away* from the level. A fade entered on
      proximity should now bank a positive, significant edge — the positive control that
      proves the harness can detect a planted reversal when one exists.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters (including the realised fraction of days spent in-band).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    moves = rng.normal(drift, daily_vol, n_days)
    bounce = rng.random(n_days)

    c = np.empty(n_days)
    c[0] = 1000.0
    in_band = np.zeros(n_days, dtype=bool)
    for t in range(1, n_days):
        m = moves[t]
        lvl = round(c[t - 1] / grid) * grid
        d = (c[t - 1] - lvl) / c[t - 1]  # signed distance to nearest round level
        if abs(d) < band:
            in_band[t] = True
            if magnet > 0 and bounce[t] < magnet:
                # Reflect away from the level: move's sign set to -sign(d), plus the base drift.
                m = -abs(m) * np.sign(d) + drift
        c[t] = c[t - 1] * (1.0 + m)

    frame = pd.DataFrame({"close": c}, index=idx)
    frame.index.name = "Date"
    truth = {
        "magnet": magnet,
        "grid": grid,
        "band": band,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "seed": seed,
        "frac_in_band": float(in_band.mean()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "^GSPC",
    start: str = "1990-01-01",
    end: str | None = None,
    mode: str = "split_only",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``mode`` carries the adjustment decision: ``"raw"`` reconstructs the
    **as-traded nominal** price (the object the price-clustering literature studies) for
    single names that have split many times; ``"split_only"`` is the natural traded level
    for the spot index ``^GSPC`` and the ETF ``SPY``. Returns a frame with a single
    lower-case ``close`` column, indexed by date.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    close = raw["Close"].rename("close")
    return close.to_frame()


def load_basket(
    as_of: str | None = None, use_cache: bool = True
) -> list[tuple[str, float, pd.Series]]:
    """Load every name in :data:`BASKET` as ``(ticker, grid, close)`` triples.

    Each close is truncated at ``as_of`` (inclusive) when given, so a headline run pins to
    an explicit date and never includes a partial bar. Cache-first.
    """
    out: list[tuple[str, float, pd.Series]] = []
    for ticker, mode, grid in BASKET:
        close = load_real(ticker, mode=mode, use_cache=use_cache)["close"].dropna()
        if as_of is not None:
            close = close.loc[:as_of]
        out.append((ticker, grid, close))
    return out


def fingerprint(closes: list[np.ndarray] | pd.DataFrame) -> str:
    """A short content fingerprint of one or more close arrays, for the as-of stamp."""
    h = hashlib.sha1()
    if isinstance(closes, pd.DataFrame):
        closes = [closes["close"].to_numpy()]
    for arr in closes:
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=float)).tobytes())
    return h.hexdigest()[:12]
