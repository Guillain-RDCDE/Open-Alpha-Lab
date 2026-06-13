"""Data layer for Study 100 (Melting-Ice).

Two kinds of tape, one shape (a daily frame with a ``close`` column):

- ``synthetic_path`` — a *deterministic, offline* generator with a ``path`` knob that
  plants the only two things that decide a leveraged ETF's fate:

  - ``path="flat_highvol"`` — a high-volatility, **trend-less** tape (zero drift, fat
    daily moves). A 3x daily-rebalanced sleeve must *decay* here: with no trend to
    compound, the variance-drag term dominates and the levered close ends *below* the
    naive "3x the period return" line. This is the folklore's home turf.
  - ``path="uptrend_lowvol"`` — a steady, **low-volatility uptrend**. A 3x sleeve must
    *compound ahead* here: the trend term beats the small drag, so the levered close
    ends *above* naive 3x. This is the regime the 2010s NASDAQ actually was.

  Together they are the study's two-sided control: the harness must report the *sign* of
  the gap correctly on both — decay on one, compounding on the other.

- ``load_real`` — the real daily total-return tape via :mod:`quantlab.data` (Yahoo,
  cached parquet). Total return folds dividends back in so the levered sleeve and its
  underlying are compared on the same, dividend-inclusive footing. Cache-first; the
  network is touched only on a cache miss.

The constant-leverage rebalancing identity (the simulator) lives in ``strategy.py``;
this module only produces the underlying close series it consumes.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The real (underlying, levered-ETF) pairs this study runs, with the leverage and the
# honest inception window of the levered fund.
PAIRS = {
    "TQQQ": {"underlying": "QQQ", "k": 3.0, "start": "2010-02-11"},
    "UPRO": {"underlying": "SPY", "k": 3.0, "start": "2009-06-25"},
}


# ---------------------------------------------------------------------------
# Synthetic tapes — the deterministic two-sided control
# ---------------------------------------------------------------------------
# Per-path default seeds: the flat tape uses a seed whose *realised* drift is ~0 (a
# genuinely trend-less draw), so the decay demonstration is clean rather than a lucky
# random uptrend masquerading as flat.
_PATH_SEED = {"flat_highvol": 38, "uptrend_lowvol": 100}


def synthetic_path(
    path: str = "flat_highvol",
    n_days: int = 4000,
    start: str = "2009-01-02",
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily underlying close for one of two opposite path types.

    - ``path="flat_highvol"`` → zero-drift, high-vol tape. A 3x daily-rebalanced sleeve
      must end **below** naive-3x: variance drag dominates with no trend to harvest.
    - ``path="uptrend_lowvol"`` → steady low-vol uptrend. A 3x sleeve must end **above**
      naive-3x: the compounding-in-a-trend term beats the small drag.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted drift/vol and which side of the gap is expected to win.
    """
    if seed is None:
        seed = _PATH_SEED.get(path, 100)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    if path == "flat_highvol":
        mu, vol = 0.0, 0.030  # no drift, ~48%/yr vol — the decay regime
        expect = "decay"  # levered close should end BELOW naive 3x
    elif path == "uptrend_lowvol":
        mu, vol = 0.18 / 252.0, 0.008  # ~18%/yr drift, ~13%/yr vol — the compounding regime
        expect = "compound"  # levered close should end ABOVE naive 3x
    else:
        raise ValueError(f"path must be 'flat_highvol' or 'uptrend_lowvol', got {path!r}")

    rets = rng.normal(mu, vol, n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "path": path,
        "n_days": n_days,
        "mu_daily": mu,
        "daily_vol": vol,
        "seed": seed,
        "expect": expect,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "QQQ",
    start: str | None = None,
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``mode="total_return"`` folds dividends back in so a levered ETF and
    its underlying are compared on the same footing. Returns a frame with a single
    lower-case ``close`` column, indexed by date.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    close = raw["Close"].rename("close")
    return close.to_frame()


def load_pair(
    name: str = "TQQQ",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a (underlying, real-levered-ETF) pair aligned on common dates.

    Returns ``(under_frame, letf_frame, meta)`` where both frames carry a single
    ``close`` column on the *intersection* of their dates (from the levered fund's
    inception), and ``meta`` records the leverage and window. The underlying is what the
    simulator levers; the real levered fund is the ground truth the simulator must
    reproduce within tolerance.
    """
    if name not in PAIRS:
        raise ValueError(f"unknown pair {name!r}; known: {list(PAIRS)}")
    spec = PAIRS[name]
    under = load_real(spec["underlying"], mode=mode, use_cache=use_cache)
    letf = load_real(name, mode=mode, use_cache=use_cache)
    common = under.index.intersection(letf.index)
    common = common[common >= pd.Timestamp(spec["start"])]
    if end is not None:
        common = common[common <= pd.Timestamp(end)]
    under, letf = under.loc[common], letf.loc[common]
    meta = {
        "name": name,
        "underlying": spec["underlying"],
        "k": spec["k"],
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "n": len(common),
    }
    return under, letf, meta


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
