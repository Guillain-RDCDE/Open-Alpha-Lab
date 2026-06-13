"""Data layer for Study 94 (Level-Pegging).

Two tapes, one shape (a daily frame of total-return prices, one column per arm):

- ``synthetic_panel`` — a *deterministic, offline* generator that builds a small cap-weighted
  universe and returns the two index curves you'd hold: an **equal-weight** (EW) index that
  rebalances to flat weights, and a **cap-weight** (CW) index that lets winners run. A
  ``mode`` knob plants the answer:

    - ``mode="broadening"`` — the *small* names carry the planted excess drift, so the
      EW index (which over-weights them vs cap-weight) **must win**. This is the world the
      "equal-weight always beats" marketing imagines.
    - ``mode="megacap"`` — a handful of *large* names carry the drift and compound into an
      ever-bigger share of the cap-weight index, so CW **must win** and EW must lag. This is
      the 2015-2024 mega-cap tape in a bottle.

  The spine tests run both: a harness that can't bank EW's edge when small names lead, and
  can't see EW lose when mega-caps lead, proves nothing on the real tape.

- ``load_real`` — the real daily tapes via :mod:`quantlab.data` (Yahoo, cached parquet),
  total-return adjusted. ``RSP`` is Invesco's **equal-weight** S&P 500 ETF (since 2003);
  ``SPY`` is the **cap-weight** S&P 500 ETF. Total return is the fair basis — an EW vs CW
  race is decided partly by reinvested dividends. Cache-first; the network is touched only
  on a cache miss.

No look-ahead is baked in here — both arms are buy-and-hold index curves; the sub-period
split and the costs live in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (positive/negative control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    n_names: int = 50,
    mode: str = "broadening",
    daily_vol: float = 0.013,
    start: str = "2003-04-30",
    seed: int = 94,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible two-column tape: an equal-weight and a cap-weight index of the same names.

    A universe of ``n_names`` stocks is given cap weights that decay geometrically (a few big
    names, a long tail of small ones — like a real index). Every name shares a common market
    factor; on top of it, ``mode`` plants *where* the extra drift lives:

    - ``mode="broadening"`` — the small-cap tail carries a positive excess drift. An
      equal-weight index over-weights that tail relative to cap-weight, so **EW must win**.
    - ``mode="megacap"`` — the few largest names carry the excess drift and, because the
      cap-weight index lets its winners compound, CW's concentration grows and **CW must win**
      (EW lags badly, exactly as 2015-2024).

    Returns ``(frame, truth)`` where ``frame`` has ``ew`` and ``cw`` total-return columns and
    ``truth`` records the planted parameters. The EW index is rebalanced to equal weights
    every ``rebal`` days (the mechanical rebalancing the folklore credits).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    rebal = 21  # ~monthly equal-weight rebalance

    # Cap weights: geometric decay -> a concentrated head, a long small-cap tail.
    rank = np.arange(n_names)
    cap0 = 0.80 ** rank
    cap0 = cap0 / cap0.sum()

    # Per-name excess drift depends on the regime we plant.
    mkt_mu = 0.06 / 252.0
    excess = np.zeros(n_names)
    if mode == "broadening":
        # small (high-rank) names get the premium
        small = rank >= n_names // 2
        excess[small] = 0.10 / 252.0
    elif mode == "megacap":
        # the top few names get a strong premium
        big = rank < 5
        excess[big] = 0.22 / 252.0
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'broadening' or 'megacap')")

    name_vol = daily_vol * (1.0 + 0.6 * rank / n_names)  # small names a touch more volatile
    # Daily simple returns per name: shared market shock + idiosyncratic + planted drift.
    mkt = rng.normal(0.0, daily_vol * 0.7, n_days)
    rets = np.empty((n_days, n_names))
    for j in range(n_names):
        idio = rng.normal(0.0, name_vol[j], n_days)
        rets[:, j] = mkt_mu + excess[j] + mkt + idio

    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)

    # Cap-weight index: drifting weights, set once from cap0 then let prices run; the index
    # return is the cap-weighted average of name returns where weights are the *previous*
    # day's market-value shares (so winners' weight grows — concentration compounds).
    shares = cap0 / prices[0]  # buy-and-hold share counts proportional to start cap weight
    mv = shares * prices
    cw_level = mv.sum(axis=1)
    cw_level = cw_level / cw_level[0] * 100.0

    # Equal-weight index: rebalance to flat weights every `rebal` days.
    ew_level = np.empty(n_days)
    ew_level[0] = 100.0
    w = np.full(n_names, 1.0 / n_names)
    for t in range(1, n_days):
        r = rets[t]
        port_r = float(w @ r)
        ew_level[t] = ew_level[t - 1] * (1.0 + port_r)
        # weights drift with returns...
        w = w * (1.0 + r)
        w = w / w.sum()
        if t % rebal == 0:  # ...then snap back to equal weight
            w = np.full(n_names, 1.0 / n_names)

    frame = pd.DataFrame({"ew": ew_level, "cw": cw_level}, index=idx)
    frame.index.name = "Date"
    truth = {
        "mode": mode,
        "n_days": n_days,
        "n_names": n_names,
        "rebal": rebal,
        "seed": seed,
        "ew_wins": bool(ew_level[-1] > cw_level[-1]),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tapes — daily total-return close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ew_ticker: str = "RSP",
    cw_ticker: str = "SPY",
    start: str = "2003-04-30",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily total-return tapes for the equal-weight and cap-weight arms, aligned.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``RSP`` (Invesco S&P 500 Equal Weight, inception 2003-04-24) is the
    equal-weight arm; ``SPY`` the cap-weight arm. ``mode="total_return"`` folds dividends
    back in — the only fair basis for an EW-vs-CW race. Returns a frame with two columns,
    ``ew`` and ``cw``, indexed by the **intersection** of both series' dates (so the race
    is run on the common window, RSP-bounded from 2003).
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    ew = qdata.fetch(ew_ticker, start=start, end=end, mode=mode, use_cache=use_cache)["Close"]
    cw = qdata.fetch(cw_ticker, start=start, end=end, mode=mode, use_cache=use_cache)["Close"]
    frame = pd.concat({"ew": ew, "cw": cw}, axis=1).dropna()
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (both columns), for the as-of stamp."""
    arr = np.ascontiguousarray(frame[["ew", "cw"]].to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
