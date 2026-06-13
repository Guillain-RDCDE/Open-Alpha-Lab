"""Data layer for Study 102 (Free-Rebalance).

Three tapes, one shape (a daily frame of **total-return** closes, one column per asset):

- ``load_real`` — a real multi-asset tape via :mod:`quantlab.data` (Yahoo, cached
  parquet), **total-return adjusted** so the buy-and-hold / drift benchmark is a fair,
  dividend-inclusive comparison. Default basket is ``SPY`` + ``TLT`` (stocks + long
  Treasuries); ``GLD`` is available too. Honest windows: SPY from 1993, **TLT from
  2002-07**, **GLD from 2004-11** — the aligned frame starts at the *latest* asset's
  first date, stated openly, never back-filled.

- ``synthetic_meanrevert`` — the **positive control** for the bonus. Two correlated
  assets with *equal* long-run drift that mean-revert around it (an Ornstein-Uhlenbeck
  log-spread). Neither asset trends away, so rebalancing repeatedly sells the temporary
  winner and buys the temporary loser: the 'rebalancing bonus' (Shannon's Demon) is
  clearly **positive**. This is the harness's positive control for a *positive* bonus.

- ``synthetic_trend`` — the **negative control**. Two assets where one strongly and
  persistently **trends** above the other. Rebalancing keeps trimming the winner and
  topping up the laggard, so the drift (buy-and-hold) portfolio pulls ahead: the bonus
  is **negative**. This proves the harness measures the *sign* correctly, not just
  magnitude — a free lunch that can turn into a tax.

No look-ahead is baked in here: the rebalancing schedule and one-period cost live in
``strategy.py``. All series are total-return so a drift portfolio compounds dividends
exactly as a held basket would.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tapes — the deterministic offline controls
# ---------------------------------------------------------------------------
def synthetic_meanrevert(
    n_days: int = 6000,
    drift: float = 0.07,
    daily_vol: float = 0.012,
    kappa: float = 0.02,
    spread_vol: float = 0.02,
    start: str = "2000-01-03",
    seed: int = 102,
) -> tuple[pd.DataFrame, dict]:
    """Two correlated, equal-drift assets that mean-revert around a common trend.

    Both assets share a common market log-return (drift + a shared shock) and add an
    Ornstein-Uhlenbeck *log-spread* of opposite sign: when asset A is temporarily rich,
    asset B is temporarily cheap, and the spread is pulled back to zero at rate
    ``kappa``. Neither asset wins in the long run (equal drift), so a rebalancer keeps
    selling the temporary winner and buying the temporary loser — the **positive**
    rebalancing bonus (Shannon's Demon).

    Returns ``(frame, truth)`` where ``frame`` has columns ``A``/``B`` of total-return
    closes and ``truth`` records the planted parameters and the expected bonus sign.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    mu = drift / 252.0
    # Shared market shock + an OU log-spread that pushes A and B apart symmetrically.
    common = rng.normal(mu, daily_vol, n_days)
    s = 0.0
    spread = np.empty(n_days)
    for t in range(n_days):
        s = (1.0 - kappa) * s + rng.normal(0.0, spread_vol)
        spread[t] = s
    log_a = np.cumsum(common) + 0.5 * spread
    log_b = np.cumsum(common) - 0.5 * spread
    frame = pd.DataFrame(
        {"A": 100.0 * np.exp(log_a), "B": 100.0 * np.exp(log_b)}, index=idx
    )
    frame.index.name = "Date"
    truth = {
        "kind": "mean_revert",
        "n_days": n_days,
        "drift": drift,
        "kappa": kappa,
        "seed": seed,
        "expected_bonus_sign": +1,
    }
    return frame, truth


def synthetic_trend(
    n_days: int = 6000,
    drift_lead: float = 0.18,
    drift_lag: float = 0.02,
    daily_vol: float = 0.012,
    start: str = "2000-01-03",
    seed: int = 102,
) -> tuple[pd.DataFrame, dict]:
    """Two correlated assets where one persistently TRENDS above the other.

    Asset A compounds at ``drift_lead`` (e.g. ~18%/yr), asset B at the much lower
    ``drift_lag`` (~2%/yr), sharing a common shock so they stay correlated. A rebalancer
    keeps trimming the runaway winner (A) and topping up the laggard (B), so the
    buy-and-hold / drift portfolio — which lets the winner run — pulls ahead. The
    rebalancing bonus here is **negative**: the negative control.

    Returns ``(frame, truth)`` with columns ``A``/``B`` and the expected (negative) sign.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    shared = rng.normal(0.0, daily_vol * 0.7, n_days)
    ra = drift_lead / 252.0 + shared + rng.normal(0.0, daily_vol * 0.7, n_days)
    rb = drift_lag / 252.0 + shared + rng.normal(0.0, daily_vol * 0.7, n_days)
    frame = pd.DataFrame(
        {"A": 100.0 * np.exp(np.cumsum(ra)), "B": 100.0 * np.exp(np.cumsum(rb))},
        index=idx,
    )
    frame.index.name = "Date"
    truth = {
        "kind": "trend",
        "n_days": n_days,
        "drift_lead": drift_lead,
        "drift_lag": drift_lag,
        "seed": seed,
        "expected_bonus_sign": -1,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return closes for a basket, cache-first, aligned
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = ("SPY", "TLT"),
    start: str = "1993-01-01",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily total-return closes for a basket as a multi-column frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``mode="total_return"`` folds dividends back in so a held basket
    compounds them exactly as a drift portfolio would. The returned frame is **aligned
    on the intersection of dates** (inner join) and therefore starts at the *latest*
    constituent's first trading day — the honest common window (TLT from 2002-07, GLD
    from 2004-11). Columns are the tickers, in the order given.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    cols = {}
    for tk in tickers:
        raw = qdata.fetch(tk, start=start, end=end, mode=mode, use_cache=use_cache)
        cols[tk] = raw["Close"]
    frame = pd.DataFrame(cols).dropna(how="any")
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (all columns), for the as-of stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
