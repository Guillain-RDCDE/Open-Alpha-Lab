"""Data layer for Study 339 (Convertible-Bonds).

Two kinds of tape, one shape (a daily frame of total-return *price* columns):

- ``synthetic_convex`` — a *deterministic, offline* generator that plants the only thing
  a convexity claim can rest on: **positive gamma** versus an equity index. A ``gamma``
  knob builds a synthetic "convertible" whose return is a *non-linear* function of the
  index return (a long-call-like payoff: rises with the index but with a soft floor on
  the downside). ``gamma > 0`` is the **positive control** — the harness must detect the
  curvature. ``gamma == 0`` collapses to a plain linear stock/bond blend (the **null**):
  no convexity to find. This control may prove the machinery works; it can NEVER stand in
  for market evidence (see METHODOLOGY → the inference bar).
- ``load_real`` — the real daily **total-return** tapes via the shared cross-asset panel
  (``_cache/cross_asset_etfs.parquet``, ``yfinance auto_adjust=True``) with a
  ``quantlab.data`` fallback per ticker. CWB (convertibles), SPY (stocks), AGG
  (aggregate bonds) and SHY (1-3y Treasuries, the cash / excess-of-cash proxy). CWB
  lists **2009-04-16**, which bounds the joint window honestly.

No look-ahead is baked in here — all the bookkeeping lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
PANEL = os.path.join(DEFAULT_CACHE, "cross_asset_etfs.parquet")

# CWB (SPDR Bloomberg Convertible Securities ETF) lists 2009-04-16 — it bounds the window.
CWB_INCEPTION = "2009-04-16"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (convexity control)
# ---------------------------------------------------------------------------
def synthetic_convex(
    n_days: int = 4000,
    gamma: float = 0.6,
    delta: float = 0.55,
    mu_idx: float = 0.08,
    vol_idx: float = 0.17,
    bond_mu: float = 0.03,
    bond_vol: float = 0.05,
    bond_beta: float = 0.0,
    start: str = "2009-04-16",
    seed: int = 339,
) -> tuple[pd.DataFrame, dict]:
    """A synthetic equity index, a bond, and a *convex* 'convertible' on the index.

    Builds three total-return price series ('IDX', 'BND', 'CVT'):

    - ``IDX`` — an equity index, geometric Brownian motion (drift ``mu_idx``, vol ``vol_idx``).
    - ``BND`` — a bond leg with its own drift/vol and a tunable ``bond_beta`` to the index.
    - ``CVT`` — the synthetic convertible. Its *daily* return is a non-linear function of the
      index return ``r``::

          cvt_r = delta * r  +  gamma * (max(r, 0)**2 - E[max(r,0)**2])  +  carry + noise

      The squared-upside term is the planted **positive gamma**: the convertible captures
      *more* of big up days than down days (option-like convexity). ``gamma == 0`` removes
      all curvature → ``CVT`` is just a linear ``delta``-beta exposure plus carry (the null).

    Returns ``(frame, truth)`` where ``frame`` has columns ``['IDX', 'BND', 'CVT']`` (levels
    starting at 100) and ``truth`` records the planted parameters (the quadratic coefficient
    is what the harness must recover with the right sign).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # equity index daily log-returns
    sd = vol_idx / np.sqrt(252.0)
    r = mu_idx / 252.0 + sd * rng.standard_normal(n_days)
    # bond daily returns: own factor + a little index beta
    bz = rng.standard_normal(n_days)
    rb = bond_mu / 252.0 + bond_beta * r + (bond_vol / np.sqrt(252.0)) * bz

    # convertible: linear delta + a convex (positive-gamma) upside term + carry + idio noise.
    # The convex term is the *scaled* squared upside: gamma * up^2 / sd, so that the quadratic
    # coefficient the regression recovers is gamma/sd (a meaningful, detectable curvature) and
    # gamma itself reads as the convexity contribution at a one-sigma up move. De-meaned so it
    # plants curvature, not drift.
    up = np.maximum(r, 0.0)
    convex = (up * up) / sd
    convex = convex - convex.mean()
    carry = 0.01 / 252.0  # small coupon-like carry
    idio = (0.04 / np.sqrt(252.0)) * rng.standard_normal(n_days)
    rc = delta * r + gamma * convex + carry + idio

    levels = lambda x: 100.0 * np.exp(np.cumsum(x))
    frame = pd.DataFrame(
        {"IDX": levels(r), "BND": levels(rb), "CVT": levels(rc)}, index=idx
    )
    frame.index.name = "Date"

    # the planted quadratic coefficient of cvt_r on (r, r^2_up): gamma is its sign/size
    truth = {
        "gamma": float(gamma),
        "delta": float(delta),
        "n_days": int(n_days),
        "seed": int(seed),
        "mu_idx": mu_idx, "vol_idx": vol_idx,
        "bond_mu": bond_mu, "bond_vol": bond_vol, "bond_beta": bond_beta,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = ("CWB", "SPY", "AGG", "SHY"),
    start: str = CWB_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Primary source is the shared cross-asset panel (``_cache/cross_asset_etfs.parquet``,
    ``yfinance auto_adjust=True`` ⇒ dividends + splits folded in ⇒ total return). On a cache
    miss for any ticker it falls back to :func:`quantlab.data.fetch` with
    ``mode="total_return"``. The frame is the inner join across tickers (dropna), so it
    begins at the latest inception — CWB (2009-04-16) bounds the joint window. Columns are
    upper-case tickers.
    """
    cols: dict[str, pd.Series] = {}
    panel = None
    if use_cache and os.path.exists(PANEL):
        panel = pd.read_parquet(PANEL)
        panel.index = pd.DatetimeIndex(panel.index).tz_localize(None)

    for tk in tickers:
        if panel is not None and tk in panel.columns:
            cols[tk] = panel[tk].rename(tk)
        else:
            import sys

            sys.path.insert(0, REPO_ROOT)
            from quantlab import data as qdata

            raw = qdata.fetch(tk, start=start, end=end, mode="total_return",
                              use_cache=use_cache)
            cols[tk] = raw["Close"].rename(tk)

    frame = pd.concat(cols.values(), axis=1, join="inner").dropna()
    frame = frame.loc[start:]
    if end is not None:
        frame = frame.loc[:end]
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame, for the as-of stamp.

    Hashes the column names and the raw float bytes of every column, so a different window,
    a different asset, or drifted data all change the fingerprint loudly.
    """
    h = hashlib.sha1()
    for c in frame.columns:
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
