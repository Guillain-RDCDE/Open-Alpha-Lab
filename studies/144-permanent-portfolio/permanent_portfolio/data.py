"""Data layer for Study 144 (Permanent-Portfolio).

Two tapes, one shape (a daily frame of total-return *price* columns):

- ``synthetic_four_asset`` — a *deterministic, offline* generator. Four assets with
  a tunable regime-cycle knob (``cycle_strength``) that plants the diversification
  the Permanent Portfolio is built to harvest: in each of four regimes one asset
  leads and the others hedge. ``cycle_strength=0`` makes all four assets i.i.d.
  — the null where equal-weight and PP are identical. ``cycle_strength>0`` plants
  the rotation across growth / inflation / deflation / recession that Brown claimed
  each of the four legs was designed to catch. This is the positive control.
- ``load_real`` — the real daily total-return panel for SPY / TLT / GLD / SHY
  from the shared cross-asset ETF cache (``_cache/cross_asset_etfs.parquet``).
  GLD incepted 2004-11-18, bounding the joint window to roughly 21 years.

No look-ahead is baked in here — the annual-rebalance bookkeeping lives in
``strategy.py``.
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

# PP quartet: stocks, long Treasuries, gold, cash
ASSETS = ["SPY", "TLT", "GLD", "SHY"]
# GLD incepted 2004-11-18 — the binding constraint for the joint window.
GLD_INCEPTION = "2004-11-18"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_four_asset(
    n_years: int = 20,
    cycle_strength: float = 0.0,
    seed: int = 144,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily four-asset world with a planted regime cycle.

    The four legs ("STK", "BOND", "GOLD", "CASH") all share the same base
    Sharpe but different volatilities matching their real-world analogues.
    With ``cycle_strength > 0`` a four-regime cycle is imposed: in each
    quarter-year one asset earns an extra return boost while the others earn
    their base, simulating the growth / inflation / deflation / recession
    rotation that Harry Browne argued the PP was built to navigate.

    - ``cycle_strength = 0.0`` → all four legs i.i.d.; PP == equal-weight.
    - ``cycle_strength > 0`` → each regime's "winner" pulls ahead; the
      cross-asset hedge lifts the portfolio Sharpe only when the cycle is real.

    Returns ``(frame, truth)`` where ``frame`` has columns
    ``['STK', 'BOND', 'GOLD', 'CASH']`` (price levels starting at 100) and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range("2005-01-03", periods=n_days, name="Date")

    # Asset-level annual Sharpe ~ 0.4, vols matching real-world order of magnitude
    ann_sharpes = np.array([0.40, 0.40, 0.40, 0.20])  # cash has lower Sharpe
    ann_vols = np.array([0.16, 0.12, 0.18, 0.005])    # stocks, bonds, gold, cash
    daily_mu = ann_sharpes * ann_vols / 252.0
    daily_vol = ann_vols / np.sqrt(252.0)

    # Light common factor (flight-to-quality: stocks vs bonds/gold negatively correlated)
    corr = np.array([
        [1.00, -0.30, 0.05, 0.00],
        [-0.30, 1.00, 0.10, 0.00],
        [0.05,  0.10, 1.00, 0.00],
        [0.00,  0.00, 0.00, 1.00],
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 4)) @ chol.T

    # Regime cycle: 4 regimes of roughly equal length, cycling through the year
    # Each regime boosts one leg by cycle_strength (as an annualised excess Sharpe)
    regime_idx = (np.arange(n_days) // (n_days // 4)) % 4  # 0..3 cycling
    boost = np.zeros((n_days, 4))
    for i in range(4):
        boost[regime_idx == i, i] = cycle_strength * ann_vols[i] / 252.0

    daily_ret = daily_mu + boost + daily_vol * z
    prices = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    cols = ["STK", "BOND", "GOLD", "CASH"]
    frame = pd.DataFrame(prices, index=idx, columns=cols)

    truth = {
        "cycle_strength": cycle_strength,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "ann_vols": ann_vols.tolist(),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = ("SPY", "TLT", "GLD", "SHY"),
    start: str = GLD_INCEPTION,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for the PP quartet.

    Primary source: the shared cross-asset ETF cache
    (``_cache/cross_asset_etfs.parquet``, yfinance ``auto_adjust=True`` so
    dividends and splits are folded in — a genuine total-return series for
    fair buy-and-hold comparison). The joint window begins at GLD's
    inception (2004-11-18). If the cache is missing and ``fetch=True`` the
    tickers are downloaded via yfinance and the panel is cached.

    Returns a price frame (NOT returns); converts to returns in strategy.py.
    """
    panel_path = os.path.join(cache_dir, "cross_asset_etfs.parquet")
    if os.path.exists(panel_path):
        raw = pd.read_parquet(panel_path)
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        cols_avail = [t for t in tickers if t in raw.columns]
        if cols_avail:
            frame = raw[cols_avail].dropna()
            frame = frame.loc[start:]
            if end is not None:
                frame = frame.loc[:end]
            frame.index.name = "Date"
            if set(cols_avail) == set(tickers):
                return frame

    # Fallback: fetch via yfinance
    if not fetch:
        raise FileNotFoundError(
            f"No cross-asset panel at {panel_path}. "
            "Call load_real(fetch=True) once to populate the cache."
        )
    import yfinance as yf

    px = yf.download(
        list(tickers), period="max", auto_adjust=True, progress=False
    )["Close"][list(tickers)].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    os.makedirs(cache_dir, exist_ok=True)
    px.to_parquet(panel_path)
    frame = px.loc[start:]
    if end is not None:
        frame = frame.loc[:end]
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame, for the as-of stamp."""
    h = hashlib.sha1()
    for c in sorted(frame.columns):
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
