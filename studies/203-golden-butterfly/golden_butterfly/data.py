"""Data layer for Study 203 (Golden-Butterfly).

Two tapes, one shape (a daily frame of total-return *price* columns):

- ``synthetic_five_asset`` — a *deterministic, offline* generator. Five assets with
  a tunable regime-cycle knob (``cycle_strength``) that plants the diversification
  the Golden Butterfly is built to harvest: each economic regime benefits one or two
  legs while the others hedge.  ``cycle_strength=0`` makes all five assets i.i.d.
  — the null where equal-weight and GB are identical. ``cycle_strength>0`` plants
  the two-growth / inflation / deflation / recession rotation the Golden Butterfly
  allocates toward.  This is the positive control.
- ``load_real`` — the real daily total-return panel for SPY / IWN / TLT / SHY / GLD
  from a per-study yfinance cache (``_cache/gb_panel.parquet``). IWN (iShares
  Russell 2000 Value) incepted 2000-07-24, but GLD only from 2004-11-18, bounding
  the joint window to roughly 21 years.

No look-ahead is baked in here — the annual-rebalance bookkeeping lives in
``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_ROOT, "_cache")

# The five GB legs and the shared cross-asset panel path
GB_TICKERS = ("SPY", "IWN", "TLT", "SHY", "GLD")
# GLD incepted 2004-11-18 — the binding constraint for the joint window.
GLD_INCEPTION = "2004-11-18"
# Per-study cache file (IWN is not in the shared cross-asset parquet)
PANEL_FILE = "gb_panel.parquet"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_five_asset(
    n_years: int = 20,
    cycle_strength: float = 0.0,
    seed: int = 203,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily five-asset world with a planted regime cycle.

    The five legs ("LCG", "SCV", "BOND", "CASH", "GOLD") share the same base
    Sharpe but different volatilities matching their real-world analogues.
    With ``cycle_strength > 0`` a four-regime cycle is imposed: in each
    quarter-year two growth assets (LCG, SCV) or one of the defensive legs
    (BOND, GOLD, CASH) earns an extra return boost, simulating the prosperity /
    inflation / deflation / recession rotation the Golden Butterfly is built to
    navigate.

    - ``cycle_strength = 0.0`` → all five legs i.i.d.; GB == equal-weight.
    - ``cycle_strength > 0`` → regime leaders pull ahead; the cross-asset hedge
      lifts the portfolio Sharpe only when the cycle is real.

    Returns ``(frame, truth)`` where ``frame`` has columns
    ``['LCG', 'SCV', 'BOND', 'CASH', 'GOLD']`` (price levels starting at 100)
    and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range("2005-01-03", periods=n_days, name="Date")

    # Asset-level annual Sharpe ~ 0.4, vols matching real-world order of magnitude
    # LCG=large-cap growth(SPY), SCV=small-cap value(IWN), BOND=TLT, CASH=SHY, GOLD=GLD
    ann_sharpes = np.array([0.40, 0.40, 0.40, 0.15, 0.30])
    ann_vols = np.array([0.16, 0.20, 0.12, 0.005, 0.18])
    daily_mu = ann_sharpes * ann_vols / 252.0
    daily_vol = ann_vols / np.sqrt(252.0)

    # Light common factor: growth (LCG, SCV) negatively correlated with bonds/gold
    corr = np.array([
        [1.00,  0.65, -0.25,  0.00,  0.05],   # LCG
        [0.65,  1.00, -0.20,  0.00,  0.05],   # SCV
        [-0.25, -0.20, 1.00,  0.00,  0.10],   # BOND
        [0.00,  0.00,  0.00,  1.00,  0.00],   # CASH
        [0.05,  0.05,  0.10,  0.00,  1.00],   # GOLD
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 5)) @ chol.T

    # Regime cycle: 4 regimes cycling through the year
    # Regime 0 (prosperity): LCG and SCV lead
    # Regime 1 (inflation):  GOLD leads
    # Regime 2 (deflation):  BOND leads
    # Regime 3 (recession):  CASH leads
    regime_idx = (np.arange(n_days) // (n_days // 4)) % 4
    winners = {0: [0, 1], 1: [4], 2: [2], 3: [3]}  # which assets benefit per regime
    boost = np.zeros((n_days, 5))
    for regime, assets in winners.items():
        for a in assets:
            boost[regime_idx == regime, a] = cycle_strength * ann_vols[a] / 252.0

    daily_ret = daily_mu + boost + daily_vol * z
    prices = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    cols = ["LCG", "SCV", "BOND", "CASH", "GOLD"]
    frame = pd.DataFrame(prices, index=idx, columns=cols)

    truth = {
        "cycle_strength": cycle_strength,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "ann_vols": ann_vols.tolist(),
        "cols": cols,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = GB_TICKERS,
    start: str = GLD_INCEPTION,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for the GB quintet.

    Primary source: the per-study cache (``_cache/gb_panel.parquet``), populated
    by ``yfinance`` with ``auto_adjust=True`` (dividends and splits folded in).
    The joint window begins at GLD's inception (2004-11-18).  If the cache is
    missing and ``fetch=True`` the tickers are downloaded via yfinance and the
    panel is cached.

    Returns a price frame (NOT returns); converts to returns in strategy.py.
    """
    panel_path = os.path.join(cache_dir, PANEL_FILE)
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
            f"No GB panel at {panel_path}. "
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
