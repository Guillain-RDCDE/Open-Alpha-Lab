"""Data layer for Study 204 (Talmud-Portfolio).

Two tapes, one shape (a daily frame of total-return *price* columns):

- ``synthetic_three_asset`` — a *deterministic, offline* generator. Three assets with a
  tunable ``cycle_strength`` knob that plants the diversification benefit the Talmud
  allocation is built to harvest: in each of three regimes one asset leads while the
  others hedge. ``cycle_strength = 0`` makes all three assets i.i.d. normal — the null
  where the Talmud 1/3–1/3–1/3 blend is indistinguishable from any other fixed-weight
  portfolio. ``cycle_strength > 0`` plants cross-asset rotation (real estate / stocks /
  bonds responding differently to inflation, growth, and safe-haven cycles) so the hedge
  value of the three-leg split shows up in reduced drawdown and higher Sharpe. This is
  the study's null in a bottle — a fair baseline for the positive-control sweep.

- ``load_real`` — the real daily total-return panel for VNQ (real estate), SPY (stocks),
  and BND (broad bonds/cash-proxy) from the shared cross-asset ETF cache
  (``_cache/cross_asset_etfs.parquet``, Yahoo Finance ``auto_adjust=True``). BND
  incepted 2007-04-10, bounding the joint window.

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

# The three Talmud legs: real estate / stocks / bonds-cash
ASSETS = ("VNQ", "SPY", "BND")
# BND (Vanguard Total Bond Market ETF) incepted 2007-04-10 — the binding constraint.
BND_INCEPTION = "2007-04-10"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_three_asset(
    n_years: int = 15,
    cycle_strength: float = 0.0,
    seed: int = 204,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily three-asset world with a planted regime cycle.

    The three legs ("REIT", "STK", "BOND") share the same base Sharpe ratio but carry
    different volatilities matching their real-world order of magnitude (REITs ~0.20,
    stocks ~0.16, bonds ~0.06). With ``cycle_strength > 0`` a three-regime cycle is
    imposed so that each year's dominant asset earns an extra return boost while the
    others earn their base — simulating the inflation / growth / safe-haven rotation
    that the Talmud diversification is built to navigate.

    - ``cycle_strength = 0.0`` → all three legs i.i.d.; equal-weight is optimal.
    - ``cycle_strength > 0``  → cross-asset rotation boosts the hedge value of the
      three-leg split; Sharpe rises and drawdown falls relative to 100 % STK.

    Returns ``(frame, truth)`` where ``frame`` has columns
    ``['REIT', 'STK', 'BOND']`` (price levels starting at 100) and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range("2007-04-10", periods=n_days, name="Date")

    # Annual Sharpes and vols calibrated to real-world analogues
    ann_sharpes = np.array([0.40, 0.45, 0.25])   # REIT, STK, BOND
    ann_vols    = np.array([0.20, 0.16, 0.06])    # REIT, STK, BOND
    daily_mu  = ann_sharpes * ann_vols / 252.0
    daily_vol = ann_vols / np.sqrt(252.0)

    # Light correlation structure (REITs and stocks correlated; bonds hedge both)
    corr = np.array([
        [1.00,  0.55, -0.15],
        [0.55,  1.00, -0.20],
        [-0.15, -0.20,  1.00],
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 3)) @ chol.T

    # Regime cycle: 3 regimes of roughly equal length per year, cycling through
    # Each regime boosts one leg by cycle_strength (as an annualised excess Sharpe)
    regime_idx = (np.arange(n_days) // (n_days // 3)) % 3
    boost = np.zeros((n_days, 3))
    for i in range(3):
        boost[regime_idx == i, i] = cycle_strength * ann_vols[i] / 252.0

    daily_ret = daily_mu + boost + daily_vol * z
    prices = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    frame = pd.DataFrame(prices, index=idx, columns=["REIT", "STK", "BOND"])

    truth = {
        "cycle_strength": cycle_strength,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "ann_vols": ann_vols.tolist(),
        "ann_sharpes": ann_sharpes.tolist(),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = ASSETS,
    start: str = BND_INCEPTION,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for the Talmud trio.

    Primary source: the shared cross-asset ETF cache
    (``_cache/cross_asset_etfs.parquet``, yfinance ``auto_adjust=True`` so dividends
    and splits are folded in — a genuine total-return series for fair buy-and-hold
    comparison). The joint window begins at BND's inception (2007-04-10). If the shared
    panel cache is missing and ``fetch=True`` the tickers are downloaded via yfinance
    and a per-study parquet is written to ``_cache/``.

    Returns a **price frame** (NOT returns); convert to returns in strategy.py.
    """
    # Primary: try the shared cross-asset panel (populated by other studies)
    panel_path = os.path.join(cache_dir, "cross_asset_etfs.parquet")
    study_cache = os.path.join(HERE, "..", "_cache", "talmud_prices.parquet")
    study_cache = os.path.normpath(study_cache)

    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        cols = [t for t in tickers if t in df.columns]
        frame = df[cols].dropna()
        frame = frame.loc[start:]
        if end is not None:
            frame = frame.loc[:end]
        frame.index.name = "Date"
        return frame

    # Try to assemble from the shared panel (may have most but not all tickers)
    shared_cols: dict[str, pd.Series] = {}
    if os.path.exists(panel_path):
        panel = pd.read_parquet(panel_path)
        if panel.index.tz is not None:
            panel.index = panel.index.tz_localize(None)
        panel.index = pd.DatetimeIndex(panel.index)
        for t in tickers:
            if t in panel.columns:
                shared_cols[t] = panel[t]

    missing = [t for t in tickers if t not in shared_cols]

    # Per-study cache supplements the shared panel for any missing tickers
    if os.path.exists(study_cache):
        sc = pd.read_parquet(study_cache)
        if sc.index.tz is not None:
            sc.index = sc.index.tz_localize(None)
        sc.index = pd.DatetimeIndex(sc.index)
        for t in list(missing):
            if t in sc.columns:
                shared_cols[t] = sc[t]
                missing.remove(t)

    if not missing:
        combined = pd.DataFrame(shared_cols)
        return _slice(combined)

    # Still missing some — fetch from Yahoo
    if not fetch:
        raise FileNotFoundError(
            f"No real price cache found for tickers: {missing}. "
            "Call load_real(fetch=True) once to populate the cache.\n"
            f"  Expected shared panel at: {panel_path}\n"
            f"  Or per-study cache at:    {study_cache}"
        )

    import yfinance as yf

    raw = yf.download(missing, period="max", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        fetched = raw["Close"]
        if len(missing) == 1:
            fetched = fetched[missing]
    else:
        fetched = raw[["Close"]].rename(columns={"Close": missing[0]})
    fetched = fetched.dropna()
    if fetched.index.tz is not None:
        fetched.index = fetched.index.tz_localize(None)

    os.makedirs(os.path.dirname(study_cache), exist_ok=True)
    fetched.to_parquet(study_cache)

    for t in missing:
        if t in fetched.columns:
            shared_cols[t] = fetched[t]

    combined = pd.DataFrame(shared_cols)
    return _slice(combined)


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame, for the as-of stamp."""
    h = hashlib.sha1()
    for c in sorted(frame.columns):
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
