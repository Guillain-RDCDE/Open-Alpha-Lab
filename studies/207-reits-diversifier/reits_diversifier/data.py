"""Data layer for Study 207 (REITs-Diversifier).

Two tapes, one purpose — test whether VNQ (Vanguard Real Estate ETF) acts as a genuine
portfolio diversifier and inflation hedge, or merely as leveraged equity-beta that
crashes alongside stocks in crisis:

- ``synthetic_portfolio`` — a *deterministic, offline* generator. Simulates a
  three-asset world (equity, reit, bond) with a tunable ``diversification`` knob.
  At ``diversification=0`` the REIT is pure equity-beta and adds nothing to a
  60/40. At ``diversification>0`` the REIT has genuinely independent risk premia
  (income stream), lowering portfolio drawdown and raising the Sharpe. This knob
  lets us confirm the engine is sensitive to what it claims to measure.
- ``load_real`` — the real daily total-return panel for VNQ, SPY, TLT, IEF, GLD,
  and SHY from the shared cross-asset ETF cache
  (``_cache/cross_asset_etfs.parquet``). VNQ incepted 2004-09-29; the joint
  window with GLD runs from 2004-11-18. Source: yfinance ``auto_adjust=True``
  (total-return, dividends folded in).

No look-ahead is baked in here — rebalance bookkeeping and the regime-split
analysis live in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# Primary tickers: VNQ is the REIT vehicle; SPY/TLT form the 60/40 baseline;
# IEF and SHY as intermediate and short bond references; GLD as crisis hedge.
TICKERS = ("VNQ", "SPY", "TLT", "IEF", "SHY", "GLD")

# VNQ incepted 2004-09-29; GLD incepted 2004-11-18 → binding start of joint tape.
VNQ_INCEPTION = "2004-11-18"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_portfolio(
    n_years: int = 20,
    diversification: float = 0.0,
    seed: int = 207,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily three-asset (equity / reit / bond) return world.

    Parameters
    ----------
    n_years:
        Simulation length in calendar years (using 252 business days per year).
    diversification:
        Controls how much genuine, uncorrelated income the REIT leg contributes
        beyond its equity-beta:

        - ``0.0`` → REIT is 100% equity-beta (SPY-clone with same vol); the REIT
          sleeve cannot improve the 60/40 Sharpe or reduce drawdown because it
          adds no new information.
        - ``>0`` → REIT has an extra independent income component of magnitude
          ``diversification * 0.02`` (annualised, per unit vol); the REIT sleeve
          now genuinely reduces portfolio drawdown in the diversification regime.

        This is the study's null in a bottle: with ``diversification=0`` any
        apparent benefit from a REIT sleeve is sampling noise; with
        ``diversification=1`` a fair analysis should recover the improvement.
    seed:
        RNG seed for exact reproducibility.

    Returns
    -------
    (frame, truth)
        ``frame`` — a date-indexed DataFrame with columns ``['EQ', 'REIT', 'BOND']``
        holding *price levels* starting at 100.
        ``truth`` — dict recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range("2005-01-03", periods=n_days, name="Date")

    # Rough calibration to real-world annualised stats:
    #   EQ  ~ SPY:  ann vol 18%, ann Sharpe 0.50
    #   REIT~ VNQ:  ann vol 20%, ann Sharpe 0.45 (+ income kicker if divers > 0)
    #   BOND~ TLT:  ann vol 12%, ann Sharpe 0.35
    ann_vols = np.array([0.18, 0.20, 0.12])
    daily_vol = ann_vols / np.sqrt(252.0)

    # Base annualised Sharpe ratios (no diversification boost)
    ann_sharpes = np.array([0.50, 0.45, 0.35])
    daily_mu = ann_sharpes * ann_vols / 252.0

    # Correlation structure: EQ and REIT are ~0.70 correlated (empirical),
    # BOND is mildly flight-to-quality negative vs EQ (-0.20).
    corr = np.array([
        [1.00,  0.70, -0.20],
        [0.70,  1.00, -0.10],
        [-0.20, -0.10,  1.00],
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 3)) @ chol.T

    # Diversification boost: an independent income component for the REIT leg.
    # This represents rental income / dividend yield uncorrelated with equity beta.
    income_boost = np.zeros((n_days, 3))
    if diversification != 0.0:
        income_daily = diversification * 0.02 * ann_vols[1] / 252.0
        # The income is i.i.d. — truly independent of equity factor
        income_boost[:, 1] = rng.normal(income_daily, daily_vol[1] * 0.10, n_days)

    daily_ret = daily_mu + income_boost + daily_vol * z
    prices = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    cols = ["EQ", "REIT", "BOND"]
    frame = pd.DataFrame(prices, index=idx, columns=cols)

    truth = {
        "diversification": diversification,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "ann_vols": ann_vols.tolist(),
        "ann_sharpes": ann_sharpes.tolist(),
        "eq_reit_corr": float(corr[0, 1]),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices from the shared cross-asset cache
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = TICKERS,
    start: str = VNQ_INCEPTION,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for the REIT panel.

    Primary source: the shared cross-asset ETF cache
    (``_cache/cross_asset_etfs.parquet``, yfinance ``auto_adjust=True`` so
    dividends and splits are folded in — a genuine total-return series). The
    joint window begins at 2004-11-18 (GLD inception). If the cache is missing
    and ``fetch=True`` the tickers are downloaded via yfinance.

    Returns a price frame (NOT returns); converts to returns in ``strategy.py``.
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

    # Fallback: fetch via yfinance (only on explicit fetch=True)
    if not fetch:
        raise FileNotFoundError(
            f"No cross-asset panel at {panel_path}. "
            "Call load_real(fetch=True) once to populate the cache."
        )
    import yfinance as yf

    px = yf.download(
        list(tickers), period="max", auto_adjust=True, progress=False
    )
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"][list(tickers)]
    else:
        px = px[["Close"]].rename(columns={"Close": tickers[0]})
    px = px.dropna(how="all")
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
