"""Data layer for Study 356 — the GLP-1 / Ozempic basket.

Two sources, both offline-friendly:

* **Real tape.** Daily *adjusted-close* (total-return) prices for **LLY, NVO, XLV, SPY**
  from Yahoo Finance (``yfinance``), 2018-01-02 → 2026-06-18, cached under
  ``_cache/prices_2018_2026.csv``. NVO is the US ADR (the freely-quoted line a US reader
  would actually buy); XLV is the Health Care Select Sector SPDR — our healthcare-sector
  benchmark; SPY is the market. Adjusted close means dividends are reinvested — labelled
  total-return, not price-only, on every table.
* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that
  plants a *known* per-name alpha on top of a one-factor market model. It is the positive
  control: the inference engine must recover the planted alpha (and its absence) — a proof
  the machinery works, never market evidence.

Pure numpy + pandas + stdlib; :func:`load_real` only touches disk, never the network. The
network fetch lives in :func:`fetch_real` and is called *only* on an explicit cache miss.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "prices_2018_2026.csv")

TICKERS = ["LLY", "NVO", "XLV", "SPY"]
START = "2018-01-01"
END = "2026-06-21"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Return the cached adjusted-close panel ``[LLY, NVO, XLV, SPY]`` (Date index)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[TICKERS].dropna(how="any")


def fetch_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the adjusted-close panel from Yahoo and cache it. Network; cache-miss only."""
    import yfinance as yf  # imported lazily so the offline core never needs it

    df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    close = df["Close"][TICKERS].dropna(how="any")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    close.to_csv(path)
    return close


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total-return matrix from an adjusted-close panel."""
    return prices.pct_change().dropna(how="any")


def basket_returns(rets: pd.DataFrame, weights=None) -> pd.Series:
    """Daily-rebalanced GLP-1 basket return. Default = 50/50 LLY/NVO.

    Daily rebalancing is the most generous reading of "hold the theme" (it keeps the
    weights fixed); turnover and its cost are charged separately in
    :func:`strategy.cost_sweep`.
    """
    if weights is None:
        weights = {"LLY": 0.5, "NVO": 0.5}
    w = pd.Series(weights)
    return (rets[w.index] * w).sum(axis=1)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 2000, alphas=None, betas=None,
                    mkt_mu: float = 0.0004, mkt_sigma: float = 0.011,
                    idio_sigma: float = 0.014, seed: int = 356) -> pd.DataFrame:
    """Deterministic daily-return panel from a one-factor market model.

    Each name ``i`` has return ``r_i = alpha_i + beta_i * mkt + eps_i`` with the market
    ``mkt`` and i.i.d. idiosyncratic noise. The planted *daily* ``alpha_i`` is recoverable
    by :func:`strategy.market_model`; a name with ``alpha_i = 0`` must test insignificant.
    Returns a returns DataFrame with columns ``[LLY, NVO, XLV, SPY]`` (SPY *is* the market).

    Defaults plant a real edge in LLY (~26%/yr), ~zero in NVO, a mild one in XLV — mirroring
    the real tape — so the control proves the engine can both *find* and *not find* alpha.
    """
    if alphas is None:
        alphas = {"LLY": 0.0010, "NVO": 0.0000, "XLV": 0.0002, "SPY": 0.0}
    if betas is None:
        betas = {"LLY": 0.66, "NVO": 0.63, "XLV": 0.70, "SPY": 1.0}
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sigma, n_days)
    cols = {}
    for name in TICKERS:
        if name == "SPY":
            cols[name] = mkt
        else:
            eps = rng.normal(0.0, idio_sigma, n_days)
            cols[name] = alphas[name] + betas[name] * mkt + eps
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame(cols, index=idx)[TICKERS]
