"""Data layer for Study 444 — Dow Theory (Industrials / Transports confirmation).

Two sources, both offline-friendly once cached:

* **Real tape.** Daily total-return-ish (auto-adjusted) closes for the two averages
  Dow Theory lives on — the **Industrials** and the **Transports**. We use the
  liquid, long-listed proxies that a believer could actually trade:

      - Industrials : ``DIA``  (SPDR Dow Jones Industrial Average ETF)
      - Transports  : ``IYT``  (iShares U.S. Transportation ETF)

  Both via yfinance (no key), auto-adjusted, cached as one wide parquet under
  ``_cache/``. We *also* pull the raw index levels ``^DJI`` / ``^DJT`` (price-only)
  for the longer-history robustness check, cached separately.

* **Synthetic.** A deterministic, fixed-seed generator with a **planted-edge knob**
  (``edge``): it builds two correlated price series where, when ``edge > 0``, the
  Industrials' next-day return is genuinely *higher* during periods when the
  Transports confirm the trend (both above their recent highs). With ``edge = 0``
  the confirmation flag carries no information and the strategy must NOT manufacture
  an edge. This is the positive control — a machinery proof, never market evidence.

``fingerprint`` hashes the cached frame so a re-run that matches holds the exact tape.
Pure numpy + pandas + stdlib on the offline path; ``fetch_*`` (network) is only used
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Tradable ETF proxies (auto-adjusted ~ total return) — the believer's real vehicle.
INDU_ETF = "DIA"   # Industrials
TRAN_ETF = "IYT"   # Transports
# Underlying price-only index levels (longer history, no dividends) — robustness only.
INDU_IDX = "^DJI"
TRAN_IDX = "^DJT"

ETF_CACHE = "dow_theory_etf.parquet"
IDX_CACHE = "dow_theory_idx.parquet"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _download(tickers, start, end, auto_adjust):
    import yfinance as yf

    last = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust,
                              progress=False)
            close = raw["Close"] if "Close" in raw else raw
            close = close.dropna(how="all")
            if len(close) > 50:
                return close
        except Exception as exc:  # pragma: no cover - network
            last = exc
        time.sleep(1.5 * (attempt + 1))
    if last is not None:  # pragma: no cover - network
        raise last
    raise RuntimeError("yfinance returned too little data")


def fetch_etf(start="2004-01-01", end=None, cache_dir=DEFAULT_CACHE) -> pd.DataFrame:
    """Download the two tradable ETF proxies (auto-adjusted) and cache one wide parquet.

    Network-only; used once to build the cache. Columns: ``indu``, ``tran``.
    """
    close = _download([INDU_ETF, TRAN_ETF], start, end, auto_adjust=True)
    df = pd.DataFrame({"indu": close[INDU_ETF], "tran": close[TRAN_ETF]}).dropna()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(os.path.join(cache_dir, ETF_CACHE))
    return df


def fetch_idx(start="1992-01-01", end=None, cache_dir=DEFAULT_CACHE) -> pd.DataFrame:
    """Download the raw index levels ``^DJI`` / ``^DJT`` (price-only) for robustness.

    Network-only; used once to build the cache. Columns: ``indu``, ``tran``.
    """
    close = _download([INDU_IDX, TRAN_IDX], start, end, auto_adjust=False)
    df = pd.DataFrame({"indu": close[INDU_IDX], "tran": close[TRAN_IDX]}).dropna()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(os.path.join(cache_dir, IDX_CACHE))
    return df


def have_real(cache_dir=DEFAULT_CACHE) -> bool:
    return os.path.exists(os.path.join(cache_dir, ETF_CACHE))


def have_idx(cache_dir=DEFAULT_CACHE) -> bool:
    return os.path.exists(os.path.join(cache_dir, IDX_CACHE))


def load_real(cache_dir=DEFAULT_CACHE) -> pd.DataFrame:
    """Cache-first tradable-ETF panel (auto-adjusted closes). Fetches once on a miss.

    Returns a wide frame indexed by date with columns ``indu`` (DIA) and ``tran`` (IYT).
    """
    path = os.path.join(cache_dir, ETF_CACHE)
    if not os.path.exists(path):
        return fetch_etf(cache_dir=cache_dir)
    return pd.read_parquet(path).sort_index()


def load_idx(cache_dir=DEFAULT_CACHE) -> pd.DataFrame:
    """Cache-first raw index-level panel (^DJI / ^DJT, price-only). Fetches once on a miss."""
    path = os.path.join(cache_dir, IDX_CACHE)
    if not os.path.exists(path):
        return fetch_idx(cache_dir=cache_dir)
    return pd.read_parquet(path).sort_index()


def fingerprint(df: pd.DataFrame, n: int = 12) -> str:
    """Stable sha1[:n] of a price frame's index + values — the reproducibility stamp."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(df.index.view("int64")).tobytes())
    h.update(np.ascontiguousarray(df.to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 5000, edge: float = 0.0, seed: int = 444,
                    lookback: int = 63, sig: float = 0.011, rho: float = 0.6
                    ) -> tuple[pd.DataFrame, dict]:
    """Deterministic two-series panel with a PLANTED confirmation edge.

    Two correlated daily-return series (``indu``, ``tran``), correlation ~``rho``.
    A "confirmation" flag is on when *both* series sit above their own ``lookback``-day
    trailing high as of the prior close. When ``edge > 0`` the Industrials' next-day
    return is boosted by ``edge`` (in daily units) on confirmed days — a genuine,
    plant-on-purpose Dow-Theory edge. With ``edge = 0`` the flag carries no
    information and the confirmation strategy must NOT beat buy-and-hold.

    Returns ``(prices, truth)`` where ``prices`` has columns ``indu``/``tran`` and
    ``truth`` records the planted edge and the realised confirmed-day fraction.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0.0, 1.0, n_days)
    z2 = rng.normal(0.0, 1.0, n_days)
    r_indu = 0.0003 + sig * z1
    r_tran = 0.0003 + sig * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)

    # Build prices iteratively so the planted edge depends on the *realised* trailing
    # high (no look-ahead): the boost on day t uses confirmation known at t-1's close.
    p_indu = np.empty(n_days)
    p_tran = np.empty(n_days)
    p_indu[0] = p_tran[0] = 100.0
    confirmed = np.zeros(n_days, dtype=bool)
    for t in range(1, n_days):
        if t > lookback:
            hi_i = p_indu[t - lookback:t].max()
            hi_t = p_tran[t - lookback:t].max()
            conf = (p_indu[t - 1] >= hi_i * 0.999) and (p_tran[t - 1] >= hi_t * 0.999)
        else:
            conf = False
        confirmed[t] = conf
        boost = edge if (conf and edge != 0.0) else 0.0
        p_indu[t] = p_indu[t - 1] * (1.0 + r_indu[t] + boost)
        p_tran[t] = p_tran[t - 1] * (1.0 + r_tran[t])

    idx = pd.bdate_range("2004-01-02", periods=n_days)
    prices = pd.DataFrame({"indu": p_indu, "tran": p_tran}, index=idx)
    truth = {"edge": edge, "lookback": lookback,
             "confirmed_frac": float(confirmed[lookback + 1:].mean())}
    return prices, truth
