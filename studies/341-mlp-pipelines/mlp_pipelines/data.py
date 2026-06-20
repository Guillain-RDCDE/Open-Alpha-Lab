"""Data layer for Study 341 (MLP-Pipelines).

Two tapes, both shaped as monthly frames carrying, per fund, a **price return** and a
**distribution yield** so that total return = price + distribution can be decomposed — the
study's whole point is that the fat AMLP "yield" is largely your own NAV (return of capital)
sitting on a price line that is really an energy bet.

- ``synthetic_mlp`` — a *deterministic, offline* generator that plants a midstream-MLP world:
  the fund's price return is ``beta`` times an energy factor (drift ``energy_mu``, vol
  ``energy_vol``) plus a little idiosyncratic noise, and it pays a steady ``dist`` distribution
  while the NAV erodes by ``nav_drift`` per month. Setting ``beta = 0, nav_drift = 0,
  dist = 0`` is the **null** (the fund is flat, market-neutral, with no payout); a high
  ``beta`` with a fat ``dist`` on a *negative* ``nav_drift`` is the **positive control** — a
  big distribution yield that is pure return of capital on a NAV that tracks (a leveraged
  multiple of) the energy complex. The synthetic is a *machinery proof*, never market
  evidence (METHODOLOGY → the inference bar).

- ``fetch_panel`` — the real Yahoo! monthly **total-return** tape (AMLP + the AMZA/MLPA peers,
  SPY/XLE benchmarks and the USO crude proxy), **cache-first** (parquet under the shared
  ``_cache``) so the test-suite and the reproducible core never touch the network. ``fetch_price
  _and_dist`` additionally pulls split-only price + the dividend stream so the distribution yield
  can be separated from the price (NAV) return on the *real* funds.

No look-ahead lives here; the racing discipline (one execution lag, excess-vs-excess) is in
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

# The MLP "income" wrappers we race, plus the benchmarks they implicitly track.
MLP_FUNDS = ["AMLP", "MLPA", "AMZA"]
BENCH = ["SPY", "XLE", "USO"]   # market, energy sector, crude-oil proxy
TICKERS = MLP_FUNDS + BENCH

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline MLP world
# ---------------------------------------------------------------------------
def synthetic_mlp(
    n_years: int = 12,
    beta: float = 1.10,
    dist: float = 0.0065,
    nav_drift: float = -0.0035,
    energy_mu: float = 0.004,
    energy_vol: float = 0.22,
    idio_vol: float = 0.04,
    start: str = "2014-01-31",
    seed: int = 341,
) -> tuple[pd.DataFrame, dict]:
    """A monthly midstream-MLP world with the income/price/energy split made explicit.

    Each month an **energy factor** draws a log return ``e`` (drift ``energy_mu``, monthly vol
    ``energy_vol/sqrt(12)``). The MLP fund:

      - **price return** = ``beta * e + nav_drift + idio`` — a (typically levered) multiple of
        the energy factor, dragged by a steady NAV bleed ``nav_drift`` (the structural/expense/
        return-of-capital erosion), plus small idiosyncratic noise.
      - **distribution yield** = ``dist`` — a constant payout (the marketed "income").

    Its **total return** is ``price + dist``. Setting ``beta = 0, dist = 0, nav_drift = 0`` makes
    the fund a flat, market-neutral, no-payout null. A high ``beta`` with a fat ``dist`` on a
    *negative* ``nav_drift`` plants the **yield trap**: a large, steady distribution that is pure
    return of capital, riding on a NAV that is really a leveraged energy bet.

    Returns ``(panel, truth)``. ``panel`` is a monthly frame with columns ``energy_tr, mlp_tr,
    mlp_price, mlp_dist`` (all simple monthly returns); ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    # Decorative label only — period_range, never date_range(periods=BIG) (CI overflow guard).
    idx = pd.period_range(start=start[:7], periods=n, freq="M").to_timestamp(how="end")
    idx = idx.normalize()
    idx.name = "date"

    e = energy_mu + (energy_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n)
    energy_tr = np.expm1(e)                                  # the energy factor's total return
    idio = (idio_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n)
    mlp_price = beta * e + nav_drift + idio                  # NAV move = levered energy + bleed
    mlp_dist = np.full(n, dist)                              # the steady "income"
    mlp_tr = mlp_price + mlp_dist                            # MLP total return

    panel = pd.DataFrame(
        {"energy_tr": energy_tr, "mlp_tr": mlp_tr, "mlp_price": mlp_price, "mlp_dist": mlp_dist},
        index=idx,
    )
    truth = {
        "beta": beta,
        "dist": dist,
        "nav_drift": nav_drift,
        "ann_dist_yield": (1.0 + dist) ** MONTHS_PER_YEAR - 1.0,
        "energy_mu": energy_mu,
        "energy_vol": energy_vol,
        "n_years": n_years,
        "seed": seed,
        "is_energy_bet": beta != 0.0,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo monthly, cache-first
# ---------------------------------------------------------------------------
def _panel_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "mlp_pipelines_tr_panel.parquet")


def _split_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "mlp_pipelines_price_dist.parquet")


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly **total-return** (auto-adjusted) returns for the MLP funds + benchmarks.

    Cache-first: reads the shared parquet if present; only touches the network on an explicit
    ``fetch=True`` (then caches). Empty frame on a cache miss with ``fetch=False`` so callers can
    fall back to the synthetic tape gracefully (the notebooks banner which tape they show).
    """
    path = _panel_path(cache_dir)
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network call

    px = yf.download(TICKERS, period="max", interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(path)
    return ret


def fetch_price_and_dist(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Split-only **price** return and **distribution yield**, per fund, monthly.

    This is what lets us separate, on the *real* funds, how much of the total return is the NAV
    moving (price) vs cash handed out (distributions) — the income-illusion decomposition.
    Columns are a MultiIndex ``(ticker, {price, dist})``. Cache-first; empty on a cache miss.
    """
    path = _split_path(cache_dir)
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf

    frames = {}
    for t in MLP_FUNDS:
        tk = yf.Ticker(t)
        # Split-only adjusted price (NOT auto_adjust, so dividends are NOT folded in).
        hist = tk.history(period="max", interval="1mo", auto_adjust=False)
        if hist.empty:
            continue
        hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
        px = hist["Close"].resample("ME").last()
        divs = hist.get("Dividends", pd.Series(dtype=float)).resample("ME").sum()
        price_ret = px.pct_change()
        dist_yield = (divs / px.shift(1)).reindex(price_ret.index).fillna(0.0)
        frames[(t, "price")] = price_ret
        frames[(t, "dist")] = dist_yield
    out = pd.DataFrame(frames).dropna(how="all")
    out.index.name = "date"
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["ticker", "leg"])
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(path)
    return out


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a returns panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.to_numpy(dtype=float, na_value=0.0))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
