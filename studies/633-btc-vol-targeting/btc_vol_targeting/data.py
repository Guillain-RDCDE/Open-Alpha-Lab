"""Data layer for Study 633 — BTC Vol Targeting.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily BTC-USD closes from yfinance (public, no key), 2014-09-17 onward —
  Bitcoin trades every calendar day, so the tape has ~365 bars/year and all annualisation
  uses 365. Cached as a single-column CSV (``btc_usd.csv``). Price-only == total-return
  for BTC (no dividends); the cash leg is assumed to earn **0%** (documented — a stablecoin
  or exchange cash balance earns nothing; this is conservative *against* the overlay, which
  holds a lot of cash).

* **Synthetic control.** A deterministic, seeded daily-return world with **persistent
  volatility clustering** (log-vol AR(1), GARCH-like — the regime structure vol targeting
  feeds on) and a **tunable vol-return disconnect** knob (same design as study
  591-vol-managed-portfolio):

  - ``disconnect = 0`` (the NULL): the conditional mean scales one-for-one with the
    conditional variance (``mu_t = lam * sig_t^2`` — risk fully priced). In this world
    vol targeting reshuffles risk but earns **no alpha** — the overlay must NOT win,
    however the noise falls.
  - ``disconnect = 2`` (the PLANTED leverage-effect world): the conditional mean FALLS
    as variance rises (vol spikes are uncompensated *and* punished). Here vol targeting
    MUST produce significant alpha; a harness that can't bank it proves nothing.

  Both worlds share the same unconditional mean and the same vol path per seed. Plain
  numpy arrays, no timestamps — immune to the pandas ns-Timestamp 250-year span trap.

Pure numpy + pandas + stdlib for the offline path. ``fetch_btc`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
BTC_CACHE = os.path.join(CACHE_DIR, "btc_usd.csv")

TICKER = "BTC-USD"
DAYS_PER_YEAR = 365          # BTC trades every calendar day

# Study-wide as-of: the last COMPLETE calendar month at build time (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_btc(start: str = "2014-01-01", end: str | None = None,
              path: str = BTC_CACHE, retries: int = 3) -> pd.Series:
    """Download daily BTC-USD closes and cache them. Network-only; run once."""
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            px = yf.download(TICKER, start=start, end=end, auto_adjust=True,
                             progress=False)["Close"]
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.name = "btc_usd"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_frame().to_csv(path)
    return px


def have_real(path: str = BTC_CACHE) -> bool:
    return os.path.exists(path)


def load_btc(path: str = BTC_CACHE, asof: str = AS_OF) -> pd.Series:
    """Daily BTC-USD close series, sliced to the study as-of (sample never creeps)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    return px[px.index <= pd.Timestamp(asof)]


def daily_returns(path: str = BTC_CACHE, asof: str = AS_OF) -> pd.Series:
    """Daily simple returns of BTC-USD (close-to-close, calendar-daily bars)."""
    return load_btc(path=path, asof=asof).pct_change().dropna()


# --------------------------------------------------------------------------- #
# Synthetic control — vol-clustered world with a tunable vol-return disconnect
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 4000, disconnect: float = 0.0, seed: int = 633,
                    sig_bar_daily: float = 0.030, phi: float = 0.985,
                    vol_of_logvol: float = 0.13,
                    mean_ann: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic BTC-like daily-return world with persistent vol clustering.

    Volatility: log-vol AR(1), ``h_t = phi h_{t-1} + vol_of_logvol * u_t``,
    ``sig_t = sig_bar_daily * exp(h_t)`` — multi-month vol regimes around a ~57%
    annualised base, BTC-like.

    Mean: ``mu_t = (1 - disconnect) * lam * sig_t^2 + disconnect * mu_flat`` where
    ``lam`` and ``mu_flat`` are set (per path) so the unconditional daily mean equals
    ``mean_ann / 365`` in EVERY world. ``disconnect = 0`` = fully priced risk (the
    null: vol targeting must NOT earn alpha); ``disconnect = 2`` = the planted
    leverage-effect world (mean falls as variance rises: vol targeting MUST earn
    alpha).

    Returns ``(daily_returns, sig_daily)`` as plain numpy arrays — no timestamps.
    """
    rng = np.random.default_rng(seed)
    h = np.empty(n_days)
    u = rng.standard_normal(n_days)
    h[0] = vol_of_logvol / np.sqrt(1 - phi**2) * u[0]
    for t in range(1, n_days):
        h[t] = phi * h[t - 1] + vol_of_logvol * u[t]
    sig = sig_bar_daily * np.exp(h)

    mu_daily = mean_ann / DAYS_PER_YEAR
    lam = mu_daily / float(np.mean(sig**2))          # priced world: mu_t = lam sig_t^2
    mu_t = (1.0 - disconnect) * lam * sig**2 + disconnect * mu_daily

    z = rng.standard_normal(n_days)
    r = mu_t + sig * z
    # a daily bar cannot lose more than 100%; clip the (rare) extreme synthetic draws
    # so log-wealth stays defined (deterministic, applied identically in every world)
    r = np.maximum(r, -0.95)
    return r, sig
