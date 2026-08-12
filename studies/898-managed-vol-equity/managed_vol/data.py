"""Data layer for Study 898 — Managed-Vol Equity.

The claim under test (Moreira & Muir 2017, *"Volatility-Managed Portfolios"*, JF): scale
your equity exposure **inversely to recent volatility** — lean in when the tape is calm,
step out when it is stormy — and the risk-adjusted return (Sharpe) *rises*. Here we take
the self-contained single-asset version: a pure volatility **thermostat** on SPY that
targets a constant ~12% annualised vol by holding

    w_t = min(cap, target_vol / RV_{t-1}(window))

of SPY and the rest in **bills** (BIL, a real total-return T-bill ETF — the honest cash
leg, not an assumed 0%). Every race is **excess-of-cash on both legs** (SPY − BIL vs the
managed book − BIL), so the comparison is a clean risk-adjusted one and the cash yield
cancels.

Two ingredients, both offline-friendly once cached.

* **Real tape.** Daily **total-return** closes (`auto_adjust=True`) for **SPY** (the equity
  leg) and **BIL** (the 1–3-month T-bill ETF, the cash leg), pulled with yfinance (public,
  no key). BIL's live history starts **2007-05-30**, so the excess-of-cash sample begins
  there — ~19 years spanning the GFC, the 2018 vol shock, COVID-2020 and the 2022 bear.
  Cached as a single aligned parquet (`spy_bil.parquet`). `auto_adjust=True` ⇒ dividends
  are in the price, so this is total-return on both legs.

* **Synthetic control — the positive control.** A deterministic, seeded daily-return world
  with **persistent volatility clustering** (log-vol AR(1), GARCH-like — the regime
  structure the thermostat feeds on) and a **tunable vol-return disconnect** knob (same
  design as study 633-btc-vol-targeting):

  - ``disconnect = 0`` (the NULL): the conditional mean scales one-for-one with the
    conditional variance (``mu_t = lam * sig_t^2`` — risk fully priced). Vol targeting
    reshuffles risk but earns **no alpha** — the overlay must NOT win.
  - ``disconnect = 2`` (the PLANTED leverage-effect world): the conditional mean FALLS as
    variance rises (vol spikes uncompensated *and* punished). Vol targeting MUST produce
    significant timing alpha; a harness that can't bank it proves nothing.

  Both worlds share the same unconditional mean and the same vol path per seed. Plain numpy
  arrays, no timestamps — immune to the pandas ns-Timestamp horizon trap. The synthetic cash
  leg is a flat 0% (the machinery test needs no real bills).

Pure numpy + pandas + stdlib for the offline path. ``fetch`` (network) builds the cache once
and is never imported by the notebooks' offline cells; ``load_prices`` reads the parquet
directly (no yfinance import).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE = os.path.join(CACHE_DIR, "spy_bil.parquet")

TICKERS = ("SPY", "BIL")
DAYS_PER_YEAR = 252            # US equity trading calendar

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"
START = "2005-01-01"           # request start; BIL's real history begins 2007-05-30

__all__ = [
    "TICKERS", "DAYS_PER_YEAR", "AS_OF", "START", "CACHE", "CACHE_DIR",
    "fetch", "have_real", "load_prices", "excess_returns", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None, path: str = CACHE,
          retries: int = 4) -> pd.DataFrame:
    """Download daily total-return closes for SPY and BIL; cache them aligned.

    Network-only; run once to build the cache. ``auto_adjust=True`` ⇒ total-return.
    """
    import yfinance as yf

    cols = {}
    for tkr in TICKERS:
        px = None
        for _ in range(retries):
            try:
                raw = yf.download(tkr, start=start, end=end, auto_adjust=True,
                                  progress=False)
                px = raw["Close"]
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
                px = px.dropna()
                if len(px) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if px is None or len(px) == 0:
            raise RuntimeError(f"could not fetch {tkr}")
        px.index = pd.DatetimeIndex(px.index).tz_localize(None)
        cols[tkr] = px

    df = pd.DataFrame(cols).dropna().sort_index()
    df.index.name = "date"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return df


def have_real(path: str = CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Aligned daily total-return closes (columns SPY, BIL), sliced to the as-of.

    OFFLINE — reads the cached parquet directly, no yfinance import. The sample is
    pinned at ``asof`` so it never creeps forward.
    """
    df = pd.read_parquet(path).sort_index()
    df = df[df.index <= pd.Timestamp(asof)]
    return df[list(TICKERS)].dropna()


def excess_returns(path: str = CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Daily simple returns: SPY, BIL (cash), and ``spy_excess = SPY − BIL``.

    The excess-of-cash SPY return is the fair benchmark leg; the managed book is also
    measured excess of BIL, so the cash yield cancels in every race.
    """
    px = load_prices(path=path, asof=asof)
    r = px.pct_change().dropna()
    out = pd.DataFrame({
        "spy": r["SPY"],
        "cash": r["BIL"],
        "spy_excess": r["SPY"] - r["BIL"],
    })
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — vol-clustered returns with a tunable vol-return disconnect
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 4000, disconnect: float = 0.0, seed: int = 898,
                    sig_bar_daily: float = 0.009, phi: float = 0.985,
                    vol_of_logvol: float = 0.16,
                    mean_ann: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic equity-like daily excess-return world with vol clustering.

    Volatility: log-vol AR(1), ``h_t = phi h_{t-1} + vol_of_logvol * u_t``,
    ``sig_t = sig_bar_daily * exp(h_t)`` — multi-month vol regimes around a ~14%
    annualised base, SPY-like.

    Mean: ``mu_t = (1 - disconnect) * lam * sig_t^2 + disconnect * mu_flat`` where ``lam``
    and ``mu_flat`` are set (per path) so the unconditional daily mean equals
    ``mean_ann / 252`` in EVERY world. ``disconnect = 0`` = fully priced risk (the null:
    vol targeting must NOT earn alpha); ``disconnect = 2`` = the planted leverage-effect
    world (mean falls as variance rises: vol targeting MUST earn timing alpha).

    Returns ``(daily_excess_returns, sig_daily)`` as plain numpy arrays — no timestamps.
    The synthetic cash leg is a flat 0%, so these ARE the excess-of-cash returns.
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
    # a daily bar cannot lose more than 100%; clip the (rare) extreme synthetic draws so
    # log-wealth stays defined (deterministic, applied identically in every world)
    r = np.maximum(r, -0.95)
    return r, sig
