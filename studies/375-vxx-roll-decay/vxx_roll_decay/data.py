"""Data layer for Study 375 — VXX-Roll-Decay (short-vol carry + the crash tail).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a tradable VIX-futures ETP and the VIX term
  structure (yfinance, no key), cached under ``_cache/vol_prices.csv`` (a wide CSV indexed
  by date). We use **VIXY** (ProShares VIX Short-Term Futures ETF) as the tradable short-vol
  vehicle: it has a *continuous* split-adjusted history from 2011 (the iPath VXX ETN was
  redeemed and re-issued in 2018, so a clean yfinance ``VXX`` series only starts in 2018 —
  VIXY is the same constant-maturity short-dated VIX-futures object with a longer tape and
  is what we short here, named as a VXX-equivalent throughout). The term structure
  (``^VIX`` spot vs ``^VIX3M``) measures **contango** — the roll-yield engine.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable
  short-vol ETP path: a steady **contango carry** knob (``edge`` = daily expected decay of
  the long ETP = daily expected gain of the short) plus an injected **crash** (a vol spike
  that blows the ETP up and the short down). It is the positive control: with ``edge=0`` the
  short carry must NOT register significance; with a large planted carry it MUST light up —
  while the planted crash always reproduces the fat left tail.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "vol_prices.csv")

# The tradable short-vol ETP (VIXY) plus the VIX term structure used to gauge contango.
TICKERS = ["VIXY", "^VIX", "^VIX9D", "^VIX3M", "SPY"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2011-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the VIX-ETP + term-structure tape via yfinance and cache a wide CSV.

    Network-only; used once to build ``_cache/vol_prices.csv``. Never imported by the offline
    notebook cells. Keeps the rows where the tradable ETP (VIXY) is present.
    """
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cached prices -> a tidy short-vol frame.

    Returns a frame indexed by date with columns:
      * ``etp``      — the tradable VIX-futures ETP (VIXY) adjusted close (the thing we short)
      * ``contango`` — VIX3M / VIX (>1 = contango = roll-yield tailwind for the short)
      * ``spy``      — SPY adjusted close (for the crash-correlation context)

    Days where the ETP is missing are dropped.
    """
    df = load_prices(path)
    out = pd.DataFrame({
        "etp": df["VIXY"],
        "contango": df["^VIX3M"] / df["^VIX"],
        "spy": df["SPY"] if "SPY" in df.columns else np.nan,
    })
    return out.dropna(subset=["etp"])


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_etp(n_days: int = 3_800, edge: float = 0.0015, crash_mag: float = 0.15,
                  n_crashes: int = 3, seed: int = 375,
                  sig_daily: float = 0.030) -> pd.DataFrame:
    """Deterministic short-vol ETP path: a contango carry knob + injected crashes.

    The **long** ETP loses ``edge`` per day in expectation (the contango bleed) plus mean-zero
    noise, so the **short** earns ``edge`` per day. Every ~1/(n_crashes+1) of the way through,
    a **crash** adds a multi-day vol spike (the ETP jumps up, the short loses), inflicting the
    fat left tail. ``crash_mag`` controls the spike size; it is deliberately calibrated so the
    crashes deliver the *negative skew / kurtosis signature* WITHOUT swamping the carry mean —
    so the ``edge`` knob, not the crashes, is what moves the HAC t. ``contango`` sits above 1
    except during crashes, where it inverts (backwardation), exactly like the real tape.

    Knob semantics (the positive control):
      * ``edge = 0``   -> the short has NO carry, only the crash drag: the HAC t is NOT
        significantly **positive** (placebo p ~ 1) — the engine never invents a carry that
        isn't there.
      * ``edge`` large -> a real planted carry: the HAC t clears 2 and the placebo p collapses.
      * the planted crash always reproduces the negative skew / fat tail, whatever ``edge`` is.

    Deterministic for fixed ``seed``; pure numpy. The date index is a decorative business-day
    ``bdate_range`` label (kept short, so no OutOfBounds risk).
    """
    rng = np.random.default_rng(seed)
    # long-ETP daily log return: a steady negative drift (-edge) + noise. short = -that.
    r = rng.normal(-edge, sig_daily, size=n_days)
    contango = np.full(n_days, 1.0) + rng.normal(0.10, 0.02, size=n_days)  # ~1.10, contango

    if n_crashes > 0:
        step = n_days // (n_crashes + 1)
        for k in range(n_crashes):
            s = step * (k + 1) + int(rng.integers(0, step // 4))
            dur = int(rng.integers(3, 6))
            end = min(s + dur, n_days)
            # vol spike: ETP jumps UP (short loses) over `dur` days; a deterministic
            # front-loaded decay profile so the spike is sharp but bounded (no runaway
            # kurtosis), reproducing a realistic crash-day signature.
            for j in range(s, end):
                r[j] += crash_mag * (0.6 ** (j - s))
            # term structure inverts (backwardation) around the crash
            end5 = min(end + 5, n_days)
            contango[s:end5] = 0.92 + rng.normal(0, 0.02, size=end5 - s)

    contango = np.clip(contango, 0.80, 1.30)
    etp = 1000.0 * np.exp(np.cumsum(r))
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, size=n_days)))
    idx = pd.bdate_range("2008-01-01", periods=n_days)
    return pd.DataFrame({"etp": etp, "contango": contango, "spy": spy}, index=idx)
