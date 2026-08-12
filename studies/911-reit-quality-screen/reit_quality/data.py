"""Data layer for Study 911 — REIT Quality Screen.

The claim under test: **not all REITs are equal.** Equity REITs with durable rents and
modest balance-sheet leverage (residential/broad-property funds) are a very different
animal from **mortgage REITs**, which lever a thin spread between long mortgage assets and
short funding and pay it out as a fat but fragile dividend. The "quality REIT" screen —
hold the durable-income equity sleeve, screen *out* the leveraged-carry sleeve — is meant
to deliver a **better risk-adjusted return than the broad REIT index, net of costs**,
separating the durable-income REIT premium from the leveraged-carry trap.

We test it on the actual liquid vehicles (yfinance, total-return closes):

  * **VNQ** — Vanguard Real Estate ETF: the **broad REIT index** everyone holds. It is an
    *equity*-REIT fund by construction (mortgage REITs are a tiny sliver), so it is both the
    benchmark and, quietly, already a light "quality" screen.
  * **REZ** — iShares Residential & Multisector Real Estate ETF: the **durable-income
    quality sleeve** (apartments, healthcare, storage — the stickiest rents).
  * **RWR** — SPDR Dow Jones REIT ETF: a second broad equity-REIT index (issuer/method
    cross-check on the benchmark).
  * **XLRE** — Real Estate Select Sector SPDR (2015-10+): the S&P-500 real-estate sector
    sleeve, a shorter cross-check.
  * **REM** — iShares Mortgage Real Estate ETF: the **leveraged-carry trap** — mortgage
    REITs, the sleeve the quality screen exists to avoid.
  * **SPY** — S&P 500, the equity control (how much of a REIT's Sharpe is just equity beta).
  * **BIL** — 1-3m T-bill ETF: the tradable risk-free leg, so every Sharpe race is
    **excess-vs-excess** and every alpha is measured over cash.

Real-tape path is deterministic (no RNG): every number is a function of the cached tape.

Synthetic world: a deterministic, fixed-seed monthly generator with a **broad** benchmark,
a **quality** leg built as ``broad + planted_edge`` (knob ``edge_ann``; null at 0), and a
**trap** leg built as a higher-vol, negatively-drifting leveraged-carry sleeve. It is the
machinery proof — the Sharpe-advantage / HAC-spread estimators must recover a planted
quality edge, must stay silent on the null, and must flag the trap's inferior Sharpe. Never
cited as market evidence. Its index is a monthly ``period_range`` kept safely inside the
pandas ns-Timestamp horizon.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "reit_prices.parquet")

# Broad index + quality sleeve + broad cross-check + sector + the trap + equity + cash.
TICKERS = ["VNQ", "REZ", "RWR", "XLRE", "REM", "SPY", "BIL"]

# The role map used across the study.
BROAD = "VNQ"           # the broad REIT index (benchmark)
QUALITY = "REZ"         # the durable-income quality sleeve (residential/multisector)
QUALITY_BLEND = ["VNQ", "REZ", "RWR"]   # equal-weight equity-REIT "quality book"
TRAP = "REM"            # mortgage REITs — the leveraged-carry trap
EQUITY = "SPY"          # equity control
CASH = "BIL"            # tradable risk-free

AS_OF = "2026-06-30"    # last complete calendar month at build time (partial month dropped)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2005-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the ETF panel (``auto_adjust=True`` -> total-return closes) and cache it.

    Network-only; used once to build the cache. Guards yfinance flakiness with retries.
    Writes a wide parquet (index = date, columns = tickers).
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None:
        raise RuntimeError("yfinance download failed after retries")
    prices = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_parquet(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide total-return close frame, sliced to the frozen as-of (cache-first)."""
    if not os.path.exists(path):
        return fetch(path=path, end=asof)
    px = pd.read_parquet(path).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof — planted quality edge + null + trap)
# --------------------------------------------------------------------------- #
def synthetic_world(edge_ann: float = 0.0, n_months: int = 228, seed: int = 911,
                    broad_mu: float = 0.006, broad_vol: float = 0.055,
                    idio_vol: float = 0.010,
                    trap_beta: float = 1.05, trap_drag_ann: float = 0.06,
                    trap_extra_vol: float = 0.02) -> pd.DataFrame:
    """Deterministic monthly-return world: BROAD, QUAL, TRAP, CASH.

    * ``BROAD`` — the broad REIT index: ``N(broad_mu, broad_vol)``.
    * ``QUAL``  — the quality sleeve: ``BROAD + edge_ann/12 + idio noise``. ``edge_ann`` is
      the TUNABLE planted annual quality edge; ``edge_ann=0`` is the **null** (the
      Sharpe-advantage / HAC-spread estimators must NOT manufacture an edge from it).
    * ``TRAP``  — the leveraged-carry sleeve: ``trap_beta * BROAD - trap_drag_ann/12`` plus
      its own extra vol — a higher-beta, negatively-drifting book whose Sharpe is
      structurally worse (the mortgage-REIT trap the screen exists to avoid).
    * ``CASH``  — a small positive risk-free drift.

    Index is a monthly ``period_range`` kept as timestamps well under the ~250-year
    ns-Timestamp cap. No look-ahead: every column is contemporaneous monthly returns.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2007-06", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    broad = rng.normal(broad_mu, broad_vol, n_months)
    qual = broad + edge_ann / 12.0 + rng.normal(0.0, idio_vol, n_months)
    trap = (trap_beta * broad - trap_drag_ann / 12.0
            + rng.normal(0.0, trap_extra_vol, n_months))
    cash = np.full(n_months, 0.02 / 12.0) + rng.normal(0.0, 0.0002, n_months)
    return pd.DataFrame({"BROAD": broad, "QUAL": qual, "TRAP": trap, "CASH": cash},
                        index=idx)
