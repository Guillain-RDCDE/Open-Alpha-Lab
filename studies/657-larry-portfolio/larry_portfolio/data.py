"""Data layer for Study 657 — Larry Portfolio.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for four ETFs, all from yfinance
  (no key), cached as CSV under the study's own ``_cache/``:

  - **IJS** (iShares S&P SmallCap 600 Value) — the small-cap VALUE sleeve Swedroe's "Larry
    Portfolio" concentrates its equity risk budget in. Lists 2000-07-24.
  - **IEF** (iShares 7-10y Treasury) — the safe-bond sleeve, both for the Larry Portfolio's
    70% and the 60/40 benchmark's 40%. Lists 2002-07-30.
  - **SPY** (S&P 500 total return) — the market/large-cap leg of the 60/40 comparator and
    the small-value premium's benchmark.
  - **SHY** (iShares 1-3y Treasury) — the cash / excess-of-cash proxy every Sharpe in this
    study is measured against (same convention as sibling study
    `97-balancing-act <../97-balancing-act/>`_). Lists 2002-07-30.

  IEF and SHY's shared 2002-07-30 inception is the binding constraint on the joint window —
  the same start date as 97-balancing-act, which lets the two studies' 60/40 numbers be
  compared apples-to-apples.

* **Synthetic world.** A deterministic, seeded three-asset generator (market, small-value,
  bond) with a TUNABLE planted small-value premium (knob ``premium``, annualised, over and
  above the market). ``premium = 0`` is the null world: small-value carries the same expected
  return as the market (just a different vol/beta), so the Larry Portfolio (30% SV / 70%
  bond) must NOT out-Sharpe or out-return a 60/40 built on the same market and bond legs.
  This is the machinery/power check — never cited in support of the real-tape stamp.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "lp_prices.csv")

TICKERS = ("IJS", "IEF", "SPY", "SHY")
CASH = "SHY"

# IEF and SHY both list 2002-07-30 — the binding joint-window constraint (IJS lists earlier,
# 2000-07-24). Same start as sibling study 97-balancing-act, for apples-to-apples 60/40s.
START = "2002-07-30"
AS_OF = "2026-06-30"           # last complete month at publication (2026-07-10 run date)

# The small-value premium's well-documented "lost decade": AQR (Asness, Frazzini, Israel &
# Moskowitz, 2015, "Fact, Fiction and Value Investing") and Swedroe's own later writing both
# date the value factor's drawdown to the years following the Global Financial Crisis — an
# externally documented turning point, not one snooped from this study's own numbers.
DECAY_SPLIT = "2007-01-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1999-01-01", end: str = "2026-07-01",
          path: str = PRICES_CACHE, retries: int = 3) -> None:
    """Download the 4 ETF closes once (auto-adjusted, i.e. total-return) and cache them.

    Network-only; never called from the offline/notebook path.
    """
    import time

    import yfinance as yf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    px = None
    for _ in range(retries):
        try:
            raw = yf.download(list(TICKERS), start=start, end=end,
                               auto_adjust=True, progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                px = raw
                break
        except Exception:
            time.sleep(2.0)
    if px is None:
        raise RuntimeError("yfinance download failed after retries")
    px[list(TICKERS)].dropna(how="all").to_csv(path)


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(start: str = START, asof: str = AS_OF,
              path: str = PRICES_CACHE) -> pd.DataFrame:
    """Cached daily total-return price frame, inner-joined across tickers, sliced to
    [start, asof]. Columns: IJS, IEF, SPY, SHY."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()[list(TICKERS)]
    px = px.dropna(how="any")
    px = px.loc[(px.index >= start) & (px.index <= asof)]
    px.index.name = "Date"
    return px


# --------------------------------------------------------------------------- #
# Synthetic world — planted small-value premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(premium: float = 0.0, seed: int = 657,
                    n_days: int = 6000, start: str = "2002-07-30",
                    mu_mkt: float = 0.08, vol_mkt: float = 0.18,
                    beta_sv: float = 1.0, vol_sv_idio: float = 0.12,
                    mu_bond: float = 0.03, vol_bond: float = 0.06,
                    corr_mkt_bond: float = -0.05,
                    ) -> pd.DataFrame:
    """Deterministic 3-asset (MKT, SV, BOND) total-return price frame with a TUNABLE planted
    small-value premium.

    SV (small-value) tracks the market with beta ``beta_sv`` (default 1.0 — CAPM-neutral, so
    the machinery check below isolates ALPHA cleanly, not a beta effect) plus an
    idiosyncratic shock (small caps carry more of it — ``vol_sv_idio``), plus a constant
    daily drift equal to ``premium`` annualised — the ONLY lever that decides whether SV
    earns more than its beta to the market implies. ``premium = 0`` is the clean null: the
    daily (SV - MKT) spread has zero expected value, so the premium-detection statistic (HAC
    *t* on that spread — the same primitive the third axis uses on the real IJS-SPY spread)
    must NOT fire across >= 10 seeds. A positive ``premium`` plants genuine excess return the
    detector should recover. (A 30%-SV/70%-bond blend still mechanically trails a 60%-market
    blend under this null — it holds far less equity beta — that structural gap is the point
    under test, not a bug in the control; see ``strategy.synthetic_detect``.)

    Business-day index, span well under the ~250-year ns-Timestamp ceiling.
    Returns a DataFrame with columns MKT, SV, BOND (price levels starting at 100).
    """
    if n_days > 3_000 * 21:
        raise ValueError("keep the synthetic span under the pandas ns-Timestamp ceiling")
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    cov = np.array([[1.0, corr_mkt_bond], [corr_mkt_bond, 1.0]])
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_days, 2)) @ chol.T
    z_mkt, z_bond = z[:, 0], z[:, 1]
    z_idio = rng.standard_normal(n_days)

    d_mkt = mu_mkt / 252.0 + (vol_mkt / np.sqrt(252.0)) * z_mkt
    d_bond = mu_bond / 252.0 + (vol_bond / np.sqrt(252.0)) * z_bond
    d_sv = (beta_sv * d_mkt + premium / 252.0
            + (vol_sv_idio / np.sqrt(252.0)) * z_idio)

    mkt = 100.0 * np.exp(np.cumsum(d_mkt))
    sv = 100.0 * np.exp(np.cumsum(d_sv))
    bond = 100.0 * np.exp(np.cumsum(d_bond))
    frame = pd.DataFrame({"MKT": mkt, "SV": sv, "BOND": bond}, index=idx)
    frame.index.name = "Date"
    return frame
