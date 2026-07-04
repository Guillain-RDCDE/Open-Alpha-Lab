"""Data layer for Study 611 — mREIT Carry.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for the mortgage-REIT complex and its benchmarks, twice over:
  once **auto-adjusted (total-return)** and once **price-only** (split-adjusted close, no
  dividend adjustment). The difference between the two monthly return series is the **dividend
  return component** — the "carry" the 10-14% headline yield claims to deliver. Tickers:

      REM   iShares Mortgage Real Estate ETF (the category bellwether, listed 2007-05)
      NLY   Annaly Capital Management (the largest agency mREIT, listed 1997)
      AGNC  AGNC Investment Corp (agency mREIT, listed 2008-05)
      IEF   iShares 7-10 Year Treasury ETF (the duration benchmark)
      SPY   SPDR S&P 500 (the equity benchmark)
      BIL   SPDR 1-3 Month T-Bill ETF (the cash / financing-rate proxy, listed 2007-05)

  Cached wide under ``_cache/mrc_tr.csv`` (total-return) and ``_cache/mrc_px.csv``
  (price-only). The common REM+BIL window starts 2007-06 — deliberately *including* the GFC.

* **Synthetic.** A deterministic, fixed-seed monthly three-factor world with a **planted
  carry** (a known monthly dividend component), a **planted alpha** versus the
  duration-matched benchmark, and tunable IEF/SPY betas. It is the positive control: with
  ``carry = 0`` the dividend-component detector must find nothing; with ``alpha = 0`` the
  HAC alpha test must NOT manufacture significance; planted values must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.

Survivorship note (named on the Signal axis): REM is an index *fund*, so its return already
contains every mortgage REIT that blew up inside the index — the fund tape is not
survivor-biased. The two single names (NLY, AGNC) *are* survivors — the two biggest agency
mREITs still standing — quoted as colour, never as the headline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TR_CACHE = os.path.join(CACHE_DIR, "mrc_tr.csv")
PX_CACHE = os.path.join(CACHE_DIR, "mrc_px.csv")

TICKERS = ["REM", "NLY", "AGNC", "IEF", "SPY", "BIL"]

# The frozen as-of: June 2026 is the last complete month at build time (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2003-01-01", end: str | None = None,
                tr_path: str = TR_CACHE, px_path: str = PX_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the panel twice (total-return and price-only) and cache both.

    Network-only; used once to build the cache. Yahoo's raw ``Close`` is split-adjusted but
    NOT dividend-adjusted, so ``auto_adjust=True`` minus ``auto_adjust=False`` isolates the
    dividend return — the decomposition this study lives on.
    """
    import yfinance as yf

    def _dl(auto_adjust: bool) -> pd.DataFrame:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(TICKERS, start=start, end=end, auto_adjust=auto_adjust,
                                  progress=False)["Close"]
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        return raw.dropna(how="all")

    tr = _dl(True)
    px = _dl(False)
    os.makedirs(os.path.dirname(tr_path), exist_ok=True)
    tr.to_csv(tr_path)
    px.to_csv(px_path)
    return tr, px


def have_real(tr_path: str = TR_CACHE, px_path: str = PX_CACHE) -> bool:
    return os.path.exists(tr_path) and os.path.exists(px_path)


def load_real(tr_path: str = TR_CACHE, px_path: str = PX_CACHE,
              as_of: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (total-return closes, price-only closes), sliced to the frozen as-of."""
    tr = pd.read_csv(tr_path, index_col=0, parse_dates=True).sort_index()
    px = pd.read_csv(px_path, index_col=0, parse_dates=True).sort_index()
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        tr = tr[tr.index <= cut]
        px = px[px.index <= cut]
    return tr, px


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 228, carry: float = 0.010, alpha: float = 0.0,
                    beta_ief: float = 2.5, beta_spy: float = 0.55,
                    carry_jitter: float = 0.003,
                    seed: int = 611) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic monthly world with a PLANTED carry and a PLANTED benchmark alpha.

    The synthetic mREIT's monthly **total return** is built as

        r_tr = cash + alpha + beta_ief * ief_excess + beta_spy * spy_excess + idio

    and its **price-only** return as ``r_px = r_tr - carry_t`` where
    ``carry_t = carry + N(0, carry_jitter)`` — i.e. the fund pays out a noisy monthly
    dividend component with a known mean ``carry`` (real payout streams jitter month to
    month; the jitter also keeps the detector test non-degenerate), while ``alpha`` plants
    a genuine surplus (or deficit) over the duration/equity-matched benchmark. Knobs:

      * ``carry``  — the planted MEAN dividend return per month (0 = no carry: the
        dividend-component HAC t must NOT manufacture significance from the payout jitter);
      * ``alpha``  — the planted spread vs the factor benchmark (0 = fair pricing: the HAC
        alpha test must NOT manufacture significance).

    Returns (tr_prices, px_prices) wide daily-style frames (monthly stamps) with columns
    ``["MREIT", "IEF", "SPY", "BIL"]``, shaped like ``load_real`` output so the same
    strategy code runs on both. Span 19 years — far below the 250-year ns-Timestamp trap;
    the index is built from a monthly ``period_range``.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2007-06", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    cash = 0.0015 * np.ones(n_months)                      # ~1.8%/yr bill yield
    ief_x = rng.normal(0.0010, 0.018, n_months)            # IEF excess return
    spy_x = rng.normal(0.0055, 0.043, n_months)            # SPY excess return
    idio = rng.normal(0.0, 0.028, n_months)                # mREIT idiosyncratic noise

    mreit_tr = cash + alpha + beta_ief * ief_x + beta_spy * spy_x + idio
    carry_t = carry + rng.normal(0.0, carry_jitter, n_months)
    mreit_px = mreit_tr - carry_t

    def _prices(rets: np.ndarray) -> np.ndarray:
        return 100.0 * np.exp(np.cumsum(np.log1p(np.clip(rets, -0.95, None))))

    tr = pd.DataFrame({
        "MREIT": _prices(mreit_tr), "IEF": _prices(cash + ief_x),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    px = pd.DataFrame({
        "MREIT": _prices(mreit_px), "IEF": _prices(cash + ief_x),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    return tr, px
