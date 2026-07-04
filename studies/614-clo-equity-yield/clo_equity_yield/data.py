"""Data layer for Study 614 — CLO Equity Yield ("the 15% machine").

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for the two listed CLO-equity closed-end funds and their
  benchmarks, twice over: once **auto-adjusted (total-return)** and once **price-only**
  (split-adjusted close, no distribution adjustment). The difference between the two monthly
  return series is the **distribution return component** — the ~15% headline "yield" the CLO
  equity pitch sells. Tickers:

      ECC   Eagle Point Credit Co (CLO-equity closed-end fund, IPO 2014-10)
      OXLC  Oxford Lane Capital (CLO-equity fund, the category's biggest, listed 2011-01)
      HYG   iShares iBoxx High Yield Corporate Bond ETF (the credit benchmark)
      SPY   SPDR S&P 500 (the equity benchmark)
      BIL   SPDR 1-3 Month T-Bill ETF (the cash proxy, listed 2007-05)

  Cached wide under ``_cache/ceq_tr.csv`` (total-return) and ``_cache/ceq_px.csv``
  (price-only). The joint ECC window starts 2014-11 (ECC's first complete month); OXLC is
  also decomposed on its own longer window from 2011-02.

* **Synthetic.** A deterministic, fixed-seed monthly world with a **planted distribution
  stream** (a known monthly payout component), a **planted alpha** versus the credit/equity
  benchmark, and tunable HYG/SPY betas. It is the positive control: with ``carry = 0`` the
  distribution-component detector must find nothing; with ``alpha = 0`` the HAC alpha test
  must NOT manufacture significance; planted values must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.

Survivorship note (named on the Signal axis): unlike the mREIT study's REM there is **no
index fund for CLO equity** — the listed category essentially *is* ECC and OXLC, the two
vehicles that made it to 2026. Funds that pivoted away or never scaled are not on this tape,
so the panel is survivor-tilted by construction and the caveat travels with the stamp.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TR_CACHE = os.path.join(CACHE_DIR, "ceq_tr.csv")
PX_CACHE = os.path.join(CACHE_DIR, "ceq_px.csv")

TICKERS = ["ECC", "OXLC", "HYG", "SPY", "BIL"]

# The frozen as-of: June 2026 is the last complete month at build time (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2010-01-01", end: str | None = None,
                tr_path: str = TR_CACHE, px_path: str = PX_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the panel twice (total-return and price-only) and cache both.

    Network-only; used once to build the cache. Yahoo's raw ``Close`` is split-adjusted but
    NOT distribution-adjusted, so ``auto_adjust=True`` minus ``auto_adjust=False`` isolates
    the distribution return — the decomposition this study lives on. (OXLC's 2025-09-08
    1-for-5 reverse split is therefore already handled on both legs.)
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
def synthetic_world(n_months: int = 140, carry: float = 0.012, alpha: float = 0.0,
                    beta_hyg: float = 1.6, beta_spy: float = 0.45,
                    carry_jitter: float = 0.003,
                    seed: int = 614) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic monthly world with a PLANTED distribution stream and a PLANTED alpha.

    The synthetic CLO-equity fund's monthly **total return** is built as

        r_tr = cash + alpha + beta_hyg * hyg_excess + beta_spy * spy_excess + idio

    and its **price-only** return as ``r_px = r_tr - carry_t`` where
    ``carry_t = carry + N(0, carry_jitter)`` — i.e. the fund pays out a noisy monthly
    distribution component with a known mean ``carry`` (real payout streams jitter month to
    month; the jitter also keeps the detector test non-degenerate), while ``alpha`` plants a
    genuine surplus (or deficit) over the credit/equity-matched benchmark. Knobs:

      * ``carry``  — the planted MEAN distribution return per month (0 = no payout: the
        distribution-component HAC t must NOT manufacture significance from the jitter);
      * ``alpha``  — the planted spread vs the factor benchmark (0 = fair pricing: the HAC
        alpha test must NOT manufacture significance).

    Returns (tr_prices, px_prices) wide daily-style frames (monthly stamps) with columns
    ``["CLOE", "HYG", "SPY", "BIL"]``, shaped like ``load_real`` output so the same strategy
    code runs on both. Span ~12 years — far below the 250-year ns-Timestamp trap; the index
    is built from a monthly ``period_range``.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2014-11", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    cash = 0.0015 * np.ones(n_months)                      # ~1.8%/yr bill yield
    hyg_x = rng.normal(0.0020, 0.023, n_months)            # HYG excess return
    spy_x = rng.normal(0.0055, 0.043, n_months)            # SPY excess return
    idio = rng.normal(0.0, 0.040, n_months)                # CLO-equity idiosyncratic noise

    cloe_tr = cash + alpha + beta_hyg * hyg_x + beta_spy * spy_x + idio
    carry_t = carry + rng.normal(0.0, carry_jitter, n_months)
    cloe_px = cloe_tr - carry_t

    def _prices(rets: np.ndarray) -> np.ndarray:
        return 100.0 * np.exp(np.cumsum(np.log1p(np.clip(rets, -0.95, None))))

    tr = pd.DataFrame({
        "CLOE": _prices(cloe_tr), "HYG": _prices(cash + hyg_x),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    px = pd.DataFrame({
        "CLOE": _prices(cloe_px), "HYG": _prices(cash + hyg_x),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    return tr, px
