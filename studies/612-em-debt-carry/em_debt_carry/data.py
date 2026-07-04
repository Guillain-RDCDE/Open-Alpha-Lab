"""Data layer for Study 612 — EM Debt Carry.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for the emerging-market sovereign debt complex and its
  benchmarks, twice over: once **auto-adjusted (total-return)** and once **price-only**
  (split-adjusted close, no dividend adjustment). The difference between the two monthly
  return series is the **coupon / distribution component** — the "fat spread over
  Treasuries" the pitch sells. Tickers:

      EMB   iShares J.P. Morgan USD Emerging Markets Bond ETF (hard-currency EM
            sovereigns, EMBI Global Core; listed 2007-12 — the GFC is on the tape)
      EMLC  VanEck J.P. Morgan EM Local Currency Bond ETF (the local-currency
            contrast — same borrowers, but you also own their currencies; listed 2010-07)
      IEF   iShares 7-10 Year Treasury ETF (the duration-matched Treasury benchmark;
            EMB's effective duration ~7y sits right on IEF's 7-8y)
      SPY   SPDR S&P 500 (the risk-off conditioning variable / equity-beta factor)
      BIL   SPDR 1-3 Month T-Bill ETF (the cash proxy; excess-vs-excess races)

  Cached wide under ``_cache/edc_tr.csv`` (total-return) and ``_cache/edc_px.csv``
  (price-only). The common EMB+BIL window starts 2008-01 — deliberately *including* the
  GFC; the EMLC contrast window starts 2010-08.

* **Synthetic.** A deterministic, fixed-seed monthly world with a **planted carry spread**
  over the duration benchmark, a **planted coupon** (dividend component) and a **planted
  risk-off beta** (the crash coupling to equities). It is the positive control: with
  ``spread = 0`` and ``beta_spy = 0`` the HAC spread test must NOT manufacture
  significance; planted values must light up, including the conditional risk-off gap.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.

Survivorship note (named on the Signal axis): EMB and EMLC are index *funds* — their
returns already contain every default, restructuring and index ejection (Venezuela 2017,
Lebanon 2020, Russia 2022, Ukraine 2022...) that happened inside the index. The fund tape
is not survivor-biased; what it *is* biased by is the index rulebook (issue-size floors,
liquidity screens), which we state rather than hide.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TR_CACHE = os.path.join(CACHE_DIR, "edc_tr.csv")
PX_CACHE = os.path.join(CACHE_DIR, "edc_px.csv")

TICKERS = ["EMB", "EMLC", "IEF", "SPY", "BIL"]

# The frozen as-of: June 2026 is the last complete month at build time (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                tr_path: str = TR_CACHE, px_path: str = PX_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the panel twice (total-return and price-only) and cache both.

    Network-only; used once to build the cache. Yahoo's raw ``Close`` is split-adjusted
    but NOT dividend-adjusted, so ``auto_adjust=True`` minus ``auto_adjust=False``
    isolates the coupon/distribution return — the decomposition this study lives on.
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
def synthetic_world(n_months: int = 222, spread: float = 0.0, coupon: float = 0.0045,
                    beta_ief: float = 1.0, beta_spy: float = 0.0,
                    coupon_jitter: float = 0.0008,
                    seed: int = 612) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic monthly world with a PLANTED carry spread and risk-off coupling.

    The synthetic EM bond fund's monthly **total return** is built as

        r_tr = cash + spread + beta_ief * ief_excess + beta_spy * spy_excess + idio

    and its **price-only** return as ``r_px = r_tr - coupon_t`` where
    ``coupon_t = coupon + N(0, coupon_jitter)`` — the fund distributes a noisy monthly
    coupon with known mean. Knobs:

      * ``spread``   — the planted MEAN total-return spread per month over the duration
        benchmark (0 = no carry premium: the EMD-minus-IEF HAC t must NOT manufacture
        significance from noise);
      * ``beta_spy`` — the planted risk-off coupling (0 = no crash profile: the
        SPY-worst-decile conditional gap must NOT reach significance; positive values
        must produce a significantly negative spread in equity-crash months);
      * ``coupon``   — the planted distribution yield (the yield-pickup detector's truth).

    Returns (tr_prices, px_prices) wide monthly-stamped frames with columns
    ``["EMD", "IEF", "SPY", "BIL"]``, shaped like ``load_real`` output so the same
    strategy code runs on both. Span 18.5 years — far below the 250-year ns-Timestamp
    trap; the index is built from a monthly ``period_range``.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2008-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    cash = 0.0012 * np.ones(n_months)                      # ~1.4%/yr bill yield
    ief_x = rng.normal(0.0012, 0.017, n_months)            # IEF excess return
    spy_x = rng.normal(0.0060, 0.044, n_months)            # SPY excess return
    idio = rng.normal(0.0, 0.012, n_months)                # EM idiosyncratic noise

    emd_tr = cash + spread + beta_ief * ief_x + beta_spy * spy_x + idio
    coupon_t = coupon + rng.normal(0.0, coupon_jitter, n_months)
    emd_px = emd_tr - coupon_t

    def _prices(rets: np.ndarray) -> np.ndarray:
        return 100.0 * np.exp(np.cumsum(np.log1p(np.clip(rets, -0.95, None))))

    ief_coupon = 0.0020 + rng.normal(0.0, 0.0002, n_months)   # a small Treasury coupon
    tr = pd.DataFrame({
        "EMD": _prices(emd_tr), "IEF": _prices(cash + ief_x),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    px = pd.DataFrame({
        "EMD": _prices(emd_px), "IEF": _prices(cash + ief_x - ief_coupon),
        "SPY": _prices(cash + spy_x), "BIL": _prices(cash),
    }, index=idx)
    return tr, px
