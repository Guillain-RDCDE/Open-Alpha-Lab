"""Data layer for Study 886 — Agency MBS Carry.

The claim under test: **agency mortgage-backed securities carry a spread over
duration-matched Treasuries** as compensation for prepayment (negative-convexity) risk,
so a duration-neutral long-MBS / short-Treasury book should earn a **real, positive
carry**. We test whether the realized excess return of an agency-MBS ETF over a
duration-matched Treasury leg is a genuine, robust premium (HAC *t*, bootstrap CI, era
cut) and whether it survives costs.

Instruments (yfinance, ``auto_adjust=True`` **total-return** closes):

  * **MBB** — iShares MBS ETF (agency pass-throughs), the longest MBS tape (2007-03+).
  * **VMBS** — Vanguard Mortgage-Backed Securities ETF (2009-11+), the corroborating MBS
    tape.
  * **IEF** — iShares 7-10y Treasury ETF, the duration-matched Treasury leg (its ~7.5y
    effective duration brackets MBB/VMBS's ~6y OAD; we duration-match empirically and by
    the published OAD ratio).
  * **AGG** — iShares Core US Aggregate Bond ETF, a broad-market reference.
  * **BIL** — SPDR 1-3 Month T-Bill ETF, the **cash / risk-free** leg (every return is
    taken excess of BIL — the excess-vs-excess rail).

The carry we harvest is the **duration-neutral** spread
``carry_t = (MBS_t − cash_t) − β·(IEF_t − cash_t)`` where ``β`` is the realized rate
sensitivity of the MBS leg on the Treasury leg (an *empirical* duration match that
already prices in negative convexity); a static ``β = OAD_MBS / OAD_IEF ≈ 0.80`` variant
is reported alongside. Both legs are excess-of-cash, so the residual cash weight
``(1 − β)`` cancels and the spread is pure carry.

Synthetic world: a deterministic monthly generator with a **tunable planted carry**
(``carry_annual``; null at 0) plus a shared rate factor and idiosyncratic noise — the
positive control that shows the duration-neutral estimator is unbiased. Index built with
``period_range`` kept safely under the ns-Timestamp horizon.

Cache-first: ``fetch`` (network, yfinance) runs once and writes
``_cache/mbs_prices.parquet``; everything else reads that cache OFFLINE.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "mbs_prices.parquet")

# ETF total-return closes: two agency-MBS funds, the duration-matched Treasury leg,
# the aggregate-bond reference, and the T-bill cash leg.
TICKERS = ["MBB", "VMBS", "IEF", "AGG", "BIL"]

AS_OF = "2026-06-30"  # last complete calendar month at build time (drop the partial month)

# Published effective durations (fund pages, mid-2026, years) — used for the static
# duration-ratio hedge alongside the empirical (regression) hedge.
OAD = {"MBB": 6.0, "VMBS": 6.0, "IEF": 7.5, "AGG": 6.0}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2007-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the ETF total-return closes and cache them as parquet (network, run once)."""
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
        raise RuntimeError("yfinance returned no data for the MBS carry tape")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide total-return close frame (MBB, VMBS, IEF, AGG, BIL), cache-first."""
    if not os.path.exists(path):
        return fetch(path=path)
    return pd.read_parquet(path).sort_index()


def monthly_panel(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly simple-return panel for every ticker, sliced to the as-of month-end.

    The partial current month is dropped (``asof`` = last complete calendar month).
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]
    return m.pct_change()


def excess_frame(panel: pd.DataFrame, mbs: str = "MBB", treasury: str = "IEF",
                 cash: str = "BIL", start: str | None = None,
                 end: str | None = None) -> pd.DataFrame:
    """Aligned monthly **excess-of-cash** returns for one MBS/Treasury pair.

    Columns ``mbs`` and ``ief`` are the MBS and Treasury total returns minus the BIL
    cash return (the excess-vs-excess rail). Rows with any missing leg are dropped so
    the pair starts at the later of the two ETF inception dates.
    """
    df = pd.DataFrame({
        "mbs": panel[mbs] - panel[cash],
        "ief": panel[treasury] - panel[cash],
    }).dropna()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 228, carry_annual: float = 0.02, seed: int = 886,
                    beta_true: float = 0.55, rate_vol: float = 0.018,
                    noise_vol: float = 0.004) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED duration-neutral MBS carry.

    A shared monthly **rate factor** ``f ~ N(0, rate_vol)`` drives both legs; the
    Treasury excess return is ``ief = f + small noise`` and the MBS excess return is
    ``mbs = beta_true·f + carry_annual/12 + eps``. The duration-neutral estimator
    ``mbs − β·ief`` (with ``β`` fit by regression) therefore recovers ``carry_annual/12``
    plus mean-zero noise. ``carry_annual = 0`` is the null: no carry to find, and the
    estimator must not manufacture one.

    Returns an ``mbs`` / ``ief`` excess-of-cash frame, same shape as
    :func:`excess_frame`. Decorative monthly index via ``period_range`` (kept well under
    the ns-Timestamp horizon).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2007-06", periods=n_months, freq="M").to_timestamp(how="end").normalize()
    f = rng.normal(0.0, rate_vol, n_months)
    ief = f + rng.normal(0.0, noise_vol, n_months)
    mbs = beta_true * f + carry_annual / 12.0 + rng.normal(0.0, noise_vol, n_months)
    return pd.DataFrame({"mbs": mbs, "ief": ief}, index=idx)
