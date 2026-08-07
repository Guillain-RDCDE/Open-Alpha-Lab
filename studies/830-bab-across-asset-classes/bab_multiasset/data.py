"""Data layer for Study 830 — BAB Across Asset Classes.

The claim under test (Frazzini & Pedersen 2014, *"Betting Against Beta"*, extended
to the cross-asset level): a **flat security-market line** is not a stocks-only
anomaly — it shows up *across asset classes* too. Rank a basket of asset classes by
their beta to a common multi-asset market portfolio; the **low-beta** assets earn
too much per unit of risk and the **high-beta** assets too little. A book that goes
long the low-beta assets (levered up to unit beta) and short the high-beta assets
(de-levered down to unit beta) is ex-ante beta-neutral and should earn a positive
alpha — the **BAB factor**.

Two tapes, one schema (a tz-naive daily frame of total-return closes, one column per
asset class):

* **Real tape — nine liquid asset-class ETFs.** ``TICKERS`` spans US equity (SPY),
  developed-ex-US equity (EFA), emerging-market equity (EEM), long Treasuries (TLT),
  investment-grade credit (LQD), high-yield credit (HYG), gold (GLD), broad
  commodities (DBC) and US real estate (VNQ). Daily total-return closes via yfinance
  (``auto_adjust=True``), cached under this study's OWN ``_cache/`` as a parquet.
  ``fetch()`` (network, retried) runs once to build the cache; ``load_panel()`` /
  ``load_series()`` read the parquet directly — OFFLINE, no yfinance import.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) of ``n_assets`` assets with dispersed **true** betas to a
  common market factor and a TUNABLE knob ``edge`` that plants a *flat-SML* alpha:
  each asset's mean return carries ``-edge*(beta_i - 1)`` (low-beta assets get a
  positive alpha, high-beta a negative one). ``edge = 0`` is the null — CAPM holds
  exactly, betas still disperse, and a beta-neutral BAB book must earn **nothing**.
  ``edge > 0`` plants exactly the Frazzini-Pedersen low-beta premium.

The offline path is pure numpy + pandas + stdlib.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_FILE = os.path.join(CACHE_DIR, "bab_multiasset_panel.parquet")

# Nine liquid, long-history asset-class ETFs — the multi-asset "market".
TICKERS = ["SPY", "EFA", "EEM", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]

# Human-readable labels (documentation / notebooks only).
ASSET_LABELS = {
    "SPY": "US equity",
    "EFA": "Dev-ex-US equity",
    "EEM": "EM equity",
    "TLT": "Long Treasuries",
    "LQD": "IG credit",
    "HYG": "High-yield credit",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "US REITs",
}

START = "2007-01-01"        # panel start (HYG/DBC inception gate the effective start ~2007-05)
AS_OF = "2026-06-30"        # last complete calendar month at publication (drop partial 2026-07)

__all__ = [
    "TICKERS", "ASSET_LABELS", "START", "AS_OF", "CACHE_DIR", "CACHE_FILE",
    "fetch", "have_real", "load_series", "load_panel", "fingerprint",
    "synthetic_panel", "synthetic_series",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily total-return closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download the cross-asset total-return closes and cache them; returns the frame.

    Network; runs once. Retries up to ``retries`` times with a short back-off on a
    failed / empty pull. ``auto_adjust=True`` gives total-return prices. Writes a wide
    ``[date x ticker]`` Close parquet under this study's ``_cache/``.
    """
    import yfinance as yf  # lazy — never imported by the offline cells

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no daily bars")
            closes = (
                raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].copy()
            )
            closes = closes.reindex(columns=TICKERS)
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            # Require every asset to have priced (drop the pre-inception warm-up rows).
            closes = closes.dropna(how="any")
            if closes.empty:
                raise RuntimeError("no fully-populated cross-asset rows after align")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_FILE)
            return closes
        except Exception as exc:  # noqa: BLE001 — retry any transient network error
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 + attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_FILE)


def load_series(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide ``[date x ticker]`` total-return closes, sliced to ``[start, asof]``.

    Reads the parquet directly — OFFLINE, no yfinance import.
    """
    raw = pd.read_parquet(CACHE_FILE)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    df = raw.reindex(columns=[t for t in TICKERS if t in raw.columns])
    df = df[(df.index >= lo) & (df.index <= hi)].dropna(how="any").sort_index()
    return df


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Close]}`` — mirrors the sibling studies'
    ``load_panel`` shape so the same ``strategy`` helpers apply. OFFLINE."""
    df = load_series(start, asof)
    return {t: df[[t]].rename(columns={t: "Close"}) for t in df.columns}


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the cross-asset close frame, for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted flat-SML (low-beta) premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_series(
    edge: float = 0.0,
    seed: int = 830,
    n_assets: int = 9,
    n_days: int = 3500,
    start: str = "2008-01-02",
    mkt_vol: float = 0.010,
    idio_vol: float = 0.008,
    beta_lo: float = 0.35,
    beta_hi: float = 1.65,
) -> pd.DataFrame:
    """Deterministic seeded wide close frame with a TUNABLE planted flat-SML premium.

    Each asset ``i`` has a fixed **true beta** ``b_i`` spread evenly across
    ``[beta_lo, beta_hi]`` and loads on a common market factor ``m_t``::

        r[i,t] = alpha_i + b_i * m_t + idio_i,t
        alpha_i = -edge * (b_i - 1)          # flat SML: low beta over-, high beta under-priced

    With ``edge = 0`` the CAPM holds exactly (``alpha_i = 0``): betas still disperse,
    the equal-weight-market beta sort still separates the names, but a beta-neutral
    BAB book earns **nothing**. With ``edge > 0`` the low-beta assets carry a positive
    alpha and the high-beta a negative one — precisely the Frazzini-Pedersen premium a
    BAB factor is built to harvest. Business-day index below the ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    betas = np.linspace(beta_lo, beta_hi, n_assets)
    alphas = -edge * (betas - 1.0)

    m = rng.normal(0.0, mkt_vol, n_days)              # common market factor
    cols: dict[str, np.ndarray] = {}
    for i in range(n_assets):
        idio = rng.normal(0.0, idio_vol, n_days)
        r = alphas[i] + betas[i] * m + idio
        close = 100.0 * np.cumprod(1.0 + r)
        cols[f"SYN{i:02d}"] = close
    df = pd.DataFrame(cols, index=idx)
    df.index.name = "date"
    return df


def synthetic_panel(edge: float = 0.0, **kw) -> dict[str, pd.DataFrame]:
    """``synthetic_series`` re-expressed as a ``{ticker: DataFrame[Close]}`` panel."""
    df = synthetic_series(edge=edge, **kw)
    return {c: df[[c]].rename(columns={c: "Close"}) for c in df.columns}
