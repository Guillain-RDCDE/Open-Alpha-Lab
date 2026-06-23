"""Data layer for Study 371 — CBOE SKEW index (tail-risk gauge) vs forward SPY.

Two sources, both offline-friendly:

* **Real tape.** Daily closes for the CBOE **SKEW** index (``^SKEW``), the CBOE **VIX**
  (``^VIX``) and **SPY** (yfinance, no key), cached under ``_cache/skew_vix_spy.csv`` (a wide
  CSV indexed by date, columns ``SKEW``, ``VIX``, ``SPY``). SKEW is CBOE's *black-swan* gauge:
  it reads the price of out-of-the-money S&P 500 puts relative to at-the-money options, so a
  high SKEW is supposed to mean the option market is *paying up for crash insurance* — a warning
  the VIX (a symmetric, at-the-money vol gauge) can't give. We test whether SKEW carries
  forward-return / tail information **beyond** what VIX already prices.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable
  (skew, vix, forward-return) panel with a **planted edge knob**: ``edge`` scales how strongly
  high SKEW depresses the next month's return. It is the positive control — with ``edge = 0``
  the inference must NOT manufacture significance; with a large ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch()`` (network) is only used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "skew_vix_spy.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1990-01-01", end: str | None = None,
          path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download ^SKEW, ^VIX and SPY via yfinance and cache a wide close CSV.

    Network-only; used once to build ``_cache/skew_vix_spy.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(["^SKEW", "^VIX", "SPY"], start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns={"^SKEW": "SKEW", "^VIX": "VIX"})
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached frame, restricted to the joint (SKEW, VIX, SPY) window.

    Returns a frame indexed by date with columns ``skew``, ``vix``, ``spy`` (all three present
    on every row — SPY's 1993 inception sets the start). This is the real tape the engine runs
    on.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.rename(columns={"^SKEW": "skew", "SKEW": "skew",
                            "^VIX": "vix", "VIX": "vix", "SPY": "spy"})
    df = df[["skew", "vix", "spy"]].dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 8_000, edge: float = 0.0, seed: int = 371,
                    mu_daily: float = 0.0003, sig_daily: float = 0.011) -> pd.DataFrame:
    """Deterministic (skew, vix, spy) panel with a PLANTED predictive edge knob.

    Builds:

    * ``vix``  — a mean-reverting volatility series around ~18 (AR(1), clipped to a realistic
      band), spiking occasionally.
    * ``skew`` — a mean-reverting series around ~120 that is **partly driven by vix** (so the
      two are correlated, as on the real tape) plus its own idiosyncratic noise.
    * ``spy``  — a price whose daily drift is the baseline ``mu_daily`` **minus** ``edge`` times
      the *standardised* SKEW level lagged one day. With ``edge = 0`` SKEW carries no forward
      information (the null); with ``edge`` large, high SKEW genuinely predicts weak forward
      returns and the inference must detect it — but only the part **orthogonal to VIX**, since
      the regression controls for VIX.

    With ``edge = 0`` the control must NOT find significance however the noise falls — that is
    the whole point: a positive control with a knob that is *off* by default.
    """
    rng = np.random.default_rng(seed)

    # mean-reverting VIX around ~18
    vix = np.empty(n_days)
    vix[0] = 18.0
    for t in range(1, n_days):
        shock = rng.normal(0, 1.2) + (8.0 if rng.random() < 0.004 else 0.0)  # rare spikes
        vix[t] = 18.0 + 0.94 * (vix[t - 1] - 18.0) + shock
    vix = np.clip(vix, 9.0, 80.0)

    # SKEW around ~120, correlated with VIX plus idiosyncratic part
    vix_z = (vix - vix.mean()) / vix.std()
    skew_idio = np.empty(n_days)
    skew_idio[0] = 0.0
    for t in range(1, n_days):
        skew_idio[t] = 0.95 * skew_idio[t - 1] + rng.normal(0, 1.0)
    skew_idio = (skew_idio - skew_idio.mean()) / skew_idio.std()
    skew = 120.0 + 6.0 * vix_z + 6.0 * skew_idio          # corr with VIX ~0.7
    skew = np.clip(skew, 105.0, 160.0)

    # the planted edge lives ONLY in the part of SKEW orthogonal to VIX (skew_idio): so the
    # VIX-controlled regression is what must recover it.
    drift = mu_daily - edge * skew_idio                    # skew_idio is the standardised resid
    drift_lag = np.concatenate([[mu_daily], drift[:-1]])   # 1-day predictive lag (no look-ahead)
    ret = rng.normal(0.0, sig_daily, size=n_days) + drift_lag
    price = 100.0 * np.exp(np.cumsum(ret))

    idx = pd.bdate_range("1993-01-29", periods=n_days)
    return pd.DataFrame({"skew": skew, "vix": vix, "spy": price}, index=idx)
