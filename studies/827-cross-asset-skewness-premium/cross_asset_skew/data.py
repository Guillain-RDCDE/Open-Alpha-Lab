"""Data layer for Study 827 — Cross-Asset Skewness Premium.

The claim under test (the asset-class analogue of the single-name realized-skewness
reversal of Amaya, Christoffersen, Jacobs & Vasquez 2015; cf. Study 803): a **skewness
premium ACROSS asset classes**. Measure each asset-class ETF's **trailing realized
skewness** of daily returns; each period sort the cross-section of asset classes and go
**long the low-skew / short the high-skew** book. If the single-name lottery-overpricing
mechanism carries up to the asset-class level, low-skew classes should out-earn high-skew
ones and the long-low/short-high spread should be positive.

Two ingredients, both offline-friendly once cached.

* **Real tape — a fixed 9-ETF cross-section of the major asset classes.** Daily total-return
  closes (``auto_adjust=True``) for a fixed list of nine liquid asset-class ETFs
  (``TICKERS`` below: US equity, developed-ex-US equity, EM equity, long Treasuries,
  IG credit, HY credit, gold, broad commodities, US REITs), pulled with yfinance and cached
  under this study's OWN ``_cache/`` as a single parquet. ``fetch()`` retries up to four
  times with a short sleep on a transient failure; ``load_panel()`` / ``load_series()`` read
  the cached parquet OFFLINE (no yfinance import).

  **Survivorship — named on the Signal axis.** ``TICKERS`` is a *current* set of nine ETFs
  that are the liquid, canonical proxy for each asset class *today*; it is a small,
  fixed, low-turnover set (the flagship class ETFs have not been delisted), so the
  survivorship exposure is far milder than a single-name universe — but the caveat still
  travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded closes panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each asset carries a persistent latent
  "skew tilt" ``c_i`` that both (a) shapes the skewness of its daily returns and (b) — only
  when ``edge > 0`` — depresses its forward mean return. ``edge = 0`` is the null world:
  realized skew still varies across assets but carries **no** information about forward
  returns, and the sort must find nothing. ``edge > 0`` plants the low-skew/high-return
  relation the claim predicts.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells; ``load_panel()`` reads the
cached parquet directly.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

START = "2007-01-01"        # panel start — the newest class ETF (HYG, 2007) bounds it
AS_OF = "2026-06-30"        # last complete calendar month at publication

# Nine liquid asset-class ETFs — a fixed, canonical, current-membership proxy set.
#   SPY  US large-cap equity        EFA  developed-ex-US equity   EEM  EM equity
#   TLT  long US Treasuries         LQD  IG corporate credit      HYG  HY corporate credit
#   GLD  gold                       DBC  broad commodities        VNQ  US REITs
TICKERS = ["SPY", "EFA", "EEM", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]

CACHE_PATH = os.path.join(CACHE_DIR, "cross_asset_closes.parquet")

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_panel",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download the nine asset-class ETF daily total-return closes; cache the panel.

    Network; runs once. ``auto_adjust=True`` (total-return closes). Retries up to
    ``retries`` times with a short sleep on a transient failure. Writes a single wide
    parquet (index=date, columns=ticker) under this study's ``_cache/``.
    """
    import yfinance as yf  # lazy — never imported on the offline path

    os.makedirs(CACHE_DIR, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            closes = (
                raw["Close"].copy()
                if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].copy()
            )
            closes = closes.reindex(columns=TICKERS)
            closes.index = pd.to_datetime(closes.index)
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes.index.name = "date"
            # A class ETF should have data across (nearly) the whole span; require all nine.
            got = [c for c in TICKERS if closes[c].notna().sum() > 250]
            if len(got) < len(TICKERS):
                raise RuntimeError(f"missing history for {set(TICKERS) - set(got)}")
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as exc:  # noqa: BLE001 — retry any transient yfinance failure
            last_err = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide closes frame (index=date, columns=ticker), sliced to ``[START, asof]``.

    Reads the parquet directly — OFFLINE, no yfinance import. Drops the partial current
    month at ``asof`` so a stamped run has no partial month.
    """
    raw = pd.read_parquet(CACHE_PATH)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    lo, hi = pd.Timestamp(START), pd.Timestamp(asof)
    closes = raw.loc[(raw.index >= lo) & (raw.index <= hi), TICKERS].copy()
    return closes.sort_index()


def load_series(ticker: str, asof: str = AS_OF) -> pd.Series:
    """One asset-class ETF's cached total-return close series (OFFLINE)."""
    return load_panel(asof=asof)[ticker].dropna()


# --------------------------------------------------------------------------- #
# Synthetic world — planted low-skew/high-return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 827,
    n_assets: int = 9,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.010,
    drift: float = 0.05 / 252,
    factor_rho: float = 0.95,
    skew_gain: float = 1.6,
) -> pd.DataFrame:
    """Deterministic seeded wide closes panel with a TUNABLE planted skew->return relation.

    Each asset ``i`` carries a persistent latent "skew tilt" ``c_i[t]`` — an AR(1) with
    autocorrelation ``factor_rho`` (very persistent, so the tilt is an asset-class trait, not
    a daily flicker). The tilt shapes the **skewness** of the daily return via a
    quadratic-in-shock term (so a trailing realized-skewness sort proxies ``c_i``), and —
    only when ``edge > 0`` — depresses the **forward mean**::

        z ~ N(0,1)
        skewed_shock = daily_vol * (z + skew_gain * c_i[t] * (z**2 - 1))
        r[i,t] = drift - edge * c_i[t] + skewed_shock

    So a high positive tilt makes an asset's returns **right-skewed** *and* (with ``edge>0``)
    lower-mean — high realized skew, low forward return, the claim's pattern. ``edge = 0`` is
    the null: skew still varies across assets but predicts nothing. Returns a wide closes
    frame (index=business day, columns ``SYN00..``) so the same ``strategy`` code runs on it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    cols: dict[str, np.ndarray] = {}
    for i in range(n_assets):
        c = np.empty(n_days)
        c[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            c[t] = factor_rho * c[t - 1] + eps[t]

        z = rng.normal(0.0, 1.0, n_days)
        skewed = daily_vol * (z + skew_gain * c * (z ** 2 - 1.0))
        r = drift - edge * c + skewed
        close = 100.0 * np.cumprod(1.0 + r)
        cols[f"SYN{i:02d}"] = close

    out = pd.DataFrame(cols, index=idx)
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------- #
# Reproducibility stamp
# --------------------------------------------------------------------------- #
def fingerprint(closes: pd.DataFrame) -> str:
    """A short, stable content hash of the closes panel (for the as-of stamp)."""
    cols = sorted(closes.columns)
    arr = np.ascontiguousarray(
        closes[cols].round(6).fillna(0.0).to_numpy(dtype=float)
    )
    h = hashlib.sha1(arr.tobytes())
    h.update(",".join(cols).encode())
    return h.hexdigest()[:12]
