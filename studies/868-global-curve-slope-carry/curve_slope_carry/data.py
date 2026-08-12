"""Data layer for Study 868 — Global Curve-Slope Carry.

The claim under test (the fixed-income *roll + carry* / curve-carry literature — Koijen,
Moskowitz, Pedersen & Vrugt 2018, *"Carry"*; the classic roll-down argument): a **steep**
yield curve pays a duration holder to hold — the bond rolls *down* the curve as it ages
and earns the term spread as carry. Cross-sectionally across bond markets a duration
investor should therefore prefer the **steep-curve / high-carry** markets and avoid the
**flat / low-carry** ones. We test the tradable version: rank a small panel of US and
international sovereign-bond ETFs by a **carry proxy** and go **long the high-carry, short
the low-carry** markets, then cost it and benchmark it against equal-weight buy-and-hold.

Two ingredients, both offline-friendly once cached.

* **Real tape — a panel of US + international sovereign-bond ETFs.** Month-end
  total-return levels for six government-bond funds spanning the maturity spectrum and two
  regions:

    - ``SHY``  — iShares 1-3y US Treasury      (short-duration US anchor)
    - ``IEF``  — iShares 7-10y US Treasury      (belly of the US curve)
    - ``TLT``  — iShares 20y+ US Treasury       (long-duration US)
    - ``BWX``  — SPDR Bloomberg Intl Treasury   (ex-US developed sovereigns, unhedged)
    - ``IGOV`` — iShares International Treasury  (ex-US developed sovereigns, unhedged)
    - ``BNDX`` — Vanguard Total International    (USD-hedged global ex-US aggregate)

  Pulled with yfinance (``auto_adjust=True`` → total-return levels, coupons reinvested),
  cached under this study's OWN ``_cache/`` as a single parquet, and read back OFFLINE.
  ``fetch()`` (network) runs once to build the cache and is never imported by the
  notebooks' offline cells; ``load_panel()`` / ``load_series()`` read the cached parquet
  directly (no yfinance import).

  **The honest limit of a price-only carry proxy.** yfinance gives *total-return* levels
  only — coupon and price change are already blended. We therefore proxy each sleeve's
  **carry / yield** by its trailing realized yield (the annualised mean monthly total
  return over a **long** ``window``, so transient price trends average out and the slow
  structural income component dominates), then divide by the sleeve's published
  **effective duration** to get a *yield-to-duration* / carry-per-unit-rate-risk score
  (:data:`DURATIONS`). This is a proxy, not a clean forward yield — a caveat carried on
  the Signal axis, and the whole reason a long window is used.

  **Survivorship — named on the Signal axis.** These six are *surviving, currently listed*
  funds; wound-up global-govvie ETFs are absent, so any positive carry result is a mild
  **upper bound**. The panel is tiny by construction (few liquid global-sovereign ETFs
  have a decade of history) and the USD-hedged ``BNDX`` only lists from 2013, so the full
  six-market cross-section is short — both facts limit statistical power.

* **Synthetic world — the positive control.** A deterministic, seeded **monthly** price
  panel (``synthetic_panel``) with a TUNABLE knob ``edge``. Each asset carries a **fixed
  structural carry level** ``c_i`` (a static per-market spread); the monthly return loads
  ``edge`` on ``c_i`` plus noise, so a high-carry market persistently out-yields and the
  trailing realized-yield proxy ranks the markets correctly. ``edge = 0`` is the null —
  every market has the same expected return, the cross-sectional carry sort has nothing to
  earn, and the detector must stay silent. ``edge > 0`` plants exactly the carry pattern
  (high-carry markets keep paying more).

No look-ahead: the carry proxy measured through the close of month ``t−1`` forms the
signal held over month ``t`` (a one-month execution lag, applied identically on both tapes
— see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers: three US-curve maturities + three international sovereign sleeves.
TICKERS = ["SHY", "IEF", "TLT", "BWX", "IGOV", "BNDX"]

# Approximate published EFFECTIVE DURATIONS (years) — public fund-fact-sheet facts, used
# as fixed sleeve characteristics for the yield-to-duration carry proxy. Sourced from the
# iShares / SPDR / Vanguard fact sheets (2024-2025 vintage); durations drift slowly, and
# the cross-sectional RANK (short SHY << belly/intl << long TLT) is stable across vintages.
# Cited in docs/references.md.
DURATIONS = {
    "SHY": 1.85,    # iShares 1-3y UST
    "IEF": 7.30,    # iShares 7-10y UST
    "TLT": 16.40,   # iShares 20y+ UST
    "BWX": 7.90,    # SPDR Bloomberg International Treasury (ex-US developed)
    "IGOV": 8.10,   # iShares International Treasury (ex-US developed)
    "BNDX": 6.90,   # Vanguard Total International Bond (USD-hedged global ex-US)
}

START = "2007-01-01"        # BWX inception era; earlier names simply carry NaN
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "DURATIONS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "cache_path", "load_prices_daily",
    "load_panel", "load_series", "synthetic_panel", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily total-return levels, cache-first
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, "curve_carry_etfs_daily.parquet")


def fetch(cache_dir: str = CACHE_DIR, start: str = START,
          end: str = "2026-07-01", retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """Download daily total-return closes for ``TICKERS`` and cache them as parquet.

    ``auto_adjust=True`` yields total-return levels (dividends/coupons reinvested).
    Retries up to ``retries`` times with a short ``pause`` on any empty/failed pull.
    Writes ``_cache/curve_carry_etfs_daily.parquet`` (a tz-naive daily Close frame,
    columns = tickers) and returns it.
    """
    import yfinance as yf  # lazy — never imported by the offline cells

    os.makedirs(cache_dir, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"].copy()
            else:  # single ticker degenerate case
                closes = raw[["Close"]].copy()
                closes.columns = [TICKERS[0]]
            closes = closes.reindex(columns=TICKERS)
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes = closes.dropna(how="all")
            if closes.shape[0] < 200:
                raise RuntimeError(f"suspiciously short pull ({closes.shape[0]} rows)")
            closes.to_parquet(cache_path(cache_dir))
            return closes
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_err = exc
            if attempt < retries:
                time.sleep(pause)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real(cache_dir: str = CACHE_DIR) -> bool:
    return os.path.exists(cache_path(cache_dir))


def load_prices_daily(cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Cached daily total-return Close frame (columns = tickers). OFFLINE."""
    path = cache_path(cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached tape at {path}. Call curve_slope_carry.data.fetch() once."
        )
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.reindex(columns=[c for c in TICKERS if c in df.columns])


def load_panel(cache_dir: str = CACHE_DIR, asof: str = AS_OF) -> pd.DataFrame:
    """Month-end total-return **price** panel (index = month-end, columns = tickers).

    Reads the cached daily parquet OFFLINE, takes the last observation of each calendar
    month, and truncates to ``asof`` (drops the partial current month). This is the frame
    every strategy function consumes; ``synthetic_panel`` returns the same schema.
    """
    daily = load_prices_daily(cache_dir)
    monthly = daily.resample("ME").last()
    monthly = monthly[monthly.index <= pd.Timestamp(asof)]
    monthly.index.name = "month"
    return monthly


def load_series(ticker: str, cache_dir: str = CACHE_DIR, asof: str = AS_OF) -> pd.Series:
    """Single ticker's month-end total-return level (OFFLINE)."""
    panel = load_panel(cache_dir, asof)
    if ticker not in panel.columns:
        raise KeyError(f"{ticker!r} not in cached panel {list(panel.columns)}")
    return panel[ticker].dropna().rename(ticker)


def fingerprint(panel: pd.DataFrame) -> str:
    """Short content fingerprint of a month-end price panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted cross-sectional carry (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 868,
    n_assets: int = 6,
    n_months: int = 360,
    start: str = "1990-01-31",
    monthly_vol: float = 0.014,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Deterministic seeded **monthly** total-return price panel with a TUNABLE carry.

    Each asset ``i`` carries a **fixed structural carry level** ``c_i`` — a static
    per-market spread, mean-zero across the panel (some markets structurally out-yield the
    others). The monthly total return loads ``edge`` on that level plus noise::

        c_i   = linspace(-1, +1, n_assets)          # fixed carry ranking
        r[i,t] = drift + edge * c_i + monthly_vol * v[i,t]

    Because ``c_i`` is constant, the **trailing realized yield** (a long-window mean of
    past returns) recovers the ranking of ``c_i``, so sorting the markets by that proxy and
    going long the high-carry / short the low-carry sleeves earns ``≈ edge × spread(c)``.
    ``edge = 0`` is the null: every market has the same expected return, the trailing-yield
    ranks are pure noise, and the cross-sectional carry sort must find nothing. ``edge > 0``
    plants exactly the carry pattern (high-carry markets persistently pay more). Returns a
    month-end price DataFrame matching :func:`load_panel`'s schema.

    Uses a plain ``pd.date_range`` at MONTH-END with ``n_months`` well under the pandas
    ns-Timestamp horizon (360 months from 1990 → 2019), so there is no overflow risk.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    carry = np.linspace(-1.0, 1.0, n_assets)          # fixed structural carry ranking

    prices = {}
    for i in range(n_assets):
        v = rng.normal(0.0, 1.0, n_months)
        r = drift + edge * carry[i] + monthly_vol * v
        level = 100.0 * np.cumprod(1.0 + r)
        prices[f"SYN{i:02d}"] = level
    return pd.DataFrame(prices, index=idx)
