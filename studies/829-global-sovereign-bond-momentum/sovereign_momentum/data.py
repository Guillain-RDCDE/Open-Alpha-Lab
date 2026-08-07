"""Data layer for Study 829 — Global Sovereign-Bond Momentum.

The claim under test (Moskowitz, Ooi & Pedersen 2012, *"Time Series Momentum"*, and
the fixed-income time-series trend literature): a bond's **own recent trend predicts
its own next return**. Measure each sovereign-bond ETF's **12-minus-1-month total
return** (the cumulative total return over the past twelve months, *excluding the most
recent month*); go **long the positive-trend** markets and **short the negative-trend**
markets (or, in the long-only variant, own only the up-trending ones). Persistent trends
in the level of global government-bond prices — driven by slow-moving policy-rate and
term-premium cycles — should make the sign of the trailing trend a positive predictor of
the forward month.

Two ingredients, both offline-friendly once cached.

* **Real tape — a small panel of global sovereign-bond ETFs.** Month-end total-return
  levels for a fixed list of **foreign / global** government-bond funds:

    - ``BWX``  — SPDR Bloomberg International Treasury Bond (ex-US developed sovereigns)
    - ``IGOV`` — iShares International Treasury Bond (ex-US developed sovereigns)
    - ``BNDX`` — Vanguard Total International Bond (USD-hedged global ex-US)
    - ``EMB``  — iShares JP Morgan USD Emerging-Markets Bond (EM sovereigns)
    - ``IEF``  — iShares 7-10y US Treasury (the domestic anchor / benchmark leg)

  Pulled with yfinance (``auto_adjust=True`` → total-return levels, coupons reinvested),
  cached under this study's OWN ``_cache/`` as a single parquet, and read back OFFLINE.
  ``fetch()`` (network) runs once to build the cache and is never imported by the
  notebooks' offline cells; ``load_panel()`` / ``load_series()`` read the cached parquet
  directly (no yfinance import).

  **Survivorship — named on the Signal axis.** These five are *surviving, currently
  listed* funds. Foreign-sovereign and EM-debt ETFs that were wound up or merged are
  absent, so any positive trend result is (mildly) an **upper bound**. The panel is tiny
  by construction (only a handful of liquid global-govvie ETFs exist with a decade of
  history), which also limits statistical power — both caveats travel with the numbers.

* **Synthetic world — the positive control.** A deterministic, seeded **monthly** price
  panel (``synthetic_panel``) with a TUNABLE knob ``edge``. Each asset carries a
  slow-moving persistent latent trend state ``x_i[t]`` (an AR(1)); the monthly return
  loads ``edge`` on the *lagged* state, so a positive persistent trend both shows up in
  the trailing 12-1 momentum **and** predicts the forward month. ``edge = 0`` is the null
  world — prices are a pure random walk, the trailing trend carries no information, and
  the time-series-momentum detector must find nothing. ``edge > 0`` plants exactly the
  Moskowitz-Ooi-Pedersen trend-persistence pattern.

No look-ahead: the 12-1 momentum measured through the close of month ``t−1`` forms the
signal that is held over month ``t`` (a one-month execution lag, applied identically on
both tapes — see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers: four foreign / global sovereign-bond ETFs + the US anchor.
TICKERS = ["BWX", "IGOV", "BNDX", "EMB", "IEF"]

START = "2007-01-01"        # BWX / EMB inception era; earlier names simply carry NaN
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "cache_path", "load_prices_daily",
    "load_panel", "load_series", "synthetic_panel", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily total-return levels, cache-first
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, "sovereign_bond_etfs_daily.parquet")


def fetch(cache_dir: str = CACHE_DIR, start: str = START,
          end: str = "2026-07-01", retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """Download daily total-return closes for ``TICKERS`` and cache them as parquet.

    ``auto_adjust=True`` yields total-return levels (dividends/coupons reinvested).
    Retries up to ``retries`` times with a short ``pause`` on any empty/failed pull.
    Writes ``_cache/sovereign_bond_etfs_daily.parquet`` (a tz-naive daily Close frame,
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
            f"No cached tape at {path}. Call sovereign_momentum.data.fetch() once."
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
# Synthetic world — planted time-series-momentum (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 829,
    n_assets: int = 8,
    n_months: int = 600,
    start: str = "1980-01-31",
    monthly_vol: float = 0.018,
    drift: float = 0.0,
    trend_rho: float = 0.94,
) -> pd.DataFrame:
    """Deterministic seeded **monthly** total-return price panel with a TUNABLE trend.

    Each asset ``i`` carries a persistent latent trend state ``x_i[t]`` — a zero-mean
    AR(1) with autocorrelation ``trend_rho`` (slow-moving, like a policy-rate / term-
    premium cycle). The monthly return loads ``edge`` on the *lagged* state plus noise::

        x[i,t] = trend_rho * x[i,t-1] + sqrt(1-trend_rho**2) * u[i,t]
        r[i,t] = drift + edge * x[i,t-1] + monthly_vol * v[i,t]

    Because ``x`` is persistent, the trailing 12-1 momentum (a sum of past returns, each
    carrying ``edge * x``) proxies the current state, which then drives the forward month
    — the Moskowitz-Ooi-Pedersen time-series-momentum pattern. ``edge = 0`` is the null:
    returns are i.i.d. noise, the trend carries no information, and the detector must stay
    silent. Returns a month-end price DataFrame matching :func:`load_panel`'s schema.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    innov_sd = np.sqrt(1.0 - trend_rho ** 2)

    prices = {}
    for i in range(n_assets):
        x = np.empty(n_months)
        x[0] = rng.normal(0.0, 1.0)
        u = rng.normal(0.0, innov_sd, n_months)
        for t in range(1, n_months):
            x[t] = trend_rho * x[t - 1] + u[t]
        v = rng.normal(0.0, 1.0, n_months)
        r = drift + edge * np.concatenate([[0.0], x[:-1]]) + monthly_vol * v
        level = 100.0 * np.cumprod(1.0 + r)
        prices[f"SYN{i:02d}"] = level
    return pd.DataFrame(prices, index=idx)
