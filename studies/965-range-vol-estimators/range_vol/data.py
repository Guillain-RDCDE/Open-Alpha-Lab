"""Data layer for Study 965 — The Range Estimators.

This study lives or dies on the **high and the low**, so it uses the OHLC(V) loader
and drops — loudly, with a published count — any bar whose high is below its low or whose
close sits outside the range. A single broken bar can dominate a variance estimate built on
squared log ranges.

Two tapes, one shape:

- ``fetch`` / ``load_ohlc`` — daily **OHLC** bars from Yahoo! Finance (``yfinance``,
  ``auto_adjust=True``: split- and dividend-adjusted, so open/high/low/close stay on one
  consistent scale), cached as parquet in the **shared** desk cache ``studies/_cache``.
  ``fetch`` is the only thing that touches the network; ``load_ohlc`` reads the cache
  offline and never imports yfinance.
- ``synthetic_ohlc`` — a deterministic, offline generator that simulates the intraday
  path so a range-based estimator has a genuine high and low to read, with a
  ``signal_strength`` knob (``1.0`` plants the effect, ``0.0`` is the matching null).

The sample is pinned at ``AS_OF`` so a rerun months later cannot quietly grow the window.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Five liquid tapes with genuinely different intraday shapes: an index fund with a
# huge open auction, a tech index, small caps, gold (a 24-hour underlying whose US
# session is a slice of the real trading day) and long Treasuries.
TICKERS = ("SPY", "QQQ", "IWM", "GLD", "TLT")

AS_OF = "2026-06-30"
START = "1993-01-01"

OHLC_COLS = ("open", "high", "low", "close")
BAR_COLS = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- #
# Real tape — daily OHLC(V) bars, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    """Own cache prefix (``ohlcv_``): these bars carry volume, the desk's older
    ``ohlc_`` parquets do not, and silently widening those would break the studies
    that read them."""
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ohlcv_{safe}_1d.parquet")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict[str, pd.DataFrame]:
    """Download daily OHLC bars for ``tickers`` and cache each as parquet (network-only)."""
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[list(BAR_COLS)].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's OHLC parquet is present in the shared cache."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_ohlc(ticker: str, cache_dir: str = DEFAULT_CACHE,
              asof: str = AS_OF) -> pd.DataFrame:
    """Read one ticker's cached daily OHLC bars OFFLINE, sliced to ``asof``.

    A bar whose high is below its low, or whose close sits outside [low, high], is a
    broken bar rather than a market event: those rows are dropped and counted by
    :func:`bad_bar_count`, never silently repaired.
    """
    path = _cache_path(ticker, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached OHLC for {ticker} at {path}. "
            f"Call range_vol.data.fetch() once to populate the shared cache."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)].sort_index()
    return df[_bar_is_sane(df)]


def load_all(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
             asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """``{ticker: ohlc frame}`` for the whole universe, offline."""
    return {tk: load_ohlc(tk, cache_dir=cache_dir, asof=asof) for tk in tickers}


def _bar_is_sane(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: high >= low and both bracket the open and the close."""
    hi, lo = df["high"], df["low"]
    return (
        (hi >= lo)
        & (df["close"] <= hi + 1e-12) & (df["close"] >= lo - 1e-12)
        & (df["open"] <= hi + 1e-12) & (df["open"] >= lo - 1e-12)
        & df[list(OHLC_COLS)].notna().all(axis=1)
        & (lo > 0)
    )


def bad_bar_count(ticker: str, cache_dir: str = DEFAULT_CACHE,
                  asof: str = AS_OF) -> int:
    """How many bars ``load_ohlc`` had to drop — a data-quality number, published."""
    path = _cache_path(ticker, cache_dir)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df[df.index <= pd.Timestamp(asof)]
    return int((~_bar_is_sane(df)).sum())


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a bar frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]




# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_ohlc(
    n_years: int = 20,
    vol_ann: float = 0.16,              # annualised close-to-close vol
    vol_of_vol: float = 0.6,            # log-vol shock size (clustering)
    vol_halflife_days: float = 21.0,    # persistence of the vol process
    n_intraday: int = 78,               # sub-steps used to build the day's range (5-min bars)
    overnight_share: float = 0.35,      # share of daily variance that arrives as a gap
    signal_strength: float = 1.0,       # 0 = constant vol (the null), 1 = full clustering
    start: str = "2004-01-02",
    seed: int = 965,
    drift_ann: float = 0.07,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic daily OHLC bars built from a simulated intraday path.

    The day's variance is split between an **overnight gap** (``overnight_share``) and a
    Brownian intraday path of ``n_intraday`` steps whose running max/min become the bar's
    high and low. Note the consequence, which is a feature and not an artefact: a path
    sampled at finitely many points has a **smaller range than the continuous path it
    approximates**, so range-based variance estimators read low on these bars exactly as
    they do on a real, discretely-traded market (Marsh & Rosenfeld 1986). Raising
    ``n_intraday`` shrinks that bias toward zero, which is itself a testable prediction — so a range estimator (Parkinson, Garman-Klass, Rogers-Satchell) reads a
    genuine range rather than a fabricated one, and its known efficiency gain over
    close-to-close is recoverable. Volatility follows an AR(1) in logs scaled by
    ``signal_strength``: at ``0.0`` vol is constant (the null), at ``1.0`` it clusters.

    Returns ``(bars, truth)`` with columns ``open, high, low, close`` plus the realised
    per-day sigma actually used, in ``truth['sigma']``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    sd_d = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    phi = float(np.exp(-np.log(2.0) / vol_halflife_days))
    shock = vol_of_vol * float(signal_strength) * np.sqrt(1.0 - phi ** 2)
    log_v = np.zeros(n_days)
    e = rng.normal(0.0, 1.0, n_days)
    for t in range(1, n_days):
        log_v[t] = phi * log_v[t - 1] + shock * e[t]
    # Centre the log-vol so the average variance stays at sd_d^2 whatever the clustering.
    sigma = sd_d * np.exp(log_v - 0.5 * np.var(log_v))

    mu_d = drift_ann / TRADING_DAYS_PER_YEAR
    gap_sd = sigma * np.sqrt(overnight_share)
    intra_sd = sigma * np.sqrt(1.0 - overnight_share)

    steps = rng.normal(0.0, 1.0, (n_days, n_intraday)) / np.sqrt(n_intraday)
    gaps = rng.normal(0.0, 1.0, n_days)

    log_close_prev = np.log(100.0)
    o = np.empty(n_days); h = np.empty(n_days); lo = np.empty(n_days); c = np.empty(n_days)
    for t in range(n_days):
        open_t = log_close_prev + mu_d + gap_sd[t] * gaps[t]
        path = open_t + intra_sd[t] * np.cumsum(steps[t])
        o[t] = open_t
        c[t] = path[-1]
        h[t] = max(open_t, path.max())
        lo[t] = min(open_t, path.min())
        log_close_prev = c[t]

    # Volume: lognormal noise around a level, with a mild tie to the day's own vol
    # (busy days are volatile days). The dispersion is deliberately modest — a real
    # index ETF's daily volume sits within a few tens of percent of its own median,
    # and an over-dispersed generator would make any volume threshold meaningless.
    vol_noise = rng.normal(0.0, 0.12, n_days)
    volume = 1e6 * np.exp(vol_noise) * (sigma / sd_d) ** 0.25

    bars = pd.DataFrame(
        {"open": np.exp(o), "high": np.exp(h), "low": np.exp(lo), "close": np.exp(c),
         "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "n_days": n_days, "n_years": n_years, "seed": seed, "vol_ann": vol_ann,
        "signal_strength": float(signal_strength), "n_intraday": n_intraday,
        "overnight_share": overnight_share, "vol_halflife_days": vol_halflife_days,
        "sigma": pd.Series(sigma, index=bars.index, name="sigma"),
        "drift_ann": drift_ann,
    }
    return bars, truth

