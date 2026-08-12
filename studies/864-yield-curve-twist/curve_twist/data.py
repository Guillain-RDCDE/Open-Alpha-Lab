"""Data layer for Study 864 (Yield-Curve Twist / Butterfly).

Beyond **level** and **slope**, the third principal component of the Treasury curve is
its **curvature** — the *butterfly*::

    fly_t  =  2 * y10_t  -  y5_t  -  y30_t          (the belly vs the wings)

A positive butterfly means the **belly** (10-year) yield is high relative to a straight
line drawn between the 5-year and 30-year wings — the belly is *cheap* (its price low).
A "twist" of the curve is a **change** in that curvature, ``dfly = fly_t - fly_{t-1}``.
This study asks whether the butterfly *level* and its *change* carry information about
forward Treasury (IEF/TLT) and equity (SPY) returns that is **distinct** from the 2s10s
slope (studies 66/132) and the roll-down carry (study 380).

Two tapes, one schema (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A single knob,
  ``fly_signal``, plants the effect: forward IEF returns load positively on the
  *lagged* butterfly rank (``fly_signal > 0`` = a genuine planted twist edge; ``= 0`` =
  the null world where the butterfly still wanders but predicts nothing). The
  self-consistent yield block (5y/10y/30y AR(1) levels) means the butterfly the engine
  reconstructs from the synthetic yields carries exactly the planted amount of
  predictive power — no more, no less. Tests never touch the network.

- ``fetch_daily`` — the real yfinance daily closes for **^FVX** (5-year yield index),
  **^TNX** (10-year yield index), **^TYX** (30-year yield index), and the ETFs **IEF**
  (7-10yr Treasury), **TLT** (20+yr Treasury), and **SPY** (S&P 500). Cache-first into
  this study's OWN ``_cache/`` so the reproducible core and the test-suite never touch
  the network. The three CBOE yield indices go back to the late 1990s; TLT/IEF start in
  2002, SPY in 1993, so the joined tape spans ~2002 onward.

No look-ahead is baked in here: the butterfly measured at the close of day *t* only
forms signals that trade at the close of day *t+1* or later (see ``strategy.py``). The
sample is pinned to :data:`AS_OF` so re-runs never creep as new sessions arrive.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers: three CBOE yield indices (5y/10y/30y) + three ETFs.
TICKERS = ["^FVX", "^TNX", "^TYX", "IEF", "TLT", "SPY"]

START = "2002-07-01"        # TLT/IEF inception window; the joined tape starts here
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "START", "AS_OF", "DEFAULT_CACHE",
    "fetch_daily", "have_real", "load_daily", "synthetic_daily",
    "cache_path", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Cache path
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Canonical parquet path for the combined curve/ETF daily cache."""
    return os.path.join(cache_dir, "daily_curve_twist.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(cache_path(cache_dir))


# --------------------------------------------------------------------------- #
# Frame assembly (shared by the real loader and — implicitly — the synthetic tape)
# --------------------------------------------------------------------------- #
def _assemble(closes: pd.DataFrame) -> pd.DataFrame:
    """From a frame carrying columns FVX, TNX, TYX, IEF_close, TLT_close, SPY_close,
    build the butterfly, slope, and forward-return-ready log returns.

    * ``fly   = 2*TNX - FVX - TYX``   (curvature / butterfly, in %-points)
    * ``dfly  = fly.diff()``          (the *twist* — day-over-day change in curvature)
    * ``slope = TNX - FVX``           (a 5s10s slope control — the dedup axis)
    * ``level = TNX``                 (the level control)
    * ``IEF_ret / TLT_ret / SPY_ret`` = daily log returns of the three ETFs
    """
    df = closes.copy()
    df["fly"] = 2.0 * df["TNX"] - df["FVX"] - df["TYX"]
    df["dfly"] = df["fly"].diff()
    df["slope"] = df["TNX"] - df["FVX"]
    df["level"] = df["TNX"]
    for etf in ("IEF", "TLT", "SPY"):
        col = f"{etf}_close"
        if col in df.columns:
            df[f"{etf}_ret"] = np.log(df[col]).diff()
    return df


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily closes, cache-first
# --------------------------------------------------------------------------- #
def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = START,
    retries: int = 4,
) -> pd.DataFrame:
    """Real daily curve/ETF data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (retried up to ``retries``
    times); the result is cached as a parquet under this study's ``_cache/``. With
    ``fetch=False`` (the default) a ``FileNotFoundError`` is raised if the cache is
    absent — the reproducible core and the tests must never hit the network.

    The returned frame has columns
    ``[FVX, TNX, TYX, IEF_close, TLT_close, SPY_close, fly, dfly, slope, level,
       IEF_ret, TLT_ret, SPY_ret]`` with a tz-naive ``DatetimeIndex`` named ``date``.
    """
    path = cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    TICKERS, start=start, interval="1d",
                    auto_adjust=True, progress=False, threads=True,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # pragma: no cover - network path
                last_err = exc
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars ({last_err})")

        closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
        closes.columns = [str(c).replace("^", "") for c in closes.columns]
        closes.index.name = "date"

        # Forward-fill ETF holidays (max 5 days) before we require them.
        for col in ("IEF", "TLT", "SPY"):
            if col in closes.columns:
                closes[col] = closes[col].ffill(limit=5)
        closes = closes.rename(columns={
            "IEF": "IEF_close", "TLT": "TLT_close", "SPY": "SPY_close",
        })
        need = ["FVX", "TNX", "TYX", "IEF_close", "TLT_close", "SPY_close"]
        closes = closes.dropna(subset=need)
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        df = _assemble(closes).dropna(subset=["IEF_ret"])
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_daily(cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached real tape sliced to ``[:, asof]`` — OFFLINE, no yfinance import."""
    df = fetch_daily(fetch=False, cache_dir=cache_dir)
    return df[df.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic tape — deterministic offline core with a planted twist edge
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_days: int = 5000,
    fly_signal: float = 0.0,
    ief_vol: float = 0.0035,
    start: str = "2003-01-02",
    seed: int = 864,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of butterfly predictive power.

    A self-consistent yield block is generated first — three AR(1) yield levels for the
    5y/10y/30y (the 10y carrying an extra mean-reverting "belly richness" term so the
    reconstructed butterfly genuinely wanders) — and the butterfly, slope and level are
    then rebuilt from those yields *exactly as the real loader does*. Forward IEF
    returns are::

        ief_ret_t  =  fly_signal * (fly_rank_{t-1} - 0.5)  +  eps_t

    where ``fly_rank`` is the causal rolling 252-day percentile of the butterfly and
    ``eps_t`` is i.i.d. normal (sd ``ief_vol``). ``fly_signal = 0`` makes IEF a pure
    random walk (the butterfly still wanders but predicts nothing — the null);
    ``fly_signal > 0`` plants a genuine positive butterfly → forward-IEF-return link
    (the positive control). TLT and SPY are related noisier series.

    Returns ``(df, truth)`` with the same schema as the real loader plus a ``truth``
    dict recording the planted parameters. Business-day index, span well below the
    pandas ns-timestamp horizon (n_days kept <= ~8000).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    def _ar1(phi, mu, sigma, x0, lo, hi):
        x = np.empty(n_days)
        x[0] = x0
        for i in range(1, n_days):
            x[i] = phi * x[i - 1] + (1 - phi) * mu + rng.normal(0, sigma)
        return np.clip(x, lo, hi)

    # 5y / 10y / 30y yield levels (%-points). The 10y gets an extra belly wobble so the
    # reconstructed butterfly = 2*y10 - y5 - y30 has real day-to-day variation.
    y5 = _ar1(0.985, 3.0, 0.05, 3.0, 0.2, 8.0)
    belly = _ar1(0.94, 0.0, 0.05, 0.0, -1.2, 1.2)  # transient belly richness
    y10 = _ar1(0.985, 3.6, 0.05, 3.6, 0.4, 8.5) + belly
    y30 = _ar1(0.988, 4.2, 0.045, 4.2, 0.6, 9.0)
    y10 = np.clip(y10, 0.4, 9.0)

    fly = 2.0 * y10 - y5 - y30
    fly_rank = pd.Series(fly, index=idx).rolling(252, min_periods=63).rank(pct=True)

    eps = rng.normal(0.0, ief_vol, n_days)
    ief_ret = np.empty(n_days)
    ief_ret[0] = eps[0]
    for i in range(1, n_days):
        r = fly_rank.iloc[i - 1]
        ief_ret[i] = eps[i] if np.isnan(r) else fly_signal * (r - 0.5) + eps[i]

    ief_close = 100.0 * np.exp(np.cumsum(ief_ret))
    tlt_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, ief_vol * 2.4, n_days)))
    spy_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, ief_vol * 3.2, n_days)))

    closes = pd.DataFrame(
        {"FVX": y5, "TNX": y10, "TYX": y30,
         "IEF_close": ief_close, "TLT_close": tlt_close, "SPY_close": spy_close},
        index=idx,
    )
    df = _assemble(closes)
    df.index.name = "date"
    truth = {
        "fly_signal": fly_signal, "ief_vol": ief_vol, "n_days": n_days,
        "seed": seed, "start": start,
    }
    return df, truth


# --------------------------------------------------------------------------- #
# Fingerprint (for the as-of stamp)
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame, col: str = "IEF_close") -> str:
    """Short content fingerprint of a daily tape (default IEF_close column)."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
