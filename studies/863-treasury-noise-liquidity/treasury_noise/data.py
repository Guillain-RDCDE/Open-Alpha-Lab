"""Data layer for Study 863 — Treasury Noise Liquidity.

The claim under test (Hu, Pan & Wang 2013, *"Noise as Information for Illiquidity"*): the
cross-maturity **roughness** of the Treasury yield curve is a market-wide illiquidity /
funding-stress gauge. When arbitrage capital is plentiful, relative-value desks iron the
curve flat against a smooth fit; when capital is scarce, individual maturities wander off
and the fitted residuals fatten. Their **noise measure** is the RMS deviation of observed
yields from a smooth fitted curve. High noise ⇒ illiquidity/stress, which is said to
**precede lower equity returns and wider credit spreads**.

Two ingredients, one daily schema.

* **Real tape — CMT yields + risk ETFs.** ``fetch()`` pulls with yfinance
  (``auto_adjust=True``) the four constant-maturity Treasury yield indices
  ``^IRX`` (13-week), ``^FVX`` (5-year), ``^TNX`` (10-year), ``^TYX`` (30-year) — all in
  percentage-point units — plus ``SPY`` (equity, total-return), ``HYG`` (high-yield
  credit), ``IEF`` (7-10y Treasury, the duration hedge), and ``LQD`` / ``TLT`` for
  reference, into this study's OWN ``_cache/`` as one parquet (retry up to 4×).
  ``have_real()`` / ``load_panel()`` read the cached parquet OFFLINE (no yfinance import).
  ``AS_OF="2026-06-30"`` — the last complete calendar month; the partial current month is
  dropped.

* **Synthetic world — the positive control.** ``synthetic_daily`` builds a deterministic,
  seeded daily tape in which a persistent latent **noise level** ``s_t`` (AR(1), positive)
  is planted *by construction*: each day the four yields are a smooth quadratic curve plus
  ``s_t``-scaled per-maturity idiosyncratic deviations, so the RMS-of-residuals roughness
  recovers ``s_t``. Only when ``edge > 0`` does the *next* day's SPY and HYG−IEF mean
  return load **negatively** on ``s_{t-1}`` (high noise ⇒ lower forward equity, wider
  credit). ``edge = 0`` is the null (the curve still roughens but that roughness carries
  no forward information). Business-day index, span far below the pandas ns horizon.

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
CACHE_PATH = os.path.join(CACHE_DIR, "treasury_noise_daily.parquet")

# CMT yield indices (percentage points) at maturities 13w / 5y / 10y / 30y, plus the
# risk assets the noise is said to warn.
YIELD_TICKERS = ["^IRX", "^FVX", "^TNX", "^TYX"]
ASSET_TICKERS = ["SPY", "HYG", "IEF", "LQD", "TLT"]
TICKERS = YIELD_TICKERS + ASSET_TICKERS
# Column names in the cache (the "^" is stripped so the parquet has clean identifiers).
YIELD_COLS = ["IRX", "FVX", "TNX", "TYX"]
MATURITIES = np.array([0.25, 5.0, 10.0, 30.0])   # years, aligned to YIELD_COLS

START = "2007-01-01"        # HYG inception is 2007-04; the panel binds there once joined
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "YIELD_TICKERS", "ASSET_TICKERS", "YIELD_COLS", "MATURITIES",
    "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "synthetic_daily", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download the CMT yields + risk ETFs with yfinance; cache under this study's _cache/.

    Network; runs once to build the cache. Retries up to ``retries`` times with a short
    sleep on transient failure. Writes a single parquet with a tz-naive daily index and
    columns ``IRX FVX TNX TYX`` (yields, %) and ``SPY HYG IEF LQD TLT`` (auto-adjusted
    total-return closes).
    """
    import yfinance as yf  # lazy — never imported on the offline path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            closes = closes.copy()
            closes.columns = [str(c).replace("^", "") for c in closes.columns]
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            want = YIELD_COLS + ASSET_TICKERS
            missing = [c for c in want if c not in closes.columns]
            if missing:
                raise RuntimeError(f"missing columns from yfinance: {missing}")
            closes = closes[want].dropna(how="any")
            if len(closes) < 1000:
                raise RuntimeError(f"suspiciously short tape ({len(closes)} rows)")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as exc:  # noqa: BLE001 — retry on any transient failure
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily frame, sliced to ``<= asof``. OFFLINE (reads parquet).

    Returns a tz-naive daily ``DataFrame`` with the yield columns (``IRX FVX TNX TYX``,
    in %) and the asset closes (``SPY HYG IEF LQD TLT``).
    """
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call treasury_noise.data.fetch() once."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)]
    return df.dropna(how="any").sort_index()


# --------------------------------------------------------------------------- #
# Synthetic world — planted noise→forward-return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    edge: float = 0.0,
    seed: int = 863,
    n_days: int = 3200,
    start: str = "2008-01-02",
    noise_base: float = 0.06,        # baseline roughness (yield %-points)
    noise_rho: float = 0.985,        # AR(1) persistence of the noise level
    noise_sd: float = 0.012,         # AR(1) innovation sd of the noise level
    spy_vol: float = 0.010,          # daily SPY vol
    spy_drift: float = 0.0003,       # daily SPY drift
    credit_vol: float = 0.0035,      # daily HYG-IEF spread-return vol
    credit_drift: float = 0.00004,   # daily HYG-IEF drift
) -> pd.DataFrame:
    """Deterministic seeded daily tape with a TUNABLE planted noise→forward-return relation.

    A persistent latent **noise level** ``s_t`` follows a positive AR(1) around
    ``noise_base``. Each day the four CMT yields are a smooth quadratic-in-maturity curve
    (a slowly wandering level/slope/curvature) **plus** ``s_t``-scaled iid per-maturity
    deviations, so the strategy's RMS-of-residuals roughness recovers ``s_t`` up to a
    constant. Only when ``edge > 0`` does the next day's SPY and HYG−IEF mean return load
    **negatively** on ``s_{t-1}`` (high noise ⇒ lower forward equity, wider credit):

        s_t         ~ AR(1) around noise_base, floored positive
        yield[m,t]  = level_t + slope_t·f1(m) + curv_t·f2(m) + s_t·ε_{m,t}
        r_SPY[t]    = spy_drift    − edge·(s_{t-1} − s̄)          + N(0, spy_vol²)
        r_(HYG−IEF) = credit_drift − edge·credit_k·(s_{t-1} − s̄) + N(0, credit_vol²)

    ``edge = 0`` is the null: the curve still roughens (s_t varies) but that roughness
    carries no forward information. Returns a tz-naive daily ``DataFrame`` with columns
    ``IRX FVX TNX TYX SPY HYG IEF LQD TLT`` — the same schema the real loader yields, so
    :func:`strategy.build_daily` consumes both identically.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # Persistent positive noise level (AR(1) around noise_base).
    s = np.empty(n_days)
    s[0] = noise_base
    innov = rng.normal(0.0, noise_sd, n_days)
    for t in range(1, n_days):
        s[t] = noise_base + noise_rho * (s[t - 1] - noise_base) + innov[t]
    s = np.maximum(s, 0.01)
    s_mean = s.mean()

    # Smooth curve factors: a slowly wandering level / slope / curvature.
    m = MATURITIES
    f1 = (m - m.mean()) / m.std()                       # slope basis
    f2 = f1 ** 2 - (f1 ** 2).mean()                     # curvature basis (centred)
    level = 3.0 + np.cumsum(rng.normal(0.0, 0.02, n_days))
    slope = 0.8 + np.cumsum(rng.normal(0.0, 0.01, n_days))
    curv = -0.2 + np.cumsum(rng.normal(0.0, 0.005, n_days))

    smooth = (level[:, None] + slope[:, None] * f1[None, :] + curv[:, None] * f2[None, :])
    dev = s[:, None] * rng.normal(0.0, 1.0, (n_days, len(m)))   # rough per-maturity noise
    yields = smooth + dev                                        # (n_days, 4), in %

    # Forward-loading returns: today's mean loads on YESTERDAY's noise (lagged predictor).
    s_lag = np.concatenate([[s_mean], s[:-1]])
    spy_ret = spy_drift - edge * (s_lag - s_mean) + rng.normal(0.0, spy_vol, n_days)
    credit_k = 0.35
    cred_ret = (credit_drift - edge * credit_k * (s_lag - s_mean)
                + rng.normal(0.0, credit_vol, n_days))

    spy_close = 100.0 * np.exp(np.cumsum(spy_ret))
    # Build HYG and IEF whose log-return DIFFERENCE equals the planted credit spread return.
    ief_ret = rng.normal(0.0002, 0.0030, n_days)     # rates leg, carries no planted edge
    hyg_ret = cred_ret + ief_ret                     # so (hyg - ief) log return == cred_ret
    ief_close = 90.0 * np.exp(np.cumsum(ief_ret))
    hyg_close = 80.0 * np.exp(np.cumsum(hyg_ret))
    lqd_close = 110.0 * np.exp(np.cumsum(0.6 * ief_ret + 0.4 * cred_ret))
    tlt_close = 95.0 * np.exp(np.cumsum(1.6 * ief_ret))

    return pd.DataFrame(
        {
            "IRX": yields[:, 0], "FVX": yields[:, 1],
            "TNX": yields[:, 2], "TYX": yields[:, 3],
            "SPY": spy_close, "HYG": hyg_close, "IEF": ief_close,
            "LQD": lqd_close, "TLT": tlt_close,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Content fingerprint (for the as-of stamp)
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the four yield columns (the as-of stamp)."""
    cols = [c for c in YIELD_COLS if c in df.columns] or [df.columns[0]]
    arr = np.ascontiguousarray(df[cols].fillna(0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
