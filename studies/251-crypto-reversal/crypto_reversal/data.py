"""Data layer for Study 251 (Crypto-Reversal).

Two tapes, one shape (a date-indexed wide frame of daily closes, one column per token):

- ``synthetic_panel`` — a *deterministic, offline* cross-sectional generator. Each token's
  daily return is a market component plus an idiosyncratic component that mean-reverts on a
  ~21-day horizon. The ``signal_strength`` knob controls how strongly idiosyncratic moves
  reverse: at ``signal_strength=0`` there is no cross-sectional predictability (the null — a
  reversal long-short and a momentum long-short both earn ~0); at higher values recent
  losers bounce back and a reversal long-short earns a positive spread while a momentum
  long-short earns a negative one. The seed is fixed so tests and the demo are fully
  deterministic and offline. This is the positive/negative control for the real-tape claim.

- ``fetch_panel`` — daily adjusted closes for a fixed crypto universe from Yahoo! Finance
  (``yfinance``), cache-only by default. ``fetch=False`` raises ``FileNotFoundError`` if the
  parquet is absent, so the test-suite and offline core never touch the network.
  ``fetch=True`` downloads and caches the whole panel as one parquet.

Survivorship is the load-bearing caveat (named on the SIGNAL axis in the verdict): the
yfinance universe contains tokens that *exist and trade today*, projected backward. Dead
tokens (LUNA, FTT, many 2021 alt-L1s) are absent — exactly the failure the paper itself
flags as inflating all reported factor returns. We do not pretend to fix it; we name it.

No look-ahead is baked in here — signal lagging lives in ``strategy.py`` (the cross-section
is ranked on data up to day ``t-1`` and the long-short return is earned over ``t-1 -> t``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Crypto trades 365 days per year (no market holidays).
TRADING_DAYS_PER_YEAR = 365

# A fixed, liquid, non-stablecoin universe with deep yfinance history (≥2019 for the core,
# 2020 for the L1 newcomers). Stablecoins, wrapped tokens and exchange-native chains
# excluded, matching the paper's universe rules. This is a *survivor* universe — see module
# docstring. ~20 tokens give quintiles of ~4 names, the same coarse cross-section the paper
# works with for its on-chain factors (53 tokens, quintiles of ~10).
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD",
    "LINK-USD", "XLM-USD", "TRX-USD", "ETC-USD", "XMR-USD", "EOS-USD", "DASH-USD",
    "ZEC-USD", "NEO-USD", "BNB-USD", "ATOM-USD", "VET-USD", "ALGO-USD",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline cross-sectional core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_tokens: int = 20,
    n_years: int = 6,
    mkt_vol_ann: float = 0.65,          # market factor vol (crypto scale)
    idio_vol_ann: float = 0.90,         # idiosyncratic vol (crypto scale)
    revert_horizon: int = 21,           # horizon (days) of the planted mean reversion
    signal_strength: float = 1.0,       # 0 = no cross-sectional predictability (null)
    start: str = "2019-01-01",
    seed: int = 213,
) -> tuple[pd.DataFrame, dict]:
    """A daily crypto-like price panel with a planted ~21-day idiosyncratic reversal.

    The data-generating process for token *i* on day *t*:

        r[i,t] = beta[i] * mkt[t] + eps[i,t]
        eps[i,t] = -kappa * (sum of eps[i, t-1-revert_horizon : t-1]) + white_noise

    where ``kappa = signal_strength * k0``. With ``kappa > 0`` a token whose idiosyncratic
    return has run up over the past ``revert_horizon`` days tends to give a *negative*
    idiosyncratic return next — recent losers bounce, winners fade — so a cross-sectional
    *reversal* long-short (long losers, short winners) earns a positive spread and the
    *momentum* long-short earns a negative one. At ``signal_strength=0`` there is no
    reversion and neither long-short has an edge (the null).

    Returns ``(prices, truth)`` where ``prices`` is a date-indexed wide frame of close
    levels (one column per ``TOK00..TOK{n-1}``) and ``truth`` records the planted knobs.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    s_mkt = mkt_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_idio = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mkt = rng.normal(0.0, s_mkt, n_days)
    beta = rng.uniform(0.7, 1.3, n_tokens)

    # k0 calibrated so signal_strength=1 plants an economically meaningful reversal
    # without exploding the series. Reversion pulls back a small fraction of the
    # accumulated idiosyncratic move each day.
    k0 = 0.04 / revert_horizon
    kappa = signal_strength * k0

    eps = np.zeros((n_days, n_tokens))
    noise = rng.normal(0.0, s_idio, (n_days, n_tokens))
    for t in range(n_days):
        lo = max(0, t - revert_horizon)
        accrued = eps[lo:t].sum(axis=0) if t > 0 else np.zeros(n_tokens)
        eps[t] = noise[t] - kappa * accrued

    r = beta[None, :] * mkt[:, None] + eps
    close = 100.0 * np.exp(np.cumsum(r, axis=0))
    cols = [f"TOK{i:02d}" for i in range(n_tokens)]
    prices = pd.DataFrame(close, index=pd.DatetimeIndex(dates, name="date"), columns=cols)

    truth = {
        "n_tokens": n_tokens,
        "n_years": n_years,
        "n_days": n_days,
        "revert_horizon": revert_horizon,
        "signal_strength": signal_strength,
        "kappa": float(kappa),
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily panel, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(cache_dir, "crypto_panel_1d.parquet")


def fetch_panel(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    min_coverage: float = 0.60,
) -> pd.DataFrame:
    """Daily adjusted-close panel for the crypto universe; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the whole panel is then cached
    as a single parquet under ``_cache/``). Returns a date-indexed wide frame, one column
    per token, with columns whose non-null coverage is below ``min_coverage`` dropped.

    Uses ``auto_adjust=True``. Crypto trades every calendar day; the index has no weekday
    gaps. A token's column may still have leading NaNs before it began trading — the
    strategy layer handles ragged starts via per-day cross-sections.
    """
    tickers = tickers or CRYPTO_UNIVERSE
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached crypto panel at {path}. "
                f"Call fetch_panel(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            tickers, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for the crypto universe")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        df = close.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # Drop thinly-covered columns (tokens that barely trade over the window).
    keep = df.columns[df.notna().mean() >= min_coverage]
    df = df[keep].sort_index()
    return df


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the panel (all closes), for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(panel.to_numpy(dtype=float)))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
