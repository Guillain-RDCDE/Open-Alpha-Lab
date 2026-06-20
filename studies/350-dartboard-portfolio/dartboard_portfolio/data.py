"""Data layer for Study 350 (Dartboard-Portfolio).

The Malkiel dartboard is a *cross-sectional selection* experiment: from a universe of
N stocks, a blindfolded monkey picks a handful at random, holds them equal-weight, and
is raced against the cap-weighted index of the same universe. So the data layer must
supply a **panel** — a (T x N) matrix of stock returns plus a (N,) vector of market
capitalisations that defines the index weights.

Two tapes, one shape (a return panel + a cap vector):

- ``synthetic_panel`` — a *deterministic, offline* generator. A single knob,
  ``size_premium``, plants the one thing that makes a random equal-weight basket beat
  the cap-weighted index: a small-cap return premium. At ``size_premium=0`` small and
  large names have the *same* expected return, so a random equal-weight dartboard and
  the cap-weighted index have the same expected return — the dartboard's "win" is then
  pure luck, symmetric around zero. At ``size_premium>0`` small caps out-earn large
  caps, so equal-weighting (which over-weights the many small names the index barely
  holds) beats the index *mechanically* — no skill, just a size tilt. This is the
  study's null and its positive control in one bottle.

- ``load_real`` — a real cross-section of large US stocks (the cap-weighted parent is
  the S&P 100 / OEF-style mega-cap set) via the shared ``quantlab.data`` loader with a
  cache-first parquet and graceful offline fallback. **Survivorship:** the panel is the
  *current* membership projected back, so it is biased upward and goes through the
  opt-in guard — see ``load_real``'s docstring and the caveat carried into the verdict.

No look-ahead is baked in here — weights for a holding window are formed from the cap
vector known *before* the window (the index is calendar-known; the dartboard's picks are
random and so need no signal lag at all).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# A small, liquid mega-cap set — the real "universe" the dartboard throws at. These are
# *current* members, so the loaded panel is survivorship-biased (see load_real).
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "XOM", "HD", "KO", "PEP", "MRK", "ABBV", "WMT", "CVX", "BAC", "DIS",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_periods: int = 240,
    n_assets: int = 100,
    size_premium: float = 0.0,
    market_vol: float = 0.04,
    idio_vol: float = 0.06,
    market_beta: float = 1.0,
    seed: int = 350,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible monthly return panel with a known small-cap premium.

    The cross-section has a power-law cap distribution (a few giant names, a long tail of
    small ones) — exactly the shape that makes cap-weighting concentrate in the top names.
    Each asset's monthly return is::

        r_it = beta_i * mkt_t + size_premium * size_score_i + eps_it

    where ``mkt_t`` is a common market factor, ``size_score_i`` is +1 for the smallest
    name and -1 for the largest (linear in cap rank), and ``eps_it`` is idiosyncratic.

    - ``size_premium = 0`` → small and large names have equal expected return. The
      cap-weighted index and a random equal-weight dartboard have the *same* expected
      return; the dartboard's edge is pure luck (symmetric, mean ~0).
    - ``size_premium > 0`` → small caps out-earn large caps, so equal-weighting beats the
      cap-weighted index *mechanically* (a size tilt, not stock-picking skill).

    Returns ``(returns_df, caps, truth)`` where ``returns_df`` is (n_periods x n_assets)
    monthly simple returns, ``caps`` is the (n_assets,) market-cap vector (index weights
    are ``caps / caps.sum()``), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)

    # Power-law caps: a handful of mega-caps, a long small tail (rank 0 = smallest).
    ranks = np.arange(n_assets)
    caps = 1.0 * (1.6 ** (ranks / max(n_assets - 1, 1) * 10.0))  # spans ~ 1 .. 110x
    caps = caps / caps.sum()

    # Size score: +1 smallest, -1 largest, linear in rank. Mean ~0 by construction.
    size_score = 1.0 - 2.0 * ranks / max(n_assets - 1, 1)

    betas = np.clip(rng.normal(market_beta, 0.20, n_assets), 0.3, 2.0)

    mkt = rng.normal(0.006, market_vol, n_periods)          # common factor
    eps = rng.normal(0.0, idio_vol, size=(n_periods, n_assets))
    # Expected-return tilt from the size premium (per period, constant in time).
    drift = size_premium * size_score                       # (n_assets,)

    raw = mkt[:, None] * betas[None, :] + drift[None, :] + eps

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M")
    # — that path overflows ns-Timestamps for long spans; see CI pitfalls).
    idx = pd.period_range("1990-01", periods=n_periods, freq="M").to_timestamp()
    cols = [f"A{i:03d}" for i in range(n_assets)]
    df = pd.DataFrame(raw, index=pd.DatetimeIndex(idx, name="date"), columns=cols)
    caps_s = pd.Series(caps, index=cols, name="cap")

    truth = {
        "n_periods": n_periods,
        "n_assets": n_assets,
        "size_premium": size_premium,
        "market_vol": market_vol,
        "idio_vol": idio_vol,
        "seed": seed,
        "size_score": size_score.tolist(),
    }
    return df, caps_s, truth


# ---------------------------------------------------------------------------
# Real tape — large US cross-section via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "dartboard_panel.parquet")


def _caps_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "dartboard_caps.parquet")


def load_real(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_survivorship_bias: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real monthly return panel + cap vector for the mega-cap universe.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and tests stay deterministic.

    **Survivorship.** ``tickers`` defaults to *current* mega-cap members projected back,
    so the panel is biased upward (the failures that fell out of the index were never in
    the sample). Cross-sectional use therefore requires ``allow_survivorship_bias=True``
    — an explicit opt-in mirroring ``quantlab.universe`` — and the caveat must travel with
    any published number. The bias here *helps the dartboard look good* relative to the
    real-world experience of a random picker, so it is named on the Signal axis.

    Returns ``(returns_df, caps)`` — monthly simple returns (T x N) and a cap vector (N,).
    """
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() builds a CURRENT-membership cross-section projected backwards, "
            "which is survivorship-biased (upward). Pass allow_survivorship_bias=True to "
            "opt in, and carry the caveat into every published number."
        )
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    rpath = _cache_path(cache_dir)
    cpath = _caps_cache_path(cache_dir)
    if not fetch:
        if not (os.path.exists(rpath) and os.path.exists(cpath)):
            raise FileNotFoundError(
                f"No cached panel at {rpath} / {cpath}. "
                f"Call load_real(fetch=True, allow_survivorship_bias=True) once to populate."
            )
        rets = pd.read_parquet(rpath)
        caps = pd.read_parquet(cpath)["cap"]
    else:
        prices = _fetch_prices(tickers, start, cache_dir)
        # Monthly simple returns from month-end TOTAL-RETURN closes.
        monthly = prices.resample("ME").last()
        rets = monthly.pct_change().dropna(how="all")
        # Drop the in-progress (partial) final month so a stamped run never includes a
        # partial bar (house rule: as-of is never in the future).
        today = pd.Timestamp.today().normalize()
        rets = rets[rets.index < today.to_period("M").to_timestamp()]
        # Market caps: a real snapshot via yfinance fast_info (a *current* snapshot, so
        # the cap weights are not point-in-time — a documented limitation that, like the
        # membership choice, biases the read; the experiment needs the cap *ordering* and
        # the concentration shape, both of which the snapshot captures faithfully).
        caps = _fetch_caps(list(rets.columns))
        os.makedirs(cache_dir, exist_ok=True)
        rets.to_parquet(rpath)
        caps.to_frame().to_parquet(cpath)

    rets = rets[[c for c in tickers if c in rets.columns]].dropna()
    caps = caps.reindex(rets.columns).dropna()
    rets = rets[caps.index]
    return rets, caps


def _fetch_prices(tickers: list[str], start: str, cache_dir: str) -> pd.DataFrame:
    """Daily adjusted closes for the universe via the shared loader (network on call)."""
    try:
        from quantlab import data as qld  # shared loader, if importable

        frames = {}
        for t in tickers:
            try:
                bars = qld.load(t, start=start, mode="total_return")
                frames[t] = bars["close"] if "close" in bars else bars.iloc[:, 0]
            except Exception:
                continue
        if frames:
            px = pd.DataFrame(frames)
            if px.index.tz is not None:
                px.index = px.index.tz_localize(None)
            return px.dropna(how="all")
    except Exception:
        pass

    # Fallback: yfinance directly.
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("no price data returned for the dartboard universe")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def _fetch_caps(tickers: list[str]) -> pd.Series:
    """Current market-cap snapshot via yfinance fast_info (network on call)."""
    import yfinance as yf

    caps = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            mc = fi.get("market_cap") or fi.get("marketCap")
            if mc:
                caps[t] = float(mc)
        except Exception:
            continue
    if not caps:
        raise RuntimeError("could not fetch any market caps for the dartboard universe")
    return pd.Series(caps, name="cap")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel, for the as-of stamp."""
    arr = np.ascontiguousarray(returns.to_numpy().ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
