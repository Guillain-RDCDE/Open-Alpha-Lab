"""Data layer for Study 936 — Tolerance Bands (calendar vs 5/25-band rebalancing).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the four sleeves this study allocates
  across: SPY (US equity), IEF (7-10y Treasuries), GLD (gold) and BIL (1-3 month
  T-bills, the cash leg used for the excess-of-cash race). ``fetch`` touches the
  network and writes parquet into the **shared** desk cache ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic only), so CI
  is green on a fresh checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_panel`` / ``synthetic_daily`` — *deterministic, offline* generators.
  A common market factor plus per-asset idiosyncratic components whose AR(1)
  persistence is controlled by ``signal_strength``: at ``signal_strength=0`` the
  idiosyncratic legs are pure random walks (the null — assets drift apart and never
  come back, so there is nothing for a rebalancing rule to harvest); at
  ``signal_strength=1`` they are strongly mean-reverting (the planted world, where
  responding to *dispersion* rather than to the calendar should pay). Seeds are
  fixed, so the tests are deterministic.

The claim under test is a portfolio-construction folk rule, not a forecasting rule:
that rebalancing a fixed-weight book on **5/25 tolerance bands** (act when a sleeve
is off target by 5 percentage points absolute or 25% relative, whichever binds
first) beats rebalancing on the **calendar** — better excess-of-cash Sharpe for less
turnover. No look-ahead is baked in here; that discipline lives in ``strategy.py``
(a breach observed at the close of day ``t`` is traded at the close of ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache: studies/_cache (two levels up from the package directory).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# SPY = US equity sleeve; IEF = 7-10y Treasury sleeve; GLD = the third (real-asset)
# sleeve of the 3-asset book; BIL = 1-3M T-bills, the cash leg of the Sharpe race.
TICKERS = ("SPY", "IEF", "GLD", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2002-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` means
    the ``close`` column is split- and dividend-adjusted total return — essential
    here, because a 60/40 book's bond sleeve earns most of its return as coupons and
    a price-only IEF series would badly understate both the return and the drift of
    the weights away from target.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call rebal_bands.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns on the common (all-columns-present) window."""
    px = prices.dropna(how="any").sort_index()
    return px.pct_change().dropna(how="any")


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a numeric frame, for the as-of data stamp.

    Stamp the **returns** frame, not the levels frame, whenever the point is
    reproducibility. ``auto_adjust=True`` closes are rescaled by a cumulative
    dividend factor, so every new distribution silently multiplies the whole
    historical level series by a constant: the *levels* fingerprint churns on each
    refetch even though not a single return has changed. The returns fingerprint is
    invariant to that adjustment vintage, so it is the one that actually pins the
    numbers in ``docs/results.md``. ``verify.py`` prints both.
    """
    prices = frame
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 25,
    n_assets: int = 2,
    drift_ann=(0.07, 0.07, 0.07),  # annualised drift per asset
    vol_ann=(0.25, 0.25, 0.25),    # annualised total vol per asset
    market_share: float = 0.05,    # share of variance coming from the common factor
    half_life_days: float = 60.0,  # idiosyncratic mean-reversion half-life at strength 1
    signal_strength: float = 1.0,  # 0 = random-walk null, 1 = strong mean reversion
    cash_rate_ann: float = 0.025,
    start: str = "2001-01-02",
    seed: int = 936,
) -> tuple[pd.DataFrame, dict]:
    """A daily multi-asset total-return panel whose *dispersion* mean-reverts on a knob.

    Each asset's log price is a common market factor plus an idiosyncratic AR(1)
    component with persistence ``phi``. The ``signal_strength`` knob interpolates
    ``phi`` between 1.0 (a pure random walk — assets drift apart and never come back,
    so a rebalancing policy can only lose money to friction: **the null**) and
    ``exp(-ln 2 / half_life_days)`` (strong reversion — dispersion is transient, so
    buying the loser and selling the winner is genuinely rewarded: **the planted
    world**). Under reversion a *dispersion-triggered* (band) rule should beat a
    *calendar* rule, because it acts when there is something to harvest.

    The defaults are two equal-vol, equal-drift sleeves with almost no common factor:
    a deliberately favourable laboratory for band rebalancing, so that a *failure* to
    separate planted from null would be a harness bug rather than a weak experiment.

    Returns ``(prices, truth)``. ``prices`` has one column per asset (``a0``, ``a1``,
    …) plus ``cash`` (a cash accrual index); ``truth`` records the planted parameters,
    the realised ``phi`` and the realised dispersion half-life. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    drift = np.asarray(drift_ann, dtype=float)[:n_assets]
    vol = np.asarray(vol_ann, dtype=float)[:n_assets]
    if drift.size < n_assets or vol.size < n_assets:
        raise ValueError("drift_ann / vol_ann must supply one entry per asset")

    phi_rev = float(np.exp(-np.log(2.0) / half_life_days))
    phi = (1.0 - signal_strength) * 1.0 + signal_strength * phi_rev

    d = drift / TRADING_DAYS_PER_YEAR
    s = vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_mkt = s * np.sqrt(market_share)
    s_idio = s * np.sqrt(1.0 - market_share)

    mkt = rng.normal(0.0, 1.0, n_days)          # shared shock, unit scale
    eps = rng.normal(0.0, 1.0, (n_days, n_assets))

    # Idiosyncratic AR(1) level; the *increment* of the level is what hits the return.
    x = np.zeros((n_days, n_assets))
    prev = np.zeros(n_assets)
    for t in range(n_days):
        prev = phi * prev + s_idio * eps[t]
        x[t] = prev
    dx = np.vstack([x[0][None, :], np.diff(x, axis=0)])

    log_ret = d[None, :] + s_mkt[None, :] * mkt[:, None] + dx
    levels = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    cols = {f"a{i}": levels[:, i] for i in range(n_assets)}
    cols["cash"] = cash_idx
    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))

    # Realised dispersion persistence (a sanity read on the knob).
    if n_assets >= 2:
        spread = x[:, 0] - x[:, 1]
        num = float(spread[1:] @ spread[:-1])
        den = float(spread[:-1] @ spread[:-1])
        phi_hat = num / den if den > 0 else float("nan")
    else:
        phi_hat = float("nan")

    truth = {
        "signal_strength": signal_strength,
        "phi": float(phi),
        "phi_random_walk": 1.0,
        "phi_reverting": phi_rev,
        "phi_hat_spread": phi_hat,
        "half_life_days": half_life_days,
        "n_assets": int(n_assets),
        "n_days": int(n_days),
        "n_years": n_years,
        "seed": int(seed),
        "drift_ann": drift.tolist(),
        "vol_ann": vol.tolist(),
        "market_share": market_share,
        "cash_rate_ann": cash_rate_ann,
    }
    return prices, truth


def synthetic_daily(
    n_years: int = 25,
    signal_strength: float = 1.0,
    seed: int = 936,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Two-asset convenience wrapper around :func:`synthetic_panel` (a 60/40 stand-in)."""
    return synthetic_panel(
        n_years=n_years, n_assets=2, signal_strength=signal_strength, seed=seed, **kwargs
    )
