"""Data layer for Study 335 (Buzz-Sentiment-ETF).

The question is a *product* question, not a signal question: the VanEck Social
Sentiment ETF (ticker **BUZZ**, launched 2021-03) packages an AI-read of social
media into a 75-stock long-only US-equity portfolio. Does that wrapper actually
beat the broad market it is implicitly trying to outperform (SPY)? This is the
tradeable-wrapper cousin of Study 254 (WSB-Mentions) and Study 256 (Twitter-Mood),
which test the *raw* sentiment signals — here we test the shrink-wrapped fund a
retail buyer can actually click.

Two tapes, one shape (a tz-naive daily frame of total-return-style close levels
for two tickers, ``buzz`` and ``spy``):

- ``synthetic_pair`` — a *deterministic, offline* generator. BUZZ is modelled as
  SPY plus a market-beta tilt (``beta``), a *daily* alpha (``alpha_bps``), an
  idiosyncratic tracking-error term (``te_vol``) and an explicit expense-ratio
  drag (``fee_bps_yr``). The two knobs that matter:
    * ``alpha_bps = 0``  → a pure dressed-up beta bet: BUZZ is SPY at higher vol
      minus its fee, so an honest excess-vs-excess test must read **no alpha**.
    * ``alpha_bps > 0``  → a planted, harvestable edge — the positive control the
      harness must be able to bank.
  This is the study's null (and its positive control) in a bottle.

- ``load_real`` — the real tapes (BUZZ + SPY), cache-first via the shared
  ``_cache`` parquet (and ``quantlab.data`` if importable), with a graceful
  fallback that *raises* offline rather than silently fabricating a "real" run.

No look-ahead is baked in here — the racing discipline (one execution lag where a
timing overlay is involved, excess-of-cash vs excess-of-cash) lives in
``strategy.py``. BUZZ-vs-SPY is a *holding* race (both fully invested), so the
core comparison needs no lag at all; the lag convention is documented there for
the overlay variant.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_pair(
    n_days: int = 1000,
    alpha_bps: float = 0.0,
    beta: float = 1.15,
    te_vol: float = 0.006,
    fee_bps_yr: float = 75.0,
    mkt_vol: float = 0.011,
    mkt_drift_bps: float = 3.0,
    start: str = "2021-03-04",
    seed: int = 335,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible BUZZ/SPY total-return pair with a *known* amount of alpha.

    SPY log returns are i.i.d. normal (drift ``mkt_drift_bps`` bps/day, vol
    ``mkt_vol``). BUZZ log returns are built as a single-factor model::

        r_buzz_t = alpha_daily + beta * r_spy_t + te_t - fee_daily

    where ``alpha_daily = alpha_bps * 1e-4``, ``te_t`` is i.i.d. normal of
    standard deviation ``te_vol`` (the idiosyncratic tracking error of a
    concentrated 75-name basket) and ``fee_daily`` is the expense ratio spread
    evenly across the year (``fee_bps_yr`` annual → per-day drag).

    The sign of the verdict lives entirely in ``alpha_bps``:

    - ``alpha_bps = 0`` → BUZZ is dressed-up beta: same market exposure scaled by
      ``beta``, plus noise, minus a fee. An honest excess-vs-excess test must find
      **no alpha** (CAPM intercept ≈ −fee, indistinguishable from zero or worse).
    - ``alpha_bps > 0`` → a planted, real edge the harness must recover.

    Returns ``(levels, truth)`` where ``levels`` has columns ``buzz`` and ``spy``
    (close levels, both starting at 100.0) and ``truth`` records the planted
    parameters and the implied annual figures.
    """
    rng = np.random.default_rng(seed)
    # bdate_range is overflow-safe at these spans (well under ~290 years of ns).
    sessions = pd.bdate_range(start=start, periods=n_days)

    fee_daily = fee_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR
    alpha_daily = alpha_bps * 1e-4

    # n_days-1 returns so both legs start *exactly* at 100.0 (matches load_real's rebase).
    r_spy = rng.normal(mkt_drift_bps * 1e-4, mkt_vol, n_days - 1)
    te = rng.normal(0.0, te_vol, n_days - 1)
    r_buzz = alpha_daily + beta * r_spy + te - fee_daily

    spy = 100.0 * np.concatenate([[1.0], np.exp(np.cumsum(r_spy))])
    buzz = 100.0 * np.concatenate([[1.0], np.exp(np.cumsum(r_buzz))])

    levels = pd.DataFrame(
        {"buzz": buzz, "spy": spy},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "alpha_bps": alpha_bps,
        "alpha_bps_ann": alpha_bps * TRADING_DAYS_PER_YEAR / 100.0,  # %/yr
        "beta": beta,
        "te_vol": te_vol,
        "fee_bps_yr": fee_bps_yr,
        "n_days": n_days,
        "seed": seed,
    }
    return levels, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first, graceful offline fallback
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def _read_one(ticker: str, cache_dir: str, fetch: bool) -> pd.Series:
    """A single ticker's total-return close series; cache-first.

    Resolution order: (1) the study's local ``_cache`` parquet; (2) the shared
    ``quantlab.data`` loader if importable; (3) a live ``yfinance`` pull *only*
    when ``fetch=True``. Offline with no cache it raises — never fabricates.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
        s = bars["close"] if "close" in bars.columns else bars.iloc[:, 0]
        s.name = ticker.lower()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s

    # Shared engine loader, if the repo's quantlab is importable.
    try:
        from quantlab import data as qld  # type: ignore

        if hasattr(qld, "load_prices"):
            df = qld.load_prices([ticker])
            s = df[ticker] if ticker in df.columns else df.iloc[:, 0]
            s = s.rename(ticker.lower())
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            return s
    except Exception:
        pass

    if not fetch:
        raise FileNotFoundError(
            f"No cached tape for {ticker} at {path} and quantlab.data unavailable. "
            f"Call load_real(fetch=True) once to populate the cache (needs network)."
        )

    import yfinance as yf  # lazy: only on an explicit network opt-in

    raw = yf.download(ticker, start="2021-01-01", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].rename(ticker.lower())
    os.makedirs(cache_dir, exist_ok=True)
    out = s.to_frame("close")
    out.index.name = "date"
    out.to_parquet(path)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_real(
    buzz_ticker: str = "BUZZ",
    spy_ticker: str = "SPY",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real BUZZ + SPY total-return close levels on a common (inner-joined) index.

    Cache-first (see ``_read_one``); raises offline on a cache miss rather than
    fabricating a tape. The two series are inner-joined to a common calendar and
    rebased to 100.0 at the first common date, matching the synthetic tape's
    shape so the same analysis code runs on either.
    """
    buzz = _read_one(buzz_ticker, cache_dir, fetch)
    spy = _read_one(spy_ticker, cache_dir, fetch)
    df = pd.concat({"buzz": buzz, "spy": spy}, axis=1, sort=True).dropna()
    if df.empty:
        raise RuntimeError("No overlapping dates between BUZZ and SPY tapes.")
    df = df / df.iloc[0] * 100.0
    df.index.name = "date"
    return df


def fingerprint(levels: pd.DataFrame) -> str:
    """A short content fingerprint of a pair tape (both close columns)."""
    arr = np.ascontiguousarray(levels[["buzz", "spy"]].to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
