"""Data layer for Study 508 (Momentum-Crashes).

Two tapes, one schema -- a (date x ticker) daily adjusted-close price frame, plus a market
proxy (SPY) used only for the bear-market / panic-regime indicators of Daniel & Moskowitz
(2016):

- ``synthetic_panel`` -- a *deterministic, offline* generator. A ``mom_strength`` knob plants
  a persistent relative-strength drift, and an optional ``crash_strength`` knob plants the
  asymmetric "loser snap-back" the paper describes: after a synthetic bear market, the
  past-loser leg rebounds violently, so the WML book takes a planted crash. ``mom_strength=0``
  is the null. Tests and the reproducible core never touch the network.

- ``fetch_panel`` / ``fetch_market`` -- real daily adjusted-close prices from yfinance, cached
  to the study's OWN ``_cache/`` dir. Cache-first: returns the cached parquet if present, else
  an empty frame (so notebooks banner "synthetic tape" on CI) unless ``fetch=True``.

The basket is **survivorship-biased**: ~38 large-cap names still trading in 2026, projected
backwards. The natural short candidates of a momentum loser leg -- firms that trended down into
delisting/bankruptcy -- are absent by construction. Crashes here are therefore an *under*-
statement of the historical momentum crash, which is exactly why this study reads the real-tape
numbers as a floor on the crash and an upper bound on the premium. Named on the SIGNAL axis.

No look-ahead lives here: the 12-1 signal (skip the most recent month), the single execution
lag (enter one month after the signal is public), and the regime conditioning live in
``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed ~38-name large-cap survivor basket (sector-spread, shared with Study 507).
# Deliberately a retail-realistic basket where the loser leg is thin -- which makes the
# crash dynamics, not statistical power, the third-axis question of this study.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "XOM", "CVX", "KO", "PEP", "MRK", "ABBV", "WMT", "COST",
    "MCD", "DIS", "CSCO", "INTC", "ORCL", "IBM", "TXN", "QCOM", "HON", "CAT",
    "GE", "BA", "MMM", "UNH", "PFE", "T", "VZ", "GS", "AXP", "LOW",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

MARKET = "SPY"
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted relative-strength drift and crash asymmetry for a synthetic panel."""

    mom_strength: float   # annualised persistent winner drift (0 = pure random-walk null)
    crash_strength: float  # planted loser-rebound severity after the synthetic bear

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    n_days: int = 2600,
    mom_strength: float = 0.0,
    crash_strength: float = 0.0,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    persistence: int = 252,
    seed: int = 508,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily adjusted-close panel with a tunable momentum drift and crash.

    Each stock loads on a common market factor (beta ~ 1) plus an idiosyncratic component.
    When ``mom_strength > 0`` a persistent latent trend adds a relative-strength drift, so
    past winners keep winning -- the Jegadeesh-Titman persistence. When ``crash_strength > 0``
    a synthetic bear market is carved into the middle of the sample (the market factor turns
    sharply negative for a stretch, then rebounds), and during the rebound the *low-trend*
    (loser) names get an extra positive kick scaled by ``crash_strength`` -- the asymmetric
    loser snap-back Daniel-Moskowitz (2016) identify as the engine of momentum crashes.

    Returns ``(prices, market, truth)`` -- ``prices`` a (date x ticker) frame, ``market`` the
    SPY-like proxy series. A *decorative* business-day index is built over a small ``n_days``
    (months, never a huge nanosecond span) -- safely under the ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"S{j:02d}" for j in range(n_stocks)]

    # Market factor: a steady drift, with a carved bear-then-rebound in the middle third.
    mkt = rng.normal(0.08 / TRADING_DAYS, market_vol / np.sqrt(TRADING_DAYS), size=n_days)
    bear_lo, bear_hi = int(n_days * 0.45), int(n_days * 0.55)
    rebound_lo, rebound_hi = bear_hi, int(n_days * 0.62)
    if crash_strength > 0.0:
        mkt[bear_lo:bear_hi] -= 0.9 / TRADING_DAYS * 4.0   # sharp synthetic bear
        mkt[rebound_lo:rebound_hi] += 0.9 / TRADING_DAYS * 6.0  # violent rebound

    idio = rng.normal(0.0, idio_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_stocks))

    # Latent persistent trend: an AR(1)-like sign-persistent process per stock.
    phi = np.exp(-1.0 / max(persistence, 1))
    shock_sd = np.sqrt(1.0 - phi**2)
    trend = np.zeros((n_days, n_stocks))
    trend[0] = rng.standard_normal(n_stocks)
    innov = rng.standard_normal((n_days, n_stocks))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + shock_sd * innov[t]

    drift = mom_strength / TRADING_DAYS * trend
    rets = mkt[:, None] + idio + drift

    # The loser snap-back: during the rebound, names with the most-negative trend at the
    # bear trough (the would-be losers) get an extra positive kick -> a planted WML crash.
    if crash_strength > 0.0:
        trough_trend = trend[bear_hi]  # latent strength at the trough
        loser_mask = (trough_trend <= np.quantile(trough_trend, 0.25)).astype(float)
        kick = crash_strength / TRADING_DAYS * 8.0
        rets[rebound_lo:rebound_hi] += kick * loser_mask[None, :]

    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"

    market = pd.Series((1.0 + mkt).cumprod() * 100.0, index=dates, name=MARKET)
    market.index.name = "date"
    return prices, market, WorldTruth(mom_strength, crash_strength)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel adjusted-close parquet."""
    return os.path.join(cache_dir, "momcrash_prices.parquet")


def market_cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached market-proxy (SPY) adjusted-close parquet."""
    return os.path.join(cache_dir, "momcrash_spy.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted-close prices for the large-cap survivor basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame (CI/offline path). Network is touched only on ``fetch=True`` (guarded
    with retries against yfinance flakiness), then cached as adjusted close (auto_adjust=True).
    """
    p = cache_path(cache_dir)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                list(UNIVERSE),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            coverage = raw.notna().mean()
            raw = raw.loc[:, coverage >= 0.80].dropna(how="all")
            raw = raw.ffill().dropna(how="any")
            raw.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def fetch_market(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> pd.Series:
    """Daily adjusted-close SPY (the market proxy for regime conditioning), cache-first."""
    p = market_cache_path(cache_dir)
    if os.path.exists(p):
        s = pd.read_parquet(p).squeeze("columns")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.name = MARKET
        return s
    if not fetch:
        return pd.Series(dtype=float, name=MARKET)

    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(MARKET, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance SPY frame")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            s = raw.squeeze("columns") if isinstance(raw, pd.DataFrame) else raw
            s.name = MARKET
            return s
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance SPY fetch failed after {retries} tries: {last_err}")


def drop_partial_last_month(prices: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop trailing in-progress-month rows (house rule: no partial month in a stamp)."""
    if prices.empty:
        return prices
    last = prices.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        keep = ~((prices.index.year == today.year) & (prices.index.month == today.month))
        return prices.loc[keep]
    return prices


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
