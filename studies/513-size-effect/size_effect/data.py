"""Data layer for Study 513 (Size-Effect / Banz 1981 SMB).

Two tapes, one schema -- a (date x ticker) daily adjusted-price frame, SPY as the market
proxy, and a per-ticker market-cap *anchor* (current shares outstanding x price) so the
cross-section can be ranked by capitalisation through time.

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``size_premium``
  knob controls how much extra return the small-cap leg earns over the large-cap leg.
  ``size_premium = 0`` is the null hypothesis. Tests / CI never touch the network.

- ``fetch_panel`` -- real daily prices from yfinance, cached in the study's own ``_cache/``.
  Returns ``(prices, spy, caps)`` or empties if the cache is absent (e.g. on CI).

The market-cap series is built as ``cap_t = shares_now * price_t`` -- shares outstanding
held at their 2026 value and scanned backwards by adjusted price. This is a documented
approximation (it ignores buybacks/issuance), adequate for *ranking* a stable basket into
small vs large halves but stated openly as a limitation.

Universe: a fixed ~40-name survivor basket deliberately spanning the size spectrum -- mega-
caps (AAPL, MSFT, ...) down to genuine small/mid-caps (small regional banks, small-cap
industrials, micro-cap consumer names) -- so the small-minus-large sort has real dispersion.
The basket is **survivorship-biased** (names still trading in 2026 projected backwards):
positive results are **upper-bound** estimates and we name it.
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
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

# A fixed survivor basket spanning the size spectrum (mega-cap -> small/mid-cap).
# ~20 large, ~20 small/mid -- chosen for genuine cross-sectional cap dispersion.
LARGE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "PG", "HD", "KO",
    "PEP", "WMT", "CVX", "XOM", "MRK", "ABT", "MCD", "CSCO", "DIS", "INTC",
]
SMALL = [
    # genuine small / mid-cap survivors across sectors
    "SXT", "WDFC", "JJSF", "UFPT", "MGEE", "SCL", "PRGS", "NPK",
    "CASS", "CSGS", "WMK", "GFF", "STBA", "TRMK", "UVV", "NWN",
    "BRC", "WTS", "AIN", "MATX",
]
UNIVERSE = LARGE + SMALL
# Deduplicate preserving order
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    size_premium: float  # extra annual return earned by the small-cap leg

    @property
    def has_premium(self) -> bool:
        return self.size_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    n_days: int = 3000,
    size_premium: float = 0.05,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    seed: int = 513,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, WorldTruth]:
    """A reproducible daily returns panel with a tunable SMB (size) premium.

    Half the stocks are assigned small caps (log-uniform low) and half large caps
    (log-uniform high). Each stock follows a one-factor market model with a small-cap
    alpha:

        r_i = beta_i * r_mkt + alpha_i + epsilon_i,   alpha_i = size_premium * small_i / 252

    where ``small_i`` is 1 for the small half and 0 for the large half. ``size_premium = 0``
    is the null. Small-caps additionally carry a modestly higher beta and idio vol, as in
    the data.

    Returns ``(stock_rets, mkt_rets, caps, truth)`` -- daily decimal returns, plus a *static*
    cap anchor Series (the synthetic analogue of shares*price), used by the ranking sort.
    """
    rng = np.random.default_rng(seed)
    # period_range -> a short decorative business index (overflow-safe; ~12y of bdays).
    dates = pd.bdate_range("2014-01-02", periods=n_days)

    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    half = n_stocks // 2
    is_small = np.zeros(n_stocks)
    is_small[half:] = 1.0

    # Cap anchors: large names ~ 1e11-1e12, small ~ 1e8-1e9 (log-uniform).
    caps = np.empty(n_stocks)
    caps[:half] = np.exp(rng.uniform(np.log(1e11), np.log(1.5e12), size=half))
    caps[half:] = np.exp(rng.uniform(np.log(2e8), np.log(2e9), size=n_stocks - half))
    cap_series = pd.Series(caps, index=tickers, name="cap")

    betas = 0.95 + 0.25 * is_small + rng.normal(0, 0.08, size=n_stocks)
    betas = np.clip(betas, 0.4, 1.8)

    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")

    idio_scale = idio_vol * (1.0 + 0.3 * is_small) / np.sqrt(252)
    idio = rng.normal(0.0, 1.0, size=(n_days, n_stocks)) * idio_scale[None, :]

    alphas = size_premium * is_small / 252.0

    stock_rets = mkt_daily[:, None] * betas[None, :] + alphas[None, :] + idio
    stock_df = pd.DataFrame(stock_rets, index=dates, columns=tickers)
    return stock_df, mkt, cap_series, WorldTruth(size_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def _download_with_retry(tickers, start, end, tries=3, pause=2.0):
    import yfinance as yf

    last = None
    for k in range(tries):
        try:
            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                progress=False, threads=True,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(pause * (k + 1))
    if last is not None:
        raise last
    return None


def _fetch_caps(tickers, cache_dir):
    """Current marketCap / sharesOutstanding per ticker via yfinance .info (best effort)."""
    import yfinance as yf

    rows = {}
    for t in tickers:
        cap = np.nan
        for _ in range(2):
            try:
                info = yf.Ticker(t).info
                cap = info.get("marketCap") or np.nan
                if cap and not np.isnan(cap):
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
        rows[t] = float(cap) if cap else np.nan
    return pd.Series(rows, name="cap")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2001-01-01",
    end: str = "2026-06-01",
    fetch: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Daily adjusted-close prices for the basket + SPY, plus a current-cap anchor.

    Cache-first: reads the study's parquets if present. On an explicit ``fetch=True`` (or
    cache miss with network) pulls from yfinance with retries and writes the cache.
    Returns ``(prices, spy, caps)`` or empties if both cache and network fail.

    ``caps`` is the *current* marketCap per ticker; the time-varying cap used for ranking is
    reconstructed in ``strategy`` as ``cap_now * price_t / price_now`` (shares-constant
    approximation, documented).
    """
    price_path = os.path.join(cache_dir, "size_prices.parquet")
    spy_path = os.path.join(cache_dir, "size_spy.parquet")
    cap_path = os.path.join(cache_dir, "size_caps.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path) and os.path.exists(cap_path):
        prices = pd.read_parquet(price_path)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        spy = pd.read_parquet(spy_path).squeeze("columns")
        spy.name = "SPY"
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        caps = pd.read_parquet(cap_path).squeeze("columns")
        caps.name = "cap"
        return prices, spy, caps

    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY"), pd.Series(dtype=float, name="cap")

    try:
        all_t = list(UNIVERSE) + ["SPY"]
        raw = _download_with_retry(all_t, start, end)["Close"]
        if raw is None or raw.empty:
            return pd.DataFrame(), pd.Series(dtype=float, name="SPY"), pd.Series(dtype=float, name="cap")
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)

        spy = raw["SPY"].dropna()
        prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")
        # Keep names with decent coverage
        coverage = prices.notna().mean()
        prices = prices.loc[:, coverage >= 0.50]

        caps = _fetch_caps(list(prices.columns), cache_dir)

        os.makedirs(cache_dir, exist_ok=True)
        prices.to_parquet(price_path)
        spy.to_frame("SPY").to_parquet(spy_path)
        caps.to_frame("cap").to_parquet(cap_path)
        return prices, spy, caps

    except Exception:  # noqa: BLE001
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY"), pd.Series(dtype=float, name="cap")


def drop_partial_last_month(ret: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop the final monthly row if it is an in-progress month (no partial bar in a stamp)."""
    if ret.empty:
        return ret
    last = ret.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        return ret.iloc[:-1]
    return ret


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
