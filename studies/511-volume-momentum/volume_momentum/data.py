"""Data layer for Study 511 (Volume-Momentum, Lee-Swaminathan).

Two tapes, one schema -- a daily ``(date x ticker)`` adjusted-close **price** frame AND a parallel
daily ``(date x ticker)`` **dollar-volume** frame (close x volume, the liquidity proxy we sort on):

- ``synthetic_panel`` -- a *deterministic, offline* generator. Two knobs: ``mom_strength``
  controls the cross-sectional relative-strength drift (the raw momentum), and ``vol_tilt``
  controls how much of that drift is *concentrated in the high-turnover names* (the
  Lee-Swaminathan "high-volume winners / low-volume losers" pattern). The planted truth: the WML
  spread is bigger inside the high-volume half. ``mom_strength = 0`` is the null. Tests and the
  reproducible core never touch the network.

- ``fetch_panel`` -- real daily adjusted-close prices + raw volume from yfinance, cached to the
  study's OWN ``_cache/`` dir. Cache-first: returns the cached parquets if present, else an empty
  pair (so notebooks banner "synthetic tape" on CI) unless ``fetch=True``.

The basket is **survivorship-biased**: ~40 large-cap names still trading in 2026, projected
backwards. The natural short candidates of a momentum loser leg -- firms that trended down into
delisting/bankruptcy -- are absent by construction. This is named openly on the SIGNAL axis (not
just tradability): it directly inflates any apparent winners-minus-losers premium, and in the
Lee-Swaminathan world the *low-volume losers* (their strongest short leg) are precisely the quiet
names that fade into delisting -- so the volume-conditioned slice carries the bias too. Opt-in
guard: pass a delisting-complete panel to ``strategy.long_short`` to lift the bias; we cannot
from yfinance, so we flag it.

No look-ahead lives here: the 12-1 momentum signal, the trailing turnover measure, the
double-sort and the single forward execution lag all live in ``strategy.py``.
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

# Fixed ~40-name large-cap survivor basket (stable subset of the S&P 500, sector-spread).
# Same basket family as Studies 507/510 (cross-sectional momentum): on ~40 names the double-sort
# into momentum x volume halves leaves only a handful of names per cell -- exactly the thin-cell
# fragility the third axis probes.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "XOM", "CVX", "KO", "PEP", "MRK", "ABBV", "WMT", "COST",
    "MCD", "DIS", "CSCO", "INTC", "ORCL", "IBM", "TXN", "QCOM", "HON", "CAT",
    "GE", "BA", "MMM", "UNH", "PFE", "T", "VZ", "GS", "AXP", "LOW",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for a synthetic panel."""

    mom_strength: float   # annualised persistent winner drift (0 = pure random walk null)
    vol_tilt: float       # share of the drift concentrated in the high-turnover names

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
    vol_tilt: float = 1.0,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    persistence: int = 252,
    seed: int = 511,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible daily ``(price, dollar-volume)`` panel with a tunable, volume-conditioned
    momentum drift.

    Each stock loads on a common market factor (beta ~ 1) plus an idiosyncratic component. When
    ``mom_strength > 0`` a *persistent* drift is added (an AR(1) sign-persistent latent trend,
    exactly as in Studies 507/510). The new ingredient is WHERE that drift lands relative to
    turnover:

    - Each stock carries a fixed **turnover rank** (a baseline log dollar-volume level plus daily
      noise). The high-turnover names (top half) receive a ``vol_tilt`` share of the persistent
      drift; the low-turnover names receive ``(1 - vol_tilt)`` of it. With ``vol_tilt > 0.5`` the
      WML spread is concentrated in the **high-volume** half -- the Lee-Swaminathan pattern.

    The planted truth (the literature claim) is that the WML spread is bigger inside the
    high-volume half. ``mom_strength = 0`` is the null: pure random walks, no cross-sectional
    persistence, so neither slice earns anything.

    Returns ``(prices, dollar_volume, truth)``. The decorative index uses ``pd.bdate_range`` over
    a small ``n_days`` (months, never a huge nanosecond span) -- safely under the ns overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"S{j:02d}" for j in range(n_stocks)]

    mkt = rng.normal(0.08 / TRADING_DAYS, market_vol / np.sqrt(TRADING_DAYS), size=n_days)
    idio = rng.normal(0.0, idio_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_stocks))

    # Latent persistent trend: AR(1)-like, sign-persistent per stock.
    phi = np.exp(-1.0 / max(persistence, 1))
    shock_sd = np.sqrt(1.0 - phi**2)
    trend = np.zeros((n_days, n_stocks))
    trend[0] = rng.standard_normal(n_stocks)
    innov = rng.standard_normal((n_days, n_stocks))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + shock_sd * innov[t]

    # Fixed per-stock turnover level: half the names are "high turnover", half "low". The level
    # is the log of a baseline dollar volume; daily noise jiggles it so the rank is realistic.
    base_turn = np.linspace(0.4, 1.6, n_stocks)
    rng.shuffle(base_turn)
    high_turn = base_turn >= np.median(base_turn)   # top half by turnover

    # Per-stock drift weight: high-turnover names get vol_tilt of the drift, low-turnover get the
    # rest. (Scaled by 2 so that at vol_tilt=0.5 every name gets the full average drift.)
    w = np.where(high_turn, 2.0 * vol_tilt, 2.0 * (1.0 - vol_tilt))
    drift = (mom_strength / TRADING_DAYS) * trend * w[None, :]

    rets = mkt[:, None] + idio + drift
    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"

    # Dollar volume = baseline level x price x daily lognormal noise (so turnover ranks are stable
    # but not frozen). Units are arbitrary; only the cross-sectional RANK is used downstream.
    vnoise = np.exp(rng.normal(0.0, 0.35, size=(n_days, n_stocks)))
    dollar_vol = pd.DataFrame(
        (base_turn[None, :] * 1e7) * prices.to_numpy() * vnoise,
        index=dates, columns=tickers,
    )
    dollar_vol.index.name = "date"
    return prices, dollar_vol, WorldTruth(mom_strength, vol_tilt)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def price_cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel adjusted-close parquet."""
    return os.path.join(cache_dir, "vm_prices.parquet")


def volume_cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel dollar-volume parquet."""
    return os.path.join(cache_dir, "vm_dollarvol.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily adjusted-close prices + dollar volume for the large-cap survivor basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquets if present, else a pair
    of **empty** DataFrames (CI/offline path). Network is touched only on ``fetch=True`` (guarded
    with retries against yfinance flakiness), then cached. Price is adjusted close
    (auto_adjust=True); dollar volume = adjusted close x raw share volume, the liquidity proxy we
    sort on. The caller computes returns via ``prices.pct_change()``.
    """
    pp = price_cache_path(cache_dir)
    vp = volume_cache_path(cache_dir)
    if os.path.exists(pp) and os.path.exists(vp):
        px = pd.read_parquet(pp)
        dv = pd.read_parquet(vp)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        if dv.index.tz is not None:
            dv.index = dv.index.tz_localize(None)
        return px, dv
    if not fetch:
        return pd.DataFrame(), pd.DataFrame()

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
            )
            close = raw["Close"]
            vol = raw["Volume"]
            if close.empty:
                raise RuntimeError("empty yfinance frame")
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            vol.index = pd.DatetimeIndex(vol.index).tz_localize(None)
            coverage = close.notna().mean()
            keep = coverage[coverage >= 0.80].index
            close = close.loc[:, keep].dropna(how="all").ffill().dropna(how="any")
            vol = vol.loc[close.index, close.columns].ffill().fillna(0.0)
            dollar_vol = close * vol
            close.index.name = "date"
            dollar_vol.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            close.to_parquet(pp)
            dollar_vol.to_parquet(vp)
            return close, dollar_vol
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def drop_partial_last_month(prices: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop trailing in-progress-month rows (house rule: no partial month in a stamp)."""
    if prices.empty:
        return prices
    today = pd.Timestamp.today().normalize()
    last = prices.index[-1]
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
