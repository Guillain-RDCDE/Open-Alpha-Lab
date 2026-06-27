"""Data layer for Study 518 (Time-Series-Momentum).

Moskowitz-Ooi-Pedersen (2012) momentum is an ASSET-OWN-SIGN signal: each instrument is
traded long or short according to the sign of ITS OWN trailing 12-month return, then
vol-scaled to a common risk target and diversified across asset classes. So the natural
tape is a small panel of liquid, cross-asset ETFs (equity / bond / commodity / FX / gold),
not a large equity cross-section.

Two tapes, one schema -- a (date x ticker) daily adjusted-close price frame:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``trend_strength``
  knob plants a persistent OWN-asset trend in each series. ``trend_strength = 0`` is the
  null: pure random walks with no own-sign persistence, so the TSMOM book earns nothing.
  Tests and the reproducible core never touch the network.

- ``fetch_panel`` -- real daily adjusted-close prices from yfinance, cached to the study's
  OWN ``_cache/`` dir. Cache-first: returns the cached parquet if present, else an empty
  frame (so notebooks can banner "synthetic tape" on CI) unless ``fetch=True``.

The basket is **survivorship-biased**: it is ~12 cross-asset ETFs still trading in 2026,
projected backwards. TSMOM's *own-sign* construction makes survivorship a far milder issue
than a cross-sectional equity sort (no name ever "delists" -- a broad ETF that fell simply
gets shorted), but it is still named on the SIGNAL axis: the ETF list itself was chosen ex
post (no ETF that closed is here), and a vehicle that survived to 2026 had no terminal crash.
We name it openly and treat any TSMOM premium as an upper bound. See the guard note in
``strategy.py`` and the README.

No look-ahead lives here: the 12-month own-return sign and the single execution lag (enter
one trading day after the signal is public) live in ``strategy.py``.
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

# Fixed cross-asset ETF basket spanning equity / bond / commodity / FX / gold.
# Deliberately diversified (low pairwise correlation) -- TSMOM's diversification, not a
# single equity cross-section, is the whole point of Moskowitz-Ooi-Pedersen.
#   SPY  US large-cap equity        EFA  developed-ex-US equity
#   EEM  emerging-market equity     IWM  US small-cap equity
#   TLT  long Treasuries            IEF  7-10y Treasuries
#   LQD  IG corporate credit        GLD  gold
#   DBC  broad commodities          USO  crude oil
#   UUP  US dollar (FX)             FXE  euro (FX)
UNIVERSE = [
    "SPY", "EFA", "EEM", "IWM",
    "TLT", "IEF", "LQD",
    "GLD", "DBC", "USO",
    "UUP", "FXE",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# Human-readable asset-class labels (for the notebooks).
ASSET_CLASS = {
    "SPY": "US equity", "EFA": "Dev-ex-US equity", "EEM": "EM equity", "IWM": "US small-cap",
    "TLT": "Long Treasuries", "IEF": "7-10y Treasuries", "LQD": "IG credit",
    "GLD": "Gold", "DBC": "Commodities", "USO": "Crude oil",
    "UUP": "US dollar", "FXE": "Euro",
}

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted own-asset trend for a synthetic panel."""

    trend_strength: float  # annualised persistent own-trend (0 = pure random walk null)

    @property
    def has_trend(self) -> bool:
        return self.trend_strength != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_assets: int = 12,
    n_days: int = 2600,
    trend_strength: float = 0.0,
    asset_vol: float = 0.16,
    persistence: int = 252,
    seed: int = 518,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily adjusted-close panel with a tunable OWN-asset trend.

    Each asset is a near-independent diffusion (low cross-correlation, like a cross-asset
    ETF panel). When ``trend_strength > 0`` a *persistent* drift is added to each series:
    every asset carries a slowly varying latent "trend" whose sign tends to persist over
    ``persistence`` days, and the drift contribution is ``trend_strength`` (annualised) times
    that latent trend. Each asset thus tends to keep going the way it has been going over the
    trailing year -- exactly the OWN-sign persistence Moskowitz-Ooi-Pedersen exploit.
    ``trend_strength = 0`` is the null: pure random walks, no own-sign persistence, so the
    TSMOM book earns nothing.

    Returns ``(prices, truth)`` where ``prices`` is a (date x ticker) adjusted-close frame.
    A *decorative* business-day index is built with ``pd.bdate_range`` over a small ``n_days``
    (months, never a huge nanosecond span) -- safely under the ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"A{j:02d}" for j in range(n_assets)]

    # Mostly idiosyncratic (cross-asset) plus a small common shock.
    common = rng.normal(0.0, 0.05 / np.sqrt(TRADING_DAYS), size=n_days)
    idio = rng.normal(0.0, asset_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_assets))

    # Latent persistent trend: an AR(1)-like sign-persistent process per asset.
    phi = np.exp(-1.0 / max(persistence, 1))  # daily persistence coefficient
    shock_sd = np.sqrt(1.0 - phi**2)
    trend = np.zeros((n_days, n_assets))
    trend[0] = rng.standard_normal(n_assets)
    innov = rng.standard_normal((n_days, n_assets))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + shock_sd * innov[t]

    drift = trend_strength / TRADING_DAYS * trend  # annualised drift -> daily
    rets = common[:, None] + idio + drift

    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"
    return prices, WorldTruth(trend_strength)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel adjusted-close parquet."""
    return os.path.join(cache_dir, "tsmom_prices.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2006-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted-close prices for the cross-asset ETF basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame (CI/offline path -> notebooks fall back to frozen numbers). Network is
    touched only on ``fetch=True`` (guarded with retries against yfinance flakiness), then the
    result is cached as adjusted close (auto_adjust=True). The caller computes returns via
    ``prices.pct_change()``.
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
            # Keep names with >= 60% coverage (some ETFs list later than 2006).
            coverage = raw.notna().mean()
            raw = raw.loc[:, coverage >= 0.60].dropna(how="all")
            raw = raw.ffill().dropna(how="any")
            raw.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def drop_partial_last_month(prices: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop trailing in-progress-month rows (house rule: no partial month in a stamp).

    If the last bar falls in the current calendar month before ``asof_day``, the whole
    current-month tail is dropped so the stamped run contains only completed months.
    """
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
