"""Data layer for Study 795 — Corporate-Bond-Momentum.

The claim (Jostova, Nikolova, Philipov & Stahel 2013) is a CROSS-SECTIONAL momentum
signal on bonds: rank a bond universe on trailing 6-12 month total return and go long
the recent winners, short the recent losers. So the natural tape is a small panel of
liquid credit + Treasury bond ETFs, ranked *against each other* each month.

Two tapes, one schema — a (date x ticker) daily adjusted-close (total-return) frame:

* ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``mom_strength``
  knob plants a PERSISTENT per-asset trend, so recent winners tend to keep winning and a
  cross-sectional winners-minus-losers book earns a premium. ``mom_strength = 0`` is the
  null: pure random walks with no rank persistence, so the momentum book earns nothing.
  Tests and the reproducible core never touch the network.

* ``fetch_panel`` — real daily total-return prices from yfinance, cached to the study's
  OWN ``_cache/`` dir. Cache-first: returns the cached parquet if present, else an empty
  frame (so notebooks can banner "synthetic tape" on CI) unless ``fetch=True``.

**Survivorship (named on the Signal axis).** The basket is credit/Treasury ETFs still
trading in 2026, chosen ex post and projected backwards. This is milder than a
single-name credit sort — a broad ETF that falls simply gets *shorted*, it does not
"delist" mid-sample — but the vehicle list itself was picked with hindsight (no ETF that
closed is here), so any momentum premium measured here is treated as an upper bound. The
panel loads through :func:`fetch_panel`, whose docstring repeats the caveat, and the
README/notebooks carry it on the Signal axis.

No look-ahead lives here: the trailing-return signal and the single execution lag (form
on the month-*t* close, earn month *t+1*) live in ``strategy.py``.
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

# Credit + Treasury bond-ETF universe. Deliberately spans the credit-quality ladder so a
# cross-sectional momentum sort has real dispersion to rank on:
#   Credit (the winners/losers the claim is about):
#     LQD   IG corporate credit          VCLT  long-dated IG corporate
#     VCSH  short-dated IG corporate     HYG   high-yield corporate
#     JNK   high-yield corporate         EMB   EM USD sovereign
#     BKLN  senior bank loans (floating) ANGL  fallen-angel high-yield
#   Treasuries (the "duration" pole of the ladder — the natural short/long when credit
#   rallies or sells off):
#     SHY   1-3y Treasuries              IEF   7-10y Treasuries
#     TLT   20y+ Treasuries
UNIVERSE = [
    "LQD", "VCLT", "VCSH", "HYG", "JNK", "EMB", "BKLN", "ANGL",
    "SHY", "IEF", "TLT",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# Human-readable labels (for the notebooks).
SLEEVE = {
    "LQD": "IG corporate", "VCLT": "Long IG corp", "VCSH": "Short IG corp",
    "HYG": "High yield", "JNK": "High yield", "EMB": "EM sovereign",
    "BKLN": "Bank loans", "ANGL": "Fallen angels",
    "SHY": "1-3y Treasury", "IEF": "7-10y Treasury", "TLT": "20y+ Treasury",
}

TRADING_DAYS = 252
START = "2007-01-01"
# Last complete calendar month at publication (built 2026-07-22); partial July dropped.
AS_OF = "2026-06-30"


@dataclass(frozen=True)
class WorldTruth:
    """The planted cross-sectional-momentum strength for a synthetic panel."""

    mom_strength: float  # annualised persistent own-trend (0 = pure random walk null)

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline positive control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_assets: int = 11,
    n_days: int = 3000,
    mom_strength: float = 0.0,
    asset_vol: float = 0.07,
    persistence: int = 189,
    seed: int = 795,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily total-return panel with a tunable cross-sectional trend.

    Each asset is a low-vol diffusion (bond-ETF-like, ~7% annual vol) sharing a small
    common factor. When ``mom_strength > 0`` a *persistent* latent trend is added to each
    series: every asset carries a slowly varying sign-persistent trend (AR(1) with
    ~``persistence``-day memory), and the drift contribution is ``mom_strength``
    (annualised) times that latent trend. Because the trend persists across the ~6-month
    formation and 1-month holding windows, an asset that has recently out-returned its
    peers tends to keep doing so — exactly the rank persistence a cross-sectional
    winners-minus-losers book harvests. ``mom_strength = 0`` is the null: pure random
    walks, no rank persistence, so the momentum book earns nothing.

    Returns ``(prices, truth)`` with ``prices`` a (date x ticker) total-return frame. The
    business-day index spans a few years (never a huge nanosecond span), safely under the
    pandas ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n_days)
    tickers = [f"B{j:02d}" for j in range(n_assets)]

    # Mostly idiosyncratic plus a small common "rates" shock (bonds co-move).
    common = rng.normal(0.0, 0.045 / np.sqrt(TRADING_DAYS), size=n_days)
    idio = rng.normal(0.0, asset_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_assets))

    # Latent persistent trend: an AR(1)-like sign-persistent process per asset.
    phi = np.exp(-1.0 / max(persistence, 1))
    shock_sd = np.sqrt(1.0 - phi**2)
    trend = np.zeros((n_days, n_assets))
    trend[0] = rng.standard_normal(n_assets)
    innov = rng.standard_normal((n_days, n_assets))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + shock_sd * innov[t]

    drift = mom_strength / TRADING_DAYS * trend      # annualised drift -> daily
    rets = common[:, None] + idio + drift

    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"
    return prices, WorldTruth(mom_strength)


# ---------------------------------------------------------------------------
# Real panel — yfinance, study-local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel total-return parquet."""
    return os.path.join(cache_dir, "bondmom_prices.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(cache_path(cache_dir))


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = START,
    end: str = "2026-07-01",
    retries: int = 3,
    min_coverage: float = 0.60,
) -> pd.DataFrame:
    """Daily total-return (auto-adjusted close) prices for the bond-ETF basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame (CI/offline path -> notebooks fall back to frozen numbers). Network
    is touched only on ``fetch=True`` (retried against yfinance flakiness), then the result
    is cached as adjusted close (``auto_adjust=True``, dividends/coupons folded in). The
    caller computes returns via ``prices.pct_change()``.

    Survivorship: this is the current-membership basket projected backwards — a
    hindsight-chosen vehicle list, named on the Signal axis. Names with < ``min_coverage``
    non-null history over the window are dropped (some ETFs listed after 2007).
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
                list(UNIVERSE), start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            coverage = raw.notna().mean()
            raw = raw.loc[:, coverage >= min_coverage]
            raw = raw.reindex(columns=[c for c in UNIVERSE if c in raw.columns])
            raw = raw.ffill().dropna(how="all")
            raw.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def load_real(cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached real panel, sliced to ``<= asof`` (empty frame on a cache miss)."""
    df = fetch_panel(cache_dir=cache_dir, fetch=False)
    if df.empty:
        return df
    return df[df.index <= pd.Timestamp(asof)].copy()


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
