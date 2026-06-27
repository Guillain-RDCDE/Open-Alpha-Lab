"""Data layer for Study 512 (High-Volume-Return-Premium).

The Gervais-Kaniel-Mingelgrin (2001) "high-volume return premium" needs daily *volume*, not
just price -- so this layer ships a (date x ticker) panel of daily simple **returns** plus a
matching (date x ticker) panel of daily **share volume**. Two tapes, one schema:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A single knob, ``hv_premium``,
  controls how strongly a high-volume *spike* this week predicts a positive return next week
  (and a low-volume week predicts a slightly negative one). ``hv_premium = 0`` is the null --
  volume is pure noise uncorrelated with the next-period return, so the long-short is a coin.
  This is the study's positive control and its null in one bottle. Tests never touch the
  network. Decorative indices use ``pd.bdate_range`` over a few thousand business days only --
  far under the ns-Timestamp overflow wall.

- ``fetch_panel`` -- real daily OHLCV from yfinance (auto-adjusted close for returns, raw
  ``Volume`` for the signal), cached to this study's own ``_cache/`` (gitignored). Returns
  ``(returns, volume)`` or empty frames if both cache and network fail. Retries guard
  yfinance flakiness.

Universe: a fixed ~40-name large-cap survivor basket (the same flavour as Studies 238/330).
It is **survivorship-biased** -- current mega/large-caps projected backwards; failed or
delisted names (a natural part of any volume-spike sort) are absent. Positive results are
**upper-bound** estimates. Named explicitly on the SIGNAL axis.

No look-ahead is baked in here: the signal is formed on a fully-closed week and the trade is
entered one bar later (see ``strategy.py`` for the single documented execution lag).
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

# Fixed large-cap survivor basket (stable subset of mega/large-cap US names).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "AVGO", "JPM", "LLY", "XOM",
    "UNH", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX",
    "BAC", "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD",
    "CSCO", "ABT", "TXN", "ADBE", "INTC", "IBM", "CAT", "HON", "GS", "QCOM",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

TRADING_DAYS = 252
WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    hv_premium: float  # how strongly a high-volume week predicts a positive next-week return

    @property
    def has_premium(self) -> bool:
        return self.hv_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    n_days: int = 2600,
    hv_premium: float = 0.04,
    market_vol: float = 0.16,
    idio_vol: float = 0.24,
    seed: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible daily (returns, volume) panel with a tunable high-volume premium.

    Each stock has a market-model daily return ``r_i = beta_i * r_mkt + epsilon_i`` plus, when
    ``hv_premium != 0``, a planted weekly tilt: in any week where the stock's average daily
    volume exceeded its own trailing average (a "high-volume" week), the *following* week's
    idiosyncratic return is nudged up by ``hv_premium`` (annualised); a low-volume week is
    nudged down by the same amount. ``hv_premium = 0`` is the null -- volume is independent
    log-normal noise with zero predictive content, so the long-short is a coin.

    Returns ``(returns, volume, truth)`` -- daily simple returns and daily share volume,
    both (date x ticker). The decorative index uses ``pd.bdate_range`` (business days), kept
    well under any ns-Timestamp overflow.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"S{j:02d}" for j in range(n_stocks)]

    betas = rng.lognormal(mean=0.0, sigma=0.30, size=n_stocks)
    mkt = rng.normal(0.08 / TRADING_DAYS, market_vol / np.sqrt(TRADING_DAYS), size=n_days)
    idio = rng.normal(0.0, idio_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_stocks))

    # Base daily log-volume: a per-name level plus AR(1)-ish daily noise.
    base_level = rng.uniform(15.5, 17.5, size=n_stocks)              # ~ log(shares)
    vol_noise = rng.normal(0.0, 0.45, size=(n_days, n_stocks))
    log_vol = base_level[None, :] + np.cumsum(0.0 * vol_noise, axis=0) + vol_noise
    # Inject occasional genuine volume spikes (event days) so weeks differ.
    spikes = (rng.random((n_days, n_stocks)) < 0.04) * rng.uniform(0.6, 1.6, (n_days, n_stocks))
    log_vol = log_vol + spikes
    volume = np.exp(log_vol)

    rets = mkt[:, None] * betas[None, :] + idio

    if hv_premium != 0.0:
        # Weekly abnormal-volume label, then nudge the FOLLOWING week's returns.
        vdf = pd.DataFrame(volume, index=dates, columns=tickers)
        rdf = pd.DataFrame(rets, index=dates, columns=tickers)
        # Weekly average daily volume per name.
        wk_vol = vdf.resample("W-FRI").mean()
        trail = wk_vol.rolling(8, min_periods=4).mean().shift(1)
        abn = wk_vol / trail - 1.0
        hi = abn.rank(axis=1, pct=True) >= 0.8       # top quintile = high volume
        lo = abn.rank(axis=1, pct=True) <= 0.2       # bottom quintile = low volume
        nudge_wk = (hi.astype(float) - lo.astype(float)) * (hv_premium / WEEKS_PER_YEAR)
        # Apply each week's nudge to the NEXT week's daily returns (forward-looking premium).
        nudge_wk = nudge_wk.shift(1)
        # Spread the weekly nudge across that week's ~5 trading days.
        for wk_end, row in nudge_wk.iterrows():
            if row.isna().all():
                continue
            mask = (rdf.index > (wk_end - pd.Timedelta(days=7))) & (rdf.index <= wk_end)
            daily = row.fillna(0.0).to_numpy() / 5.0
            rdf.loc[mask] = rdf.loc[mask].to_numpy() + daily[None, :]
        rets = rdf.to_numpy()

    returns = pd.DataFrame(rets, index=dates, columns=tickers)
    vol_df = pd.DataFrame(volume, index=dates, columns=tickers)
    returns.index.name = "date"
    vol_df.index.name = "date"
    return returns, vol_df, WorldTruth(hv_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance daily OHLCV, study-local cache
# ---------------------------------------------------------------------------
def _cache_paths(cache_dir: str) -> tuple[str, str]:
    return (
        os.path.join(cache_dir, "hv_returns.parquet"),
        os.path.join(cache_dir, "hv_volume.parquet"),
    )


def _existing_cache() -> str | None:
    """The first cache dir that holds both parquets (study-local first, then shared)."""
    for c in (DEFAULT_CACHE, SHARED_CACHE):
        rp, vp = _cache_paths(c)
        if os.path.exists(rp) and os.path.exists(vp):
            return c
    return None


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2014-01-01",
    end: str = "2025-12-31",
    fetch: bool = False,
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily simple returns and daily share volume for the large-cap survivor basket.

    Cache-first: if both parquets exist (study-local, else shared repo cache) they are read
    and returned -- the reproducible core and any CI run never touch the network. With
    ``fetch=False`` and no cache, returns ``(empty, empty)`` so the notebook can banner
    "synthetic tape" and fall back. Network is touched only on an explicit ``fetch=True``;
    the auto-adjusted close gives returns, the raw ``Volume`` gives the signal, and the
    result is cached. yfinance flakiness is guarded with ``retries``.
    """
    existing = _existing_cache()
    if existing is not None:
        rp, vp = _cache_paths(existing)
        ret = pd.read_parquet(rp)
        vol = pd.read_parquet(vp)
        for df in (ret, vol):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        return ret, vol

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
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            close = raw["Close"]
            volume = raw["Volume"]
            close = close.dropna(how="all")
            volume = volume.reindex(close.index)
            # Drop names with thin coverage.
            cov = close.notna().mean()
            keep = cov[cov >= 0.80].index
            close = close[keep]
            volume = volume[keep]
            ret = close.pct_change().dropna(how="all")
            volume = volume.reindex(ret.index)
            ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
            volume.index = ret.index
            ret.index.name = "date"
            volume.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            rp, vp = _cache_paths(cache_dir)
            ret.to_parquet(rp)
            volume.to_parquet(vp)
            return ret, volume
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"fetch_panel: all retries failed ({last_err})")
    return pd.DataFrame(), pd.DataFrame()


def drop_partial_last_week(ret: pd.DataFrame, asof_day: int = 5) -> pd.DataFrame:
    """Drop trailing rows of an in-progress final calendar week (house rule: no partial bar).

    A row inside the current ISO week before ``asof_day`` business days have elapsed is treated
    as partial. Conservative: trims any rows sharing the last row's ISO (year, week).
    """
    if ret.empty:
        return ret
    today = pd.Timestamp.today().normalize()
    last = ret.index[-1]
    iso_last = last.isocalendar()
    iso_today = today.isocalendar()
    if iso_last.year == iso_today.year and iso_last.week == iso_today.week:
        iso = ret.index.isocalendar()
        same = (iso.year == iso_last.year) & (iso.week == iso_last.week)
        if same.sum() < asof_day:
            return ret.loc[~same.to_numpy()]
    return ret


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a frame (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
