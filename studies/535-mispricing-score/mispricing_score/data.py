"""Data layer for Study 535 (Mispricing-Score).

One schema -- a (date x ticker) daily adjusted-close price frame plus SPY as a
context series. The composite mispricing score is built from PRICE-COMPUTABLE
anomaly signals (momentum, short-term reversal, low-volatility, lottery/MAX,
52-week-high distance, and a price-trend "asset-growth" proxy), so the panel is
reproducible from yfinance daily bars alone -- no flaky per-name fundamentals.

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable
  ``mispricing_premium`` knob controls how strongly the planted "overpriced"
  names underperform the planted "underpriced" names. ``mispricing_premium = 0``
  is the null hypothesis. Tests never touch the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's
  own ``_cache/`` dir (gitignored). Returns ``(prices, spy)`` or empty frames
  if the cache is absent (e.g., on CI).

No look-ahead: every anomaly signal is computed on a trailing window and the
portfolio enters the close ONE day after the signal is public (see strategy).

The panel is **survivorship-biased** (current S&P 500 membership projected
backwards): positive results should be read as **upper-bound** estimates. The
short (overpriced) leg -- where SYY say the edge lives -- is exactly the leg
most damaged by survivorship, since the worst overpriced names delist.

Universe: ~45 large-cap S&P 500 names (a stable subset), plus SPY for context.
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

# Fixed large-cap survivor universe (stable subset of the S&P 500).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "LLY", "AVGO", "XOM",
    "UNH", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX", "BAC",
    "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD", "CSCO",
    "ABT", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC", "IBM", "GE",
    "CAT", "HON", "UPS", "LOW", "INTU",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    mispricing_premium: float  # how much "underpriced" names beat "overpriced" ones

    @property
    def has_premium(self) -> bool:
        return self.mispricing_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 45,
    n_days: int = 3000,
    mispricing_premium: float = 0.06,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    seed: int = 535,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily returns panel with a tunable mispricing premium.

    The plant is **state-dependent and forward-looking**, exactly as in the SYY
    story: a name is "overpriced" *because* its observable glamour fingerprint
    (high realised volatility) is extreme, and such names then UNDERPERFORM next
    month. Concretely, each name has a fixed glamour loading ``glam_i`` in
    [0, 1]; high-glamour names carry higher idiosyncratic volatility (so the
    volatility / MAX / dispersion signals flag them) and a NEGATIVE forward
    alpha proportional to ``glam_i``:

        r_i,t = beta_i * r_mkt,t - mispricing_premium * (glam_i - 0.5) / 252 + eps_i,t,
        Var(eps_i) increasing in glam_i.

    ``mispricing_premium = 0`` is the null -- the signals are then pure noise
    w.r.t. future returns. Because the underperformance is wired to the SAME
    glamour that drives the high-vol signal, the composite score (which the
    volatility leg dominates) detects it. Returns are levels (start 100).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    true_betas = rng.lognormal(mean=0.0, sigma=0.30, size=n_stocks)
    # Hidden glamour loading: 1 = most overpriced (high vol + negative alpha).
    glam = rng.uniform(0.0, 1.0, size=n_stocks)
    overpriced = 2.0 * glam - 1.0  # in [-1, 1] for the WorldTruth contract
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")

    # Idio vol is much HIGHER for high-glamour names -- glamour/lottery names are
    # more volatile. This makes the volatility & MAX signals the dominant, valid
    # detectors of the planted effect.
    name_idio = idio_vol * (0.6 + 1.4 * glam)
    idio = rng.normal(0.0, 1.0, size=(n_days, n_stocks)) * (name_idio[None, :] / np.sqrt(252))

    # Forward alpha: high-glamour names UNDERPERFORM. Centred so the null
    # (premium 0) plants no systematic L-S.
    alphas = -mispricing_premium * (glam - 0.5) / 252.0
    _ = overpriced

    stock_rets = mkt_daily[:, None] * true_betas[None, :] + alphas[None, :] + idio
    prices = (1.0 + pd.DataFrame(stock_rets, index=dates, columns=tickers)).cumprod() * 100.0
    spy_px = (1.0 + mkt).cumprod() * 100.0
    spy_px.name = "SPY"
    return prices, spy_px, WorldTruth(mispricing_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap universe + SPY.

    Cache-first: reads ``<cache_dir>/msp_prices.parquet`` and
    ``<cache_dir>/msp_spy.parquet`` if present. On cache miss, fetches from
    yfinance (with retries) and writes the parquets. Returns ``(prices, spy)``
    or empty frames if both cache and network fail.
    """
    price_path = os.path.join(cache_dir, "msp_prices.parquet")
    spy_path = os.path.join(cache_dir, "msp_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze()
        spy.name = "SPY"
        return prices, spy

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            import yfinance as yf

            all_tickers = list(UNIVERSE) + ["SPY"]
            raw = yf.download(
                all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")

            spy = raw["SPY"].dropna()
            prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")

            coverage = prices.notna().mean()
            prices = prices.loc[:, coverage >= 0.50]

            os.makedirs(cache_dir, exist_ok=True)
            prices.to_parquet(price_path)
            spy.to_frame("SPY").to_parquet(spy_path)
            return prices, spy
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))

    _ = last_err
    return pd.DataFrame(), pd.Series(dtype=float, name="SPY")


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
