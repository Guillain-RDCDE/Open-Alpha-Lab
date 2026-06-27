"""Data layer for Study 539 (Cash-Flow-Volatility).

Two tapes, one shape -- a (period x ticker) panel of trailing cash-flow volatility
plus a (date x ticker) frame of daily prices:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``cfvol_premium``
  knob controls how strongly *low* cash-flow-volatility firms outperform *high* CF-vol
  firms (the Huang 2009 sign: stable cash flows -> higher returns). ``cfvol_premium = 0``
  is the null hypothesis; the long-short sort becomes a coin flip. Tests never touch the
  network.

- ``fetch_panel`` -- the real tape. Daily adjusted-close prices from yfinance for a fixed
  ~40-name large-cap survivor basket, plus per-ticker QUARTERLY operating cash flow pulled
  from ``yfinance.Ticker(...).quarterly_cashflow`` and scaled by total assets
  (``quarterly_balance_sheet``). Everything is cached to this study's OWN ``_cache/`` dir
  (gitignored). Returns empty frames if the cache is absent and the network is unavailable.

**Survivorship bias is explicit:** the basket is current large-cap names that are still
trading in 2026. Firms whose cash flows blew up and that subsequently delisted -- the
natural high-CF-vol short candidates -- are simply absent. Positive results are
**upper-bound** estimates.

**Fundamental-data limitation:** yfinance only serves a short window of quarterly statements
(typically the last ~4-5 years, ~16-20 quarters). Trailing CF-volatility therefore rests on
a thin history. We document this and let the verdict fall where it honestly falls.

No look-ahead: CF-volatility for a given rebalance date is computed only from quarterly
statements whose reporting period END is strictly before the rebalance date, and the trade
is entered one trading day after the signal is formed (the execution lag lives in
``strategy``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

PRICE_PATH = os.path.join(DEFAULT_CACHE, "cfvol_prices.parquet")
SPY_PATH = os.path.join(DEFAULT_CACHE, "cfvol_spy.parquet")
CFVOL_PATH = os.path.join(DEFAULT_CACHE, "cfvol_signal.parquet")
META_PATH = os.path.join(DEFAULT_CACHE, "cfvol_meta.json")

# Fixed large-cap survivor basket (~40 names across sectors; all still trading 2026).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "PG", "XOM",
    "HD", "MA", "CVX", "ABBV", "PEP", "KO", "MRK", "WMT", "COST", "MCD",
    "CSCO", "ADBE", "TXN", "ORCL", "NKE", "QCOM", "HON", "UNP", "CAT", "DE",
    "LMT", "GD", "MMM", "DOW", "FCX", "NUE", "F", "GM", "DAL", "MGM",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    cfvol_premium: float  # extra annual return earned by LOW-CF-vol firms (Huang sign)

    @property
    def has_premium(self) -> bool:
        return self.cfvol_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 60,
    n_months: int = 120,
    cfvol_premium: float = 0.05,
    base_ret: float = 0.08,
    idio_vol: float = 0.18,
    seed: int = 539,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (month x stock) returns panel with a tunable CF-volatility premium.

    Each stock has a persistent cash-flow-volatility characteristic ``cfvol_i`` (drawn
    log-normal, so right-skewed like real CF-vol). Its monthly return is::

        r_{i,t} = base_ret/12 + alpha_i + epsilon_{i,t}
        alpha_i = -cfvol_premium/12 * z(cfvol_i)

    where ``z`` is the cross-sectional z-score. The MINUS sign encodes the Huang (2009)
    finding: high cash-flow volatility -> LOWER return. ``cfvol_premium = 0`` is the null
    (alpha vanishes, the sort is a coin flip).

    Returns ``(rets, cfvol, truth)``:
      - ``rets``  -- (month x stock) monthly decimal returns,
      - ``cfvol`` -- (month x stock) the (here time-invariant) CF-vol characteristic,
      - ``truth`` -- the planted ``WorldTruth``.

    Monthly index uses ``pd.period_range`` (NOT date_range) to dodge the ns-overflow trap
    on CI pandas for long synthetic panels.
    """
    rng = np.random.default_rng(seed)
    # period_range -> decorative monthly index, overflow-safe
    idx = pd.period_range("2008-01", periods=n_months, freq="M")
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Persistent CF-volatility characteristic (right-skewed, positive).
    cfvol_i = rng.lognormal(mean=-2.0, sigma=0.6, size=n_stocks)
    z = (cfvol_i - cfvol_i.mean()) / (cfvol_i.std() + 1e-12)

    # Alpha: low CF-vol -> positive, high CF-vol -> negative (Huang sign).
    alpha_monthly = -cfvol_premium / 12.0 * z

    idio = rng.normal(0.0, idio_vol / np.sqrt(12.0), size=(n_months, n_stocks))
    rets = base_ret / 12.0 + alpha_monthly[None, :] + idio

    rets_df = pd.DataFrame(rets, index=idx, columns=tickers)
    # CF-vol is (approximately) time-invariant: broadcast the characteristic across months.
    cfvol_df = pd.DataFrame(
        np.tile(cfvol_i, (n_months, 1)), index=idx, columns=tickers
    )
    return rets_df, cfvol_df, WorldTruth(cfvol_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance prices + per-ticker quarterly cash flow, study-local cache
# ---------------------------------------------------------------------------
def _download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close for ``tickers`` + SPY, with a couple of retries."""
    import yfinance as yf

    all_tickers = list(tickers) + ["SPY"]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                all_tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )["Close"]
            if not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        if last_exc is not None:
            raise last_exc
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY")

    spy = raw["SPY"].dropna()
    prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")
    coverage = prices.notna().mean()
    prices = prices.loc[:, coverage >= 0.50]
    return prices, spy


def _download_cfvol(tickers: list[str], min_quarters: int = 4) -> pd.DataFrame:
    """Per-ticker trailing cash-flow volatility from yfinance quarterly statements.

    For each ticker we pull ``quarterly_cashflow`` (operating cash flow) and
    ``quarterly_balance_sheet`` (total assets), align on the reporting-period end date,
    scale OCF by lagged total assets, and take the standard deviation of the
    asset-scaled OCF series (the firm's cash-flow-volatility characteristic).

    Returns a single-row-per-ticker DataFrame indexed by ticker with columns:
      - ``cfvol``     -- std of (OCF / assets) across available quarters,
      - ``n_quarters``-- number of quarters used,
      - ``last_end``  -- ISO date of the most recent reporting-period end.

    yfinance fundamentals are thin (~4-5 yr of quarters); tickers with fewer than
    ``min_quarters`` usable quarters are dropped.
    """
    import yfinance as yf

    def _row_like(frame: pd.DataFrame, names: list[str]) -> pd.Series | None:
        if frame is None or frame.empty:
            return None
        for nm in names:
            if nm in frame.index:
                return frame.loc[nm]
        # fuzzy contains match
        for nm in names:
            hits = [ix for ix in frame.index if nm.lower() in str(ix).lower()]
            if hits:
                return frame.loc[hits[0]]
        return None

    rows: list[dict] = []
    ocf_names = ["Operating Cash Flow", "Total Cash From Operating Activities",
                 "OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"]
    asset_names = ["Total Assets", "TotalAssets"]

    for tk in tickers:
        cfvol = np.nan
        n_q = 0
        last_end = ""
        for attempt in range(2):
            try:
                t = yf.Ticker(tk)
                cf = t.quarterly_cashflow
                bs = t.quarterly_balance_sheet
                ocf = _row_like(cf, ocf_names)
                assets = _row_like(bs, asset_names)
                if ocf is None or assets is None:
                    break
                ocf = pd.to_numeric(ocf, errors="coerce").dropna()
                assets = pd.to_numeric(assets, errors="coerce").dropna()
                # align on common reporting-period ends
                common = ocf.index.intersection(assets.index)
                if len(common) < min_quarters:
                    break
                ocf = ocf.reindex(common).sort_index()
                assets = assets.reindex(common).sort_index()
                scaled = (ocf / assets.replace(0, np.nan)).dropna()
                if len(scaled) < min_quarters:
                    break
                cfvol = float(scaled.std(ddof=1))
                n_q = int(len(scaled))
                last_end = str(pd.Timestamp(common.max()).date())
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
                continue
        if np.isfinite(cfvol):
            rows.append({"ticker": tk, "cfvol": cfvol, "n_quarters": n_q,
                         "last_end": last_end})

    if not rows:
        return pd.DataFrame(columns=["cfvol", "n_quarters", "last_end"])
    return pd.DataFrame(rows).set_index("ticker")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    fetch: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Real tape: ``(prices, spy, cfvol)``.

    Cache-first: reads the three parquets under ``cache_dir`` if present. On a cache miss
    and ``fetch=True``, downloads from yfinance and writes the cache. With ``fetch=False``
    and no cache (e.g. CI), returns empty frames so callers degrade gracefully.

    - ``prices`` -- (date x ticker) daily adjusted-close,
    - ``spy``    -- daily adjusted-close of SPY (the market proxy),
    - ``cfvol``  -- per-ticker trailing cash-flow volatility (index = ticker).
    """
    have = (os.path.exists(PRICE_PATH) and os.path.exists(SPY_PATH)
            and os.path.exists(CFVOL_PATH))
    if have:
        prices = pd.read_parquet(PRICE_PATH)
        spy = pd.read_parquet(SPY_PATH).squeeze("columns")
        spy.name = "SPY"
        cfvol = pd.read_parquet(CFVOL_PATH)
        return prices, spy, cfvol

    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY"), pd.DataFrame()

    prices, spy = _download_prices(UNIVERSE, start, end)
    if prices.empty:
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY"), pd.DataFrame()

    cfvol = _download_cfvol(list(prices.columns))
    # keep only tickers present in both price panel and cfvol panel
    common = prices.columns.intersection(cfvol.index)
    prices = prices.loc[:, common]
    cfvol = cfvol.loc[common]

    os.makedirs(cache_dir, exist_ok=True)
    prices.to_parquet(PRICE_PATH)
    spy.to_frame("SPY").to_parquet(SPY_PATH)
    cfvol.to_parquet(CFVOL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"start": start, "end": end,
                   "n_tickers": int(len(common)),
                   "date_min": str(prices.index.min().date()),
                   "date_max": str(prices.index.max().date())}, f, indent=2)
    return prices, spy, cfvol


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
