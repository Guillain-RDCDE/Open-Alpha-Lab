"""Data layer for Study 520 (External-Financing-Anomaly).

The Bradshaw–Richardson–Sloan (2006) signal is built from the **cash-flow statement's
financing activities**, not from prices. Two tapes share one schema — an annual
(fiscal-year × ticker) frame of net external financing scaled by average total assets,
plus a daily price panel to measure forward returns:

- ``synthetic_world`` — a *deterministic, offline* generator. A single knob, ``xfin_penalty``,
  dials the only thing the trade can harvest: big external-finance *raisers* underperform
  *retirers* the following year. ``xfin_penalty = 0`` is the null. This is the study's positive
  control and its null in one bottle; tests never touch the network.

- ``fetch_fundamentals`` / ``fetch_prices`` — the real Yahoo tape. yfinance exposes only the
  most recent ~5 fiscal years of cash-flow / balance-sheet line items per name, so the real
  cross-section is deliberately **thin** (a handful of annual rebalances on a fixed survivor
  basket). Cache-first: the reproducible core and the notebooks never hit the network unless an
  explicit ``fetch=True`` is passed.

Two honest caveats baked into this study and stated openly:

1. **Survivorship.** The basket is names still trading in 2026; the dead raisers (the ones the
   anomaly says should have cratered) are absent. Real results are an *upper bound* on how badly
   raisers underperform.
2. **Look-ahead.** The signal is the *prior* fiscal year's financing; it is treated as public
   only after a reporting lag (``REPORT_LAG_DAYS``), and the forward return is the next 12
   months — so no fundamental is used before it could have been read.

The decorative monthly index on the synthetic tape is built with ``pd.period_range`` (never a
huge ``date_range`` periods= span), to stay clear of the ns-Timestamp overflow wall on CI pandas.
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

# A fundamental is treated as public only this many days after fiscal-year end (10-K filing lag).
REPORT_LAG_DAYS = 90

# Fixed large-cap survivor basket (~45 names, sector-spread). Names still trading in 2026 — the
# survivorship caveat the SIGNAL axis states openly. Drawn from the same pool as Studies 238/330.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "LLY", "AVGO", "XOM",
    "UNH", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX",
    "BAC", "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD",
    "CSCO", "ABT", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC", "IBM",
    "GE", "CAT", "HON", "UPS", "GS",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# Cash-flow / balance-sheet line items we read (yfinance row labels). We try several aliases per
# concept because Yahoo's labelling drifts across names and vintages.
DEBT_ISSUE_KEYS = ["Issuance Of Debt", "Long Term Debt Issuance", "Net Issuance Payments Of Debt"]
DEBT_REPAY_KEYS = ["Repayment Of Debt", "Long Term Debt Payments"]
DEBT_NET_KEYS = ["Net Issuance Payments Of Debt", "Net Long Term Debt Issuance"]
EQUITY_ISSUE_KEYS = ["Issuance Of Capital Stock", "Common Stock Issuance"]
EQUITY_REPURCHASE_KEYS = ["Repurchase Of Capital Stock", "Common Stock Payments"]
DIVIDEND_KEYS = ["Cash Dividends Paid", "Common Stock Dividend Paid"]
FINANCING_CF_KEYS = ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"]
ASSET_KEYS = ["Total Assets"]


@dataclass(frozen=True)
class WorldTruth:
    """The planted ground truth for a synthetic world."""

    xfin_penalty: float  # forward-return penalty per unit of (positive) scaled external financing

    @property
    def has_penalty(self) -> bool:
        return self.xfin_penalty != 0.0


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_stocks: int = 45,
    n_years: int = 12,
    xfin_penalty: float = 0.40,
    xfin_spread: float = 0.10,
    base_drift: float = 0.09,
    idio_vol: float = 0.12,
    seed: int = 520,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible annual panel of scaled external financing + forward returns.

    Each year, every name draws a scaled net-external-financing value ``xfin`` (centred near
    zero, std ``xfin_spread``). The next-year return is::

        r = base_drift - xfin_penalty * xfin + idio

    so big *raisers* (xfin > 0) earn *less* and *retirers* (xfin < 0) earn *more* — exactly the
    Bradshaw–Richardson–Sloan claim. ``xfin_penalty = 0`` is the null (financing tells you
    nothing about forward returns).

    Returns ``(xfin, fwd, truth)`` — two (year × ticker) DataFrames of the scaled financing
    signal and the realised forward annual return, plus the planted truth.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    # period_range -> timestamps is overflow-safe for any sane n (years, not nanoseconds).
    years = pd.period_range("2012", periods=n_years, freq="Y").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(years, name="fy_end")

    xfin = rng.normal(0.0, xfin_spread, size=(n_years, n_stocks))
    idio = rng.normal(0.0, idio_vol, size=(n_years, n_stocks))
    fwd = base_drift - xfin_penalty * xfin + idio

    xfin_df = pd.DataFrame(xfin, index=idx, columns=tickers)
    fwd_df = pd.DataFrame(fwd, index=idx, columns=tickers)
    return xfin_df, fwd_df, WorldTruth(xfin_penalty=xfin_penalty)


# ---------------------------------------------------------------------------
# Real tape — yfinance, study-local cache, retry-guarded
# ---------------------------------------------------------------------------
def _cache_file(name: str, cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or DEFAULT_CACHE, name)


def _existing(name: str) -> str | None:
    for c in (DEFAULT_CACHE, SHARED_CACHE):
        p = _cache_file(name, c)
        if os.path.exists(p):
            return p
    return None


def _first_row(frame: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    """First matching row (by Yahoo label alias) from a cash-flow / balance-sheet frame."""
    for k in keys:
        if k in frame.index:
            row = frame.loc[k]
            if isinstance(row, pd.DataFrame):  # duplicate labels
                row = row.iloc[0]
            return row.astype(float)
    return None


def fetch_fundamentals(
    cache_dir: str | None = None,
    fetch: bool = False,
    tickers: list[str] | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Per-name annual financing line items + total assets, long-format, cache-first.

    Returns a tidy DataFrame indexed by ``(ticker, fy_end)`` with columns for net debt issuance,
    net equity issuance, dividends, total financing cash flow and total assets (all in raw
    currency). Cache-only by default; on an explicit ``fetch=True`` it pulls each name's
    ``.cashflow`` and ``.balance_sheet`` from yfinance with retries and caches the result.
    """
    cached = _existing("xfin_fundamentals.parquet")
    if cached is not None:
        return pd.read_parquet(cached)
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    names = tickers or UNIVERSE
    rows: list[dict] = []
    for tk in names:
        cf = bs = None
        for attempt in range(max_retries):
            try:
                t = yf.Ticker(tk)
                cf = t.cashflow
                bs = t.balance_sheet
                if cf is not None and not cf.empty:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.0 + attempt)
        if cf is None or cf.empty or bs is None or bs.empty:
            continue

        debt_issue = _first_row(cf, DEBT_ISSUE_KEYS)
        debt_repay = _first_row(cf, DEBT_REPAY_KEYS)
        debt_net = _first_row(cf, DEBT_NET_KEYS)
        eq_issue = _first_row(cf, EQUITY_ISSUE_KEYS)
        eq_repurchase = _first_row(cf, EQUITY_REPURCHASE_KEYS)
        div = _first_row(cf, DIVIDEND_KEYS)
        fin_cf = _first_row(cf, FINANCING_CF_KEYS)
        assets = _first_row(bs, ASSET_KEYS)
        if assets is None:
            continue

        for col in cf.columns:
            fy = pd.Timestamp(col)

            def g(s: pd.Series | None) -> float:
                if s is None or col not in s.index:
                    return np.nan
                return float(s[col])

            rows.append(
                {
                    "ticker": tk,
                    "fy_end": fy,
                    "debt_issue": g(debt_issue),
                    "debt_repay": g(debt_repay),
                    "debt_net": g(debt_net),
                    "equity_issue": g(eq_issue),
                    "equity_repurchase": g(eq_repurchase),
                    "dividends": g(div),
                    "financing_cf": g(fin_cf),
                    "total_assets": g(assets) if assets is not None and fy in assets.index else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index(["ticker", "fy_end"]).sort_index()
    os.makedirs(cache_dir or DEFAULT_CACHE, exist_ok=True)
    out.to_parquet(_cache_file("xfin_fundamentals.parquet", cache_dir))
    return out


def fetch_prices(
    cache_dir: str | None = None,
    fetch: bool = False,
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2026-06-26",
) -> pd.DataFrame:
    """Daily adjusted-close prices for the basket, cache-first.

    Cache-only by default; on ``fetch=True`` pulls from yfinance (auto-adjusted) and caches.
    """
    cached = _existing("xfin_prices.parquet")
    if cached is not None:
        px = pd.read_parquet(cached)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    names = tickers or UNIVERSE
    raw = yf.download(names, start=start, end=end, auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir or DEFAULT_CACHE, exist_ok=True)
    raw.to_parquet(_cache_file("xfin_prices.parquet", cache_dir))
    return raw


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a frame, for the as-of stamp in docs/results.md."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
