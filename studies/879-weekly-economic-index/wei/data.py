"""Data layer for Study 879 — Weekly Economic Index (WEI + SPY / XLY / XLP).

Two sources, both offline-friendly once the caches exist.

* **Real tape.** The **Weekly Economic Index** (Lewis, Mertens & Stock, 2020) is a
  real-time activity nowcast that blends **10 weekly** series into one year-over-year
  growth number, published every week by the **Dallas Fed**. Its public workbook
  (``weekly-economic-index.xlsx``) carries the full ``2008-current`` weekly history — one
  row per week-ending Saturday with the top-line ``WEI`` (and several dated real-time
  vintages). We keep the current-vintage weekly ``WEI`` level and derive its weekly change
  ``dwei``. Cached as ``_cache/wei.csv``. The return side is SPY (broad tape) plus **XLY**
  (consumer-discretionary, the *cyclical* leg) and **XLP** (consumer-staples, the
  *defensive* leg) daily total-return closes (yfinance, ``auto_adjust=True``), cached as
  ``_cache/market.csv``.

  The **FRED** CSV endpoint (series ``WEI``) is firewalled in this build (as noted in
  siblings 385 / 877), so the nowcast history comes straight from the Dallas Fed workbook;
  FRED's ``WEI`` series is named as the documented fallback in :func:`fetch_wei`.

* **Synthetic.** A deterministic, fixed-seed generator that emits a WEI-like nowcast and
  SPY-like / rotation forward-return series with a PLANTED edge knob: when ``edge != 0`` a
  higher nowcast genuinely lifts the next-period return. ``edge = 0`` is the null and must
  NOT manufacture significance; a large planted edge MUST light the regression up. It is
  the positive control — it proves the inference is unbiased, and is **never** cited for a
  real-tape stamp.

Pure numpy + pandas + stdlib for the offline path. ``fetch_*`` (network) are used only to
build the caches and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
WEI_CACHE = os.path.join(CACHE_DIR, "wei.csv")
MARKET_CACHE = os.path.join(CACHE_DIR, "market.csv")

AS_OF = "2026-06-30"        # last complete calendar month at publication

# Dallas Fed WEI public workbook. The ``2008-current`` sheet holds the full weekly history
# (one row per week-ending Saturday) with the top-line current-vintage WEI.
WEI_XLSX_URL = (
    "https://www.dallasfed.org/-/media/documents/research/wei/weekly-economic-index.xlsx"
)
WEI_SHEET = "2008-current"
# Documented fallback if the Dallas Fed workbook is unreachable: FRED's weekly WEI series
# (the same top-line nowcast, one dated observation per week).
FRED_FALLBACK_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WEI"

MARKET_TICKERS = ("SPY", "XLY", "XLP")

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenAlphaLab-research/1.0"

__all__ = [
    "AS_OF", "CACHE_DIR", "WEI_CACHE", "MARKET_CACHE", "MARKET_TICKERS",
    "fetch_wei", "fetch_market", "have_real",
    "load_wei", "load_market", "build_real", "synthetic",
]


# --------------------------------------------------------------------------- #
# Real tape — fetchers (network; used once to build the caches)
# --------------------------------------------------------------------------- #
def _get(url: str, tries: int = 4, timeout: int = 90) -> bytes:
    """GET ``url`` with a browser User-Agent and up to ``tries`` retries."""
    import urllib.request

    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:  # pragma: no cover - network path
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to GET {url}: {last}")


def fetch_wei(path: str = WEI_CACHE) -> pd.DataFrame:
    """Download the Dallas Fed WEI workbook, extract the weekly history, cache a CSV.

    Parses the ``2008-current`` sheet (``Date``, ``WEI``) into a two-column
    ``date, wei`` CSV. Falls back to FRED's weekly ``WEI`` series (documented) if the
    Dallas Fed workbook cannot be read.
    """
    try:  # pragma: no cover - network path
        raw = _get(WEI_XLSX_URL)
        xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        df = xl.parse(WEI_SHEET, header=0)[["Date", "WEI"]].copy()
        df.columns = ["date", "wei"]
    except Exception:  # pragma: no cover - fallback path
        raw = _get(FRED_FALLBACK_URL).decode()
        fr = pd.read_csv(io.StringIO(raw))
        fr.columns = ["date", "wei"]
        df = fr

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wei"] = pd.to_numeric(df["wei"], errors="coerce")
    df = (df.dropna().drop_duplicates(subset=["date"]).sort_values("date")
          .reset_index(drop=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_market(path: str = MARKET_CACHE, start: str = "1999-01-01",
                 end: str | None = "2026-07-01") -> pd.DataFrame:
    """Download SPY / XLY / XLP daily total-return closes and cache a wide CSV (network)."""
    import yfinance as yf  # pragma: no cover - network path

    cols = {}
    for tk in MARKET_TICKERS:
        px = yf.download(tk, start=start, end=end, auto_adjust=True,
                         progress=False)["Close"]
        if hasattr(px, "columns"):
            px = px.iloc[:, 0]
        cols[tk] = px
    wide = pd.DataFrame(cols).sort_index()
    wide.index.name = "Date"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wide.to_csv(path)
    return wide


# --------------------------------------------------------------------------- #
# Real tape — offline loaders
# --------------------------------------------------------------------------- #
def have_real(wei: str = WEI_CACHE, market: str = MARKET_CACHE) -> bool:
    return os.path.exists(wei) and os.path.exists(market)


def load_wei(path: str = WEI_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached weekly WEI with the weekly change ``dwei``, sliced to ``asof``.

    Columns: ``wei`` (the top-line current-vintage nowcast level) and ``dwei`` (the
    week-over-week change of the level). Indexed by week-ending Saturday.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= pd.Timestamp(asof)].copy()
    df = df.set_index("date")
    df["dwei"] = df["wei"].diff()
    return df


def load_market(path: str = MARKET_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily SPY / XLY / XLP total-return closes, sliced to ``asof``."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[df.index <= pd.Timestamp(asof)]
    return df.astype(float)


def _forward_returns(px: pd.Series, dates: pd.DatetimeIndex, horizon: int,
                     lag: int) -> np.ndarray:
    """``horizon``-trading-day forward return anchored to each weekly WEI ``date``.

    For WEI week-ending date ``t`` we find the last trading day on or before ``t``
    (position ``p``), then measure ``close[p+lag+horizon] / close[p+lag] - 1``. The WEI for
    a week-ending Saturday is only *published* the following week, so ``lag`` is set to one
    trading week (5 days) by default — the position is taken *after* the nowcast is known,
    zero look-ahead. Vectorised via ``searchsorted`` — no per-date loop.
    """
    sidx = px.index
    close = px.to_numpy(dtype=float)
    pos = sidx.searchsorted(np.asarray(dates, dtype="datetime64[ns]"), side="right") - 1
    base = pos + lag
    tgt = pos + lag + horizon
    ok = (pos >= 0) & (tgt < len(close)) & (base < len(close)) & (base >= 0)
    out = np.full(len(dates), np.nan)
    out[ok] = close[tgt[ok]] / close[base[ok]] - 1.0
    return out


def build_real(wei_path: str = WEI_CACHE, market_path: str = MARKET_CACHE,
               asof: str = AS_OF, lag: int = 5) -> pd.DataFrame:
    """Weekly frame indexed by WEI week-ending date.

    Columns: ``wei`` (level), ``dwei`` (weekly change); the forward broad-tape returns
    ``spy_h1`` / ``spy_h4`` (1- and 4-week forward SPY, in trading days 5 / 20); and the
    forward **cyclical-minus-defensive rotation** ``rot_h1`` / ``rot_h4`` = forward XLY
    return minus forward XLP return over the same window. Every forward return uses a
    documented one-week execution lag (``lag`` trading days) so the position is taken only
    after the weekly nowcast is published — this frame carries no look-ahead by itself.
    """
    w = load_wei(wei_path, asof=asof)
    m = load_market(market_path, asof=asof)
    dates = w.index
    out = pd.DataFrame({"wei": w["wei"].to_numpy(), "dwei": w["dwei"].to_numpy()},
                       index=dates)
    spy, xly, xlp = m["SPY"], m["XLY"], m["XLP"]
    for tag, h in (("h1", 5), ("h4", 20)):
        out[f"spy_{tag}"] = _forward_returns(spy, dates, horizon=h, lag=lag)
        rot = (_forward_returns(xly, dates, horizon=h, lag=lag)
               - _forward_returns(xlp, dates, horizon=h, lag=lag))
        out[f"rot_{tag}"] = rot
    return out.dropna(subset=["dwei", "spy_h1"])


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n: int = 700, edge: float = 0.0, seed: int = 879,
              phi: float = 0.92, mu_weekly: float = 0.0015,
              sig_weekly: float = 0.018, rot_sig: float = 0.012) -> pd.DataFrame:
    """Deterministic weekly WEI + SPY / rotation forward-return frame with a PLANTED edge.

    ``wei`` is a persistent AR(1) (autocorrelation ``phi``, the WEI is very smooth)
    standardised to ~unit variance; ``dwei`` is its weekly change. SPY-like 1-week forward
    returns have drift ``mu_weekly`` and vol ``sig_weekly``; the rotation leg has vol
    ``rot_sig``. When ``edge != 0`` an *extra* return of ``edge * wei_t`` is injected into
    both the broad and rotation forward legs, so a high nowcast on week ``t`` genuinely
    predicts a higher forward SPY return *and* cyclical outperformance (the claim, planted).
    The 4-week forward return compounds four such weekly draws.

    ``edge = 0`` is the null: the nowcast carries no forward information and the regression
    must NOT manufacture significance. A large ``edge`` must light it up. The index is a
    plain weekly ``bdate``-derived label (``n`` well under the ns-timestamp horizon) —
    decorative only.
    """
    rng = np.random.default_rng(seed)
    wei = np.empty(n)
    wei[0] = rng.normal()
    innov = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        wei[t] = phi * wei[t - 1] + rng.normal(0.0, innov)
    dwei = np.concatenate([[np.nan], np.diff(wei)])

    spy1 = rng.normal(mu_weekly, sig_weekly, size=n)
    rot1 = rng.normal(0.0, rot_sig, size=n)
    if edge != 0.0:
        spy1 = spy1 + edge * wei
        rot1 = rot1 + edge * wei
    # 4-week forward returns: the anchored week plus three fresh weekly draws
    spy_extra = rng.normal(mu_weekly, sig_weekly, size=(n, 3)).sum(axis=1)
    rot_extra = rng.normal(0.0, rot_sig, size=(n, 3)).sum(axis=1)
    spy4 = spy1 + spy_extra
    rot4 = rot1 + rot_extra

    idx = pd.date_range("2008-01-05", periods=n, freq="W-SAT")  # weekly, n<<ns horizon
    out = pd.DataFrame({"wei": wei, "dwei": dwei, "spy_h1": spy1, "spy_h4": spy4,
                        "rot_h1": rot1, "rot_h4": rot4}, index=idx)
    return out.dropna(subset=["dwei"])
