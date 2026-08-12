"""Data layer for Study 877 — GDPNow Revisions (Atlanta Fed nowcast + SPY).

Two sources, both offline-friendly once the caches exist.

* **Real tape.** The Atlanta Fed **GDPNow** model publishes a *real-time* nowcast of the
  current quarter's real-GDP growth and **revises it almost daily** as each new data release
  arrives. Its public workbook (``GDPTrackingModelDataAndForecasts.xlsx``) carries a
  long-format history of *every* dated forecast in the ``TrackingDeepArchives`` (2011–2014)
  and ``TrackingArchives`` (2014–today) sheets — one row per **forecast date** with the
  top-line ``GDP Nowcast`` and the ``Quarter being forecasted``. We keep the top-line nowcast
  and, **within each quarter**, take the day-over-day **revision** (the change in the
  current-quarter nowcast). That revision is the real-time growth surprise under test. Cached
  as ``_cache/gdpnow.csv``. The return side is SPY daily total-return closes (yfinance,
  ``auto_adjust=True``), cached as ``_cache/spy.csv``.

  The FRED CSV endpoint is firewalled in this build (as noted in siblings 385 / 268), so the
  nowcast history comes straight from the Atlanta Fed workbook; FRED's ``GDPNOW`` series is
  named as the documented fallback in :func:`fetch_nowcast`.

* **Synthetic.** A deterministic, fixed-seed generator that emits a revision series and a
  SPY-like forward-return series with a PLANTED edge knob: when ``edge != 0`` an upward
  revision genuinely lifts the next-period return. ``edge = 0`` is the null and must NOT
  manufacture significance; a large planted edge MUST light the regression up. It is the
  positive control — it proves the inference is unbiased, and is **never** cited for a
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
NOWCAST_CACHE = os.path.join(CACHE_DIR, "gdpnow.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")

AS_OF = "2026-06-30"        # last complete calendar month at publication

# Atlanta Fed GDPNow public workbook. The two long-format archive sheets hold the full
# daily forecast history; ``TrackingHistory`` holds only the current quarter's evolution.
GDPNOW_XLSX_URL = (
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
    "cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx"
)
ARCHIVE_SHEETS = ("TrackingDeepArchives", "TrackingArchives")
# Documented fallback if the Atlanta Fed workbook is unreachable: FRED's daily GDPNow series
# (the same top-line nowcast, one dated observation per update).
FRED_FALLBACK_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPNOW"

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenAlphaLab-research/1.0"

__all__ = [
    "AS_OF", "CACHE_DIR", "NOWCAST_CACHE", "SPY_CACHE",
    "fetch_nowcast", "fetch_spy", "have_real",
    "load_nowcast", "load_spy", "build_real", "synthetic",
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


def fetch_nowcast(path: str = NOWCAST_CACHE) -> pd.DataFrame:
    """Download the GDPNow workbook, extract the top-line nowcast history, cache a CSV.

    Parses the ``TrackingDeepArchives`` + ``TrackingArchives`` sheets (long format: one row
    per forecast date) into ``date, qtr, nowcast``. Falls back to FRED's daily ``GDPNOW``
    series (documented) if the Atlanta Fed workbook cannot be read.
    """
    try:  # pragma: no cover - network path
        raw = _get(GDPNOW_XLSX_URL)
        xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        frames = []
        for sh in ARCHIVE_SHEETS:
            df = xl.parse(sh, header=0)[
                ["Forecast Date", "Quarter being forecasted", "GDP Nowcast"]
            ].copy()
            df.columns = ["date", "qtr", "nowcast"]
            frames.append(df)
        g = pd.concat(frames, ignore_index=True)
    except Exception:  # pragma: no cover - fallback path
        raw = _get(FRED_FALLBACK_URL).decode()
        fr = pd.read_csv(io.StringIO(raw))
        fr.columns = ["date", "nowcast"]
        fr["nowcast"] = pd.to_numeric(fr["nowcast"], errors="coerce")
        fr["date"] = pd.to_datetime(fr["date"], errors="coerce")
        # FRED gives no quarter label; stamp each date with its own calendar quarter.
        fr["qtr"] = fr["date"] + pd.offsets.QuarterEnd(0)
        g = fr[["date", "qtr", "nowcast"]]

    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g["qtr"] = pd.to_datetime(g["qtr"], errors="coerce")
    g["nowcast"] = pd.to_numeric(g["nowcast"], errors="coerce")
    g = (g.dropna().drop_duplicates(subset=["date"]).sort_values("date")
         .reset_index(drop=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.to_csv(path, index=False)
    return g


def fetch_spy(path: str = SPY_CACHE, start: str = "2011-01-01",
              end: str | None = "2026-07-01") -> pd.Series:
    """Download SPY daily total-return closes and cache them (network; once)."""
    import yfinance as yf  # pragma: no cover - network path

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    if hasattr(spy, "columns"):
        spy = spy.iloc[:, 0]
    spy.name = "SPY"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spy.sort_index().to_frame().to_csv(path)
    return spy


# --------------------------------------------------------------------------- #
# Real tape — offline loaders
# --------------------------------------------------------------------------- #
def have_real(nowcast: str = NOWCAST_CACHE, spy: str = SPY_CACHE) -> bool:
    return os.path.exists(nowcast) and os.path.exists(spy)


def load_nowcast(path: str = NOWCAST_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached GDPNow history with the within-quarter revision, sliced to ``asof``.

    Columns: ``qtr`` (quarter being forecasted), ``nowcast`` (top-line real-GDP nowcast,
    annualised %), ``rev`` (the day-over-day change of the nowcast *within* the quarter —
    the first forecast of each quarter has no revision, so ``rev`` is NaN there). Indexed by
    forecast date.
    """
    g = pd.read_csv(path, parse_dates=["date", "qtr"]).sort_values("date")
    g = g[g["date"] <= pd.Timestamp(asof)].copy()
    g["rev"] = g.groupby("qtr")["nowcast"].diff()
    return g.set_index("date")


def load_spy(path: str = SPY_CACHE, asof: str = AS_OF) -> pd.Series:
    """Cached SPY daily total-return close as a Series named 'spy', sliced to ``asof``."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "spy"
    return s


def _forward_returns(spy: pd.Series, dates: pd.DatetimeIndex, horizon: int,
                     lag: int) -> np.ndarray:
    """``horizon``-trading-day forward SPY return anchored to each forecast ``date``.

    For forecast date ``t`` we find the last SPY trading day on or before ``t`` (position
    ``p``), then measure ``close[p+lag+horizon] / close[p+lag] - 1``. ``lag=0`` acts at the
    close of the release day (the nowcast is posted intraday); ``lag=1`` is the stricter
    next-day execution. Vectorised via ``searchsorted`` — no per-date loop.
    """
    sidx = spy.index
    close = spy.to_numpy(dtype=float)
    pos = sidx.searchsorted(np.asarray(dates, dtype="datetime64[ns]"), side="right") - 1
    base = pos + lag
    tgt = pos + lag + horizon
    ok = (pos >= 0) & (tgt < len(close)) & (base < len(close)) & (base >= 0)
    out = np.full(len(dates), np.nan)
    out[ok] = close[tgt[ok]] / close[base[ok]] - 1.0
    return out


def build_real(nowcast_path: str = NOWCAST_CACHE, spy_path: str = SPY_CACHE,
               asof: str = AS_OF) -> pd.DataFrame:
    """Frame indexed by forecast date: ``nowcast``, ``rev`` and the aligned forward SPY
    returns ``fwd1`` / ``fwd5`` (lag 0, the headline) and ``fwd1_l1`` / ``fwd5_l1`` (the
    stricter next-day lag). Rows without a revision (the first forecast of a quarter) are
    dropped so the regression sees only genuine revisions; the strategy layer carries the
    documented execution lag, so this frame has no look-ahead by itself.
    """
    g = load_nowcast(nowcast_path, asof=asof)
    spy = load_spy(spy_path, asof=asof)
    dates = g.index
    out = pd.DataFrame(
        {"nowcast": g["nowcast"].to_numpy(), "rev": g["rev"].to_numpy()}, index=dates
    )
    out["fwd1"] = _forward_returns(spy, dates, horizon=1, lag=0)
    out["fwd5"] = _forward_returns(spy, dates, horizon=5, lag=0)
    out["fwd1_l1"] = _forward_returns(spy, dates, horizon=1, lag=1)
    out["fwd5_l1"] = _forward_returns(spy, dates, horizon=5, lag=1)
    return out.dropna(subset=["rev", "fwd1"])


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n: int = 2000, edge: float = 0.0, seed: int = 877,
              rev_sd: float = 0.35, mu_daily: float = 0.0003,
              sig_daily: float = 0.010) -> pd.DataFrame:
    """Deterministic revision + SPY-like forward-return frame with a PLANTED edge.

    ``rev`` is a mean-zero nowcast-revision series (roughly the dispersion of the real one,
    ~0.35pp). SPY-like next-day returns have drift ``mu_daily`` and vol ``sig_daily``; when
    ``edge != 0`` an *extra* return of ``edge * rev_t`` is injected, so an upward revision on
    day ``t`` genuinely predicts a higher day-``t`` forward return (the claim, planted). The
    5-day forward return sums five such daily draws.

    ``edge = 0`` is the null: the revision carries no forward information and the regression
    must NOT manufacture significance. A large ``edge`` must light it up. The index is a plain
    business-day label (``n`` well under the ns-timestamp horizon) — decorative only.
    """
    rng = np.random.default_rng(seed)
    rev = rng.normal(0.0, rev_sd, size=n)
    r1 = rng.normal(mu_daily, sig_daily, size=n)
    if edge != 0.0:
        r1 = r1 + edge * rev
    # a 5-day forward return: the anchored day plus four fresh daily draws
    extra = rng.normal(mu_daily, sig_daily, size=(n, 4)).sum(axis=1)
    r5 = r1 + extra
    idx = pd.bdate_range("2011-01-03", periods=n)
    return pd.DataFrame({"rev": rev, "fwd1": r1, "fwd5": r5,
                         "fwd1_l1": r1, "fwd5_l1": r5}, index=idx)
