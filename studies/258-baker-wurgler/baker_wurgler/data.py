"""Data layer for Study 258 (Baker-Wurgler investor-sentiment index).

Three components, all offline for the reproducible core:

- ``BW_SENTIMENT_ANNUAL`` -- a hardcoded, year-end *reconstruction* of the
  Baker & Wurgler (2006) orthogonalized investor-sentiment index (the
  "SENT-perp" series, standardized to mean 0 / unit variance over its sample).
  The original monthly file is published on Jeffrey Wurgler's NYU website and is
  updated periodically; we cannot fetch it offline, so we hardcode the well-known
  *shape* of the index at annual resolution -- the documented peaks (1968-69
  electronics/conglomerate bubble, the 1980-81 biotech/energy froth, the 1987
  pre-crash spike, the 2000 dot-com top, the 2007 pre-GFC high) and troughs
  (post-1974, the early-1990s, the 1991/2003/2009 bottoms).  Values are the
  standardized index level; positive = optimistic/speculative, negative =
  pessimistic.  This is a curated table, not the verbatim BW file -- it captures
  the index's regime structure so the contrarian sort can be tested offline.

- ``bw_monthly`` -- expands the annual anchors to a deterministic monthly series
  by linear interpolation plus a fixed-seed AR-flavoured wiggle, so the monthly
  signal has realistic persistence.  ``seed`` is fixed (258); fully reproducible.

- ``fetch_market_monthly`` -- real ^GSPC (and size-proxy ^RUT) monthly closes
  from the repo-level ``_cache`` parquets (split-only OHLCV).  Cache-only by
  default; ``fetch=True`` does a lazy yfinance download.  Price-only (no
  dividends) -- labeled on the Signal axis.

The Baker-Wurgler claim (2006, 2007) is **contrarian**: when sentiment is HIGH,
subsequent aggregate returns are LOW; when sentiment is LOW, subsequent returns
are HIGH.  The effect is strongest for "hard-to-arbitrage / hard-to-value"
stocks -- small caps, young firms, non-dividend payers, extreme-growth and
distressed names.  We proxy the cross-sectional leg with small-minus-large
(^RUT - ^GSPC) where the size proxy is available.

No look-ahead: sentiment measured at the end of month t predicts the return
*earned during* month t+1 (one-month execution lag; positions formed at the
month-end close on the published sentiment reading).

**Honesty caveats** (named on the Signal axis):
  - Price-only returns (no dividends) -- the level effect is understated, the
    *spread* across regimes is roughly dividend-neutral.
  - The hardcoded index is a reconstruction; the verbatim BW file would shift
    individual months but preserves the high/low regime structure.
  - The size proxy (^RUT) only starts in 1987, so the cross-sectional leg uses a
    shorter window than the index-level test.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_CACHE = os.path.join(REPO_ROOT, "_cache")

GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
RUT_CACHE = os.path.join(REPO_CACHE, "^RUT_split_only.parquet")


# ---------------------------------------------------------------------------
# Hardcoded Baker-Wurgler sentiment reconstruction (annual year-end anchors)
# ---------------------------------------------------------------------------
# Standardized orthogonalized index level (mean ~0, unit-ish variance over the
# sample).  Positive = high sentiment / speculative optimism; negative =
# pessimism.  Anchors capture the documented BW regime structure.
#
# Source/spirit:
#   Baker, M. & Wurgler, J. (2006). "Investor Sentiment and the Cross-Section
#     of Stock Returns." Journal of Finance 61(4), 1645-1680.
#   Baker, M. & Wurgler, J. (2007). "Investor Sentiment in the Stock Market."
#     Journal of Economic Perspectives 21(2), 129-151.
#   Index updates: Jeffrey Wurgler, NYU Stern (people.stern.nyu.edu/jwurgler).
# As-of curation: 2026-06-17.
BW_SENTIMENT_ANNUAL: dict[int, float] = {
    1965: 0.55, 1966: 0.20, 1967: 1.05, 1968: 2.05, 1969: 1.55,
    1970: 0.05, 1971: 0.45, 1972: 1.15, 1973: 0.10, 1974: -0.85,
    1975: -0.45, 1976: 0.20, 1977: -0.20, 1978: 0.10, 1979: 0.55,
    1980: 1.35, 1981: 1.55, 1982: 0.10, 1983: 1.05, 1984: 0.20,
    1985: 0.05, 1986: 0.60, 1987: 1.30, 1988: -0.20, 1989: 0.35,
    1990: -0.45, 1991: -0.55, 1992: -0.10, 1993: 0.25, 1994: -0.05,
    1995: 0.45, 1996: 1.05, 1997: 1.35, 1998: 1.55, 1999: 2.30,
    2000: 2.55, 2001: 0.85, 2002: -0.55, 2003: -0.75, 2004: 0.25,
    2005: 0.55, 2006: 0.85, 2007: 1.10, 2008: -0.45, 2009: -1.05,
    2010: -0.35, 2011: -0.20, 2012: -0.10, 2013: 0.45, 2014: 0.65,
    2015: 0.35, 2016: 0.25, 2017: 0.75, 2018: 0.55, 2019: 0.40,
    2020: 0.95, 2021: 1.45, 2022: -0.55, 2023: 0.15, 2024: 0.70,
    2025: 0.85,
}


def sentiment_annual() -> pd.Series:
    """Year-end standardized BW sentiment reconstruction as a Series (index = year)."""
    s = pd.Series(BW_SENTIMENT_ANNUAL, name="bw_sent").sort_index()
    s.index.name = "year"
    return s


# ---------------------------------------------------------------------------
# Monthly expansion -- deterministic, offline
# ---------------------------------------------------------------------------
def bw_monthly(
    wiggle: float = 0.12,
    seed: int = 258,
) -> pd.Series:
    """Monthly BW sentiment level: interpolated annual anchors + fixed-seed wiggle.

    Year-end anchors are linearly interpolated to month-ends, then a small
    persistent (AR(1)-flavoured) wiggle is added so the monthly signal looks like
    a real sentiment tape (smooth, mean-reverting) rather than a piecewise line.
    Fully deterministic given ``seed``.

    Returns a month-end-indexed Series named ``bw_sent`` spanning Jan of the first
    anchor year through Dec of the last.
    """
    ann = sentiment_annual()
    years = ann.index.to_numpy()
    y0, y1 = int(years.min()), int(years.max())

    months = pd.date_range(f"{y0}-01-31", f"{y1}-12-31", freq="ME")
    # Fractional-year position of each month-end (year-end anchor = Dec of that year)
    frac = np.array([d.year + (d.month - 12) / 12.0 for d in months])
    anchor_x = years.astype(float)
    anchor_y = ann.to_numpy(dtype=float)
    base = np.interp(frac, anchor_x, anchor_y)

    # Persistent wiggle: AR(1) noise, fixed seed
    rng = np.random.default_rng(seed)
    n = len(months)
    eps = rng.normal(0.0, wiggle, n)
    ar = np.zeros(n)
    phi = 0.6
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + eps[t]
    level = base + ar

    out = pd.Series(level, index=months, name="bw_sent")
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------
# Synthetic market panel with a *known* contrarian sentiment effect
# ---------------------------------------------------------------------------
def synthetic_market(
    n_months: int = 360,
    contrarian_bps: float = 60.0,
    base_ret: float = 0.006,
    ret_vol: float = 0.043,
    seed: int = 258,
) -> tuple[pd.DataFrame, dict]:
    """A deterministic monthly market tape with a planted *contrarian* sentiment effect.

    Returns a frame with columns ``bw_sent`` (the lagged sentiment signal at the
    start of the month) and ``ret`` (the simple return earned that month).  The
    data-generating process is::

        ret_t = base_ret - (contrarian_bps * 1e-4) * sent_{t-1} + noise_t

    so HIGH prior sentiment depresses the next month's return (Baker-Wurgler
    contrarian sign).  ``contrarian_bps = 0`` is the null: sentiment carries no
    forward information.

    Returns ``(df, truth)``.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("1990-01-31", periods=n_months, freq="ME")

    # A persistent sentiment process in [-2.5, 2.5]
    eps = rng.normal(0.0, 0.35, n_months)
    sent = np.zeros(n_months)
    for t in range(1, n_months):
        sent[t] = 0.92 * sent[t - 1] + eps[t]
    sent = np.clip(sent, -2.5, 2.5)

    noise = rng.normal(0.0, ret_vol, n_months)
    ret = np.full(n_months, base_ret)
    # Prior-month sentiment predicts THIS month's return with a contrarian sign
    ret[1:] = base_ret - (contrarian_bps * 1e-4) * sent[:-1] + noise[1:]
    ret[0] = base_ret + noise[0]

    df = pd.DataFrame(
        {"bw_sent": sent, "ret": ret},
        index=months,
    )
    df.index.name = "date"
    truth = {
        "n_months": n_months,
        "contrarian_bps": contrarian_bps,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC / ^RUT monthly closes from repo cache
# ---------------------------------------------------------------------------
def _load_monthly_close(cache_path: str, fetch: bool, ticker: str) -> pd.Series:
    """Load a single ticker's month-end close as a Series (cache-only unless fetch)."""
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached price parquet at {cache_path}. "
                "Pass fetch=True once to populate (network)."
            )
        df = pd.read_parquet(cache_path)
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        monthly = close.resample("ME").last().dropna()
        monthly.name = ticker
        return monthly

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(ticker, start="1927-01-01", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker] if ticker in raw["Close"].columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close.index = pd.to_datetime(close.index)
    monthly = close.resample("ME").last().dropna()
    monthly.name = ticker
    return monthly


def fetch_market_monthly(
    fetch: bool = False,
    gspc_cache: str = GSPC_CACHE,
    rut_cache: str = RUT_CACHE,
) -> pd.DataFrame:
    """Month-end ^GSPC close (and ^RUT size proxy when available), cache-only by default.

    Returns a DataFrame indexed by month-end with columns:
      - ``gspc`` : S&P 500 month-end close (price-only)
      - ``rut``  : Russell 2000 month-end close (price-only), or absent if no cache

    Price-only (no dividends) -- labeled on the Signal axis.  Raises
    ``FileNotFoundError`` if the GSPC cache is absent and ``fetch=False``.
    """
    gspc = _load_monthly_close(gspc_cache, fetch, "gspc")
    cols = {"gspc": gspc}
    # Size proxy is optional (RUT starts 1987)
    if fetch or os.path.exists(rut_cache):
        try:
            rut = _load_monthly_close(rut_cache, fetch, "rut")
            cols["rut"] = rut
        except (FileNotFoundError, RuntimeError):
            pass
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    return out


def merge_sentiment_returns(
    market: pd.DataFrame,
    sentiment: pd.Series,
) -> pd.DataFrame:
    """Join lagged month-end sentiment to the *next* month's market return.

    For each month-end t, the sentiment reading sent_t predicts the return earned
    during month t+1 (one-month execution lag, no look-ahead).  Returns a frame
    indexed by the *return* month with columns:
      - ``bw_sent`` : sentiment at the prior month-end (the signal)
      - ``ret``     : ^GSPC simple price return earned this month
      - ``smb``     : ^RUT - ^GSPC return (small-minus-large), where available
    """
    gspc_ret = market["gspc"].pct_change()
    df = pd.DataFrame({"ret": gspc_ret})
    if "rut" in market.columns:
        rut_ret = market["rut"].pct_change()
        df["smb"] = rut_ret - gspc_ret
    # Signal is prior month-end sentiment (shift the sentiment forward by 1 month)
    sent_aligned = sentiment.reindex(market.index, method="ffill")
    df["bw_sent"] = sent_aligned.shift(1)
    df = df.dropna(subset=["ret", "bw_sent"])
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """Short content fingerprint (sha1[:12]) of a Series/DataFrame's numeric tail."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.select_dtypes("number").to_numpy(dtype=float).ravel()
    else:
        arr = pd.Series(obj).to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr[-512:]).tobytes())
    return h.hexdigest()[:12]
