"""Data layer for Study 217 (Lipstick Index).

Three components:

- ``RECESSION_DATES`` — a hardcoded NBER recession table (start/end months
  1980-2020) used as the offline reference for the folklore claim. Source: NBER.

- ``synthetic_monthly`` — a deterministic, offline monthly-return generator with
  an optional planted "lipstick premium" knob. ``signal_bps = 0`` is the null
  (no defensive tilt in recessions), so tests can confirm the machinery is
  honest before looking at real data.

- ``fetch_cosmetics`` — yfinance pull for a small cosmetics basket (EL, ULTA,
  COTY, ELF) and SPY, with a local parquet cache in studies/217-lipstick-index/_cache/.
  Cache-first; network only on first run.

The Lipstick Index folklore (Leonard Lauder, ~2001): during recessions consumers
trade down from luxury goods to affordable luxuries — lipstick or nail polish —
so cosmetics companies outperform the broad market in downturns. We test whether
cosmetics stocks show a measurable defensive relative-return vs SPY around NBER
recessions.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# NBER recession dates — hardcoded offline reference
# ---------------------------------------------------------------------------
# Columns:
#   start : first month of recession (YYYY-MM-DD, first day of month)
#   end   : last month of recession (YYYY-MM-DD, first day of trough month)
#
# Source: NBER Business Cycle Dating Committee
#   https://www.nber.org/research/business-cycles
# We use 1980 onward (the era in which EL / Estee Lauder has been public).
#
# Recessions included:
#   1980-01 to 1980-07  (6 months)
#   1981-07 to 1982-11  (16 months)
#   1990-07 to 1991-03  (8 months)
#   2001-03 to 2001-11  (8 months)
#   2007-12 to 2009-06  (18 months — Global Financial Crisis)
#   2020-02 to 2020-04  (2 months — COVID)

RECESSION_DATES: list[dict] = [
    {"start": "1980-01-01", "end": "1980-07-01", "name": "1980 recession"},
    {"start": "1981-07-01", "end": "1982-11-01", "name": "1981-82 recession"},
    {"start": "1990-07-01", "end": "1991-03-01", "name": "1990-91 recession"},
    {"start": "2001-03-01", "end": "2001-11-01", "name": "Dot-com recession"},
    {"start": "2007-12-01", "end": "2009-06-01", "name": "Global Financial Crisis"},
    {"start": "2020-02-01", "end": "2020-04-01", "name": "COVID recession"},
]

RECESSION_DF: pd.DataFrame = pd.DataFrame(RECESSION_DATES)
RECESSION_DF["start"] = pd.to_datetime(RECESSION_DF["start"])
RECESSION_DF["end"] = pd.to_datetime(RECESSION_DF["end"])


# Cosmetics basket tickers (all listed on US exchanges, price history to ~1995+)
# EL  = Estee Lauder Companies (IPO 1995) — the founding family behind the "index"
# ULTA = Ulta Beauty (IPO 2007)
# COTY = Coty Inc. (IPO 2013, but available via yfinance adjusted for splits)
# ELF  = e.l.f. Beauty (IPO 2016)
# SPY  = S&P 500 ETF — benchmark
COSMETICS_TICKERS = ["EL", "ULTA", "COTY", "ELF"]
BENCHMARK_TICKER = "SPY"
ALL_TICKERS = COSMETICS_TICKERS + [BENCHMARK_TICKER]


def recession_table() -> pd.DataFrame:
    """Return a clean copy of the NBER recession dates table.

    Columns: ``start`` (Timestamp), ``end`` (Timestamp), ``name`` (str).
    """
    return RECESSION_DF.copy()


def is_recession(date: pd.Timestamp, rec_df: pd.DataFrame | None = None) -> bool:
    """Return True if ``date`` falls inside any NBER recession month."""
    if rec_df is None:
        rec_df = recession_table()
    d = pd.Timestamp(date).normalize()
    return any(
        (d >= row["start"]) and (d <= row["end"])
        for _, row in rec_df.iterrows()
    )


def recession_flag_series(index: pd.DatetimeIndex,
                           rec_df: pd.DataFrame | None = None) -> pd.Series:
    """Return a boolean Series aligned to ``index``, True in recession months."""
    if rec_df is None:
        rec_df = recession_table()
    flags = pd.Series(False, index=index, name="in_recession")
    for _, row in rec_df.iterrows():
        flags[(index >= row["start"]) & (index <= row["end"])] = True
    return flags


# ---------------------------------------------------------------------------
# Synthetic monthly return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 240,
    signal_bps: float = 0.0,
    base_return_ann: float = 0.08,
    vol_ann: float = 0.15,
    recession_fraction: float = 0.15,
    seed: int = 217,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly-return series with an optional planted lipstick premium.

    Each month is drawn i.i.d. from N(base_return_ann/12, (vol_ann/sqrt(12))^2).
    In 'recession' months (selected randomly with probability ``recession_fraction``),
    the cosmetics basket gets an extra drift of ``signal_bps`` bps/month over SPY.
    ``signal_bps = 0`` is the null — no defensive tilt.

    Returns ``(df, truth)`` where ``df`` has columns:
      ``month`` (period), ``spy_ret`` (monthly return), ``cosm_ret`` (basket return),
      ``rel_ret`` (cosm_ret - spy_ret), ``in_recession`` (bool).
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    base_m = base_return_ann / 12.0
    vol_m = vol_ann / np.sqrt(12)

    months = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    in_rec = rng.random(n_months) < recession_fraction

    spy_rets = rng.normal(base_m, vol_m, n_months)
    # Cosmetics: same as SPY plus planted premium in recession months
    cosm_extra = np.where(in_rec, signal_bps * 1e-4 / 12.0, 0.0)
    cosm_rets = spy_rets + cosm_extra + rng.normal(0.0, vol_m * 0.3, n_months)

    rel_rets = cosm_rets - spy_rets

    df = pd.DataFrame({
        "month": months,
        "spy_ret": spy_rets,
        "cosm_ret": cosm_rets,
        "rel_ret": rel_rets,
        "in_recession": in_rec,
    })
    truth = {
        "n_months": n_months,
        "signal_bps": signal_bps,
        "base_return_ann": base_return_ann,
        "vol_ann": vol_ann,
        "recession_fraction": recession_fraction,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — yfinance cosmetics basket (study-local cache)
# ---------------------------------------------------------------------------
def fetch_cosmetics(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1996-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Fetch monthly total-return data for the cosmetics basket + SPY.

    Cache-first: writes parquet to ``cache_dir/cosmetics_monthly.parquet``.
    On subsequent calls, reads from cache without touching the network.

    Returns a DataFrame with columns:
      ``month`` (period start), ``ticker``, ``ret`` (monthly simple return).
    Tickers with IPOs after ``start`` begin from their first available month.

    Note: EL (Estee Lauder) IPO'd in 1995, ULTA in 2007, COTY in 2013,
    ELF in 2016. The basket shrinks in earlier periods — cells that use
    this function should be aware of the survivorship window.
    """
    cache_path = os.path.join(cache_dir, "cosmetics_monthly.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    import yfinance as yf

    frames = []
    for ticker in ALL_TICKERS:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            continue
        close = raw["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        ret = close.pct_change().dropna()
        ret.index = ret.index.to_period("M").to_timestamp()
        df_t = pd.DataFrame({"month": ret.index, "ticker": ticker, "ret": ret.values})
        frames.append(df_t)

    if not frames:
        raise RuntimeError("yfinance returned no data — check network access.")

    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(out["month"])
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def build_analysis_frame(
    raw: pd.DataFrame,
    rec_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pivot raw ticker returns into a wide frame and add recession flag.

    Returns a DataFrame indexed by month with columns:
      ``EL``, ``ULTA``, ``COTY``, ``ELF``, ``SPY``,
      ``cosm_ret`` (equal-weight average of available cosmetics tickers that month),
      ``rel_ret`` (cosm_ret - SPY),
      ``in_recession`` (bool).

    Only months where SPY is available are kept. Cosmetics mean is computed
    on available tickers per month (unbalanced panel).
    """
    if rec_df is None:
        rec_df = recession_table()

    pivot = raw.pivot_table(index="month", columns="ticker", values="ret", aggfunc="first")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    if "SPY" not in pivot.columns:
        raise ValueError("SPY data missing from raw frame.")

    cosm_cols = [c for c in COSMETICS_TICKERS if c in pivot.columns]
    pivot["cosm_ret"] = pivot[cosm_cols].mean(axis=1)
    pivot["rel_ret"] = pivot["cosm_ret"] - pivot["SPY"]
    pivot["in_recession"] = recession_flag_series(pivot.index, rec_df)
    pivot = pivot.dropna(subset=["SPY", "cosm_ret"])
    pivot.index.name = "month"
    return pivot.reset_index()


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the rel_ret column."""
    vals = df["rel_ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
