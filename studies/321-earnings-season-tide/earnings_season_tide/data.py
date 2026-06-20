"""Data layer for Study 321 (Earnings-Season-Tide).

Two tapes, one shape (a tz-naive daily close series of a broad-market index):

- ``synthetic_daily`` — a *deterministic, offline* generator. A single knob, ``tide_bps``,
  plants a constant per-day drift *bump* on the days that fall inside the four peak
  earnings windows of each year (mid-late January, April, July and October). With
  ``tide_bps = 0`` the tape is a pure random walk with no calendar structure — the study's
  null in a bottle, so a test can assert the calendar test fires only when a real tide is
  planted, and not otherwise.

- ``load_real`` — the real broad-market daily tape (SPY total-return), loaded **cache-only**
  from the shared repo ``_cache/`` parquet (or a study-local copy), with a graceful
  ``quantlab.data`` fallback and finally a clear error so the test-suite and the
  reproducible core never silently touch the network.

The earnings-window calendar is *known in advance* — it is a function of the date alone —
so there is **no execution lag** here (a calendar-known rule needs no ``shift``). That
discipline, and the choice to treat the windows as decided before the fact, lives with the
``in_earnings_window`` mask in ``strategy.py``; this module only manufactures and serves
the price tape.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The four peak-earnings windows, as (month, start_day, end_day) inclusive of calendar
# days. US large-cap earnings cluster in the second half of the month following each
# fiscal quarter-end: mid-late Jan (Q4), Apr (Q1), Jul (Q2) and Oct (Q3). These ranges
# are HARD-CODED and fixed in advance — the whole point is a pre-declared calendar rule.
PEAK_WINDOWS = (
    (1, 15, 31),   # mid-late January  — Q4 prior year
    (4, 12, 30),   # mid-late April    — Q1
    (7, 12, 31),   # mid-late July     — Q2
    (10, 12, 31),  # mid-late October  — Q3
)


# ---------------------------------------------------------------------------
# The earnings-window calendar mask (pure function of the date)
# ---------------------------------------------------------------------------
def in_earnings_window(index: pd.DatetimeIndex, windows=PEAK_WINDOWS) -> pd.Series:
    """Boolean mask, True on the trading days inside any of the four peak windows.

    Purely a function of the calendar date — no price information, no look-ahead, hence
    no execution lag. Returned as a ``pd.Series`` aligned to ``index``.
    """
    idx = pd.DatetimeIndex(index)
    mask = np.zeros(len(idx), dtype=bool)
    m = idx.month.to_numpy()
    d = idx.day.to_numpy()
    for (mon, lo, hi) in windows:
        mask |= (m == mon) & (d >= lo) & (d <= hi)
    return pd.Series(mask, index=idx, name="in_window")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 30,
    tide_bps: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "1995-01-02",
    seed: int = 321,
    windows=PEAK_WINDOWS,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close tape with a *known* earnings-season tide.

    Daily log returns are i.i.d. normal of standard deviation ``daily_vol_bps`` (in basis
    points), plus a constant drift bump of ``tide_bps`` (bps/day) on every trading day that
    falls inside one of the four peak earnings windows.

    - ``tide_bps = 0``  → no calendar structure at all (a pure random walk); the calendar
      test must read insignificant here.
    - ``tide_bps > 0``  → a genuine in-window drift the calendar test should recover.

    Returns ``(bars, truth)`` where ``bars`` carries a single ``close`` column on a tz-naive
    business-day index and ``truth`` records the planted parameters.

    NOTE on span: we build the index with ``pd.bdate_range`` over a span of ~``n_years`` and
    keep it far under the ns-Timestamp overflow horizon (~292 yr from the 1970 epoch). We
    never use ``pd.date_range(periods=BIG, freq="M")``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(round(n_years * TRADING_DAYS_PER_YEAR))
    sessions = pd.bdate_range(start=start, periods=n_days)

    mask = in_earnings_window(sessions, windows).to_numpy()
    vol = daily_vol_bps * 1e-4
    bump = tide_bps * 1e-4

    eps = rng.normal(0.0, vol, n_days)
    log_ret = eps + np.where(mask, bump, 0.0)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    bars = pd.DataFrame(
        {"close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "tide_bps": tide_bps,
        "daily_vol_bps": daily_vol_bps,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "n_in_window": int(mask.sum()),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — cache-only, shared parquet first, then quantlab, then error
# ---------------------------------------------------------------------------
def _read_shared_parquet(path: str) -> pd.DataFrame:
    """Read a shared OHLCV parquet (capitalised columns, ``Date`` index) into a one-column
    ``close`` frame on a tz-naive index."""
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close")
    if close_col is None:
        raise ValueError(f"parquet at {path} has no Close column: {list(df.columns)}")
    out = df[[close_col]].rename(columns={close_col: "close"})
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    return out


def shared_cache_path(ticker: str = "SPY", mode: str = "total_return") -> str:
    """Where the shared repo cache parquet for ``ticker`` lives (may not exist offline)."""
    return os.path.join(REPO_CACHE, f"{ticker}_{mode}.parquet")


def load_real(
    ticker: str = "SPY",
    mode: str = "total_return",
    fetch: bool = False,
) -> pd.DataFrame:
    """Real daily close for a broad-market index; **cache-first**, network only if asked.

    Resolution order:

    1. The study-local ``_cache/{ticker}_{mode}.parquet`` if present.
    2. The shared repo ``_cache/{ticker}_{mode}.parquet`` (the desk's pinned tapes).
    3. ``quantlab.data`` (a cached desk loader), if importable and it returns bars.
    4. Only if ``fetch=True``, hit the network via ``yfinance`` and cache it study-locally.

    Otherwise raise ``FileNotFoundError`` — the offline core and tests never touch the wire.
    Returns a one-column ``close`` frame, total-return labelled (SPY carries dividends).
    """
    local = os.path.join(STUDY_CACHE, f"{ticker}_{mode}.parquet")
    if os.path.exists(local):
        return _read_shared_parquet(local)

    shared = shared_cache_path(ticker, mode)
    if os.path.exists(shared):
        return _read_shared_parquet(shared)

    # Graceful desk-loader fallback.
    try:
        from quantlab import data as qld  # type: ignore

        for fn_name in ("load_prices", "load_total_return", "get_prices"):
            fn = getattr(qld, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(ticker)
            except Exception:
                continue
            if df is not None and len(df):
                if isinstance(df, pd.Series):
                    df = df.to_frame("close")
                return _read_shared_parquet_from_frame(df)
    except Exception:
        pass

    if not fetch:
        raise FileNotFoundError(
            f"No cached tape for {ticker} ({mode}). Looked in {local} and {shared}. "
            f"Call load_real(fetch=True) once to populate the study cache, or run offline "
            f"on data.synthetic_daily()."
        )

    import yfinance as yf  # lazy: only on an explicit network opt-in

    raw = yf.download(ticker, start="1993-01-01", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.rename(columns=str.lower)[["close"]]
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    os.makedirs(STUDY_CACHE, exist_ok=True)
    out.to_parquet(local)
    return out


def _read_shared_parquet_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLC(V) or close frame from quantlab into our ``close`` shape."""
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close", df.columns[-1])
    out = df[[close_col]].rename(columns={close_col: "close"})
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "date"
    return out


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
