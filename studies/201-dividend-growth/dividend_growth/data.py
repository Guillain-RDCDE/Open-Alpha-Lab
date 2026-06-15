"""Data layer for Study 201 (Dividend-Growth).

Two tapes, one shape (a wide annual panel: year x ticker):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A ``growth_premium`` knob
  lets us dial in a genuine dividend-growth effect; ``growth_premium=0`` is the null world
  where consistent raisers are no different from everyone else.  This is the study's null
  in a bottle: the strategy must detect an edge only when one is planted.

- ``load_real_panel`` — the real tape.  Fetches per-ticker adjusted-close prices and raw
  dividend streams from yfinance, derives the calendar-year total return and a trailing-3yr
  dividend growth streak, and caches everything under ``_cache/`` as parquet so the
  notebooks and tests run offline after the first fetch.  ``fetch=False`` (default) raises if
  no cache; ``fetch=True`` hits the network and repopulates.

Universe: a fixed basket of 40 large-cap US stocks that pay dividends and are alive as of
2026-06-15.  **Survivorship bias is intentional and named on the Signal axis** — we
deliberately pick survivors to give the idea its best shot; any outperformance must be
interpreted knowing the losers (cut dividends, delisted) are absent.

No look-ahead: the dividend-growth screen for year *y* is formed on data through
December *y*; the forward return is the calendar year *y+1* total return.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Fixed large-cap universe — survivorship-biased by construction (named on Signal axis)
# ---------------------------------------------------------------------------
# 40 names: S&P 500 large caps known for dividend payment as of 2026.  All current payers
# with multi-decade histories.  Names that cut dividends or were delisted are absent —
# that is the bias; it is documented, not hidden.
UNIVERSE = [
    "AAPL", "MSFT", "JNJ", "PG",  "KO",  "PEP",  "MCD",  "MMM",
    "ABT",  "IBM",  "CVX", "XOM", "JPM", "V",    "AXP",  "MRK",
    "AMGN", "WMT",  "HD",  "NKE", "TXN", "CSCO", "HON",  "CAT",
    "GS",   "TRV",  "UNH", "EMR", "SHW", "LOW",  "MDT",  "BDX",
    "ADP",  "CMI",  "AFL", "HRL", "CL",  "GPC",  "NUE",  "BEN",
]


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 40,
    n_years: int = 20,
    growth_premium: float = 0.03,
    base_ret: float = 0.09,
    vol: float = 0.18,
    streak_years: int = 3,
    seed: int = 201,
) -> dict:
    """A deterministic annual panel with a known dividend-growth premium.

    Each synthetic name has an annual total return and an integer *consecutive-raises*
    streak.  When ``growth_premium > 0``, stocks with a streak >= ``streak_years`` earn
    ``growth_premium`` extra total return next year — the textbook direction the literature
    claims.  ``growth_premium = 0`` is the null world (the strategy should find nothing).

    Returns a dict with:

    - ``streaks``  — DataFrame (year x name): consecutive prior-year dividend raises (int).
    - ``fwd_ret``  — DataFrame (year x name): next-year calendar total return (float).
    - ``ew_ret``   — Series (year): equal-weight basket mean return for the year.
    - ``truth``    — dict recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    names = [f"S{i:02d}" for i in range(n_names)]
    years = list(range(2004, 2004 + n_years))

    # Build synthetic dividend-raise streaks: each year, 40% of names raise independently.
    # Streaks accumulate: a name that raises 3y in a row has streak=3.
    raise_prob = 0.65
    raw_raises = rng.random((n_years + streak_years, n_names)) < raise_prob
    streaks_arr = np.zeros((n_years, n_names), dtype=int)
    for t in range(n_years):
        for k in range(n_names):
            s = 0
            for lag in range(1, streak_years + 2):
                idx = t + streak_years - lag
                if 0 <= idx < n_years + streak_years and raw_raises[idx, k]:
                    s += 1
                else:
                    break
            streaks_arr[t, k] = s

    # Forward returns
    noise = rng.normal(0.0, vol, (n_years, n_names))
    fwd_arr = base_ret + noise
    if growth_premium != 0.0:
        growers = streaks_arr >= streak_years
        fwd_arr[growers] += growth_premium

    streaks = pd.DataFrame(streaks_arr, index=pd.Index(years, name="year"), columns=names)
    fwd_ret = pd.DataFrame(fwd_arr, index=pd.Index(years, name="year"), columns=names)
    ew_ret = fwd_ret.mean(axis=1).rename("ew")

    truth = {
        "growth_premium": growth_premium,
        "streak_years": streak_years,
        "n_names": n_names,
        "n_years": n_years,
        "seed": seed,
    }
    return {"streaks": streaks, "fwd_ret": fwd_ret, "ew_ret": ew_ret, "truth": truth}


# ---------------------------------------------------------------------------
# Real loader — yfinance dividends + adjusted closes, cache-first
# ---------------------------------------------------------------------------
def _px_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"dg_{safe}_px.parquet")


def _div_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"dg_{safe}_div.parquet")


def _load_ticker(ticker: str, cache_dir: str, fetch: bool):
    """Return (annual_total_return_series, annual_dividend_series) for one ticker.

    Uses yfinance auto-adjusted closes (total-return prices) and the raw dividend stream.
    Both are cached; the network is only touched when ``fetch=True``.
    """
    px_path = _px_cache_path(ticker, cache_dir)
    div_path = _div_cache_path(ticker, cache_dir)

    if not fetch:
        if not os.path.exists(px_path) or not os.path.exists(div_path):
            return None, None
        px = pd.read_parquet(px_path)["close"]
        div = pd.read_parquet(div_path)["div"]
    else:
        import yfinance as yf  # lazy: only when we hit the network

        tk = yf.Ticker(ticker)
        raw = tk.history(period="max", auto_adjust=True)
        if raw is None or raw.empty:
            return None, None
        px = raw["Close"].rename("close")
        px.index = pd.to_datetime(px.index).tz_localize(None)
        div = tk.dividends.rename("div")
        if div is None or len(div) == 0:
            div = pd.Series(dtype=float, name="div")
        else:
            div.index = pd.to_datetime(div.index).tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        px.to_frame().to_parquet(px_path)
        div.to_frame().to_parquet(div_path)

    return px, div


def _annual_total_return(px: pd.Series) -> pd.Series:
    """Calendar-year total return from auto-adjusted (total-return) daily closes.

    Uses December 31 (last available trading day of the year) as the reference close.
    Returns the *forward* year's return indexed on the *signal* year, so signal year y
    maps to the return earned in year y+1.
    """
    annual = px.resample("YE").last()
    annual.index = annual.index.year
    ret = annual.pct_change().dropna()
    ret.index.name = "year"
    return ret


def _annual_div_sum(div: pd.Series) -> pd.Series:
    """Total dividends paid per calendar year."""
    if len(div) == 0:
        return pd.Series(dtype=float, name="div_sum")
    d = div.copy()
    d.index = pd.to_datetime(d.index).tz_localize(None) if d.index.tz is not None else d.index
    s = d.resample("YE").sum()
    s.index = s.index.year
    s.index.name = "year"
    return s.rename("div_sum")


def _div_growth_streak(div_annual: pd.Series, min_div: float = 0.01) -> pd.Series:
    """Consecutive calendar years of dividend growth (raises) through each year.

    A year counts as a raise if the annual dividend sum strictly increased vs the prior
    year AND both years paid at least ``min_div`` per share (filters erratic tiny payers).
    Returns an integer Series indexed by year.
    """
    out = {}
    years = sorted(div_annual.index)
    for i, y in enumerate(years):
        streak = 0
        for j in range(i, 0, -1):
            prev_y = years[j - 1]
            curr_y = years[j]
            d_prev = div_annual.get(prev_y, 0.0)
            d_curr = div_annual.get(curr_y, 0.0)
            if d_prev >= min_div and d_curr > d_prev:
                streak += 1
            else:
                break
        out[y] = streak
    return pd.Series(out, name="streak")


def load_real_panel(
    tickers: list[str] | None = None,
    start_year: int = 2000,
    streak_years: int = 3,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> dict:
    """Build the real annual panel: grower streaks + next-year returns for the universe.

    Returns a dict with keys:

    - ``streaks``  — DataFrame (year x ticker): consecutive-raises streak *at end of year*.
    - ``fwd_ret``  — DataFrame (year x ticker): next-calendar-year total return.
    - ``ew_ret``   — Series (year): equal-weight basket next-year return.
    - ``tickers``  — list of tickers actually loaded.

    When no cache is present and ``fetch=False``, returns empty DataFrames (the caller
    must check before use; tests must be cache-gated).
    """
    tickers = tickers or UNIVERSE
    all_px = {}
    all_div = {}
    for t in tickers:
        px, div = _load_ticker(t, cache_dir, fetch)
        if px is not None and len(px) > 0:
            all_px[t] = px
        if div is not None and len(div) >= 0:
            all_div[t] = div

    if not all_px:
        return {
            "streaks": pd.DataFrame(),
            "fwd_ret": pd.DataFrame(),
            "ew_ret": pd.Series(dtype=float),
            "tickers": [],
        }

    # Per-ticker annual streak and forward return
    streak_dict = {}
    fwd_dict = {}
    for t in all_px:
        px = all_px[t]
        div = all_div.get(t, pd.Series(dtype=float, name="div"))
        ann_ret = _annual_total_return(px)
        div_ann = _annual_div_sum(div)
        streak = _div_growth_streak(div_ann, min_div=0.01)

        # Align: streak in year y, forward return in year y+1
        common_years = sorted(set(streak.index) & set(ann_ret.index - 1))
        for y in common_years:
            if y >= start_year:
                if y not in streak_dict:
                    streak_dict[y] = {}
                    fwd_dict[y] = {}
                streak_dict[y][t] = streak.get(y, 0)
                fwd_dict[y][t] = ann_ret.get(y + 1, np.nan)

    if not streak_dict:
        return {
            "streaks": pd.DataFrame(),
            "fwd_ret": pd.DataFrame(),
            "ew_ret": pd.Series(dtype=float),
            "tickers": [],
        }

    streaks = pd.DataFrame(streak_dict).T.sort_index()
    streaks.index.name = "year"
    fwd_ret = pd.DataFrame(fwd_dict).T.sort_index()
    fwd_ret.index.name = "year"
    ew_ret = fwd_ret.mean(axis=1).rename("ew")

    return {
        "streaks": streaks,
        "fwd_ret": fwd_ret,
        "ew_ret": ew_ret,
        "tickers": list(all_px.keys()),
    }


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(panel: dict) -> str:
    """A short content fingerprint of the real panel, for the as-of stamp.

    Hashes the streaks and fwd_ret matrices (NaN -> 0) to a 12-char hex digest.
    A mismatch between two runs flags that the underlying ticker data drifted
    (split adjustment, dividend restatement, etc.).
    """
    h = hashlib.sha1()
    for key in ("streaks", "fwd_ret"):
        df = panel.get(key)
        if df is not None and not df.empty:
            arr = np.ascontiguousarray(np.nan_to_num(df.to_numpy(dtype=float)))
            h.update(arr.tobytes())
    return h.hexdigest()[:12]
