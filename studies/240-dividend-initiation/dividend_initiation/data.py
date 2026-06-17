"""Data layer for Study 240 (Dividend-Initiation).

Two tapes, one shape (an annual event panel: year x ticker):

- ``synthetic_panel`` — a *deterministic, offline* generator.  An ``initiation_premium``
  knob lets us dial in a genuine post-initiation effect; ``initiation_premium=0`` is the
  null world where initiators are no different from non-initiators.  This is the study's
  null in a bottle: the strategy must detect an edge only when one is planted.

- ``load_real_panel`` — the real tape.  Fetches per-ticker adjusted-close prices and raw
  dividend streams from yfinance, identifies the first dividend year (initiation event),
  and caches everything under ``_cache/`` as parquet so the notebooks and tests run offline
  after the first fetch.  ``fetch=False`` (default) raises if no cache; ``fetch=True`` hits
  the network and repopulates.

Universe: a fixed basket of 50 large-cap US stocks, mixing confirmed dividend payers
(initiators — they once paid a first dividend) and long-standing non-payers, alive as of
2026-06-16.  **Survivorship bias is intentional and named on the Signal axis** — we
deliberately pick survivors to give the idea its best shot.

No look-ahead: a ticker is classified as an "initiator" in year *y* if year *y* is the
first calendar year with positive dividends in the full history.  The forward return is the
calendar year *y+1* total return.  Non-initiators (no dividend in year *y*, not having
previously initiated) serve as the control arm for their matched year.
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
# 50 names: a mix of established dividend payers that once initiated dividends and
# large-cap growth names that historically did not pay dividends (or initiated later).
# Survivorship bias: all are large-cap survivors as of 2026-06-16.
UNIVERSE = [
    # Established payers — all once paid a first dividend
    "AAPL", "MSFT", "JNJ", "PG",  "KO",  "PEP",  "MCD",  "MMM",
    "ABT",  "IBM",  "CVX", "XOM", "JPM", "V",    "AXP",  "MRK",
    "AMGN", "WMT",  "HD",  "NKE", "TXN", "CSCO", "HON",  "CAT",
    "GS",   "TRV",  "UNH", "EMR", "SHW", "LOW",
    # Growth names that did not pay for many years, some later initiated
    "GOOG", "META", "NVDA", "AMZN", "TSLA", "CRM", "ADBE", "NFLX",
    "PYPL", "UBER", "LYFT", "SNAP", "TWLO", "ZM",  "DDOG", "SNOW",
    "PLTR", "RBLX", "RIVN", "COIN",
]


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 50,
    n_years: int = 20,
    initiation_premium: float = 0.04,
    base_ret: float = 0.09,
    vol: float = 0.25,
    initiation_rate: float = 0.05,
    seed: int = 240,
) -> dict:
    """A deterministic annual panel with a known dividend-initiation premium.

    Each synthetic name can "initiate" a dividend in a given year with probability
    ``initiation_rate``.  Once initiated, a firm remains a payer.  When
    ``initiation_premium > 0``, the initiation year and the following year earn
    ``initiation_premium`` extra total return — the textbook signalling direction.
    ``initiation_premium = 0`` is the null world (the strategy should find nothing).

    Returns a dict with:

    - ``initiation_mask`` — DataFrame (year x name): True if this year is the
      firm's first dividend year (initiation event).
    - ``control_mask``    — DataFrame (year x name): True if the firm has never paid a
      dividend through year y (the non-initiator control arm for year y).
    - ``fwd_ret``         — DataFrame (year x name): next-year calendar total return.
    - ``ew_ret``          — Series (year): equal-weight basket mean return for that year.
    - ``truth``           — dict recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    names = [f"S{i:02d}" for i in range(n_names)]
    years = list(range(2004, 2004 + n_years))

    # Determine initiation year for each name (if any)
    # A name initiates at most once; some names never initiate
    initiation_year = np.full(n_names, -1, dtype=int)  # -1 = never initiates
    for k in range(n_names):
        for t_idx, y in enumerate(years):
            if rng.random() < initiation_rate and initiation_year[k] == -1:
                initiation_year[k] = t_idx  # index into years

    # Build masks
    init_arr = np.zeros((n_years, n_names), dtype=bool)
    ctrl_arr = np.zeros((n_years, n_names), dtype=bool)
    for t_idx in range(n_years):
        for k in range(n_names):
            if initiation_year[k] == t_idx:
                init_arr[t_idx, k] = True  # this is the first dividend year
            elif initiation_year[k] == -1 or initiation_year[k] > t_idx:
                ctrl_arr[t_idx, k] = True  # has never paid a dividend through year y

    # Forward returns — initiation_premium applied at the initiation year
    noise = rng.normal(0.0, vol, (n_years, n_names))
    fwd_arr = base_ret + noise
    if initiation_premium != 0.0:
        fwd_arr[init_arr] += initiation_premium

    initiation_mask = pd.DataFrame(
        init_arr, index=pd.Index(years, name="year"), columns=names
    )
    control_mask = pd.DataFrame(
        ctrl_arr, index=pd.Index(years, name="year"), columns=names
    )
    fwd_ret = pd.DataFrame(fwd_arr, index=pd.Index(years, name="year"), columns=names)
    ew_ret = fwd_ret.mean(axis=1).rename("ew")

    truth = {
        "initiation_premium": initiation_premium,
        "initiation_rate": initiation_rate,
        "n_names": n_names,
        "n_years": n_years,
        "seed": seed,
    }
    return {
        "initiation_mask": initiation_mask,
        "control_mask": control_mask,
        "fwd_ret": fwd_ret,
        "ew_ret": ew_ret,
        "truth": truth,
    }


# ---------------------------------------------------------------------------
# Real loader — yfinance dividends + adjusted closes, cache-first
# ---------------------------------------------------------------------------
def _px_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"di_{safe}_px.parquet")


def _div_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"di_{safe}_div.parquet")


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
    """Calendar-year total return from auto-adjusted (total-return) daily closes."""
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


def _find_initiation_year(div_annual: pd.Series, min_div: float = 0.01) -> int | None:
    """Return the first calendar year in which dividends exceeded ``min_div``.

    Returns None if the ticker never paid a meaningful dividend.  The initiation year is
    identified from the full history without look-ahead (the full history is available on
    the signal date; we identify the event in hindsight).
    """
    paying_years = div_annual[div_annual >= min_div].index
    if len(paying_years) == 0:
        return None
    return int(paying_years.min())


def load_real_panel(
    tickers: list[str] | None = None,
    start_year: int = 2000,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> dict:
    """Build the real annual panel: initiation events + next-year returns.

    For each ticker we identify the first year with positive dividends (the initiation year).
    In that year, the ticker is flagged as an "initiator".  Non-initiators (tickers that
    have NEVER paid a dividend through year y, in the slice of history available) form the
    control arm.  Note: because we use full-history dividends, a ticker's initiation year
    is identified by looking at all available data — this mirrors real-world event studies
    where the initiation is observable from public announcements.

    Returns a dict with:

    - ``initiation_mask`` — DataFrame (year x ticker): True in the initiation year only.
    - ``control_mask``    — DataFrame (year x ticker): True if ticker has not yet initiated
      through year y.
    - ``fwd_ret``         — DataFrame (year x ticker): next-calendar-year total return.
    - ``ew_ret``          — Series (year): equal-weight basket next-year return.
    - ``tickers``         — list of tickers actually loaded.
    - ``initiation_years``— dict {ticker: first_div_year}.

    When no cache is present and ``fetch=False``, returns empty DataFrames.
    """
    tickers = tickers or UNIVERSE
    all_px: dict = {}
    all_div_ann: dict = {}
    initiation_years: dict = {}

    for t in tickers:
        px, div = _load_ticker(t, cache_dir, fetch)
        if px is not None and len(px) > 0:
            all_px[t] = px
        if div is not None:
            div_ann = _annual_div_sum(div)
            all_div_ann[t] = div_ann
            init_yr = _find_initiation_year(div_ann)
            if init_yr is not None:
                initiation_years[t] = init_yr

    if not all_px:
        return {
            "initiation_mask": pd.DataFrame(),
            "control_mask": pd.DataFrame(),
            "fwd_ret": pd.DataFrame(),
            "ew_ret": pd.Series(dtype=float),
            "tickers": [],
            "initiation_years": {},
        }

    # Build per-year annual returns
    ann_ret_dict: dict = {}
    for t, px in all_px.items():
        ann_ret_dict[t] = _annual_total_return(px)

    # Determine the year range from available data
    all_years_set: set = set()
    for t, ann_ret in ann_ret_dict.items():
        all_years_set.update(ann_ret.index.tolist())
    all_years = sorted(y for y in all_years_set if y >= start_year)

    if not all_years:
        return {
            "initiation_mask": pd.DataFrame(),
            "control_mask": pd.DataFrame(),
            "fwd_ret": pd.DataFrame(),
            "ew_ret": pd.Series(dtype=float),
            "tickers": [],
            "initiation_years": {},
        }

    init_rows: dict = {}
    ctrl_rows: dict = {}
    fwd_rows: dict = {}

    for y in all_years:
        init_rows[y] = {}
        ctrl_rows[y] = {}
        fwd_rows[y] = {}
        for t in all_px:
            ann_ret = ann_ret_dict[t]
            fwd_yr = y + 1
            if fwd_yr not in ann_ret.index:
                continue
            fwd_rows[y][t] = float(ann_ret[fwd_yr])
            init_yr = initiation_years.get(t)
            if init_yr == y:
                init_rows[y][t] = True
                ctrl_rows[y][t] = False
            elif init_yr is None or init_yr > y:
                # never paid through year y → control
                init_rows[y][t] = False
                ctrl_rows[y][t] = True
            else:
                # already paid before year y (established payer)
                init_rows[y][t] = False
                ctrl_rows[y][t] = False

    # Drop years with no forward returns
    valid_years = [y for y in all_years if fwd_rows.get(y)]
    if not valid_years:
        return {
            "initiation_mask": pd.DataFrame(),
            "control_mask": pd.DataFrame(),
            "fwd_ret": pd.DataFrame(),
            "ew_ret": pd.Series(dtype=float),
            "tickers": list(all_px.keys()),
            "initiation_years": initiation_years,
        }

    initiation_mask = pd.DataFrame(init_rows).T.reindex(valid_years).fillna(False).astype(bool)
    initiation_mask.index.name = "year"
    control_mask = pd.DataFrame(ctrl_rows).T.reindex(valid_years).fillna(False).astype(bool)
    control_mask.index.name = "year"
    fwd_ret = pd.DataFrame(fwd_rows).T.reindex(valid_years)
    fwd_ret.index.name = "year"
    ew_ret = fwd_ret.mean(axis=1).rename("ew")

    return {
        "initiation_mask": initiation_mask,
        "control_mask": control_mask,
        "fwd_ret": fwd_ret,
        "ew_ret": ew_ret,
        "tickers": list(all_px.keys()),
        "initiation_years": initiation_years,
    }


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(panel: dict) -> str:
    """A short content fingerprint of the real panel, for the as-of stamp."""
    h = hashlib.sha1()
    for key in ("initiation_mask", "control_mask", "fwd_ret"):
        df = panel.get(key)
        if df is not None and not df.empty:
            arr = np.ascontiguousarray(np.nan_to_num(df.to_numpy(dtype=float)))
            h.update(arr.tobytes())
    return h.hexdigest()[:12]
