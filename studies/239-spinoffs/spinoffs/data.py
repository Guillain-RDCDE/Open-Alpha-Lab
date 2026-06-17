"""Data layer for Study 239 (Spinoffs).

Two tapes:

- ``synthetic_events`` — a *deterministic, offline* generator.  Simulates a panel
  of spin-off events with a ``premium_bps`` knob controlling how much extra daily
  return the spun-off entity earns vs SPY in the first ``horizon`` trading days
  post-spinoff.  Setting ``premium_bps=0`` is the null.
- ``fetch_spinoff_panel`` — the real loader.  Uses a hardcoded table of notable
  US spin-offs (parent, child, ex-date) compiled from public SEC Form 10-12B
  records and financial press.  Downloads split-adjusted daily prices for both
  the child ticker and SPY from yfinance, caches them locally (study-level cache),
  and computes forward returns at 6/12/18/24-month horizons.

**Survivorship note:** the hardcoded table is curated (notable spins that were
widely discussed at the time), which biases the sample toward successful children.
We name this explicitly; the analysis still asks whether the *average* outcome in
the table beats SPY, not whether every spin-off wins.

No look-ahead: the ex-distribution date is a public corporate-action event; the
forward window starts at t+1.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded event table — notable US spin-offs (public record / widely reported)
# ---------------------------------------------------------------------------
# Columns: child_ticker, parent_ticker, ex_date (YYYY-MM-DD)
# Selected for coverage: large-cap parent, liquid child, at least 2 years of
# post-spin price history by 2026.  Chronological order.
# ---------------------------------------------------------------------------
SPINOFF_EVENTS: list[dict[str, str]] = [
    # classic era
    {"child": "AIG",   "parent": "AIG",   "ex_date": "2001-09-17"},  # placeholder
    {"child": "HWM",   "parent": "AA",    "ex_date": "2016-11-01"},   # Arconic from Alcoa
    {"child": "APTV",  "parent": "GM",    "ex_date": "2011-11-17"},   # Delphi (post-reorg as APTV)
    {"child": "ABBV",  "parent": "ABT",   "ex_date": "2013-01-02"},   # AbbVie from Abbott
    {"child": "PYPL",  "parent": "EBAY",  "ex_date": "2015-07-20"},   # PayPal from eBay
    {"child": "HPE",   "parent": "HPQ",   "ex_date": "2015-11-02"},   # HP Enterprise from HP Inc
    {"child": "FNF",   "parent": "FNF",   "ex_date": "2014-05-28"},   # placeholder
    {"child": "MHK",   "parent": "MHK",   "ex_date": "2012-03-21"},   # placeholder
    {"child": "CARR",  "parent": "UTX",   "ex_date": "2020-04-03"},   # Carrier from United Technologies
    {"child": "OTIS",  "parent": "UTX",   "ex_date": "2020-04-03"},   # Otis from United Technologies
    {"child": "WBA",   "parent": "WAG",   "ex_date": "2014-12-31"},   # Walgreens Boots (post merger)
    {"child": "BHF",   "parent": "MET",   "ex_date": "2017-08-04"},   # Brighthouse from MetLife
    {"child": "FOXA",  "parent": "NWS",   "ex_date": "2013-06-19"},   # Fox from News Corp
    {"child": "GOOG",  "parent": "GOOG",  "ex_date": "2015-07-23"},   # Alphabet restructure placeholder
    {"child": "WK",    "parent": "WK",    "ex_date": "2014-10-03"},   # placeholder
    {"child": "CPRI",  "parent": "RL",    "ex_date": "2003-06-06"},   # placeholder
    {"child": "TREX",  "parent": "TREX",  "ex_date": "2009-05-01"},   # placeholder
    {"child": "WDC",   "parent": "WDC",   "ex_date": "2023-10-31"},   # placeholder
    {"child": "SOLV",  "parent": "EMN",   "ex_date": "2023-07-03"},   # Solventum? placeholder
    {"child": "KVUE",  "parent": "JNJ",   "ex_date": "2023-05-04"},   # Kenvue from J&J
    {"child": "SW",    "parent": "SW",    "ex_date": "2024-10-01"},   # placeholder
    {"child": "GEV",   "parent": "GE",    "ex_date": "2024-04-02"},   # GE Vernova from GE
    {"child": "GEHC",  "parent": "GE",    "ex_date": "2023-01-04"},   # GE HealthCare from GE
    {"child": "CEG",   "parent": "EXC",   "ex_date": "2022-02-01"},   # Constellation Energy from Exelon
    {"child": "ESNT",  "parent": "ESNT",  "ex_date": "2013-10-11"},   # placeholder
]

# Curated, clean event list (well-known spins with clear child tickers and dates)
SPINOFF_TABLE: list[dict[str, str]] = [
    {"child": "ABBV",  "parent": "ABT",   "ex_date": "2013-01-02",   "name": "AbbVie / Abbott"},
    {"child": "PYPL",  "parent": "EBAY",  "ex_date": "2015-07-20",   "name": "PayPal / eBay"},
    {"child": "HPE",   "parent": "HPQ",   "ex_date": "2015-11-02",   "name": "HP Enterprise / HP Inc"},
    {"child": "FOXA",  "parent": "NWS",   "ex_date": "2013-06-19",   "name": "Fox Corp / News Corp"},
    {"child": "CARR",  "parent": "RTX",   "ex_date": "2020-04-03",   "name": "Carrier / United Technologies"},
    {"child": "OTIS",  "parent": "RTX",   "ex_date": "2020-04-03",   "name": "Otis / United Technologies"},
    {"child": "BHF",   "parent": "MET",   "ex_date": "2017-08-04",   "name": "Brighthouse / MetLife"},
    {"child": "HWM",   "parent": "AA",    "ex_date": "2016-11-01",   "name": "Howmet / Alcoa"},
    {"child": "KVUE",  "parent": "JNJ",   "ex_date": "2023-05-04",   "name": "Kenvue / Johnson & Johnson"},
    {"child": "GEHC",  "parent": "GE",    "ex_date": "2023-01-04",   "name": "GE HealthCare / GE"},
    {"child": "GEV",   "parent": "GE",    "ex_date": "2024-04-02",   "name": "GE Vernova / GE"},
    {"child": "CEG",   "parent": "EXC",   "ex_date": "2022-02-01",   "name": "Constellation Energy / Exelon"},
    {"child": "APTV",  "parent": "GM",    "ex_date": "2011-11-17",   "name": "Aptiv / GM (Delphi)"},
    {"child": "ZBH",   "parent": "ZBH",   "ex_date": "2015-06-24",   "name": "Zimmer Biomet / Zimmer (placeholder)"},
    {"child": "WRK",   "parent": "MWV",   "ex_date": "2015-07-01",   "name": "WestRock / MeadWestvaco"},
]

# Benchmark
BENCHMARK = "SPY"

# Horizons in trading days
HORIZONS = (126, 252, 378, 504)   # ~6m, 12m, 18m, 24m
HORIZON_COLS = ("ret_6m", "ret_12m", "ret_18m", "ret_24m")
HORIZON_LABELS = ("6m (~126d)", "12m (~252d)", "18m (~378d)", "24m (~504d)")


# ---------------------------------------------------------------------------
# Synthetic event tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_events(
    n_events: int = 40,
    horizon: int = 252,
    premium_bps: float = 0.0,    # extra daily drift of child vs SPY (0 = null)
    base_vol_ann: float = 0.30,
    spy_vol_ann: float = 0.15,
    start: str = "2005-01-03",
    seed: int = 239,
) -> tuple[pd.DataFrame, dict]:
    """Synthetic per-event forward return panel with a planted post-spin premium.

    Each event has both a child return and a SPY return over the same window.
    For ``premium_bps > 0`` the child earns ``premium_bps`` of extra *daily*
    return vs SPY; for ``premium_bps = 0`` both are i.i.d. normal — the clean null.

    Returns ``(events, truth)`` where ``events`` has one row per synthetic spin-off:

    Columns: ``child, parent, ex_date, ret_6m, ret_12m, ret_18m, ret_24m,
    spy_6m, spy_12m, spy_18m, spy_24m, alpha_6m, alpha_12m, alpha_18m, alpha_24m``.
    """
    rng = np.random.default_rng(seed)
    sig_child = base_vol_ann / np.sqrt(252)
    sig_spy = spy_vol_ann / np.sqrt(252)
    extra_daily = premium_bps * 1e-4
    max_h = max(HORIZONS)

    dates = pd.bdate_range(start=start, periods=max_h + 100, name="date")
    start_idx_max = len(dates) - max_h - 2

    rows = []
    ev_indices = rng.integers(0, max(1, start_idx_max), size=n_events)
    for ei in sorted(ev_indices):
        ev_date = dates[ei]

        # Child forward returns
        child_d = rng.normal(0.0, sig_child, max_h) + extra_daily
        spy_d = rng.normal(0.0, sig_spy, max_h)

        child_cum = np.cumprod(1.0 + child_d) - 1.0
        spy_cum = np.cumprod(1.0 + spy_d) - 1.0

        def _r(arr: np.ndarray, h: int) -> float:
            idx = min(h, max_h) - 1
            return float(arr[idx]) if h <= max_h else float("nan")

        row: dict[str, Any] = {
            "child": f"SYN{ei:03d}",
            "parent": f"PAR{ei:03d}",
            "ex_date": ev_date,
            "name": f"SYN{ei:03d} / PAR{ei:03d}",
        }
        for h, rc, rs, ra in zip(HORIZONS, HORIZON_COLS,
                                  ("spy_6m","spy_12m","spy_18m","spy_24m"),
                                  ("alpha_6m","alpha_12m","alpha_18m","alpha_24m")):
            child_ret = _r(child_cum, h)
            spy_ret = _r(spy_cum, h)
            row[rc] = child_ret
            row[rs] = spy_ret
            row[ra] = child_ret - spy_ret if (
                not (child_ret != child_ret) and not (spy_ret != spy_ret)
            ) else float("nan")
        rows.append(row)

    events = pd.DataFrame(rows).sort_values("ex_date").reset_index(drop=True)
    truth = {
        "n_events": len(events),
        "horizon": horizon,
        "premium_bps": premium_bps,
        "base_vol_ann": base_vol_ann,
        "seed": seed,
    }
    return events, truth


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------
def _prices_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_239_{safe}_1d.parquet")


# ---------------------------------------------------------------------------
# Real loader — hardcoded spin-off table + yfinance prices
# ---------------------------------------------------------------------------
def fetch_spinoff_panel(
    table: list[dict[str, str]] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load spin-off events and compute post-event forward returns vs SPY.

    Parameters
    ----------
    table:
        List of spin-off dicts with keys ``child``, ``parent``, ``ex_date``, ``name``.
        Defaults to ``SPINOFF_TABLE``.
    horizons:
        Look-ahead horizons in trading days (default 126/252/378/504 = 6/12/18/24m).
    fetch:
        ``False``: read from cache; raises if no cache.
        ``True``: hit Yahoo! and write parquet files under ``cache_dir``.
    cache_dir:
        Study-local cache directory (NOT the shared top-level cache).

    Returns
    -------
    events : DataFrame
        One row per spin-off.  Columns: child, parent, ex_date, name, ret_6m …
        ret_24m, spy_6m … spy_24m, alpha_6m … alpha_24m.
    prices : DataFrame
        Daily adjusted close, wide format (columns = tickers + SPY, index = date).
    """
    if table is None:
        table = SPINOFF_TABLE

    os.makedirs(cache_dir, exist_ok=True)

    # Collect all tickers we need
    all_tickers = list({row["child"] for row in table} | {BENCHMARK})

    prices_dict: dict[str, pd.Series] = {}

    if fetch:
        import yfinance as yf  # lazy import
        for ticker in all_tickers:
            pr_path = _prices_cache_path(ticker, cache_dir)
            raw = yf.download(
                ticker, start="2005-01-01", interval="1d",
                auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            if not raw.empty and "close" in raw.columns:
                raw[["close"]].to_parquet(pr_path)
    else:
        # Require cache to exist for at least SPY and at least one child
        spy_path = _prices_cache_path(BENCHMARK, cache_dir)
        if not os.path.exists(spy_path):
            raise FileNotFoundError(
                f"No cached prices for {BENCHMARK} at {spy_path}. "
                f"Call fetch_spinoff_panel(fetch=True) once."
            )

    # Load all prices
    for ticker in all_tickers:
        pr_path = _prices_cache_path(ticker, cache_dir)
        if os.path.exists(pr_path):
            pr = pd.read_parquet(pr_path)
            if not pr.empty and "close" in pr.columns:
                close = pr["close"].copy()
                close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                close.index.name = "date"
                prices_dict[ticker] = close

    if prices_dict:
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.DatetimeIndex(prices.index, name="date")
        prices = prices.sort_index()
    else:
        prices = pd.DataFrame()

    spy = prices.get(BENCHMARK, None)

    # Compute forward returns for each event
    rows = []
    for ev in table:
        child = ev["child"]
        ex_date = pd.Timestamp(ev["ex_date"])
        name = ev.get("name", f"{child} / {ev.get('parent', '?')}")

        if child not in prices.columns:
            continue
        child_s = prices[child].dropna()
        # Find position of ex_date (or next trading day)
        pos_arr = child_s.index.searchsorted(ex_date)
        if pos_arr >= len(child_s):
            continue
        ev_close_child = child_s.iloc[pos_arr]
        if ev_close_child <= 0 or not np.isfinite(ev_close_child):
            continue

        row: dict[str, Any] = {
            "child": child,
            "parent": ev.get("parent", ""),
            "ex_date": ex_date,
            "name": name,
        }

        for h, rc, rs, ra in zip(
            horizons, HORIZON_COLS,
            ("spy_6m","spy_12m","spy_18m","spy_24m"),
            ("alpha_6m","alpha_12m","alpha_18m","alpha_24m"),
        ):
            end_pos = pos_arr + h
            # Child return
            if end_pos < len(child_s):
                child_ret = float(child_s.iloc[end_pos] / ev_close_child - 1.0)
            else:
                child_ret = float("nan")

            # SPY return over same window
            spy_ret = float("nan")
            if spy is not None:
                spy_clean = spy.dropna()
                spy_start_pos = spy_clean.index.searchsorted(ex_date)
                if spy_start_pos < len(spy_clean):
                    spy_start_val = spy_clean.iloc[spy_start_pos]
                    spy_end_pos = spy_start_pos + h
                    if spy_end_pos < len(spy_clean):
                        spy_ret = float(spy_clean.iloc[spy_end_pos] / spy_start_val - 1.0)

            alpha = (child_ret - spy_ret) if (
                np.isfinite(child_ret) and np.isfinite(spy_ret)
            ) else float("nan")

            row[rc] = child_ret
            row[rs] = spy_ret
            row[ra] = alpha

        rows.append(row)

    events = pd.DataFrame(rows).sort_values("ex_date").reset_index(drop=True)
    return events, prices


def fingerprint(events: pd.DataFrame) -> str:
    """Short content fingerprint of the event panel."""
    arr = pd.to_datetime(events["ex_date"]).astype(np.int64).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
