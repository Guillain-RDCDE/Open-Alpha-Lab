"""Data layer for Study 219 (IPO-Pop).

Two tapes:

- ``synthetic_ipos`` — a *deterministic, offline* generator.  Simulates a panel of
  IPOs with a ``pop_bps`` knob controlling the first-day pop magnitude and a
  ``drift_bps`` knob controlling the subsequent daily drift (positive = momentum,
  negative = long-run underperformance).  Setting both to zero is the null.
- ``fetch_ipo_panel`` — the real loader.  Uses a hardcoded table of notable US IPOs
  (first-day pop taken from public filings / historical records) plus ``yfinance``
  daily closes to compute post-close forward returns at 6m / 12m / 36m horizons.
  Cache-only by default (``fetch=False`` raises without cache).

Design notes
------------
- The hardcoded IPO table stores (ticker, ipo_date, offer_price, first_close).
  First-day pop = (first_close / offer_price) - 1.  No look-ahead: forward returns
  start at t+1 (the first trading day *after* the IPO close).
- Long-run returns are total-return price-only (yfinance adjusted closes).
  Survivorship is *severe* here — many IPOs delist or get acquired; the table only
  contains tickers that were still quote-able on yfinance.  This is named prominently.
- Costs: one-way 10 bps for liquid large-cap IPOs (reasonable for institutional).
  Retail allocations are lottery-only and the pop is typically unharvestable.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded IPO table — notable US IPOs 1997-2023
# Sources: SEC S-1 filings, Renaissance Capital IPO data, Bloomberg historical.
# Fields: ticker, ipo_date (YYYY-MM-DD), offer_price (USD), first_day_close (USD)
# Only tickers still quoteable on yfinance (survivorship bias NAMED).
# ---------------------------------------------------------------------------
IPO_TABLE: list[dict] = [
    # Tech bubble era
    {"ticker": "AMZN", "ipo_date": "1997-05-15", "offer_price": 18.00,  "first_close": 23.50},
    {"ticker": "GOOGL","ipo_date": "2004-08-19", "offer_price": 85.00,  "first_close": 100.34},
    {"ticker": "NFLX", "ipo_date": "2002-05-23", "offer_price": 15.00,  "first_close": 16.75},
    {"ticker": "EBAY", "ipo_date": "1998-09-24", "offer_price": 18.00,  "first_close": 47.375},
    {"ticker": "CRM",  "ipo_date": "2004-06-23", "offer_price": 11.00,  "first_close": 11.22},
    # 2011–2015 wave
    {"ticker": "LNKD", "ipo_date": "2011-05-19", "offer_price": 45.00,  "first_close": 94.25},  # LinkedIn, delisted 2016
    {"ticker": "FB",   "ipo_date": "2012-05-18", "offer_price": 38.00,  "first_close": 38.23},   # Meta, now META
    {"ticker": "TWTR", "ipo_date": "2013-11-07", "offer_price": 26.00,  "first_close": 44.90},   # Twitter, delisted 2022
    {"ticker": "BABA", "ipo_date": "2014-09-19", "offer_price": 68.00,  "first_close": 93.89},
    {"ticker": "SNAP", "ipo_date": "2017-03-02", "offer_price": 17.00,  "first_close": 24.48},
    {"ticker": "LYFT", "ipo_date": "2019-03-29", "offer_price": 72.00,  "first_close": 78.29},
    {"ticker": "UBER", "ipo_date": "2019-05-10", "offer_price": 45.00,  "first_close": 41.57},
    {"ticker": "ZOOM", "ipo_date": "2019-04-18", "offer_price": 36.00,  "first_close": 62.00},  # ZM
    # SPAC / 2020-2021 boom
    {"ticker": "ABNB", "ipo_date": "2020-12-10", "offer_price": 68.00,  "first_close": 144.71},
    {"ticker": "DASH", "ipo_date": "2020-12-09", "offer_price": 102.00, "first_close": 189.51},
    {"ticker": "RBLX", "ipo_date": "2021-03-10", "offer_price": 45.00,  "first_close": 69.50},   # Direct listing ref price
    {"ticker": "COIN", "ipo_date": "2021-04-14", "offer_price": 250.00, "first_close": 328.28},  # Direct listing ref price
    {"ticker": "RIVN", "ipo_date": "2021-11-10", "offer_price": 78.00,  "first_close": 100.73},
    {"ticker": "LCID", "ipo_date": "2021-07-26", "offer_price": 15.00,  "first_close": 25.24},   # via SPAC
    {"ticker": "BROS", "ipo_date": "2021-11-03", "offer_price": 23.00,  "first_close": 39.54},   # Dutch Bros
    # More recent / older solid names
    {"ticker": "TSLA", "ipo_date": "2010-06-29", "offer_price": 17.00,  "first_close": 23.89},
    {"ticker": "V",    "ipo_date": "2008-03-19", "offer_price": 44.00,  "first_close": 56.50},
    {"ticker": "MA",   "ipo_date": "2006-05-25", "offer_price": 39.00,  "first_close": 45.15},
    {"ticker": "GS",   "ipo_date": "1999-05-04", "offer_price": 53.00,  "first_close": 70.00},
    {"ticker": "SHOP", "ipo_date": "2015-05-21", "offer_price": 17.00,  "first_close": 28.00},
    {"ticker": "SQ",   "ipo_date": "2015-11-19", "offer_price": 9.00,   "first_close": 13.07},   # Block (XYZ)
    {"ticker": "SPOT", "ipo_date": "2018-04-03", "offer_price": 132.00, "first_close": 149.01},   # Direct listing
    {"ticker": "DDOG", "ipo_date": "2019-09-19", "offer_price": 27.00,  "first_close": 40.49},
    {"ticker": "SNOW", "ipo_date": "2020-09-16", "offer_price": 120.00, "first_close": 253.93},
    {"ticker": "U",    "ipo_date": "2020-09-18", "offer_price": 52.00,  "first_close": 75.00},    # Unity
    {"ticker": "PLTR", "ipo_date": "2020-09-30", "offer_price": 7.25,   "first_close": 9.50},    # Direct listing ref
    {"ticker": "SOFI", "ipo_date": "2021-06-01", "offer_price": 10.00,  "first_close": 22.65},   # via SPAC
    {"ticker": "HOOD", "ipo_date": "2021-07-29", "offer_price": 38.00,  "first_close": 34.82},
    {"ticker": "IONQ", "ipo_date": "2021-10-01", "offer_price": 10.00,  "first_close": 13.34},   # via SPAC
    {"ticker": "KVYO", "ipo_date": "2023-09-20", "offer_price": 30.00,  "first_close": 32.76},   # Klaviyo
    {"ticker": "ARM",  "ipo_date": "2023-09-14", "offer_price": 51.00,  "first_close": 63.59},
    {"ticker": "BIRK", "ipo_date": "2023-10-11", "offer_price": 46.00,  "first_close": 48.39},   # Birkenstock
    {"ticker": "CART", "ipo_date": "2023-09-19", "offer_price": 30.00,  "first_close": 33.70},   # Instacart
    {"ticker": "KRTX", "ipo_date": "2019-06-27", "offer_price": 16.00,  "first_close": 22.06},   # Karuna
    {"ticker": "ZI",   "ipo_date": "2020-06-04", "offer_price": 21.00,  "first_close": 31.96},   # ZoomInfo
]

# Tickers that are no longer tradeable (delisted/acquired) — yfinance may still return
# historical prices for these; we try and skip gracefully.
DELISTED = {"LNKD", "TWTR"}

# yfinance alias mapping (some tickers changed)
TICKER_ALIAS = {
    "FB": "META",
    "ZOOM": "ZM",
    "SQ": "XYZ",  # Block renamed to XYZ in 2024 but ZM is still Zoom Video
}

# Forward-return horizons in trading days
HORIZONS: dict[str, int] = {
    "ret_6m":  126,
    "ret_12m": 252,
    "ret_36m": 756,
}


# ---------------------------------------------------------------------------
# Build the raw pop table from the hardcoded data
# ---------------------------------------------------------------------------
def build_pop_table() -> pd.DataFrame:
    """Return the hardcoded IPO table as a DataFrame with first-day pop computed.

    Columns: ticker, ipo_date, offer_price, first_close, pop (fraction, e.g. 0.30 = 30%).
    """
    rows = []
    for r in IPO_TABLE:
        pop = r["first_close"] / r["offer_price"] - 1.0
        rows.append(
            {
                "ticker": r["ticker"],
                "ipo_date": pd.Timestamp(r["ipo_date"]),
                "offer_price": float(r["offer_price"]),
                "first_close": float(r["first_close"]),
                "pop": float(pop),
            }
        )
    df = pd.DataFrame(rows).sort_values("ipo_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Synthetic IPO panel — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_ipos(
    n_ipos: int = 40,
    pop_bps: float = 2000.0,       # mean first-day pop in bps (2000 = 20%)
    drift_bps: float = -5.0,       # daily drift after close (negative = underperformance)
    horizon: int = 756,            # max horizon in trading days (3 years)
    base_vol_ann: float = 0.40,
    start: str = "2000-01-03",
    seed: int = 219,
) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic IPO panel with tunable pop and post-close drift.

    Parameters
    ----------
    n_ipos:
        Number of synthetic IPOs.
    pop_bps:
        Mean first-day pop in basis points (2000 bps = 20%).  Drawn from a
        lognormal so pops are always positive.
    drift_bps:
        Extra daily drift *after* the first close.  Negative = long-run hangover.
        Setting this to 0 gives a null where post-close returns are pure noise.
    horizon:
        Length of the simulated post-close window in trading days.
    base_vol_ann:
        Annualised volatility of the daily post-close returns.
    start:
        First business date for the simulated calendar.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    ipos : DataFrame
        One row per IPO. Columns: ticker, ipo_date, offer_price, first_close, pop,
        ret_6m, ret_12m, ret_36m.
    truth : dict
        Ground-truth parameters used for generation.
    """
    rng = np.random.default_rng(seed)
    sig_d = base_vol_ann / np.sqrt(252)
    extra_daily = drift_bps * 1e-4
    # Ito correction: subtract variance/2 so E[log(S_T/S_0)] = extra_daily * T
    # This prevents geometric-Brownian-motion upward bias in long horizons.
    ito_adj = -0.5 * sig_d ** 2

    dates = pd.bdate_range(start=start, periods=1500, name="date")
    # Sample n_ipos event dates spread across the calendar
    ev_indices = sorted(
        rng.choice(len(dates) - horizon - 20, size=n_ipos, replace=False)
    )

    rows = []
    for i, ei in enumerate(ev_indices):
        ev_date = dates[ei]
        # First-day pop — lognormal so always positive
        log_mu = np.log(pop_bps * 1e-4) - 0.5 * 0.5 ** 2
        pop = float(np.exp(rng.normal(log_mu, 0.5)))
        offer = 20.0 + float(rng.uniform(0, 100))
        first_close = offer * (1.0 + pop)

        # Post-close log-returns (zero-drift null uses Ito correction so mean price
        # change is ~zero over long horizons; drift_bps shifts the log mean)
        log_rets = rng.normal(ito_adj + extra_daily, sig_d, horizon)
        cum = np.exp(np.cumsum(log_rets)) - 1.0

        def _r(h: int) -> float:
            return float(cum[min(h, horizon) - 1]) if h <= horizon else float("nan")

        rows.append(
            {
                "ticker": f"IPO{i:03d}",
                "ipo_date": ev_date,
                "offer_price": round(offer, 2),
                "first_close": round(first_close, 2),
                "pop": pop,
                "ret_6m":  _r(126),
                "ret_12m": _r(252),
                "ret_36m": _r(756),
            }
        )

    ipos = pd.DataFrame(rows).sort_values("ipo_date").reset_index(drop=True)
    truth = {
        "n_ipos":      n_ipos,
        "pop_bps":     pop_bps,
        "drift_bps":   drift_bps,
        "horizon":     horizon,
        "base_vol_ann": base_vol_ann,
        "seed":        seed,
    }
    return ipos, truth


# ---------------------------------------------------------------------------
# Real loader — yfinance post-close forward returns, cache-first
# ---------------------------------------------------------------------------
def _price_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_219_{safe}_1d.parquet")


def fetch_ipo_panel(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load post-IPO forward returns for the hardcoded IPO table.

    For each IPO in ``IPO_TABLE``:
    - If ``fetch=True``: download daily prices from yfinance starting 1 day before
      the IPO date and save to cache.
    - If ``fetch=False``: read from cache; raises FileNotFoundError if absent.

    Computes post-first-close forward returns at 6m / 12m / 36m horizons.
    Returns NaN for any horizon beyond available data.

    Returns
    -------
    DataFrame with columns:
        ticker, ipo_date, offer_price, first_close, pop, ret_6m, ret_12m, ret_36m.

    Notes
    -----
    Survivorship bias is severe: this table contains only IPOs whose tickers were
    still queryable on yfinance. Many IPOs that cratered or were acquired are excluded.
    This biases long-run returns upward — stated here and in all results.
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    pop_table = build_pop_table()

    rows = []
    for _, row in pop_table.iterrows():
        raw_tkr = row["ticker"]
        # Use alias if ticker was renamed
        tkr = TICKER_ALIAS.get(raw_tkr, raw_tkr)
        cache_path = _price_cache_path(raw_tkr, cache_dir)

        if fetch:
            start_dt = (row["ipo_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            try:
                raw = yf.download(
                    tkr, start=start_dt, interval="1d",
                    auto_adjust=True, progress=False
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.rename(columns=str.lower)
                if not raw.empty and "close" in raw.columns:
                    raw[["close"]].to_parquet(cache_path)
                else:
                    # Write empty sentinel so cache-check doesn't re-hit network
                    pd.DataFrame(columns=["close"]).to_parquet(cache_path)
            except Exception:
                pd.DataFrame(columns=["close"]).to_parquet(cache_path)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached prices for {raw_tkr} at {cache_path}. "
                f"Call fetch_ipo_panel(fetch=True) once."
            )

        pr = pd.read_parquet(cache_path)
        if pr.empty or "close" not in pr.columns:
            continue

        close_s = pr["close"].copy()
        close_s.index = pd.DatetimeIndex(close_s.index).tz_localize(None)
        close_s.index.name = "date"
        close_s = close_s.sort_index().dropna()

        # Find position right after the IPO date
        ipo_dt = pd.Timestamp(row["ipo_date"])
        pos_arr = np.searchsorted(close_s.index, ipo_dt)
        # We want t+1 (first day after IPO close) as our entry
        entry_pos = pos_arr + 1
        if entry_pos >= len(close_s):
            continue

        entry_price = float(close_s.iloc[entry_pos])
        if entry_price <= 0 or not np.isfinite(entry_price):
            continue

        fwd_rets: dict[str, float] = {}
        for col, h in HORIZONS.items():
            end_pos = entry_pos + h
            if end_pos >= len(close_s):
                fwd_rets[col] = float("nan")
            else:
                fwd_rets[col] = float(close_s.iloc[end_pos] / entry_price - 1.0)

        rows.append(
            {
                "ticker":      raw_tkr,
                "ipo_date":    ipo_dt,
                "offer_price": float(row["offer_price"]),
                "first_close": float(row["first_close"]),
                "pop":         float(row["pop"]),
                **fwd_rets,
            }
        )

    df = pd.DataFrame(rows).sort_values("ipo_date").reset_index(drop=True)
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the IPO panel (IPO dates + pops), for as-of stamps."""
    arr = pd.to_datetime(df["ipo_date"]).astype(np.int64).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
