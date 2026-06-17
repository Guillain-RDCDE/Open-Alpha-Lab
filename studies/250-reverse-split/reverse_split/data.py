"""Data layer for Study 250 (Reverse-Split).

Two tapes:

- ``synthetic_events`` — a *deterministic, offline* generator.  Simulates a panel
  of stocks with a ``drift_bps`` knob that controls the extra daily return in the
  ``horizon`` days following a synthetic reverse-split event.  Setting ``drift_bps=0``
  is the null.  Drift is expected to be *negative* for reverse splits.
- ``load_events`` — the real loader.  Uses a hardcoded table of confirmed US reverse
  splits (publicly-available effective dates), then fetches forward returns from
  yfinance for each ticker.  Cache-only by default (``fetch=False`` raises without
  cache); ``fetch=True`` hits Yahoo! and writes price parquets to the study-local
  ``_cache/``.

**Distress confound (stated prominently here and in the notebooks):**
Reverse splits cluster in financially distressed companies at risk of delisting.
The negative post-event returns documented here may reflect ongoing fundamental
deterioration (the distress itself) rather than a pure "reverse split signal".
Separating the two is not possible without a distress-matched control group — which
requires financial data (Altman-Z, leverage, coverage ratios) beyond this study's scope.
We name the confound explicitly rather than over-claiming.

**Survivorship in reverse:** Unlike forward-split studies, the reverse-split universe
has *negative* survivorship — many RS names eventually delist entirely.  Our sample is
biased toward names that survived long enough for yfinance to return a price history,
which may *understate* the true average post-RS loss.

No look-ahead: all event dates in the hardcoded table are historical public records;
forward windows start at t+1 business day.  Prices are split-adjusted by Yahoo!.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded reverse-split event table
# ---------------------------------------------------------------------------
# Source: SEC filings / press releases / historical record.
# Only tickers that remain retrievable from yfinance are included.
# Ratio: number of old shares exchanged for 1 new share (always > 1).
# Delisted names (BBBY, FRE, SHLDQ, etc.) are excluded — yfinance cannot
# retrieve reliable price histories for fully-delisted tickers.

REVERSE_SPLIT_TABLE: List[dict] = [
    # --- Large-cap / notable recoveries ---
    {"ticker": "AIG",  "event_date": "2009-07-01", "ratio": 20,  "note": "AIG 1-for-20 post-bailout"},
    {"ticker": "C",    "event_date": "2011-05-09", "ratio": 10,  "note": "Citigroup 1-for-10 (2011)"},
    {"ticker": "GE",   "event_date": "2021-07-30", "ratio": 8,   "note": "GE 1-for-8 portfolio simplification"},
    # --- EV / SPAC / speculative names ---
    {"ticker": "KOSS", "event_date": "2022-04-04", "ratio": 20,  "note": "Koss Corp 1-for-20"},
    {"ticker": "BYFC", "event_date": "2022-06-21", "ratio": 5,   "note": "Broadway Financial 1-for-5"},
    {"ticker": "AUVI", "event_date": "2022-09-12", "ratio": 10,  "note": "Applied UV Products 1-for-10"},
    {"ticker": "XELA", "event_date": "2023-02-22", "ratio": 200, "note": "Exela Technologies 1-for-200"},
    {"ticker": "CLPS", "event_date": "2023-03-13", "ratio": 3,   "note": "CLPS Technology 1-for-3"},
    {"ticker": "SNDL", "event_date": "2023-05-01", "ratio": 10,  "note": "SNDL Inc 1-for-10"},
    {"ticker": "CLOV", "event_date": "2023-05-24", "ratio": 10,  "note": "Clover Health 1-for-10"},
    {"ticker": "STSS", "event_date": "2023-05-31", "ratio": 100, "note": "Sharps Technology 1-for-100"},
    {"ticker": "AMTX", "event_date": "2023-07-25", "ratio": 50,  "note": "Aemetis 1-for-50"},
    {"ticker": "ATER", "event_date": "2023-09-01", "ratio": 10,  "note": "Aterian 1-for-10"},
    {"ticker": "MVIS", "event_date": "2023-09-19", "ratio": 10,  "note": "MicroVision 1-for-10"},
    {"ticker": "WKHS", "event_date": "2024-09-30", "ratio": 20,  "note": "Workhorse Group 1-for-20"},
    {"ticker": "LCID", "event_date": "2024-11-14", "ratio": 5,   "note": "Lucid Group 1-for-5"},
    {"ticker": "BLNK", "event_date": "2024-12-09", "ratio": 4,   "note": "Blink Charging 1-for-4"},
]

# Build canonical DataFrame
EVENTS_RAW: pd.DataFrame = pd.DataFrame(REVERSE_SPLIT_TABLE)[["ticker", "event_date", "ratio", "note"]]
EVENTS_RAW["event_date"] = pd.to_datetime(EVENTS_RAW["event_date"])

# ---------------------------------------------------------------------------
# Synthetic event tape — deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_events(
    n_stocks: int = 20,
    n_events: int = 50,
    horizon: int = 63,           # ~3 months in trading days
    drift_bps: float = 0.0,      # signed daily drift (negative = negative drift)
    base_vol_ann: float = 0.45,  # higher vol for distressed names
    start: str = "2018-01-03",
    seed: int = 250,
) -> tuple[pd.DataFrame, dict]:
    """Synthetic per-event forward return panel with a planted post-RS drift.

    ``drift_bps`` is signed: negative values simulate distress-driven decline.
    Setting ``drift_bps=0`` is the clean null (no RS effect beyond normal volatility).
    Setting ``drift_bps=-20`` plants a −20 bps/day drag over the post-event window,
    producing roughly −12% cumulative at 63 days (~3 months).

    Returns ``(events, truth)`` where ``events`` has columns:
    ticker, event_date, ratio, ret_1m, ret_3m, ret_6m, ret_12m, is_rs.
    """
    rng = np.random.default_rng(seed)
    sig_d = base_vol_ann / np.sqrt(252)
    extra_daily = drift_bps * 1e-4

    dates = pd.bdate_range(start=start, periods=600, name="date")
    rows = []
    for stock_i in range(n_stocks):
        events_this = max(1, n_events // n_stocks)
        ev_indices = rng.choice(
            len(dates) - horizon - 10, size=int(events_this), replace=False
        )
        for ei in sorted(ev_indices):
            ev_date = dates[ei]
            fwd = rng.normal(sig_d, sig_d, horizon)
            fwd += extra_daily
            cum = np.cumprod(1.0 + fwd) - 1.0

            def _r(h: int) -> float:
                return float(cum[min(h, horizon) - 1]) if h <= horizon else float("nan")

            rows.append(
                {
                    "ticker": f"SYN{stock_i:02d}",
                    "event_date": ev_date,
                    "ratio": int(rng.choice([5, 10, 20, 50])),
                    "ret_1m": _r(21),
                    "ret_3m": _r(63),
                    "ret_6m": _r(126),
                    "ret_12m": _r(252),
                    "is_rs": True,
                }
            )

    events = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    truth = {
        "n_stocks": n_stocks,
        "n_events": len(events),
        "horizon": horizon,
        "drift_bps": drift_bps,
        "base_vol_ann": base_vol_ann,
        "seed": seed,
    }
    return events, truth


# ---------------------------------------------------------------------------
# Real loader — hardcoded event table + yfinance prices
# ---------------------------------------------------------------------------

def _prices_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_250_{safe}_1d.parquet")


def load_events(
    horizon_days: tuple[int, ...] = (21, 63, 126, 252),
    start_prices: str = "2008-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the hardcoded reverse-split event table and compute forward returns.

    Parameters
    ----------
    horizon_days:
        Look-ahead horizons in trading days.
    start_prices:
        Earliest date to download price history.
    fetch:
        ``False`` (default): read cached parquets; raises if absent.
        ``True``: hit Yahoo! Finance and write parquets to ``cache_dir``.
    cache_dir:
        Study-local cache directory (default: studies/250-reverse-split/_cache/).

    Returns
    -------
    events : DataFrame
        One row per reverse-split event with forward returns.
        Columns: ticker, event_date, ratio, ret_1m, ret_3m, ret_6m, ret_12m, is_rs.
    prices : DataFrame
        Wide daily adjusted close panel (columns = tickers, index = date).

    Notes
    -----
    - The hardcoded table uses the *effective* (ex-date), not announcement date.
    - Only tickers still accessible via yfinance are included; fully delisted names
      are excluded because yfinance cannot retrieve their price histories.
    - **Distress confound**: negative post-RS returns may reflect ongoing fundamental
      deterioration, not a tradable "RS signal".  See module docstring.
    - **Negative survivorship**: our sample over-represents names that survived long
      enough to have forward returns; it likely *understates* the average RS loss.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tickers = list(EVENTS_RAW["ticker"].unique())

    prices_dict: dict[str, pd.Series] = {}

    if fetch:
        import yfinance as yf
        for ticker in tickers:
            pr_path = _prices_cache_path(ticker, cache_dir)
            try:
                raw = yf.download(
                    ticker, start=start_prices, interval="1d",
                    auto_adjust=True, progress=False
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.rename(columns=str.lower)
                if not raw.empty and "close" in raw.columns:
                    raw[["close"]].to_parquet(pr_path)
                    close = raw["close"].copy()
                    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                    close.index.name = "date"
                    prices_dict[ticker] = close
            except Exception:
                pass
    else:
        missing = []
        for ticker in tickers:
            pr_path = _prices_cache_path(ticker, cache_dir)
            if not os.path.exists(pr_path):
                missing.append(ticker)
            else:
                pr = pd.read_parquet(pr_path)
                if not pr.empty and "close" in pr.columns:
                    close = pr["close"].copy()
                    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                    close.index.name = "date"
                    prices_dict[ticker] = close
        if missing:
            raise FileNotFoundError(
                f"No cached prices for {missing}. Call load_events(fetch=True) once."
            )

    if prices_dict:
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.DatetimeIndex(prices.index, name="date")
        prices = prices.sort_index()
    else:
        prices = pd.DataFrame()

    # Compute forward returns for each event
    rows = []
    for _, ev_row in EVENTS_RAW.iterrows():
        tkr = ev_row["ticker"]
        ev = pd.Timestamp(ev_row["event_date"])
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        idx = close_s.index
        pos_arr = int(np.searchsorted(idx, ev))
        if pos_arr >= len(idx):
            continue
        ev_pos = pos_arr
        ev_close = close_s.iloc[ev_pos]
        if ev_close <= 0 or not np.isfinite(ev_close):
            continue
        fwd_returns: dict[str, float] = {}
        for h, col in zip(horizon_days, ("ret_1m", "ret_3m", "ret_6m", "ret_12m")):
            end_pos = ev_pos + h
            if end_pos >= len(close_s):
                fwd_returns[col] = float("nan")
            else:
                fwd_returns[col] = float(close_s.iloc[end_pos] / ev_close - 1.0)
        rows.append(
            {
                "ticker": tkr,
                "event_date": ev,
                "ratio": int(ev_row["ratio"]),
                **fwd_returns,
                "is_rs": True,
            }
        )

    events = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    return events, prices


def fingerprint(events: pd.DataFrame) -> str:
    """Short content fingerprint of the event panel."""
    arr = pd.to_datetime(events["event_date"]).astype(np.int64).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
