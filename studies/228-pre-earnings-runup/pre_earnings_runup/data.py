"""Data layer for the pre-earnings-runup study.

The desk's offline/cache split:

  * :func:`synthetic_runup` — fully **offline, deterministic**. A daily panel of ``n_stocks`` stocks;
    each name has scheduled earnings on a quarterly cadence (every ``quarter_days`` trading days, staggered
    per name). For the ``drift_strength > 0`` *control*, the ``pre_days`` trading days *before* each event
    carry a positive daily drift proportional to ``drift_strength`` that ramps up linearly as the event
    approaches (maximum drift on the day before). ``drift_strength = 0`` is the **null**: the same calendar
    is baked in but the pre-event window is pure noise. Returns ``(panel, events, truth)`` where ``events``
    is a frame ``[date, ticker]`` giving the *scheduled earnings date* (the day of the announcement, not
    tradable on that day — entries must be placed before it). Deterministic given ``seed``.

  * :func:`fetch_real_data` — the **real hook**, reads cached yfinance prices + earnings calendars for a
    fixed large-cap universe from the study's own ``_cache/`` directory. Cache-first, offline; network only
    on an explicit ``fetch=True`` miss. Returns ``{'panel': DataFrame, 'events': DataFrame}`` where events
    is ``[date, ticker]`` (announcement dates, one row per event).

Survivorship caveat: the real price panel uses *current* large-cap membership, so delisted names are absent,
which biases measured runup magnitudes upward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
STUDY_CACHE = os.path.join(_HERE, "..", "_cache")
TRADING_DAYS = 252

# Large-cap tickers used for the real run (a fixed list to avoid look-ahead in ticker selection).
LARGE_CAP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "JPM", "V", "JNJ",
    "UNH", "XOM", "PG", "HD", "MA", "CVX", "LLY", "MRK", "ABBV", "PEP",
    "KO", "AVGO", "TMO", "COST", "CSCO", "ACN", "MCD", "BAC", "NKE", "WMT",
    "ABT", "DHR", "TXN", "QCOM", "CRM", "NEE", "PM", "HON", "AMGN", "IBM",
    "LIN", "ORCL", "RTX", "SPGI", "INTU", "SBUX", "AXP", "GS", "BLK", "CAT",
    "MDT", "DE", "ISRG", "GILD", "BMY", "VRTX", "SYK", "ADI", "REGN", "ZTS",
    "CI", "CVS", "ANTM", "HUM", "AIG", "MMM", "EMR", "ETN", "PLD", "AMT",
    "CCI", "DLR", "PSA", "EQR", "SPG", "O", "WELL", "VTR", "HSY", "GIS",
    "CPB", "K", "CAG", "MKC", "SJM", "HRL", "CLX", "CHD", "EL", "CL",
    "PFE", "MO", "PM", "T", "VZ", "CMCSA", "DIS", "NFLX", "CHTR", "FOXA",
]


@dataclass(frozen=True)
class RunupTruth:
    """What the synthetic generator baked in, so a test can verify the strategy recovers it."""
    n_stocks: int
    n_bars: int
    drift_strength: float       # daily drift per day in the pre-event window; 0 == the null
    pre_days: int               # trading days before the event that carry the drift
    quarter_days: int           # event cadence per name

    @property
    def has_runup(self) -> bool:
        return self.drift_strength != 0.0


def synthetic_runup(
    n_stocks: int = 100,
    n_bars: int = 252 * 12,
    drift_strength: float = 0.0004,
    pre_days: int = 5,
    quarter_days: int = 63,
    mkt_vol: float = 0.009,
    idio: float = 0.013,
    seed: int = 228,
) -> tuple[pd.DataFrame, pd.DataFrame, RunupTruth]:
    """A daily cross-section where a pre-earnings runup is planted, by construction.

    Each stock reports on a quarterly cadence (every ``quarter_days`` trading days, with a per-name phase
    offset). In the ``pre_days`` trading days *before* each announcement the stock earns a positive daily
    drift of ``drift_strength * ramp(τ)``, where ``τ`` counts down from 0 (first pre-event day) to
    ``pre_days - 1`` (the day before the announcement), and ``ramp(τ) = (τ + 1) / pre_days`` rises linearly
    toward the announcement (maximum drift on the last pre-event day — the accumulation-into-event pattern).
    With ``drift_strength = 0`` the same calendar exists but carries no drift — the null.

    Returns ``(panel, events, truth)`` where ``panel`` is a ``dates × ticker`` daily-return frame and
    ``events`` is a long frame ``[date, ticker]`` giving the *announcement date* (entry must happen *before*
    this date to be strictly causal). One row per announcement. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2012-01-02", periods=n_bars, name="date")
    cols = [f"STK{i:03d}" for i in range(n_stocks)]

    market = 0.0003 + mkt_vol * rng.standard_normal(n_bars)
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)
    rets = betas[None, :] * market[:, None] + idio * rng.standard_normal((n_bars, n_stocks))

    event_rows = []
    for j in range(n_stocks):
        phase = int(rng.integers(0, quarter_days))
        for t0 in range(phase, n_bars, quarter_days):
            event_rows.append((idx[t0], cols[j]))
            if drift_strength != 0.0:
                # Plant a ramp drift in the pre_days BEFORE t0 (not including t0 itself)
                for k in range(pre_days):
                    t = t0 - pre_days + k      # day index: from t0-pre_days to t0-1
                    if 0 <= t < n_bars:
                        ramp = (k + 1) / pre_days    # 1/pre_days on first day, 1.0 on last day before event
                        rets[t, j] += drift_strength * ramp

    panel = pd.DataFrame(rets, index=idx, columns=cols)
    events = (
        pd.DataFrame(event_rows, columns=["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    truth = RunupTruth(
        n_stocks=n_stocks,
        n_bars=n_bars,
        drift_strength=drift_strength,
        pre_days=pre_days,
        quarter_days=quarter_days,
    )
    return panel, events, truth


# ---------------------------------------------------------------------------
# Real tape — cached yfinance prices + earnings dates.
# ---------------------------------------------------------------------------

PRICES_CACHE = "prices.parquet"
EVENTS_CACHE = "earnings_dates.parquet"


def fetch_real_data(
    tickers: list[str] | None = None,
    start: str = "2014-01-01",
    cache_dir: str = STUDY_CACHE,
    fetch: bool = False,
    rebuild: bool = False,
) -> dict:
    """Return ``{'panel': DataFrame, 'events': DataFrame}`` from the study-local cache, or fetch via
    yfinance if ``fetch=True`` and the cache is absent.

    ``panel`` is a ``dates × ticker`` frame of daily total-return (adjusted close pct_change).
    ``events`` is a ``[date, ticker]`` frame of earnings announcement dates (one row per event;
    the date is the *day of the announcement* — strategy must enter *before* this date).
    """
    if tickers is None:
        tickers = LARGE_CAP_TICKERS

    os.makedirs(cache_dir, exist_ok=True)
    prices_path = os.path.join(cache_dir, PRICES_CACHE)
    events_path = os.path.join(cache_dir, EVENTS_CACHE)

    if not rebuild and os.path.exists(prices_path) and os.path.exists(events_path):
        panel = pd.read_parquet(prices_path)
        events = pd.read_parquet(events_path)
        return {"panel": panel, "events": events}

    if not fetch:
        return {}

    try:
        import yfinance as yf
    except ImportError:
        return {}

    # Download adjusted close prices
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}

    # Handle both single-ticker and multi-ticker MultiIndex returns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    close = close.dropna(how="all").sort_index()
    panel = close.pct_change().iloc[1:].astype(float)
    panel.index.name = "date"

    # Collect earnings dates via yfinance
    event_rows = []
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            cal = t.calendar
            # yfinance returns calendar as dict with 'Earnings Date' key or a DataFrame
            if isinstance(cal, dict) and "Earnings Date" in cal:
                for d in (cal["Earnings Date"] if isinstance(cal["Earnings Date"], list)
                          else [cal["Earnings Date"]]):
                    try:
                        event_rows.append({"date": pd.Timestamp(d).normalize(), "ticker": tk})
                    except Exception:
                        pass
        except Exception:
            pass

        # Also pull historical earnings dates from quarterly_earnings
        try:
            t2 = yf.Ticker(tk)
            # earnings_dates attribute (newer yfinance)
            ed = t2.earnings_dates
            if ed is not None and not ed.empty:
                for d in ed.index:
                    try:
                        ts = pd.Timestamp(d)
                        # Strip timezone to get tz-naive date
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert("UTC").tz_localize(None)
                        event_rows.append({"date": ts.normalize(), "ticker": tk})
                    except Exception:
                        pass
        except Exception:
            pass

    if not event_rows:
        return {}

    df_events = pd.DataFrame(event_rows)
    # Ensure date column is tz-naive Timestamp
    df_events["date"] = pd.to_datetime(df_events["date"]).dt.tz_localize(None)
    events = (
        df_events
        .drop_duplicates()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    # Keep only events within the price panel's date range
    events = events[
        (events["date"] >= panel.index[0]) & (events["date"] <= panel.index[-1])
    ].reset_index(drop=True)
    # Keep only tickers present in the panel
    events = events[events["ticker"].isin(panel.columns)].reset_index(drop=True)

    panel.to_parquet(prices_path)
    events.to_parquet(events_path)
    return {"panel": panel, "events": events}
