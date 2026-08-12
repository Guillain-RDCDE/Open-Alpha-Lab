"""Data layer for Study 880 — Aggregate Short Interest.

The claim under test (Rapach, Ringgenberg & Zhou 2016, *"Short Interest and
Aggregate Stock Returns"*): the **market-wide** short-interest index is
"arguably the strongest known predictor" of the aggregate equity return — when
short sellers crowd the whole tape, forward market returns are **lower** (a
*negative* predictive slope). This is the **aggregate / time-series** cousin of
the cross-sectional short-interest sort in study 262.

Three pieces; the offline path is pure numpy + pandas + stdlib.

* **Real aggregate short-interest tape — FINRA.** ``fetch()`` pulls the FINRA
  *Consolidated Short Interest* file (the official bi-monthly, mid-month and
  end-of-month, settlement-date short-position report) for a fixed liquid
  ``PANEL`` of large-cap US names, one symbol at a time, from the public FINRA
  Query API (``api.finra.org/data``, no key). For each name and settlement date
  it records the reported short position, the average daily volume, and the
  exchange-published **days-to-cover** (short interest / ADV — the canonical
  *short-interest ratio*). The market-level index is the **equal-weight
  cross-sectional mean days-to-cover** across the panel on each settlement date
  (see :func:`aggregate_index`). Cached under this study's own ``_cache/``.

* **Real return tape — SPY.** ``fetch()`` also pulls SPY daily total-return
  (``auto_adjust=True``) via yfinance and caches it; the return side of the
  predictive regression is the forward SPY return over the settlement grid.

* **Synthetic positive control.** :func:`synthetic_frame` is a deterministic,
  fixed-seed generator of a bi-monthly short-interest index and a daily SPY-like
  price with a TUNABLE planted link: when the *detrended* short-interest index is
  high, the forward market return is knocked down by an ``edge`` knob. ``edge=0``
  is the null (short interest carries no forward information) and must NOT
  manufacture significance; a large ``edge`` must light the regression up.

**Frequency & availability honesty.** Aggregate short interest is **not** a daily
series: FINRA settles and publishes it only **twice a month** (the 15th and the
last business day), and each print is released with an ~8-business-day reporting
lag. So the real sample is ~24 observations a year since 2017-12; the strategy
documents the publication lag explicitly (a signal dated ``t`` is acted on at the
*next* settlement, after it is public). We do **not** have shares-outstanding from
FINRA, so the index is a **days-to-cover** average — a short-interest ratio in the
volume sense — rather than the shares-outstanding ratio of the original paper;
this is stated on the Signal axis.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SI_CACHE = os.path.join(CACHE_DIR, "finra_si_panel.parquet")
SPY_CACHE = os.path.join(CACHE_DIR, "spy_prices.csv")

START = "2017-12-01"        # first FINRA settlement in the dataset is 2017-12-29
AS_OF = "2026-06-30"        # last complete settlement month at publication

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"

# A fixed liquid large-cap panel — *current* membership, a survivor set. 49 names
# have the full FINRA history back to 2017-12-29; META (ex-FB symbol change) joins
# in 2021-07 and simply enters the equal-weight average once it is present.
PANEL = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "V", "JNJ", "WMT",
    "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV", "COST",
    "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD", "TXN",
    "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON", "UNH",
    "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS", "META",
]

MIN_NAMES = 20              # drop any settlement date with fewer ranked names

__all__ = [
    "PANEL", "START", "AS_OF", "CACHE_DIR", "SI_CACHE", "SPY_CACHE",
    "fetch", "have_real", "load_si_panel", "aggregate_index",
    "load_spy", "load_real", "synthetic_frame",
]


# --------------------------------------------------------------------------- #
# Real tape — FINRA consolidated short interest (network; builds the cache)
# --------------------------------------------------------------------------- #
def _finra_symbol(symbol: str, retries: int = 4, timeout: int = 60) -> list[dict]:
    """POST the FINRA Query API for one symbol's full short-interest history."""
    body = {
        "limit": 500,
        "compareFilters": [
            {"fieldName": "symbolCode", "compareType": "EQUAL", "fieldValue": symbol}
        ],
    }
    headers = {
        "User-Agent": "Open-Alpha-Lab research (contact: research@example.com)",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                FINRA_URL, data=json.dumps(body).encode(), headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 (retry any transport error)
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FINRA fetch failed for {symbol} after {retries} tries: {last}")


def fetch(panel: list[str] | None = None) -> None:
    """Download the FINRA short-interest panel + SPY and cache both (network-only).

    Runs once to build the cache; never imported by the offline notebook cells.
    """
    panel = panel or PANEL
    os.makedirs(CACHE_DIR, exist_ok=True)

    rows = []
    for sym in panel:
        recs = _finra_symbol(sym)
        for d in recs:
            dtc = d.get("daysToCoverQuantity")
            short = d.get("currentShortPositionQuantity")
            adv = d.get("averageDailyVolumeQuantity")
            sd = d.get("settlementDate")
            if sd is None or dtc is None:
                continue
            rows.append((pd.Timestamp(sd), sym, float(dtc),
                         float(short) if short is not None else np.nan,
                         float(adv) if adv is not None else np.nan))
        time.sleep(0.15)
    panel_df = pd.DataFrame(rows, columns=["date", "symbol", "dtc", "short", "adv"])
    panel_df = panel_df.dropna(subset=["dtc"]).sort_values(["date", "symbol"])
    panel_df.to_parquet(SI_CACHE, index=False)

    # SPY total-return daily
    import yfinance as yf

    raw = yf.download("SPY", start="2017-11-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):        # yfinance may return a 1-col frame
        raw = raw.iloc[:, 0]
    spy = pd.DataFrame({"SPY": pd.Series(raw).astype(float)}).dropna()
    spy.to_csv(SPY_CACHE)


def have_real() -> bool:
    return os.path.exists(SI_CACHE) and os.path.exists(SPY_CACHE)


# --------------------------------------------------------------------------- #
# Offline loaders (read the cache directly — no network)
# --------------------------------------------------------------------------- #
def load_si_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Long-form FINRA panel ``[date, symbol, dtc, short, adv]`` sliced to ``asof``."""
    df = pd.read_parquet(SI_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(asof)].sort_values(["date", "symbol"]).reset_index(drop=True)


def aggregate_index(panel_df: pd.DataFrame | None = None, asof: str = AS_OF,
                    min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Market-level short-interest index = equal-weight mean days-to-cover per date.

    Returns a settlement-date-indexed frame with ``si_index`` (average
    short-interest ratio across the panel) and ``n`` (names contributing). Dates
    with fewer than ``min_names`` names are dropped.
    """
    if panel_df is None:
        panel_df = load_si_panel(asof)
    g = panel_df.groupby("date")["dtc"]
    out = pd.DataFrame({"si_index": g.mean(), "n": g.size()}).sort_index()
    return out[out["n"] >= min_names]


def load_spy(asof: str = AS_OF) -> pd.Series:
    """Cached daily SPY total-return close, sliced to ``asof``."""
    df = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    s = df["SPY"].astype(float)
    return s[s.index <= pd.Timestamp(asof)]


def load_real(asof: str = AS_OF, min_names: int = MIN_NAMES) -> dict:
    """The real-tape object the strategy runs on.

    ``{"index": <settlement-dated si_index frame>, "spy": <daily SPY series>}``.
    Both offline (read straight from ``_cache/``).
    """
    return {"index": aggregate_index(asof=asof, min_names=min_names),
            "spy": load_spy(asof)}


# --------------------------------------------------------------------------- #
# Synthetic positive control — planted negative SI->return relation, null at 0
# --------------------------------------------------------------------------- #
def synthetic_frame(
    edge: float = 0.0,
    seed: int = 880,
    n_periods: int = 200,
    period_days: int = 10,
    start: str = "2010-01-04",
    daily_vol: float = 0.011,
    drift: float = 0.06 / 252,
    ar_rho: float = 0.90,
    si_center: float = 3.0,
    si_sd: float = 0.35,
) -> dict:
    """Deterministic bi-monthly-ish SI index + daily SPY-like price with a planted link.

    A mean-reverting log short-interest index (AR(1), ``ar_rho``) is sampled on a
    settlement grid every ``period_days`` business days. The **detrended** index
    ``s_t`` (its deviation from the sample mean, standardised) drives the forward
    period return only when ``edge > 0``::

        r_period[t+1] += drift_period - edge * s_t

    So a high detrended short-interest index at ``t`` depresses the *next* period's
    market return — the Rapach-Ringgenberg-Zhou negative aggregate relation, with a
    knob. ``edge = 0`` is the null. Business-day index, span far below the pandas
    ns-timestamp horizon (no OutOfBounds risk).
    """
    rng = np.random.default_rng(seed)
    n_days = n_periods * period_days + period_days
    idx = pd.bdate_range(start, periods=n_days)

    # mean-reverting log short-interest index on the settlement grid
    log_s = np.empty(n_periods)
    log_s[0] = np.log(si_center)
    tgt = np.log(si_center)
    innov = si_sd * np.sqrt(1.0 - ar_rho ** 2)
    for t in range(1, n_periods):
        log_s[t] = tgt + ar_rho * (log_s[t - 1] - tgt) + rng.normal(0.0, innov)
    si = np.exp(log_s)

    # standardised detrended index (deviation from mean / sd)
    s_std = (log_s - log_s.mean()) / (log_s.std(ddof=0) or 1.0)

    # per-period market return, with the planted link on the *next* period
    drift_period = drift * period_days
    per_ret = rng.normal(drift_period, daily_vol * np.sqrt(period_days), n_periods)
    if edge != 0.0:
        for t in range(n_periods - 1):
            per_ret[t + 1] -= edge * s_std[t]

    # spread each period's return across its business days to a daily SPY-like path
    daily_ret = np.zeros(n_days)
    for t in range(n_periods):
        seg = slice(t * period_days, (t + 1) * period_days)
        base = rng.normal(0.0, daily_vol, period_days)
        base = base - base.mean() + per_ret[t] / period_days
        daily_ret[seg] = base
    price = 100.0 * np.cumprod(1.0 + daily_ret)
    spy = pd.Series(price, index=idx, name="SPY")

    # settlement dates = the last business day of each period segment
    settle_pos = [(t + 1) * period_days - 1 for t in range(n_periods)]
    settle_idx = idx[settle_pos]
    index = pd.DataFrame({"si_index": si, "n": np.full(n_periods, len(PANEL))},
                         index=settle_idx)
    return {"index": index, "spy": spy}
