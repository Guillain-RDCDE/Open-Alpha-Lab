"""Data layer for Study 760 — Michigan-Sentiment-Day (UMich sentiment + SPY).

Three components, the first and third fully offline and deterministic:

* **Real sentiment tape (hardcoded snapshot).** ``UMCSENT_BY_YEAR`` is a hardcoded
  monthly snapshot of the **University of Michigan Index of Consumer Sentiment**
  (1966:Q1 = 100), 1978..2026. Source: University of Michigan, Surveys of Consumers
  (FRED series ``UMCSENT``). FRED's CSV endpoint is firewalled in this build, so —
  exactly like Study 385 (Jobless-Claims) hardcodes FRED ``IC4WSA`` and Study 268
  (Sahm-Rule) hardcodes FRED ``UNRATE`` — we hardcode a public, well-known monthly
  snapshot. Michigan reports a **preliminary** reading mid-month (usually the second
  Friday) and a **final** reading late-month; the settled monthly value hardcoded here
  is the final print. The believers' signals are computed *from* this series (see
  :mod:`michigan_sentiment_day.strategy`).

* **Real SPY tape.** ``load_spy`` reads the cached daily SPY adjusted close
  (``_cache/spy.csv``, yfinance, no key). ``fetch_spy`` (network) rebuilds the cache and
  is never imported by the offline notebook cells. Price = total-return adjusted close
  (``auto_adjust=True``); labelled as such. Release-day tests need the **daily** tape;
  the level/regime test uses month-end sampling.

* **Release-date proxy (labelled).** :func:`release_dates` returns the **second Friday**
  of each month — the canonical UMich *preliminary*-release day. The true calendar shifts
  a day or two around holidays; this is a small, clearly-labelled proxy for the release
  schedule, never presented as the official calendar.

* **Synthetic positive control.** :func:`synthetic` is a deterministic, fixed-seed
  generator producing a monthly sentiment path and a daily SPY-like price with a *planted*
  link (a knob that makes low-then-rising sentiment genuinely precede higher forward
  returns). ``edge = 0`` is the null and must NOT manufacture significance; a large planted
  ``edge`` must light the test up.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")


# --------------------------------------------------------------------------- #
# Real sentiment tape — hardcoded monthly snapshot of FRED UMCSENT (1966:Q1=100)
# --------------------------------------------------------------------------- #
# University of Michigan Index of Consumer Sentiment, monthly, NOT seasonally
# adjusted, index 1966:Q1 = 100. Source: U. Michigan Surveys of Consumers (FRED
# series UMCSENT). Monthly rows Jan..Dec. As-of: 2026-04 final prints.
# Public, widely-reported values; the famous troughs — 1980 (~52), 2008-09 (~55-57),
# 2011 (~56), June 2022 (50.0, an all-time low), 2025 (~52) — are included faithfully.
UMCSENT_BY_YEAR: dict[int, list[float]] = {
    1978: [83.7, 84.3, 78.8, 81.6, 82.9, 80.0, 82.4, 78.4, 80.4, 79.3, 75.0, 66.1],
    1979: [72.1, 73.9, 68.4, 66.0, 68.1, 65.8, 60.4, 64.5, 66.7, 62.1, 63.3, 61.0],
    1980: [67.0, 66.9, 56.5, 52.7, 51.7, 58.7, 62.3, 67.3, 73.7, 75.0, 76.7, 64.5],
    1981: [71.4, 66.9, 66.5, 72.4, 76.3, 73.1, 74.1, 77.2, 73.1, 70.3, 62.5, 64.3],
    1982: [71.0, 66.5, 62.0, 65.5, 67.5, 65.7, 65.4, 65.4, 69.3, 73.4, 72.1, 71.9],
    1983: [70.4, 74.6, 80.8, 89.1, 93.3, 92.2, 92.8, 90.9, 89.9, 89.3, 91.1, 94.2],
    1984: [100.1, 97.4, 101.0, 96.1, 98.1, 95.5, 96.6, 99.1, 100.9, 96.3, 95.7, 92.9],
    1985: [96.0, 93.7, 93.7, 94.6, 91.8, 96.5, 94.0, 92.4, 92.1, 88.4, 90.9, 93.9],
    1986: [95.6, 95.9, 95.1, 96.2, 94.8, 99.3, 97.7, 94.9, 91.9, 95.6, 91.4, 89.1],
    1987: [90.4, 90.2, 90.8, 92.8, 91.1, 91.5, 93.7, 94.4, 93.6, 89.3, 83.1, 86.8],
    1988: [90.8, 91.6, 94.6, 91.2, 94.8, 94.7, 93.4, 97.4, 97.3, 94.1, 93.0, 91.9],
    1989: [97.9, 95.4, 94.3, 91.5, 90.7, 90.6, 92.0, 89.6, 95.8, 93.9, 90.9, 90.5],
    1990: [93.0, 89.5, 91.3, 93.9, 90.6, 88.3, 88.2, 76.4, 72.8, 63.9, 66.0, 65.5],
    1991: [66.8, 70.4, 87.7, 81.8, 78.3, 82.1, 82.9, 82.0, 83.0, 78.3, 69.1, 68.2],
    1992: [67.5, 68.8, 76.0, 77.2, 79.2, 80.4, 76.6, 76.1, 75.6, 73.3, 85.3, 91.0],
    1993: [89.3, 86.6, 85.9, 85.6, 80.3, 81.5, 77.0, 77.3, 77.9, 82.7, 81.2, 88.2],
    1994: [94.3, 93.2, 91.5, 92.6, 92.8, 91.2, 89.0, 91.7, 91.5, 92.7, 91.6, 95.1],
    1995: [97.6, 95.1, 90.3, 92.5, 89.8, 92.7, 94.4, 96.2, 88.9, 90.2, 88.2, 91.0],
    1996: [89.3, 88.5, 93.7, 92.7, 89.4, 92.4, 94.7, 95.3, 94.7, 96.5, 99.2, 96.9],
    1997: [97.4, 99.7, 100.0, 101.4, 103.2, 104.5, 107.1, 104.4, 106.0, 105.6, 107.2, 102.1],
    1998: [106.6, 110.4, 106.5, 108.7, 106.5, 105.6, 105.2, 104.4, 100.9, 97.4, 102.7, 100.5],
    1999: [103.9, 108.1, 105.7, 104.6, 106.8, 107.3, 106.0, 104.5, 107.2, 103.2, 107.2, 105.4],
    2000: [112.0, 111.3, 107.1, 109.2, 110.7, 106.4, 108.3, 107.3, 106.8, 105.8, 107.6, 98.4],
    2001: [94.7, 90.6, 91.5, 88.4, 92.0, 92.6, 92.4, 91.5, 81.8, 82.7, 83.9, 88.8],
    2002: [93.0, 90.7, 95.7, 93.0, 96.9, 92.4, 88.1, 87.6, 86.1, 80.6, 84.2, 86.7],
    2003: [82.4, 79.9, 77.6, 86.0, 92.1, 89.7, 90.9, 89.3, 87.7, 89.6, 93.7, 92.6],
    2004: [103.8, 94.4, 95.8, 94.2, 90.2, 95.6, 96.7, 95.9, 94.2, 91.7, 92.8, 97.1],
    2005: [95.5, 94.1, 92.6, 87.7, 86.9, 96.0, 96.5, 89.1, 76.9, 74.2, 81.6, 91.5],
    2006: [91.2, 86.7, 88.9, 87.4, 79.1, 84.9, 84.7, 82.0, 85.4, 93.6, 92.1, 91.7],
    2007: [96.9, 91.3, 88.4, 87.1, 88.3, 85.3, 90.4, 83.4, 83.4, 80.9, 76.1, 75.5],
    2008: [78.4, 70.8, 69.5, 62.6, 59.8, 56.4, 61.2, 63.0, 70.3, 57.6, 55.3, 60.1],
    2009: [61.2, 56.3, 57.3, 65.1, 68.7, 70.8, 66.0, 65.7, 73.5, 70.6, 67.4, 72.5],
    2010: [74.4, 73.6, 73.6, 72.2, 73.6, 76.0, 67.8, 68.9, 68.2, 67.7, 71.6, 74.5],
    2011: [74.2, 77.5, 67.5, 69.8, 74.3, 71.5, 63.7, 55.8, 59.5, 60.8, 63.7, 69.9],
    2012: [75.0, 75.3, 76.2, 76.4, 79.3, 73.2, 72.3, 74.3, 78.3, 82.6, 82.7, 72.9],
    2013: [73.8, 77.6, 78.6, 76.4, 84.5, 84.1, 85.1, 82.1, 77.5, 73.2, 75.1, 82.5],
    2014: [81.2, 81.6, 80.0, 84.1, 81.9, 82.5, 81.8, 82.5, 84.6, 86.9, 88.8, 93.6],
    2015: [98.1, 95.4, 93.0, 95.9, 90.7, 96.1, 93.1, 91.9, 87.2, 90.0, 91.3, 92.6],
    2016: [92.0, 91.7, 91.0, 89.0, 94.7, 93.5, 90.0, 89.8, 91.2, 87.2, 93.8, 98.2],
    2017: [98.5, 96.3, 96.9, 97.0, 97.1, 95.0, 93.4, 96.8, 95.1, 100.7, 98.5, 95.9],
    2018: [95.7, 99.7, 101.4, 98.8, 98.0, 98.2, 97.9, 96.2, 100.1, 98.6, 97.5, 98.3],
    2019: [91.2, 93.8, 98.4, 97.2, 100.0, 98.2, 98.4, 89.8, 93.2, 95.5, 96.8, 99.3],
    2020: [99.8, 101.0, 89.1, 71.8, 72.3, 78.1, 72.5, 74.1, 80.4, 81.8, 76.9, 80.7],
    2021: [79.0, 76.8, 84.9, 88.3, 82.9, 85.5, 81.2, 70.3, 72.8, 71.7, 67.4, 70.6],
    2022: [67.2, 62.8, 59.4, 65.2, 58.4, 50.0, 51.5, 58.2, 58.6, 59.9, 56.7, 59.8],
    2023: [64.9, 66.9, 62.0, 63.7, 59.0, 64.2, 71.5, 69.4, 67.8, 63.8, 61.3, 69.7],
    2024: [79.0, 76.9, 79.4, 77.2, 69.1, 68.2, 66.4, 67.9, 70.1, 70.5, 71.8, 74.0],
    2025: [71.7, 64.7, 57.0, 52.2, 52.2, 60.7, 61.7, 58.2, 55.1, 53.6, 51.0, 52.9],
    2026: [56.4, 56.6, 53.3, 49.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def have_real(path: str = SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the sentiment table is always available)."""
    return os.path.exists(path)


def sentiment_series() -> pd.Series:
    """Monthly UMich Index of Consumer Sentiment, indexed by month-*end* date.

    In-progress months of the final year (zeros) are dropped so a stamped run never
    includes a partial bar.
    """
    rows = []
    for yr in sorted(UMCSENT_BY_YEAR):
        for mo, v in enumerate(UMCSENT_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="sent").sort_index()


def second_friday(year: int, month: int) -> pd.Timestamp:
    """The second Friday of ``year``-``month`` — the canonical UMich prelim-release day."""
    first = pd.Timestamp(year, month, 1)
    offset = (4 - first.weekday()) % 7          # days until the first Friday
    return first + pd.Timedelta(days=offset + 7)


def release_dates(start: int = 1993, end: int = 2026) -> pd.DatetimeIndex:
    """Second-Friday-of-month prelim-release schedule proxy over [start, end]."""
    ds = [second_friday(y, m) for y in range(start, end + 1) for m in range(1, 13)]
    return pd.DatetimeIndex(ds)


def fetch_spy(path: str = SPY_CACHE, start: str = "1993-01-01",
              end: str | None = None) -> pd.Series:
    """Download SPY daily adjusted close via yfinance and cache it (network-only).

    Used once to build ``_cache/spy.csv``. Never imported by offline cells. Total-return
    adjusted close (``auto_adjust=True``).
    """
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    out = pd.DataFrame({"SPY": spy}).dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out["SPY"]


def load_spy(path: str = SPY_CACHE) -> pd.Series:
    """Load the cached daily SPY adjusted-close series (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s.name = "spy"
    return s


def spy_monthly(path: str = SPY_CACHE) -> pd.Series:
    """Month-end SPY adjusted close (the grid the level/regime test runs on)."""
    return load_spy(path).resample("ME").last().dropna()


def load_monthly(path: str = SPY_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: sentiment level + month-end SPY close.

    Columns ``sent`` and ``spy``; only months present in both are kept. Michigan's
    preliminary print is public *mid-month*, so pairing month-``t`` sentiment with the
    month-``t`` end close carries no look-ahead (the print precedes the close).
    """
    df = pd.DataFrame({"sent": sentiment_series(), "spy": spy_monthly(path)}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 396, edge: float = 0.0, seed: int = 760,
              mu_d: float = 0.0003, sig_d: float = 0.011) -> dict:
    """Deterministic monthly sentiment + daily SPY-like price with a PLANTED link.

    Builds a mean-reverting monthly sentiment level (AR(1) around ~85) and a daily
    SPY-like price. When ``edge != 0`` the *forward* monthly return is lifted by ``edge``
    whenever sentiment is **low-and-rising** (below its trailing 30th percentile *and*
    above its value three months prior) — the believers' "bottom-timer" story injected by
    construction, with a knob.

    ``edge = 0`` => low-then-rising sentiment carries no information (the null); the
    inference must NOT manufacture significance. A large planted ``edge`` (e.g. 0.03
    monthly) must drive the regime test well past ``t = 2``. Returns a dict with a monthly
    ``frame`` (``sent`` + month-end ``spy``) and the ``daily`` SPY series.
    """
    rng = np.random.default_rng(seed)

    # mean-reverting sentiment around 85, AR(1)
    sent = np.empty(n_months)
    sent[0] = 85.0
    target = 85.0
    for t in range(1, n_months):
        sent[t] = target + 0.90 * (sent[t - 1] - target) + rng.normal(0, 6.0)

    # daily SPY-like path, ~21 trading days per month
    dpm = 21
    n_days = n_months * dpm
    dret = rng.normal(mu_d, sig_d, size=n_days)

    # low-and-rising regime on the synthetic sentiment
    s = pd.Series(sent)
    pct = s.expanding(min_periods=12).apply(lambda x: (x.iloc[-1] >= x).mean())
    rising = s > s.shift(3)
    low_rising = ((pct <= 0.30) & rising).fillna(False).values

    # plant the believers' link: low-and-rising at month t lifts the NEXT 12 months of
    # returns by a total of ``edge`` (spread evenly across days) — "a year of recovery
    # after a sentiment bottom", the exact horizon the regime test reads at.
    if edge != 0.0:
        h_plant = 12
        for t in range(n_months - h_plant):
            if low_rising[t]:
                lo = (t + 1) * dpm
                hi = lo + h_plant * dpm
                dret[lo:hi] += edge / (h_plant * dpm)

    price = 100.0 * np.exp(np.cumsum(dret))
    didx = pd.bdate_range("1990-01-01", periods=n_days)
    daily = pd.Series(price, index=didx, name="spy")
    # month-end price = last synthetic day of each fixed 21-day block (exact, length n_months)
    month_end_px = price[np.minimum((np.arange(n_months) + 1) * dpm - 1, n_days - 1)]
    midx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    frame = pd.DataFrame({"sent": sent, "spy": month_end_px}, index=midx)
    return {"frame": frame, "daily": daily}
