"""Data layer for Study 360 (NAAIM-Exposure).

Three pieces, all offline-friendly for the reproducible core.

* **Real NAAIM weekly tape.** The National Association of Active Investment Managers
  (NAAIM) publishes a free *Exposure Index* every week since 2006-07-05: the mean
  reported equity exposure of its member firms on a 0-200% scale (0 = all cash,
  100 = fully invested, 200 = 2x long; the mean can dip slightly negative when
  members are net short). We cache the published spreadsheet's "NAAIM Number"
  column under ``_cache/naaim_weekly.csv`` (built once from
  ``USE_Data-since-Inception_*.xlsx`` on naaim.org). This is the genuine, free,
  primary source -- not a reconstruction.

* **Real SPY weekly tape.** SPY (SPDR S&P 500 ETF) **total-return** closes from
  yfinance (``auto_adjust=True``: dividends reinvested), cached daily under
  ``_cache/spy_daily.csv`` and resampled to the NAAIM survey dates. We measure the
  forward **1-week** SPY return earned *after* each weekly reading.

* **Synthetic positive control.** ``synthetic_weekly`` plants a *known* contrarian
  edge (next-week return loads negatively on the current exposure deviation) on a
  deterministic AR(1) exposure series. ``edge = 0`` is the null. This is the study's
  null in a bottle: it lets us prove the harness can detect a contrarian signal
  when one is present, and reads ~zero when it is not.

Plus ``NAAIM_FALLBACK`` -- a compact, **real** quarterly snapshot of the published
NAAIM Number (2006Q3 .. 2026Q2), clearly cited, so the package and the offline
notebook cells still run if the weekly cache is deleted. It is coarser than the
weekly tape and is labelled as such wherever it is used.

**No look-ahead.** The NAAIM survey closes Wednesday and is published the same day;
the exposure observed at week *t* predicts the SPY return earned over week *t+1*
(one-week execution lag, applied once). Calendar-known, conservative.

**Vintage caveat.** The published series has had minor revisions over its history
and the file is the *current* vintage, not strictly point-in-time. Any edge is an
upper bound on what a live trader watching the noisy weekly print could have banked.
Named on the Signal axis.

Pure numpy/pandas; network is touched only on an explicit ``fetch=True``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.join(_HERE, "..", "_cache")
NAAIM_CSV = os.path.join(STUDY_CACHE, "naaim_weekly.csv")
SPY_CSV = os.path.join(STUDY_CACHE, "spy_daily.csv")

# Long-run NAAIM Number average over the published history (2006-2026): ~65%.
NAAIM_AVG = 65.0

# ---------------------------------------------------------------------------
# Compact REAL fallback -- quarterly snapshots of the published NAAIM Number.
# (year, month, day, naaim_number).  Source: NAAIM Exposure Index history
# spreadsheet, naaim.org -- the last weekly reading of each calendar quarter.
# This is the genuine published series sub-sampled to quarters (NOT fabricated),
# used only when the weekly cache is absent; it is coarser and labelled so.
# ---------------------------------------------------------------------------
NAAIM_FALLBACK: list[tuple[int, int, int, float]] = [
    (2006, 9, 27, 49.9), (2006, 12, 27, 85.2),
    (2007, 3, 28, 44.5), (2007, 6, 27, 74.2), (2007, 9, 26, 77.4), (2007, 12, 26, 74.8),
    (2008, 3, 26, 26.7), (2008, 6, 25, 32.0), (2008, 9, 24, 19.1), (2008, 12, 31, 22.1),
    (2009, 3, 25, 47.8), (2009, 6, 24, 28.0), (2009, 9, 30, 86.4), (2009, 12, 30, 74.6),
    (2010, 3, 31, 83.3), (2010, 6, 30, 36.1), (2010, 9, 29, 68.3), (2010, 12, 29, 80.2),
    (2011, 3, 30, 70.9), (2011, 6, 29, 32.0), (2011, 9, 28, 4.2), (2011, 12, 28, 43.6),
    (2012, 3, 28, 72.0), (2012, 6, 27, 44.8), (2012, 9, 26, 69.8), (2012, 12, 26, 88.1),
    (2013, 3, 27, 80.1), (2013, 6, 26, 34.2), (2013, 9, 25, 80.5), (2013, 12, 26, 98.9),
    (2014, 3, 26, 91.1), (2014, 6, 25, 88.2), (2014, 9, 24, 59.8), (2014, 12, 31, 95.9),
    (2015, 3, 25, 84.3), (2015, 6, 24, 59.0), (2015, 9, 30, 16.4), (2015, 12, 30, 47.7),
    (2016, 3, 30, 67.9), (2016, 6, 29, 74.5), (2016, 9, 28, 84.5), (2016, 12, 28, 100.6),
    (2017, 3, 29, 68.3), (2017, 6, 28, 97.5), (2017, 9, 27, 106.6), (2017, 12, 27, 96.3),
    (2018, 3, 28, 49.4), (2018, 6, 27, 85.1), (2018, 9, 26, 90.7), (2018, 12, 26, 47.6),
    (2019, 3, 27, 61.7), (2019, 6, 26, 72.3), (2019, 9, 25, 64.8), (2019, 12, 25, 97.4),
    (2020, 3, 25, 25.9), (2020, 6, 24, 76.6), (2020, 9, 30, 58.2), (2020, 12, 30, 83.0),
    (2021, 3, 31, 52.0), (2021, 6, 30, 91.7), (2021, 9, 29, 55.0), (2021, 12, 29, 85.7),
    (2022, 3, 30, 79.7), (2022, 6, 29, 30.7), (2022, 9, 28, 12.6), (2022, 12, 28, 43.5),
    (2023, 3, 29, 65.2), (2023, 6, 28, 75.9), (2023, 9, 27, 43.0), (2023, 12, 27, 102.7),
    (2024, 3, 27, 103.9), (2024, 6, 26, 85.4), (2024, 9, 25, 86.6), (2024, 12, 25, 80.4),
    (2025, 3, 26, 57.5), (2025, 6, 25, 81.4), (2025, 9, 24, 86.2), (2025, 12, 31, 92.9),
    (2026, 3, 25, 68.5), (2026, 6, 17, 92.8),
]


# ---------------------------------------------------------------------------
# Real NAAIM weekly tape
# ---------------------------------------------------------------------------
def have_real() -> bool:
    return os.path.exists(NAAIM_CSV) and os.path.exists(SPY_CSV)


def naaim_table(path: str = NAAIM_CSV) -> pd.DataFrame:
    """The weekly NAAIM Number, indexed by survey (Wednesday) date.

    Falls back to the compact real quarterly snapshot (:data:`NAAIM_FALLBACK`)
    when the weekly cache is absent, so the offline core still runs. The returned
    frame has a single column ``naaim`` (the exposure number, percent).
    """
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[["date", "naaim"]].dropna().drop_duplicates("date").sort_values("date")
        return df.set_index("date")
    rows = [{"date": pd.Timestamp(y, m, d), "naaim": float(v)}
            for (y, m, d, v) in NAAIM_FALLBACK]
    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_spy(fetch: bool = False, path: str = SPY_CSV) -> pd.Series:
    """Daily SPY total-return close (``auto_adjust=True``) as a Series.

    Cache-only by default (reads ``_cache/spy_daily.csv``). With ``fetch=True`` it
    re-pulls from yfinance (lazy import; network only then). Total-return because
    a timing study that sits in cash forgoes dividends, so the honest benchmark is
    dividend-reinvested buy-and-hold on the *same* total-return index.
    """
    if not fetch and os.path.exists(path):
        s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["spy"]
        return s.astype(float).sort_index()

    import yfinance as yf  # lazy: network only on fetch

    raw = yf.download("SPY", start="2006-06-01", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no SPY data")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.astype(float).dropna().sort_index()


def _spy_on_or_before(daily: pd.Series, when: pd.Timestamp) -> float:
    """Last available SPY close at or before ``when`` (handles weekend/holiday)."""
    sub = daily.loc[:when]
    return float(sub.iloc[-1]) if len(sub) else float("nan")


def build_real_panel(fetch: bool = False) -> pd.DataFrame:
    """Join the weekly NAAIM Number to SPY *forward 1-week* total returns.

    For each survey date *t* we take the SPY close on/just-before *t* and the SPY
    close on/just-before the *next* survey date; the simple return between them is
    the ``ret`` the exposure reading at *t* is asked to predict (one-week lag).

    Returns a frame indexed by survey date with columns:
      ``naaim`` (exposure %, observed at *t*),
      ``ret``   (SPY total return earned over the following week).
    The last NAAIM row has no forward return and is dropped.
    """
    naaim = naaim_table()
    spy = fetch_spy(fetch=fetch)
    dates = list(naaim.index)
    px = np.array([_spy_on_or_before(spy, d) for d in dates], dtype=float)
    fwd = np.full(len(dates), np.nan)
    fwd[:-1] = px[1:] / px[:-1] - 1.0
    panel = naaim.copy()
    panel["ret"] = fwd
    panel = panel.dropna(subset=["naaim", "ret"])
    return panel


# ---------------------------------------------------------------------------
# Synthetic positive control -- deterministic, with a contrarian knob.
# ---------------------------------------------------------------------------
def synthetic_weekly(
    n_weeks: int = 1000,
    edge: float = 0.04,
    base_ret: float = 0.0016,
    ret_vol: float = 0.022,
    persistence: float = 0.85,
    seed: int = 360,
) -> tuple[pd.DataFrame, dict]:
    """A weekly (NAAIM exposure, SPY-return) tape with a *known* contrarian edge.

    The exposure ``e_t`` is an AR(1) process centred on the long-run NAAIM average
    (~65%), clipped to [0, 200]. The next-week return is::

        ret_{t+1} = base_ret - edge * (e_t - mean) / scale + noise

    so **high** (all-in) exposure predicts a **lower** next-week return and **low**
    (cash) exposure a **higher** one -- the contrarian story. ``edge = 0`` is the
    null (exposure carries no forward information). ``scale`` = the exposure std,
    so ``edge`` is roughly the return swing per +1 sd of positioning.

    Returns ``(df, truth)`` where ``df`` is indexed weekly with columns ``naaim``,
    ``ret`` (return earned *during* that week, already aligned so ``ret`` at row t
    is driven by ``naaim`` at row t-1).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2006-07-05", periods=n_weeks, freq="W-WED")

    e = np.empty(n_weeks)
    e[0] = NAAIM_AVG
    innov = rng.normal(0.0, 16.0, n_weeks)
    for t in range(1, n_weeks):
        e[t] = NAAIM_AVG + persistence * (e[t - 1] - NAAIM_AVG) + innov[t]
    e = np.clip(e, 0.0, 200.0)
    scale = float(np.std(e)) or 1.0

    noise = rng.normal(0.0, ret_vol, n_weeks)
    ret = np.full(n_weeks, base_ret) + noise
    ret[1:] += -edge * (e[:-1] - NAAIM_AVG) / scale  # contrarian loading, lagged

    df = pd.DataFrame({"naaim": e, "ret": ret}, index=idx)
    truth = {"n_weeks": n_weeks, "edge": edge, "base_ret": base_ret,
             "ret_vol": ret_vol, "persistence": persistence,
             "mean": NAAIM_AVG, "scale": scale, "seed": seed}
    return df, truth


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame, col: str = "naaim") -> str:
    """A short content fingerprint of a column, for the as-of stamp."""
    arr = pd.Series(df[col]).dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(np.round(arr, 6)).tobytes())
    return h.hexdigest()[:12]
