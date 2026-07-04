"""Data layer for Study 626 — Unemployment-Trend-Timing ("Growth-Trend Timing").

Four real inputs, all offline-friendly once cached under the study's ``_cache/``:

* **Unemployment.** BLS series ``LNS14000000`` — civilian unemployment rate (U-3),
  16 years+, **seasonally adjusted**, monthly, 1948-01 onward. Fetched once from the
  BLS public API v2 (keyless, 10-year chunks; the same series also lives in the BLS
  flat file ``download.bls.gov/pub/time.series/ln/ln.data.1.AllData`` — the API is
  simply the lighter one-time path to the identical numbers) and cached as
  ``unrate_lns14000000.csv``. **Vintage caveat (named on the Signal axis):** this is
  the *current* vintage — seasonal factors and benchmark revisions have reshaped the
  historical path since the original real-time prints, so a genuinely-real-time
  backtest would have seen slightly different values.

* **S&P 500 daily closes.** ``^GSPC`` via yfinance (1927-12-30 onward), **price-only**
  (the long index carries no dividends) — cached as ``gspc_daily.csv``. Used for the
  200-day SMA trend signal and for month-end monthly price returns.

* **Dividend yield.** Shiller's monthly S&P dataset (repo-staged parquet, falling back
  to the ``datasets/s-and-p-500`` mirror on raw.githubusercontent.com). The trailing
  annual dividend rate D over the monthly price P gives an annual dividend yield; we
  add ``dy/12`` to each monthly price return so the headline race is **total-return**,
  not price-only. Shiller's not-yet-populated trailing zeros are forward-filled from
  the last real print (documented; affects only the last few months).

* **Cash (T-bills).** 1960-01 onward: ``^IRX`` (13-week T-bill discount rate, %) via
  yfinance, monthly mean, cached ``irx_daily.csv``. 1948–1959: yfinance has no bill
  rate, so we hardcode the **annual average 3-month Treasury-bill (new issue) rate**
  from the Economic Report of the President 2011, Table B-73 (verified against the
  published table; constant within each year — a documented approximation). Monthly
  cash return = annual rate / 12 (simple).

The synthetic world is a deterministic, seeded two-regime (expansion/recession)
generator with a **tunable planted recession drag** on equity returns and a switch
that either ties unemployment to the true return regime (the effect) or drives it
from an independent regime chain (the null). Spans stay well under 250 years
(pandas ns-Timestamp CI trap; the index is built with ``period_range``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CACHE_DIR = os.path.join(STUDY_ROOT, "_cache")

UNRATE_CACHE = os.path.join(CACHE_DIR, "unrate_lns14000000.csv")
GSPC_CACHE = os.path.join(CACHE_DIR, "gspc_daily.csv")
IRX_CACHE = os.path.join(CACHE_DIR, "irx_daily.csv")
DIVYIELD_CACHE = os.path.join(CACHE_DIR, "shiller_divyield_monthly.csv")

# Repo-wide staged caches (reused if present; never re-fetched live).
_REPO_GSPC = [os.path.join(REPO_ROOT, "_cache", "^GSPC_split_only.parquet"),
              os.path.join(REPO_ROOT, "_cache", "_cache", "^GSPC_split_only.parquet")]
_REPO_SHILLER = [os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet"),
                 os.path.join(REPO_ROOT, "_cache", "_cache", "shiller_sp500.parquet")]
SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_SERIES = "LNS14000000"
_UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us",
       "Content-Type": "application/json"}

# As-of guard: June 2026 is the last complete calendar month for a 2026-07-03 run.
AS_OF = "2026-06-30"

# Annual average 3-month Treasury-bill (new issue) rate, percent per annum,
# 1948-1959 — Economic Report of the President 2011, Table B-73 ("Bond yields and
# interest rates, 1933-2010", 3-month bills at auction), verified against the
# published table at govinfo.gov/content/pkg/ERP-2011/pdf/ERP-2011-table73.pdf.
TBILL_PRE1960 = {
    1948: 1.040, 1949: 1.102, 1950: 1.218, 1951: 1.552, 1952: 1.766, 1953: 1.931,
    1954: 0.953, 1955: 1.753, 1956: 2.658, 1957: 3.267, 1958: 1.839, 1959: 3.405,
}


# --------------------------------------------------------------------------- #
# Fetch-once builders (network; never imported by the offline notebook cells)
# --------------------------------------------------------------------------- #
def fetch_unrate(path: str = UNRATE_CACHE, start_year: int = 1948,
                 end_year: int = 2026, retries: int = 3) -> pd.Series:
    """Fetch LNS14000000 (U-3 SA, monthly) from the keyless BLS API v2, cache CSV.

    The keyless API caps each request at 10 years, so the 1948→now span is pulled
    in chunks and stitched. Identical values ship in the BLS flat file
    ``pub/time.series/ln/ln.data.1.AllData`` (389 MB) — the API is the light path.
    """
    rows: dict[pd.Timestamp, float] = {}
    for y0 in range(start_year, end_year + 1, 10):
        y1 = min(y0 + 9, end_year)
        payload = json.dumps({"seriesid": [BLS_SERIES],
                              "startyear": str(y0), "endyear": str(y1)}).encode()
        out = None
        for _ in range(retries):
            try:
                req = urllib.request.Request(BLS_API, data=payload, headers=_UA)
                out = json.loads(urllib.request.urlopen(req, timeout=60).read())
                if out.get("status") == "REQUEST_SUCCEEDED":
                    break
            except Exception:
                time.sleep(2.0)
        if out is None or out.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API failed for {y0}-{y1}")
        for rec in out["Results"]["series"][0]["data"]:
            if not rec["period"].startswith("M"):
                continue
            try:
                val = float(rec["value"])
            except ValueError:          # BLS prints '-' for not-yet-available cells
                continue
            ts = pd.Period(f"{rec['year']}-{rec['period'][1:]}", freq="M").to_timestamp(how="end").normalize()
            rows[ts] = val
    s = pd.Series(rows).sort_index()
    s.index.name = "date"
    s.name = "unrate"
    os.makedirs(CACHE_DIR, exist_ok=True)
    s.to_csv(path)
    return s


def fetch_gspc(path: str = GSPC_CACHE) -> pd.Series:
    """Daily ^GSPC closes (price-only) via yfinance; repo parquet as the fallback.

    A live pull is preferred so the tape covers the full as-of month (the staged
    repo parquet can lag a couple of weeks and would leave a partial month).
    """
    close = None
    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if close.dropna().empty:
            close = None
    except Exception:
        close = None
    if close is None:
        for p in _REPO_GSPC:
            if os.path.exists(p):
                close = pd.read_parquet(p)["Close"]
                break
    if close is None:
        raise RuntimeError("no ^GSPC source available")
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "date"
    close.name = "close"
    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_csv(path)
    return close


def fetch_irx(path: str = IRX_CACHE) -> pd.Series:
    """Daily ^IRX (13-week T-bill discount rate, % p.a.) via yfinance, cached."""
    import yfinance as yf
    raw = yf.download("^IRX", start="1960-01-01", auto_adjust=False, progress=False)
    rate = raw["Close"]
    if isinstance(rate, pd.DataFrame):
        rate = rate.iloc[:, 0]
    rate = rate.dropna()
    rate.index = pd.to_datetime(rate.index).tz_localize(None).normalize()
    rate.index.name = "date"
    rate.name = "irx"
    os.makedirs(CACHE_DIR, exist_ok=True)
    rate.to_csv(path)
    return rate


def fetch_divyield(path: str = DIVYIELD_CACHE) -> pd.Series:
    """Monthly annual dividend yield D/P from Shiller's dataset, cached.

    Uses the repo-staged Shiller parquet if present, else the public
    ``datasets/s-and-p-500`` mirror. Trailing months where Shiller has not yet
    populated D (stored as 0) are forward-filled from the last real print.
    """
    df = None
    for p in _REPO_SHILLER:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    if df is None:
        req = urllib.request.Request(SHILLER_URL, headers={"User-Agent": _UA["User-Agent"]})
        df = pd.read_csv(urllib.request.urlopen(req, timeout=60))
    df = df.rename(columns={c: c.strip() for c in df.columns})
    dt = pd.to_datetime(df["Date"])
    dy = pd.Series(df["Dividend"].to_numpy(dtype=float)
                   / df["SP500"].to_numpy(dtype=float),
                   index=dt.dt.to_period("M").dt.to_timestamp(how="end").dt.normalize())
    dy = dy.replace(0.0, np.nan).ffill().dropna()
    dy.index.name = "date"
    dy.name = "divyield"
    os.makedirs(CACHE_DIR, exist_ok=True)
    dy.to_csv(path)
    return dy


def fetch_all() -> None:
    """Build every cache once (network). Prints a one-line stamp per input."""
    u = fetch_unrate()
    print(f"unrate  : {len(u)} months {u.index.min().date()} -> {u.index.max().date()}")
    g = fetch_gspc()
    print(f"gspc    : {len(g)} days   {g.index.min().date()} -> {g.index.max().date()}")
    r = fetch_irx()
    print(f"irx     : {len(r)} days   {r.index.min().date()} -> {r.index.max().date()}")
    d = fetch_divyield()
    print(f"divyield: {len(d)} months {d.index.min().date()} -> {d.index.max().date()}")


# --------------------------------------------------------------------------- #
# Cache-first loaders
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    return all(os.path.exists(p) for p in
               (UNRATE_CACHE, GSPC_CACHE, IRX_CACHE, DIVYIELD_CACHE))


def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].sort_index()


def load_monthly(as_of: str = AS_OF, sma_days: int = 200,
                 u_window: int = 12) -> pd.DataFrame:
    """The master month-end frame every test runs on (cache-only, deterministic).

    Columns (index = month-end):

    - ``eq_ret``     : monthly S&P 500 **total** return = price return + dy/12
    - ``eq_ret_px``  : monthly price-only return (robustness leg)
    - ``cash_ret``   : monthly T-bill return (annual rate / 12)
    - ``unrate``     : U-3 SA unemployment for the *reference* month
    - ``trend_up``   : month-end close > its 200-day SMA (signal formed at that close)
    - ``unemp_rising``: latest **available** print (the prior month's, honouring the
      1-month reporting lag) above the 12-month SMA of available prints
    - ``tr_level``   : cumulative total-return index (for drawdowns)

    Signals dated month *m* are formed at the close of month *m*; positions are
    taken for month *m+1* by ``strategy.backtest`` via ``shift(1)`` — exactly one
    execution lag, plus the documented reporting lag inside ``unemp_rising``.
    """
    asof = pd.Timestamp(as_of)
    px = _load_series(GSPC_CACHE)
    px = px[px.index <= asof]
    sma = px.rolling(sma_days).mean()
    me_px = px.resample("ME").last()
    me_sma = sma.resample("ME").last()
    pret = me_px.pct_change()

    dy = _load_series(DIVYIELD_CACHE)
    dy = dy.reindex(pret.index, method="ffill")
    eq_ret = pret + dy / 12.0

    irx = _load_series(IRX_CACHE)
    irx = irx[irx.index <= asof]
    rate = irx.resample("ME").mean().reindex(pret.index)
    pre = pd.Series({ts: TBILL_PRE1960[ts.year] for ts in pret.index
                     if ts.year in TBILL_PRE1960})
    rate = rate.fillna(pre)
    cash_ret = rate / 100.0 / 12.0

    u = _load_series(UNRATE_CACHE)
    u = u[u.index <= asof].reindex(pret.index)
    u_avail = u.shift(1)                       # the 1-month reporting lag
    unemp_rising = u_avail > u_avail.rolling(u_window).mean()

    out = pd.DataFrame({
        "eq_ret": eq_ret, "eq_ret_px": pret, "cash_ret": cash_ret,
        "unrate": u, "trend_up": me_px > me_sma, "unemp_rising": unemp_rising,
    })
    out = out.dropna(subset=["eq_ret", "cash_ret", "unrate"])
    out = out[out.index > out.index[0] + pd.DateOffset(months=u_window)]
    out["tr_level"] = (1.0 + out["eq_ret"]).cumprod()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — deterministic positive control with a TUNABLE planted effect
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 80, gap: float = 0.020, linked: bool = True,
                    seed: int = 626) -> pd.DataFrame:
    """A monthly two-regime world with a planted recession drag on equities.

    Expansion months draw N(+0.9%, 3.8%); recession months draw
    N(+0.9% − ``gap``, 5.5%). The regime chain targets ~15% of months in
    recession with ~12-month average spells. Unemployment falls slowly in
    expansions and climbs in recessions.

    - ``linked=True`` : unemployment tracks the **same** regime that drives
      returns — the filter carries genuine information (the planted effect;
      ``gap`` is the knob).
    - ``linked=False``: unemployment is driven by an **independent** regime
      chain of identical dynamics — the null; the filter is pure noise and no
      GTT-vs-Faber edge may appear however the draws fall.

    Monthly period index (well under 250 years). Same columns as
    :func:`load_monthly`, so every strategy function runs unchanged.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * 12)
    pidx = pd.period_range("1950-01", periods=n, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    def regime_path(r):
        st = np.zeros(n, dtype=bool)         # False=expansion, True=recession
        cur = False
        for i in range(n):
            if cur:
                cur = r.random() >= 1.0 / 12.0      # E[spell] ~ 12 months
            else:
                cur = r.random() < 0.015            # ~15% unconditional share
            st[i] = cur
        return st

    rec = regime_path(rng)
    rec_u = rec if linked else regime_path(rng)

    ret = np.where(rec, rng.normal(0.009 - gap, 0.055, n),
                   rng.normal(0.009, 0.038, n))
    u = np.empty(n)
    u[0] = 5.0
    for i in range(1, n):
        du = (0.28 if rec_u[i] else -0.045) + rng.normal(0.0, 0.06)
        u[i] = float(np.clip(u[i - 1] + du, 2.5, 12.0))

    price = 100.0 * np.cumprod(1.0 + ret)
    price_s = pd.Series(price, index=idx)
    sma10 = price_s.rolling(10).mean()           # 10-month SMA ~ the 200-day SMA
    u_s = pd.Series(u, index=idx)
    u_avail = u_s.shift(1)

    out = pd.DataFrame({
        "eq_ret": pd.Series(ret, index=idx),
        "eq_ret_px": pd.Series(ret, index=idx),
        "cash_ret": 0.003,
        "unrate": u_s,
        "trend_up": price_s > sma10,
        "unemp_rising": u_avail > u_avail.rolling(12).mean(),
    }).iloc[13:]                     # drop the SMA / rolling-mean warm-up rows
    out["tr_level"] = (1.0 + out["eq_ret"]).cumprod()
    return out
