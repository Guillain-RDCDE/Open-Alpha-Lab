"""Data layer for Study 628 — Buffett's Alpha.

Two sources, both offline-friendly once cached:

* **Real tape.** One monthly frame (``_cache/ba_monthly.csv``) with:

  - ``brk``  — Berkshire Hathaway (BRK-A) monthly total return. Berkshire has paid **no
    dividend since 1967**, so the yfinance adjusted close IS the total-return series — no
    price-only/total-return ambiguity on the fund leg.
  - ``mkt``  — the US market **total return**, spliced honestly and labeled per row
    (``mkt_src``): from **1993-02** on it is SPY's adjusted (total-return) monthly return;
    before that it is the **^GSPC month-end price return + the Shiller monthly dividend
    yield** (annual dividend rate / 12 / price), the standard way to rebuild a pre-ETF S&P
    total return without a paywalled index.
  - ``rf``   — the 13-week T-bill (^IRX) discount yield observed at the **previous**
    month-end, /100/12: the risk-free rate an investor could actually lock for the coming
    month (known in advance — no look-ahead).
  - ``qual``/``usmv`` — investable factor-lite proxies for the FKP quality and low-beta
    stories: iShares QUAL (MSCI USA Quality, live 2013-07) and USMV (MSCI USA Min-Vol,
    live 2011-10), monthly total returns. Honest labeling: these are the *investable ETF
    era* proxies, NOT the academic QMJ/BAB long-short factors — the attribution is
    "factor-lite" and said so everywhere.

* **Synthetic.** A deterministic, seeded generator producing a (market, manager) pair where
  the manager has a **planted** monthly CAPM alpha (knob ``alpha_m``) on a chosen beta, plus
  autocorrelated noise. It is the machinery proof: with ``alpha_m = 0`` the HAC alpha *t*
  must NOT manufacture significance; with a fat planted alpha it must light up.

Pure numpy + pandas + stdlib on the offline path. ``fetch_all`` (network) is used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
MONTHLY_CACHE = os.path.join(CACHE_DIR, "ba_monthly.csv")

# The desk-wide frozen as-of: June 2026 is the last complete month on a 2026-07-03 run.
AS_OF = "2026-06-30"

# Repo-level Shiller cache (already fetched by earlier studies; raw.githubusercontent
# fallback lives in the desk's shared tooling). Used only inside fetch_all().
_SHILLER_CANDIDATES = [
    os.path.join(HERE, "..", "..", "..", "_cache", "_cache", "shiller_sp500.parquet"),
    os.path.join(HERE, "..", "..", "..", "_cache", "shiller_sp500.parquet"),
]


# --------------------------------------------------------------------------- #
# Real tape (network — used once to build the cache)
# --------------------------------------------------------------------------- #
def _monthly_tr(ticker: str, retries: int = 3) -> pd.Series:
    """Month-end adjusted close -> monthly total return (simple), tz-naive index."""
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            h = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if h is not None and len(h) > 0:
                px = h["Close"]
                break
        except Exception:
            time.sleep(2.0)
    px.index = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) for t in px.index])
    m = px.resample("ME").last()
    # drop the trailing partial month: if the tape stops before that month-end, the last
    # bucket is built from a truncated window
    if px.index.max() < m.index.max():
        m = m.iloc[:-1]
    return m.pct_change().dropna()


def fetch_all(path: str = MONTHLY_CACHE) -> pd.DataFrame:
    """Build and cache the single monthly frame (brk, mkt, mkt_src, rf, qual, usmv).

    Network-only; run once. Everything downstream (verify.py, notebooks) is cache-first.
    """
    import yfinance as yf

    brk = _monthly_tr("BRK-A")            # starts 1980-03 (partial) -> first full 1980-04
    spy = _monthly_tr("SPY")              # total return from 1993-02
    qual = _monthly_tr("QUAL")
    usmv = _monthly_tr("USMV")

    # ^GSPC month-end price return for the pre-SPY leg
    gspc_px = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)["Close"]
    gspc_px.index = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) for t in gspc_px.index])
    gspc_m = gspc_px.resample("ME").last().pct_change().dropna()

    # Shiller monthly dividend yield (annual dividend rate / 12 / price) for the pre-SPY leg
    sh = None
    for cand in _SHILLER_CANDIDATES:
        if os.path.exists(cand):
            sh = pd.read_parquet(cand)
            break
    if sh is None:
        raise FileNotFoundError("shiller_sp500.parquet not found in the repo _cache")
    sh = sh.copy()
    sh["Date"] = pd.to_datetime(sh["Date"])
    sh = sh.set_index("Date")
    div_y = (sh["Dividend"] / 12.0 / sh["SP500"]).replace([np.inf, -np.inf], np.nan)
    div_y.index = div_y.index.to_period("M")

    # spliced market total return: Shiller-augmented ^GSPC before SPY, SPY after
    first_spy = spy.index.min()
    pre = gspc_m[gspc_m.index < first_spy].copy()
    pre_y = pre.index.to_period("M").map(lambda p: div_y.get(p, np.nan))
    pre = pre + pd.Series(np.asarray(pre_y, dtype=float), index=pre.index).fillna(0.0)
    mkt = pd.concat([pre, spy]).sort_index()
    mkt_src = pd.Series(np.where(mkt.index < first_spy, "gspc+shiller_dy", "spy_tr"),
                        index=mkt.index)

    # risk-free: previous month-end ^IRX (13w T-bill discount yield, % annual) / 100 / 12
    irx = yf.Ticker("^IRX").history(period="max", auto_adjust=False)["Close"]
    irx.index = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) for t in irx.index])
    irx_m = irx.resample("ME").last() / 100.0 / 12.0
    rf = irx_m.shift(1)                    # the rate KNOWN at the start of each month

    df = pd.DataFrame({"brk": brk, "mkt": mkt, "rf": rf,
                       "qual": qual, "usmv": usmv})
    df["mkt_src"] = mkt_src
    df = df.dropna(subset=["brk", "mkt", "rf"])
    df = df[df.index <= pd.Timestamp(AS_OF)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    return df


def have_real(path: str = MONTHLY_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = MONTHLY_CACHE) -> pd.DataFrame:
    """Cached monthly frame, sliced to the frozen as-of (defensive re-slice)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df[df.index <= pd.Timestamp(AS_OF)]


# --------------------------------------------------------------------------- #
# Synthetic control (deterministic, offline)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 480, alpha_m: float = 0.0, beta: float = 0.7,
                    seed: int = 628, mkt_mu: float = 0.009, mkt_sig: float = 0.045,
                    idio_sig: float = 0.040, idio_ar1: float = 0.15) -> pd.DataFrame:
    """A seeded (mkt, rf, manager) monthly frame with a PLANTED CAPM alpha.

    manager_t = rf + beta*(mkt_t - rf) + alpha_m + e_t, where e is AR(1) idiosyncratic
    noise (the autocorrelation is what the HAC correction must absorb). ``alpha_m`` is the
    tunable planted effect; ``alpha_m = 0`` is the null. The index is a decorative monthly
    period_range (<= 250 years, the pandas ns-Timestamp CI trap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("1980-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    rf = np.full(n_months, 0.003)
    mkt = rng.normal(mkt_mu, mkt_sig, size=n_months)
    e = np.zeros(n_months)
    innov = rng.normal(0.0, idio_sig * np.sqrt(1 - idio_ar1 ** 2), size=n_months)
    for t in range(1, n_months):
        e[t] = idio_ar1 * e[t - 1] + innov[t]
    mgr = rf + beta * (mkt - rf) + alpha_m + e
    return pd.DataFrame({"brk": mgr, "mkt": mkt, "rf": rf}, index=idx)
