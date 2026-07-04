"""Data layer for Study 640 — Gold-Overnight ("Gold Trades in Its Sleep").

Two sources, both offline-friendly once cached:

* **Real tape.** Daily **open + close** (yfinance, auto-adjusted / total-return) for
  ``GLD`` (SPDR Gold Shares, the claim's home instrument, listed 2004-11-18), ``IAU``
  (iShares Gold Trust, the confirmation tape — a different sponsor/custodian tracking the
  same metal), and ``SPY`` (the equity placebo — study 01 showed US equities carry their own
  overnight effect, so gold's must be judged *against* it, not in a vacuum). From open and
  close we split every session into the **overnight** leg (prior close → today's open, the
  hours when US exchanges are shut but gold trades in Asia and into the London AM fix) and
  the **intraday** leg (open → close, the London-PM-fix / NY window). Cached wide as
  ``go_ohlc.csv`` with columns ``{TICKER}_open`` / ``{TICKER}_close``.

* **Synthetic.** A deterministic, fixed-seed world generator with a **tunable planted
  overnight drift** (knob ``night_edge``): daily close-to-close returns are built from a
  night piece with mean ``night_edge`` and a day piece with mean ``day_edge``, then rolled
  into open/close prices. With both edges 0 (the null) the night-vs-day split must NOT
  manufacture significance; with a planted night edge the machinery must light up. Span is
  ~20 years — far under the 250-year ns-Timestamp ceiling.

Pure numpy + pandas + stdlib on the offline path; ``fetch_panel`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.

Adjustment note (a house rule: say it, because it moves return between night and day):
prices are yfinance ``auto_adjust=True`` — **total-return, split-adjusted** opens and
closes. GLD has never split and pays no dividend; IAU's 2021-05-24 1-for-2 reverse split is
adjusted on both open and close, so the night/day split is not distorted by the corporate
action. SPY's dividends land in the adjusted close (the usual convention), which if anything
*flatters* the equity placebo's overnight leg — a conservative choice for the "is gold's
bigger?" comparison.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
OHLC_CACHE = os.path.join(CACHE_DIR, "go_ohlc.csv")

TICKERS = ["GLD", "IAU", "SPY"]
START = "2004-11-18"          # GLD's first session

# The London Gold Fix (twice-daily, phone-based, five bullion banks, est. 1919) was replaced
# by the electronic, auditable LBMA Gold Price auction (ICE Benchmark Administration) on
# 2015-03-20, after the 2014 Barclays fix-manipulation fine and the FCA/IOSCO benchmark
# reforms. The claim parks the anomaly around the fixes, so this is the pre-registered,
# externally-dated structural break — not a snooped split.
FIX_REFORM = "2015-03-20"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = START, end: str | None = None,
                path: str = OHLC_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download daily adjusted open/close for GLD, IAU, SPY and cache them wide.

    Network-only; used once to build the cache. Guards yfinance flakiness with simple
    retries. Writes one wide CSV with columns ``{TICKER}_open`` / ``{TICKER}_close``.
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    out = {}
    for tk in TICKERS:
        out[f"{tk}_open"] = raw["Open"][tk]
        out[f"{tk}_close"] = raw["Close"][tk]
    wide = pd.DataFrame(out).dropna(how="all")
    wide.index = pd.DatetimeIndex(wide.index).tz_localize(None).normalize()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wide.to_csv(path)
    return wide


def have_real(path: str = OHLC_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = OHLC_CACHE) -> pd.DataFrame:
    """Cached wide open/close frame (index = date, columns = ``{TICKER}_{open|close}``)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def session_returns(wide: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Split one ticker's tape into overnight and intraday simple returns.

    * ``overnight``  = today's open / yesterday's close − 1   (exchange shut; Asia + London AM)
    * ``intraday``   = today's close / today's open − 1        (London PM fix + NY hours)
    * ``close2close``= today's close / yesterday's close − 1   (what a holder actually earns)

    The two legs compound exactly to close-to-close: (1+on)(1+id) = 1+cc. Rows with a
    missing open or close (or no prior close) are dropped, so every kept day has both legs.
    """
    o = wide[f"{ticker}_open"].astype(float)
    c = wide[f"{ticker}_close"].astype(float)
    df = pd.DataFrame({"open": o, "close": c}).dropna()
    on = df["open"] / df["close"].shift(1) - 1.0
    idr = df["close"] / df["open"] - 1.0
    cc = df["close"] / df["close"].shift(1) - 1.0
    out = pd.DataFrame({"overnight": on, "intraday": idr, "close2close": cc}).dropna()
    # guard against corrupt rows (a zero/garbage open prints an absurd leg)
    return out[(out.abs() < 0.5).all(axis=1)]


# --------------------------------------------------------------------------- #
# Synthetic control world
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 5000, night_edge: float = 0.0, day_edge: float = 0.0,
                    seed: int = 640, sig_night: float = 0.007,
                    sig_day: float = 0.008) -> pd.DataFrame:
    """Deterministic open/close tape with a PLANTED overnight drift (the tunable effect).

    Each day draws an overnight return ~ N(night_edge, sig_night) and an intraday return
    ~ N(day_edge, sig_day); prices roll as close_t = close_{t-1}(1+on)(1+id) with
    open_t = close_{t-1}(1+on). With ``night_edge = day_edge = 0`` (the null) the split
    must NOT manufacture a significant night-vs-day gap however the noise falls; with a
    planted ``night_edge`` the detector must recover it. Business-day index from 2005 —
    ~20 years, far below the 250-year ns-Timestamp ceiling.

    Returns a wide frame with columns ``SYN_open`` / ``SYN_close`` (same shape contract as
    ``load_real``), so the whole real-tape pipeline runs on it unchanged.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    on = rng.normal(night_edge, sig_night, size=n_days)
    idr = rng.normal(day_edge, sig_day, size=n_days)
    close = np.empty(n_days)
    opn = np.empty(n_days)
    prev = 100.0
    for i in range(n_days):
        opn[i] = prev * (1.0 + on[i])
        close[i] = opn[i] * (1.0 + idr[i])
        prev = close[i]
    return pd.DataFrame({"SYN_open": opn, "SYN_close": close}, index=idx)
