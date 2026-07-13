"""Data layer for Study 766 — Memecoin-Season.

The folklore: in euphoric "memecoin seasons" the dog-coins — DOGE and SHIB — don't just
rally, they *blow past* Bitcoin by one, two, three orders of magnitude, and a nimble
momentum rotation can hop onto whichever coin is hot and harvest the mania net of costs.

Two ingredients, both **real tape** (no proxy needed — all three trade on yfinance):

* **BTC-USD, DOGE-USD, SHIB-USD** daily closes from yfinance (no key), cached as CSV.
  Price-only == total-return for crypto (no dividends). BTC goes back to 2014; DOGE to
  Nov-2017; SHIB only to Aug-2020 (it launched then). The rotation universe therefore only
  exists in full from **2020-08** onward — which is, not coincidentally, exactly the window
  the "memecoin season" story is told about (the 2021 mania).

* **The survivorship label — named here because it is the whole ballgame.** DOGE and SHIB are
  the *two memecoins that survived* out of literally thousands launched in 2020-2021 (the vast
  majority went to zero within months). Backtesting a rotation on the two ex-post winners is
  survivorship bias in its purest form: in real time you did not know DOGE and SHIB would be
  the survivors, you were staring at ten thousand coins that mostly evaporated. Every return
  in this study is therefore an *upper bound* that a real-time trader could not have banked —
  we say so on the Signal axis, loudly, wherever a number appears.

Pure numpy + pandas + stdlib on the offline path once cached. ``fetch_all`` (network) runs
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07-12)

# Rotation universe: Bitcoin plus the two surviving large-cap memecoins.
TICKERS = {"BTC": "BTC-USD", "DOGE": "DOGE-USD", "SHIB": "SHIB-USD"}
ASSETS = ["BTC", "DOGE", "SHIB"]

CACHE = {k: os.path.join(CACHE_DIR, f"ms_{k.lower()}_usd.csv") for k in TICKERS}


# --------------------------------------------------------------------------- #
# Real tape — daily closes, one CSV per asset
# --------------------------------------------------------------------------- #
def fetch_all(start: str = "2013-01-01", end: str | None = None) -> None:
    """Download daily closes for BTC/DOGE/SHIB and cache them. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for key, ticker in TICKERS.items():
        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px = px.dropna()
        px.index = pd.DatetimeIndex(px.index).tz_localize(None)
        px.name = key.lower()
        px.to_frame().to_csv(CACHE[key])


def have_real() -> bool:
    return all(os.path.exists(p) for p in CACHE.values())


def _load_one(key: str, asof: str) -> pd.Series:
    px = pd.read_csv(CACHE[key], index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    px = px[px.index <= pd.Timestamp(asof)]
    px.name = key
    return px


def load_prices(asof: str = AS_OF) -> pd.DataFrame:
    """Daily closes for BTC/DOGE/SHIB, aligned on the window where **all three** exist.

    The common window starts the day all three first carry a **positive** price. SHIB is the
    binding constraint: yfinance stores its sub-1e-10 launch price as literal ``0.0`` for its
    first ~8 months, so the tradable memecoin universe only truly exists from ~2021 — which is
    itself the honest headline (the "memecoin season" data set is a handful of years dominated
    by one mania). Sliced to the study as-of so the sample never creeps. Forward-fills the odd
    single missing day (crypto trades every calendar day; a rare gap shouldn't drop the whole
    cross-section) but never back-fills before an asset's first real print.
    """
    cols = {k: _load_one(k, asof) for k in ASSETS}
    df = pd.DataFrame(cols).where(lambda x: x > 0)      # non-positive == not yet real
    first_all = max(df[c].first_valid_index() for c in df.columns)
    df = df[df.index >= first_all].sort_index()
    df = df.ffill().dropna(how="any")
    return df[ASSETS]


def weekly_prices(asof: str = AS_OF) -> pd.DataFrame:
    """Friday-close weekly bars (the rotation rebalances weekly).

    Weekly bars tame the day-to-day noise of memecoins and make weekly-turnover costs the
    honest accounting unit. A partial final week is dropped so a stamped run never includes an
    in-progress bar.
    """
    daily = load_prices(asof)
    wk = daily.resample("W-FRI").last().dropna(how="any")
    # drop a trailing partial week (its Friday is after the as-of)
    return wk[wk.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world — planted momentum persistence in ONE asset
# --------------------------------------------------------------------------- #
def synthetic_world(n_weeks: int = 300, persistence: float = 0.0, seed: int = 766,
                    base_ann: float = 0.20, vol_ann: float = 1.20
                    ) -> pd.DataFrame:
    """Deterministic 3-asset weekly-return world with TUNABLE momentum persistence.

    Builds three crypto-like weekly return streams (one "BTC-like" at lower vol, two
    "memecoin-like" at higher vol). When ``persistence > 0``, the two memecoin streams get an
    AR(1)-style momentum component: last week's shock partially carries into this week, so a
    trailing-momentum rotation *should* be able to harvest it. ``persistence = 0`` is the null —
    pure random walks, and a momentum rotation must NOT beat a naive equal-weight basket by any
    meaningful margin. Returns a weekly-return DataFrame (columns A/B/C), no timestamps
    dependence, fully reproducible from ``seed``.
    """
    rng = np.random.default_rng(seed)
    mu_w = base_ann / 52.0
    sig_lo = vol_ann / np.sqrt(52.0) * 0.5      # "BTC-like"
    sig_hi = vol_ann / np.sqrt(52.0)            # "memecoin-like"
    cols = {}
    for j, sig in enumerate([sig_lo, sig_hi, sig_hi]):
        shocks = rng.standard_normal(n_weeks)
        if persistence > 0 and j >= 1:          # plant momentum in the two "memecoins"
            r = np.zeros(n_weeks)
            prev = 0.0
            for i in range(n_weeks):
                r[i] = persistence * prev + shocks[i]
                prev = r[i]
            shocks = r
        cols[chr(ord("A") + j)] = mu_w + sig * shocks
    return pd.DataFrame(cols)
