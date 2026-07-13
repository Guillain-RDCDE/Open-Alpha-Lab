"""Data layer for Study 744 — Tetraphobia (the fear of the number 4).

In Mandarin, Cantonese, Japanese and Korean the word for "four" is a near-homophone
of the word for "death" (sì / sei / shi). The superstition — *tetraphobia* — is real
and everywhere: buildings skip the 4th, 14th and 24th floors, licence plates and phone
numbers ending in 4 sell at a discount, "8" (a homophone of *prosperity*) at a premium.
The market-folklore version has two testable halves, and this study builds the data for
both:

* **The clustering half (does the superstition shape *where prices settle*?).** The
  academic finding — Brown & Mitchell (2008), Bhattacharya et al. (2018) — is that the
  *trailing digit* of stock prices in Greater China avoids 4 and prefers 8. To test it
  on live tape we need **raw, un-adjusted, local-currency** closes (a split/dividend
  adjustment destroys the actual traded price's last digit), for a basket of tickers
  from tetraphobic markets — Taiwan (``.TW``), Hong Kong (``.HK``), mainland China
  A-shares (``.SS`` / ``.SZ``), Korea (``.KS``) — against a **US control basket** priced
  in a culture with no such superstition. The control IS the placebo: whatever round-
  number clustering (digits 0 and 5) both baskets share is not tetraphobia; only an
  *asymmetry between 4 and 8 that appears in Asia and not in the US* is.

* **The returns half (does the superstition shape *when returns happen*?).** The tradable
  folklore says a maximally-unlucky calendar date — **4/4**, the "double-death" day —
  should see the relevant markets sell off. We fetch **total-return** ETF closes for the
  three cleanest China-sphere vehicles (``EWT`` Taiwan, ``EWH`` Hong Kong, ``MCHI``
  China), plus an extended set (``EWY`` Korea, ``EWJ`` Japan, ``FXI`` China large-cap),
  and measure the 4/4 session return against a random-calendar placebo. **8/8** (the
  "lucky" double-prosperity date) is carried as the natural contrast.

Two synthetic worlds (deterministic, seeded) are the positive controls: a digit-stream
with a tunable planted 4-deficit, and a daily tape with a tunable planted 4/4 dip. With
the knob at zero, neither detector may manufacture significance.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-13)
CLUSTER_START = "2010-01-01"  # trailing-digit basket window (raw local-currency closes)
CAL_START = "2000-01-01"      # calendar-returns window (total-return ETF closes)

# --------------------------------------------------------------------------- #
# The clustering baskets. Local-currency, integer-ish price levels so the price has a
# meaningful trailing digit. Region tags let the notebook show that the effect is
# strongest exactly where tetraphobia is strongest (Taiwan, mainland China) and weakest
# in the most internationalised market (Hong Kong).
# --------------------------------------------------------------------------- #
ASIA_CLUSTER: dict[str, str] = {
    "2330.TW": "Taiwan",   # TSMC
    "2317.TW": "Taiwan",   # Hon Hai / Foxconn
    "2454.TW": "Taiwan",   # MediaTek
    "0700.HK": "HongKong", # Tencent
    "0005.HK": "HongKong", # HSBC
    "0941.HK": "HongKong", # China Mobile
    "600519.SS": "ChinaA", # Kweichow Moutai (Shanghai)
    "601398.SS": "ChinaA", # ICBC (Shanghai)
    "000001.SZ": "ChinaA", # Ping An Bank (Shenzhen)
    "005930.KS": "Korea",  # Samsung Electronics (won-priced, no sub-unit digit)
}

# US control — a culture with no "4 = death" homophone. Same methodology; the ONLY
# fair placebo for whatever generic round-number clustering both baskets share.
US_CONTROL: list[str] = ["AAPL", "MSFT", "JPM", "XOM", "KO",
                         "PG", "WMT", "JNJ", "CVX", "HD"]

# --------------------------------------------------------------------------- #
# The calendar-returns instruments. Core = the three cleanest China-sphere ETFs; the
# extended set widens the net (and shows the null is not an artefact of three tickers).
# --------------------------------------------------------------------------- #
CALENDAR_CORE = ["EWT", "EWH", "MCHI"]
CALENDAR_EXT = ["EWY", "EWJ", "FXI"]
CALENDAR_ALL = CALENDAR_CORE + CALENDAR_EXT

CAL_YEARS = tuple(range(2000, 2026))   # 2000..2025 inclusive


# --------------------------------------------------------------------------- #
# A small, CLEARLY-LABELLED literature proxy. NOT the study's evidence (the real-tape
# trailing-digit computation is) — an illustrative anchor for the qualitative finding
# reported by Brown & Mitchell (2008) for Chinese A-share prices: 8 the most common
# trailing digit, 4 among the least. Stylised to that ordering, summing to 100%; used
# only to sanity-check the SHAPE of the real-tape result, never quoted as a measurement.
# --------------------------------------------------------------------------- #
LIT_PROXY_TRAILING_DIGIT = {  # digit -> stylised % (illustrative, per B&M 2008 ordering)
    0: 14.0, 1: 8.5, 2: 9.5, 3: 9.0, 4: 6.5,
    5: 12.0, 6: 9.5, 7: 9.5, 8: 13.5, 9: 8.0,
}


def _cluster_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"tetra_cl_{ticker.lower().replace('.', '_')}.csv")


def _cal_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"tetra_cal_{ticker.lower()}.csv")


def all_cluster_tickers() -> list[str]:
    return list(ASIA_CLUSTER.keys()) + US_CONTROL


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(end: str = "2026-07-01") -> None:
    """Download both tapes and cache them. Network; run once.

    Clustering basket: ``auto_adjust=False`` — the RAW traded close, whose trailing
    digit is the choice the superstition acts on (an adjusted price is a back-computed
    number no human ever traded, and its last digit is meaningless for this test).
    Price-only, local currency, by design.

    Calendar ETFs: ``auto_adjust=True`` — total-return, since the returns half is an
    ordinary return study where dividends belong in the number.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    for t in all_cluster_tickers():
        d = yf.download(t, start=CLUSTER_START, end=end,
                        auto_adjust=False, progress=False)
        if d is None or len(d) == 0:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d[["Close"]].dropna().to_csv(_cluster_path(t))

    for t in CALENDAR_ALL:
        d = yf.download(t, start=CAL_START, end=end,
                        auto_adjust=True, progress=False)
        if d is None or len(d) == 0:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d[["Close"]].dropna().to_csv(_cal_path(t))


def have_real() -> bool:
    cl = all(os.path.exists(_cluster_path(t)) for t in all_cluster_tickers())
    ca = all(os.path.exists(_cal_path(t)) for t in CALENDAR_ALL)
    return cl and ca


def load_cluster(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: raw local-currency close}, sliced to <= asof."""
    out = {}
    for t in all_cluster_tickers():
        p = _cluster_path(t)
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


def load_calendar(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return close} for the calendar ETFs, sliced to <= asof."""
    out = {}
    for t in CALENDAR_ALL:
        p = _cal_path(t)
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


def calendar_panel(asof: str = AS_OF) -> pd.DataFrame:
    """The six calendar ETFs as one aligned frame — the object the fingerprint stamps."""
    prices = load_calendar(asof)
    return pd.DataFrame({t: prices[t] for t in CALENDAR_ALL if t in prices})


# --------------------------------------------------------------------------- #
# Synthetic world 1 — a digit stream with a tunable planted 4-deficit (control for the
# clustering detector). bias = 0 is the null: every non-zero, non-five digit equally
# likely, no tetraphobia.
# --------------------------------------------------------------------------- #
def synthetic_digits(bias: float = 0.0, seed: int = 744, n: int = 40000
                     ) -> np.ndarray:
    """Draw ``n`` trailing digits 0..9. Round digits 0 and 5 are over-weighted (the
    universal round-number effect, present in every market); ``bias`` then moves mass
    from digit 4 to digit 8 (the tetraphobia asymmetry). ``bias = 0`` -> 4 and 8 equal.
    """
    rng = np.random.default_rng(seed)
    w = np.ones(10)
    w[0] = w[5] = 3.0                    # round-number clustering, culture-neutral
    w[8] += bias                          # prosperity pull
    w[4] = max(w[4] - bias, 0.01)         # death aversion
    w = w / w.sum()
    return rng.choice(10, size=n, p=w)


# --------------------------------------------------------------------------- #
# Synthetic world 2 — a daily tape with a tunable planted 4/4 dip (control for the
# calendar detector). dip = 0 is the null: 4/4 is an ordinary day.
# --------------------------------------------------------------------------- #
def synthetic_calendar(dip: float = 0.0, seed: int = 744,
                       daily_vol: float = 0.012) -> tuple[pd.Series, list]:
    """A reproducible daily tape 2000..2025 with a planted extra return of ``dip`` on the
    first session on/after 4 April each year. ``dip = 0`` -> 4/4 statistically identical
    to every other day. Returns (close Series, list of 4/4 session Timestamps).
    """
    idx = pd.bdate_range("2000-01-03", "2025-12-31")
    rng = np.random.default_rng(seed)
    logret = rng.normal(0.0, daily_vol, len(idx))
    ev = []
    for y in CAL_YEARS:
        pos = idx.searchsorted(pd.Timestamp(f"{y}-04-04"))
        if pos < len(idx):
            logret[pos] += dip
            ev.append(idx[pos])
    close = pd.Series(100.0 * np.exp(np.cumsum(logret)), index=idx)
    return close, ev
