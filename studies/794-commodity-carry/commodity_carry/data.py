"""Data layer for Study 794 — Commodity Carry.

Three ingredients, all offline-friendly once cached:

* **Real futures term structure (the carry SIGNAL).** EIA v2 daily near-month energy
  futures: WTI ``RCLC1..RCLC4`` ($/bbl) and Henry Hub natural gas ``RNGC1..RNGC4``
  ($/MMBtu). The front-vs-deferred slope of these curves is the *ex-ante* carry (roll
  yield) — known at each date's settle, no look-ahead. Cached as ``cc_energy_curve.csv``.

* **Investable ETF returns (the OUTCOME).** yfinance total-return closes for the
  front-month funds (USO = WTI front, UNG = gas front) and the 12-month-laddered
  counterparts (USL, UNL). The front funds are the returns we sort; the
  front-minus-laddered pair (USO vs USL) is the direct roll-drag proxy. Cached as
  ``cc_etfs.csv``.

* **Synthetic world (the positive control).** A deterministic, seeded two-commodity
  panel with a TUNABLE planted carry premium (knob ``premium``): the more-backwardated
  name earns an extra ``premium`` per period. ``premium = 0`` is the null — the
  cross-sectional carry sort must NOT manufacture significance from it.

PROXY LIMITATION, stated plainly: broad, clean historical commodity term structure is
not freely available, so this is a **two-name energy proxy** (WTI, natural gas) for the
general cross-sectional commodity carry premium. A two-name cross-section is the thinnest
possible; treat the cross-sectional numbers as underpowered by construction and lean on
the pooled panel test.

Pure numpy + pandas + stdlib on the offline path. ``fetch_*`` (network) run once to
build the cache and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
CURVE_CACHE = os.path.join(CACHE_DIR, "cc_energy_curve.csv")
ETF_CACHE = os.path.join(CACHE_DIR, "cc_etfs.csv")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-22)
UA = {"User-Agent": "OpenAlphaLab research"}

# Commodity -> (EIA curve prefix, front ETF, laddered ETF)
COMMODITIES = {
    "WTI": {"prefix": "WTI", "front_etf": "USO", "ladder_etf": "USL"},
    "GAS": {"prefix": "HH", "front_etf": "UNG", "ladder_etf": "UNL"},
}
# The carry SIGNAL start: ETFs (the returns) begin 2006-04 (USO/USL) and 2007-04 (UNG),
# so the joint monthly panel effectively starts mid-2007.


# --------------------------------------------------------------------------- #
# Real futures term structure (EIA) — the carry signal
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 30, tries: int = 4) -> bytes:
    last = None
    for _ in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise last


def _eia_series(route: str, series_id: str, key: str = "DEMO_KEY",
                start: str = "2005-01-01") -> pd.Series:
    """One EIA daily near-month futures series -> a date-indexed price Series.

    ``start`` trims the download (the ETF outcome series only begin 2006-2007, so the
    joint panel never uses pre-2005 curve data) — this keeps the request count inside
    the free DEMO_KEY rate budget."""
    rows, offset = [], 0
    while True:
        url = (f"https://api.eia.gov/v2/{route}/data/?api_key={key}&frequency=daily"
               f"&data[0]=value&facets[series][]={series_id}&start={start}"
               f"&sort[0][column]=period&sort[0][direction]=asc&offset={offset}&length=5000")
        j = json.loads(_get(url))
        data = j["response"]["data"]
        if not data:
            break
        rows += [(d["period"], d["value"]) for d in data]
        if len(data) < 5000:
            break
        offset += 5000
        time.sleep(0.4)
    s = pd.Series({pd.Timestamp(p): (float(v) if v is not None else np.nan)
                   for p, v in rows})
    return s[s.notna()].sort_index()


def fetch_curve(contracts: tuple[int, ...] = (1, 2), pace_s: float = 90.0) -> None:
    """Download the WTI + Henry Hub near-month futures curves; cache. Network; once.

    Only the front two contracts (C1, C2) are fetched by default — the carry signal is
    ``ln(C1/C2)`` and two series per commodity keeps the request count inside the free
    EIA ``DEMO_KEY`` rate budget. ``pace_s`` spaces requests to avoid the burst
    cooldown. Raises if a series comes back empty (so a rate-limited run fails loudly
    rather than caching a hollow curve)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cols = {}
    spec = [("petroleum/pri/fut", "RCLC", "WTI"), ("natural-gas/pri/fut", "RNGC", "HH")]
    for route, root, tag in spec:
        for n in contracts:
            s = _eia_series(route, f"{root}{n}")
            if len(s) == 0:
                raise RuntimeError(f"EIA {root}{n} returned no data (rate-limited?)")
            cols[f"{tag}{n}"] = s
            time.sleep(pace_s)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df.to_csv(CURVE_CACHE)


def fetch_etfs() -> None:
    """Download total-return closes for the four energy ETFs; cache. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    tks = ["USO", "USL", "UNG", "UNL"]
    h = yf.download(tks, period="max", interval="1d", progress=False, auto_adjust=True)
    close = h["Close"] if hasattr(h.columns, "levels") else h
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.index.name = "date"
    close = close[[c for c in tks if c in close.columns]].dropna(how="all")
    close.to_csv(ETF_CACHE)


def fetch() -> None:
    fetch_curve()
    fetch_etfs()


def have_real() -> bool:
    return os.path.exists(CURVE_CACHE) and os.path.exists(ETF_CACHE)


def load_curve(asof: str = AS_OF) -> pd.DataFrame:
    df = pd.read_csv(CURVE_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[df.index <= asof].copy()


def load_etfs(asof: str = AS_OF) -> pd.DataFrame:
    df = pd.read_csv(ETF_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[df.index <= asof].copy()


# --------------------------------------------------------------------------- #
# Synthetic world — planted cross-sectional carry premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(premium: float = 0.0, seed: int = 794, n_months: int = 220,
                    vol: float = 0.09, carry_vol: float = 0.30,
                    ) -> pd.DataFrame:
    """Deterministic two-commodity monthly panel with a TUNABLE planted carry premium.

    Each month, each of the two synthetic commodities draws a carry (annualized roll
    yield, mean 0, sd ``carry_vol``) and a monthly return = idiosyncratic noise
    (sd ``vol``) PLUS ``premium * (carry - cross_sectional_mean_carry)`` — i.e. the
    more-backwardated name of the pair earns an extra ``premium`` slice, the more-
    contangoed one loses it. ``premium = 0`` is the null world: carry carries no
    information and the cross-sectional sort must NOT reach significance.

    Returns a long frame with columns ``[month, commodity, carry, ret]`` — the exact
    shape the real panel is reduced to, so the same estimator runs on both.
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range("2007-07", periods=n_months, freq="M")
    recs = []
    for m in months:
        carries = rng.normal(0.0, carry_vol, 2)
        xmean = carries.mean()
        for j, name in enumerate(("A", "B")):
            noise = rng.normal(0.0, vol)
            ret = noise + premium * (carries[j] - xmean)
            recs.append({"month": m.to_timestamp("M"), "commodity": name,
                         "carry": float(carries[j]), "ret": float(ret)})
    return pd.DataFrame(recs)
