"""Data access for the hedging-pressure study — commodity futures, CFTC positioning, and where they come from.

The hedging-pressure premium (Keynes' normal backwardation; Cootner 1960; Gorton-Hayashi-Rouwenhorst
2013) says: when commercial *hedgers* in a commodity are heavily net **short** (producers selling forward
to lock in prices), speculators must take the other side net **long**, and are paid a risk premium for
bearing that price risk. The signal is therefore the **hedging pressure** read off the CFTC Commitments
of Traders (COT) report, and the asset is the commodity futures return. The data layer keeps the desk's
offline/cache split:

    * :func:`synthetic_commodities` — fully **offline**. A panel of synthetic commodities, each with a
      time-varying hedging-pressure signal that *predicts* its next return (the baked premium) plus a
      no-predictability null. ``hp_strength`` sets the premium; ``hp_strength = 0`` is the null.
    * :func:`fetch_cot` — download the CFTC **legacy futures-only COT** (free, no key), filter to a
      mapped commodity basket, and compute weekly hedging pressure ``HP = (CommShort − CommLong) /
      (CommShort + CommLong)``. **Cache-only** unless ``fetch=True`` (needs network).
    * :func:`fetch_futures` — commodity futures weekly returns from Yahoo (continuous front-month).
      Cache-only unless ``fetch=True``.

Data choice: weekly COT (Tuesday positions, published Friday) lagged so the signal is tradable, against
the next week's futures return; a small liquid commodity basket. Stated, not hidden.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
WEEKS_PER_YEAR = 52

# Exact CFTC legacy market name -> Yahoo continuous front-month futures ticker.
COT_MARKET = {
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "COPPER- #1 - COMMODITY EXCHANGE INC.": "HG=F",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
    "CORN - CHICAGO BOARD OF TRADE": "ZC=F",
    "WHEAT-SRW - CHICAGO BOARD OF TRADE": "ZW=F",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "ZS=F",
    "SUGAR NO. 11 - ICE FUTURES U.S.": "SB=F",
    "COFFEE C - ICE FUTURES U.S.": "KC=F",
    "COTTON NO. 2 - ICE FUTURES U.S.": "CT=F",
    "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE": "LE=F",
}


@dataclass(frozen=True)
class PanelTruth:
    n_comm: int
    n_weeks: int
    hp_strength: float

    @property
    def has_premium(self) -> bool:
        return self.hp_strength != 0.0


def synthetic_commodities(n_comm: int = 12, n_weeks: int = 1040, hp_strength: float = 0.0045,
                          hp_persist: float = 0.95, vol: float = 0.035, seed: int = 0
                          ) -> tuple[pd.DataFrame, pd.DataFrame, PanelTruth]:
    """A panel of synthetic commodities whose **hedging pressure predicts the next return**, by construction.

    Each commodity has a slow, persistent hedging-pressure signal ``hp_{i,t}`` (an AR(1) in [-1,1]-ish),
    and its weekly return is ``r_{i,t} = hp_strength * hp_{i,t-1} + vol * eps`` — so a commodity whose
    commercials are net short (hp high) earns a positive expected return next week (the premium accruing
    to the long speculator). ``hp_strength = 0`` is the null (hp carries no information). Returns
    ``(returns, hedging_pressure, truth)`` as weekly ``weeks x commodity`` frames.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2005-01-04", periods=n_weeks, freq="W-TUE", name="date")
    hp = np.zeros((n_weeks, n_comm))
    innov = np.sqrt(1.0 - hp_persist**2)
    nu = rng.standard_normal((n_weeks, n_comm))
    for t in range(1, n_weeks):
        hp[t] = np.clip(hp_persist * hp[t - 1] + innov * nu[t], -3, 3)
    rets = hp_strength * np.vstack([np.zeros((1, n_comm)), hp[:-1]]) + vol * rng.standard_normal((n_weeks, n_comm))
    cols = [f"CMD{i:02d}" for i in range(n_comm)]
    return (pd.DataFrame(rets, index=idx, columns=cols),
            pd.DataFrame(hp, index=idx, columns=cols),
            PanelTruth(n_comm=n_comm, n_weeks=n_weeks, hp_strength=hp_strength))


def align_hp(hp: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Reindex the (weekly) hedging-pressure panel onto the futures-return index, carrying forward.

    COT positions are stamped on the report's Tuesday and the futures returns on a week-ending grid, so
    the two indices rarely match exactly; a forward-fill puts the most recent *known* hedging pressure
    against each return week (and never uses future positioning). Returns hp restricted to the basket
    columns present in both.
    """
    cols = [c for c in hp.columns if c in returns.columns]
    return hp[cols].reindex(returns.index, method="ffill")


def _getzip(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def fetch_cot(years=range(2010, 2025), cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Return a weekly ``date x ticker`` hedging-pressure panel from the CFTC legacy COT, cache-first.

    ``HP = (CommShort − CommLong) / (CommShort + CommLong)`` per mapped commodity (positive = commercials
    net short = hedging demand). **Cache-only** unless ``fetch=True``; fetching downloads each year's
    ``deacot<YYYY>.zip`` from cftc.gov.
    """
    cache = os.path.join(cache_dir, "cot_hedging_pressure.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    frames = []
    for y in years:
        try:
            z = zipfile.ZipFile(io.BytesIO(_getzip(f"https://www.cftc.gov/files/dea/history/deacot{y}.zip")))
            df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        except Exception:
            continue
        mc = "Market and Exchange Names"
        # the proper ISO date column (avoid the YYMMDD-integer "As of Date in Form YYMMDD" column)
        date_col = next((c for c in df.columns if "Report_Date_as_YYYY-MM-DD" in c), None) \
            or next((c for c in df.columns if "YYYY-MM-DD" in c), None)
        cl = "Commercial Positions-Long (All)"; cs = "Commercial Positions-Short (All)"
        if mc not in df.columns or cl not in df.columns or date_col is None:
            continue
        df = df[df[mc].isin(COT_MARKET)][[mc, date_col, cl, cs]].copy()
        df["date"] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        df["ticker"] = df[mc].map(COT_MARKET)
        tot = df[cs] + df[cl]
        df["hp"] = (df[cs] - df[cl]) / tot.where(tot > 0)
        frames.append(df[["date", "ticker", "hp"]].dropna())
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames, ignore_index=True)
    panel = allc.pivot_table(index="date", columns="ticker", values="hp").sort_index()
    panel.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(cache)
    return panel


def fetch_futures(tickers=None, cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Return weekly (W-TUE) ``date x ticker`` futures returns for the commodity basket, cache-first."""
    cache = os.path.join(cache_dir, "commodity_futures_weekly.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[[t for t in (tickers or df.columns) if t in df.columns]]
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy
    tickers = tickers or list(dict.fromkeys(COT_MARKET.values()))
    closes = {}
    for tk in tickers:
        try:
            raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            closes[tk] = raw["Close"].dropna()
        except Exception:
            continue
    if not closes:
        return pd.DataFrame()
    prices = pd.DataFrame(closes).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    weekly = prices.resample("W-TUE").last()
    rets = weekly.pct_change().dropna(how="all")
    rets.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(cache)
    return rets
