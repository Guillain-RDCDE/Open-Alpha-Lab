"""Real-tape energy roll yield — measured from ETF pairs, no futures-curve feed required.

The roll yield a front-month commodity future pays is **observable without a term-structure data feed** by
contrasting two ETFs on the *same* underlying that differ only in *where on the curve* they sit:

  * **WTI crude** — ``USO`` (front-month) vs ``USL`` (12-month laddered, United States 12 Month Oil Fund)
  * **Natural gas** — ``UNG`` (front-month) vs ``UNL`` (12-month laddered, US 12 Month Natural Gas Fund)

The laddered fund spreads its exposure across the first twelve contracts, so it is far less exposed to the
front-month roll. The weekly return spread ``laddered − front`` is therefore the realized **roll cost of the
front contract**: positive in **contango** (front bleeds as it rolls *down* the curve), negative in
**backwardation** (front rolls *up*). This is exactly the mechanism §9.1 of Kakushadze–Serur describes, and
the famous ``USO`` bleed is its real-world fingerprint. Both ETFs are liquid and have clean yfinance history
back to 2007–2010, so this real run needs **no FRED, no EIA, no paid curve** — the desk's reliable workhorse.

Two reads:

  * :func:`bleed_table` — the descriptive fact: how much the front-month roll costs per commodity (the
    laddered-minus-front drag), with a Newey–West *t* on the spread.
  * :func:`roll_timing_book` — the strategy: hold the front contract only when the curve is **backwardated**
    (recent roll positive), short it when **contangoed** — a time-series roll-yield carry book per commodity
    and equal-weighted across the two. With only two liquid energy curves the cross-section is too thin for a
    bucket sort (that machinery is proved on the synthetic panel in :mod:`contango.data`), so the real tape is
    timed per curve.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .strategy import summary

# commodity -> (front-month ETF, 12-month laddered ETF)
PAIRS = {"WTI": ("USO", "USL"), "GAS": ("UNG", "UNL")}
ALL_TICKERS = sorted({t for pair in PAIRS.values() for t in pair})

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
ETF_CACHE = os.path.join(DEFAULT_CACHE, "energy_carry_etfs.parquet")

WEEKS_PER_YEAR = 52
DEFAULT_LOOKBACK = 13          # ~one quarter of weekly bars to read the curve regime
AS_OF = "2026-06-05"           # pin the sample so headline numbers don't drift with the calendar


def fetch_energy_pairs(cache: str = ETF_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Weekly (W-FRI) adjusted closes for the four energy ETFs, cache-first.

    Returns the cached frame if present; with ``fetch=True`` downloads from yfinance (period='max',
    auto-adjusted), resamples to weekly, writes the cache and returns it. Returns an empty frame if the
    cache is absent and ``fetch`` is False.
    """
    if os.path.exists(cache) and not fetch:
        df = pd.read_parquet(cache)
        df.index.name = "date"
        return df
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf

    h = yf.download(ALL_TICKERS, period="max", interval="1d", progress=False, auto_adjust=True)
    px = (h["Close"] if isinstance(h.columns, pd.MultiIndex) else h).copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    wk = px.resample("W-FRI").last().dropna(how="all")
    wk = wk[[t for t in ALL_TICKERS if t in wk.columns]]
    wk.index.name = "date"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    wk.to_parquet(cache)
    return wk


def load_pairs(cache: str = ETF_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached weekly ETF closes, sliced to ``asof`` (≤). Empty frame on a cache miss."""
    df = fetch_energy_pairs(cache=cache, fetch=False)
    if df.empty:
        return df
    if asof is not None:
        df = df[df.index <= pd.Timestamp(asof)]
    return df


def _pair_returns(prices: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Weekly returns of the (front, laddered) pair for one commodity, aligned & dropna'd."""
    front, lad = PAIRS[commodity]
    sub = prices[[front, lad]].dropna()
    out = pd.DataFrame({"front": sub[front].pct_change(), "lad": sub[lad].pct_change()}).dropna()
    return out


def bleed_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-commodity realized roll cost of the front-month contract (the contango bleed).

    For each commodity: the total return of the front and laddered ETFs over their common sample, the gap,
    the annualized mean ``laddered − front`` drag (the roll cost the front pays), the share of weeks the
    front is in contango (underperforms the ladder), and the Newey–West *t* of the drag spread.
    """
    from quantlab.analytics import mean_tstat_hac

    rows = {}
    for cmd, (front, lad) in PAIRS.items():
        sub = prices[[front, lad]].dropna()
        r = _pair_returns(prices, cmd)
        drag = (r["lad"] - r["front"])               # laddered minus front == roll cost borne by the front
        rows[cmd] = {
            "start": sub.index.min().date(),
            "years": round(len(sub) / WEEKS_PER_YEAR, 1),
            "front_total_pct": (sub[front].iloc[-1] / sub[front].iloc[0] - 1) * 100,
            "lad_total_pct": (sub[lad].iloc[-1] / sub[lad].iloc[0] - 1) * 100,
            "gap_pct": (sub[lad].iloc[-1] / sub[lad].iloc[0] - sub[front].iloc[-1] / sub[front].iloc[0]) * 100,
            "ann_drag_pct": drag.mean() * WEEKS_PER_YEAR * 100,
            "weeks_in_contango_pct": (drag > 0).mean() * 100,
            "drag_hac_t": mean_tstat_hac(drag)["tstat"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "commodity"
    return out


def roll_timing_book(prices: pd.DataFrame, commodity: str, lookback: int = DEFAULT_LOOKBACK,
                     cost_bps: float = 0.0) -> pd.Series:
    """Time-series roll-yield carry book on one energy curve.

    Signal: the sign of the trailing ``lookback``-week mean roll (``front − laddered``) — positive when the
    curve has recently been **backwardated**, negative in **contango** — lagged one week (causal). Position
    ∈ {−1, 0, +1} on the front-month ETF; net of ``cost_bps`` per unit traded. Long the front when the curve
    pays you to roll, short it when the curve taxes you (the USO bleed, harvested instead of suffered).
    """
    r = _pair_returns(prices, commodity)
    roll = r["front"] - r["lad"]                       # + == backwardation (front out-rolls the ladder)
    signal = np.sign(roll.rolling(lookback).mean()).shift(1).fillna(0.0)
    gross = signal * r["front"]
    cost = (cost_bps * 1e-4) * signal.diff().abs().fillna(0.0)
    return (gross - cost).rename(commodity).dropna()


def combined_book(prices: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK, cost_bps: float = 0.0) -> pd.Series:
    """Equal-weight the per-commodity roll-timing books over their common weeks."""
    books = [roll_timing_book(prices, c, lookback=lookback, cost_bps=cost_bps) for c in PAIRS]
    df = pd.concat(books, axis=1).dropna()
    return df.mean(axis=1).rename("combined")


def book_summary(prices: pd.DataFrame, commodity: str | None = None, lookback: int = DEFAULT_LOOKBACK,
                 cost_bps: float = 0.0) -> dict:
    """Sharpe/CAGR/DD/skew (annualised, weekly) plus the Newey–West *t* for a roll-timing book.

    ``commodity=None`` summarises the combined WTI+GAS book.
    """
    from quantlab.analytics import mean_tstat_hac

    book = (combined_book(prices, lookback=lookback, cost_bps=cost_bps) if commodity is None
            else roll_timing_book(prices, commodity, lookback=lookback, cost_bps=cost_bps))
    s = summary(book)
    s["hac_t"] = mean_tstat_hac(book)["tstat"]
    s["turnover_per_yr"] = _turnover(prices, commodity, lookback) * WEEKS_PER_YEAR
    return s


def _turnover(prices: pd.DataFrame, commodity: str | None, lookback: int) -> float:
    """Average weekly one-way turnover of the {−1,0,+1} position(s)."""
    if commodity is None:
        per = []
        for c in PAIRS:
            r = _pair_returns(prices, c)
            sig = np.sign((r["front"] - r["lad"]).rolling(lookback).mean()).shift(1).fillna(0.0)
            per.append(sig.diff().abs())
        return float(pd.concat(per, axis=1).mean(axis=1).mean())
    r = _pair_returns(prices, commodity)
    sig = np.sign((r["front"] - r["lad"]).rolling(lookback).mean()).shift(1).fillna(0.0)
    return float(sig.diff().abs().mean())


def always_long_front(prices: pd.DataFrame, commodity: str) -> pd.Series:
    """The naive buy-and-hold front-month ETF return — the contrast the timing book has to beat."""
    return _pair_returns(prices, commodity)["front"].rename(f"{commodity}_long")
