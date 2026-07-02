"""Data layer for Study 548 (Happiness-Index-Country) — the WHR-rank / country-ETF sort.

The folklore: sort the world's investable equity markets by their **World Happiness Report**
(WHR) rank and ask whether the *happiest* nations' stock markets beat the *gloomiest* ones. It is
a textbook cross-country **spurious-correlation** demo — the kind of alt-data story that sounds
plausible ("optimistic, well-governed societies must have better markets") and dies the instant
you count the degrees of freedom: roughly two dozen investable single-country ETFs, one happiness
number apiece.

Why this study is **synthetic-first**. There is no free, clean, point-in-time panel that lines up
the WHR ladder with tradable country-index total returns over a long horizon. Single-country ETFs
have short, dividend-noisy, currency-mangled histories, the WHR methodology has changed across
editions, and the investable set is tiny (n ~ 24). A robust real-tape *t* >= 2 across countries is
therefore not reachable here — so per the desk rule a synthetic-only study is capped at WEAK/NONE,
and the small-N / no-real-panel limitation is named on the SIGNAL axis.

Two tapes, one schema (a per-country cross-section):

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``rank_alpha``, plants
  the folklore: the happier a country (better/lower WHR rank), the higher its market's alpha.
  ``rank_alpha > 0`` reproduces the "happy markets win" story; ``= 0`` is the null (returns
  unrelated to happiness). The positive control and the null in one bottle. Tests never touch the
  network.
- ``fetch_panel`` — an *illustrative* real cross-section: a curated public WHR-2024 rank table
  (hardcoded below) joined to cached single-country ETF total returns from ``yfinance``,
  cache-first into this study's own ``_cache/``. It exists to show the machinery on a real tape,
  but with n ~ 24 and a short window it can only ever be suggestive — never REAL.

No look-ahead: happiness rank is the *published* WHR value for the edition preceding the holding
window, and the forward return is measured strictly *after* that edition's publication. The signal
is fully public at entry; the position is entered at that close and held over the forward window.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Curated public WHR-2024 rank table joined to an investable single-country ETF.
# Rank = World Happiness Report 2024 country rank (1 = happiest). Only countries
# with a liquid, USD-denominated single-country ETF are kept (the *investable* set).
# This is a hardcoded public snapshot; the tradable panel is small by construction.
# ---------------------------------------------------------------------------
WHR_ETF = {
    # country:        (WHR-2024 rank, single-country ETF)
    "Finland":        (1,  "EFNL"),
    "Denmark":        (2,  "EDEN"),
    "Sweden":         (4,  "EWD"),
    "Netherlands":    (6,  "EWN"),
    "Norway":         (7,  "NORW"),
    "Switzerland":    (9,  "EWL"),
    "Australia":      (10, "EWA"),
    "Israel":         (5,  "EIS"),
    "Canada":         (15, "EWC"),
    "Ireland":        (17, "EIRL"),
    "United States":  (23, "SPY"),
    "Germany":        (24, "EWG"),
    "Belgium":        (16, "EWK"),
    "United Kingdom": (20, "EWU"),
    "Austria":        (14, "EWO"),
    "France":         (27, "EWQ"),
    "Singapore":      (30, "EWS"),
    "Spain":          (36, "EWP"),
    "Italy":          (41, "EWI"),
    "Japan":          (51, "EWJ"),
    "Korea":          (52, "EWY"),
    "Mexico":         (25, "EWW"),
    "Brazil":         (44, "EWZ"),
    "South Africa":   (83, "EZA"),
}


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    rank_alpha: float  # extra annual return per unit of *happiness* (>0 = happy markets win)

    @property
    def has_effect(self) -> bool:
        return self.rank_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_countries: int = 24,
    rank_alpha: float = 0.05,
    idio_vol: float = 0.10,
    seed: int = 548,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-country panel with a tunable happiness *effect*.

    Each country gets a happiness *score* ``h`` in [0, 1] (higher = happier), converted to a WHR
    rank (1 = happiest). Its forward market return is::

        forward_ret = drift + rank_alpha * (h - h_bar) + idio

    With ``rank_alpha > 0`` the happier countries earn *more* (the folklore); ``= 0`` is the null
    (return unrelated to happiness). The returned frame has one row per country with the columns the
    strategy consumes: ``rank`` (WHR, 1 = best), ``happiness`` (score) and the realised
    ``forward_ret`` — the same schema ``fetch_panel`` produces from real data.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.0, 1.0, size=n_countries)  # happiness score in [0,1]
    h_bar = h.mean()

    drift = 0.08
    idio = idio_vol * rng.standard_normal(n_countries)
    forward_ret = drift + rank_alpha * (h - h_bar) + idio

    # WHR rank: happiest (highest h) is rank 1.
    order = np.argsort(-h)                 # indices from happiest to gloomiest
    rank = np.empty(n_countries, dtype=int)
    rank[order] = np.arange(1, n_countries + 1)

    countries = [f"C{j:02d}" for j in range(n_countries)]
    panel = pd.DataFrame(
        {
            "rank": rank,
            "happiness": h,
            "forward_ret": forward_ret,
        },
        index=pd.Index(countries, name="country"),
    )
    return panel, WorldTruth(rank_alpha=rank_alpha)


# ---------------------------------------------------------------------------
# Real tape — cached single-country ETF total returns, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "country_etf_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2015-01-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily adjusted-close (total-return) prices for the country ETFs, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    tickers = sorted({etf for (_, etf) in WHR_ETF.values()})
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    cov = raw.notna().mean()
    raw = raw.loc[:, cov >= 0.5]
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    split_date: str = "2024-03-20",
    end_date: str = "2026-06-27",
) -> pd.DataFrame:
    """Build the illustrative real cross-section: WHR-2024 rank + forward ETF total return.

    ``split_date`` is the WHR-2024 publication (2024-03-20); the forward return runs strictly after
    it (no look-ahead). Cache-only by default. Returns an empty frame offline.

    Schema mirrors ``synthetic_panel``: ``rank`` (1 = happiest), ``happiness`` (a 0..1 score derived
    from rank for the regression), ``forward_ret``.
    """
    px = fetch_prices(cache_dir=cache_dir, fetch=fetch)
    if px.empty:
        return pd.DataFrame()
    rows = {}
    for country, (rank, etf) in WHR_ETF.items():
        if etf not in px.columns:
            continue
        seg = px[etf].loc[split_date:end_date].dropna()
        if len(seg) < 2:
            continue
        fret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        rows[country] = {"rank": int(rank), "forward_ret": fret}
    panel = pd.DataFrame.from_dict(rows, orient="index")
    if not panel.empty:
        # happiness score in [0,1] over the countries actually present: best rank -> 1.0, worst
        # rank -> 0.0 (monotone in rank, so the regression slope's sign is the folklore's sign).
        rmin, rmax = panel["rank"].min(), panel["rank"].max()
        span = max(1, rmax - rmin)
        panel["happiness"] = 1.0 - (panel["rank"] - rmin) / span
        panel = panel[["rank", "happiness", "forward_ret"]]
    panel.index.name = "country"
    return panel.dropna()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
