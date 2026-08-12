"""Data layer for Study 876 — Industry-Relative MAX.

The claim under test refines study **365** (Bali, Cakici & Whitelaw 2011, *"Maxing Out"*):
the raw MAX effect sorts a name on its own **maximum daily return** over the prior month and
finds the lottery-like high-MAX names *under-earn*. Here we sort on the **industry-relative**
MAX — a name's MAX **minus the median MAX of its sector peers** — to strip out sector-wide
volatility (a whole sector can be jumpy for macro reasons) and isolate the *idiosyncratic*
lottery demand. Does the industry adjustment **sharpen** or **kill** the negative MAX→return
relation?

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid US
  large-caps (``UNIVERSE`` below), pulled with yfinance through the ``quantlab.universe``
  **survivorship guard** (``download_panel(..., allow_survivorship_bias=True)``).
  ``auto_adjust=True`` (total-return prices). The panel parquet is cached under this study's
  OWN ``_cache/`` (we point ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE``
  *before* importing it). Each name carries a fixed **GICS sector** label (``SECTORS``) — a
  public fact, encoded here with its source — which defines the industry peer group.

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership list of
  names that are liquid mega-caps *today*; feeding it to a backward-looking panel omits the
  delisted / de-rated names and biases any cross-sectional result. The guard forces the opt-in;
  the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded monthly panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``. Each name's monthly MAX is the sum of a
  **sector-wide** lottery level (shared by all peers, and *un-priced*) and an **idiosyncratic**
  lottery intensity (name-specific, and — only when ``edge > 0`` — priced: high idiosyncratic
  MAX depresses next-month return). ``edge = 0`` is the null: MAX carries no forward
  information. The industry adjustment removes the sector-wide term, so at ``edge > 0`` the
  *industry-relative* MAX recovers the planted relation **more sharply** than the raw MAX —
  the whole thesis of this study, provable in the toy world.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells; ``load_panel()`` reads the cached
parquet directly (no yfinance import).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Point quantlab.universe's cache at THIS study's _cache/ before importing it.
os.environ.setdefault("OVERNIGHT_CACHE", CACHE_DIR)

from quantlab.universe import (  # noqa: E402  (after the env var is set)
    SurvivorshipBiasError,
    download_panel,
    panel_cache_path,
)

START = "2010-01-01"        # panel start (matches quantlab.universe default)
AS_OF = "2026-06-30"        # last complete calendar month at publication

# A fixed list of ~50 liquid US large-caps — *current* membership, a survivor set.
# (Identical to studies 803 / 788, so the shared panel cache is reused byte-for-byte.)
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "COST", "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD",
    "TXN", "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UNH", "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS",
]

# GICS sector for each name — the industry peer group for the industry-relative MAX.
# Public fact (S&P Dow Jones Indices / MSCI GICS classification, sector level). Encoded
# statically so the study is fully offline. Visa/Mastercard/American Express are placed in
# **Financials** (their post-2023 GICS home under Financial Services) — a documented choice;
# nothing here hinges on that single call. See docs/references.md for the source.
SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "CSCO": "Information Technology",
    "ORCL": "Information Technology", "ADBE": "Information Technology",
    "CRM": "Information Technology", "TXN": "Information Technology",
    "INTC": "Information Technology", "QCOM": "Information Technology",
    "AMD": "Information Technology", "IBM": "Information Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials",
    "WFC": "Financials", "GS": "Financials", "MS": "Financials", "C": "Financials",
    "AXP": "Financials",
    "JNJ": "Health Care", "ABBV": "Health Care", "MRK": "Health Care",
    "PFE": "Health Care", "UNH": "Health Care",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "XOM": "Energy", "CVX": "Energy",
    "GE": "Industrials", "CAT": "Industrials", "BA": "Industrials", "MMM": "Industrials",
    "HON": "Industrials", "LMT": "Industrials", "UPS": "Industrials",
}

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "SECTORS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_panel", "build_panel", "load_real", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START) -> None:
    """Download the cross-section panel through the survivorship guard; cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_panel(
        UNIVERSE, start=start, use_cache=True, allow_survivorship_bias=True,
    )


def have_real() -> bool:
    return os.path.exists(panel_cache_path(UNIVERSE, START))


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][["Open", "High", "Low", "Close"]].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# The monthly MAX panel (raw MAX, forward return, and the sector map aligned)
# --------------------------------------------------------------------------- #
def build_panel(panel: dict[str, pd.DataFrame], min_obs: int = 15) -> dict:
    """From a daily OHLC panel, build the monthly MAX panel.

    Returns ``{"max", "fwd_ret", "mret", "sectors"}``:

    * ``max``     — DataFrame (month-end index × ticker): each name's **highest daily simple
                    return** over that calendar month (the raw MAX signal, observed at
                    month-end). Months with fewer than ``min_obs`` daily observations are NaN.
    * ``fwd_ret`` — DataFrame: each name's **next** calendar-month simple total return (what a
                    portfolio formed at this month-end earns). ``max`` row ``t`` pairs with
                    ``fwd_ret`` row ``t`` = month ``t+1``'s return — already lagged, no
                    look-ahead.
    * ``mret``    — each name's own-month total return (diagnostic).
    * ``sectors`` — a Series ``ticker -> GICS sector`` aligned to the columns.
    """
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    daily = closes.pct_change()
    grp = daily.groupby(pd.Grouper(freq="ME"))
    cnt = grp.count()
    mx = grp.max().where(cnt >= min_obs)
    mret = grp.apply(lambda d: (1.0 + d).prod() - 1.0).where(cnt >= min_obs)
    fwd = mret.shift(-1)
    sectors = pd.Series({c: SECTORS.get(c, "Unknown") for c in closes.columns})
    return {"max": mx, "fwd_ret": fwd, "mret": mret, "sectors": sectors}


def load_real() -> dict:
    """Convenience: cached OHLC panel -> monthly MAX panel in one call."""
    return build_panel(load_panel())


# --------------------------------------------------------------------------- #
# Synthetic world — planted idiosyncratic-MAX -> return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 876,
    n_months: int = 240,
    per_sector: int = 8,
    n_sectors: int = 8,
    mkt_vol: float = 0.045,
    idio_vol: float = 0.06,
    mkt_drift: float = 0.006,
    sector_max_scale: float = 1.5,
) -> dict:
    """Deterministic monthly panel with a *planted* idiosyncratic-MAX -> return relation and
    an explicit SECTOR structure, so the industry adjustment matters.

    Each name ``i`` in sector ``g`` has a monthly MAX that is the sum of two positive parts::

        max[i,t] = sector_level[g,t]  +  idio_max[i,t]

    * ``sector_level[g,t]`` — a **sector-wide** lottery/volatility level, shared by every peer
      in the sector, and **un-priced** (pure sector volatility, no forward-return content).
    * ``idio_max[i,t]``     — a **name-specific** lottery intensity, and — only when
      ``edge > 0`` — **priced**: a high idiosyncratic MAX this month depresses the name's
      *next*-month return by ``edge × z(idio_max)`` (z = cross-sectional standardisation).

    Because the raw MAX mixes the two parts, a raw-MAX sort is contaminated by the un-priced
    sector level. The **industry-relative** MAX (raw MAX minus the sector's median MAX) removes
    ``sector_level`` and recovers ``idio_max`` — so at ``edge > 0`` the industry-adjusted sort
    detects the planted relation MORE sharply than the raw sort. ``edge = 0`` is the null:
    neither sort should fire.

    A decorative month-end index is built with ``pd.period_range`` (never a huge ``date_range``
    span) to stay far under the ns-Timestamp overflow wall.

    Returns the same ``{"max", "fwd_ret", "mret", "sectors"}`` shape as :func:`build_panel`.
    """
    rng = np.random.default_rng(seed)
    n = n_months
    k = per_sector * n_sectors
    idx = pd.period_range("2005-01", periods=n, freq="M").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")
    cols = [f"N{i:02d}" for i in range(k)]
    sec_of = np.repeat(np.arange(n_sectors), per_sector)   # sector index per name
    sec_labels = pd.Series({cols[i]: f"SEC{sec_of[i]}" for i in range(k)})

    betas = rng.uniform(0.7, 1.3, size=k)
    mkt = rng.normal(mkt_drift, mkt_vol, size=n)                 # market factor
    sec_ret = rng.normal(0.0, mkt_vol * 0.6, size=(n, n_sectors))  # sector return factor
    idio = rng.normal(0.0, idio_vol, size=(n, k))               # name idiosyncratic return

    # MAX parts (positive by construction):
    #   sector_level: shared across peers, un-priced
    sec_level = np.abs(rng.normal(0.0, idio_vol, size=(n, n_sectors))) * sector_max_scale
    sector_level = sec_level[:, sec_of]                         # (n, k) broadcast to names
    #   idio_max: name-specific lottery intensity, priced (when edge>0)
    idio_max = np.abs(rng.normal(0.0, idio_vol, size=(n, k))) + 0.3 * idio_vol
    mx_raw = sector_level + idio_max

    # base monthly returns: market + sector + idiosyncratic
    base = mkt[:, None] * betas[None, :] + sec_ret[:, sec_of] + idio

    # plant the lottery penalty on the IDIOSYNCRATIC max only (standardise cross-sectionally)
    z = (idio_max - idio_max.mean(axis=1, keepdims=True)) / (idio_max.std(axis=1, keepdims=True) + 1e-9)
    penalty = np.zeros_like(base)
    penalty[1:, :] = edge * z[:-1, :]     # this month's idio-MAX penalises next month's return
    ret = base - penalty

    mret = pd.DataFrame(ret, index=idx, columns=cols)
    mx = pd.DataFrame(mx_raw, index=idx, columns=cols)
    fwd = mret.shift(-1)
    return {"max": mx, "fwd_ret": fwd, "mret": mret, "sectors": sec_labels}
