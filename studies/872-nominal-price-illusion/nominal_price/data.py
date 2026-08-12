"""Data layer for Study 872 — Nominal-Price Illusion.

The claim under test (Kumar 2009, *"Who Gambles in the Stock Market?"*; Birru &
Wang 2016, *"Nominal Price Illusion"*): the **nominal share price** of a stock — the
dollar number a single share trades at, $10 vs $500 — is a pure **money-illusion**
characteristic. It says nothing about a firm's value (price × shares = value; the
denominator is arbitrary), yet **retail lottery demand clusters in low-priced names**.
If that demand over-prices cheap-looking stocks, low **nominal-price** names should
carry higher volatility / positive skew *and* **lower risk-adjusted returns** — an
over-priced lottery.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking
  panel omits the delisted / de-rated names and biases any cross-sectional result. And
  crucially for *this* signal: mega-caps are **rarely cheap** — the cross-section spans
  ~$22 to ~$1,000 a share, with no true low-priced (single-digit) names — so the sort
  has little of the retail-lottery segment to bite on. **Honest low power** is the
  headline risk here, and it is named up front.

  **Nominal price ≠ adjusted price — the second, quieter caveat.** The cache is
  ``auto_adjust=True`` (total-return): historical prices are **split- and
  dividend-back-adjusted**, so ``Close[t]`` equals the *true* nominal trading price only
  near the as-of date and drifts below it going back in time (a name that later split
  4:1 shows a divided-down history). The sort here is therefore on an **adjusted**
  price level — an honest *proxy* for nominal price that is exact at the right edge and
  degrades into the past. A pure-nominal replication needs raw (unadjusted) closes,
  which the total-return cache does not preserve; this is stated with every number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "cheapness" ``c_i`` that (a) fixes its **price level** (cheap names start
  low), (b) inflates its daily **volatility** and right-tail **skew** (the lottery
  look), and (c) — only when ``edge > 0`` — depresses its **forward mean** return.
  ``edge = 0`` is the null world: cheap names still look lottery-like (more vol, more
  right skew) but carry **no** forward-return information, and a sort on price must find
  nothing. ``edge > 0`` plants the Kumar / Birru-Wang over-priced-lottery relation.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells; ``load_panel()``
reads the cached parquet directly (no yfinance import).
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
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "COST", "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD",
    "TXN", "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UNH", "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS",
]

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_panel", "synthetic_panel",
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
# Synthetic world — planted low-price -> lottery-look + under-earn (the control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 872,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    base_vol: float = 0.011,
    drift: float = 0.06 / 252,
    vol_gain: float = 1.6,
    skew_gain: float = 1.1,
    price_lo: float = 8.0,
    price_hi: float = 520.0,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted price->return relation.

    Each asset ``i`` gets a persistent latent **cheapness** ``c_i`` ∈ [0, 1] (0 = most
    expensive, 1 = cheapest). It drives three things at once:

    * **Price level.** The starting price is ``price_hi`` for ``c_i = 0`` down to
      ``price_lo`` for ``c_i = 1`` (log-spaced), so a cross-sectional sort on the price
      level recovers ``c_i`` — cheap names sit at the bottom.
    * **Lottery look.** Daily volatility scales as ``base_vol * (1 + vol_gain * c_i)``
      and a positive-skew (right-tail) shock scales with ``c_i`` — cheap names are more
      volatile and more right-skewed, *regardless of* ``edge``. This is the money-illusion
      characteristic that is always present.
    * **Forward mean (only when ``edge > 0``).** ``r = drift - edge * c_i + shock`` — a
      cheap name's mean return is *depressed*, the over-priced-lottery penalty. At
      ``edge = 0`` cheapness still buys the lottery look but predicts **nothing** about
      forward returns (the honest null: a pure characteristic with no return content).

        z ~ N(0,1)
        vol_i        = base_vol * (1 + vol_gain * c_i)
        skewed_shock = vol_i * (z + skew_gain * c_i * (z**2 - 1))
        r[i,t]       = drift - edge * c_i + skewed_shock

    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    # Evenly spread cheapness across [0, 1] so both ends of the sort are populated.
    cheap = np.linspace(0.0, 1.0, n_assets)
    cheap = rng.permutation(cheap)                      # break any name<->order coupling
    log_lo, log_hi = np.log(price_lo), np.log(price_hi)

    for i in range(n_assets):
        c = float(cheap[i])
        p0 = float(np.exp(log_hi + (log_lo - log_hi) * c))  # cheap c=1 -> price_lo
        vol_i = base_vol * (1.0 + vol_gain * c)

        z = rng.normal(0.0, 1.0, n_days)
        skewed = vol_i * (z + skew_gain * c * (z ** 2 - 1.0))
        r = drift - edge * c + skewed

        close = p0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[p0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, vol_i / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, vol_i / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, vol_i / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
