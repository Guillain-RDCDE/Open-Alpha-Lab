"""Data layer for Study 392 — Glassdoor-Sentiment (employee-happiness factor).

The claim under test: *do companies with the happiest employees beat the market?*
Edmans (2011) famously found that a value-weighted portfolio of Fortune's "100 Best
Companies to Work For" earned a real, risk-adjusted long-short premium. The folk version
("buy the firms with the best Glassdoor ratings") is the retail descendant of that paper.

Real employer-review data (Glassdoor ratings, satisfaction surveys) is **not free** — it
sits behind paid APIs and licensing. So this study is built on TWO transparent sources, both
offline-friendly:

* **Real tape.** Daily adjusted closes for a fixed 40-name large-cap basket (yfinance, no
  key), cached under ``_cache/basket_prices.csv``. These are genuine prices.

* **Constructed employee-sentiment panel.** Because we cannot buy the real ratings, we attach
  to each name a **clearly-labelled, deterministic illustrative satisfaction score** drawn
  from a fixed seed — an explicit PROXY / placeholder, *not* scraped Glassdoor data, and we
  say so everywhere. By construction the placeholder score is **independent of returns**: it
  is the honest null. A real Glassdoor feed could be dropped in via :func:`attach_sentiment`'s
  ``scores`` argument.

* **Synthetic positive control.** A fully synthetic panel (prices + sentiment) with a
  **planted** happiness→return edge knob. With ``edge=0`` the long-short premium must NOT
  reach significance; with a large planted edge it must light up. This proves the engine is
  faithful and unbiased.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the price cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")

# A transparent, fixed basket of large, long-listed US large-caps. Chosen for long price
# history and sector spread. Each name is assigned a CONSTRUCTED satisfaction score below;
# this is a PROXY for true Glassdoor / employee-review data, named as such everywhere.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
]


# --------------------------------------------------------------------------- #
# Real tape (genuine prices)
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/basket_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.90]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


# --------------------------------------------------------------------------- #
# Constructed employee-sentiment panel (the explicit PROXY)
# --------------------------------------------------------------------------- #
def constructed_sentiment(tickers, seed: int = 392) -> pd.Series:
    """A CONSTRUCTED, deterministic employee-satisfaction score per ticker.

    THIS IS NOT REAL GLASSDOOR DATA. It is a fixed-seed placeholder on a 1.0–5.0 "stars"
    scale, assigned independently of returns. We use it to demonstrate the *method* — sort
    firms by happiness, go long the happiest, short the grumpiest — on an honest null where
    we know there is no built-in link between sentiment and price. A real ratings feed would
    replace this Series wholesale.
    """
    rng = np.random.default_rng(seed)
    # centre ~3.6 stars with realistic spread, clipped to the Glassdoor 1.0–5.0 range
    stars = np.clip(rng.normal(3.6, 0.45, size=len(tickers)), 1.0, 5.0)
    return pd.Series(np.round(stars, 2), index=list(tickers), name="stars")


def load_real(path: str = DEFAULT_CACHE, seed: int = 392) -> dict:
    """Convenience: cached prices + the constructed sentiment panel in one call.

    Returns ``{"prices": DataFrame, "stars": Series}`` where ``stars`` is the EXPLICIT
    constructed proxy (see :func:`constructed_sentiment`), aligned to the surviving columns.
    """
    prices = load_prices(path)
    stars = constructed_sentiment(list(prices.columns), seed=seed)
    return {"prices": prices, "stars": stars}


# --------------------------------------------------------------------------- #
# Synthetic positive control (planted edge)
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_months: int = 240, edge: float = 0.0,
                    seed: int = 392, mu_m: float = 0.006, sig_m: float = 0.055
                    ) -> dict:
    """Deterministic monthly panel (prices + sentiment) with a PLANTED happiness edge.

    Each name gets a fixed satisfaction score ``s_i`` (1–5 stars). Monthly returns are
    ``mu_m + edge * (s_i - mean(s)) + noise``: with ``edge = 0`` happiness carries NO
    information (the long-short must stay insignificant); with ``edge > 0`` the happiest
    names out-earn the grumpiest by a known amount the engine must recover.

    Returns ``{"prices": DataFrame(month x name), "stars": Series}``. The date index is a
    decorative monthly label built with ``period_range`` (no OutOfBounds risk).
    """
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n_names)]
    stars = np.clip(rng.normal(3.6, 0.45, size=n_names), 1.0, 5.0)
    s_demean = stars - stars.mean()

    # monthly simple returns with an (optional) planted cross-sectional happiness tilt
    base = rng.normal(mu_m, sig_m, size=(n_months, n_names))
    rets = base + edge * s_demean[None, :]

    price = 100.0 * np.cumprod(1.0 + rets, axis=0)
    idx = pd.period_range("2005-01", periods=n_months, freq="M").to_timestamp()
    prices = pd.DataFrame(price, index=idx, columns=names)
    return {"prices": prices, "stars": pd.Series(np.round(stars, 2), index=names,
                                                 name="stars")}
