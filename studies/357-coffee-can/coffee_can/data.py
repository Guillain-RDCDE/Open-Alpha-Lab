"""Data layer for Study 357 — the Coffee-Can portfolio.

Two sources, both offline-friendly:

* **Real tape.** Monthly **total-return** (``auto_adjust=True`` — splits *and* dividends)
  prices for a quality large-cap basket and the index (SPY), pulled from **yfinance** (public
  Yahoo endpoint, no key) and cached under ``_cache/coffee_can_monthly.csv``. The basket is the
  kind of names a 2005-vintage "coffee can" investor would have called *quality large-caps* —
  but note they are also names we *know* survived, which is exactly the bias the study measures.
* **Synthetic.** A deterministic, fixed-seed generator of a stock panel WITH a realistic
  death process (firms that drift to zero and delist). It exposes the survivorship knob
  directly: a "famous-names" selector that peeks at the terminal value (look-ahead) versus an
  honest selector that picks ex-ante on early data. The positive control is that the look-ahead
  selector manufactures a large fake out-performance even when every stock has the *same* true
  drift — i.e. the harness can detect survivorship where it is planted, and zero where it is not.

Pure numpy / pandas / stdlib for the offline core; only :func:`fetch_real` touches the network,
and only on an explicit cache miss.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache", "coffee_can_monthly.csv")

# A 2005-vintage "quality large-cap" basket — survivors we recognise today (that is the point:
# the names are chosen with hindsight, which is the survivorship the Signal axis must name).
BASKET = ["KO", "JNJ", "PG", "PEP", "WMT", "MMM", "MO", "XOM", "IBM", "GE"]
INDEX = "SPY"
START = "2005-01-01"
END = "2026-06-01"


# --------------------------------------------------------------------------- #
# Real tape (yfinance) — cached
# --------------------------------------------------------------------------- #
def fetch_real(tickers=None, index: str = INDEX, start: str = START, end: str = END,
               path: str = CACHE) -> pd.DataFrame:
    """Download monthly total-return closes for the basket + index and cache them.

    Network only. Writes a tidy CSV (Date index, one column per ticker) to ``path`` and
    returns it. Offline readers use :func:`load_real`, which never calls this.
    """
    import yfinance as yf

    tickers = list(tickers or BASKET)
    cols = tickers + [index]
    raw = yf.download(cols, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)
    close = raw["Close"][cols].dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    close.to_csv(path)
    return close


def load_real(path: str = CACHE) -> pd.DataFrame:
    """Load the cached monthly total-return panel (Date index, ticker columns)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna(how="all")


def have_real(path: str = CACHE) -> bool:
    return os.path.exists(path)


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns from a price panel (first row dropped)."""
    return prices.pct_change().iloc[1:]


# --------------------------------------------------------------------------- #
# Synthetic positive control — a panel WITH a death process
# --------------------------------------------------------------------------- #
def synthetic_panel(n_firms: int = 400, n_months: int = 240,
                    mu_ann: float = 0.08, sigma_ann: float = 0.22,
                    death_hazard_ann: float = 0.03, seed: int = 357):
    """Deterministic monthly total-return panel with delistings.

    Every firm draws the **same** true monthly drift ``mu_ann/12`` and vol — so there is *no*
    real cross-sectional edge to find. But each firm carries a constant annual death hazard:
    once it dies its price path goes to (near) zero and it delists. This is the honest universe
    a real coffee-can investor faced; the dead names are the ones survivorship erases.

    Returns ``(prices, alive)`` as DataFrames of shape ``(n_months+1, n_firms)``:
    ``prices`` are total-return levels (start = 100, NaN after death), ``alive`` is a 0/1 mask.
    Pure numpy, fixed seed — no network, fully reproducible.
    """
    rng = np.random.default_rng(seed)
    mu_m = mu_ann / 12.0
    sig_m = sigma_ann / np.sqrt(12.0)
    haz_m = 1.0 - (1.0 - death_hazard_ann) ** (1.0 / 12.0)

    rets = rng.normal(mu_m, sig_m, size=(n_months, n_firms))
    # death times: geometric with monthly hazard, capped past the window if it never dies.
    # haz_m == 0 (no-death control) -> every firm outlives the window.
    if haz_m <= 0:
        death = np.full(n_firms, n_months + 1)
    else:
        death = rng.geometric(haz_m, size=n_firms)
        death = np.where(death > n_months, n_months + 1, death)

    prices = np.empty((n_months + 1, n_firms))
    prices[0] = 100.0
    alive = np.ones((n_months + 1, n_firms))
    for t in range(1, n_months + 1):
        step = 1.0 + rets[t - 1]
        prices[t] = prices[t - 1] * step
        dead = death < t                          # already dead before this month
        dying = death == t                        # delists this month -> crash to ~0
        prices[t] = np.where(dead, np.nan, prices[t])
        prices[t] = np.where(dying, prices[t - 1] * 0.05, prices[t])  # 95% wipe on delist
        alive[t] = np.where(death <= t, 0.0, 1.0)

    idx = pd.date_range("2005-01-31", periods=n_months + 1, freq="ME")
    cols = [f"F{i:03d}" for i in range(n_firms)]
    return (pd.DataFrame(prices, index=idx, columns=cols),
            pd.DataFrame(alive, index=idx, columns=cols))


def famous_names(prices: pd.DataFrame, k: int = 10) -> list[str]:
    """LOOK-AHEAD selector: the ``k`` firms with the highest *terminal* value.

    This is exactly the coffee-can selection bias — you build the basket from the names that,
    with hindsight, turned out to be the famous winners. It peeks at the end of the panel."""
    terminal = prices.iloc[-1]
    return list(terminal.sort_values(ascending=False).head(k).index)


def honest_names(prices: pd.DataFrame, k: int = 10, lookback: int = 24) -> list[str]:
    """EX-ANTE selector: the ``k`` firms with the best return over the FIRST ``lookback`` months.

    No peeking — picks at the start, when a real investor would have. Used as the honest
    counterfactual to :func:`famous_names`."""
    early = prices.iloc[lookback] / prices.iloc[0] - 1.0
    return list(early.sort_values(ascending=False).head(k).index)
