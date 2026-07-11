"""Data layer for Study 681 — Relative-Rotation-Graph (RRG).

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily adjusted closes for the **11 SPDR sector ETFs**
  (XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLB/XLU/XLRE/XLC) plus **SPY** as the benchmark, all from
  yfinance (no key), cached as a single CSV under the study's own ``_cache/``. The RRG
  needs *daily* granularity (the rolling normalisation windows are tens of trading days)
  even though the rotation strategy itself only rebalances monthly — same universe as
  siblings [225-sector-rotation](../225-sector-rotation/) and
  [506-industry-momentum](../506-industry-momentum/), same known quirk: XLRE launched
  2015-10, XLC launched 2018-06 (the other 9 launched 1998-12).

* **Synthetic world.** A deterministic, seeded multi-sector *daily* price panel with a
  TUNABLE persistent relative-drift component per sector (an AR(1) "who's hot" process,
  same spirit as sibling 225's synthetic panel but simulated day-by-day so the RRG's
  rolling-window machinery has something realistic to chew on). ``mom_strength = 0`` is
  the null: sectors share no persistent relative drift, and the RRG quadrant detector
  must not manufacture a rotation edge from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "rrg_prices.csv")

# --------------------------------------------------------------------------- #
# Universe — the 11 SPDR sector ETFs (same table as siblings 225 / 506) + SPY.
# --------------------------------------------------------------------------- #
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLV": "Health_Care",
    "XLF": "Financials",
    "XLY": "Consumer_Discretionary",
    "XLI": "Industrials",
    "XLP": "Consumer_Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real_Estate",
    "XLC": "Communication_Services",
}
BENCHMARK = "SPY"
TICKERS = list(SECTOR_ETFS.keys())
ALL_TICKERS = TICKERS + [BENCHMARK]

START = "1998-12-01"     # first sector SPDRs launch
AS_OF = "2026-06-30"      # last complete month at publication (2026-07-10)

# RRG default windows (trading days): the classic JdK "tail" is ~10-12 *weeks* on a
# weekly chart; on a daily tape the equivalent normalisation window is ~63 sessions
# (one quarter) for the RS-Ratio level and ~21 sessions (one month) for the RS-Momentum
# rate-of-change — stated once here, used everywhere, never re-tuned per result.
RS_WINDOW = 63
MOM_WINDOW = 21


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1998-11-01", end: str = "2026-07-01") -> None:
    """Download daily adjusted closes for the 11 sector ETFs + SPY; cache as one CSV."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    closes: dict[str, pd.Series] = {}
    for tk in ALL_TICKERS:
        raw = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        closes[tk] = s.astype(float)
    if not closes:
        raise RuntimeError("yfinance returned no data for any RRG ticker")
    prices = pd.DataFrame(closes).sort_index()
    prices.to_csv(PRICES_CACHE)


def have_real() -> bool:
    return os.path.exists(PRICES_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily adjusted closes (columns = 11 sector tickers + SPY), sliced."""
    df = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


# --------------------------------------------------------------------------- #
# Synthetic world — planted persistent relative-drift rotation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 6300,
    n_sectors: int = 9,
    mom_strength: float = 0.00035,
    phi: float = 0.995,
    market_vol: float = 0.011,
    idio_vol: float = 0.009,
    seed: int = 681,
    start: str = "1999-01-04",
) -> tuple[pd.DataFrame, dict]:
    """Deterministic daily price panel: a market factor + AR(1) relative-drift per sector.

    Each sector ``i`` carries a slowly mean-reverting daily relative-drift
    ``alpha_{i,t} = phi * alpha_{i,t-1} + sqrt(1-phi^2) * mom_strength * nu_{i,t}``, so a
    sector that has drifted hot tends to *stay* hot for a while — the structure any
    momentum-flavoured rotation signal (RRG included) is trying to harvest.
    ``mom_strength = 0`` is the null: no persistent relative drift, only noise and the
    shared market factor — the quadrant detector must not manufacture a rotation edge.

    Business-day index, ``n_days`` ~ 6,300 (~25 years) — far below the ~250-year pandas
    ns-timestamp trap. Returns ``(prices, truth)``: prices is a DataFrame with columns
    ``SEC00..SEC{n-1}`` plus a ``BENCH`` column (the equal-weight market itself, so RS
    ratios are always well-defined), ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    n = len(idx)

    market_ret = rng.normal(0.0003, market_vol, n)

    innov_scale = mom_strength * np.sqrt(1.0 - phi**2)
    alpha = np.zeros((n, n_sectors))
    nu = rng.standard_normal((n, n_sectors))
    for t in range(1, n):
        alpha[t] = phi * alpha[t - 1] + innov_scale * nu[t]

    eps = rng.standard_normal((n, n_sectors))
    rets = market_ret[:, None] + alpha + idio_vol * eps

    cols = [f"SEC{i:02d}" for i in range(n_sectors)]
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    prices["BENCH"] = 100.0 * np.exp(np.cumsum(market_ret))

    truth = {
        "n_days": n, "n_sectors": n_sectors, "mom_strength": mom_strength, "phi": phi,
        "market_vol": market_vol, "idio_vol": idio_vol, "seed": seed,
        "has_momentum": mom_strength != 0.0,
    }
    return prices, truth
