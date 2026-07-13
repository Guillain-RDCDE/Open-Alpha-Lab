"""Data layer for Study 758 — TSA-Throughput (checkpoint volumes as a travel nowcast).

Three components, the first and third fully offline and deterministic:

* **Real TSA tape (hardcoded snapshot).** ``TSA_THROUGHPUT_BY_YEAR`` is a hardcoded,
  monthly snapshot of the U.S. **TSA checkpoint travel numbers** — the *average daily*
  number of travellers screened at TSA airport checkpoints, in **millions per day**,
  2019..2026. Source: Transportation Security Administration, *TSA checkpoint travel
  numbers* (``https://www.tsa.gov/travel/passenger-volumes``), aggregated from the daily
  series to a monthly average. TSA publishes the daily counts next-day; there is no free
  tidy CSV history and the site is not reachable from this build's sandbox, so — exactly
  like Study 385 (Jobless-Claims) hardcodes a FRED `IC4WSA` snapshot and Study 358
  (Watch-Index) hardcodes an alt-data index — we hardcode a public, well-known monthly
  snapshot as a **LABELLED PROXY** (rounded, approximate; the famous COVID-2020 collapse,
  when daily throughput fell ~95% to ~0.1M in April 2020, is included faithfully). It is
  never presented under a real-tape banner. The believers' momentum signal is computed
  *from* this series (see :mod:`tsa_throughput.strategy`).

* **Real travel-basket tape.** ``load_prices`` reads cached daily total-return adjusted
  closes (``_cache/travel_prices.csv``, yfinance, no key) for **JETS** (U.S. Global Jets
  ETF — airlines), **MAR** (Marriott) and **HLT** (Hilton) — the "airlines + hotels"
  travel basket — plus **SPY** as the market control. ``build_basket`` forms an
  equal-weight, monthly-rebalanced total-return travel index (½ airlines · ½ hotels).
  ``fetch_prices`` (network) rebuilds the cache and is never imported by offline cells.

* **Synthetic positive control.** :func:`synthetic_tsa` is a deterministic, fixed-seed
  generator producing a monthly TSA path and a basket-like price with a *planted* link:
  when TSA momentum is high, forward travel-basket returns are lifted by a controllable
  ``edge`` knob. ``edge = 0`` is the null (no forward information) and must NOT manufacture
  significance; a large ``edge`` must light the test up. Runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRICE_CACHE = os.path.join(HERE, "..", "_cache", "travel_prices.csv")

# The travel basket: airlines (JETS) + hotels (MAR, HLT), with SPY as the market control.
AIRLINE_TICKERS = ["JETS"]
HOTEL_TICKERS = ["MAR", "HLT"]
MARKET_TICKER = "SPY"
ALL_TICKERS = AIRLINE_TICKERS + HOTEL_TICKERS + [MARKET_TICKER]


# --------------------------------------------------------------------------- #
# Real TSA tape — hardcoded monthly snapshot (millions of travellers / day, avg)
# --------------------------------------------------------------------------- #
# U.S. TSA checkpoint travel numbers: MONTHLY AVERAGE of the daily count of travellers
# screened, in MILLIONS PER DAY. Source: TSA, "TSA checkpoint travel numbers" (daily
# series aggregated to a monthly mean). Approximate, rounded, LABELLED PROXY — as-of
# 2026-06-30. The COVID-2020 collapse (daily throughput bottomed near 0.09M on
# 2020-04-14; the April 2020 monthly average is ~0.11M) is the famous outlier and is
# included faithfully. In-progress months (zeros) are dropped by ``tsa_series``.
TSA_THROUGHPUT_BY_YEAR: dict[int, list[float]] = {
    2019: [2.00, 2.05, 2.25, 2.28, 2.40, 2.50, 2.55, 2.48, 2.22, 2.40, 2.35, 2.30],
    2020: [2.15, 2.15, 1.05, 0.11, 0.25, 0.50, 0.72, 0.75, 0.72, 0.85, 0.90, 0.92],
    2021: [0.95, 1.05, 1.30, 1.45, 1.75, 1.95, 2.05, 1.95, 1.85, 1.95, 2.05, 1.90],
    2022: [1.75, 1.90, 2.10, 2.15, 2.20, 2.30, 2.35, 2.30, 2.15, 2.25, 2.25, 2.10],
    2023: [1.95, 2.15, 2.30, 2.30, 2.45, 2.55, 2.60, 2.50, 2.35, 2.45, 2.50, 2.45],
    2024: [2.10, 2.35, 2.50, 2.45, 2.60, 2.70, 2.75, 2.65, 2.50, 2.60, 2.65, 2.55],
    2025: [2.25, 2.40, 2.55, 2.55, 2.70, 2.80, 2.85, 2.75, 2.60, 2.70, 2.75, 2.60],
    2026: [2.35, 2.45, 2.60, 2.60, 2.70, 2.80, 0, 0, 0, 0, 0, 0],
}


def have_real(path: str = DEFAULT_PRICE_CACHE) -> bool:
    """True iff the price cache exists (the TSA table is always available)."""
    return os.path.exists(path)


def tsa_series() -> pd.Series:
    """Monthly average daily TSA throughput (millions/day), indexed by month-end date.

    In-progress months of the final year (zeros) are dropped so a stamped run never
    includes a partial bar.
    """
    rows = []
    for yr in sorted(TSA_THROUGHPUT_BY_YEAR):
        for m, v in enumerate(TSA_THROUGHPUT_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="tsa").sort_index()


# --------------------------------------------------------------------------- #
# Real travel-basket tape — fetch (network) + offline loaders
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2015-01-01", end: str | None = None,
                 path: str = DEFAULT_PRICE_CACHE) -> pd.DataFrame:
    """Download JETS/MAR/HLT/SPY daily adjusted closes via yfinance and cache them.

    Used once to build ``_cache/travel_prices.csv``. Never imported by offline cells.
    Total-return adjusted close (``auto_adjust=True``).
    """
    import yfinance as yf

    raw = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    out = raw[ALL_TICKERS].dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def load_prices(path: str = DEFAULT_PRICE_CACHE) -> pd.DataFrame:
    """Load the cached daily adjusted-close panel (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df[ALL_TICKERS].astype(float)


def build_basket(path: str = DEFAULT_PRICE_CACHE) -> pd.DataFrame:
    """Daily total-return prices for the travel basket + SPY.

    The travel basket is an **equal-weight, monthly-rebalanced** total-return index that
    puts **half its weight on airlines** (JETS) and **half on hotels** (the equal-weight
    mean of MAR and HLT) — the believers' "airlines + hotels" travel bet. We build it from
    daily component returns (equal-weight mean of the two sleeves), cumulated to a price
    index based at 100. Returns a frame with columns ``basket`` and ``spy`` (both
    total-return, price levels).
    """
    px = load_prices(path)
    rets = px.pct_change()
    air = rets[AIRLINE_TICKERS].mean(axis=1)
    hotel = rets[HOTEL_TICKERS].mean(axis=1)
    basket_ret = 0.5 * air + 0.5 * hotel
    basket = 100.0 * (1.0 + basket_ret.fillna(0.0)).cumprod()
    spy = px[MARKET_TICKER]
    return pd.DataFrame({"basket": basket, "spy": spy}).dropna()


def basket_monthly(path: str = DEFAULT_PRICE_CACHE) -> pd.DataFrame:
    """Month-end travel-basket + SPY prices aligned to the monthly TSA grid."""
    daily = build_basket(path)
    return daily.resample("ME").last().dropna()


def load_real(path: str = DEFAULT_PRICE_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: TSA throughput + basket + SPY prices.

    Columns: ``tsa`` (millions/day), ``basket`` (travel-basket total-return price), and
    ``spy`` (market total-return price). Only months present in all series are kept. This
    is the real-tape object the strategy runs on.
    """
    tsa = tsa_series()
    px = basket_monthly(path)
    df = pd.DataFrame({"tsa": tsa}).join(px, how="inner").dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_tsa(n_months: int = 240, edge: float = 0.0, seed: int = 758,
                  mu_m: float = 0.008, sig_m: float = 0.070) -> pd.DataFrame:
    """Deterministic monthly TSA + basket-like price with a PLANTED momentum->returns link.

    Builds a seasonal-plus-trend TSA throughput path (a smooth annual travel season around a
    slowly growing level) and a travel-basket-like monthly return series (high vol, like real
    airlines/hotels). When ``edge != 0`` the *forward* monthly return of an ACCELERATING month
    (12-month TSA momentum > 0) is lifted by a clean level shift ``edge`` — the believers'
    story (accelerating travel lifts travel stocks) injected by construction, with a knob.

    ``edge = 0`` => TSA momentum carries no information about returns (the null); the
    inference must NOT manufacture significance. A large planted ``edge`` must drive the Welch
    t well past 2. A ``spy`` column (independent market path) is included so beta controls run.
    The date index is a decorative monthly label built with ``period_range``.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_months)
    season = 0.25 * np.sin(2 * np.pi * (t % 12) / 12.0)          # +/- travel season
    trend = np.log(2.0) + 0.0015 * t                              # slow growth
    tsa = np.exp(trend + season + rng.normal(0, 0.02, size=n_months))

    # 12-month TSA momentum; ACCELERATING when momentum > 0
    mom = np.zeros(n_months)
    mom[12:] = tsa[12:] / tsa[:-12] - 1.0
    accel = mom > 0.0

    ret = rng.normal(mu_m, sig_m, size=n_months)                 # basket-like returns
    if edge != 0.0:
        # clean planted link honouring the one-month execution lag: an accelerating month t
        # (signal at close t, entered at close t+1) lifts the held return over [t+1, t+2].
        ret[2:] += edge * accel[:-2]
    basket = 100.0 * np.exp(np.cumsum(ret))

    spy_ret = rng.normal(0.006, 0.040, size=n_months)            # independent market
    spy = 100.0 * np.exp(np.cumsum(spy_ret))

    idx = pd.period_range("2005-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"tsa": tsa, "basket": basket, "spy": spy}, index=idx)
