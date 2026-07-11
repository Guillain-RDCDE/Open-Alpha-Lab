"""Data layer for Study 663 — Hash-Ribbons.

Capriole Investments' "Hash Ribbons" (Charles Edwards, 2019) says: watch Bitcoin's 30-day
simple moving average of network hashrate against its 60-day SMA. When miners are
capitulating (shutting down unprofitable rigs), the 30-day SMA sinks below the 60-day; when
the 30-day SMA claws back above the 60-day, capitulation is ending — historically a strong buy
zone, because the weakest, most-levered miners have already been forced to sell their BTC and
shut off, removing forced-seller pressure right as the network's economics stabilize.

Two ingredients:

* **The hashrate series.** A *hardcoded, curated* month-end series of the Bitcoin network
  hashrate in EH/s, Jan-2014 -> May-2026 — the **identical table** used by sibling study
  292-bitcoin-hashrate, digitised to round figures from the public Blockchain.com 7-day-average
  hashrate chart (https://www.blockchain.com/explorer/charts/hash-rate). Capriole's actual
  ribbon runs on *daily* hashrate; we don't have a free, hardcodable daily series, so we
  linearly interpolate between the curated month-end anchors to get a daily path. This is a
  named, honest limitation: interpolation smooths away real day-to-day hashrate noise, so the
  exact calendar day of a crossover is only accurate to within roughly the anchor month, not
  the session. It does NOT erase the multi-month capitulation-and-recovery swings the ribbon is
  built to catch (the 2018 bear, the 2020 COVID/halving dip, the 2021 China-mining-ban crash,
  the 2022 FTX shock) — those are visible even in monthly-anchored data, and they are exactly
  the ~4-6 episodes in Bitcoin's history usually cited as "hash-ribbon buy signals".

* **Real BTC-USD tape.** Daily closes from yfinance (no key), cached as CSV. Price-only ==
  total-return for BTC (no dividends).

Pure numpy + pandas + stdlib on the offline path once cached. ``fetch_btc`` (network) runs
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
BTC_CACHE = os.path.join(CACHE_DIR, "hr_btc_usd.csv")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07-10)
TICKER = "BTC-USD"

# --------------------------------------------------------------------------- #
# Curated month-end Bitcoin network hashrate, EH/s. IDENTICAL hardcoded table to
# sibling study 292-bitcoin-hashrate — digitised to ~2 significant figures from the public
# Blockchain.com 7-day-average hashrate chart (https://www.blockchain.com/explorer/charts/hash-rate).
# Reused here rather than re-digitised, so both studies agree on the one fact they share.
# --------------------------------------------------------------------------- #
HASHRATE_MONTHLY = {
    # 2014: network in the low single-digit EH/s
    "2014-01-31": 0.010, "2014-02-28": 0.020, "2014-03-31": 0.035,
    "2014-04-30": 0.050, "2014-05-31": 0.080, "2014-06-30": 0.110,
    "2014-07-31": 0.140, "2014-08-31": 0.190, "2014-09-30": 0.250,
    "2014-10-31": 0.270, "2014-11-30": 0.290, "2014-12-31": 0.310,
    # 2015
    "2015-01-31": 0.320, "2015-02-28": 0.330, "2015-03-31": 0.360,
    "2015-04-30": 0.380, "2015-05-31": 0.350, "2015-06-30": 0.380,
    "2015-07-31": 0.420, "2015-08-31": 0.420, "2015-09-30": 0.480,
    "2015-10-31": 0.560, "2015-11-30": 0.620, "2015-12-31": 0.780,
    # 2016
    "2016-01-31": 0.900, "2016-02-29": 1.10, "2016-03-31": 1.20,
    "2016-04-30": 1.40, "2016-05-31": 1.40, "2016-06-30": 1.50,
    "2016-07-31": 1.60, "2016-08-31": 1.70, "2016-09-30": 1.80,
    "2016-10-31": 2.00, "2016-11-30": 2.10, "2016-12-31": 2.50,
    # 2017
    "2017-01-31": 3.00, "2017-02-28": 3.40, "2017-03-31": 3.70,
    "2017-04-30": 3.80, "2017-05-31": 4.80, "2017-06-30": 5.50,
    "2017-07-31": 6.40, "2017-08-31": 7.50, "2017-09-30": 9.30,
    "2017-10-31": 11.0, "2017-11-30": 11.0, "2017-12-31": 14.0,
    # 2018: ATH then bear
    "2018-01-31": 16.0, "2018-02-28": 21.0, "2018-03-31": 25.0,
    "2018-04-30": 27.0, "2018-05-31": 34.0, "2018-06-30": 40.0,
    "2018-07-31": 39.0, "2018-08-31": 51.0, "2018-09-30": 53.0,
    "2018-10-31": 51.0, "2018-11-30": 42.0, "2018-12-31": 41.0,
    # 2019
    "2019-01-31": 41.0, "2019-02-28": 43.0, "2019-03-31": 46.0,
    "2019-04-30": 52.0, "2019-05-31": 56.0, "2019-06-30": 66.0,
    "2019-07-31": 72.0, "2019-08-31": 78.0, "2019-09-30": 90.0,
    "2019-10-31": 95.0, "2019-11-30": 90.0, "2019-12-31": 97.0,
    # 2020: halving in May, China still dominant
    "2020-01-31": 110.0, "2020-02-29": 120.0, "2020-03-31": 110.0,
    "2020-04-30": 120.0, "2020-05-31": 110.0, "2020-06-30": 120.0,
    "2020-07-31": 125.0, "2020-08-31": 130.0, "2020-09-30": 135.0,
    "2020-10-31": 130.0, "2020-11-30": 140.0, "2020-12-31": 150.0,
    # 2021: China mining ban mid-year -> big drawdown then recovery
    "2021-01-31": 160.0, "2021-02-28": 165.0, "2021-03-31": 170.0,
    "2021-04-30": 175.0, "2021-05-31": 150.0, "2021-06-30": 90.0,
    "2021-07-31": 95.0, "2021-08-31": 120.0, "2021-09-30": 140.0,
    "2021-10-31": 160.0, "2021-11-30": 165.0, "2021-12-31": 175.0,
    # 2022: bear market in price, hashrate keeps climbing
    "2022-01-31": 195.0, "2022-02-28": 210.0, "2022-03-31": 215.0,
    "2022-04-30": 220.0, "2022-05-31": 220.0, "2022-06-30": 215.0,
    "2022-07-31": 225.0, "2022-08-31": 240.0, "2022-09-30": 250.0,
    "2022-10-31": 265.0, "2022-11-30": 250.0, "2022-12-31": 255.0,
    # 2023
    "2023-01-31": 300.0, "2023-02-28": 320.0, "2023-03-31": 340.0,
    "2023-04-30": 350.0, "2023-05-31": 365.0, "2023-06-30": 375.0,
    "2023-07-31": 400.0, "2023-08-31": 410.0, "2023-09-30": 420.0,
    "2023-10-31": 440.0, "2023-11-30": 470.0, "2023-12-31": 500.0,
    # 2024: halving in April
    "2024-01-31": 520.0, "2024-02-29": 540.0, "2024-03-31": 600.0,
    "2024-04-30": 620.0, "2024-05-31": 600.0, "2024-06-30": 600.0,
    "2024-07-31": 620.0, "2024-08-31": 640.0, "2024-09-30": 650.0,
    "2024-10-31": 700.0, "2024-11-30": 720.0, "2024-12-31": 780.0,
    # 2025
    "2025-01-31": 800.0, "2025-02-28": 820.0, "2025-03-31": 830.0,
    "2025-04-30": 880.0, "2025-05-31": 900.0, "2025-06-30": 920.0,
    "2025-07-31": 940.0, "2025-08-31": 960.0, "2025-09-30": 980.0,
    "2025-10-31": 1000.0, "2025-11-30": 1020.0, "2025-12-31": 1050.0,
    # 2026
    "2026-01-31": 1070.0, "2026-02-28": 1080.0, "2026-03-31": 1100.0,
    "2026-04-30": 1120.0, "2026-05-31": 1140.0,
}


def hashrate_monthly() -> pd.Series:
    """The curated month-end hashrate anchors (EH/s), as a float Series."""
    idx = pd.to_datetime(list(HASHRATE_MONTHLY.keys()))
    vals = np.array(list(HASHRATE_MONTHLY.values()), dtype=float)
    return pd.Series(vals, index=idx, name="hashrate_ehs").sort_index()


def hashrate_daily() -> pd.Series:
    """Daily hashrate path (EH/s): linear interpolation between the month-end anchors.

    Deterministic, offline, a pure function of ``HASHRATE_MONTHLY``. Named limitation: this
    smooths away genuine day-to-day hashrate noise (real daily hashrate is much choppier than
    a straight line between two monthly points), so the SMA-crossover *day* is only accurate to
    roughly the anchor month; the multi-month capitulation/recovery cycles the ribbon targets
    survive the smoothing intact.
    """
    m = hashrate_monthly()
    idx = pd.date_range(m.index.min(), m.index.max(), freq="D")
    return m.reindex(idx).interpolate(method="linear").rename("hashrate_ehs")


# --------------------------------------------------------------------------- #
# Real tape — BTC-USD daily close
# --------------------------------------------------------------------------- #
def fetch_btc(start: str = "2014-01-01", end: str | None = None,
              path: str = BTC_CACHE) -> pd.Series:
    """Download daily BTC-USD closes and cache them. Network; run once."""
    import yfinance as yf

    px = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.name = "btc_usd"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_frame().to_csv(path)
    return px


def have_real(path: str = BTC_CACHE) -> bool:
    return os.path.exists(path)


def load_btc(path: str = BTC_CACHE, asof: str = AS_OF) -> pd.Series:
    """Cached daily BTC-USD close, sliced to the study as-of (sample never creeps)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    return px[px.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world — planted hashrate-crossover -> forward-return effect
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 4200, effect: float = 0.0, seed: int = 663,
                    base_ann: float = 0.40, vol_ann: float = 0.65,
                    n_cycles: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic BTC-like daily-return world with TUNABLE post-signal drift.

    Plants ``n_cycles`` evenly-spaced "capitulation -> recovery" signal days. On the
    ``horizon``-day window right after each signal the daily mean return gets an extra
    ``effect`` (annualised) on top of the flat unconditional drift ``base_ann``.
    ``effect = 0`` is the null: signal windows are statistically identical to any other day,
    and the detector must NOT fire. Returns ``(daily_returns, signal_idx)`` as plain arrays —
    no timestamps, so this is immune to the pandas ns-Timestamp span trap.
    """
    rng = np.random.default_rng(seed)
    mu_flat = base_ann / 365.0
    sig = vol_ann / np.sqrt(365.0)
    r = mu_flat + sig * rng.standard_normal(n_days)

    signal_idx = np.linspace(300, n_days - 300, n_cycles).astype(int)
    horizon = 180
    mu_extra = effect / 365.0
    for s in signal_idx:
        r[s: s + horizon] += mu_extra
    return r, signal_idx
