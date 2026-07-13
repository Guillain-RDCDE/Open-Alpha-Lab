"""Data layer for Study 752 — TGA-Drawdown (Treasury cash balance + SPY).

Three components, the first and third fully offline and deterministic:

* **TGA tape (LABELLED monthly proxy).** ``TGA_BY_YEAR`` is a hardcoded, monthly
  end-of-month snapshot of the **U.S. Treasury General Account operating cash
  balance** ($ billions) — the Treasury's checking account at the Federal Reserve.
  Source: U.S. Treasury *Daily Treasury Statement* operating cash balance / FRED
  series ``WTREGEN`` (*"Liabilities and Capital: Liabilities: Deposits with F.R.
  Banks, Other Than Reserve Balances: U.S. Treasury, General Account"*, weekly
  Wednesday level). FRED's CSV endpoint is firewalled in this build, and the true
  series is **weekly**; hardcoding ~1,000 weekly prints by hand would be false
  precision, so — exactly like Study 358 (watch-index) and Study 708
  (eurovision-effect) use a small labelled proxy — we hardcode an **approximate
  month-end proxy** of the balance (2005..2026, $B). It is a PROXY, named as such
  on every axis: the big, well-documented moves (the 2020 COVID surge to ~$1.8T,
  the 2021/2023/2025 debt-ceiling drawdowns to near-zero) are faithful; the exact
  monthly levels are approximate. The believers' "liquidity" signal is computed
  *from* this series (see :mod:`tga_drawdown.strategy`).

* **Real SPY tape.** ``load_spy`` reads the cached daily SPY adjusted close
  (``_cache/spy_prices.csv``, yfinance, no key). ``fetch_spy`` (network) rebuilds
  the cache and is never imported by the offline notebook cells. Price =
  total-return adjusted close (``auto_adjust=True``); labelled as such.

* **Synthetic positive control.** :func:`synthetic_tga` is a deterministic,
  fixed-seed generator producing a monthly TGA path and a daily SPY-like price with
  a *planted* link: when TGA is **drawing down** (liquidity injected), forward
  equity returns are lifted by a controllable ``edge`` knob. ``edge = 0`` is the
  null (TGA changes and returns are independent) and must NOT manufacture
  significance; a large planted ``edge`` must light the test up. Runs with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPY_CACHE = os.path.join(HERE, "..", "_cache", "spy_prices.csv")


# --------------------------------------------------------------------------- #
# TGA tape — hardcoded monthly PROXY of the Treasury operating cash balance ($B)
# --------------------------------------------------------------------------- #
# U.S. Treasury General Account operating cash balance, END-OF-MONTH, in $ BILLIONS.
# Source: Treasury Daily Treasury Statement operating cash balance / FRED WTREGEN
# (weekly). Monthly end-of-month PROXY, Jan..Dec per row. As-of: 2026-06-22.
# APPROXIMATE — the landmark moves are faithful (2008 Supplementary Financing surge,
# the 2020 COVID balloon toward ~$1.8T, and the 2021 / 2023 / 2025 debt-ceiling
# drawdowns toward near-zero), the exact monthly levels are proxy figures. Named as a
# PROXY everywhere it appears. Values rounded to the nearest ~$10B.
TGA_BY_YEAR: dict[int, list[float]] = {
    2005: [45, 30, 55, 60, 35, 40, 45, 30, 50, 40, 35, 45],
    2006: [40, 35, 50, 55, 30, 35, 40, 30, 45, 35, 40, 40],
    2007: [35, 30, 45, 55, 35, 40, 45, 35, 50, 45, 55, 60],
    2008: [45, 40, 30, 45, 40, 45, 45, 40, 200, 480, 500, 370],
    2009: [310, 250, 290, 260, 240, 260, 200, 250, 220, 180, 150, 200],
    2010: [110, 180, 100, 130, 90, 80, 100, 120, 100, 90, 120, 100],
    2011: [110, 90, 120, 130, 90, 80, 90, 100, 90, 80, 100, 90],
    2012: [95, 80, 110, 120, 80, 70, 90, 100, 90, 80, 100, 90],
    2013: [100, 60, 90, 110, 80, 60, 70, 80, 50, 90, 120, 100],
    2014: [90, 70, 100, 120, 90, 80, 90, 100, 90, 100, 120, 110],
    2015: [110, 200, 220, 280, 200, 180, 200, 220, 150, 30, 250, 200],
    2016: [330, 380, 280, 430, 320, 350, 400, 380, 330, 380, 400, 350],
    2017: [380, 300, 100, 180, 180, 130, 100, 60, 40, 220, 190, 210],
    2018: [310, 280, 290, 320, 320, 350, 320, 350, 380, 350, 380, 400],
    2019: [400, 350, 280, 300, 250, 260, 180, 130, 180, 350, 400, 400],
    2020: [400, 480, 420, 800, 1500, 1600, 1750, 1650, 1750, 1650, 1500, 1600],
    2021: [1600, 1400, 1100, 950, 750, 650, 500, 300, 200, 120, 60, 350],
    2022: [650, 550, 700, 850, 750, 600, 550, 600, 650, 600, 500, 450],
    2023: [450, 500, 200, 300, 50, 400, 550, 750, 700, 800, 780, 770],
    2024: [830, 800, 780, 930, 750, 780, 780, 800, 850, 780, 700, 720],
    2025: [680, 500, 400, 620, 350, 300, 500, 650, 750, 800, 820, 810],
    2026: [800, 820, 810, 830, 800, 810, 0, 0, 0, 0, 0, 0],
}


def have_real(path: str = DEFAULT_SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the TGA proxy table is always available)."""
    return os.path.exists(path)


def tga_series() -> pd.Series:
    """Monthly Treasury General Account balance ($B), indexed by month-end date.

    The in-progress months of the final year (zeros) are dropped so a stamped run
    never includes a partial bar.
    """
    rows = []
    for yr in sorted(TGA_BY_YEAR):
        for m, v in enumerate(TGA_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="tga").sort_index()


def fetch_spy(start: str = "2004-06-01", end: str | None = None,
              path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Download SPY daily adjusted close via yfinance and cache it (network-only).

    Used once to build ``_cache/spy_prices.csv``. Never imported by offline cells.
    Total-return adjusted close (``auto_adjust=True``).
    """
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    out = pd.DataFrame({"SPY": raw}).dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def load_spy(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Load the cached daily SPY adjusted-close series (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df["SPY"].astype(float)


def spy_monthly(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Month-end SPY adjusted close aligned to the monthly TGA grid."""
    spy = load_spy(path)
    return spy.resample("ME").last().dropna()


def load_real(path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: TGA balance ($B) + month-end SPY close.

    Columns: ``tga`` ($B) and ``spy`` (price). Only months present in both series are
    kept. This is the real-tape object the strategy runs on (TGA is a labelled proxy).
    """
    tga = tga_series()
    spy_m = spy_monthly(path)
    df = pd.DataFrame({"tga": tga, "spy": spy_m}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_tga(n_months: int = 252, edge: float = 0.0, seed: int = 752,
                  mu_m: float = 0.007, sig_m: float = 0.040) -> pd.DataFrame:
    """Deterministic monthly TGA + SPY-like price with a PLANTED drawdown->returns link.

    Builds a mean-reverting monthly TGA level (AR(1) in logs, around ~$500B) and a
    monthly SPY-like return series. When ``edge != 0`` the *forward* monthly return is
    perturbed by ``+edge`` whenever the TGA is **drawing down** (the 1-month change in
    the balance is negative) — the believers' story (a TGA drawdown injects liquidity
    and lifts equities) injected by construction, with a knob.

    ``edge = 0`` => TGA changes carry no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.04 monthly)
    must drive the HAC t and Welch t well past 2. The date index is a decorative monthly
    label built with ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)

    # mean-reverting log-TGA around log(500), AR(1)
    log_t = np.empty(n_months)
    log_t[0] = np.log(500.0)
    target = np.log(500.0)
    for t in range(1, n_months):
        log_t[t] = target + 0.90 * (log_t[t - 1] - target) + rng.normal(0, 0.12)
    tga = np.exp(log_t)

    # base monthly returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # drawdown flag: 1-month change in the balance is negative (liquidity injected)
    chg = np.zeros(n_months)
    chg[1:] = tga[1:] - tga[:-1]
    drawing = chg < 0

    # plant the believers' link, respecting the 1-month execution lag: a drawdown at t
    # is only acted on at t+1's close, so the boosted return is the [t+1, t+2] month —
    # i.e. ret[t+2]. Planting into ret[t+1] instead would be uncapturable by a lagged
    # strategy and the positive control would (correctly) fail to detect it.
    if edge != 0.0:
        for t in range(n_months - 2):
            if drawing[t]:
                ret[t + 2] += edge

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"tga": tga, "spy": price}, index=idx)
