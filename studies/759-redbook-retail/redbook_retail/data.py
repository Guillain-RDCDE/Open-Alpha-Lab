"""Data layer for Study 759 — Redbook-Retail (Redbook same-store-sales nowcast + XRT).

Three components, the first and third fully offline and deterministic:

* **Redbook same-store-sales tape (hardcoded, LABELLED PROXY).** ``REDBOOK_YOY_BY_YEAR``
  is a hardcoded, monthly reconstruction of the **Johnson Redbook Index** — the weekly
  **same-store (comparable) retail-sales** growth rate, reported **year-over-year, in
  percent**, by Redbook Research Inc. (a proprietary series distributed to subscribers and
  quoted by Investing.com / Reuters). The weekly Redbook print is *not* on FRED and its
  history is paywalled, so — exactly like Study 358 (Watch-Index) and Study 708
  (Eurovision-Effect) hardcode a small, clearly-cited approximate series — we hardcode a
  **monthly-averaged, approximate reconstruction** of the headline YoY same-store number.
  It is a **labelled proxy**, never presented under a real-tape banner: the *shape* (soft
  2015–16, strong 2018–19, the COVID-2020 collapse, the 2021–22 nominal reopening/inflation
  surge, the 2023 deceleration) is faithful to the public record; individual months are
  approximate. The believers' momentum signal is computed *from* this series (see
  :mod:`redbook_retail.strategy`).

* **Real XRT / SPY tape.** ``load_prices`` reads cached daily adjusted closes for **XRT**
  (SPDR S&P Retail ETF — the "retail stocks" of the claim) and **SPY** (the broad-market
  benchmark, for the retail-vs-market relative test), yfinance, no key
  (``_cache/xrt_spy_prices.csv``). ``fetch_prices`` (network) rebuilds the cache and is
  never imported by the offline notebook cells. Price = total-return adjusted close
  (``auto_adjust=True``); labelled as such. XRT lists **2006-06**, which sets the sample.

* **Synthetic positive control.** :func:`synthetic_redbook` is a deterministic, fixed-seed
  generator producing a monthly Redbook-like path and a daily XRT-like price with a
  *planted* link: when Redbook momentum accelerates, forward equity returns are lifted by a
  controllable ``edge`` knob. ``edge = 0`` is the null (Redbook momentum and returns are
  independent) and must NOT manufacture significance; a large planted ``edge`` must light
  the test up. The control runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "xrt_spy_prices.csv")


# --------------------------------------------------------------------------- #
# Redbook same-store-sales tape — hardcoded monthly LABELLED PROXY (YoY %)
# --------------------------------------------------------------------------- #
# Johnson Redbook Index: weekly SAME-STORE (comparable) retail sales growth, YEAR-OVER-YEAR,
# in PERCENT. Source: Redbook Research Inc. (proprietary; quoted by Investing.com / Reuters).
# Monthly reconstruction (approximate month-average of the weekly YoY prints). As-of
# 2026-07-12. This is a LABELLED PROXY, not the licensed tape: the broad shape is faithful to
# the public record — the 2008-09 slump into NEGATIVE same-store growth, the soft 2015-16
# patch, the strong 2018-19 run, the COVID-2020 collapse (April/May deeply negative), and the
# 2021-22 *nominal* reopening/inflation surge into double digits, then the 2023 deceleration —
# but individual monthly values are approximate. Values are percent (e.g. 4.2 = +4.2% YoY).
REDBOOK_YOY_BY_YEAR: dict[int, list[float]] = {
    2005: [4.4, 4.0, 4.3, 4.6, 3.8, 4.1, 4.5, 4.2, 3.6, 3.9, 3.7, 3.5],
    2006: [4.0, 3.6, 4.2, 4.0, 3.5, 3.3, 3.1, 3.4, 3.0, 2.8, 3.2, 3.0],
    2007: [2.7, 2.9, 2.6, 2.4, 2.8, 2.5, 2.3, 2.1, 2.4, 2.0, 1.8, 1.5],
    2008: [1.6, 1.2, 1.9, 1.4, 1.1, 1.3, 0.9, 0.6, 0.2, -0.8, -1.6, -1.9],
    2009: [-1.4, -2.3, -3.1, -3.6, -4.2, -4.5, -4.0, -3.3, -2.4, -1.6, -0.7, 0.4],
    2010: [1.2, 2.0, 3.1, 3.5, 2.8, 3.0, 3.2, 2.9, 3.1, 3.3, 3.6, 3.4],
    2011: [3.2, 3.6, 4.0, 4.4, 3.8, 3.5, 3.9, 4.2, 4.0, 3.7, 3.9, 3.6],
    2012: [3.4, 3.1, 3.5, 2.9, 2.6, 2.3, 2.1, 2.5, 2.8, 2.4, 2.1, 2.6],
    2013: [2.8, 3.0, 2.6, 2.9, 3.1, 3.3, 2.9, 3.2, 2.7, 3.0, 3.4, 3.6],
    2014: [3.8, 3.5, 4.1, 4.4, 4.0, 4.3, 3.9, 4.2, 3.7, 4.0, 3.6, 4.2],
    2015: [3.4, 2.6, 1.9, 1.4, 1.7, 1.2, 0.9, 1.3, 1.0, 0.7, 0.5, 0.9],
    2016: [0.6, 1.1, 0.8, 1.3, 1.0, 0.7, 1.2, 0.9, 1.4, 1.1, 1.6, 1.3],
    2017: [1.5, 1.9, 1.6, 2.1, 1.8, 2.3, 2.0, 2.6, 2.9, 3.2, 3.6, 3.4],
    2018: [4.1, 4.5, 4.0, 4.8, 5.2, 5.6, 5.9, 6.1, 5.7, 5.4, 5.8, 5.5],
    2019: [4.9, 4.6, 5.1, 4.7, 5.3, 5.0, 5.4, 4.8, 5.2, 4.9, 5.5, 5.1],
    2020: [5.4, 5.2, 1.1, -8.4, -6.2, 1.8, 2.9, 2.6, 3.4, 3.1, 2.4, 5.6],
    2021: [6.2, 9.8, 12.4, 15.1, 16.3, 15.7, 14.9, 14.2, 13.8, 15.2, 16.8, 15.4],
    2022: [14.1, 12.6, 13.9, 11.8, 10.4, 12.1, 9.8, 10.6, 8.9, 8.2, 7.4, 6.8],
    2023: [5.1, 4.4, 3.2, 2.6, 1.9, 1.4, 2.1, 3.3, 4.0, 3.6, 4.4, 5.2],
    2024: [4.6, 5.3, 4.9, 5.6, 5.1, 4.8, 5.4, 6.0, 5.5, 5.2, 5.9, 6.3],
    2025: [5.7, 6.4, 5.9, 6.6, 6.1, 5.8, 6.4, 7.0, 6.5, 6.2, 6.8, 7.2],
    2026: [6.6, 7.1, 6.5, 7.3, 6.9, 6.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
# months flagged 0.0 are in-progress / not-yet-printed and dropped by ``redbook_series``.


def have_real(path: str = DEFAULT_CACHE) -> bool:
    """True iff the price cache exists (the Redbook proxy table is always available)."""
    return os.path.exists(path)


def redbook_series() -> pd.Series:
    """Monthly Redbook same-store-sales YoY (percent), indexed by month-end date.

    The in-progress months of the final year (zeros) are dropped so a stamped run never
    includes a partial bar.
    """
    rows = []
    for yr in sorted(REDBOOK_YOY_BY_YEAR):
        for m, v in enumerate(REDBOOK_YOY_BY_YEAR[yr], start=1):
            if v == 0.0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="redbook_yoy").sort_index()


def fetch_prices(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download XRT + SPY daily adjusted close via yfinance and cache it (network-only).

    Used once to build ``_cache/xrt_spy_prices.csv``. Never imported by offline cells.
    Total-return adjusted close (``auto_adjust=True``). XRT lists 2006-06, which bounds the
    usable sample.
    """
    import yfinance as yf

    raw = yf.download(["XRT", "SPY"], start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    out = raw[["XRT", "SPY"]].dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached daily XRT + SPY adjusted-close frame (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df[["XRT", "SPY"]].astype(float)


def prices_monthly(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Month-end XRT + SPY adjusted closes aligned to the monthly Redbook grid."""
    px = load_prices(path)
    return px.resample("ME").last().dropna()


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: Redbook YoY + month-end XRT & SPY closes.

    Columns: ``redbook`` (percent YoY), ``xrt`` (price), ``spy`` (price). Only months present
    in all three series are kept. This is the real-tape object the strategy runs on.
    """
    rb = redbook_series()
    px = prices_monthly(path)
    df = pd.DataFrame({"redbook": rb, "xrt": px["XRT"], "spy": px["SPY"]}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_redbook(n_months: int = 360, edge: float = 0.0, seed: int = 759,
                      mu_m: float = 0.008, sig_m: float = 0.060) -> pd.DataFrame:
    """Deterministic monthly Redbook + XRT-like price with a PLANTED momentum->returns link.

    Builds a persistent Redbook-like YoY series (AR(1) around ~+4%) and a monthly XRT-like
    return series (retail is higher-vol than the broad market: ``sig_m`` default 6%/month).
    When ``edge != 0`` the *forward* monthly return is perturbed by ``+edge`` whenever Redbook
    momentum (the 3-month change in the smoothed YoY level) is **accelerating** — the
    believers' story (accelerating same-store sales lift retail stocks) injected by
    construction, with a knob.

    ``edge = 0`` => Redbook momentum carries no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.05 monthly) must
    drive the Welch t well past 2. A flat ``spy`` column is included so the relative
    (XRT-minus-SPY) machinery runs on the control too. The date index is a decorative monthly
    label built with ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)

    # persistent Redbook-like YoY around +4%, AR(1)
    yoy = np.empty(n_months)
    yoy[0] = 4.0
    target = 4.0
    for t in range(1, n_months):
        yoy[t] = target + 0.90 * (yoy[t - 1] - target) + rng.normal(0, 0.8)

    # base monthly returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # momentum: 3-month change of the 3-month-smoothed YoY level
    smooth = pd.Series(yoy).rolling(3, min_periods=1).mean().values
    mom = np.zeros(n_months)
    mom[3:] = smooth[3:] - smooth[:-3]
    accel = mom > 0

    # plant the believers' link: accelerating Redbook at t -> higher return at t+1
    if edge != 0.0:
        for t in range(n_months - 1):
            if accel[t]:
                ret[t + 1] += edge

    price = 100.0 * np.exp(np.cumsum(ret))
    spy_ret = rng.normal(0.006, 0.040, size=n_months)          # independent broad-market leg
    spy = 100.0 * np.exp(np.cumsum(spy_ret))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"redbook": yoy, "xrt": price, "spy": spy}, index=idx)
