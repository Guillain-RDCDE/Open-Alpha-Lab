"""Data layer for Study 762 — Vegas-Gaming-Win (Strip GGR tape + casino basket).

Three components, the first and third fully offline and deterministic:

* **Strip GGR tape (hardcoded reconstruction).** ``STRIP_GGR_BY_YEAR`` is a hardcoded,
  monthly, clearly-labelled **approximate reconstruction** of the Nevada Gaming Control
  Board's "**Las Vegas Strip**" gross-gaming-revenue line (US$ millions). The NGCB
  publishes the number monthly (in the *Nevada Gaming Revenue Report*, a PDF released about
  five weeks after the reference month); those PDFs aren't machine-fetchable in this build,
  so — exactly as Study 385 hardcodes a snapshot of FRED ``IC4WSA`` and Study 358
  (Watch-Index) / Study 708 (Eurovision) hardcode a labelled proxy series — we hardcode a
  monthly series whose **annual sums match the published Strip totals** (~$6.6B/yr pre-COVID,
  the near-total 2020 closure, the record ~$8.8B of 2023–24) with a plausible seasonal
  shape. It is a *labelled reconstruction*, not the settled NGCB print, and that caveat
  travels on the Signal axis. The believers' momentum signal is computed *from* this series
  (see :mod:`vegas_gaming_win.strategy`).

* **Real casino basket.** ``load_prices`` reads cached daily adjusted closes for a fixed
  basket of listed casino operators (``_cache/casino_prices.csv``, yfinance, no key).
  ``fetch_basket`` (network) rebuilds the cache and is never imported by the offline
  notebook cells. Prices are total-return adjusted (``auto_adjust=True``); labelled as such.
  Survivorship is named on the Signal axis: the basket is *today's* listed operators, so it
  omits names that delisted/were acquired (Caesars restructured in 2015–17; Station, Pinnacle,
  Isle, Ameristar were absorbed) — a mild upward tilt on the basket's realised return.

* **Synthetic positive control.** :func:`synthetic_ggr` is a deterministic, fixed-seed
  generator producing a monthly GGR path and a daily basket-like price with a *planted*
  link: when GGR momentum turns up, forward equity returns are lifted by a controllable
  ``edge`` knob. ``edge = 0`` is the null (GGR momentum and returns are independent) and must
  NOT manufacture significance; a large planted ``edge`` must light the test up. The control
  runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "casino_prices.csv")

# Fixed basket of listed casino operators. Chosen for gaming-sector exposure and a usable
# price history; the panel fills in as each name lists (LVS Dec-2004, WYNN Oct-2002, PENN,
# BYD, MGM long-listed; CZR = the Eldorado/Caesars combined entity). This is a SURVIVING
# basket — named on the Signal axis. Equal-weight of whatever names have data each month.
BASKET = ["MGM", "LVS", "WYNN", "CZR", "BYD", "PENN"]

# --------------------------------------------------------------------------- #
# Strip GGR tape — hardcoded monthly reconstruction (US$ millions)
# --------------------------------------------------------------------------- #
# Nevada Gaming Control Board "Las Vegas Strip" gross gaming revenue, US$ MILLIONS, by
# calendar month (Jan..Dec per row). This is a LABELLED APPROXIMATE RECONSTRUCTION whose
# annual sums track the published Strip totals; it is NOT the settled monthly NGCB print.
# The 2020 COVID closure (casinos shut Mar 18 -> reopened Jun 4, 2020) is represented
# faithfully (Apr-May ~0). As-of: 2026-07-13. Partial 2025 (H2 zeros) is dropped downstream.
STRIP_GGR_BY_YEAR: dict[int, list[float]] = {
    2000: [390, 375, 435, 400, 425, 390, 410, 405, 370, 410, 380, 410],
    2001: [385, 370, 425, 390, 415, 380, 405, 395, 365, 400, 370, 400],
    2002: [380, 365, 420, 390, 410, 375, 400, 390, 360, 395, 370, 395],
    2003: [410, 390, 455, 415, 440, 405, 430, 420, 390, 425, 395, 425],
    2004: [435, 420, 485, 445, 470, 430, 455, 450, 415, 455, 420, 455],
    2005: [490, 470, 550, 500, 535, 485, 520, 510, 465, 515, 475, 515],
    2006: [545, 525, 610, 560, 590, 540, 575, 565, 520, 570, 530, 570],
    2007: [560, 535, 620, 570, 605, 550, 585, 575, 530, 580, 540, 580],
    2008: [500, 480, 555, 510, 540, 495, 525, 515, 475, 520, 485, 520],
    2009: [455, 435, 505, 460, 490, 450, 475, 465, 430, 470, 440, 470],
    2010: [470, 455, 525, 480, 510, 465, 495, 485, 450, 490, 460, 490],
    2011: [495, 475, 550, 505, 535, 490, 520, 510, 470, 515, 480, 515],
    2012: [505, 485, 565, 520, 550, 500, 535, 525, 480, 530, 490, 530],
    2013: [530, 510, 590, 540, 575, 525, 560, 550, 505, 555, 515, 555],
    2014: [520, 500, 580, 530, 565, 515, 545, 535, 495, 540, 505, 540],
    2015: [520, 495, 575, 530, 560, 515, 545, 535, 490, 540, 505, 540],
    2016: [525, 500, 580, 535, 565, 515, 550, 540, 495, 545, 505, 545],
    2017: [530, 505, 590, 540, 570, 525, 555, 545, 500, 550, 510, 550],
    2018: [540, 515, 600, 550, 580, 535, 565, 555, 510, 560, 520, 560],
    2019: [540, 515, 600, 550, 580, 535, 565, 555, 510, 560, 520, 560],
    2020: [605, 565, 183, 0, 5, 158, 335, 405, 368, 430, 175, 415],
    2021: [575, 555, 640, 590, 625, 570, 605, 595, 545, 600, 560, 600],
    2022: [675, 650, 755, 690, 730, 670, 710, 700, 640, 705, 655, 705],
    2023: [720, 690, 800, 735, 780, 715, 760, 745, 685, 750, 700, 750],
    2024: [720, 690, 800, 735, 780, 715, 760, 745, 685, 750, 700, 750],
    2025: [725, 695, 805, 740, 780, 715, 0, 0, 0, 0, 0, 0],
}


def have_real(path: str = DEFAULT_CACHE) -> bool:
    """True iff the casino-basket cache exists (the GGR table is always available)."""
    return os.path.exists(path)


def ggr_series() -> pd.Series:
    """Monthly Strip GGR (US$ millions), indexed by month-end date.

    In-progress / unpublished months (zeros) are dropped so a stamped run never includes a
    partial bar. Note the *legitimate* zeros of Apr-May 2020 (the COVID closure) are kept —
    they are real revenue of ~0, not missing data — while the trailing zeros of the final
    partial year are dropped. We distinguish the two by position: only trailing zeros at the
    end of the last year are treated as unpublished.
    """
    rows: list[tuple[pd.Timestamp, float]] = []
    years = sorted(STRIP_GGR_BY_YEAR)
    last_year = years[-1]
    for yr in years:
        vals = STRIP_GGR_BY_YEAR[yr]
        for m, v in enumerate(vals, start=1):
            # drop only the trailing unpublished zeros of the final year
            if yr == last_year and v == 0 and all(x == 0 for x in vals[m - 1:]):
                break
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="ggr").sort_index()


def fetch_basket(start: str = "2002-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the casino basket via yfinance and cache a wide adj-close CSV (network-only).

    Used once to build ``_cache/casino_prices.csv``. Never imported by offline cells.
    Total-return adjusted close (``auto_adjust=True``). Keeps names present for >=25% of the
    window (each operator's listed history).
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.25]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = casino tickers)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def basket_monthly(path: str = DEFAULT_CACHE) -> pd.Series:
    """Equal-weight casino-basket price index at month-ends (total-return adjusted).

    Each month we equal-weight the *available* names' monthly returns (a name enters the
    average once it has a price), then compound into a level index normalised to 100 at the
    first month. This is the tradable proxy for "the casino stocks" the folklore points at.
    """
    px = load_prices(path)
    m = px.resample("ME").last()
    rets = m.pct_change()
    basket_ret = rets.mean(axis=1, skipna=True)          # equal-weight of available names
    basket_ret = basket_ret.dropna()
    level = 100.0 * (1.0 + basket_ret).cumprod()
    level.name = "basket"
    return level


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: Strip GGR + equal-weight casino-basket level.

    Columns: ``ggr`` (US$ millions) and ``basket`` (level index, total-return adjusted).
    Only months present in both series are kept. This is the real-tape object the strategy
    runs on.
    """
    ggr = ggr_series()
    basket = basket_monthly(path)
    df = pd.DataFrame({"ggr": ggr, "basket": basket}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_ggr(n_months: int = 360, edge: float = 0.0, seed: int = 762,
                  mu_m: float = 0.008, sig_m: float = 0.075) -> pd.DataFrame:
    """Deterministic monthly GGR + basket-like level with a PLANTED GGR->returns link.

    Builds a trending, seasonal monthly GGR level and a monthly casino-basket-like return
    series. When ``edge != 0`` the *forward* monthly return is lifted by ``+edge`` whenever
    GGR momentum (the 3-month change in the trailing-12-month GGR sum) is **rising** — the
    believers' story (rising GGR momentum -> casino stocks run) injected by construction,
    with a knob. Casino equities are ~2x as volatile as the market, so ``sig_m`` is large.

    ``edge = 0`` => GGR momentum carries no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.05 monthly) must
    drive the Welch t well past 2. The date index is a decorative monthly label built with
    ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)

    # trending, seasonal GGR level around a slow random walk
    trend = np.cumsum(rng.normal(0.0, 6.0, size=n_months)) + 500.0
    seas = 30.0 * np.sin(2.0 * np.pi * (np.arange(n_months) % 12) / 12.0)
    ggr = np.clip(trend + seas + rng.normal(0, 8.0, size=n_months), 50.0, None)

    # base monthly basket returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # GGR momentum: 3-month change of the trailing-12-month GGR sum
    ttm = pd.Series(ggr).rolling(12, min_periods=1).sum().values
    mom = np.zeros(n_months)
    mom[3:] = ttm[3:] - ttm[:-3]
    rising = mom > 0

    # plant the believers' link: rising GGR momentum at t -> higher return at t+1
    if edge != 0.0:
        for t in range(n_months - 1):
            if rising[t]:
                ret[t + 1] += edge

    level = 100.0 * np.cumprod(1.0 + ret)
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"ggr": ggr, "basket": level}, index=idx)
