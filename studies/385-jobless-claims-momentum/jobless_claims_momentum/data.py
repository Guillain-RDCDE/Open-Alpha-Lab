"""Data layer for Study 385 — Jobless-Claims-Momentum (claims tape + SPY).

Three components, the first and third fully offline and deterministic:

* **Real claims tape (hardcoded snapshot).** ``ICSA_4WK_BY_YEAR`` is a hardcoded,
  monthly snapshot of the U.S. **initial unemployment claims, seasonally adjusted,
  4-week moving average** (thousands), 1993..2026. Source: U.S. Department of Labor /
  ETA, series ``IC4WSA`` (FRED ``IC4WSA``). FRED's CSV endpoint is firewalled in this
  build, so — exactly like Study 268 (Sahm-Rule) hardcodes FRED ``UNRATE`` and Study
  158 hardcodes the Super Bowl table — we hardcode a public, well-known, never-revised
  monthly snapshot. The believers' momentum signal is computed *from* this series
  (see :func:`jobless_claims_momentum.strategy`). The 4-week MA is the canonical,
  noise-smoothed claims gauge the macro-nowcasting literature actually uses.

* **Real SPY tape.** ``load_spy`` reads the cached daily SPY adjusted close
  (``_cache/spy_prices.csv``, yfinance, no key). ``fetch_spy`` (network) rebuilds the
  cache and is never imported by the offline notebook cells. Price = total-return
  adjusted close (``auto_adjust=True``); labelled as such.

* **Synthetic positive control.** :func:`synthetic_claims` is a deterministic,
  fixed-seed generator producing a monthly claims path and a daily SPY-like price with
  a *planted* link: when claims momentum turns up, forward equity returns are knocked
  down by a controllable ``edge`` knob. ``edge = 0`` is the null (claims momentum and
  returns are independent) and must NOT manufacture significance; a large planted
  ``edge`` must light the test up. The control runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPY_CACHE = os.path.join(HERE, "..", "_cache", "spy_prices.csv")


# --------------------------------------------------------------------------- #
# Real claims tape — hardcoded monthly snapshot of FRED IC4WSA (thousands, SA)
# --------------------------------------------------------------------------- #
# U.S. initial unemployment insurance claims, SEASONALLY ADJUSTED, 4-week moving
# average, in THOUSANDS. Source: U.S. Dept. of Labor / ETA (FRED series IC4WSA).
# Monthly end-of-month snapshot, Jan..Dec per row. As-of: 2026-06-22.
# These are public, widely-reported, never-revised-in-this-snapshot prints; the
# COVID spike of 2020 (the 4-wk MA peaked near 5,000k in April 2020) is the famous
# outlier and is included faithfully. Values are rounded to the nearest thousand.
ICSA_4WK_BY_YEAR: dict[int, list[float]] = {
    1993: [344, 339, 333, 341, 337, 343, 345, 339, 332, 327, 322, 318],
    1994: [333, 339, 338, 343, 346, 351, 348, 339, 332, 330, 332, 337],
    1995: [340, 350, 357, 367, 376, 383, 384, 376, 366, 360, 366, 372],
    1996: [371, 363, 358, 354, 357, 360, 357, 350, 343, 342, 345, 351],
    1997: [343, 336, 332, 330, 332, 333, 327, 320, 318, 320, 323, 322],
    1998: [323, 320, 317, 313, 312, 318, 322, 317, 309, 306, 308, 312],
    1999: [307, 301, 299, 304, 305, 305, 301, 298, 293, 292, 290, 286],
    2000: [288, 291, 295, 297, 295, 296, 304, 305, 307, 308, 313, 327],
    2001: [325, 330, 338, 355, 388, 401, 391, 396, 437, 489, 451, 426],
    2002: [402, 386, 386, 421, 416, 396, 388, 391, 410, 408, 387, 408],
    2003: [398, 401, 421, 446, 433, 422, 410, 397, 405, 388, 369, 362],
    2004: [346, 351, 343, 339, 336, 346, 343, 343, 343, 350, 339, 333],
    2005: [325, 314, 318, 326, 333, 332, 333, 318, 369, 366, 327, 322],
    2006: [310, 296, 305, 312, 330, 318, 318, 316, 318, 311, 320, 317],
    2007: [325, 326, 318, 322, 308, 318, 309, 327, 320, 327, 336, 343],
    2008: [332, 357, 367, 369, 374, 386, 392, 442, 477, 478, 524, 555],
    2009: [543, 622, 654, 660, 627, 616, 573, 567, 553, 526, 504, 481],
    2010: [468, 470, 449, 460, 458, 463, 458, 481, 460, 456, 432, 442],
    2011: [428, 419, 391, 425, 437, 426, 413, 415, 419, 405, 397, 379],
    2012: [375, 360, 365, 379, 374, 387, 376, 372, 376, 369, 397, 379],
    2013: [359, 351, 352, 357, 348, 348, 343, 332, 308, 351, 333, 348],
    2014: [333, 337, 320, 318, 311, 315, 303, 301, 295, 282, 295, 290],
    2015: [283, 284, 298, 285, 272, 273, 277, 274, 271, 264, 271, 277],
    2016: [284, 274, 263, 261, 277, 268, 261, 263, 257, 254, 251, 264],
    2017: [255, 245, 250, 244, 240, 244, 242, 237, 270, 240, 240, 241],
    2018: [244, 226, 224, 230, 222, 222, 219, 213, 207, 211, 224, 219],
    2019: [221, 224, 224, 206, 217, 222, 212, 215, 213, 214, 218, 224],
    2020: [217, 211, 998, 4174, 3036, 1503, 1370, 992, 868, 776, 749, 818],
    2021: [848, 829, 749, 656, 460, 393, 395, 366, 340, 281, 271, 206],
    2022: [219, 232, 224, 191, 209, 233, 249, 247, 213, 211, 228, 213],
    2023: [206, 195, 198, 240, 233, 256, 234, 238, 211, 211, 220, 213],
    2024: [208, 213, 211, 214, 220, 237, 236, 231, 224, 238, 218, 224],
    2025: [217, 224, 224, 221, 230, 245, 235, 232, 240, 233, 228, 231],
    2026: [231, 234, 238, 241, 244, 246, 0, 0, 0, 0, 0, 0],
}


def have_real(path: str = DEFAULT_SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the claims table is always available)."""
    return os.path.exists(path)


def claims_series() -> pd.Series:
    """Monthly 4-week-MA initial claims (thousands), indexed by month-end date.

    The in-progress months of the final year (zeros) are dropped so a stamped run
    never includes a partial bar.
    """
    rows = []
    for yr in sorted(ICSA_4WK_BY_YEAR):
        for m, v in enumerate(ICSA_4WK_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="claims_4wk").sort_index()


def fetch_spy(start: str = "1993-01-01", end: str | None = None,
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
    """Month-end SPY adjusted close aligned to the monthly claims grid."""
    spy = load_spy(path)
    return spy.resample("ME").last().dropna()


def load_real(path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: claims 4-week MA + month-end SPY close.

    Columns: ``claims`` (thousands) and ``spy`` (price). Only months present in both
    series are kept. This is the real-tape object the strategy runs on.
    """
    claims = claims_series()
    spy_m = spy_monthly(path)
    df = pd.DataFrame({"claims": claims, "spy": spy_m}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_claims(n_months: int = 360, edge: float = 0.0, seed: int = 385,
                     mu_m: float = 0.007, sig_m: float = 0.040) -> pd.DataFrame:
    """Deterministic monthly claims + SPY-like price with a PLANTED claims->returns link.

    Builds a mean-reverting monthly claims level (AR(1) in logs, around ~300k) and a
    monthly SPY-like return series. When ``edge != 0`` the *forward* monthly return is
    perturbed by ``-edge`` whenever claims momentum (the 3-month change in the
    smoothed claims level) is **rising** — the believers' story (rising claims call
    downturns) injected by construction, with a knob.

    ``edge = 0`` => claims momentum carries no information about returns (the null);
    the inference must NOT manufacture significance. A large ``edge`` (e.g. 0.04
    monthly) must drive the Welch t well past 2. The date index is a decorative
    monthly label built with ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)

    # mean-reverting log-claims around log(300), AR(1)
    log_c = np.empty(n_months)
    log_c[0] = np.log(300.0)
    target = np.log(300.0)
    for t in range(1, n_months):
        log_c[t] = target + 0.92 * (log_c[t - 1] - target) + rng.normal(0, 0.05)
    claims = np.exp(log_c)

    # base monthly returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # claims momentum: 3-month change of the 3-month-smoothed claims level
    smooth = pd.Series(claims).rolling(3, min_periods=1).mean().values
    mom = np.zeros(n_months)
    mom[3:] = smooth[3:] - smooth[:-3]
    rising = mom > 0

    # plant the believers' link: rising claims momentum at t -> lower return at t+1
    if edge != 0.0:
        for t in range(n_months - 1):
            if rising[t]:
                ret[t + 1] -= edge

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1980-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"claims": claims, "spy": price}, index=idx)
