"""Data layer for Study 544 (Oyster-R-Months) — the folk 'eat oysters only in R months' calendar.

The oyster rule: eat oysters only in months whose name contains the letter **R** — September,
October, November, December, January, February, March, April — and skip the four R-less months
of **May, June, July, August**. (The folk rationale is spawning + summer heat before
refrigeration; see docs/references.md.) It is, calendrically, almost exactly the
**sell-in-May / Halloween** split (Bouman & Jacobsen 2002), just one month wider on the winter
side (R-months add September; the Halloween "winter" is Nov–Apr, "summer" is May–Oct).

This study tests whether *market* and *consumer-staples/seafood* stock returns follow the same
R-month calendar: do R-months (Sep–Apr) out-earn R-less months (May–Aug)?

Two tapes, one schema (a monthly simple-return Series with a DatetimeIndex):

- ``synthetic_monthly`` — a deterministic, offline generator. A single knob, ``r_premium_bp``,
  plants an extra drift in the eight R-months; ``= 0`` is the null (no seasonal). The positive
  control and the null in one bottle. Tests never touch the network.
- ``fetch_monthly`` — the real yfinance tape (daily adjusted close → month-end total return),
  cache-first into this study's own ``_cache/`` so the reproducible core never needs the network.
  Two instruments: the broad market (**^GSPC**, S&P 500, back to 1928) and consumer staples
  (**XLP**, the SPDR staples ETF that holds the food/beverage/grocery names an "oyster" seasonal
  would plausibly touch, back to 1998).

Survivorship / look-ahead notes:

- The market index and the ETF are *indices*, so single-name survivorship does not apply; the
  calendar label (does the month name contain 'R'?) is fully known in advance — there is no
  estimation and therefore no look-ahead in the signal. The one honest execution point is the
  fill: a month's return is realised over that whole month, and the R/non-R label is known before
  the month starts, so the position is on for the full month (the documented convention; a
  one-day-later entry is immaterial on a monthly bar).
- Partial months are dropped (we keep only complete calendar months).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# The eight months whose English name contains the letter 'R': Sep–Apr.
# (janua*R*y, feb*R*ua*R*y, ma*R*ch, ap*R*il, septembe*R*, octobe*R*, novembe*R*, decembe*R*)
R_MONTHS = {1, 2, 3, 4, 9, 10, 11, 12}
NON_R_MONTHS = {5, 6, 7, 8}  # may, june, july, august — no 'R'

# The two real instruments: broad market and consumer staples.
INSTRUMENTS = {
    "market": "^GSPC",   # S&P 500 price index (long history back to 1928)
    "staples": "XLP",    # SPDR Consumer-Staples ETF (food/beverage/grocery; from 1998)
}


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic monthly series."""

    r_premium_bp: float  # extra monthly drift (bp) added to the eight R-months

    @property
    def has_seasonal(self) -> bool:
        return self.r_premium_bp != 0.0


# ---------------------------------------------------------------------------
# Synthetic monthly series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_years: int = 80,
    base_bp: float = 50.0,
    r_premium_bp: float = 40.0,
    vol_ann: float = 0.15,
    seed: int = 544,
) -> tuple[pd.Series, WorldTruth]:
    """A monthly simple-return series with a tunable R-month bump — deterministic given ``seed``.

    Each month gets ``base_bp`` of drift; the eight R-months (Sep–Apr) get an extra
    ``r_premium_bp``. ``r_premium_bp = 0`` is the null (returns unrelated to the R/non-R label).
    Returns ``(series, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1946-01-31", periods=n, freq="ME", name="date")
    is_r = pd.DatetimeIndex(idx).month.isin(R_MONTHS).astype(float)
    drift = base_bp * 1e-4 + (r_premium_bp * 1e-4) * is_r
    r = drift + (vol_ann / np.sqrt(12)) * rng.standard_normal(n)
    return pd.Series(r, index=idx, name="ret"), WorldTruth(r_premium_bp)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily close → month-end total return, cache-first
# ---------------------------------------------------------------------------
def _cache_path(name: str) -> str:
    return os.path.join(DEFAULT_CACHE, f"oyster_{name}_monthly.parquet")


def fetch_monthly(
    name: str = "market",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    end: str = "2026-06-30",
) -> pd.Series:
    """Monthly simple returns for one instrument (``market`` or ``staples``), cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    Series. Network is touched only on an explicit ``fetch=True`` (then cached). Daily adjusted
    close is resampled to month-end; the last (partial) month is dropped so every bar is a
    complete calendar month.
    """
    if name not in INSTRUMENTS:
        raise ValueError(f"unknown instrument {name!r}; use one of {list(INSTRUMENTS)}")
    path = _cache_path(name)
    if os.path.exists(path):
        return pd.read_parquet(path)["ret"]
    if not fetch:
        return pd.Series(dtype=float)

    import yfinance as yf  # lazy

    ticker = INSTRUMENTS[name]
    px = yf.download(ticker, period="max", interval="1d", auto_adjust=True,
                     progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px.loc[:end]
    mret = px.resample("ME").last().pct_change().dropna().squeeze()
    # Drop a trailing partial month: keep only months whose month-end is <= end and complete.
    mret = mret[mret.index <= pd.Timestamp(end)]
    mret.name = "ret"
    mret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    mret.to_frame().to_parquet(path)
    return mret


def r_month_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True where the month name contains the letter 'R' (Sep–Apr)."""
    return pd.DatetimeIndex(index).month.isin(R_MONTHS).to_numpy()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
