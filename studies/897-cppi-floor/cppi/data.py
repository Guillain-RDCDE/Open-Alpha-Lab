"""Data layer for Study 897 — CPPI Floor.

The claim under test — **Constant-Proportion Portfolio Insurance** (Black & Jones 1987;
Perold & Sharpe 1988): protect a **floor** ``F`` (a fraction of NAV that accretes at the
cash rate) by holding a fixed **multiplier** ``m`` times the **cushion** ``C = V − F`` in
equities and the rest in bills, rebalancing daily as the cushion moves. As the market
falls the cushion shrinks and the rule mechanically de-risks — a *self-financing* dynamic
floor. The questions: does the floor actually hold across 2008 / 2020 / 2022, and what
does the insurance cost in upside (excess-of-cash Sharpe) versus buy-and-hold?

Two tapes, one shape (a daily frame of total-return price columns, one per asset):

* :func:`synthetic_prices` — fully **offline**, deterministic. A geometric risky asset with a
  tunable daily ``drift`` / ``sigma``. The mechanical claim CPPI *guarantees* (absent gaps) is
  a bounded drawdown, so the positive control is a **bear** tape (``drift < 0``): buy-and-hold
  craters while CPPI de-risks and its drawdown stays pinned near the floor — versus a **calm**
  null (small positive drift, low vol) where nothing draws down and there is nothing to
  protect. An independent ``gap_prob`` / ``gap_size`` knob injects occasional overnight crashes
  so the **gap-risk** floor-breach machinery can be shown to fire only when a one-day drop
  exceeds ``1/m``. A near-constant positive ``BIL`` column stands in for the risk-free /
  floor-accretion leg. Deterministic given ``seed``; no network.

* :func:`fetch` / :func:`load_prices` — the real daily **total-return** closes (yfinance
  ``auto_adjust=True`` ⇒ dividends + splits folded in) for SPY (the risky leg), IEF
  (7–10y Treasuries, a bond-safe-leg robustness) and **BIL** (1–3m T-bills, the cash /
  floor-accretion / excess-of-cash proxy), each cached to parquet under this study's OWN
  ``_cache/``. **Cache-only** unless ``fetch=True``: the offline core (and CI) never
  imports ``yfinance``. BIL lists **2007-05-30**, which bounds the joint window honestly —
  a young-ETF, single-cycle short-sample caveat we name on the Signal axis (but it does put
  the 2008 crash in-sample, exactly where a floor is supposed to earn its keep).

Daily total-return closes are the right input: CPPI is a daily position-sizing overlay in
the source literature, and total return is the fair series for a long-horizon book.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ("SPY", "IEF", "BIL")
RISKY_TICKER = "SPY"
CASH_TICKER = "BIL"
START = "2007-05-30"          # BIL inception — the binding joint-window start
AS_OF = "2026-06-30"          # last complete calendar month at publication
TRADING_DAYS = 252

__all__ = [
    "TICKERS", "RISKY_TICKER", "CASH_TICKER", "START", "AS_OF", "CACHE_DIR",
    "TRADING_DAYS", "GroundTruth", "synthetic_prices", "fetch", "have_real",
    "load_prices",
]


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, a trending risky asset + optional crash gaps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the machinery."""
    drift: float              # constant daily risky drift (negative -> a bear tape)
    sigma: float              # daily risky vol scale
    gap_prob: float           # per-day probability of an overnight crash gap
    gap_size: float           # size of the crash gap (a positive number, subtracted)
    cash_rate_ann: float      # annual cash / floor-accretion rate
    n_days: int

    @property
    def is_bear(self) -> bool:
        """True on a falling tape — where CPPI's mechanical floor is supposed to bite."""
        return self.drift < 0.0

    @property
    def has_gap_risk(self) -> bool:
        """True when the tape carries crash gaps big enough to threaten a CPPI floor."""
        return self.gap_prob > 0.0 and self.gap_size > 0.0


def synthetic_prices(
    n_days: int = 6000,
    drift: float = 0.00030,
    sigma: float = 0.011,
    gap_prob: float = 0.0,
    gap_size: float = 0.0,
    bond_drift: float = 0.00012,
    bond_sigma: float = 0.0035,
    cash_rate_ann: float = 0.02,
    start: str = "2007-05-30",
    seed: int = 897,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A geometric risky asset (``SPY``) with tunable ``drift`` / ``sigma``, plus a bond and cash.

    Risky returns are ``r_t = drift + sigma · z_t``. The positive control turns on ``drift < 0``
    (a **bear** tape): buy-and-hold craters while CPPI de-risks into cash and its drawdown stays
    pinned near the floor — the mechanical guarantee. A **calm** null (small positive drift, low
    vol) draws down little, so CPPI and buy-and-hold nearly coincide (nothing to protect). With
    ``gap_prob > 0`` a fraction of days receive an extra ``−gap_size`` overnight crash so the
    floor-breach machinery has something to catch (a breach needs a one-day drop past ``1/m``).
    A ``BIL`` column accrues a near-constant daily bill rate (the floor accretes at this rate);
    ``IEF`` is a mild low-vol bond.

    Returns ``(frame, truth)`` with columns ``['SPY', 'IEF', 'BIL']`` (price levels from 100, so
    the same downstream code runs on synthetic and real) on a business-day ``Date`` index;
    ``truth`` carries the baked-in parameters and the ``is_bear`` / ``has_gap_risk`` flags.
    Deterministic given ``seed``. ``n_days`` well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="Date")

    z = rng.standard_normal(n_days)
    r_risky = drift + sigma * z

    # Optional overnight crash gaps — deterministic given the seed.
    if gap_prob > 0.0 and gap_size > 0.0:
        gap_days = rng.random(n_days) < gap_prob
        r_risky = np.where(gap_days, r_risky - gap_size, r_risky)

    r_bond = bond_drift + bond_sigma * rng.standard_normal(n_days)
    cash_daily = cash_rate_ann / TRADING_DAYS
    r_cash = np.full(n_days, cash_daily)

    frame = pd.DataFrame(
        {
            "SPY": 100.0 * np.cumprod(1.0 + r_risky),
            "IEF": 100.0 * np.cumprod(1.0 + r_bond),
            "BIL": 100.0 * np.cumprod(1.0 + r_cash),
        },
        index=idx,
    )
    truth = GroundTruth(
        drift=drift, sigma=sigma, gap_prob=gap_prob, gap_size=gap_size,
        cash_rate_ann=cash_rate_ann, n_days=n_days,
    )
    return frame, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! daily total-return closes, cached per ticker
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"close_{safe}.parquet")


def _fetch_one(ticker: str, retries: int = 4) -> pd.Series:
    """Download one ticker's total-return close, retrying transient network failures."""
    import time

    import yfinance as yf  # lazy: offline core never imports it

    last = None
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, period="max", interval="1d",
                              auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                col = raw["Close"]
                if isinstance(col, pd.DataFrame):  # yfinance MultiIndex on a single ticker
                    col = col.iloc[:, 0]
                close = col.astype("float64")
                close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                close.index.name = "Date"
                close.name = ticker
                return close[~close.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # pragma: no cover - network path
            last = exc
        time.sleep(1.5 * (attempt + 1))
    if last is not None:  # pragma: no cover
        raise last
    return pd.Series(dtype="float64", name=ticker)


def fetch(tickers: tuple[str, ...] = TICKERS) -> None:
    """Download each ticker's total-return close and cache it to parquet (network path)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for tk in tickers:
        close = _fetch_one(tk)
        if not close.empty:
            close.to_frame().to_parquet(_cache_path(tk))


def have_real(tickers: tuple[str, ...] = TICKERS) -> bool:
    """True only if every ticker's parquet is present in this study's cache."""
    return all(os.path.exists(_cache_path(tk)) for tk in tickers)


def load_prices(
    tickers: tuple[str, ...] = TICKERS,
    start: str = START,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Cached real total-return price frame, inner-joined and sliced to ``[start, asof]``.

    Reads the per-ticker parquet directly — **OFFLINE**, no yfinance import. Columns are the
    upper-case tickers; the frame is the inner join across tickers (dropna) so it begins at
    the latest inception (BIL, 2007-05-30). Returns an empty frame if any ticker is uncached.
    """
    cols: dict[str, pd.Series] = {}
    for tk in tickers:
        path = _cache_path(tk)
        if not os.path.exists(path):
            return pd.DataFrame()
        cols[tk] = pd.read_parquet(path)[tk]
    frame = pd.concat(cols.values(), axis=1, join="inner").dropna()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    frame = frame[(frame.index >= lo) & (frame.index <= hi)]
    frame.index.name = "Date"
    return frame
