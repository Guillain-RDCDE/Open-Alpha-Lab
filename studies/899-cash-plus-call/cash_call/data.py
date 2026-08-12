"""Data layer for Study 899 — Cash + Call "90/10".

The claim under test — Bill **Gross's "90/10"** (and the older Zvi Bodie "T-bills + calls"
capital-protection idea): keep **~90% in T-bills** so the bills accrete back toward par
(capital is roughly protected) and spend the remaining **~10%** on **convex upside** — buying
call options. If the market falls the calls expire worthless and you lose only the ~10% premium
(the bills have grown, so you end roughly flat); if it rockets, the calls give leveraged,
capped-loss upside. The question the desk asks: does the asymmetric *"protect capital, rent
upside"* profile beat plain buy-and-hold on **risk-adjusted** terms across crashes, or is the
option premium a standing drag that a convex payoff cannot pay for?

**Single-name / index option history is not freely available**, so the ~10% convex sleeve is a
**documented proxy**: a rolling **1-year at-the-money SPY call** *marked-to-market daily with the
Black–Scholes formula* (strike = spot at each annual roll, priced off SPY's trailing realized
vol and the ^IRX bill rate). The 10% premium each year buys as much call notional as that fair
price affords — so protection is correctly **more expensive in high-vol regimes** (exactly when
you want it). This is a **model proxy for a real listed call**, clearly labelled: it carries
Black–Scholes model risk and it ignores the dividend the call-holder forgoes (a small optimistic
tilt we name on the Signal axis). An honest structural test — likely Weak/Fragile.

Two tapes, one shape (a daily frame of SPY / BIL total-return price columns + an ^IRX rate column):

* :func:`synthetic_prices` — fully **offline**, deterministic. A geometric SPY with tunable
  ``drift`` / ``sigma`` plus optional **up-jumps** (``up_jump_prob`` / ``up_jump_size`` ⇒ a
  positively-skewed tape where a *convex* call sleeve should out-capture a linear book) and a
  **crash** knob (negative ``drift`` ⇒ a bear tape where the capital-protection floor must
  visibly hold). A near-constant ``BIL`` column and a near-constant ``IRX`` rate stand in for the
  cash / risk-free legs. Deterministic given ``seed``; no network.

* :func:`fetch` / :func:`load_prices` — the real daily closes for **SPY** (``auto_adjust=True``
  total return — the underlying path), **BIL** (1–3m T-bills, the cash leg *and* the
  excess-of-cash denominator) and **^IRX** (13-week T-bill discount rate, the Black–Scholes
  risk-free input), each cached to parquet under this study's OWN ``_cache/``. **Cache-only**
  unless ``fetch=True``: the offline core (and CI) never imports ``yfinance``. BIL lists
  **2007-05-30**, which bounds the joint window — a young-ETF single-cycle caveat named on the
  Signal axis (but it puts 2008, 2020 and 2022 in-sample, exactly where capital protection earns
  its keep).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ("SPY", "BIL", "^IRX")
RISKY_TICKER = "SPY"
CASH_TICKER = "BIL"
RATE_TICKER = "^IRX"
START = "2007-05-30"          # BIL inception — the binding joint-window start
AS_OF = "2026-06-30"          # last complete calendar month at publication
TRADING_DAYS = 252

__all__ = [
    "TICKERS", "RISKY_TICKER", "CASH_TICKER", "RATE_TICKER", "START", "AS_OF",
    "CACHE_DIR", "TRADING_DAYS", "GroundTruth", "synthetic_prices", "fetch",
    "have_real", "load_prices",
]


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, a trending SPY with optional up-jumps / crash drift
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the machinery."""
    drift: float              # constant daily SPY drift (negative -> a bear/crash tape)
    sigma: float              # daily SPY vol scale
    up_jump_prob: float       # per-day probability of a large UP jump (positive skew)
    up_jump_size: float       # size of the up jump (a positive number, added)
    cash_rate_ann: float      # annual cash / bill rate (BIL accrual & the IRX level)
    n_days: int

    @property
    def is_bear(self) -> bool:
        """True on a falling tape — where the 90/10 capital-protection floor should bite."""
        return self.drift < 0.0

    @property
    def has_convex_tape(self) -> bool:
        """True when the tape carries positive-skew up-jumps a convex call sleeve can monetize."""
        return self.up_jump_prob > 0.0 and self.up_jump_size > 0.0


def synthetic_prices(
    n_days: int = 6000,
    drift: float = 0.00030,
    sigma: float = 0.011,
    up_jump_prob: float = 0.0,
    up_jump_size: float = 0.0,
    cash_rate_ann: float = 0.03,
    start: str = "2007-05-30",
    seed: int = 899,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A geometric SPY with tunable ``drift`` / ``sigma``, plus optional up-jumps, cash & rate.

    SPY returns are ``r_t = drift + sigma · z_t`` with, when ``up_jump_prob > 0``, an extra
    ``+up_jump_size`` on a fraction of days (a **positively-skewed** tape where the convex call
    sleeve should out-capture a matched linear book). A **crash** control turns on ``drift < 0``
    (a bear tape): buy-and-hold craters while the 90% bills preserve capital and the call sleeve
    simply expires worthless. ``BIL`` accrues a near-constant daily bill rate; ``IRX`` is a
    near-constant annual rate (in percent) at that same level — the Black–Scholes risk-free input.

    Returns ``(frame, truth)`` with columns ``['SPY', 'BIL', 'IRX']`` (SPY/BIL are price levels
    from 100 so the same downstream code runs on synthetic and real; IRX is a *rate in percent*)
    on a business-day ``Date`` index; ``truth`` carries the baked-in parameters and the
    ``is_bear`` / ``has_convex_tape`` flags. Deterministic given ``seed``. ``n_days`` stays well
    below the pandas ns-timestamp horizon (bdate_range, n small).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="Date")

    z = rng.standard_normal(n_days)
    r_spy = drift + sigma * z
    if up_jump_prob > 0.0 and up_jump_size > 0.0:
        jump_days = rng.random(n_days) < up_jump_prob
        r_spy = np.where(jump_days, r_spy + up_jump_size, r_spy)

    cash_daily = cash_rate_ann / TRADING_DAYS
    r_cash = np.full(n_days, cash_daily)

    frame = pd.DataFrame(
        {
            "SPY": 100.0 * np.cumprod(1.0 + r_spy),
            "BIL": 100.0 * np.cumprod(1.0 + r_cash),
            "IRX": np.full(n_days, cash_rate_ann * 100.0),   # a rate in percent (^IRX convention)
        },
        index=idx,
    )
    truth = GroundTruth(
        drift=drift, sigma=sigma, up_jump_prob=up_jump_prob, up_jump_size=up_jump_size,
        cash_rate_ann=cash_rate_ann, n_days=n_days,
    )
    return frame, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! daily closes, cached per ticker
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"close_{safe}.parquet")


def _fetch_one(ticker: str, retries: int = 4) -> pd.Series:
    """Download one ticker's close, retrying transient network failures."""
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
    """Download each ticker's close and cache it to parquet (network path)."""
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
    """Cached real frame, inner-joined and sliced to ``[start, asof]`` — columns SPY / BIL / IRX.

    Reads the per-ticker parquet directly — **OFFLINE**, no yfinance import. SPY and BIL are
    total-return price levels; ^IRX is stored under column ``IRX`` as a *rate in percent* (the
    Black–Scholes risk-free input). The frame is the inner join across tickers (dropna) so it
    begins at the latest inception (BIL, 2007-05-30). Returns an empty frame if any ticker is
    uncached.
    """
    cols: dict[str, pd.Series] = {}
    for tk in tickers:
        path = _cache_path(tk)
        if not os.path.exists(path):
            return pd.DataFrame()
        s = pd.read_parquet(path)[tk]
        cols["IRX" if tk == RATE_TICKER else tk] = s
    frame = pd.concat(cols.values(), axis=1, keys=list(cols.keys()), join="inner").dropna()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    frame = frame[(frame.index >= lo) & (frame.index <= hi)]
    frame.index.name = "Date"
    return frame
