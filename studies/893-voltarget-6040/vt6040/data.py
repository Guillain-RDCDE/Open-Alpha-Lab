"""Data layer for Study 893 — Vol-Target 60/40.

The whole study reduces to a few daily **total-return** price columns and the folk claim it
tests: *run a volatility thermostat on the balanced 60/40 book — scale total exposure down when
realized portfolio vol spikes and up when it is calm, holding risk roughly constant — and you buy
a better excess-of-cash Sharpe and a shallower drawdown than the static 60/40, net of the extra
turnover.* This is the equities vol-target of [Study 16 (Storm-Shy)](../../16-storm-shy/) applied to
the diversified book of [Study 97 (Balancing-Act)](../../97-balancing-act/).

Two tapes, one shape (a daily frame of total-return price columns, one per asset):

* :func:`synthetic_prices` — fully **offline**, deterministic. A two-asset (stock/bond) world
  whose *common* volatility switches between a calm regime ``sigma_lo`` and a stormy ``sigma_hi``
  via a sticky Markov chain, so the **portfolio** variance clusters and is therefore forecastable.
  The drift is **regime-independent** on purpose: a storm carries the same expected return as a
  calm stretch but far more risk — the exact Moreira–Muir (2017) condition under which sizing by
  inverse realized vol raises the Sharpe. Pass ``sigma_lo == sigma_hi`` for the **flat-vol null**:
  no clustering, nothing for the thermostat to read, no gain. A near-constant positive ``CASH``
  column stands in for the risk-free leg. Deterministic given ``seed``; no network.

* :func:`fetch` / :func:`load_prices` — the real daily **total-return** closes (yfinance
  ``auto_adjust=True`` ⇒ dividends + splits folded in) for SPY (stocks), IEF (7–10y Treasuries),
  AGG (aggregate bonds, a robustness leg) and **BIL** (1–3m T-bills, the cash / excess-of-cash
  proxy), each cached to parquet under this study's OWN ``_cache/``. **Cache-only** unless
  ``fetch=True``: the offline core (and CI) never imports ``yfinance``. BIL lists **2007-05-30**,
  which bounds the joint window honestly — a young-ETF short-sample caveat we name on the Signal
  axis.

Daily total-return closes are the right input: vol-targeting is a daily position-sizing overlay in
the source literature, and total return is the fair series for a long-horizon risk-premium book.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ("SPY", "IEF", "AGG", "BIL")
CASH_TICKER = "BIL"
START = "2007-05-30"          # BIL inception — the binding joint-window start
AS_OF = "2026-06-30"          # last complete calendar month at publication
TRADING_DAYS = 252

__all__ = [
    "TICKERS", "CASH_TICKER", "START", "AS_OF", "CACHE_DIR", "TRADING_DAYS",
    "GroundTruth", "synthetic_prices", "fetch", "have_real", "load_prices",
]


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, portfolio-vol clustering with a regime-independent drift
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the machinery."""
    drift_stk: float          # constant daily stock drift, SAME in both vol regimes
    drift_bnd: float          # constant daily bond drift, SAME in both vol regimes
    sigma_lo: float           # calm-regime common vol scale
    sigma_hi: float           # stormy-regime common vol scale
    stay_calm: float          # P(calm -> calm)
    stay_storm: float         # P(storm -> storm)
    corr: float               # stock/bond shock correlation
    n_days: int

    @property
    def has_regime(self) -> bool:
        """True when there is a genuine vol regime to read (storm != calm)."""
        return self.sigma_hi != self.sigma_lo

    @property
    def calm_fraction(self) -> float:
        """Stationary share of time in the calm regime."""
        out_calm = 1.0 - self.stay_calm
        out_storm = 1.0 - self.stay_storm
        denom = out_calm + out_storm
        return float(out_storm / denom) if denom > 0 else 1.0

    @property
    def theoretical_sharpe_gain(self) -> float:
        """Perfect-foresight Sharpe multiple a constant-risk overlay earns over buy-&-hold.

        With a regime-independent drift and stationary regime weights, buy-&-hold Sharpe
        ≈ ``mu / sqrt(E[sigma^2])`` while a constant-risk overlay earns ≈ ``mu * E[1/sigma]``;
        the ratio ``E[1/sigma] * sqrt(E[sigma^2])`` is ≥ 1 by Cauchy–Schwarz, strictly > 1 when
        the regimes differ. The *ceiling* the realized, lagged, capped overlay approaches.
        """
        pc = self.calm_fraction
        sl, sh = self.sigma_lo, self.sigma_hi
        e_inv = pc * (1.0 / sl) + (1.0 - pc) * (1.0 / sh)
        e_var = pc * sl**2 + (1.0 - pc) * sh**2
        return float(e_inv * np.sqrt(e_var))


def synthetic_prices(
    n_days: int = 5000,
    drift_stk: float = 0.00060,
    drift_bnd: float = 0.00022,
    sigma_lo: float = 0.006,
    sigma_hi: float = 0.030,
    stay_calm: float = 0.985,
    stay_storm: float = 0.94,
    corr: float = -0.20,
    cash_rate_ann: float = 0.02,
    start: str = "2007-05-30",
    seed: int = 893,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A two-asset (STK/BND) total-return world whose **portfolio vol clusters** in regimes.

    A single sticky two-state Markov chain (calm / storm) scales the **common** shock size of both
    assets, so the *portfolio* realized variance is autocorrelated — the only thing the thermostat
    can forecast. Stock and bond shocks are correlated at ``corr`` (default −0.2, a mild hedge).
    Both drifts are **regime-independent**, so storms add risk without return — the setup where
    sizing by inverse realized vol lifts the Sharpe. Set ``sigma_lo == sigma_hi`` for the flat-vol
    null (no clustering ⇒ no gain). A ``CASH`` column accrues a near-constant daily bill rate.

    Returns ``(frame, truth)`` where ``frame`` has columns ``['SPY', 'IEF', 'BIL']`` (price levels
    from 100, so the same downstream code runs on synthetic and real) on a business-day ``Date``
    index; ``truth`` carries the baked-in parameters and the perfect-foresight Sharpe ceiling.
    Deterministic given ``seed``. ``n_days`` well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="Date")

    # Sticky two-state common-vol regime path: 0 = calm, 1 = storm.
    scale = np.empty(n_days)
    state = 0
    for t in range(n_days):
        stay = stay_calm if state == 0 else stay_storm
        if rng.random() > stay:
            state = 1 - state
        scale[t] = sigma_lo if state == 0 else sigma_hi

    c = float(np.clip(corr, -0.99, 0.99))
    chol = np.linalg.cholesky(np.array([[1.0, c], [c, 1.0]]))
    z = rng.standard_normal((n_days, 2)) @ chol.T
    # Common regime scale multiplies each asset's per-unit vol; bonds run ~40% of stock vol.
    r_stk = drift_stk + scale * z[:, 0]
    r_bnd = drift_bnd + 0.4 * scale * z[:, 1]

    cash_daily = cash_rate_ann / TRADING_DAYS
    r_cash = np.full(n_days, cash_daily)

    frame = pd.DataFrame(
        {
            "SPY": 100.0 * np.cumprod(1.0 + r_stk),
            "IEF": 100.0 * np.cumprod(1.0 + r_bnd),
            "BIL": 100.0 * np.cumprod(1.0 + r_cash),
        },
        index=idx,
    )
    truth = GroundTruth(
        drift_stk=drift_stk, drift_bnd=drift_bnd, sigma_lo=sigma_lo, sigma_hi=sigma_hi,
        stay_calm=stay_calm, stay_storm=stay_storm, corr=c, n_days=n_days,
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
    upper-case tickers; the frame is the inner join across tickers (dropna) so it begins at the
    latest inception (BIL, 2007-05-30). Returns an empty frame if any ticker is uncached.
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
