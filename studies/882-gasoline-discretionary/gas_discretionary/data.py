"""Data layer for Study 882 — Gas-Price → Discretionary (the "pump tax" rotation).

The claim under test: a spike in the **retail gasoline price** is a tax on the consumer's
wallet, so a *rising* gas price should forecast **underperformance of consumer-discretionary
(XLY) versus staples (XLP)** — the discretionary-minus-staples spread turns *negative* — and
a **tailwind for energy (XLE)**. The tradable reading is a monthly predictive regression

    r_spread[t+1] = alpha + beta * r_gas[t] + eps ,   beta expected < 0 ,

where ``r_spread`` is the discretionary-minus-staples (XLY − XLP) monthly return and
``r_gas`` is the trailing-month gasoline-price return. A parallel regression on the energy
tilt (XLE − SPY) expects ``beta > 0``.

Two tapes, one shape (a daily adjusted-close frame, resampled to month-end for the
regression).

* **Real tape — the gas price and four liquid ETFs.** Daily adjusted close for **RB=F**
  (front-month RBOB gasoline futures, the pump-price proxy), **XLY** (consumer
  discretionary), **XLP** (consumer staples), **XLE** (energy) and **SPY** (the market),
  pulled with yfinance (``auto_adjust=True``, total-return) back to RB=F's 2005 history —
  the binding constraint, the sector ETFs go back to 1998. The aligned five-column frame is
  cached as a single parquet under this study's OWN ``_cache/``. ``fetch()`` (network,
  retried) builds the cache once; ``load_series()`` / ``load_panel()`` read it back OFFLINE
  (no yfinance import). *(FRED weekly retail gasoline ``GASREGW`` is the textbook pump
  series and would be an ideal confirmatory tape, but FRED was unreachable from the build
  host — see ``docs/references.md`` — so RB=F, a real traded gasoline price, is the signal.
  RB=F and GASREGW co-move ~0.9 at the monthly frequency.)*

* **Synthetic world — the positive control.** A deterministic, seeded daily generator
  (``synthetic_panel``) with a TUNABLE knob ``edge``: the discretionary-minus-staples
  month-``t+1`` return is ``-edge * gas_ret[t] + noise`` (and the energy tilt loads
  ``+edge``), so ``edge > 0`` plants exactly the pump-tax rotation and ``edge = 0`` is the
  null (gas and the sector spreads are independent random walks, the regression must find
  nothing). Used only to prove the regression machinery is unbiased — never to support a
  real-tape stamp.

Monthly frequency is the whole point: the pump-tax rotation is a *slow-diffusion* monthly
signal on the **discretionary-vs-staples spread**, distinct from crude's calendar
seasonality (study 226), the lagged crude→aggregate-equity forecast (study 825), the
same-period oil↔equity co-movement (study 245) and gasoline's own RVP-driven seasonality
(study 639). No look-ahead is baked in here — the discipline lives in ``strategy.py`` (the
predictor is the trailing-month gas return, the target the *forward*-month spread return).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["RB=F", "XLY", "XLP", "XLE", "SPY"]
# Column-safe names for the cache (RB=F -> GAS) so the parquet round-trips cleanly.
COLMAP = {"RB=F": "GAS", "XLY": "XLY", "XLP": "XLP", "XLE": "XLE", "SPY": "SPY"}
COLS = ["GAS", "XLY", "XLP", "XLE", "SPY"]
START = "2005-01-01"        # RB=F (RBOB gasoline futures) history is the binding constraint
AS_OF = "2026-06-30"        # last complete calendar month at publication
CACHE_PATH = os.path.join(CACHE_DIR, "gas_disc_RBOB_XLY_XLP_XLE_SPY.parquet")

__all__ = [
    "TICKERS", "COLS", "COLMAP", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_series", "load_panel", "fingerprint",
    "synthetic_series", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily adjusted close, cached under the study's _cache/
# --------------------------------------------------------------------------- #
def fetch(start: str = START, retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """Download the gas + four-ETF daily adjusted close and cache the aligned frame.

    Network is touched only here. yfinance is flaky, so we retry up to ``retries`` times
    with a short ``pause`` between attempts before giving up. On success the five-column
    daily frame (index=date, columns ``GAS``, ``XLY``, ``XLP``, ``XLE``, ``SPY``) is
    written to ``CACHE_PATH`` as a parquet and returned.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(CACHE_DIR, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                TICKERS, start=start, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned an empty frame")
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:  # single ticker degenerate case
                close = raw[["Close"]].copy()
                close.columns = [TICKERS[0]]
            close = close[[c for c in TICKERS if c in close.columns]]
            if close.shape[1] < len(TICKERS):
                raise RuntimeError(f"missing tickers, got columns {list(close.columns)}")
            close = close.rename(columns=COLMAP)[COLS]
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            close.index.name = "date"
            close = close.sort_index().ffill(limit=3).dropna()
            if len(close) < 500:
                raise RuntimeError(f"too few aligned rows: {len(close)}")
            close.to_parquet(CACHE_PATH)
            return close
        except Exception as exc:  # noqa: BLE001 — retry any failure
            last_err = exc
            if attempt < retries:
                time.sleep(pause)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    """True iff the real cache parquet exists (offline-readable)."""
    return os.path.exists(CACHE_PATH)


def load_series(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily adjusted-close frame (columns ``GAS``, ``XLY``, ``XLP``, ``XLE``,
    ``SPY``), sliced to ``[, asof]``.

    Reads the parquet directly — OFFLINE, no yfinance import. Drops the partial current
    month by cutting at ``asof`` (the last complete calendar month at publication).
    """
    df = pd.read_parquet(CACHE_PATH)
    df.index = pd.DatetimeIndex(df.index)
    hi = pd.Timestamp(asof)
    return df[df.index <= hi].sort_index()


def load_panel(asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached tape as ``{name: DataFrame[Close]}`` — mirrors the house panel shape."""
    df = load_series(asof)
    return {c: df[[c]].rename(columns={c: "Close"}) for c in COLS if c in df.columns}


def fingerprint(df: pd.DataFrame, col: str = "XLY") -> str:
    """Short content fingerprint of the panel (``col`` column) for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted pump-tax rotation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_series(
    edge: float = 0.0,
    seed: int = 882,
    n_days: int = 5292,          # ~21 years of business days (span < ns horizon)
    start: str = "2005-01-03",
    gas_annual_vol: float = 0.36,
    sec_annual_vol: float = 0.16,
    spread_annual_vol: float = 0.09,
    mkt_drift: float = 0.07 / 252,
) -> pd.DataFrame:
    """Deterministic seeded daily gas + four-ETF tape with a TUNABLE planted rotation.

    Gasoline is a pure random walk. The **discretionary-minus-staples** spread carries a
    small drag proportional to *last month's* gas return: on each day ``t`` we carry the
    trailing 21-day gas return ``m`` and add ``-edge * m / 21`` to the spread's daily
    return, so the *forward monthly* XLY−XLP spread loads on the *trailing monthly* gas
    return with slope ``≈ -edge`` — exactly the pump-tax pattern. The **energy** tilt
    (XLE − SPY) loads the opposite way (``+edge``, a tailwind). ``edge = 0`` is the null:
    gas and the sector spreads are independent random walks, and the predictive regression
    must find nothing. Business-day index; span well below the ns-timestamp horizon.

    Construction guarantees the identities used by the strategy: ``XLY − XLP`` monthly
    return equals the planted spread, and ``XLE − SPY`` equals the planted energy tilt.
    """
    rng = np.random.default_rng(seed)
    dv_gas = gas_annual_vol / np.sqrt(252)
    dv_sec = sec_annual_vol / np.sqrt(252)
    dv_sp = spread_annual_vol / np.sqrt(252)

    gas_r = rng.normal(0.0, dv_gas, n_days)

    # Trailing 21-day gas return, known at t-1, drives the day-t spread/energy returns.
    cum = np.concatenate([[0.0], np.cumsum(gas_r)])
    trail = np.zeros(n_days)
    trail[21:] = cum[21:n_days] - cum[:n_days - 21]      # gas log-return over (t-21, t-1]

    # Staples = market-like leg; discretionary = staples + planted spread.
    xlp_r = mkt_drift + rng.normal(0.0, dv_sec, n_days)
    spread_r = rng.normal(0.0, dv_sp, n_days) - edge * trail / 21.0
    xly_r = xlp_r + spread_r

    # SPY = market; energy = market + planted (positive) tilt on the same gas trail.
    spy_r = mkt_drift + rng.normal(0.0, dv_sec, n_days)
    enr_r = rng.normal(0.0, dv_sp, n_days) + edge * trail / 21.0
    xle_r = spy_r + enr_r

    idx = pd.bdate_range(start=start, periods=n_days)
    gas_p = 1.50 * np.exp(np.cumsum(gas_r))
    xlp_p = 40.0 * np.exp(np.cumsum(xlp_r))
    xly_p = 40.0 * np.exp(np.cumsum(xly_r))
    spy_p = 120.0 * np.exp(np.cumsum(spy_r))
    xle_p = 45.0 * np.exp(np.cumsum(xle_r))
    return pd.DataFrame(
        {"GAS": gas_p, "XLY": xly_p, "XLP": xlp_p, "XLE": xle_p, "SPY": spy_p},
        index=pd.DatetimeIndex(idx, name="date"),
    )


def synthetic_panel(edge: float = 0.0, seed: int = 882, **kw) -> dict[str, pd.DataFrame]:
    """``synthetic_series`` reshaped to ``{name: DataFrame[Close]}`` (house panel shape)."""
    df = synthetic_series(edge=edge, seed=seed, **kw)
    return {c: df[[c]].rename(columns={c: "Close"}) for c in COLS}
