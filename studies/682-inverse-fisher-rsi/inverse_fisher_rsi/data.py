"""Data layer for Study 682 — Inverse-Fisher-RSI.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily total-return closes (dividend + split adjusted, ``auto_adjust=True``)
  for SPY plus a five-name liquid basket (QQQ, IWM, AAPL, MSFT, NVDA) from yfinance (no key),
  cached as CSV under the study's own ``_cache/``. No hardcoded calendar is needed — Ehlers'
  Inverse Fisher Transform of RSI is a purely algorithmic oscillator computed from price alone,
  not tied to a scheduled public event. Same universe as sibling
  [669-rsi-divergence](../../669-rsi-divergence/) for direct comparability.

* **Synthetic world.** A deterministic, seeded AR(1) log-return process with a TUNABLE
  mean-reversion knob ``rho``: ``rho = 0`` is a pure random walk (the null world — an RSI
  extreme carries zero forward-return information, the Welch machinery must NOT manufacture
  significance from it); ``rho > 0`` plants genuine short-horizon return anti-persistence
  (a real "buy the oversold dip" world) so the detector's power can be checked honestly.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# SPY + a liquid five-name basket — the same universe as sibling 669-rsi-divergence, so the
# two RSI-family studies on this desk are directly comparable.
TICKERS = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA")

START = "2010-01-04"     # 16+ full years, matches sibling technical-pattern studies
AS_OF = "2026-06-30"     # last complete calendar month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ifrsi_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2009-11-01", end: str = "2026-07-01") -> None:
    """Download daily total-return closes for every ticker in TICKERS; cache each as CSV.

    ``auto_adjust=True`` — dividend + split adjusted, so the oscillator and every forward
    return below are computed on a genuine total-return tape (not price-only). Network; once.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = pd.DataFrame({"Close": raw["Close"]}).dropna(how="all")
        df.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def load_real(start: str = START, asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return Close series}, each sliced to [start, asof]."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df.loc[(df.index >= start) & (df.index <= asof), "Close"]
        out[t] = s.copy()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted mean-reversion strength (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(rho: float = 0.0, seed: int = 682, n: int = 6500,
                     mu: float = 0.00025, sig: float = 0.011) -> pd.Series:
    """Deterministic AR(1) log-return price path with a TUNABLE mean-reversion knob ``rho``.

    Daily log return ``r_t = mu + rho * (mu - r_{t-1}) + eps_t``: ``rho = 0`` collapses to an
    i.i.d. random walk around drift ``mu`` (returns are NOT predictable from anything, including
    an RSI extreme — the null world). ``rho > 0`` makes returns genuinely anti-persistent
    (a big down day is, on average, partly reversed the next), so an oversold RSI/IFT-RSI
    extreme should carry real predictive content for the next few sessions — the faithful
    positive control for a mean-reversion oscillator claim.

    Business-day index of ``n`` bars (~25 years) — far below the ~250-year pandas
    ns-timestamp trap. Returns a total-return-style Close series (no dividends to model; the
    process already IS the total-return path).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    eps = rng.normal(0.0, sig, size=n)
    r = np.empty(n)
    r[0] = mu + eps[0]
    for t in range(1, n):
        r[t] = mu + rho * (mu - r[t - 1]) + eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.Series(close, index=idx, name="Close")
