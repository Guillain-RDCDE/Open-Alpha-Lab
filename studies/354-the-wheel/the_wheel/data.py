"""Data layer for Study 354 — The Wheel.

Two sources, both offline-friendly:

* **Real tape.** Daily SPY (total-return adjusted close) and ^VIX (the CBOE 30-day implied
  vol index) from yfinance, cached under ``_cache/the_wheel_spy_vix.parquet``. We resample to
  **month-end** observations — the Wheel's natural rebalance — keeping for each month the SPY
  level at the open of the month, the SPY level at the close, and the VIX at the open (the
  implied vol the option premium is priced from). VIX/100 is used as the Black-Scholes implied
  vol for the one-month ATM option — a **transparent, clearly-labelled model**, not a live
  option chain. This slightly *flatters* the seller (realised vol is usually below VIX, the
  variance risk premium), so any shortfall the Wheel shows is conservative.

* **Synthetic.** A deterministic geometric-Brownian-with-jumps world (fixed seed) used as the
  positive control: monthly log-returns drawn with a known sigma plus rare crash months. With
  the premium priced at the *same* sigma the path is generated from, the Wheel's behaviour is
  exactly the textbook short-vol payoff — it must keep the premium, cap the up months, and eat
  the crash months in full. The control runs with no network.

Pure numpy + pandas; the loader is cache-first and only touches yfinance when asked to fetch.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "the_wheel_spy_vix.parquet")
TICKERS = ["SPY", "^VIX"]
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(cache: str = DEFAULT_CACHE, do_fetch: bool = False) -> pd.DataFrame:
    """Daily ``SPY`` (adjusted close) and ``VIX`` (level) frame, cache-first.

    Returns an empty frame when neither cache nor ``do_fetch`` is available, so the offline
    notebooks fall back to the frozen headline numbers (exactly like the desk exemplar).
    """
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not do_fetch:
        return pd.DataFrame()
    import yfinance as yf

    px = yf.download(TICKERS, period="max", interval="1d",
                     auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    df = pd.DataFrame({"SPY": px["SPY"], "VIX": px["^VIX"]}).dropna()
    df.index.name = "date"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_parquet(cache)
    return df


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Fold the daily SPY/VIX frame into month-end Wheel observations.

    Each row is one month: ``spy_open`` (SPY at the first trading day of the month),
    ``spy_close`` (last trading day), ``vix_open`` (VIX/100 at the open — the Black-Scholes
    implied vol used to price that month's ATM option), and ``spy_ret`` (the realised SPY
    total return over the month). Index is the month-end date. Drops a trailing partial month
    only if the source data does (we use whatever closed bars are cached).
    """
    if df.empty:
        return df
    g = df.groupby(df.index.to_period("M"))
    rows = []
    for per, sub in g:
        sub = sub.sort_index()
        spy_open = float(sub["SPY"].iloc[0])
        spy_close = float(sub["SPY"].iloc[-1])
        vix_open = float(sub["VIX"].iloc[0]) / 100.0
        rows.append((per.to_timestamp("M"), spy_open, spy_close, vix_open,
                     spy_close / spy_open - 1.0))
    out = pd.DataFrame(rows, columns=["month_end", "spy_open", "spy_close",
                                      "vix_open", "spy_ret"]).set_index("month_end")
    return out


def load_real(cache: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached daily frame -> month-end Wheel observations."""
    return monthly(fetch(cache))


def have_real(cache: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(cache)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_months(n: int = 360, mu_ann: float = 0.07, sigma_ann: float = 0.16,
                     crash_prob: float = 0.0, crash_mean: float = -0.18,
                     vix_premium: float = 1.0, seed: int = 354) -> pd.DataFrame:
    """Deterministic month-end observations from a lognormal-with-jumps SPY world.

    Monthly log-returns are ``N(mu_ann/12 - sigma_m^2/2, sigma_m)`` with ``sigma_m =
    sigma_ann/sqrt(12)``; with probability ``crash_prob`` a month is replaced by a crash of
    mean ``crash_mean`` (the steamroller — set ``crash_prob = 0`` for the no-tail null).

    The ``vix_open`` column is the implied vol the premium is priced from: ``sigma_ann *
    vix_premium``. ``vix_premium = 1`` makes the option *fairly* priced at the world's own
    vol — the clean positive control where the Wheel is exactly a short straddle/strangle and
    its expected edge over buy-and-hold is ~0 before costs. ``vix_premium > 1`` mimics the
    real variance risk premium (VIX above realised vol), which pays the seller.

    Same column layout as :func:`monthly` so the strategy code is source-agnostic.
    """
    rng = np.random.default_rng(seed)
    sigma_m = sigma_ann / np.sqrt(MONTHS_PER_YEAR)
    drift = mu_ann / MONTHS_PER_YEAR - 0.5 * sigma_m ** 2
    logr = rng.normal(drift, sigma_m, size=n)
    if crash_prob > 0:
        crash = rng.random(n) < crash_prob
        crash_logr = np.log1p(crash_mean + 0.05 * rng.standard_normal(n))
        logr = np.where(crash, crash_logr, logr)
    ret = np.expm1(logr)
    spy_close_level = 100.0 * np.cumprod(1.0 + ret)
    spy_open_level = np.concatenate([[100.0], spy_close_level[:-1]])
    idx = pd.period_range("1996-01", periods=n, freq="M").to_timestamp("M")
    idx.name = "month_end"
    return pd.DataFrame({
        "spy_open": spy_open_level,
        "spy_close": spy_close_level,
        "vix_open": np.full(n, sigma_ann * vix_premium),
        "spy_ret": ret,
    }, index=idx)
