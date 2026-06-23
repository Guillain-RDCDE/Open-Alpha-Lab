"""Data layer for Study 373 — Dispersion-Trade (index vs single-name realized vol).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for SPY and a large basket of liquid US large-caps
  (yfinance, no key), cached under ``_cache/basket_prices.csv`` (a wide CSV indexed by date,
  one column per ticker plus ``SPY``). From the basket we build the **dispersion proxy**:
  rolling realized vol of each single name, the equal-weight average of those, the rolling
  realized vol of SPY (the index), and the gap ``avg_single_vol − index_vol``. True
  dispersion is an implied-vol / implied-correlation trade; implied vols are not free, so
  this realized-vol gap is an explicit, transparent **proxy** for the same object the desk
  trades.

* **Synthetic.** A deterministic, fixed-seed generator that produces a correlated basket
  with a *controllable* dispersion-carry edge planted by construction. It is the positive
  control: the engine must recover a planted carry, and — with the carry knob set to zero —
  must NOT manufacture significance out of an autocorrelated, fat-tailed vol-spread series.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")

# A transparent, fixed basket of large, long-listed US large-caps used as the single-name
# leg of the dispersion proxy. Chosen for long price history and sector spread; this is a
# PROXY for "the constituents of SPY", named as such everywhere. (Survivorship is
# acknowledged on the Signal axis: a fixed surviving-names basket is the vol-of-the-
# survivors, a mild bias toward calmer single names and so a *narrower* dispersion gap —
# i.e. the bias runs *against* the believers' free-lunch claim, if anything.)
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
]

ANNUAL = 252.0  # trading days per year (vol annualisation)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + the basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/basket_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = ["SPY"] + BASKET
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    # keep names present for >=80% of the window (long-listed survivors)
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.80]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def realized_vol(returns: pd.Series | pd.DataFrame, window: int = 21) -> pd.Series | pd.DataFrame:
    """Rolling annualised realized vol from daily simple returns (window trading days)."""
    return returns.rolling(window).std() * np.sqrt(ANNUAL)


def dispersion_proxy(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Build the dispersion proxy from basket + SPY prices.

    Returns a frame indexed by date with columns:
      * ``index_vol``   — rolling annualised realized vol of SPY (the index leg)
      * ``avg_vol``     — equal-weight average of single-name rolling realized vols
      * ``gap``         — ``avg_vol − index_vol`` (the dispersion gap; the proxy for the
                          single-vs-index volatility premium the trade harvests)
      * ``impl_corr``   — implied-by-realized correlation = ``(index_vol/avg_vol)**2``, the
                          single-factor approximation (lower ⇒ wider dispersion)
      * ``spy``         — SPY adjusted close (for the carry book)

    This is the realized-vol PROXY for the implied-vol / implied-correlation dispersion
    trade. Days before the window fills, or with fewer than 10 single names, are dropped.
    """
    cols = [c for c in prices.columns if c != "SPY"]
    basket = prices[cols]
    spy = prices["SPY"]

    rb = basket.pct_change()
    rs = spy.pct_change()

    single_vol = realized_vol(rb, window)           # one column per name
    index_vol = realized_vol(rs, window)            # series

    enough = single_vol.notna().sum(axis=1) >= 10
    avg_vol = single_vol.mean(axis=1).where(enough)
    gap = avg_vol - index_vol
    # single-factor implied correlation: sigma_index = sqrt(rho) * avg_single (approx, EW)
    ratio = (index_vol / avg_vol).clip(upper=1.0)
    impl_corr = (ratio ** 2)

    out = pd.DataFrame({
        "spy": spy,
        "index_vol": index_vol,
        "avg_vol": avg_vol,
        "gap": gap,
        "impl_corr": impl_corr,
    }).dropna()
    return out


def load_real(path: str = DEFAULT_CACHE, window: int = 21) -> pd.DataFrame:
    """Convenience: cached prices -> dispersion-proxy frame in one call."""
    return dispersion_proxy(load_prices(path), window=window)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_basket(n_days: int = 4_000, n_names: int = 40,
                     carry: float = 0.0, seed: int = 373,
                     base_corr: float = 0.35, sig_daily: float = 0.018,
                     window: int = 21) -> pd.DataFrame:
    """Deterministic correlated basket with a *planted* dispersion-carry edge.

    Builds ``n_names`` single-name daily return series as a one-factor model:
    ``r_i = beta_i * f + eps_i`` with a common factor ``f`` and idiosyncratic noise, so the
    realized pairwise correlation is controlled by ``base_corr``. The index is the
    equal-weight average of the single names (a clean, self-consistent index). Realized
    single-name vols therefore exceed the index vol by construction (subadditivity) — the
    mechanical dispersion gap.

    The ``carry`` knob plants a *tradable* edge **on top of** the mechanical gap: it scales
    an extra daily carry accrual to the long-single / short-index variance book (the P&L the
    trade is supposed to harvest above and beyond the identity). With ``carry = 0`` the book
    must earn nothing the null can't explain — the mechanical gap is real but, framed as a
    *trade* P&L, it must NOT cross t = 2 by itself once the carry is honestly zero. A large
    planted ``carry`` must light the test up. That is the whole positive control.

    Returns the same wide price frame shape as the real cache (``SPY`` + one col per name)
    plus a hidden ``_carry`` attr on the frame for the engine's planted-edge accrual.
    """
    rng = np.random.default_rng(seed)
    # common factor + idiosyncratic, calibrated so realized pairwise corr ~= base_corr
    rho = float(np.clip(base_corr, 0.01, 0.99))
    f = rng.normal(0, sig_daily, size=n_days)
    betas = np.sqrt(rho)                                 # equal loadings -> corr = rho
    idio_sig = sig_daily * np.sqrt(1.0 - rho)
    eps = rng.normal(0, idio_sig, size=(n_days, n_names))
    # mild cross-sectional spread in single-name vol (dispersion of vols)
    vol_mult = rng.uniform(0.7, 1.5, size=n_names)
    single_ret = (betas * f)[:, None] + eps * vol_mult[None, :]

    index_ret = single_ret.mean(axis=1)                 # equal-weight index

    # build prices (start at 100); index column named SPY for cache-shape parity
    single_px = 100.0 * np.exp(np.cumsum(single_ret, axis=0))
    index_px = 100.0 * np.exp(np.cumsum(index_ret))

    # period_range avoids OutOfBounds on long synthetic series; the date index is decorative
    idx = pd.period_range("2000-01-03", periods=n_days, freq="D").to_timestamp()
    data = {"SPY": index_px}
    for j in range(n_names):
        data[f"N{j:02d}"] = single_px[:, j]
    frame = pd.DataFrame(data, index=idx)
    # stash the planted carry so the strategy can add it to the realized book deterministically
    frame.attrs["_carry"] = float(carry)
    frame.attrs["_window"] = int(window)
    return frame
