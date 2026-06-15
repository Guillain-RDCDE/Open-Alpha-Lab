"""Data layer for Study 172 (Hundred-Minus-Age).

Two tapes, one shape (a monthly real-return frame for equities and bonds):

- ``synthetic_lifecycle`` — a *deterministic, offline* generator.  Two assets
  (stocks, bonds) with tunable Sharpe / vol / correlation parameters.  A
  ``equity_premium`` knob lets us dial the equity risk premium up (so de-risking
  costs something) or to zero (so it is free insurance).  ``equity_premium=0``
  is the null world where the glidepath has no cost and no benefit.  This is the
  positive control that shows what the engine does when there *is* something to
  find.
- ``load_shiller`` — the real 150-year Shiller S&P 500 dataset (already staged at
  ``_cache/shiller_sp500.parquet``) plus a synthetic monthly bond-return series
  derived from the Shiller long interest rate column.  Joint window: 1881-01 to
  2023-06 (sufficient history for many 40-year lifecycle simulations).

No look-ahead is baked in here — age lookup and weight computation live in
``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SHILLER_PATH = os.path.join(DEFAULT_CACHE, "shiller_sp500.parquet")

# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_lifecycle(
    n_years: int = 40,
    equity_premium: float = 0.04,       # annualised real equity risk premium over bonds
    bond_real_yield: float = 0.015,     # annualised real return on bonds
    equity_vol_ann: float = 0.16,       # annual equity vol
    bond_vol_ann: float = 0.06,         # annual bond vol
    corr_eq_bond: float = -0.10,        # equity-bond correlation (mildly negative)
    seed: int = 172,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly two-asset (equity + bond) total-return tape.

    Log-returns are drawn from a bivariate normal calibrated to the supplied
    parameters.  The equity risk premium is the *only* tunable "signal" — the
    thing a 100-minus-age glidepath extracts (or forfeits) relative to buy-and-
    hold equities.

    - ``equity_premium = 0.0``  → equities and bonds have the same expected return;
      the glidepath costs nothing and gains nothing in expectation.
    - ``equity_premium > 0.0``  → equities earn more; a declining equity weight
      forfeits the premium for lower volatility.

    Returns ``(prices, truth)`` where ``prices`` has columns ``['EQ', 'BD']``
    (price levels starting at 100, in real terms) indexed by monthly dates, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * 12
    idx = pd.date_range("1970-01-01", periods=n_months, freq="MS", name="Date")

    mu_eq = (bond_real_yield + equity_premium) / 12.0
    mu_bd = bond_real_yield / 12.0
    sig_eq = equity_vol_ann / np.sqrt(12.0)
    sig_bd = bond_vol_ann / np.sqrt(12.0)

    cov = np.array([
        [sig_eq**2, corr_eq_bond * sig_eq * sig_bd],
        [corr_eq_bond * sig_eq * sig_bd, sig_bd**2],
    ])
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_months, 2)) @ chol.T

    ret = np.column_stack([
        mu_eq + z[:, 0],
        mu_bd + z[:, 1],
    ])
    prices = 100.0 * np.exp(np.cumsum(ret, axis=0))
    frame = pd.DataFrame(prices, index=idx, columns=["EQ", "BD"])

    truth = {
        "equity_premium": equity_premium,
        "bond_real_yield": bond_real_yield,
        "equity_vol_ann": equity_vol_ann,
        "bond_vol_ann": bond_vol_ann,
        "corr_eq_bond": corr_eq_bond,
        "n_years": n_years,
        "n_months": n_months,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller S&P 500 monthly (1871-2023), cache-first
# ---------------------------------------------------------------------------

def load_shiller(
    start: str = "1881-01-01",
    end: str | None = None,
    cache_path: str = SHILLER_PATH,
) -> pd.DataFrame:
    """Real monthly total-return frame from the Shiller S&P 500 dataset.

    Returns a DataFrame with columns ``['EQ', 'BD']`` (monthly real total
    returns, NOT price levels) indexed by month-start dates.

    Equity (``EQ``):
      Real price return (Shiller ``Real Price``) + monthly dividend yield
      (annualised ``Dividend`` / ``SP500`` / 12), shifted one period so the
      dividend received in month t is the yield *implied at* month t-1.

    Bond (``BD``):
      Approximated from the Shiller long-rate column (10-year yield): each
      month's bond return is the carry component (yield / 12) minus the
      duration-weighted capital gain from the yield change:
          BD_t ≈ y_{t-1}/12  −  D * Δy_t
      where D = 7 (approximate modified duration of a 10-year bond).
      This is a standard first-order approximation widely used in academic
      asset allocation research.

    Both series are trimmed to the overlap of valid real-price and long-rate
    data.  The joint window is 1881-01 to 2023-06 (after leading NaN removal).
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Shiller parquet not found at {cache_path}. "
            "The file should be pre-staged in _cache/shiller_sp500.parquet."
        )
    raw = pd.read_parquet(cache_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").set_index("Date")

    # Keep rows with real data (exclude zero/NaN placeholders at the tail)
    raw = raw[(raw["SP500"] > 0) & (raw["Real Price"] > 0) & (raw["Long Interest Rate"] > 0)]

    # Monthly equity real total return
    div_yield_monthly = raw["Dividend"] / raw["SP500"] / 12.0          # monthly cash yield
    real_price_ret = raw["Real Price"].pct_change()                     # real price change
    eq_ret = real_price_ret + div_yield_monthly.shift(1)               # no look-ahead on div

    # Monthly bond return (approximate, 10-year Treasuries, mod-dur ≈ 7)
    DURATION = 7.0
    y = raw["Long Interest Rate"] / 100.0  # convert % to decimal
    carry = y.shift(1) / 12.0             # carry earned during the month
    capital_gain = -DURATION * y.diff()   # price impact of yield change
    bd_ret = carry + capital_gain

    df = pd.DataFrame({"EQ": eq_ret, "BD": bd_ret})
    df = df.dropna()

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    df.index = df.index.to_period("M").to_timestamp("M")  # normalise to month-start
    df.index.name = "Date"
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a return frame, for the as-of stamp."""
    h = hashlib.sha1()
    for c in sorted(df.columns):
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(df[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
