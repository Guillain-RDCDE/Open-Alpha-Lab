"""Data layer for Study 120 (Excess-CAPE-Yield).

Two tapes, one shape (a monthly macro panel):

- ``synthetic_world`` — a *deterministic, offline* generator.  A synthetic CAPE
  series and a matching real-bond-yield series drive an ECY (= 1/CAPE minus real
  yield) that predicts the forward 10-year excess return by ``predicts``; set
  ``predicts=0`` for the null world.  This pins the machinery entirely offline.

- ``load_shiller`` — Shiller's own monthly dataset, read from the repo-wide
  cache at ``_cache/shiller_sp500.parquet``.  **Never re-fetched live**; the
  parquet is a read-only staged input.  Returns a cleaned frame with ``ecy``,
  ``earnings_yield``, ``real_bond_yield``, ``fwd10_excess`` (10-year forward
  annualised real equity excess return) and ``fwd1_excess`` (1-year), with
  NaN rows properly dropped.

  ECY construction:
    - earnings yield   = 1 / CAPE  (= 1 / PE10)
    - real bond yield  = nominal 10-year yield − trailing 12-month CPI inflation
    - ECY              = earnings yield − real bond yield

  Forward excess return:
    - real equity total return reinvests dividends monthly via Real Price +
      Real Dividend columns
    - ``fwd10_excess`` = annualised 10-year real equity total return − real bond yield
      at the *start* of the holding period (Shiller's own definition)
    - ``fwd1_excess``  = 1-year real equity total return − real bond yield (start)

  The last ~120 months of the file have zero-placeholder Dividend/CPI/Earnings
  columns; these rows are dropped before any calculation.

No look-ahead: the signal at date *t* (ECY) uses only data available at *t*;
the forward return looks forward from *t* onward.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SHILLER_FILE = os.path.join(DEFAULT_CACHE, "shiller_sp500.parquet")


# ---------------------------------------------------------------------------
# Synthetic world — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 130,
    predicts: float = 1.0,
    cape_mean: float = 17.0,
    bond_mean: float = 0.02,
    seed: int = 120,
) -> tuple[pd.DataFrame, dict]:
    """A monthly ECY/forward-excess-return world — deterministic given ``seed``.

    CAPE mean-reverts around ``cape_mean`` (AR(1) with high persistence); the
    real bond yield oscillates around ``bond_mean`` with moderate variance.
    ECY = 1/CAPE − real_bond_yield; the forward 10-year excess return loads on
    ECY by ``predicts`` plus noise:

    - ``predicts = 1.0`` → a 1:1 linear relationship, close to empirical.
    - ``predicts = 0.0`` → the null (ECY has no predictive power).
    - ``predicts = 0.5`` → a weak partial relationship.

    Returns ``(frame, truth)`` where ``frame`` has columns ``cape``,
    ``earnings_yield``, ``real_bond_yield``, ``ecy``, ``fwd10_excess``,
    ``fwd1_excess`` and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1882-01-31", periods=n, freq="ME", name="date")

    # CAPE: high-persistence AR(1) around cape_mean
    cape = np.empty(n)
    cape[0] = cape_mean
    for t in range(1, n):
        cape[t] = cape_mean + 0.995 * (cape[t - 1] - cape_mean) + 0.8 * rng.standard_normal()
    cape = np.clip(cape, 5.0, 50.0)

    # Real bond yield: lower persistence, bounded
    bond = np.empty(n)
    bond[0] = bond_mean
    for t in range(1, n):
        bond[t] = bond_mean + 0.98 * (bond[t - 1] - bond_mean) + 0.003 * rng.standard_normal()
    bond = np.clip(bond, -0.05, 0.15)

    earnings_yield = 1.0 / cape
    ecy = earnings_yield - bond

    # Forward 10-year excess return loads on ECY + noise
    noise_10yr = 0.025 * rng.standard_normal(n)
    fwd10_excess = predicts * ecy + (1.0 - predicts) * (0.04 + noise_10yr) + noise_10yr * predicts * 0.2

    # Forward 1-year is much noisier (horizon shrinkage)
    noise_1yr = 0.15 * rng.standard_normal(n)
    fwd1_excess = predicts * 0.4 * ecy + noise_1yr

    truth = {
        "predicts": predicts,
        "cape_mean": cape_mean,
        "bond_mean": bond_mean,
        "n_years": n_years,
        "n_obs": n,
        "seed": seed,
    }
    frame = pd.DataFrame(
        {
            "cape": cape,
            "earnings_yield": earnings_yield,
            "real_bond_yield": bond,
            "ecy": ecy,
            "fwd10_excess": fwd10_excess,
            "fwd1_excess": fwd1_excess,
        },
        index=idx,
    )
    return frame, truth


# ---------------------------------------------------------------------------
# Real data — Shiller monthly panel, staged parquet
# ---------------------------------------------------------------------------
def load_shiller(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load and clean the Shiller S&P 500 monthly panel from the staged parquet.

    Reads ``_cache/shiller_sp500.parquet`` (already staged in the repo; never
    re-fetched). Drops rows where Dividend, Earnings, CPI or PE10 are zero or
    NaN (the last ~120 months have zero placeholders). Computes:

      * ``earnings_yield``   = 1 / PE10
      * ``cpi_yoy``          = trailing 12-month CPI inflation
      * ``real_bond_yield``  = Long Interest Rate / 100 − cpi_yoy
      * ``ecy``              = earnings_yield − real_bond_yield
      * real total return index ``real_tr``
      * ``fwd10_excess``     = annualised 10-year real equity total return minus
                               real_bond_yield at start (NaN for last 120 months)
      * ``fwd1_excess``      = 1-year real equity total return minus real_bond_yield
                               at start (NaN for last 12 months)

    Returns a DataFrame indexed by Date, sorted ascending, with no other NaN
    in the computed columns (rows with NaN dropped after forward-return shift).
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    raw = pd.read_parquet(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    df = raw.set_index("Date").sort_index()

    # Drop rows where required source columns are zero or missing
    mask = (
        (df["PE10"] > 0)
        & (df["Long Interest Rate"] > 0)
        & (df["Dividend"] > 0)
        & (df["Earnings"] > 0)
        & (df["Consumer Price Index"] > 0)
    )
    df = df[mask].copy()

    # Trailing 12-month CPI inflation (annualised)
    df["cpi_yoy"] = df["Consumer Price Index"].pct_change(12)

    # Earnings yield and real bond yield
    df["earnings_yield"] = 1.0 / df["PE10"]
    df["real_bond_yield"] = df["Long Interest Rate"] / 100.0 - df["cpi_yoy"]

    # Excess CAPE Yield (Shiller's definition)
    df["ecy"] = df["earnings_yield"] - df["real_bond_yield"]

    # Real total return index (monthly reinvestment of real dividend yield)
    monthly_ret = df["Real Price"].pct_change() + df["Real Dividend"] / df["Real Price"] / 12
    df["real_tr"] = (1.0 + monthly_ret.fillna(0.0)).cumprod()

    # Forward 10-year annualised real equity total return
    df["fwd10_eq"] = (df["real_tr"].shift(-120) / df["real_tr"]) ** (1.0 / 10.0) - 1.0

    # Forward 1-year real equity total return (simple, not annualised per Shiller)
    df["fwd1_eq"] = (df["real_tr"].shift(-12) / df["real_tr"]) - 1.0

    # Excess return = real equity return minus real bond yield at start of period
    df["fwd10_excess"] = df["fwd10_eq"] - df["real_bond_yield"]
    df["fwd1_excess"] = df["fwd1_eq"] - df["real_bond_yield"]

    # Keep only the core columns; drop rows where ecy or cpi_yoy is NaN
    cols = [
        "PE10",
        "Long Interest Rate",
        "earnings_yield",
        "cpi_yoy",
        "real_bond_yield",
        "ecy",
        "fwd10_excess",
        "fwd1_excess",
        "fwd10_eq",
        "fwd1_eq",
    ]
    out = df[cols].dropna(subset=["ecy", "cpi_yoy"])
    out.index.name = "date"
    return out


def fingerprint(df: pd.DataFrame, col: str = "ecy") -> str:
    """A short content fingerprint of the ECY series, for the as-of stamp."""
    arr = df[col].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
