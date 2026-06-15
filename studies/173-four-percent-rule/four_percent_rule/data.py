"""Data layer for Study 173 (Four-Percent-Rule).

Two tapes, one shape (annual real return series for equities and bonds):

- ``synthetic_cohorts`` — a *deterministic, offline* generator.  A mean-return
  knob (``equity_real_mean``) and a sequence-of-returns knob (``bad_start``) let
  us dial the two effects the 4% rule lives or dies on:

    - ``equity_real_mean = 0.068`` reproduces the long-run Shiller real return ~6.8%,
      the fertile ground that makes the rule work.
    - ``equity_real_mean = 0.03`` (low-yield world) stresses survivorship badly.
    - ``bad_start = True`` plants a severe bear market in years 1-3 to demonstrate
      sequence-of-returns risk even when the 30-year mean is unchanged.

- ``load_shiller`` — the real monthly Shiller dataset already staged at the repo's
  shared cache.  Returns annual real total-return series for stocks and 10-year
  bonds, so the 60/40 retirement engine has no network dependency.

No look-ahead is baked in here — the retirement engine in ``strategy.py`` only
uses returns that have been *realised* by the time a cohort starts.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SHILLER_CACHE = os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet")

# Long-run empirical reference values (Shiller 1871-2023, ~6.8% real equity).
LONG_RUN_EQUITY_REAL = 0.068
LONG_RUN_BOND_REAL = 0.016  # 10yr nominal minus avg ~3% inflation


# ---------------------------------------------------------------------------
# Synthetic generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_cohorts(
    start_year: int = 1926,
    n_years_total: int = 100,
    horizon: int = 30,
    equity_real_mean: float = LONG_RUN_EQUITY_REAL,
    equity_vol: float = 0.175,
    bond_real_mean: float = LONG_RUN_BOND_REAL,
    bond_vol: float = 0.065,
    equity_bond_corr: float = -0.05,
    bad_start: bool = False,
    seed: int = 173,
) -> tuple[pd.DataFrame, dict]:
    """Generate a reproducible run of annual real total-return pairs (equity, bond).

    Log-returns follow a bivariate normal with the correlation ``equity_bond_corr``.
    When ``bad_start=True``, the first three years of each 30-year window receive a
    large negative equity shock (-0.30 log-return excess) to demonstrate
    sequence-of-returns risk without changing the mean (the shock is offset later).

    Returns ``(returns_df, truth)`` where ``returns_df`` has columns
    ``['year', 'eq_real', 'bond_real']`` and ``truth`` records parameters.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years_total)
    n = len(years)

    # Bivariate normal correlated draws.
    cov = np.array([
        [equity_vol**2, equity_bond_corr * equity_vol * bond_vol],
        [equity_bond_corr * equity_vol * bond_vol, bond_vol**2],
    ])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 2))
    raw = z @ L.T
    log_eq = equity_real_mean - 0.5 * equity_vol**2 + raw[:, 0]
    log_bond = bond_real_mean - 0.5 * bond_vol**2 + raw[:, 1]

    if bad_start:
        # Sequence-of-returns stress: plant a bad first-3-year run, offset in yrs 4-6
        # so the 30-year CAGR is unchanged.  The offset is approximate but deterministic.
        shock = 0.30
        log_eq[:3] -= shock
        log_eq[3:6] += shock  # rebalance so arithmetic mean is preserved over 30 yrs

    eq_real = np.exp(log_eq) - 1.0
    bond_real = np.exp(log_bond) - 1.0

    df = pd.DataFrame({
        "year": years,
        "eq_real_tr": eq_real,
        "bond_real_tr": bond_real,
    }).set_index("year")

    truth = {
        "start_year": int(start_year),
        "n_years_total": int(n_years_total),
        "horizon": int(horizon),
        "equity_real_mean": float(equity_real_mean),
        "equity_vol": float(equity_vol),
        "bond_real_mean": float(bond_real_mean),
        "bond_vol": float(bond_vol),
        "equity_bond_corr": float(equity_bond_corr),
        "bad_start": bool(bad_start),
        "seed": int(seed),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data loader — Shiller annual real total returns
# ---------------------------------------------------------------------------
def load_shiller(path: str = SHILLER_CACHE) -> pd.DataFrame:
    """Load the Shiller dataset and return annual real total returns (equity + bond).

    Columns returned: ``['year', 'eq_real_tr', 'bond_real_tr']``.

    *Equity real total return* is computed as the monthly real price change plus the
    real dividend yield, compounded annually.  *Bond real return* is approximated by
    the annual change in the long-rate column (10-year US rate) using the duration
    formula, deflated by realised CPI.  Both are calendar-year totals; we drop the
    final year if it lacks a full set of 12 months.

    Sources:
      - Shiller (2000) *Irrational Exuberance*, online data supplement
        (http://www.econ.yale.edu/~shiller/data.htm).
      - Dividend yield from Shiller ``Dividend`` / ``SP500``.
      - Bond real return approximation: Pfau (2010) *An International Perspective
        on Safe Withdrawal Rates* uses identical Shiller inputs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shiller cache not found at {path}. "
            "Expected the repo-level _cache/shiller_sp500.parquet."
        )
    raw = pd.read_parquet(path)
    raw = raw[raw["SP500"] > 0].copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)

    # Monthly real total return for equity:
    #   price appreciation: Real Price(t) / Real Price(t-1) - 1
    #   dividend: Real Dividend(t) / 12 / Real Price(t-1)
    raw["real_div_monthly"] = raw["Real Dividend"] / 12.0
    raw["rp"] = raw["Real Price"]
    raw["eq_monthly_tr"] = (
        (raw["rp"] + raw["real_div_monthly"]) / raw["rp"].shift(1) - 1.0
    )

    # Monthly real bond return approximation.
    # Using modified duration ≈ 9 years for a 10yr bond:
    # r_bond ≈ coupon_income/12 - duration * d(yield)/12 - inflation
    # We approximate as: nominal coupon / 12 - 9 * delta_rate / 12 - cpi_inflation/12
    # CPI deflation already handled by using Real Price for equity; for bond, we note:
    # real_bond_nominal ≈ long_rate/1200 - 9 * d(rate)/12
    rate = raw["Long Interest Rate"] / 100.0
    raw["bond_monthly_nom"] = rate / 12.0 - 9.0 * rate.diff() / 12.0
    cpi = raw["Consumer Price Index"]
    raw["cpi_infl_monthly"] = cpi / cpi.shift(1) - 1.0
    raw["bond_monthly_real"] = raw["bond_monthly_nom"] - raw["cpi_infl_monthly"]

    raw["year"] = raw["Date"].dt.year
    raw = raw.dropna(subset=["eq_monthly_tr", "bond_monthly_real"])

    # Compound monthly returns within each year.
    def compound(s):
        return (1.0 + s).prod() - 1.0

    annual = (
        raw.groupby("year")
        .agg(
            eq_real_tr=("eq_monthly_tr", compound),
            bond_real_tr=("bond_monthly_real", compound),
            n_months=("eq_monthly_tr", "count"),
        )
        .reset_index()
    )

    # Keep only complete years (12 months).
    annual = annual[annual["n_months"] == 12].copy()
    annual = annual.drop(columns=["n_months"]).set_index("year")
    return annual


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the annual return frame, for as-of stamping."""
    arr = df.values.astype(float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
