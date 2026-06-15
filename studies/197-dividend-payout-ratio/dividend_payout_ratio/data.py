"""Data layer for Study 197 (Dividend-Payout-Ratio).

Two tapes, one shape (a monthly panel of payout ratio, forward earnings growth, and
forward real price return):

- ``synthetic_monthly`` — a *deterministic, offline* generator. A ``payout_signal``
  knob injects a linear relationship between the payout ratio and the forward outcome;
  ``payout_signal=0`` is a pure null (no predictability). This is the study's null in
  a bottle — the engine should detect the slope only when it is planted.
- ``load_shiller`` — the real Shiller monthly S&P 500 dataset, read from the shared
  parquet at ``_cache/shiller_sp500.parquet``. Cache-only (no network needed): the
  parquet is already staged in the repo. Returns a DataFrame with columns
  ``Payout``, ``fwd_10y_eg``, ``fwd_10y_price`` plus the raw Shiller fields.

No look-ahead: the payout ratio is built from current-period Dividend and Earnings;
the forward outcomes look 10 years ahead from the *same* observation date — the
same horizon used by Arnott & Asness (2003).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SHILLER_CACHE = os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet")

# Default horizon for the forward outcome (months).
HORIZON = 120  # 10 years


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_years: int = 100,
    payout_signal: float = 0.0,
    payout_mean: float = 0.55,
    payout_vol: float = 0.15,
    outcome_vol: float = 0.025,
    horizon: int = HORIZON,
    seed: int = 197,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly tape with an optional payout-ratio → forward-outcome slope.

    Each month *t* draws a payout ratio from N(``payout_mean``, ``payout_vol``^2). The
    10-year-ahead outcome (real earnings growth proxy) is:

        outcome_t = payout_signal * (payout_t - payout_mean) + eps_t,  eps ~ N(0, outcome_vol)

    so ``payout_signal = 0`` (the null) gives pure noise — the payout carries no
    information — while a positive ``payout_signal`` plants the Arnott-Asness
    relationship. Returns a DataFrame with columns ``payout``, ``fwd_outcome``,
    and ``truth``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1871-01-01", periods=n + horizon, freq="MS", name="date")

    # Generate payout ratio for the full span (n + horizon months)
    payout_full = payout_mean + payout_vol * rng.standard_normal(n + horizon)
    payout_full = np.clip(payout_full, 0.05, 2.0)  # keep economically plausible

    # Forward outcome at t is determined by payout at t (no horizon shift in the signal)
    # i.e., outcome[t+horizon] = payout_signal*(payout[t] - payout_mean) + noise[t]
    # So for observations 0..n-1 we need: outcome_obs = signal*(payout[:n]-mean) + noise
    noise = outcome_vol * rng.standard_normal(n)
    fwd_outcome = payout_signal * (payout_full[:n] - payout_mean) + noise

    df = pd.DataFrame(
        {
            "payout": payout_full[:n],
            "fwd_outcome": fwd_outcome,
        },
        index=idx[:n],
    )
    truth = {
        "payout_signal": payout_signal,
        "payout_mean": payout_mean,
        "payout_vol": payout_vol,
        "outcome_vol": outcome_vol,
        "horizon": horizon,
        "n_years": n_years,
        "n_obs": n,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller monthly dataset, cache-only
# ---------------------------------------------------------------------------
def load_shiller(
    cache_path: str = SHILLER_CACHE,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Load and prepare the Shiller monthly dataset for this study.

    Reads the staged parquet at ``cache_path`` (never fetches from the network).
    Computes:

    - ``Payout`` — 12-month rolling sum of Dividend / 12-month rolling sum of
      Earnings. Both Dividend and Earnings in the Shiller dataset are monthly
      per-share values; a 12-month rolling sum annualises both before dividing,
      yielding a true annual payout ratio. Months where either is zero or NaN
      are excluded.
    - ``fwd_10y_eg`` — annualised forward 10-year real earnings growth:
      ``(RealEarnings[t+120] / RealEarnings[t])^(1/10) - 1``.
    - ``fwd_10y_price`` — annualised forward 10-year real price return:
      ``(RealPrice[t+120] / RealPrice[t])^(1/10) - 1``.

    Returns a DataFrame indexed by date, dropping rows where any of the three
    signal or outcome columns are NaN.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {cache_path}. "
            "The staged parquet must be present at _cache/shiller_sp500.parquet."
        )
    raw = pd.read_parquet(cache_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()

    # Keep only rows with positive Dividend and Earnings
    valid = raw[(raw["Dividend"] > 0) & (raw["Earnings"] > 0)].copy()

    # Annual payout ratio: rolling 12-month sums for both numerator and denominator
    valid["Payout"] = (
        valid["Dividend"].rolling(12).sum() / valid["Earnings"].rolling(12).sum()
    )

    # Forward 10-year real earnings growth (annualised)
    valid["fwd_10y_eg"] = (
        valid["Real Earnings"].shift(-horizon) / valid["Real Earnings"]
    ) ** (1.0 / 10.0) - 1.0

    # Forward 10-year real price return (annualised)
    valid["fwd_10y_price"] = (
        valid["Real Price"].shift(-horizon) / valid["Real Price"]
    ) ** (1.0 / 10.0) - 1.0

    # Drop rows missing any of the three key columns
    out = valid.dropna(subset=["Payout", "fwd_10y_eg", "fwd_10y_price"])
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the Payout column, for the as-of stamp."""
    arr = df["Payout"].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
