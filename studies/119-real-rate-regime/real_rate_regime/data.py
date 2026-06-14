"""Data layer for Study 119 (Real-Rate-Regime).

Two tapes, one shape — a monthly panel with the real long-term interest rate
(Shiller long rate minus trailing 12-month CPI inflation) and the forward real
S&P 500 return:

- ``synthetic_world`` — **offline, deterministic**. A fake real-rate series with
  a tunable ``regime_effect`` knob: positive values bake in the "don't fight the
  Fed" prior (rising rates → lower equity returns), zero is the null. Fixed seed
  so tests never touch the network.
- ``fetch_shiller`` — loads the Shiller monthly dataset from the **already-staged
  parquet** at ``_cache/shiller_sp500.parquet`` (read-only; never refetched). Real
  rate = nominal long rate − trailing 12-month CPI; forward return computed from
  Shiller's real price index to side-step dividend reinvestment assumptions.

No look-ahead: the real rate and its 12-month change are formed on data through
month *t*; the forward return starts at month *t+1* (the shift is applied in
``fetch_shiller``; in ``strategy.py`` the timing overlay uses a lag=1 on the
signal so the signal at *t* drives the return at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# Name of the shared Shiller parquet (read-only; already staged in CI).
SHILLER_PARQUET = "shiller_sp500.parquet"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_months: int = 600,
    regime_effect: float = 0.0,
    rr_mean: float = 0.02,
    rr_ar1: float = 0.95,
    rr_vol: float = 0.015,
    equity_vol: float = 0.18,
    seed: int = 119,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly real-rate / equity-return world.

    The real rate follows a mean-reverting AR(1):
        rr_t = rr_mean + ar1*(rr_{t-1} - rr_mean) + eps, eps ~ N(0, rr_vol)

    The 12-month real rate change is computed from the level series. The forward
    12-month equity real return is driven by the 12-month change in real rates:
        fwd12_t = 0.04 + regime_effect * rr_chg_{t-12..t} + noise

    ``regime_effect = 0`` is the null (returns are unrelated to the rate regime).
    ``regime_effect < 0`` encodes the "don't fight the Fed" prior: rising real
    rates (positive rr_chg) precede lower equity returns. Returns ``(df, truth)``
    where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1900-01-31", periods=n_months, freq="ME", name="date")

    # Real rate — persistent AR(1) around a mean
    rr = np.empty(n_months)
    rr[0] = rr_mean
    for t in range(1, n_months):
        rr[t] = rr_mean + rr_ar1 * (rr[t - 1] - rr_mean) + rng.normal(0, rr_vol)

    # 12-month change in real rate (known at month t)
    rr_chg = np.full(n_months, np.nan)
    for t in range(12, n_months):
        rr_chg[t] = rr[t] - rr[t - 12]

    # Forward 12-month equity real return starting from t+1
    # (plant the signal; the study checks whether it exists in real data)
    equity_noise = rng.normal(0, equity_vol / np.sqrt(12), n_months)
    fwd12 = 0.04 + regime_effect * np.where(np.isnan(rr_chg), np.nan, rr_chg) + equity_noise

    df = pd.DataFrame(
        {"real_rate": rr, "real_rate_chg": rr_chg, "fwd12_real": fwd12},
        index=idx,
    )
    truth = {
        "regime_effect": regime_effect,
        "rr_mean": rr_mean,
        "rr_ar1": rr_ar1,
        "rr_vol": rr_vol,
        "equity_vol": equity_vol,
        "n_months": n_months,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller monthly dataset, already staged as parquet
# ---------------------------------------------------------------------------
def fetch_shiller(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 1873,
) -> pd.DataFrame:
    """Shiller monthly real rate and forward real equity return, from the staged parquet.

    Reads ``_cache/shiller_sp500.parquet`` (read-only; staged in CI; never
    re-downloaded). Computes:
        - ``cpi_inflation``  : trailing 12-month CPI inflation (month t−12 → t)
        - ``real_rate``      : Shiller long rate / 100 − cpi_inflation
        - ``real_rate_chg``  : 12-month change in the real rate (rr_t − rr_{t−12})
        - ``fwd12_real``     : forward 12-month real S&P total-price return
                               (Shiller real-price index, shift −12, no dividends
                               reinvested — a lower bound; consistent with the study's
                               "price only" framing, and comparable across the whole
                               1871–2026 history)

    Rows where CPI, Long Interest Rate, or Real Price are zero (the ~35 trailing
    months that Shiller has not yet filled in) are dropped. Rows near the end where
    the forward window is unavailable are also dropped via ``dropna``.

    Returns a DataFrame indexed by month-end dates with columns:
        ``real_rate, real_rate_chg, cpi_inflation, fwd12_real, long_rate, sp500_real``.
    """
    path = os.path.join(cache_dir, SHILLER_PARQUET)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shiller parquet not found at {path}. "
            "The file should be pre-staged in the repo's _cache/ folder."
        )

    raw = pd.read_parquet(path)
    # The parquet has a numeric RangeIndex and a 'Date' string column.
    raw["date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("date").sort_index()

    # Rename long columns for convenience
    raw = raw.rename(
        columns={
            "Long Interest Rate": "LongRate",
            "Consumer Price Index": "CPI",
        }
    )

    # Drop months with unfilled trailing zeros (Shiller's recent placeholder rows)
    raw = raw[(raw["CPI"] > 0) & (raw["LongRate"] > 0) & (raw["SP500"] > 0)].copy()

    # Trailing 12-month CPI inflation (rate of change, not level shift)
    raw["cpi_inflation"] = raw["CPI"].pct_change(12)

    # Real long rate: nominal rate − trailing CPI inflation
    raw["real_rate"] = raw["LongRate"] / 100.0 - raw["cpi_inflation"]

    # 12-month change in real rate (the "rising / falling regime" signal)
    raw["real_rate_chg"] = raw["real_rate"].diff(12)

    # Forward 12-month real price return (Shiller real-price index)
    rp = raw["Real Price"].replace(0, np.nan)
    raw["fwd12_real"] = rp.shift(-12) / rp - 1.0

    out = raw[["real_rate", "real_rate_chg", "cpi_inflation", "fwd12_real",
               "LongRate", "Real Price"]].rename(
        columns={"LongRate": "long_rate", "Real Price": "sp500_real"}
    )
    # Restrict to start_year and drop rows missing the core signals
    out = out[out.index.year >= start_year]
    return out.dropna(subset=["real_rate", "real_rate_chg"])


def fingerprint(df: pd.DataFrame) -> str:
    """Short SHA-1 fingerprint of the ``real_rate`` column — for the as-of stamp."""
    arr = df["real_rate"].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
