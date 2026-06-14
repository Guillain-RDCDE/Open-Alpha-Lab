"""Data layer for Study 151 (Stocks-For-Long-Run).

Two tapes, one shape (a monthly macro panel of rolling real returns):

- ``synthetic_world`` — a *deterministic, offline* generator.  A synthetic
  real-equity return series and a matching real-bond-return series let us dial
  the *horizon* and the *equity premium* (``equity_premium`` knob). Set
  ``equity_premium=0`` for the null world where stocks have no edge over bonds.
  This is the study's null in a bottle, allowing tests to run entirely offline.

- ``load_shiller`` — the Shiller S&P 500 monthly dataset, read from the repo-wide
  cache at ``_cache/shiller_sp500.parquet`` (never re-fetched; it is a staged
  read-only input).  Computes:

    * ``real_tr``        — real total-return index (monthly real price + real
                          dividend reinvestment), normalised to 1.0 at start.
    * ``real_bond_ret``  — monthly real bond return proxy = Long Interest Rate / 12
                          (as a monthly rate) minus monthly CPI inflation.
    * ``roll_N_eq``      — rolling N-year *annualised* real equity total return.
    * ``roll_N_bd``      — rolling N-year *annualised* real bond proxy return.
    * ``roll_N_excess``  — equity minus bond annualised return over N years.

  Horizons computed: 1, 5, 10, 20, 30 years.

No look-ahead: every rolling window uses only data available at the *start* of
that window looking forward, so the result is purely out-of-the-window (the
forward N-year return known only in hindsight).
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

HORIZONS = (1, 5, 10, 20, 30)  # years


# ---------------------------------------------------------------------------
# Synthetic world — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 155,
    equity_premium: float = 0.05,
    equity_vol: float = 0.17,
    bond_return: float = 0.01,
    bond_vol: float = 0.04,
    seed: int = 151,
) -> tuple[pd.DataFrame, dict]:
    """A monthly real-equity + real-bond world — deterministic given ``seed``.

    Monthly log returns are drawn independently (i.i.d.) for equity and bond:
      - equity: mean = equity_premium/12 + bond_return/12, std = equity_vol/sqrt(12)
      - bond:   mean = bond_return/12,                    std = bond_vol/sqrt(12)

    Set ``equity_premium = 0`` for the null world (no equity advantage). Returns
    ``(frame, truth)`` where ``frame`` has columns ``real_eq_ret``, ``real_bd_ret``,
    ``real_tr_eq``, ``real_tr_bd`` and the rolling N-year annualised excess and
    component returns for each horizon in ``HORIZONS``; ``truth`` records parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1871-01-31", periods=n, freq="ME", name="date")

    monthly_eq_mu = (equity_premium + bond_return) / 12
    monthly_eq_sig = equity_vol / np.sqrt(12)
    monthly_bd_mu = bond_return / 12
    monthly_bd_sig = bond_vol / np.sqrt(12)

    eq_ret = rng.normal(monthly_eq_mu, monthly_eq_sig, n)
    bd_ret = rng.normal(monthly_bd_mu, monthly_bd_sig, n)

    # Total return indices
    tr_eq = np.cumprod(1.0 + eq_ret)
    tr_bd = np.cumprod(1.0 + bd_ret)

    frame = pd.DataFrame(
        {"real_eq_ret": eq_ret, "real_bd_ret": bd_ret, "real_tr_eq": tr_eq, "real_tr_bd": tr_bd},
        index=idx,
    )

    # Rolling forward N-year annualised returns
    for h in HORIZONS:
        months = h * 12
        # forward h-year equity annualised: (TR[t+months] / TR[t])^(1/h) - 1
        # Using shift(-months): the value at t is the forward return starting at t
        eq_fwd = (frame["real_tr_eq"].shift(-months) / frame["real_tr_eq"]) ** (1.0 / h) - 1.0
        bd_fwd = (frame["real_tr_bd"].shift(-months) / frame["real_tr_bd"]) ** (1.0 / h) - 1.0
        frame[f"roll_{h}y_eq"] = eq_fwd
        frame[f"roll_{h}y_bd"] = bd_fwd
        frame[f"roll_{h}y_excess"] = eq_fwd - bd_fwd

    truth = {
        "equity_premium": equity_premium,
        "bond_return": bond_return,
        "equity_vol": equity_vol,
        "bond_vol": bond_vol,
        "n_years": n_years,
        "n_obs": n,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real data — Shiller monthly panel, staged parquet
# ---------------------------------------------------------------------------
def load_shiller(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load and clean the Shiller S&P 500 monthly panel from the staged parquet.

    Reads ``_cache/shiller_sp500.parquet`` (already staged; never re-fetched).
    Drops rows where Dividend, CPI, Long Interest Rate, or Real Price are zero
    or NaN (the last ~30 months have zero placeholders). Computes:

      * ``real_eq_ret``   — monthly real total equity return (price change + real
                           dividend yield, monthly reinvestment)
      * ``real_tr``       — cumulative real total return index, normalised to 1.0
      * ``real_bd_ret``   — monthly real bond return proxy:
                           (Long Interest Rate / 100 / 12) − monthly CPI inflation
      * ``real_tr_bd``    — cumulative real bond return index
      * ``roll_{N}y_eq``  — forward N-year annualised real equity total return
      * ``roll_{N}y_bd``  — forward N-year annualised real bond proxy return
      * ``roll_{N}y_excess`` — equity minus bond, same horizon

    Returns a DataFrame indexed by Date, sorted ascending, NaN rows in the
    core columns dropped appropriately.
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    raw = pd.read_parquet(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    df = raw.set_index("Date").sort_index()

    # Drop rows where required source columns are zero or missing
    mask = (
        (df["Real Price"] > 0)
        & (df["Real Dividend"] > 0)
        & (df["Long Interest Rate"] > 0)
        & (df["Consumer Price Index"] > 0)
    )
    df = df[mask].copy()

    # Monthly real equity total return: price change + real dividend yield / 12
    df["real_eq_ret"] = (
        df["Real Price"].pct_change()
        + df["Real Dividend"] / df["Real Price"] / 12.0
    )
    # Fill the first NaN from pct_change with 0
    df["real_eq_ret"] = df["real_eq_ret"].fillna(0.0)

    # Cumulative real total return index (equity)
    df["real_tr"] = (1.0 + df["real_eq_ret"]).cumprod()

    # Monthly real bond return proxy:
    # Long Interest Rate is the 10-year nominal yield in % per annum.
    # Convert to monthly: yield_monthly = rate / 100 / 12
    # Monthly CPI inflation: CPI pct_change(1)  (monthly, not annualised)
    df["monthly_cpi"] = df["Consumer Price Index"].pct_change().fillna(0.0)
    df["real_bd_ret"] = df["Long Interest Rate"] / 100.0 / 12.0 - df["monthly_cpi"]

    # Cumulative real bond return index
    df["real_tr_bd"] = (1.0 + df["real_bd_ret"]).cumprod()

    # Forward N-year rolling annualised returns (shift(-months) gives the *forward* return)
    for h in HORIZONS:
        months = h * 12
        eq_fwd = (df["real_tr"].shift(-months) / df["real_tr"]) ** (1.0 / h) - 1.0
        bd_fwd = (df["real_tr_bd"].shift(-months) / df["real_tr_bd"]) ** (1.0 / h) - 1.0
        df[f"roll_{h}y_eq"] = eq_fwd
        df[f"roll_{h}y_bd"] = bd_fwd
        df[f"roll_{h}y_excess"] = eq_fwd - bd_fwd

    # Core columns
    cols = (
        ["real_eq_ret", "real_bd_ret", "real_tr", "real_tr_bd"]
        + [f"roll_{h}y_eq" for h in HORIZONS]
        + [f"roll_{h}y_bd" for h in HORIZONS]
        + [f"roll_{h}y_excess" for h in HORIZONS]
    )
    out = df[cols].copy()
    out.index.name = "date"
    return out


def fingerprint(df: pd.DataFrame, col: str | None = None) -> str:
    """A short content fingerprint of the real total return index, for the as-of stamp.

    Uses ``real_tr`` (from :func:`load_shiller`) if present, otherwise ``real_tr_eq``
    (from :func:`synthetic_world`).  Pass ``col`` explicitly to override.
    """
    if col is None:
        col = "real_tr" if "real_tr" in df.columns else "real_tr_eq"
    arr = df[col].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
