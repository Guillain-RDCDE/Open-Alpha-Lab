"""Data layer for Study 152 (Inflation-Hedge).

Two tapes, one shape (a monthly macro panel):

- ``synthetic_world`` — a *deterministic, offline* generator.  A synthetic CPI
  series and nominal S&P 500 return series let us dial the Fisher relationship
  via ``fisher_beta``:

  - ``fisher_beta = 1.0`` → full Fisher pass-through: nominal returns rise
    one-for-one with inflation, real returns are invariant; the inflation hedge
    works perfectly.
  - ``fisher_beta = 0.0`` → no pass-through (the null): nominal returns have no
    relationship with inflation; real returns *fall* when inflation rises by the
    full inflation amount.  This matches the Fama-Schwert (1977) finding.
  - Values in between → partial hedge.

  This pins the machinery entirely offline and gives a clean positive/negative
  control for the strategy tests.

- ``load_shiller`` — Shiller's own monthly dataset, read from the repo-wide
  cache at ``_cache/shiller_sp500.parquet``.  **Never re-fetched live**; the
  parquet is a read-only staged input.  Returns a cleaned frame with:

    - ``cpi_yoy``         : trailing 12-month CPI inflation (annualised rate)
    - ``cpi_regime``      : "high" (cpi_yoy > regime_threshold) else "low"
    - ``nom_ret_12m``     : trailing 12-month nominal S&P 500 price return
    - ``real_ret_12m``    : trailing 12-month real S&P 500 price return
    - ``real_tr_ret_12m`` : trailing 12-month real *total* return (dividends
                            reinvested, using Real Price + Real Dividend)
    - ``fwd_real_tr_12m`` : forward 12-month real total return (the outcome
                            we want the inflation hedge to protect)
    - ``fwd_real_tr_60m`` : forward 60-month (5-year) annualised real total
                            return (longer horizon cross-check)

  No look-ahead: all signals use data up to the month of observation; forward
  returns look forward from there.

  The last ~6 months of the Shiller file have zero-placeholder Dividend, CPI,
  and Earnings values — these rows are dropped before any calculation.
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

# Inflation threshold: months with CPI YoY above this are the "high-inflation" regime.
# 3% is a natural break: it separates the 1970s-1980s episodes from "normal" post-WWII
# inflation.  We treat this as a pre-specified split, not data-mined.
DEFAULT_REGIME_THRESHOLD = 0.03   # 3% annualised CPI YoY


# ---------------------------------------------------------------------------
# Synthetic world — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 130,
    fisher_beta: float = 0.0,
    mean_inflation: float = 0.03,
    equity_real_mean: float = 0.07,
    equity_vol: float = 0.16,
    seed: int = 152,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly inflation/equity panel — deterministic given ``seed``.

    Monthly CPI inflation follows an AR(1) with persistence 0.95, mean
    ``mean_inflation/12``, moderate shocks.  Nominal equity returns are
    constructed as:

        nom_ret_m  =  real_component  +  fisher_beta * infl_m

    where ``real_component`` is drawn i.i.d. normal with mean
    ``equity_real_mean/12`` and std ``equity_vol/sqrt(12)``.  The
    ``fisher_beta`` knob controls pass-through:

    - ``fisher_beta = 1.0`` → full inflation hedge (real return invariant).
    - ``fisher_beta = 0.0`` → no hedge (Fama-Schwert null); real returns fall
      when inflation is high.

    Returns ``(frame, truth)`` where ``frame`` has monthly columns matching
    those returned by ``load_shiller`` (so tests and notebooks can use the same
    code path), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1882-01-31", periods=n, freq="ME", name="date")

    # Monthly CPI inflation: AR(1) around mean_inflation/12
    mu_m = mean_inflation / 12.0
    infl_m = np.empty(n)
    infl_m[0] = mu_m
    for t in range(1, n):
        infl_m[t] = mu_m + 0.95 * (infl_m[t - 1] - mu_m) + 0.004 * rng.standard_normal()
    infl_m = np.clip(infl_m, -0.01, 0.02)  # keep monthly inflation in plausible range

    # Nominal equity return: real component + Fisher pass-through on inflation
    real_comp = rng.normal(equity_real_mean / 12.0, equity_vol / 12.0**0.5, n)
    nom_ret_m = real_comp + fisher_beta * infl_m
    real_ret_m = nom_ret_m - infl_m

    # Cumulative levels
    sp500 = 100.0 * np.cumprod(1.0 + nom_ret_m)
    cpi = 100.0 * np.cumprod(1.0 + infl_m)

    # Trailing 12-month aggregates (annualised where appropriate)
    nom_price = pd.Series(sp500, index=idx)
    cpi_s = pd.Series(cpi, index=idx)

    cpi_yoy = cpi_s.pct_change(12)  # trailing 12m CPI growth
    nom_ret_12m = nom_price.pct_change(12)
    real_tr_12m = (nom_price / cpi_s).pct_change(12)  # simplified real (no divs in synth)

    # Forward returns (shift backward so date t carries what happens from t onward)
    fwd_real_12m = real_tr_12m.shift(-12)
    fwd_real_60m = (
        (nom_price / cpi_s).shift(-60) / (nom_price / cpi_s)
    ) ** (1.0 / 5.0) - 1.0

    cpi_regime = pd.Series(
        np.where(cpi_yoy > DEFAULT_REGIME_THRESHOLD, "high", "low"),
        index=idx,
        dtype=str,
    )

    frame = pd.DataFrame(
        {
            "cpi_yoy": cpi_yoy,
            "cpi_regime": cpi_regime,
            "nom_ret_12m": nom_ret_12m,
            "real_ret_12m": real_tr_12m,   # simplified synonym in synthetic world
            "real_tr_ret_12m": real_tr_12m,
            "fwd_real_tr_12m": fwd_real_12m,
            "fwd_real_tr_60m": fwd_real_60m,
        },
        index=idx,
    ).dropna(subset=["cpi_yoy", "nom_ret_12m"])

    truth = {
        "fisher_beta": fisher_beta,
        "mean_inflation": mean_inflation,
        "equity_real_mean": equity_real_mean,
        "equity_vol": equity_vol,
        "n_years": n_years,
        "n_obs": len(frame),
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real data — Shiller monthly panel, staged parquet
# ---------------------------------------------------------------------------
def load_shiller(
    cache_dir: str = DEFAULT_CACHE,
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
) -> pd.DataFrame:
    """Load and clean the Shiller S&P 500 monthly panel from the staged parquet.

    Reads ``_cache/shiller_sp500.parquet`` (never re-fetched; the file is a
    read-only staged input in the repo).  Drops rows where Dividend, CPI or
    Long Interest Rate are zero or NaN (the last ~6 months have zero
    placeholders in the Shiller file).  Computes:

      * ``cpi_yoy``         — trailing 12-month CPI YoY inflation (annualised,
                              from the raw CPI index column).
      * ``cpi_regime``      — "high" if ``cpi_yoy > regime_threshold``, else
                              "low"  (``regime_threshold`` default 3%).
      * ``nom_ret_12m``     — trailing 12-month *nominal* S&P 500 price return.
      * ``real_ret_12m``    — trailing 12-month real price return (SP500 deflated
                              by CPI).
      * ``real_tr_ret_12m`` — trailing 12-month real *total* return, reinvesting
                              the real dividend monthly via Real Price + Real Dividend
                              (Shiller's own total return construction).
      * ``fwd_real_tr_12m`` — forward 12-month real total return from month t.
      * ``fwd_real_tr_60m`` — forward 60-month annualised real total return.

    No look-ahead: all signal columns are formed from data available at the row's
    date; forward-return columns look forward from there.
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    raw = pd.read_parquet(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    df = raw.set_index("Date").sort_index()

    # Drop rows with zero/missing source values (last ~6 months of the Shiller file)
    mask = (
        (df["Consumer Price Index"] > 0)
        & (df["Dividend"] > 0)
        & (df["Long Interest Rate"] > 0)
        & (df["SP500"] > 0)
        & (df["Real Price"] > 0)
        & (df["Real Dividend"] > 0)
    )
    df = df[mask].copy()

    # Trailing 12-month CPI inflation (annualised pct change of CPI index)
    df["cpi_yoy"] = df["Consumer Price Index"].pct_change(12)

    # Inflation regime label (pre-specified at 3% threshold)
    df["cpi_regime"] = np.where(df["cpi_yoy"] > regime_threshold, "high", "low")

    # Nominal price return (trailing 12m)
    df["nom_ret_12m"] = df["SP500"].pct_change(12)

    # Real price return (SP500 deflated by CPI, trailing 12m)
    df["real_price"] = df["SP500"] / df["Consumer Price Index"]
    df["real_ret_12m"] = df["real_price"].pct_change(12)

    # Real total return index (Shiller method: reinvest real dividend monthly)
    # Monthly real total return = real_price pct_change + real_dividend_yield/12
    monthly_rp_ret = df["Real Price"].pct_change()
    monthly_div_yield = df["Real Dividend"] / df["Real Price"] / 12.0
    monthly_real_tr = monthly_rp_ret + monthly_div_yield
    df["real_tr_idx"] = (1.0 + monthly_real_tr.fillna(0.0)).cumprod()

    # Trailing 12-month real total return
    df["real_tr_ret_12m"] = df["real_tr_idx"].pct_change(12)

    # Forward 12-month and 60-month real total return (shift backward)
    df["fwd_real_tr_12m"] = df["real_tr_idx"].shift(-12) / df["real_tr_idx"] - 1.0
    df["fwd_real_tr_60m"] = (
        df["real_tr_idx"].shift(-60) / df["real_tr_idx"]
    ) ** (1.0 / 5.0) - 1.0

    cols = [
        "cpi_yoy",
        "cpi_regime",
        "nom_ret_12m",
        "real_ret_12m",
        "real_tr_ret_12m",
        "fwd_real_tr_12m",
        "fwd_real_tr_60m",
    ]
    out = df[cols].dropna(subset=["cpi_yoy", "nom_ret_12m", "real_tr_ret_12m"])
    out.index.name = "date"
    return out


def fingerprint(df: pd.DataFrame, col: str = "cpi_yoy") -> str:
    """A short content fingerprint of the CPI series, for the as-of stamp."""
    arr = df[col].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
