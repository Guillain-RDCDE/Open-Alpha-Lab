"""Data layer for Study 118 (Fed-Model).

Two tapes, one shape (a monthly macro panel indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``predicts``
  knob controls how strongly the earnings-yield-minus-rate spread forecasts forward
  10-year real returns. ``predicts=0`` is the null — the spread carries zero
  information — so tests can assert that the regression finds a slope *only* when we
  plant a real effect, and not otherwise.
- ``load_shiller`` — reads the pre-staged Shiller S&P 500 dataset from the repo's
  shared cache (``_cache/shiller_sp500.parquet``). Never touches the network; the
  cache is a read-only input. Returns a DataFrame with columns:
    EP              — trailing earnings yield (Earnings/SP500)
    rate            — 10y nominal yield (Long Interest Rate / 100)
    fed_spread      — EP minus rate (the classic Fed Model gap)
    infl            — trailing 12m CPI inflation
    real_rate       — rate minus infl
    ep_vs_real      — EP minus real_rate (Asness's corrected comparison)
    fwd1            — annualised forward 1yr real total return
    fwd10           — annualised forward 10yr real total return (geometric)
    unconditional1  — expanding-window unconditional mean of fwd1 (the baseline)
    unconditional10 — expanding-window unconditional mean of fwd10
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SHILLER_PATH = os.path.join(DEFAULT_CACHE, "shiller_sp500.parquet")


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 1_200,
    predicts: float = 0.9,
    ep_mean: float = 0.065,
    rate_mean: float = 0.045,
    noise_scale: float = 0.04,
    start: str = "1900-01-31",
    seed: int = 118,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly macro panel with a known amount of Fed-Model predictability.

    The earnings yield (``EP``) follows a stationary AR(1) around ``ep_mean``. The 10y
    yield (``rate``) follows a separate AR(1) around ``rate_mean``. The Fed-Model spread
    is ``fed_spread = EP - rate``. The forward 10-year real return is a linear function
    of ``fed_spread`` scaled by ``predicts`` plus noise — so ``predicts=0`` is a pure
    null where no regression should find a slope, and ``predicts>0`` plants a real effect.

    Returns ``(panel, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_months, freq="ME", name="date")

    # AR(1) earnings yield
    ep = np.empty(n_months)
    ep[0] = ep_mean
    for t in range(1, n_months):
        ep[t] = ep_mean + 0.98 * (ep[t - 1] - ep_mean) + 0.004 * rng.standard_normal()
    ep = np.clip(ep, 0.02, 0.20)

    # AR(1) nominal 10y yield
    rate = np.empty(n_months)
    rate[0] = rate_mean
    for t in range(1, n_months):
        rate[t] = rate_mean + 0.98 * (rate[t - 1] - rate_mean) + 0.003 * rng.standard_normal()
    rate = np.clip(rate, 0.005, 0.20)

    fed_spread = ep - rate

    # Forward 10yr real return: linear in spread + noise
    noise = rng.standard_normal(n_months) * noise_scale
    fwd10 = 0.04 + predicts * fed_spread + noise  # unconditional ~4% real + signal + noise

    # Forward 1yr (much noisier, weaker signal)
    noise1 = rng.standard_normal(n_months) * 0.20
    fwd1 = 0.05 + predicts * 0.3 * fed_spread + noise1

    # Expanding-window unconditional means (baseline the strategy must beat)
    uncond10 = pd.Series(fwd10, index=idx).expanding(min_periods=60).mean()
    uncond1 = pd.Series(fwd1, index=idx).expanding(min_periods=60).mean()

    panel = pd.DataFrame(
        {
            "EP": ep,
            "rate": rate,
            "fed_spread": fed_spread,
            "infl": 0.03 * np.ones(n_months),  # fixed for synthetic
            "real_rate": rate - 0.03,
            "ep_vs_real": ep - (rate - 0.03),
            "fwd1": fwd1,
            "fwd10": fwd10,
            "unconditional1": uncond1.values,
            "unconditional10": uncond10.values,
        },
        index=idx,
    )
    truth = {
        "predicts": predicts,
        "ep_mean": ep_mean,
        "rate_mean": rate_mean,
        "n_months": n_months,
        "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller monthly, from shared cache
# ---------------------------------------------------------------------------
def load_shiller(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 1871,
) -> pd.DataFrame:
    """Load and derive the Fed-Model panel from the staged Shiller dataset.

    Reads ``_cache/shiller_sp500.parquet`` (never fetches from the network). Drops
    rows where Earnings, the Long Interest Rate, or SP500 are zero or NaN — these are
    the recent placeholder rows with no fundamental data. Returns a frame indexed by
    monthly date with the columns described in the module docstring.

    The forward return columns (``fwd1``, ``fwd10``) are NaN for the last 12 / 120
    months respectively because the target window extends beyond the available price
    data — callers must ``dropna`` before fitting regressions.
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shiller cache not found at {path}. "
            "The shared _cache/shiller_sp500.parquet must be present (read-only input)."
        )
    raw = pd.read_parquet(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()

    # Drop rows with zero/placeholder fundamentals
    mask = (raw["Earnings"] > 0) & (raw["Long Interest Rate"] > 0) & (raw["SP500"] > 0)
    df = raw[mask].copy()
    df = df[df.index.year >= start_year]

    # Core signals
    df["EP"] = df["Earnings"] / df["SP500"]
    df["rate"] = df["Long Interest Rate"] / 100.0
    df["fed_spread"] = df["EP"] - df["rate"]

    # Asness (2003) critique: compare E/P to the REAL rate
    df["infl"] = df["Consumer Price Index"].pct_change(12)  # trailing 12m CPI inflation
    df["real_rate"] = df["rate"] - df["infl"]
    df["ep_vs_real"] = df["EP"] - df["real_rate"]

    # Cumulative real total return index
    rp = df["Real Price"]
    rd = df["Real Dividend"]
    monthly_rtr = rp.pct_change() + rd / rp.shift(1) / 12.0
    monthly_rtr = monthly_rtr.fillna(0.0)
    cum_tr = (1.0 + monthly_rtr).cumprod()

    # Forward returns (annualised geometric)
    df["fwd1"] = (cum_tr.shift(-12) / cum_tr) - 1.0
    df["fwd10"] = (cum_tr.shift(-120) / cum_tr) ** (1.0 / 10.0) - 1.0

    # Expanding-window unconditional mean (the honest baseline — uses only past data)
    df["unconditional1"] = df["fwd1"].expanding(min_periods=60).mean()
    df["unconditional10"] = df["fwd10"].expanding(min_periods=60).mean()

    return df[
        [
            "EP", "rate", "fed_spread",
            "infl", "real_rate", "ep_vs_real",
            "fwd1", "fwd10",
            "unconditional1", "unconditional10",
        ]
    ]


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the panel's EP column, for the as-of stamp."""
    arr = panel["EP"].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
