"""Critique layer (SOCLE 1) — all OFFLINE, no network.

These functions exist to keep three things separate, which the original
pamphlet tends to blur:

  1. The empirical fact (overnight >> intraday) is REAL.
  2. The *magnitudes* are inflated by compounding, data artefacts and selection.
  3. Attribution to an orchestrated fraud is NOT proven.

Here we demonstrate (2) mechanically so a reader can see *why* a tiny, totally
innocent night/day bias produces "billions of percent" on a log chart, and how
a single mis-adjusted close manufactures a fake overnight signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import decompose, TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Synthetic market generator (the honest "clone" of Knuteson's pattern)
# ---------------------------------------------------------------------------
def synthetic_ohlc(
    n_days: int = 252 * 32,
    overnight_bias_bps: float = 3.0,
    intraday_bias_bps: float = -1.0,
    daily_vol_bps: float = 100.0,
    night_share_of_vol: float = 0.5,
    start_price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate OHLC for a market with a *constant, innocent* night/day drift.

    No manipulation, no agent, no fraud — just two Gaussian legs per day with a
    small mean difference. Decomposing it reproduces the Knuteson shape:
    overnight curve soars, intraday curve sags. That's the point.

    Parameters are in basis points (1 bps = 0.01%). ``night_share_of_vol``
    splits the daily noise between the overnight and intraday legs.
    """
    rng = np.random.default_rng(seed)
    mu_on = overnight_bias_bps * 1e-4
    mu_id = intraday_bias_bps * 1e-4
    sd = daily_vol_bps * 1e-4
    sd_on = sd * np.sqrt(night_share_of_vol)
    sd_id = sd * np.sqrt(1.0 - night_share_of_vol)

    r_on = rng.normal(mu_on, sd_on, n_days)
    r_id = rng.normal(mu_id, sd_id, n_days)

    dates = pd.bdate_range("1992-01-01", periods=n_days)
    close = np.empty(n_days)
    open_ = np.empty(n_days)
    prev_close = start_price
    for t in range(n_days):
        open_[t] = prev_close * (1.0 + r_on[t])
        close[t] = open_[t] * (1.0 + r_id[t])
        prev_close = close[t]

    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# (A) Compounding: why a minuscule bias becomes "billions of percent"
# ---------------------------------------------------------------------------
def compounding_table(
    bias_bps=(1, 5, 10, 30),
    horizons_years=(10, 20, 32),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Cumulative return of a *constant* overnight bias, by horizon.

    Pure arithmetic, fully deterministic: (1 + bias)^(periods*years) - 1.
    This is the single most important "reality check" on the headline numbers —
    the explosion comes from the exponent, not from anything sinister.
    Values are returns as a fraction (multiply by 100 for percent).
    """
    rows = {}
    for b in bias_bps:
        r = b * 1e-4
        rows[f"{b} bps/night"] = {
            f"~{y}y": (1.0 + r) ** (periods_per_year * y) - 1.0 for y in horizons_years
        }
    return pd.DataFrame(rows).T


def format_compounding(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable percent strings (handles the absurdly large cells)."""

    def fmt(x):
        pct = x * 100.0
        if pct < 1e6:
            return f"+{pct:,.0f}%"
        if pct < 1e9:
            return f"+{pct/1e6:,.0f}M%"
        if pct < 1e12:
            return f"+{pct/1e9:,.0f}bn%"
        return f"+{pct/1e12:,.1f}tn%"

    # pandas >=2.1: DataFrame.map (applymap was removed). Fallback for older.
    return table.map(fmt) if hasattr(table, "map") else table.applymap(fmt)


# ---------------------------------------------------------------------------
# (B) Data artefacts: a mis-adjusted close fabricates an overnight signal
# ---------------------------------------------------------------------------
def inject_split_artifact(
    ohlc: pd.DataFrame,
    rows=(0.25, 0.5, 0.75),
    factor: float = 1.5,
) -> pd.DataFrame:
    """Corrupt a few opens (un-adjusted split/dividend style) and return a copy.

    Free data sources retroactively adjust the *close* for a split/dividend but
    can leave the recorded *open* on the mismatch day un-adjusted. That bumps
    the open relative to the prior close, so the overnight leg (open/prev_close)
    spikes up while the intraday leg (close/open) spikes down: return is
    mechanically dumped from the day INTO the night — exactly the Yahoo artefact
    behind Knuteson's wildest emerging-market figures, and large enough to trip
    the >40% detector. ``rows`` are fractional positions; ``factor`` multiplies
    the open (and high, to keep OHLC self-consistent) on those days.
    """
    df = ohlc.copy()
    n = len(df)
    idx = sorted({int(p * n) for p in rows if 0 < p < 1})
    oi = df.columns.get_loc("Open")
    for i in idx:
        df.iloc[i, oi] *= factor
        if "High" in df.columns:
            hi = df.columns.get_loc("High")
            df.iloc[i, hi] = max(df.iloc[i, hi], df.iloc[i, oi])
    return df


def flag_suspicious_returns(dec: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """Return the rows whose overnight move exceeds ``threshold`` (default 40%).

    A blunt but effective artefact detector: legitimate index opens almost
    never gap ±40% overnight, so hits are overwhelmingly bad data. Pair with an
    official split calendar to harden it (see roadmap).
    """
    mask = dec["r_overnight"].abs() > threshold
    flagged = dec.loc[mask, ["r_overnight", "r_intraday", "r_close_close"]].copy()
    flagged["abs_overnight"] = flagged["r_overnight"].abs()
    return flagged.sort_values("abs_overnight", ascending=False)


# ---------------------------------------------------------------------------
# (C) Weighting / selection: how a portfolio's "overnight alpha" is built
# ---------------------------------------------------------------------------
def portfolio_decompose(
    panel: dict[str, pd.DataFrame],
    weights: str = "equal",
) -> pd.DataFrame:
    """Decompose a portfolio of tickers and show the weighting effect.

    Parameters
    ----------
    panel : mapping ticker -> OHLC DataFrame
    weights : 'equal' (rebalanced equal-weight each day) or
              'first' (everything in the first ticker; a crude proxy for the
              concentration / selection that puffs up a headline portfolio).

    Returns the decomposition of the resulting portfolio close-close series.
    """
    decs = {k: decompose(v) for k, v in panel.items()}
    common = None
    for d in decs.values():
        common = d.index if common is None else common.intersection(d.index)
    tickers = list(decs)
    if weights == "equal":
        w = {t: 1.0 / len(tickers) for t in tickers}
    elif weights == "first":
        w = {t: (1.0 if i == 0 else 0.0) for i, t in enumerate(tickers)}
    else:
        raise ValueError("weights must be 'equal' or 'first'")

    r_on = sum(w[t] * decs[t].loc[common, "r_overnight"] for t in tickers)
    r_id = sum(w[t] * decs[t].loc[common, "r_intraday"] for t in tickers)
    # Portfolio close-close from the weighted legs keeps the identity per day.
    out = pd.DataFrame({"r_overnight": r_on, "r_intraday": r_id})
    out["r_close_close"] = (1.0 + out["r_overnight"]) * (1.0 + out["r_intraday"]) - 1.0
    out["cum_overnight"] = (1.0 + out["r_overnight"]).cumprod() - 1.0
    out["cum_intraday"] = (1.0 + out["r_intraday"]).cumprod() - 1.0
    out["cum_close_close"] = (1.0 + out["r_close_close"]).cumprod() - 1.0
    return out
