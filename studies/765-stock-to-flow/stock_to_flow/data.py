"""Data layer for Study 765 — Stock-to-Flow.

PlanB's "Stock-to-Flow" (S2F) model (Twitter/@100trillionUSD, "Modeling Bitcoin's Value with
Scarcity", 2019-03-22) values Bitcoin from its *scarcity*: the ratio of the existing stock of
coins to the annual flow of newly-mined coins.

    SF(t) = stock(t) / flow(t)

and the model claims a power law between market value and SF:

    market_value = exp(a) * SF ** b     <=>     ln(price) = a + b * ln(SF)

with a reported in-sample R^2 ~ 0.95. The model's headline predictions — a six-figure BTC by
end-2021, ~$288k for the 2024 cycle — are what made it famous, and what made it a byword for a
busted model after the 2022 crash to ~$16k.

Two ingredients:

* **The S2F curve itself — RECONSTRUCTED FROM THE ISSUANCE SCHEDULE, not a proxy.** This is the
  clean part: Bitcoin's issuance is *deterministic consensus law*, not an estimated series. The
  block reward starts at 50 BTC and halves every 210,000 blocks; the halving block heights
  (0, 210,000, 420,000, 630,000, 840,000, ...) are exact, and the total coins minted in each
  4-year epoch is exactly ``reward * 210_000``. So ``stock`` at each halving is an EXACT integer
  (10.5M, 15.75M, 18.375M, 19.6875M, ...). The only modelling choice is the *within-epoch*
  interpolation by calendar date (real block intervals wobble around 10 minutes); we anchor
  supply exactly at each historical halving date and interpolate linearly between them. ``flow``
  is the annualised current issuance ``reward * blocks_per_year`` (144 blocks/day x 365). This is
  labelled a *reconstruction* rather than a proxy because the anchors are exact and only the
  daily granularity between anchors is smoothed.

* **Real BTC-USD tape.** Daily closes from yfinance (no key), cached as CSV. Price-only ==
  total-return for BTC (no dividends).

Pure numpy + pandas + stdlib on the offline path once cached. ``fetch_btc`` (network) runs once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
BTC_CACHE = os.path.join(CACHE_DIR, "s2f_btc_usd.csv")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07)
TICKER = "BTC-USD"

BLOCKS_PER_DAY = 144          # 10-minute target block time
BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365   # 52,560; the annualised-flow convention

# PlanB's article "Modeling Bitcoin's Value with Scarcity" went public on this date. Everything
# after it is genuinely out-of-sample for the published model.
PUBLICATION_DATE = "2019-03-22"

# --------------------------------------------------------------------------- #
# Bitcoin's issuance schedule — EXACT consensus law, not an estimate.
# Each row: (halving date, block height at halving, block reward AFTER that height).
# The reward is 50 BTC from genesis; it halves at each height. The historical halving DATES are
# public record; the block heights and per-epoch coin totals are exact by protocol.
# --------------------------------------------------------------------------- #
GENESIS_DATE = "2009-01-03"
HALVINGS = [
    # date         height    reward_after
    ("2012-11-28", 210_000, 25.0),
    ("2016-07-09", 420_000, 12.5),
    ("2020-05-11", 630_000, 6.25),
    ("2024-04-20", 840_000, 3.125),
    # future halvings are scheduled by height; the date is the network's own ~4-year estimate,
    # used only to extend the deterministic curve forward for the model line.
    ("2028-04-17", 1_050_000, 1.5625),
]
INITIAL_REWARD = 50.0


def _epoch_table() -> pd.DataFrame:
    """Piecewise-constant reward epochs with EXACT cumulative supply at each halving date.

    Supply at halving k is exactly ``sum_j reward_j * 210_000`` — an integer number of coins
    fixed by consensus (10,500,000 / 15,750,000 / 18,375,000 / 19,687,500 / ...). Within an
    epoch we interpolate supply linearly by date between the two bounding halving anchors, so
    the curve hits every halving exactly and is smooth in between.
    """
    dates = [pd.Timestamp(GENESIS_DATE)] + [pd.Timestamp(d) for d, _, _ in HALVINGS]
    rewards = [INITIAL_REWARD] + [r for _, _, r in HALVINGS]   # reward DURING each epoch
    # cumulative coins minted at the START of each epoch (i.e. at each halving date)
    supply_at = [0.0]
    for reward in rewards[:-1]:
        supply_at.append(supply_at[-1] + reward * 210_000)
    rows = []
    for i in range(len(dates) - 1):
        rows.append({
            "start": dates[i], "end": dates[i + 1],
            "reward": rewards[i],
            "supply_start": supply_at[i], "supply_end": supply_at[i + 1],
        })
    return pd.DataFrame(rows)


def supply_flow_daily(start: str = GENESIS_DATE, end: str = "2028-01-01") -> pd.DataFrame:
    """Daily stock, flow and stock-to-flow, reconstructed from the issuance schedule.

    Deterministic, offline, a pure function of ``HALVINGS``. Columns:

    * ``supply`` — circulating BTC (stock), exact at every halving, linearly interpolated by
      date within each epoch.
    * ``flow`` — annualised current issuance ``reward * BLOCKS_PER_YEAR`` (step function that
      halves at each halving).
    * ``sf`` — stock / flow, the raw Stock-to-Flow ratio (doubles at each halving).
    * ``sf_ma365`` — the 365-day trailing average of ``sf`` (PlanB's smoothed variant, which
      turns the halving step into a ramp).

    Named limitation: within-epoch daily granularity assumes a constant issuance rate per day
    (real block intervals wobble around the 10-minute target), so the *level* between halvings is
    smoothed. The halving anchors — the only points that matter for the model's structure — are
    exact.
    """
    ep = _epoch_table()
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    supply = np.full(len(idx), np.nan)
    flow = np.full(len(idx), np.nan)
    for _, row in ep.iterrows():
        mask = (idx >= row["start"]) & (idx < row["end"])
        if not mask.any():
            continue
        span_days = (row["end"] - row["start"]).days
        days_in = (idx[mask] - row["start"]).days
        frac = days_in / span_days
        supply[mask] = row["supply_start"] + frac * (row["supply_end"] - row["supply_start"])
        flow[mask] = row["reward"] * BLOCKS_PER_YEAR
    df = pd.DataFrame({"supply": supply, "flow": flow}, index=idx).dropna()
    df["sf"] = df["supply"] / df["flow"]
    df["sf_ma365"] = df["sf"].rolling(365, min_periods=1).mean()
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# Real tape — BTC-USD daily close
# --------------------------------------------------------------------------- #
def fetch_btc(start: str = "2010-07-01", end: str | None = None,
              path: str = BTC_CACHE) -> pd.Series:
    """Download daily BTC-USD closes and cache them. Network; run once."""
    import yfinance as yf

    px = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.name = "btc_usd"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_frame().to_csv(path)
    return px


def have_real(path: str = BTC_CACHE) -> bool:
    return os.path.exists(path)


def load_btc(path: str = BTC_CACHE, asof: str = AS_OF) -> pd.Series:
    """Cached daily BTC-USD close, sliced to the study as-of (sample never creeps)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    return px[px.index <= pd.Timestamp(asof)]


def join_price_sf(btc: pd.Series, sf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Align the BTC price tape with the reconstructed S2F curve on the price calendar."""
    if sf is None:
        sf = supply_flow_daily()
    df = pd.DataFrame({"price": btc}).join(sf, how="inner").dropna()
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# Synthetic world — planted valuation-residual -> forward-return effect
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 3000, beta: float = 0.0, seed: int = 765,
                    base_ann: float = 0.20, vol_ann: float = 0.70) -> pd.DataFrame:
    """Deterministic BTC-like world with a TUNABLE, explicitly-planted valuation signal.

    Construction (kept clean so the null is genuinely null):

    * ``resid`` — an exogenous, *stationary* AR(1) "valuation gap" (the number a trader reads off
      an S2F chart: negative = cheap vs model, positive = rich). Stationary so it can't manufacture
      the spurious-regression inflation a non-stationary regressor would.
    * ``ret`` — daily return = market noise + drift **minus** ``beta * resid[t-1]``: when
      ``beta > 0``, yesterday's cheapness (resid < 0) lifts today's return — a genuine planted
      mean-reversion-to-model effect. At ``beta = 0`` the return is pure noise, independent of the
      valuation gap: the NULL, where a valuation signal must read ~zero.

    Returns a frame with ``price``, a decorative rising ``sf``, and ``true_resid`` (the observable
    gap the detector regresses forward returns on).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_days, freq="D")
    sf = np.linspace(20.0, 120.0, n_days)        # decorative rising issuance curve
    phi, gap_sd = 0.90, 0.30                      # persistent but stationary valuation gap
    resid = np.zeros(n_days)
    innov = gap_sd * np.sqrt(1 - phi ** 2)
    for t in range(1, n_days):
        resid[t] = phi * resid[t - 1] + innov * rng.standard_normal()
    mu = base_ann / 365.0
    sig = vol_ann / np.sqrt(365.0)
    noise = mu + sig * rng.standard_normal(n_days)
    ret = noise.copy()
    ret[1:] -= beta * resid[:-1]                  # planted: cheap yesterday -> up today
    ln_price = 8.0 + np.cumsum(ret)
    df = pd.DataFrame({"price": np.exp(ln_price), "sf": sf, "true_resid": resid}, index=idx)
    df.index.name = "date"
    return df
