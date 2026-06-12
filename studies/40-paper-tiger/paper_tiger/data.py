"""Data for the dual-momentum study — an offline synthetic world, and a cached real ETF tape.

The desk's offline/cache split (same contract as study 31):

  * :func:`synthetic_world` — fully **offline, deterministic**. Two equity series (US, INTL), one bond
    series and a T-bill rate, with a controllable **momentum premium** and a **crash** regime. With the
    premium on, the trailing-12-month winner tends to keep winning *and* both equities periodically
    crash together (so the absolute-momentum filter has something to dodge). ``premium = 0`` is the
    null: three correlated random walks where dual momentum should add nothing.
  * :func:`fetch_etfs` — the real GEM universe (SPY, EFA, AGG) plus the 13-week T-bill yield (^IRX),
    monthly, total-return, **cache-first**; network only on an explicit ``fetch=True`` cache miss. The
    fingerprinted real run lives in ``docs/results.md``.

Why these series: GEM is a *three-door* rule — US equity, foreign equity, or bonds — gated by a T-bill
hurdle. SPY/EFA/AGG are the cheapest, longest, most liquid ETF proxies for those doors; ^IRX supplies
the cash hurdle back to the 1990s. Monthly total-return prices because the rule rebalances monthly on
total return. Stated, not hidden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
MONTHS_PER_YEAR = 12

# The real GEM doors (Yahoo tickers) + the cash hurdle.
ETFS = {"SPY": "US", "EFA": "INTL", "AGG": "BOND"}
TBILL_TICKER = "^IRX"  # 13-week T-bill discount yield, percent annualised


@dataclass(frozen=True)
class WorldTruth:
    premium: float
    crash: bool

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


def synthetic_world(
    n_months: int = 12 * 25,
    premium: float = 0.12,
    crash: bool = True,
    vol_ann: float = 0.15,
    regime_months: float = 18.0,
    tbill_ann: float = 0.02,
    seed: int = 40,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A monthly synthetic world for the dual-momentum machinery — deterministic given ``seed``.

    Two equities (US, INTL) follow slow up/flat/down drift regimes (switching ~every
    ``regime_months``); in a trending regime the monthly drift is ``±premium·σ_monthly`` so the
    trailing-12-month winner tends to persist (relative momentum has signal). When ``crash`` is set,
    rare shared bear regimes drag *both* equities below the T-bill hurdle, which is exactly what the
    absolute-momentum filter exists to dodge (so the bond door earns its keep). The bond series is a
    quiet positive drift. ``premium = 0`` kills the drift → three random walks (the null).

    Returns ``(prices[US,INTL,BOND], tbill_monthly_return, truth)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME", name="date")
    sig_m = vol_ann / np.sqrt(MONTHS_PER_YEAR)
    p_switch = 1.0 / regime_months

    states = rng.choice([-1, 0, 1], size=2)  # US, INTL
    bear = 0
    eq = np.empty((n_months, 2))
    for t in range(n_months):
        flip = rng.random(2) < p_switch
        states = np.where(flip, rng.choice([-1, 0, 1], size=2), states)
        # rare shared bear market (both equities down hard) when crash is enabled
        if crash and rng.random() < 1.0 / 60.0:
            bear = rng.integers(4, 10)  # a multi-month drawdown
        drift = premium * sig_m * states
        if bear > 0:
            drift = drift - 2.5 * sig_m  # both equities bleed together
            bear -= 1
        eq[t] = drift + sig_m * rng.standard_normal(2)

    bond = 0.25 * sig_m + 0.30 * sig_m * rng.standard_normal(n_months)  # quiet, low-vol positive drift
    rets = pd.DataFrame({"US": eq[:, 0], "INTL": eq[:, 1], "BOND": bond}, index=idx)
    prices = (1.0 + rets).cumprod() * 100.0
    tbill_m = pd.Series(np.full(n_months, tbill_ann / MONTHS_PER_YEAR), index=idx, name="tbill")
    return prices, tbill_m, WorldTruth(premium, crash)


def fetch_etfs(
    cache_dir: str = DEFAULT_CACHE, fetch: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """The real GEM tape: monthly total-return prices for SPY/EFA/AGG + the ^IRX T-bill return.

    **Cache-only** unless ``fetch=True`` (then downloads from Yahoo). Returns
    ``(prices[US,INTL,BOND], tbill_monthly_return)`` aligned from the first month all three ETFs exist,
    or ``(empty, empty)`` on a cache miss with ``fetch=False`` so offline runs never hit the network.
    """
    cache = os.path.join(cache_dir, "paper_tiger_etfs.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[["US", "INTL", "BOND"]], df["tbill"]
    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float)

    import yfinance as yf  # lazy

    raw = yf.download(list(ETFS) + [TBILL_TICKER], period="max", interval="1mo",
                      auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px.resample("ME").last()
    # A month only exists once its month-end has passed. Yahoo ships the current month as a
    # partial bar; resampling relabels it to a *future* month-end, which would poison the
    # as-of stamp and fingerprint. Drop it — the tape ends at the last completed month.
    px = px[px.index < pd.Timestamp.now().normalize()]
    prices = px[list(ETFS)].rename(columns=ETFS)
    tbill_m = (px[TBILL_TICKER].ffill() / 100.0) / MONTHS_PER_YEAR  # yield% → monthly return
    start = prices.dropna().index.min()
    prices, tbill_m = prices.loc[start:], tbill_m.loc[start:]
    tbill_m.name = "tbill"
    out = prices.join(tbill_m)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return prices, tbill_m
