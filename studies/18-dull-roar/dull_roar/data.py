"""Data access for the low-volatility study — the cross-section of returns and where it comes from.

This study lives in the **cross-section**: it sorts a universe of stocks by their past volatility and
asks whether the calm ones beat the wild ones on a risk-adjusted basis. So unlike the desk's
single-series studies, the tape here is a *panel* — daily returns for many names at once — and the
data layer keeps the desk's standing split between an offline synthetic universe and a cache-only
real reader:

    * :func:`synthetic_panel` — fully **offline**. A single-factor model with a *baked-in* low-vol
      anomaly: ``r_i = alpha_i + beta_i * r_market + nu_i * eps_i``. The anomaly is injected as a
      **flat security-market line** — ``alpha_i = -sml_slope * (beta_i - mean_beta)`` — so with
      ``sml_slope > 0`` the low-beta (hence low-vol) names carry *positive* alpha and the high-beta
      names *negative* alpha, exactly the Frazzini-Pedersen (2014) "the SML is too flat" mechanism.
      Idiosyncratic vol is **proportional to beta** (``nu_i = idio_load * beta_i`` — the lottery-stock
      pattern, Bali-Cakici-Whitelaw 2011), which buys two things: a sort on *total* realized vol
      becomes a clean sort on beta, and — crucially — under the null every stock's Sharpe is *identical*
      (``mkt_drift / sqrt(mkt_vol^2 + idio_load^2)``), so the control is genuinely flat. Set
      ``sml_slope = 0`` for that **null**: a textbook-CAPM universe where vol carries *no* alpha and a
      low-vol sort earns nothing on a beta-adjusted basis. Deterministic given ``seed``; no network.
    * :func:`fetch_panel` — pull a real cross-section (current S&P 500 constituents) via the shared
      :mod:`quantlab.universe` engine, which caches one batched OHLC download to parquet and returns a
      ``{ticker: close-series}`` map. **Cache-only** unless ``fetch=True``: a missing cache is returned
      empty, never a silent network stall, so the offline core (and CI) never imports ``yfinance``.

Two data choices, named up front per the house rule. **Total-return (split/dividend-adjusted)
closes**: the anomaly is a statement about *total* realized returns, and dividends differ
systematically across the vol cross-section (low-vol names are dividend-heavy), so dropping them would
bias the very sort we study. **Current index membership**: the real panel uses *today's* S&P 500, so
it carries **survivorship bias** (it excludes names that were delisted) — the qualitative low-minus-
high ranking is robust to this, precise magnitudes are not, and we say so wherever a real number
appears.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with a baked-in low-volatility anomaly
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PanelTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_stocks: int
    n_bars: int
    sml_slope: float          # the anomaly: per-unit-beta alpha tilt (daily). 0 == textbook CAPM null
    mkt_drift: float          # daily market drift
    mkt_vol: float            # daily market vol
    idio_load: float          # idiosyncratic vol per unit beta
    beta_lo: float
    beta_hi: float

    @property
    def has_anomaly(self) -> bool:
        """True when the security-market line is tilted (low-beta earns alpha) — something to find."""
        return self.sml_slope != 0.0

    @property
    def null_stock_sharpe(self) -> float:
        """The Sharpe *every* stock shares under the null (alpha=0), since nu_i scales with beta.

        ``Sharpe_i = beta_i*mkt_drift / (beta_i*sqrt(mkt_vol^2+idio_load^2))`` — the betas cancel, so a
        fair-CAPM universe is perfectly flat across the vol sort. Annualised.
        """
        return float(self.mkt_drift / np.sqrt(self.mkt_vol**2 + self.idio_load**2)
                     * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def annual_sml_alpha_spread(self) -> float:
        """Baked annual alpha gap between the lowest-beta and highest-beta stock — the anomaly's size.

        ``alpha(beta) = -sml_slope*(beta - mean_beta)``, so the low-minus-high-beta alpha spread is
        ``sml_slope*(beta_hi - beta_lo)`` per day, annualised by ``*252``. The ceiling a *perfect*,
        beta-neutral long-short could extract before idiosyncratic noise, costs or shorting frictions.
        """
        return float(self.sml_slope * (self.beta_hi - self.beta_lo) * TRADING_DAYS_PER_YEAR)


def synthetic_panel(
    n_stocks: int = 200,
    n_bars: int = 4032,
    sml_slope: float = 0.00018,
    mkt_drift: float = 0.0003,
    mkt_vol: float = 0.010,
    idio_load: float = 0.011,
    beta_lo: float = 0.4,
    beta_hi: float = 1.8,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A toy daily cross-section whose **low-volatility names carry positive alpha** by construction.

    The return process for stock ``i`` is the textbook single-factor model with a *tilted* intercept::

        r_market[t] = mkt_drift + mkt_vol * z[t],                 z ~ N(0, 1)   (one shared factor)
        alpha_i     = -sml_slope * (beta_i - mean(beta))          (the anomaly: a flat SML)
        nu_i        = idio_load * beta_i                          (lottery effect: high-beta => high idio)
        r_i[t]      = alpha_i + beta_i * r_market[t] + nu_i * eps_i[t],   eps ~ N(0, 1)  (i.i.d.)

    Two facts are baked in on purpose. **(1)** Because ``alpha_i`` *decreases* in ``beta_i`` while CAPM
    says it should be flat at zero, the low-beta (and so low-total-vol) names are *underpriced* — the
    Frazzini-Pedersen (2014) leverage-constraint story, where the security-market line is too flat.
    **(2)** Idiosyncratic vol is proportional to beta, so total vol ``sigma_i = beta_i *
    sqrt(mkt_vol^2 + idio_load^2)`` is monotone in beta — a sort on *observable past vol* recovers the
    *unobservable* alpha gradient, and the proportionality makes the null's per-stock Sharpe identical
    across the cross-section (a genuinely flat control). Pass ``sml_slope = 0`` for that null.

    Returns ``(panel, market, truth)`` — ``panel`` is a ``dates x stock`` returns DataFrame (columns
    ``STK000...``), ``market`` is the shared factor's daily return, ``truth`` carries the baked
    parameters, the null Sharpe and the annual alpha spread. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2010-01-04", periods=n_bars, name="date")

    market = pd.Series(mkt_drift + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")

    betas = np.linspace(beta_lo, beta_hi, n_stocks)
    rng.shuffle(betas)
    alphas = -sml_slope * (betas - betas.mean())                 # the flat SML: low beta => +alpha
    nu = idio_load * betas                                       # lottery: idio vol scales with beta

    mkt = market.to_numpy()[:, None]                             # (T, 1)
    eps = rng.standard_normal((n_bars, n_stocks))
    rets = alphas[None, :] + betas[None, :] * mkt + nu[None, :] * eps

    cols = [f"STK{i:03d}" for i in range(n_stocks)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    truth = PanelTruth(
        n_stocks=n_stocks, n_bars=n_bars, sml_slope=sml_slope,
        mkt_drift=mkt_drift, mkt_vol=mkt_vol, idio_load=idio_load,
        beta_lo=beta_lo, beta_hi=beta_hi,
    )
    return panel, market, truth


# --------------------------------------------------------------------------- #
# Real cross-section — current S&P 500 closes via the shared quantlab engine
# --------------------------------------------------------------------------- #

def fetch_panel(
    start: str = "2010-01-01",
    min_days: int = 2500,
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Return a ``dates x ticker`` **daily-returns** panel for the current S&P 500, cache-first.

    Leans on :mod:`quantlab.universe` (the desk's breadth engine): it reads the cached batched OHLC
    download — ``_cache/panel_<N>_<start>.parquet`` — and we reduce each name to its split/dividend-
    adjusted close, keep those with at least ``min_days`` of history (so a freshly-listed name can't
    distort a vol sort), and difference to daily returns. **Cache-only by default**: if the panel is
    not cached an empty frame is returned *unless* ``fetch=True``, which lets ``quantlab.universe`` hit
    Yahoo once. The network import stays lazy, so the offline core never imports ``yfinance``.
    """
    import sys
    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if cache_dir is not None:
        os.environ["OVERNIGHT_CACHE"] = cache_dir

    from quantlab import universe  # lazy; pulls in the shared breadth engine

    symbols = universe.sp500_symbols(use_cache=True)
    cache_file = os.path.join(
        os.environ.get("OVERNIGHT_CACHE", os.path.join(repo_root, "_cache")),
        f"panel_{len(symbols)}_{start}.parquet",
    )
    if not os.path.exists(cache_file) and not fetch:
        return pd.DataFrame()

    panel = universe.download_panel(symbols, start=start, use_cache=True)

    closes = {}
    for tk, ohlc in panel.items():
        c = ohlc["Close"].dropna()
        if len(c) >= min_days:
            closes[tk] = c
    if not closes:
        return pd.DataFrame()

    prices = pd.DataFrame(closes).sort_index()
    rets = prices.pct_change().dropna(how="all")
    rets.index.name = "date"
    return rets
