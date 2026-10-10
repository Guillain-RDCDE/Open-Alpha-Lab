"""Data layer for Study 1018 — Correlations Go to One.

Two real tapes and one synthetic world, all offline.

Real — :mod:`quantlab.bundled`, skfolio 1.8.1, SHA-256 pinned
    ``sp500_dataset``  daily **adjusted** closes of 20 large US stocks, 1990-01-02 → 2022-12-28.
                       A **survivor sample**: twenty companies that were large and alive in 2022.
                       For *average returns* that would be disqualifying. For *co-movement* it is
                       far less dangerous — the question here is how correlated these stocks were
                       with each other in calm versus turbulent markets, not how much they earned —
                       but it is not harmless either, and the direction is reasoned out in the
                       results caveats: firms that died in a crisis are exactly the ones whose
                       returns decoupled into idiosyncratic collapse, so dropping them most
                       plausibly *raises* measured crisis correlation slightly. Named, not hidden.
    ``sp500_index``    the daily S&P 500 **price index** (no dividends) — used only as the
                       *market state variable* (its volatility, drawdown and big down days define
                       the regimes) and as the market leg of the exceedance correlations. A
                       missing dividend is a ~2%/yr drift; it moves no volatility or correlation
                       number in this study by any visible amount.

The tape stops on 2022-12-28, mid-month. ``AS_OF`` is the last full month, **2022-11-30**; the
partial December is dropped before any statistic is computed.

Synthetic — :func:`synthetic_panel`
    A deterministic panel with a ``signal_strength`` knob. Every asset shares one stochastic
    volatility process, so in the **null** (``signal_strength=0``) the true pairwise correlation
    is *exactly constant* while volatility clusters and spikes — the artefact alone. At
    ``signal_strength=1`` the true correlation jumps by a planted amount whenever the latent
    volatility state is in its top decile: a genuine crisis-correlation effect of known size.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantlab import bundled  # noqa: E402

TRADING_DAYS = 252

STOCK_TAPE = "sp500_dataset"
INDEX_TAPE = "sp500_index"
INDEX = "SP500"
TICKERS = ("AAPL", "AMD", "BAC", "BBY", "CVX", "GE", "HD", "JNJ", "JPM", "KO", "LLY", "MRK",
           "MSFT", "PEP", "PFE", "PG", "RRC", "UNH", "WMT", "XOM")

START = "1990-01-02"
# Last FULL month of the tape (it ends 2022-12-28): the partial December is dropped.
AS_OF = "2022-11-30"

PROVENANCE = {
    "package": "skfolio 1.8.1 (wheel fetched once from PyPI, SHA-256 verified)",
    "wheel_sha256": bundled.SKFOLIO_WHEEL_SHA256,
    "files": {k: {"file": v[0], "sha256": v[1]} for k, v in bundled.SKFOLIO_FILES.items()
              if k in (STOCK_TAPE, INDEX_TAPE)},
    "stocks": "daily adjusted closes, 20 large US stocks — SURVIVOR sample",
    "index": "daily S&P 500 PRICE index (no dividends) — the market state variable",
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def have_real(cache_dir: str = bundled.DEFAULT_CACHE) -> bool:
    """True iff the pinned skfolio tapes are in the local cache (never touches the network)."""
    return bundled.have_skfolio(cache_dir)


def _slice(df, asof: str):
    return df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(asof))]


def load_prices(asof: str = AS_OF, cache_dir: str = bundled.DEFAULT_CACHE) -> pd.DataFrame:
    """The 20-stock adjusted-close panel (survivor sample), sliced to ``asof``."""
    df = bundled.load_skfolio(STOCK_TAPE, cache_dir=cache_dir)
    return _slice(df[list(TICKERS)], asof)


def load_index(asof: str = AS_OF, cache_dir: str = bundled.DEFAULT_CACHE) -> pd.Series:
    """The S&P 500 **price** index, sliced to ``asof``."""
    df = bundled.load_skfolio(INDEX_TAPE, cache_dir=cache_dir)
    return _slice(df[INDEX], asof)


def load_returns(asof: str = AS_OF, cache_dir: str = bundled.DEFAULT_CACHE
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Daily simple returns ``(stocks, index)`` on a common calendar, first row dropped.

    Simple, not log: the diversified holder's portfolio is a weighted sum of *simple* returns,
    and correlations computed either way agree to the third decimal at daily frequency.
    """
    px = load_prices(asof, cache_dir)
    ix = load_index(asof, cache_dir)
    both = pd.concat([px, ix.rename(INDEX)], axis=1, sort=False).dropna()
    r = both.pct_change().iloc[1:]
    return r[list(TICKERS)], r[INDEX]


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex chars) — :func:`quantlab.bundled.fingerprint`."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# Synthetic world — constant correlation + stochastic volatility, plus a planted jump
# --------------------------------------------------------------------------- #
def synthetic_panel(n_assets: int = 10, n_years: int = 16, rho0: float = 0.30,
                    crisis_jump: float = 0.30, signal_strength: float = 1.0,
                    vol_ann: float = 0.25, vol_persistence: float = 0.985,
                    log_vol_sd: float = 0.45, crisis_quantile: float = 0.90,
                    start: str = "2000-01-03", seed: int = 1018
                    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A daily return panel whose true correlation is known on every single day.

    Construction (one factor, one shared stochastic volatility)::

        h_t   = phi * h_{t-1} + log_vol_sd * sqrt(1 - phi^2) * xi_t        (latent log-vol)
        s_t   = vol_ann / sqrt(252) * exp(h_t - log_vol_sd^2 / 2)
        rho_t = rho0 + signal_strength * crisis_jump * 1{h_t > q_crisis(h)}
        r_it  = s_t * v_i * ( sqrt(rho_t) * f_t + sqrt(1 - rho_t) * e_it )

    with ``f, e, xi`` i.i.d. standard normal and ``v_i`` a fixed per-asset vol scale. Because
    the volatility multiplies the common and the idiosyncratic parts *alike*, the conditional
    correlation of any pair is exactly ``rho_t``. In the null (``signal_strength=0``) that is a
    constant ``rho0`` while volatility clusters and spikes — so any rise a procedure reports in
    high-volatility periods is artefact by construction. At ``signal_strength=1`` correlation
    genuinely jumps to ``rho0 + crisis_jump`` in the top-``crisis_quantile`` volatility states.

    The "market" is the equal-weight average of the assets, playing the role the S&P 500 plays
    on the real tape. Returns ``(returns, market, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS)
    phi = float(vol_persistence)
    h = np.empty(n)
    h[0] = rng.normal(0.0, log_vol_sd)
    xi = rng.normal(0.0, 1.0, n)
    k = log_vol_sd * np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        h[t] = phi * h[t - 1] + k * xi[t]
    s = vol_ann / np.sqrt(TRADING_DAYS) * np.exp(h - log_vol_sd ** 2 / 2.0)
    crisis = h > np.quantile(h, crisis_quantile)
    jump = float(signal_strength) * float(crisis_jump)
    rho = rho0 + jump * crisis.astype(float)
    if np.any(rho >= 1.0) or np.any(rho < 0.0):
        raise ValueError("planted correlation must stay in [0, 1)")
    v = rng.uniform(0.8, 1.3, n_assets)
    f = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, 1.0, (n, n_assets))
    z = np.sqrt(rho)[:, None] * f[:, None] + np.sqrt(1.0 - rho)[:, None] * e
    r = s[:, None] * v[None, :] * z
    idx = pd.bdate_range(start=start, periods=n, name="date")
    rets = pd.DataFrame(r, index=idx, columns=[f"S{i:02d}" for i in range(n_assets)])
    market = rets.mean(axis=1).rename("MKT")
    truth = {"n_assets": n_assets, "n_days": n, "seed": seed, "rho0": rho0,
             "crisis_jump": crisis_jump, "signal_strength": float(signal_strength),
             "rho_crisis": rho0 + jump, "planted_jump": jump,
             "crisis_quantile": crisis_quantile, "phi": phi, "log_vol_sd": log_vol_sd,
             "crisis": pd.Series(crisis, index=idx, name="crisis"),
             "sigma": pd.Series(s, index=idx, name="sigma"),
             "rho": pd.Series(rho, index=idx, name="rho")}
    return rets, market, truth
