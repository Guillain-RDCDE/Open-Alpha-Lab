"""Data layer for Study 578 (Cross-Asset-Correlation-Regime).

The signal is the **mean pairwise correlation** across a small multi-asset ETF panel, measured over
a trailing window and read as a fragility gauge. When correlations rise toward 1, the assets stop
diversifying each other — the folk claim is that this clustering *precedes* market stress, so a
high/rising correlation regime should flag lower forward returns / higher forward vol on a risk
asset (SPY).

Two tapes, one schema (a daily-return frame across the panel):

- ``synthetic_series`` — a deterministic, offline generator. A single knob, ``fragility``, plants
  the effect: a latent regime process raises the *common-factor loading* (so realised cross-asset
  correlation rises) AND simultaneously drags the risk asset's forward drift down and its vol up.
  ``fragility > 0`` reproduces the claim (high correlation → bad forward tape); ``= 0`` is the null
  (correlation still varies, but carries no forward information). The positive control and null in
  one bottle; tests never touch the network.
- ``fetch_returns`` — the real yfinance tape: daily adjusted-close returns for a multi-asset ETF
  panel (equities, bonds, gold, commodities, real estate, an FX/EM slice), cache-first into this
  study's own ``_cache/`` so the reproducible core never needs the network.

The panel is deliberately *cross-asset* (stocks, bonds, gold, oil, REITs, EM) rather than
cross-equity, so the "average correlation" really measures how much *distinct* asset classes are
moving together — the quantity the fragility folklore is about.

Data-availability note: the panel is ETFs that *exist today*, so there is no cross-sectional
survivorship story (these are broad index vehicles, not single names); the real limitation is that
liquid multi-asset ETF history only reaches back ~2007 (GLD/TLT/DBC era), so the sample spans one
GFC and one COVID stress, not a century of crises — named on the SIGNAL axis.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# A cross-asset ETF panel: broad equity, size/tech tilts, bonds (long + intermediate + credit),
# gold, broad commodities, oil, real estate, and an EM/international slice. These are the distinct
# asset classes whose *co-movement* the fragility folklore is about. The risk asset we forecast is
# SPY (kept first).
PANEL = [
    "SPY",   # US large-cap equity (the risk asset we forecast)
    "QQQ",   # US tech / growth
    "IWM",   # US small-cap
    "EFA",   # developed ex-US equity
    "EEM",   # emerging-market equity
    "TLT",   # long Treasuries
    "IEF",   # intermediate Treasuries
    "LQD",   # investment-grade credit
    "HYG",   # high-yield credit
    "GLD",   # gold
    "DBC",   # broad commodities
    "USO",   # oil
    "VNQ",   # US real estate
    "SLV",   # silver
]
_seen: set[str] = set()
PANEL = [t for t in PANEL if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

RISK_ASSET = "SPY"
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    fragility: float  # >0 = high correlation predicts bad forward tape; 0 = null

    @property
    def has_effect(self) -> bool:
        return self.fragility != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 4000,
    n_assets: int = 12,
    fragility: float = 0.9,
    base_corr: float = 0.2,
    seed: int = 578,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible multi-asset daily-return panel with a tunable fragility regime.

    A latent AR(1) regime process ``g[t]`` in [0, 1] drives the **common-factor loading**: on
    high-``g`` days every asset loads more heavily on the single market factor, so the realised
    cross-asset correlation rises toward 1 (the "everything moves together" state). The *same*
    ``g`` drags the risk asset's drift down and its volatility up by an amount scaled by
    ``fragility``:

        asset_ret[t,i] = load(g[t]) * factor[t] + sqrt(1 - load^2) * idio[t,i]
        risk_ret[t]    = asset_ret[t,0] - fragility * beta_dd * g[t] + fragility * beta_v * g[t]*shock

    With ``fragility > 0`` the high-correlation state carries genuinely worse forward returns and
    higher vol (the claim); ``fragility = 0`` leaves correlation varying but forward-uninformative
    (the null). The returned frame is daily returns, one column per asset, ``RISK_ASSET`` first —
    the identical schema ``fetch_returns`` produces from real prices.

    Returns ``(returns, truth)``.
    """
    rng = np.random.default_rng(seed)

    # Latent regime g in [0,1]: persistent, occasionally spiking (a fragility/stress state).
    g = np.zeros(n_days)
    x = 0.0
    for t in range(1, n_days):
        x = 0.985 * x + rng.standard_normal() * 0.16
        g[t] = x
    # squash to [0,1] with a right-skew so the high-corr state is the minority (stress is rare)
    g = 1.0 / (1.0 + np.exp(-2.2 * (g - g.mean())))

    # Common-factor loading rises with g: low-corr floor -> high-corr near 1.
    load = np.sqrt(np.clip(base_corr + (0.9 - base_corr) * g, 0.02, 0.97))

    factor = rng.standard_normal(n_days) * 0.010
    idio = rng.standard_normal((n_days, n_assets)) * 0.010
    rets = load[:, None] * factor[:, None] + np.sqrt(1.0 - load[:, None] ** 2) * idio

    # The risk asset (column 0): fragility ties the high-corr state to worse drift + fatter vol.
    beta_dd = 0.0009   # drift drag per unit g per day
    beta_v = 0.010     # extra vol shock scaled by g
    shock = rng.standard_normal(n_days)
    rets[:, 0] = (
        rets[:, 0]
        + 0.00035                                   # base equity drift
        - fragility * beta_dd * g
        + fragility * beta_v * g * shock
    )

    cols = [RISK_ASSET] + [f"A{j:02d}" for j in range(1, n_assets)]
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    return panel, WorldTruth(fragility=fragility)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices -> returns, cache-first
# ---------------------------------------------------------------------------
def _returns_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "cac_returns.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    """Tiny retry wrapper for yfinance flakiness."""
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_returns(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2007-01-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily returns for the cross-asset ETF panel, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached). The returned frame
    is daily simple returns, one column per ETF, ``RISK_ASSET`` first; rows are dropped only where
    the risk asset itself is missing so the panel-average correlation is honest.
    """
    path = _returns_cache()
    if os.path.exists(path):
        rets = pd.read_parquet(path)
        if rets.index.tz is not None:
            rets.index = rets.index.tz_localize(None)
        return rets
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download(
            list(PANEL), start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    # keep the columns we asked for, in panel order, that actually came back
    cols = [c for c in PANEL if c in raw.columns]
    raw = raw[cols]
    rets = raw.pct_change()
    rets = rets.dropna(subset=[RISK_ASSET])
    # require the risk asset + at least a few other assets present each row
    rets = rets.dropna(thresh=6)
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(path)
    return rets


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0.0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
