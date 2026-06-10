"""Data for the macro-momentum study — an offline synthetic cross-asset world driven by latent macro
state, and the real hook that *would* pull FRED macro series + asset prices.

Macro momentum is a slow, cross-asset premium: the *trend* in fundamental macro data (growth, inflation)
predicts the next stretch of asset returns. Improving growth lifts pro-cyclical assets (equities,
commodities); rising inflation favours *real* assets (commodities, a TIPS/gold proxy) over *nominal*
bonds. The signal is the **change** in the (slow, persistent, regime-switching) macro state — not its
level — and it must be lagged so it is causally tradable.

The desk's offline/cache split, mirrored from Study 27 (Steamroller) because the real macro tape is, in
this environment, **not reliably fetchable**:

  * :func:`synthetic_macro` — fully **offline, deterministic**. A small cross-asset panel (equities,
    nominal bonds, commodities, a real-asset/TIPS proxy, gold) is driven by two latent macro state
    variables — *growth* and *inflation* — each a persistent, regime-switching process. The **momentum**
    (one-period change) of each macro driver predicts next-period asset returns through fixed, signed
    betas, and rising-inflation regimes tilt the world toward real assets (the positive control).
    ``macro_strength`` sets how strongly macro momentum drives returns; ``macro_strength = 0`` is the
    **null** (assets are pure noise — no macro predictability). Deterministic given ``seed``.
  * :func:`fetch_macro` — the real hook. It *would* pull **FRED macro series** (growth: ``INDPRO`` /
    ``PAYEMS``; inflation: ``CPIAUCSL`` / ``T10YIE``) plus asset proxies, align them monthly, and cache
    the result. **Cache-first; network only behind ``fetch=True``.** Daily FRED rate series time out in
    this sandbox and even the small monthly CPI series only succeeds intermittently, so on a cache-miss
    this returns ``{}`` (Steamroller-style) and the synthetic core stands alone — see ``docs/results.md``.

Data choice, named up front: a **monthly** frequency. Macro releases are monthly and the signal is a
slow trend, so monthly is the right (and the cleanly available) horizon; using one public source (FRED)
for the macro state and liquid proxies for the assets keeps the panel internally consistent.
"""

from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

MONTHS_PER_YEAR = 12

# The toy cross-asset world. Each asset has a sign on growth-momentum and on inflation-momentum, and a
# flag for whether it is a *real* asset (the inflation hedge overweights these when inflation rises).
#   growth +  → pro-cyclical (equities, commodities);  growth − → defensive (nominal bonds)
#   infl   +  → real asset    (commodities, TIPS proxy, gold);  infl − → nominal bond (hurt by inflation)
ASSETS: dict[str, dict] = {
    "EQ":   {"g_beta": +1.0, "i_beta": -0.3, "real": False, "vol": 0.040},  # equities
    "BOND": {"g_beta": -0.6, "i_beta": -0.8, "real": False, "vol": 0.018},  # nominal bonds
    "CMDTY":{"g_beta": +0.7, "i_beta": +1.0, "real": True,  "vol": 0.050},  # commodities
    "TIPS": {"g_beta": -0.2, "i_beta": +0.6, "real": True,  "vol": 0.020},  # inflation-linked / real-rate proxy
    "GOLD": {"g_beta": -0.1, "i_beta": +0.7, "real": True,  "vol": 0.045},  # gold
}
ASSET_ORDER = list(ASSETS.keys())


@dataclass(frozen=True)
class MacroTruth:
    """What the synthetic generator baked in, so a test can check the strategy recovers it."""
    n_assets: int
    n_months: int
    macro_strength: float       # how strongly macro momentum drives returns; 0 == null (no predictability)

    @property
    def has_macro(self) -> bool:
        return self.macro_strength != 0.0


def _regime_state(rng: np.random.Generator, n: int, p_stay: float) -> np.ndarray:
    """A sticky two-state (low/high) Markov chain in {-1, +1}, persistent so regimes last for years."""
    s = np.empty(n)
    state = 1.0 if rng.random() < 0.5 else -1.0
    for t in range(n):
        if rng.random() > p_stay:               # flip with prob (1 - p_stay)
            state = -state
        s[t] = state
    return s


def synthetic_macro(n_months: int = 600, macro_strength: float = 1.0, growth_persist: float = 0.96,
                    infl_persist: float = 0.985, regime_stay: float = 0.985, macro_vol: float = 0.6,
                    noise: float = 0.7, seed: int = 37
                    ) -> tuple[pd.DataFrame, pd.DataFrame, MacroTruth]:
    """A small cross-asset world where the *trend* in latent macro state predicts next-month returns.

    Two latent macro state variables evolve as persistent, regime-switching AR(1) processes::

        regime_x_t        : sticky two-state chain in {-1,+1}  (multi-year up/down regimes)
        x_t = persist · x_{t-1} + (1-persist) · drift·regime_x_t + macro_vol·√(1-persist²)·ε

    for ``x in {growth, inflation}``. The **macro momentum** is the one-month change ``Δx_t = x_t -
    x_{t-1}``. Each asset's next-month excess return loads on the *lagged* macro momentum through its
    fixed signed betas plus idiosyncratic noise::

        r_{a,t} = macro_strength · ( g_beta_a · Δgrowth_{t-1} + i_beta_a · Δinfl_{t-1} ) · vol_a
                  + noise · vol_a · η_{a,t}

    Because real assets carry positive inflation betas, a **rising-inflation regime** (Δinfl > 0 for a
    sustained stretch) tilts returns toward commodities / TIPS / gold — the positive control the
    inflation-hedge tilt is built to harvest. ``macro_strength = 0`` removes the macro term entirely:
    assets become pure noise, a clean **null** with nothing to predict.

    Returns ``(asset_returns, macro_state, truth)``: ``asset_returns`` a monthly ``months × asset`` frame,
    ``macro_state`` a monthly frame with columns ``growth`` and ``inflation`` (the *levels*; the strategy
    differences them itself). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1975-01", periods=n_months, freq="M").to_timestamp(how="end")
    idx.name = "date"

    def latent(persist: float, drift: float) -> np.ndarray:
        reg = _regime_state(rng, n_months, regime_stay)
        x = np.empty(n_months)
        x[0] = 0.0
        shock_sd = macro_vol * np.sqrt(max(1e-9, 1.0 - persist ** 2))
        for t in range(1, n_months):
            x[t] = persist * x[t - 1] + (1.0 - persist) * drift * reg[t] + shock_sd * rng.standard_normal()
        return x

    growth = latent(growth_persist, drift=8.0)
    infl = latent(infl_persist, drift=10.0)
    macro = pd.DataFrame({"growth": growth, "inflation": infl}, index=idx)

    # macro momentum = one-month change in the latent state; LAG it one month so returns are causal.
    dg = np.diff(growth, prepend=growth[0])
    di = np.diff(infl, prepend=infl[0])
    dg_lag = np.concatenate([[0.0], dg[:-1]])
    di_lag = np.concatenate([[0.0], di[:-1]])

    rets = np.empty((n_months, len(ASSET_ORDER)))
    for j, a in enumerate(ASSET_ORDER):
        cfg = ASSETS[a]
        signal = cfg["g_beta"] * dg_lag + cfg["i_beta"] * di_lag
        eta = rng.standard_normal(n_months)
        rets[:, j] = macro_strength * signal * cfg["vol"] + noise * cfg["vol"] * eta

    returns = pd.DataFrame(rets, index=idx, columns=ASSET_ORDER)
    return returns, macro, MacroTruth(n_assets=len(ASSET_ORDER), n_months=n_months,
                                      macro_strength=macro_strength)


# --------------------------------------------------------------------------- #
# Real tape — FRED macro series + asset proxies (CSV, no API key). Network only behind fetch=True.
# In THIS environment the daily rate series time out and even monthly CPI is intermittent, so the
# cache-miss path returns {} and the synthetic core is the offline proof (Steamroller pattern).
# --------------------------------------------------------------------------- #

# Growth and inflation macro drivers (monthly FRED ids). T10YIE is daily (breakeven inflation expectn).
FRED_MACRO = {
    "growth_indpro": "INDPRO",      # industrial production index (monthly)
    "growth_payems": "PAYEMS",      # nonfarm payrolls (monthly)
    "infl_cpi":      "CPIAUCSL",    # CPI, all urban consumers (monthly)  -- the small series we MAY try
    "infl_be10y":    "T10YIE",      # 10y breakeven inflation (DAILY -- times out here)
}
# Liquid asset proxies (FRED has a few; the rest the real run would pull from yfinance via quantlab).
FRED_ASSETS = {
    "EQ":   "SP500",                # S&P 500 level (FRED, ~10y history only -- a known limitation)
    "BOND": "DGS10",                # 10y Treasury yield (proxy; real run uses a total-return bond index)
}


def _fred_csv(series_id: str, timeout: int = 8) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted URL)
        df = pd.read_csv(io.StringIO(r.read().decode()))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fetch_macro(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, timeout: int = 8,
                retries: int = 2) -> dict:
    """Return ``{'macro': DataFrame, 'assets': DataFrame}`` of monthly macro state + asset proxies,
    cache-first. Returns ``{}`` on a cache-miss when the fetch is unavailable.

    Reads a cached ``barometer_macro.parquet`` if present. Otherwise, only if ``fetch=True``, it *would*
    download the FRED macro series (growth: INDPRO/PAYEMS; inflation: CPIAUCSL/T10YIE) and the asset
    proxies, align them to month-end, cache and return.

    **Honest network note.** In this sandbox the daily FRED rate/breakeven series (``T10YIE``, ``DGS10``)
    reliably **time out**, and even the small monthly ``CPIAUCSL`` succeeds only intermittently. To keep
    the verdict from depending on a flaky fetch, this *tries a single small series* (``CPIAUCSL``) with a
    short timeout and a couple of retries; if that one succeeds it is cached for provenance, but the full
    panel still needs the daily series, so a partial result is treated as a cache-miss and ``{}`` is
    returned. The synthetic core (:func:`synthetic_macro`) is the validated offline proof meanwhile — see
    ``docs/results.md``. This mirrors Study 27 (Steamroller)'s pending-fetch pattern exactly.
    """
    cache = os.path.join(cache_dir, "barometer_macro.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        asset_cols = [c for c in df.columns if c.startswith("asset_")]
        if macro_cols and asset_cols:
            macro = df[macro_cols].rename(columns=lambda c: c[6:])
            assets = df[asset_cols].rename(columns=lambda c: c[6:])
            return {"macro": macro, "assets": assets}
        return {}
    if not fetch:
        return {}

    # Best-effort: try ONE small monthly series so a transient success is captured for provenance, but
    # never block the verdict on it. The full panel needs the daily series, which time out here.
    got: dict[str, pd.Series] = {}
    for attempt in range(retries + 1):
        try:
            got["infl_cpi"] = _fred_csv(FRED_MACRO["infl_cpi"], timeout=timeout).resample("ME").last()
            break
        except Exception:
            continue
    if got:
        # Cache the one small series for provenance, but it is NOT a usable panel on its own.
        os.makedirs(cache_dir, exist_ok=True)
        probe = pd.DataFrame(got).add_prefix("probe_")
        probe.to_parquet(os.path.join(cache_dir, "barometer_cpi_probe.parquet"))
    # The full macro+asset panel is unavailable in this environment → treated as a cache-miss.
    return {}
