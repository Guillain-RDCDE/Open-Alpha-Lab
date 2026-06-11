"""Data for the macro-momentum study — an offline synthetic cross-asset world driven by latent macro
state, and the REAL cross-asset book read from the desk's cached macro + ETF tape.

Macro momentum is a slow, cross-asset premium: the *trend* in fundamental macro data (growth, inflation)
predicts the next stretch of asset returns. Improving growth lifts pro-cyclical assets (equities,
commodities); rising inflation favours *real* assets (commodities, a TIPS/gold proxy) over *nominal*
bonds. The signal is the **change** in the (slow, persistent, regime-switching) macro state — not its
level — and it must be lagged so it is causally tradable.

The desk's offline/cache split:

  * :func:`synthetic_macro` — fully **offline, deterministic**. A small cross-asset panel (equities,
    nominal bonds, commodities, a real-asset/TIPS proxy, gold) is driven by two latent macro state
    variables — *growth* and *inflation* — each a persistent, regime-switching process. The **momentum**
    (one-period change) of each macro driver predicts next-period asset returns through fixed, signed
    betas, and rising-inflation regimes tilt the world toward real assets (the positive control).
    ``macro_strength`` sets how strongly macro momentum drives returns; ``macro_strength = 0`` is the
    **null** (assets are pure noise — no macro predictability). Deterministic given ``seed``. The
    synthetic tests depend on this and on ``ASSETS`` / ``ASSET_ORDER``.
  * :func:`fetch_macro` — the REAL book, **cache-first and offline**. It reads three cached parquets the
    desk pre-fetched — ``macro_us.parquet`` (BLS CPI level, industrial-production index, unemployment
    rate, monthly), ``us_treasury_yields.parquet`` (^IRX/^FVX/^TNX/^TYX daily yields in percent, the
    ``y10y − y3m`` slope a clean daily growth proxy), and ``cross_asset_etfs.parquet`` (18 liquid ETFs,
    adjusted close) — and returns an aligned **monthly** ``(asset-returns, macro-state)`` pair.

Two data choices, stated up front.

  * **Frequency = monthly.** Macro releases are monthly and the signal is a slow trend, so monthly is the
    right horizon. The yield-curve slope (daily) is sampled to month-end.
  * **Publication lag.** Macro data is released *after* the month it describes (CPI ~2 weeks, industrial
    production ~2 weeks), so every macro driver is lagged **one month** (``PUBLICATION_LAG_M``) before it
    can inform a position — a strict, causal floor on the release delay. The Treasury slope is observable
    in real time but is lagged the same one month for a single clean alignment.

**Sample span.** The real-asset ETFs that make the inflation hedge tradable only exist from the mid-2000s
(DBC 2006-02, DBA/UUP/HYG 2007, SLV 2006, USO 2006, TIP 2003, GLD 2004), so the real book begins ~**2007**.
The right edge is set by the macro series: the cached CPI runs to 2025-01 (industrial production only to
2023-11, so it is reported as a robustness cross-check, not the primary growth driver), so after the
one-month publication lag the book spans roughly **2007-02 → 2025-02** (~18 years, ~217 months). That is a
**short, post-2007 sample** — long enough to test the mechanism, too short to nail the magnitude.
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
PUBLICATION_LAG_M = 1   # macro data is released a month late → lag every driver by one month (causal)

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
# Real tape — cached macro (BLS CPI / industrial production) + Treasury yields + 18 cross-asset ETFs.
# Cache-first and fully offline: reads three parquets the desk pre-fetched. No network.
# --------------------------------------------------------------------------- #

# The 18 cached cross-asset ETFs and their macro priors. Each carries a sign on growth-momentum and on
# inflation-momentum, and a *real-asset* flag (the inflation hedge overweights these when inflation
# rises). Priors, not fitted betas — the book never sees in-sample returns to set these:
#   growth +  → pro-cyclical (equities, EM, REITs, HY, commodities);  growth − → long-duration nominal bonds
#   infl   +  → real asset (commodities, energy, ags, metals, TIPS, REITs);  infl − → nominal bonds/credit
REAL_ASSETS: dict[str, dict] = {
    "SPY": {"g_beta": +1.0, "i_beta": -0.2, "real": False},  # US large cap
    "QQQ": {"g_beta": +1.0, "i_beta": -0.3, "real": False},  # US tech (long-duration equity)
    "IWM": {"g_beta": +1.0, "i_beta": -0.2, "real": False},  # US small cap
    "EFA": {"g_beta": +0.8, "i_beta": -0.1, "real": False},  # developed-ex-US equity
    "EEM": {"g_beta": +1.0, "i_beta": +0.2, "real": False},  # EM equity (commodity-linked)
    "VNQ": {"g_beta": +0.6, "i_beta": +0.4, "real": True},   # REITs (real, pro-cyclical)
    "HYG": {"g_beta": +0.6, "i_beta": -0.2, "real": False},  # high-yield credit
    "LQD": {"g_beta": +0.1, "i_beta": -0.6, "real": False},  # investment-grade credit (rate-sensitive)
    "TLT": {"g_beta": -0.7, "i_beta": -1.0, "real": False},  # long Treasuries (the nominal-bond loser)
    "IEF": {"g_beta": -0.5, "i_beta": -0.7, "real": False},  # 7-10y Treasuries
    "SHY": {"g_beta": -0.2, "i_beta": -0.3, "real": False},  # 1-3y Treasuries
    "TIP": {"g_beta": -0.1, "i_beta": +0.7, "real": True},   # TIPS (inflation-linked)
    "GLD": {"g_beta": -0.1, "i_beta": +0.8, "real": True},   # gold
    "SLV": {"g_beta": +0.3, "i_beta": +0.7, "real": True},   # silver
    "DBC": {"g_beta": +0.7, "i_beta": +1.0, "real": True},   # broad commodities
    "USO": {"g_beta": +0.6, "i_beta": +0.9, "real": True},   # crude oil
    "DBA": {"g_beta": +0.3, "i_beta": +0.7, "real": True},   # agriculture
    "UUP": {"g_beta": -0.2, "i_beta": -0.3, "real": False},  # US dollar (rises when risk/infl falls)
}
REAL_ASSET_ORDER = list(REAL_ASSETS.keys())

# Sample floor: the real-asset ETFs that make the inflation hedge tradable only exist from ~2007.
REAL_SAMPLE_START = "2007-02-28"

CACHE_MACRO = "macro_us.parquet"            # index month-end; cols [cpi, indpro, unrate]
CACHE_YIELDS = "us_treasury_yields.parquet"  # index daily; cols [y3m, y5y, y10y, y30y] in percent
CACHE_ETFS = "cross_asset_etfs.parquet"      # index daily; 18 ETF adjusted closes


def _macro_state(cache_dir: str, lag_m: int = PUBLICATION_LAG_M) -> pd.DataFrame:
    """Monthly macro-state frame ``[growth, inflation, indpro_yoy]``, strictly lagged for publication.

    * ``inflation`` = CPI year-over-year (12-month % change of the BLS CPI level) — the inflation trend.
    * ``growth``    = the **yield-curve slope** ``y10y − y3m`` (in percent), sampled to month-end — a clean,
      real-time growth/macro proxy (a steeper curve ↦ stronger expected growth).
    * ``indpro_yoy`` = industrial-production YoY — a second growth read, reported as a robustness
      cross-check (the cached series only runs to 2023-11, so it is *not* the primary growth driver).

    Every column is lagged ``lag_m`` month(s): macro data is published after the month it describes, so a
    position formed at month-end can only use data through the prior month.
    """
    macro = pd.read_parquet(os.path.join(cache_dir, CACHE_MACRO))
    yields = pd.read_parquet(os.path.join(cache_dir, CACHE_YIELDS))

    cpi_yoy = macro["cpi"].pct_change(MONTHS_PER_YEAR)
    indpro_yoy = macro["indpro"].pct_change(MONTHS_PER_YEAR)
    slope = (yields["y10y"] - yields["y3m"]).resample("ME").last()

    state = pd.DataFrame({"growth": slope, "inflation": cpi_yoy, "indpro_yoy": indpro_yoy})
    state.index = state.index.to_period("M").to_timestamp(how="end")
    state = state[~state.index.duplicated(keep="last")].sort_index()
    return state.shift(lag_m)


def _asset_returns(cache_dir: str) -> pd.DataFrame:
    """Monthly returns of the 18 cross-asset ETFs (month-end adjusted-close % change)."""
    etfs = pd.read_parquet(os.path.join(cache_dir, CACHE_ETFS))[REAL_ASSET_ORDER]
    monthly = etfs.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp(how="end")
    return monthly.pct_change()


def fetch_macro(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Return ``{'macro': DataFrame, 'assets': DataFrame}`` — the REAL monthly macro state + ETF returns,
    aligned, lagged and sliced to the post-2007 real-asset sample. Cache-first and offline.

    Reads the three cached parquets (``macro_us``, ``us_treasury_yields``, ``cross_asset_etfs``), builds
    the lagged monthly macro state (CPI-YoY inflation, yield-curve-slope growth, IndPro-YoY robustness) and
    the 18-ETF return panel, and aligns them on the months where both the macro drivers and the real-asset
    ETFs exist (``REAL_SAMPLE_START`` onward). ``fetch`` is accepted for interface parity but unused — the
    tape is pre-cached, so the real run is fully offline. Returns ``{}`` if the cache parquets are absent.

    ``macro`` carries ``[growth, inflation, indpro_yoy]``; ``assets`` carries the 18 ETF monthly returns in
    :data:`REAL_ASSET_ORDER`. The publication lag is documented in this module's docstring.
    """
    paths = [os.path.join(cache_dir, f) for f in (CACHE_MACRO, CACHE_YIELDS, CACHE_ETFS)]
    if not all(os.path.exists(p) for p in paths):
        return {}

    macro = _macro_state(cache_dir)
    assets = _asset_returns(cache_dir)

    # Align: months where the primary macro drivers (growth, inflation) exist AND the ETFs trade.
    macro = macro.loc[REAL_SAMPLE_START:].dropna(subset=["growth", "inflation"])
    common = macro.index.intersection(assets.index)
    macro = macro.loc[common]
    assets = assets.loc[common].dropna(how="all")
    macro = macro.loc[assets.index]
    return {"macro": macro, "assets": assets}
