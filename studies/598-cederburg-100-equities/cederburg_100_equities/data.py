"""Data layer for Study 598 (Cederburg "100% Equities for Life").

Three monthly real-return series, one shape — ``['US', 'INTL', 'BD']``:

- ``US`` — Shiller S&P 500 real total return (real price change + trailing
  dividend yield, one-month lag on the dividend — no look-ahead). Cache-first:
  the study's own ``_cache/shiller_sp500.parquet``, then the repo-root staged
  copy, then a one-time refetch from the GitHub raw mirror.
- ``BD`` — 10-year Treasury first-order approximation (carry − 7 × Δy),
  **CPI-deflated** so every leg is real (same construction as sibling
  study 596, stated as a decision).
- ``INTL`` — the honest compromise this study is explicit about, because no
  public 150-year international monthly tape exists:

  * **EFA era (2001-09 → tape end): market data.** iShares MSCI EAFE (EFA)
    monthly total return (yfinance auto-adjusted closes), deflated by realised
    US CPI inflation from the Shiller panel. Cached once to
    ``_cache/efa_monthly.csv``.
  * **Pre-EFA (1871-02 → 2001-08): literature-calibrated, NOT market data.**
    A deterministic (seed 598) monthly log-return series built as
    ``mu_intl + beta * (us_t − mean(us)) + resid`` with the three calibration
    constants below taken from the Dimson-Marsh-Staunton (DMS) Global Returns
    yearbook: world-ex-US real equity geometric return ≈ **4.3 %/yr**,
    volatility ≈ **17 %/yr**, correlation with US ≈ **0.60**. This is a
    *simulation calibrated to the literature* — it is labeled as such in every
    table it touches, and the study always reports a pure-domestic variant
    (``intl_mode='domestic'``) that uses **only** market data, plus a
    ``corr1`` stress variant that kills the diversification benefit entirely.

- ``synthetic_world`` — a deterministic, offline annual three-asset generator
  whose planted knob is the **arithmetic equity premium over bonds**
  (``premium``). With ``premium = 0`` the expected terminal wealth of any
  weight schedule is identical (products of independent gross returns with
  equal arithmetic means), so the all-equity-vs-60/40 detector must read ~0:
  the exact null. Synthetic spans stay <= 250 years (ns-Timestamp CI trap).
"""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(STUDY_ROOT, "..", ".."))
STUDY_CACHE = os.path.join(STUDY_ROOT, "_cache")
SHILLER_PATH = os.path.join(STUDY_CACHE, "shiller_sp500.parquet")
EFA_PATH = os.path.join(STUDY_CACHE, "efa_monthly.csv")

_FALLBACKS = [
    os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet"),
    os.path.join(REPO_ROOT, "_cache", "_cache", "shiller_sp500.parquet"),
]

SHILLER_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
)

MOD_DURATION = 7.0   # approximate modified duration of a 10-year Treasury

# ---- literature calibration of the pre-EFA international leg (DMS yearbook) --
INTL_REAL_GEO = 0.043   # world-ex-US real equity, geometric, ~1900-2022
INTL_VOL = 0.17         # annualised volatility of world-ex-US real equity
CORR_US_INTL = 0.60     # long-run monthly correlation with US equity
INTL_SEED = 598         # fixed seed for the calibrated pre-EFA residuals


# ---------------------------------------------------------------------------
# Shiller tape — cache-first
# ---------------------------------------------------------------------------

def load_shiller(cache_path: str = SHILLER_PATH) -> pd.DataFrame:
    """Raw Shiller monthly frame, cache-first (study cache -> repo cache -> web)."""
    if os.path.exists(cache_path):
        raw = pd.read_parquet(cache_path)
    else:
        raw = None
        for fb in _FALLBACKS:
            if os.path.exists(fb):
                raw = pd.read_parquet(fb)
                break
        if raw is None:  # pragma: no cover — one-time refetch
            import urllib.request

            req = urllib.request.Request(
                SHILLER_URL,
                headers={"User-Agent": "OpenAlphaLab research desk "
                                        "guillain@poulpe.us"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = pd.read_csv(io.BytesIO(r.read()))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw.to_parquet(cache_path)
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw.sort_values("Date").set_index("Date")


# ---------------------------------------------------------------------------
# EFA tape — one-time yfinance fetch, then cache-first
# ---------------------------------------------------------------------------

def fetch_efa(cache_path: str = EFA_PATH) -> pd.DataFrame:
    """One-time yfinance fetch of EFA monthly total-return closes (network)."""
    import yfinance as yf  # pragma: no cover — network path

    px = yf.download("EFA", start="2001-01-01", auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    m = px.resample("ME").last().dropna()
    out = m.to_frame("EFA")
    out.index.name = "Date"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_csv(cache_path)
    return out


def load_efa(cache_path: str = EFA_PATH) -> pd.DataFrame:
    """Monthly EFA total-return closes, cache-first."""
    if not os.path.exists(cache_path):  # pragma: no cover — one-time fetch
        return fetch_efa(cache_path)
    out = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    out.index.name = "Date"
    return out


# ---------------------------------------------------------------------------
# The joint monthly real-return tape
# ---------------------------------------------------------------------------

def real_returns(
    intl_mode: str = "hybrid",
    start: str = "1871-02-01",
    end: str | None = None,
    cache_path: str = SHILLER_PATH,
    efa_path: str = EFA_PATH,
    seed: int = INTL_SEED,
) -> pd.DataFrame:
    """Monthly real returns ``['US', 'INTL', 'BD']`` plus ``intl_is_market``.

    ``intl_mode``:

    - ``'hybrid'`` (default) — EFA-era market data + literature-calibrated
      pre-EFA leg (deterministic, seed 598; corr 0.60, vol 17 %, geo 4.3 %/yr).
    - ``'domestic'`` — ``INTL = US`` (the pure-tape variant: a 50/50 blend
      degenerates to 100 % domestic; nothing simulated anywhere).
    - ``'corr1'`` — pre-EFA leg perfectly correlated with US (same log
      deviations, shifted to the DMS mean): the stress case with **zero**
      pre-EFA diversification benefit.

    The frame carries a boolean column ``intl_is_market`` naming, month by
    month, whether the INTL entry is market data (EFA era / domestic mode) or
    the literature-calibrated simulation.
    """
    raw = load_shiller(cache_path)
    ok = (
        (raw["SP500"] > 0)
        & (raw["Real Price"] > 0)
        & (raw["Long Interest Rate"] > 0)
        & (raw["Dividend"] > 0)
        & (raw["Consumer Price Index"] > 0)
    )
    raw = raw[ok]

    div_yield_monthly = raw["Dividend"] / raw["SP500"] / 12.0
    us = raw["Real Price"].pct_change() + div_yield_monthly.shift(1)

    y = raw["Long Interest Rate"] / 100.0
    bd_nominal = y.shift(1) / 12.0 - MOD_DURATION * y.diff()
    infl = raw["Consumer Price Index"].pct_change()
    bd = (1.0 + bd_nominal) / (1.0 + infl) - 1.0

    df = pd.DataFrame({"US": us, "BD": bd}).dropna()
    df["infl"] = infl.reindex(df.index)

    if intl_mode == "domestic":
        df["INTL"] = df["US"]
        df["intl_is_market"] = True
    else:
        # EFA era — market data, CPI-deflated
        efa = load_efa(efa_path)["EFA"]
        efa_ret = efa.pct_change().dropna()
        # align EFA month-end returns onto the Shiller month-start grid
        efa_ret.index = efa_ret.index.to_period("M").to_timestamp()
        efa_real = ((1.0 + efa_ret) / (1.0 + df["infl"]) - 1.0).reindex(df.index)

        is_market = efa_real.notna()
        pre = ~is_market

        # Pre-EFA leg — literature-calibrated (DMS), deterministic seed
        lus = np.log1p(df.loc[pre, "US"].to_numpy())
        mu_l = np.log1p(INTL_REAL_GEO) / 12.0
        sd_l = INTL_VOL / np.sqrt(12.0)
        dev = lus - lus.mean()
        if intl_mode == "corr1":
            l_intl = mu_l + dev
        elif intl_mode == "hybrid":
            beta = CORR_US_INTL * sd_l / lus.std(ddof=0)
            resid_sd = sd_l * np.sqrt(1.0 - CORR_US_INTL ** 2)
            rng = np.random.default_rng(seed)
            l_intl = mu_l + beta * dev + resid_sd * rng.standard_normal(pre.sum())
        else:
            raise ValueError(f"unknown intl_mode {intl_mode!r}")

        intl = efa_real.copy()
        intl.loc[pre] = np.expm1(l_intl)
        df["INTL"] = intl
        df["intl_is_market"] = is_market

    df = df[["US", "INTL", "BD", "intl_is_market"]]
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    df.index.name = "Date"
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a return frame, for the as-of stamp."""
    h = hashlib.sha1()
    for c in ("US", "INTL", "BD"):
        h.update(c.encode())
        h.update(np.ascontiguousarray(df[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline generator (annual returns)
# ---------------------------------------------------------------------------

def synthetic_world(
    n_years: int = 200,
    premium: float = 0.045,      # ARITHMETIC equity premium over bonds (both legs)
    bd_mu: float = 0.020,        # arithmetic annual real bond return
    eq_vol: float = 0.17,
    intl_vol: float = 0.17,
    bd_vol: float = 0.07,
    corr_ui: float = 0.60,       # US-INTL correlation
    corr_eb: float = 0.10,       # equity-bond correlation
    seed: int = 598,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible **annual** three-asset real-return world.

    Simple returns are normal with **arithmetic** means ``bd_mu + premium``
    (both equity legs) and ``bd_mu`` (bonds). Because expected terminal wealth
    of any rebalanced schedule is a function of arithmetic means only (products
    of independent gross returns), ``premium = 0`` is the *exact null* for the
    all-equity-vs-60/40 mean-terminal-wealth detector.

    Returns ``(frame, truth)``; ``frame`` has columns ``['US', 'INTL', 'BD']``
    of annual simple real returns on a decorative ``period_range`` (<= 250 y).
    """
    if n_years > 250:
        raise ValueError("keep synthetic spans <= 250 years (ns-Timestamp trap)")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_years, 3))
    z_us = z[:, 0]
    z_in = corr_ui * z[:, 0] + np.sqrt(1.0 - corr_ui ** 2) * z[:, 1]
    z_bd = corr_eb * z[:, 0] + np.sqrt(1.0 - corr_eb ** 2) * z[:, 2]

    eq_mu = bd_mu + premium
    us = eq_mu + eq_vol * z_us
    intl = eq_mu + intl_vol * z_in
    bd = bd_mu + bd_vol * z_bd

    idx = pd.period_range("1900", periods=n_years, freq="Y")
    frame = pd.DataFrame({"US": us, "INTL": intl, "BD": bd}, index=idx)
    truth = dict(n_years=n_years, premium=premium, bd_mu=bd_mu, eq_vol=eq_vol,
                 intl_vol=intl_vol, bd_vol=bd_vol, corr_ui=corr_ui,
                 corr_eb=corr_eb, seed=seed)
    return frame, truth
