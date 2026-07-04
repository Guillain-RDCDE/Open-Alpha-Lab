"""Data layer for Study 638 — Value-Momentum-Everywhere.

Four asset-class *sleeves* of monthly returns, all offline-friendly once cached:

* **EQ — country equity ETFs** (yfinance, dividend-adjusted = total-return closes):
  SPY + 12 single-country iShares MSCI funds (Japan, Germany, UK, France, Italy, Spain,
  Australia, Canada, Hong Kong, Singapore, Sweden, Switzerland), listed since 1996.
  A *survivor* panel (every fund still trades in 2026) — named on the Signal axis.
* **FX — G10 spot pairs vs USD** (yfinance, price-only spot — **no carry**, labeled):
  EUR, GBP, AUD, NZD directly; JPY, CAD, CHF, SEK, NOK inverted from their USD/XXX quotes
  so every column is the return of holding the *foreign* currency against USD.
* **BOND — US Treasury futures** (repo futures cache): ZF (5y), ZN (10y), ZB (30y)
  daily *excess* returns compounded to monthly. Thin cross-section (3 assets) — the
  long/short is essentially a curve bet; named in the docs.
* **CMD — commodity futures** (repo futures cache): CL, GC, SI, HG, NG, ZC, ZS, ZW
  daily excess returns compounded to monthly.

Futures sleeves come from the shared repo cache
``_cache/_cache/trade_winds_futures.parquet`` (built once for the trade-winds studies);
we *copy* the needed columns into this study's own ``_cache/`` so the study stays
self-contained. Equity/FX dailies are fetched once from yfinance and cached the same way.

The **synthetic world** generator builds a multi-sleeve monthly panel month-by-month with
a TUNABLE planted cross-sectional value effect (``val_edge``: long-run losers outperform)
and momentum effect (``mom_edge``: recent winners keep winning), plus the ``edge = 0``
null. Deterministic, fixed seed, 480 months (40 years — far below the 250-year
ns-Timestamp ceiling; decorative ``period_range`` monthly index).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
EQ_CACHE = os.path.join(CACHE_DIR, "vme_eq_daily.parquet")
FX_CACHE = os.path.join(CACHE_DIR, "vme_fx_daily.parquet")
FUT_CACHE = os.path.join(CACHE_DIR, "vme_futures_daily.parquet")
# shared repo-root futures cache (read-only fallback used once to seed the study cache)
REPO_FUT = os.path.join(HERE, "..", "..", "..", "_cache", "_cache",
                        "trade_winds_futures.parquet")

# Last complete calendar month across ALL sleeves: the shared futures cache ends
# 2026-06-11, so June 2026 is partial and the panel is frozen at May 2026.
AS_OF = "2026-05-31"

# EQ sleeve — SPY + single-country iShares MSCI ETFs (all listed 1996 or earlier).
# Survivors: every fund still trades in 2026 (named on the Signal axis).
EQ_TICKERS = ["SPY", "EWJ", "EWG", "EWU", "EWQ", "EWI", "EWP",
              "EWA", "EWC", "EWH", "EWS", "EWD", "EWL"]

# FX sleeve — G10 vs USD. Direct XXX/USD quotes, plus USD/XXX quotes to invert.
FX_DIRECT = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"]     # already XXX/USD
FX_INVERT = ["JPY=X", "CAD=X", "CHF=X", "SEK=X", "NOK=X"]        # USD/XXX -> invert

BOND_COLS = ["ZF=F", "ZN=F", "ZB=F"]
CMD_COLS = ["CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZC=F", "ZS=F", "ZW=F"]

SLEEVES = ["EQ", "FX", "BOND", "CMD"]


# --------------------------------------------------------------------------- #
# Fetch (network — used ONCE to build the study cache)
# --------------------------------------------------------------------------- #
def _yf_close(tickers: list[str], start: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance download failed for {tickers}")
    return raw.dropna(how="all").sort_index()


def fetch_all(start: str = "1996-01-01") -> None:
    """Build the three study caches (network-only; never imported by offline cells)."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    eq = _yf_close(EQ_TICKERS, start=start)[EQ_TICKERS]
    eq.to_parquet(EQ_CACHE)

    fx = _yf_close(FX_DIRECT + FX_INVERT, start=start)
    fx.to_parquet(FX_CACHE)

    if os.path.exists(REPO_FUT):
        fut = pd.read_parquet(REPO_FUT)[BOND_COLS + CMD_COLS]
    else:  # rebuild from yfinance continuous front-month tickers (price returns)
        px = _yf_close(BOND_COLS + CMD_COLS, start="2000-01-01")
        fut = px.pct_change()
    fut.to_parquet(FUT_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (EQ_CACHE, FX_CACHE, FUT_CACHE))


# --------------------------------------------------------------------------- #
# Monthly sleeve panels (offline, cache-first)
# --------------------------------------------------------------------------- #
def _monthly_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end-to-month-end simple returns; each column's first month dropped."""
    m = px.resample("ME").last()
    return m.pct_change()


def _monthly_from_daily_returns(r: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns to calendar months; a column's first (possibly partial)
    month is dropped so no month is built from a truncated window."""
    m = (1.0 + r).groupby(pd.Grouper(freq="ME")).prod(min_count=1) - 1.0
    for c in m.columns:
        first = m[c].first_valid_index()
        if first is not None:
            m.loc[first, c] = np.nan
    return m


def load_sleeves(as_of: str | None = AS_OF) -> dict[str, pd.DataFrame]:
    """The four monthly-return sleeve panels, sliced to the frozen as-of month-end.

    EQ is total-return (dividend-adjusted); FX is price-only spot (no carry — labeled);
    BOND and CMD are futures *excess* returns. Every panel's index is month-end.
    """
    eq_px = pd.read_parquet(EQ_CACHE)
    fx_px = pd.read_parquet(FX_CACHE)
    fut = pd.read_parquet(FUT_CACHE)

    eq = _monthly_from_prices(eq_px)

    # invert USD/XXX quotes so every FX column is the foreign currency's return vs USD
    fx_cols = {}
    for t in FX_DIRECT:
        if t in fx_px.columns:
            fx_cols[t.replace("USD=X", "")] = fx_px[t]
    for t in FX_INVERT:
        if t in fx_px.columns:
            fx_cols[t.replace("=X", "")] = 1.0 / fx_px[t]
    fx = _monthly_from_prices(pd.DataFrame(fx_cols))

    bond = _monthly_from_daily_returns(fut[BOND_COLS])
    cmd = _monthly_from_daily_returns(fut[CMD_COLS])

    out = {"EQ": eq, "FX": fx, "BOND": bond, "CMD": cmd}
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        out = {k: v[v.index <= cut] for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — the machinery control
# --------------------------------------------------------------------------- #
def synthetic_world(n_sleeves: int = 4, n_assets: int = 8, n_months: int = 480,
                    val_edge: float = 0.0, mom_edge: float = 0.0,
                    seed: int = 638, sig_m: float = 0.04) -> dict[str, pd.DataFrame]:
    """Deterministic multi-sleeve monthly panel with PLANTED value & momentum effects.

    Generated month-by-month: each month every asset's return is idiosyncratic noise
    plus a sleeve-common factor, *plus* ``val_edge`` times the cross-sectional z-score
    of its value signal (negative of its month t-59..t-12 cumulative return) and
    ``mom_edge`` times the z-score of its momentum signal (month t-11..t-1 cumulative
    return, skipping the current month). With both edges zero the world is a pure
    random walk: the pipeline must NOT manufacture a significant VAL/MOM/COMBO.
    The 480-month (40-year) span stays far below the 250-year ns-Timestamp ceiling;
    the decorative monthly index comes from ``period_range``.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("1980-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    world = {}
    for s in range(n_sleeves):
        r = np.zeros((n_months, n_assets))
        for t in range(n_months):
            common = rng.normal(0.0, sig_m * 0.5)
            eps = rng.normal(0.0, sig_m, n_assets)
            extra = np.zeros(n_assets)
            if t >= 60 and (val_edge != 0.0 or mom_edge != 0.0):
                cum = np.log1p(r[:t]).cumsum(axis=0)
                valsig = -(cum[t - 12] - cum[t - 60])      # 5y reversal, skip last 12m
                momsig = cum[t - 1] - cum[t - 12]          # 12-1 momentum
                for sig, edge in ((valsig, val_edge), (momsig, mom_edge)):
                    sd = sig.std()
                    if sd > 0:
                        extra += edge * (sig - sig.mean()) / sd
            r[t] = common + eps + extra
        cols = [f"S{s}A{i:02d}" for i in range(n_assets)]
        world[f"SYN{s}"] = pd.DataFrame(r, index=idx, columns=cols)
    return world
