"""Data layer for Study 641 — Sell in May (the Halloween indicator).

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily ^GSPC (S&P 500 index, **price-only** — no dividends, named honestly),
  daily SPY (total-return-ish via yfinance's split/dividend adjustment) and, where available,
  the ^SP500TR total-return index (inception 1988), all from yfinance (no key), cached as CSV
  under the study's own ``_cache/``. A 13-week T-bill yield proxy (^IRX) supplies the "go to
  cash" leg of the Halloween timer.

* **The calendar split, hardcoded as a fact, not fit.** Nov→Apr = "winter" (the Halloween
  indicator's *in* half), May→Oct = "summer" (the *out* half). This is a fixed calendar
  partition, known in full years ahead — a scheduled rule, not a discovered one, so entering
  either half involves zero look-ahead (see ``strategy.py`` for the one documented convention).

* **Synthetic world.** A deterministic, seeded monthly log-return series with a TUNABLE planted
  winter premium (knob ``premium_bp``): winter months earn an extra ``premium_bp`` of drift.
  ``premium_bp = 0`` is the null world — winter and summer months statistically identical; the
  Welch/HAC machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
GSPC_CACHE = os.path.join(CACHE_DIR, "sim_gspc.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "sim_spy.csv")
TR_CACHE = os.path.join(CACHE_DIR, "sim_sp500tr.csv")
IRX_CACHE = os.path.join(CACHE_DIR, "sim_irx.csv")

START = "1950-01-03"          # ^GSPC's effective start on Yahoo
SPY_START = "1993-02-01"      # SPY inception 1993-01-29
TR_START = "1988-01-05"       # ^SP500TR inception on Yahoo
IRX_START = "1960-01-04"      # ^IRX (13-week T-bill discount yield) history on Yahoo
AS_OF = "2026-06-30"          # last complete month at publication (2026-07-10)

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}   # Nov -> Apr, the Halloween indicator's "in" half
SUMMER_MONTHS = {5, 6, 7, 8, 9, 10}    # May -> Oct, the "go away" half


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(end: str = "2026-07-01") -> None:
    """Download ^GSPC, SPY, ^SP500TR (best-effort) and ^IRX; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    def _dl(ticker, start, auto_adjust):
        raw = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    # ^GSPC — PRICE-ONLY index (no dividends baked in). Named honestly throughout.
    gspc = _dl("^GSPC", START, auto_adjust=False)
    gspc[["Close"]].dropna().to_csv(GSPC_CACHE)

    # SPY — dividend/split-adjusted close (auto_adjust=True): a real total-return proxy.
    spy = _dl("SPY", SPY_START, auto_adjust=True)
    spy[["Close"]].dropna().to_csv(SPY_CACHE)

    # ^SP500TR — the genuine total-return index, since 1988. Best-effort: some yfinance/Yahoo
    # configurations don't serve it; if the pull comes back empty we simply don't cache it and
    # ``have_tr()`` reports it absent (the study still runs on ^GSPC price-only + SPY).
    try:
        tr = _dl("^SP500TR", TR_START, auto_adjust=False)
        tr = tr[["Close"]].dropna()
        if len(tr) > 100:
            tr.to_csv(TR_CACHE)
    except Exception:
        pass

    # ^IRX — 13-week T-bill discount yield, quoted in percent (e.g. 5.00 = 5%/yr). The cash leg
    # of the Halloween timer.
    irx = _dl("^IRX", IRX_START, auto_adjust=False)
    irx[["Close"]].dropna().to_csv(IRX_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (GSPC_CACHE, SPY_CACHE, IRX_CACHE))


def have_tr() -> bool:
    return os.path.exists(TR_CACHE)


def _load_close(path: str, start: str, asof: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df["Close"]
    return s.loc[(s.index >= start) & (s.index <= asof)].copy()


def load_real(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Raw daily ``Close`` frames, asof-sliced — for the data-stamp / fingerprint banner
    (mirrors sibling studies: the fingerprint is taken on the RAW tape, before any
    resample/log-return transform)."""
    out = {
        "gspc": pd.DataFrame({"Close": _load_close(GSPC_CACHE, start, asof)}),
        "spy": pd.DataFrame({"Close": _load_close(SPY_CACHE, max(start, SPY_START), asof)}),
        "irx": pd.DataFrame({"Close": _load_close(IRX_CACHE, max(start, IRX_START), asof)}),
    }
    if have_tr():
        out["sp500tr"] = pd.DataFrame({"Close": _load_close(TR_CACHE, max(start, TR_START), asof)})
    return out


def load_monthly(start: str = START, asof: str = AS_OF) -> dict[str, pd.Series]:
    """Month-end log-return series for ^GSPC, SPY, ^SP500TR (if cached) and the ^IRX cash rate.

    Each equity series is resampled to the last trading close of the month, then log-differenced
    (one return per completed calendar month — the natural unit for a Nov->Apr / May->Oct split).
    ^IRX is resampled to its month-end level (percent, annualized) — the cash-rate INPUT, not a
    return; ``strategy.py`` turns it into a monthly cash return.
    """
    out: dict[str, pd.Series] = {}
    gspc = _load_close(GSPC_CACHE, start, asof).resample("ME").last()
    out["gspc"] = np.log(gspc).diff().dropna()
    spy = _load_close(SPY_CACHE, max(start, SPY_START), asof).resample("ME").last()
    out["spy"] = np.log(spy).diff().dropna()
    if have_tr():
        tr = _load_close(TR_CACHE, max(start, TR_START), asof).resample("ME").last()
        out["sp500tr"] = np.log(tr).diff().dropna()
    irx = _load_close(IRX_CACHE, max(start, IRX_START), asof).resample("ME").last()
    out["irx_pct"] = irx.dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted winter premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(premium_bp: float = 0.0, seed: int = 641,
                    n_years: int = 76, base_bp: float = 60.0, vol_ann: float = 0.15,
                    ) -> pd.Series:
    """Deterministic monthly log-return series with a TUNABLE planted winter premium.

    Every month gets ``base_bp`` of drift; Nov->Apr months get an EXTRA ``premium_bp`` (in
    basis points of monthly log return). ``premium_bp = 0`` is the null world: winter and
    summer are statistically identical, and the Welch/HAC split must NOT reach significance.

    ``n_years`` * 12 monthly points (default ~912) is far below the >~3000-point Timestamp
    trap and the span (~76 years) is far below the ~250-year ns-timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1950-01-31", periods=n, freq="ME")
    winter = pd.Series(idx.month, index=idx).isin(WINTER_MONTHS).astype(float)
    drift = base_bp * 1e-4 + (premium_bp * 1e-4) * winter
    shocks = rng.normal(0.0, vol_ann / np.sqrt(12), n)
    return pd.Series(drift + shocks, index=idx, name="ret")
