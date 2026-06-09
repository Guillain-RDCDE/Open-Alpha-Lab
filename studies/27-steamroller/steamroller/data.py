"""Data access for the carry-trade study — the rates, the FX, and where they come from.

The carry trade harvests the *interest-rate differential*: borrow a low-rate currency, lend a high-rate
one, and pocket the gap — betting that the high-rate currency won't depreciate enough to wipe it out
(i.e. that uncovered interest-rate parity, UIRP, fails). So the tape is a panel of currencies, each with
a short rate and a spot FX level, and the data layer keeps the desk's offline/cache split:

    * :func:`synthetic_carry` — fully **offline**. A toy G10: each currency has a fixed short rate; the
      spot moves so that, *most* of the time, the high-rate currencies do **not** depreciate enough to
      offset the rate gap (the baked carry premium), but in occasional **risk-off** events they crash
      together (the "steamroller" — carry's fat negative tail). ``carry_strength`` sets how much of UIRP
      fails (the premium); ``crash_size`` the depth of the risk-off crashes. ``carry_strength = 0`` is
      the **null**: full UIRP, spot exactly offsets the rate gap, no premium. Deterministic given ``seed``.
    * :func:`fetch_carry` — pull real **G10 short rates and FX spot from FRED** (CSV download, no API
      key). **Cache-only** unless ``fetch=True``. *Network note:* this requires internet; in a fully
      offline environment the real run is skipped and the synthetic core stands alone.

Data choice, named up front: monthly 3-month interbank rates and end-of-month FX from FRED — the carry
signal is a slow, rate-differential signal, so a monthly frequency is the right (and cleanly available)
horizon, and using one public source keeps rates and FX internally consistent.
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


@dataclass(frozen=True)
class CarryTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_ccy: int
    n_months: int
    carry_strength: float     # fraction of UIRP that fails (the premium); 0 == full UIRP null
    crash_size: float

    @property
    def has_premium(self) -> bool:
        return self.carry_strength != 0.0


def synthetic_carry(n_ccy: int = 9, n_months: int = 600, carry_strength: float = 0.9,
                    crash_size: float = 0.05, crash_prob: float = 0.012, fx_vol: float = 0.016,
                    crash_persist: float = 0.85, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, CarryTruth]:
    """A toy G10 where high-rate currencies earn a carry premium — punctuated by joint risk-off crashes.

    For currency ``i`` with monthly rate ``r_i`` (fixed, spread across the panel) and base-average rate
    ``r̄``, the monthly **excess return** of holding it (funded at the average) is::

        crash_t  ~ Bernoulli(crash_prob)                      (a shared risk-off month)
        ds_i     = -(1 - carry_strength)*(r_i - r̄)            (partial UIRP: high rates depreciate, but not fully)
                   + fx_vol*eps_i  - crash_t*crash_size*(r_i - r̄)/scale   (high-carry crash hardest in risk-off)
        xret_i   = (r_i - r̄) + ds_i

    With ``carry_strength > 0`` the high-rate currencies keep part of the rate gap (the premium); the
    risk-off crashes give the strategy its negative skew. ``carry_strength = 0`` makes ``ds_i`` exactly
    offset the rate gap (full UIRP) — no premium. Returns ``(excess_returns, rates, truth)`` as monthly
    ``months x currency`` frames. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1995-01", periods=n_months, freq="M").to_timestamp(how="end")
    idx.name = "date"
    rates_ann = np.linspace(0.00, 0.12, n_ccy)
    rng.shuffle(rates_ann)
    r = rates_ann / MONTHS_PER_YEAR                      # monthly
    rbar = r.mean()
    dr = r - rbar                                        # rate gap vs average
    scale = np.abs(dr).max() if np.abs(dr).max() > 0 else 1.0

    # risk-off as a *sticky* two-state Markov chain (calm/crash), so crashes cluster into multi-month
    # episodes (1998, 2008) and produce realistic carry drawdowns rather than isolated bad months.
    crash = np.zeros(n_months)
    state = 0
    for t in range(n_months):
        p_enter, p_stay = crash_prob, crash_persist
        if rng.random() < (p_stay if state == 1 else p_enter):
            state = 1
        else:
            state = 0
        crash[t] = float(state)
    eps = rng.standard_normal((n_months, n_ccy))
    # the crash is part of the *carry trade's* risk, so it scales with carry_strength: at full UIRP
    # (carry_strength=0) there is neither premium nor crash, and the carry book earns ~0 (a clean null).
    ds = (-(1.0 - carry_strength) * dr[None, :]
          + fx_vol * eps
          - carry_strength * crash[:, None] * crash_size * (dr[None, :] / scale))
    xret = dr[None, :] + ds
    cols = [f"C{i:01d}" for i in range(n_ccy)]
    xr = pd.DataFrame(xret, index=idx, columns=cols)
    rates = pd.DataFrame(np.tile(r, (n_months, 1)), index=idx, columns=cols)
    return xr, rates, CarryTruth(n_ccy=n_ccy, n_months=n_months, carry_strength=carry_strength, crash_size=crash_size)


# --------------------------------------------------------------------------- #
# Real tape — G10 short rates + FX spot from FRED (CSV, no API key). Network only behind fetch=True.
# --------------------------------------------------------------------------- #

# 3-month interbank rates (% p.a., monthly) and USD FX (FRED ids). FX as USD-per-unit where noted.
FRED_RATES = {  # ccy: 3-month or immediate interbank rate series
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N", "SEK": "IR3TIB01SEM156N", "NOK": "IR3TIB01NOM156N",
}
FRED_FX = {  # USD per 1 unit of foreign currency (so a rise = foreign appreciation). DEX* are mixed; handled below.
    "EUR": ("DEXUSEU", False), "GBP": ("DEXUSUK", False), "AUD": ("DEXUSAL", False),
    "JPY": ("DEXJPUS", True), "CAD": ("DEXCAUS", True), "CHF": ("DEXSZUS", True),
    "SEK": ("DEXSDUS", True), "NOK": ("DEXNOUS", True),
}


def _fred_csv(series_id: str, timeout: int = 30) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted URL)
        df = pd.read_csv(io.StringIO(r.read().decode()))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fetch_carry(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Return ``{'rates': DataFrame, 'fx': DataFrame}`` of monthly G10 rates and USD FX, cache-first.

    Reads a cached ``carry_g10.parquet`` if present; otherwise, only if ``fetch=True``, downloads the
    FRED series (rates + FX), aligns them to month-end, caches, and returns. **Requires network** when
    fetching; in an offline environment this returns ``{}`` and the study's real run is skipped (the
    synthetic core is the offline proof).
    """
    cache = os.path.join(cache_dir, "carry_g10.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        rate_cols = [c for c in df.columns if c.startswith("rate_")]
        fx_cols = [c for c in df.columns if c.startswith("fx_")]
        rates = df[rate_cols].rename(columns=lambda c: c[5:])
        fx = df[fx_cols].rename(columns=lambda c: c[3:])
        return {"rates": rates, "fx": fx}
    if not fetch:
        return {}
    rates = {}
    for ccy, sid in FRED_RATES.items():
        try:
            rates[ccy] = _fred_csv(sid).resample("ME").last()
        except Exception:
            continue
    fx = {}
    for ccy, (sid, invert) in FRED_FX.items():
        try:
            s = _fred_csv(sid).resample("ME").last()
            fx[ccy] = (1.0 / s) if invert else s          # to USD-per-foreign-unit
        except Exception:
            continue
    if not rates or not fx:
        return {}
    rates_df = pd.DataFrame(rates).sort_index()
    fx_df = pd.DataFrame(fx).sort_index()
    os.makedirs(cache_dir, exist_ok=True)
    out = pd.concat([rates_df.add_prefix("rate_"), fx_df.add_prefix("fx_")], axis=1).dropna(how="all")
    out.to_parquet(cache)
    return {"rates": rates_df, "fx": fx_df}
