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
    * :func:`fetch_carry` — the **real G10 tape**, served from the repo-wide cache: OECD MEI 3-month
      interbank short rates (via DBnomics) + yfinance FX spot, the *same two parquets* Study 36
      (Greenback) runs on, so the two FX studies share one tape and one fingerprint lineage.
      **Cache-first**; on a cache miss with ``fetch=True`` it calls the shared fetchers in
      [`tools/fetch_altdata.py`](../../../tools/fetch_altdata.py) (no duplicated download code). In a
      fully offline environment with no cache the real run is skipped and the synthetic core stands alone.

Data choice, named up front: monthly 3-month interbank rates (OECD MEI, % p.a.) and end-of-month FX
(yfinance, USD per 1 unit of foreign currency) — the carry signal is a slow, rate-differential signal, so
a monthly frequency is the right horizon. OECD's MEI series was discontinued at **2024-01**, so the
headline as-of is pinned there (:data:`DATA_AS_OF`) and the published numbers never creep.
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
# Real tape — OECD MEI 3-month short rates (DBnomics) + yfinance FX, the SAME repo-wide cache as
# Study 36 (Greenback). Cache-first; network only behind fetch=True, via tools/fetch_altdata.py.
# --------------------------------------------------------------------------- #

RATES_CACHE = "g10_short_rates.parquet"   # OECD 3-month interbank, % p.a., monthly (cols incl. USD, NZD).
FX_CACHE = "g10_fx.parquet"               # daily, USD per 1 unit of each foreign currency (yfinance).

# OECD MEI (the rates source) was discontinued at 2024-01 — pin the headline as-of to the data end,
# not the live calendar, so the published numbers never creep.
DATA_AS_OF = "2024-01-31"


def fetch_carry(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Return ``{'rates': DataFrame, 'fx': DataFrame}`` of monthly G10 rates and month-end USD FX,
    cache-first.

    Reads ``_cache/g10_short_rates.parquet`` (OECD MEI 3-month interbank short rates, % p.a., monthly,
    columns ``USD,JPY,GBP,EUR,CAD,AUD,CHF,SEK,NOK,NZD``) and ``_cache/g10_fx.parquet`` (daily FX,
    **USD per 1 unit** of nine foreign currencies) — the same two parquets Study 36 (Greenback) runs on.
    On a cache miss with ``fetch=True`` it **reuses** the shared fetchers in ``tools/fetch_altdata.py``
    (one download path for the whole desk); with ``fetch=False`` it returns ``{}`` so offline runs never
    touch the network. FX is resampled to month-end and the rates index is snapped to month-end so the
    carry accrual and the FX appreciation line up.
    """
    rates_path = os.path.join(cache_dir, RATES_CACHE)
    fx_path = os.path.join(cache_dir, FX_CACHE)
    if not (os.path.exists(rates_path) and os.path.exists(fx_path)):
        if not fetch:
            return {}
        import sys
        tools_dir = os.path.join(REPO_ROOT, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        try:
            import fetch_altdata  # the desk's shared fetchers — no duplicated download code here
            if not os.path.exists(rates_path):
                fetch_altdata.fetch_g10_short_rates()
            if not os.path.exists(fx_path):
                fetch_altdata.fetch_g10_fx()
        except Exception:
            return {}
        if not (os.path.exists(rates_path) and os.path.exists(fx_path)):
            return {}
    rates = pd.read_parquet(rates_path).sort_index()
    rates.index = pd.DatetimeIndex(rates.index) + pd.offsets.MonthEnd(0)
    fx = pd.read_parquet(fx_path).sort_index()
    fx_m = fx.resample("ME").last()
    return {"rates": rates, "fx": fx_m}
