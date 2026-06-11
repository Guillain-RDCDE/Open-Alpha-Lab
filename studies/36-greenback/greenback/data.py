"""Data for the dollar-carry / carry⊕momentum study — an offline synthetic FX panel, and the real hook.

Greenback needs three ingredients per currency: a (persistent) **interest-rate differential** that
predicts excess return (the carry premium), a **negative-skew crash** component (carry crashes — the
steamroller), and a **trend** component so a momentum sleeve has something to ride. The desk's offline /
cache split:

  * :func:`synthetic_fx` — fully **offline, deterministic**. A toy currency panel where each currency has
    a fixed rate differential that earns a partial-UIRP premium (carry), occasional joint risk-off crashes
    that hit the high-carry names hardest (negative skew), *and* a slow autocorrelated trend so a
    trailing-return momentum book is profitable too. ``carry_strength`` sets the premium; ``trend_strength``
    the momentum signal; ``carry_strength = 0`` is the carry **null** (full UIRP — no premium, no crash).
    Returns ``(excess-returns panel, rate-differential frame, truth)``.
  * :func:`fetch_fx_rates` — the **real tape**, served from cache. Reads two pre-fetched parquets —
    ``_cache/g10_short_rates.parquet`` (OECD 3-month interbank short rates, % p.a., monthly, columns
    ``USD,JPY,GBP,EUR,CAD,AUD,CHF,SEK,NOK,NZD``, sample ~1994→**2024-01**; OECD MEI was discontinued so the
    rates end 2024-01) and ``_cache/g10_fx.parquet`` (daily FX, **USD per 1 unit** of nine foreign
    currencies ``EUR,GBP,AUD,NZD,JPY,CAD,CHF,SEK,NOK``, yfinance ~2003→2026) — and builds the monthly carry
    book. **Cache-first, offline.** :func:`build_carry_panel` does the carry math (below).

The carry math, stated up front. For a USD investor the **excess return** of being long 1 unit of a
foreign currency, funded in USD, over a month is ≈ **FX appreciation + the rate gap accrual** =
``Δlog(usd_per_unit) + (rate_foreign − rate_USD)/12`` (rates in decimal p.a.). The **carry signal** is the
forward discount ≈ the rate differential ``rate_foreign − rate_USD`` (high = high carry). Two choices:
**monthly horizon** (rates are monthly; carry is slow; FX resampled to month-end) and **USD base** (the
USD is the funding leg, so the dollar-carry tilt is naturally long/short USD vs the basket). Effective
sample after rates ∩ FX and a 12-month momentum warm-up ≈ **2003→2024-01**.
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
class FXTruth:
    """What the synthetic generator baked in, so a test can check the books recover it."""
    n_ccy: int
    n_months: int
    carry_strength: float     # fraction of UIRP that fails (the carry premium); 0 == full-UIRP null
    trend_strength: float     # persistence of the FX trend the momentum sleeve rides
    crash_size: float

    @property
    def has_carry(self) -> bool:
        return self.carry_strength != 0.0

    @property
    def has_trend(self) -> bool:
        return self.trend_strength != 0.0


def synthetic_fx(n_ccy: int = 9, n_months: int = 600, carry_strength: float = 0.9,
                 trend_strength: float = 0.35, crash_size: float = 0.06, crash_prob: float = 0.012,
                 fx_vol: float = 0.011, crash_persist: float = 0.85, seed: int = 36
                 ) -> tuple[pd.DataFrame, pd.DataFrame, FXTruth]:
    """A toy currency panel with a carry premium, carry crashes, AND a tradable trend (for momentum).

    For currency ``i`` with monthly rate ``r_i`` (fixed, spread across the panel) and base-average rate
    ``r̄``, the monthly **excess return** of holding it (funded at the average) is built from three pieces::

        crash_t   ~ sticky 2-state Markov (calm / risk-off)              (a shared risk-off month)
        trend_i,t = trend_strength * trend_i,t-1 + trend_shock_i,t       (a slow autocorrelated drift)
        ds_i      = -(1 - carry_strength)*(r_i - r̄)                       (partial UIRP: high rates drift down, but not fully)
                    + trend_i,t                                          (the momentum sleeve rides this)
                    + fx_vol*eps_i
                    - carry_strength*crash_t*crash_size*(r_i - r̄)/scale   (high-carry crashes hardest in risk-off)
        xret_i    = (r_i - r̄) + ds_i

    ``carry_strength > 0`` leaves the high-rate names a real premium (and the risk-off crashes give the
    negative skew — the steamroller); ``trend_strength > 0`` makes returns autocorrelated so a trailing
    momentum book is profitable. Crucially the crash is driven by the *carry* tilt while the trend is
    independent, so carry and momentum pay at **different times** — the whole point of the combo.
    ``carry_strength = 0`` is the carry null (full UIRP, no premium, no crash). Returns
    ``(excess_returns, rate_diffs, truth)`` as monthly ``months × currency`` frames; deterministic given
    ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1995-01", periods=n_months, freq="M").to_timestamp(how="end")
    idx.name = "date"
    rates_ann = np.linspace(0.00, 0.18, n_ccy)
    rng.shuffle(rates_ann)
    r = rates_ann / MONTHS_PER_YEAR                      # monthly
    rbar = r.mean()
    dr = r - rbar                                        # rate differential vs the cross-sectional average
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

    # a slow autocorrelated trend per currency — what the momentum sleeve rides. Independent of the
    # carry tilt and the crash, so the two premia are decorrelated by construction (the combo thesis).
    trend = np.zeros((n_months, n_ccy))
    tshock = 0.010 * rng.standard_normal((n_months, n_ccy))
    prev = np.zeros(n_ccy)
    for t in range(n_months):
        prev = trend_strength * prev + tshock[t]
        trend[t] = prev

    eps = rng.standard_normal((n_months, n_ccy))
    ds = (-(1.0 - carry_strength) * dr[None, :]
          + trend
          + fx_vol * eps
          - carry_strength * crash[:, None] * crash_size * (dr[None, :] / scale))
    xret = dr[None, :] + ds
    cols = [f"C{i:01d}" for i in range(n_ccy)]
    xr = pd.DataFrame(xret, index=idx, columns=cols)
    rate_diffs = pd.DataFrame(np.tile(dr, (n_months, 1)), index=idx, columns=cols)
    return xr, rate_diffs, FXTruth(n_ccy=n_ccy, n_months=n_months, carry_strength=carry_strength,
                                   trend_strength=trend_strength, crash_size=crash_size)


# --------------------------------------------------------------------------- #
# Real tape — OECD 3-month short rates + yfinance FX, served from the offline cache.
# --------------------------------------------------------------------------- #

RATES_CACHE = "g10_short_rates.parquet"      # OECD 3-month interbank, % p.a., monthly, month-end index.
FX_CACHE = "g10_fx.parquet"                  # daily, USD per 1 unit of each foreign currency (yfinance).

# The data ends 2024-01 — OECD MEI (the rates source) was discontinued — so we pin the headline as-of to
# the rates end, not the live calendar, and the published numbers never creep.
DATA_AS_OF = "2024-01-31"


def fetch_fx_rates(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Return ``{'rates': DataFrame, 'fx': DataFrame}`` of monthly short rates and month-end USD FX spot.

    **Cache-first, offline.** Reads ``_cache/g10_short_rates.parquet`` (OECD 3-month interbank short rates,
    % p.a., monthly, columns ``USD,JPY,GBP,EUR,CAD,AUD,CHF,SEK,NOK,NZD``) and ``_cache/g10_fx.parquet``
    (daily FX, **USD per 1 unit** of nine foreign currencies). FX is resampled to month-end and both frames
    are snapped to a month-end index so the carry accrual and the FX appreciation line up. Returns ``{}``
    only if a cache parquet is genuinely missing — ``fetch`` is accepted for signature compatibility but the
    real tape is the cache (no network is used).
    """
    rates_path = os.path.join(cache_dir, RATES_CACHE)
    fx_path = os.path.join(cache_dir, FX_CACHE)
    if not (os.path.exists(rates_path) and os.path.exists(fx_path)):
        return {}
    rates = pd.read_parquet(rates_path).sort_index()
    rates.index = pd.DatetimeIndex(rates.index) + pd.offsets.MonthEnd(0)
    fx = pd.read_parquet(fx_path).sort_index()
    fx_m = fx.resample("ME").last()
    return {"rates": rates, "fx": fx_m}


def build_carry_panel(rates: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the monthly ``(excess_returns, rate_diffs)`` carry panel for a USD investor, offline.

    For each foreign currency present in **both** frames, the monthly **excess return** of holding 1 unit
    funded in USD is ``Δlog(usd_per_unit) + (rate_foreign − rate_USD)/12`` (rates → decimal p.a.). The
    **carry signal** (``rate_diffs``) is the rate differential ``rate_foreign − rate_USD`` — the forward
    discount, high = high carry. Both frames are returned on the common monthly index after a ``dropna`` of
    all-empty rows; USD is the funding leg (no USD column on either side of the output).
    """
    ccys = [c for c in fx.columns if c in rates.columns and c != "USD"]
    r = rates[ccys] / 100.0 / MONTHS_PER_YEAR              # monthly decimal foreign rates
    r_usd = rates["USD"] / 100.0 / MONTHS_PER_YEAR
    fx_ret = np.log(fx[ccys]).diff()
    rate_diffs = r.sub(r_usd, axis=0)                      # forward-discount carry signal vs USD
    xret = (fx_ret + rate_diffs).dropna(how="all")
    rate_diffs = rate_diffs.reindex(xret.index)
    return xret, rate_diffs


def load_real_panel(cache_dir: str = DEFAULT_CACHE, as_of: str | None = DATA_AS_OF
                    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Convenience: read the cache and build the carry panel, pinned to ``as_of`` (the data end). Offline.

    Returns ``(excess_returns, rate_diffs)`` or ``None`` if the cache is missing. Pins both frames to
    ``as_of`` so the headline numbers don't creep past the rates' 2024-01 end.
    """
    tape = fetch_fx_rates(cache_dir=cache_dir)
    if not tape:
        return None
    xret, rate_diffs = build_carry_panel(tape["rates"], tape["fx"])
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        xret = xret[xret.index <= cut]
        rate_diffs = rate_diffs.reindex(xret.index)
    return xret, rate_diffs
