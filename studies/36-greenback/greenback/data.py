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
  * :func:`fetch_fx_rates` — the real hook. *Would* pull FX spot (yfinance) + short rates (FRED) and build
    the carry panel; **cache-first**, network only behind ``fetch=True``. In this environment the FRED
    rates fetch **times out**, so on a cache-miss it returns ``{}`` and the real run is skipped — the
    synthetic core stands alone, exactly as [Study 27 (Steamroller)](../../27-steamroller/) does.

Two data choices, stated up front. **Monthly horizon** — carry is a slow rate-differential signal and
momentum a 12-month trend; monthly is the right (and cleanly available) frequency. **USD base** — the
dollar-carry tilt is naturally expressed against the USD funding leg.
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
# Real tape — FX spot (yfinance) + short rates (FRED). PENDING: the FRED fetch times out in this sandbox.
# --------------------------------------------------------------------------- #

# Short-term (3-month) interbank rates, % p.a., monthly, FRED ids.
FRED_RATES = {
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N", "SEK": "IR3TIB01SEM156N", "NOK": "IR3TIB01NOM156N",
}
# FX spot via yfinance (USD per 1 unit of foreign currency where the pair quotes that way).
YF_FX = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "AUD": "AUDUSD=X", "JPY": "JPY=X",
    "CAD": "CAD=X", "CHF": "CHF=X", "SEK": "SEK=X", "NOK": "NOK=X",
}


def _fred_csv(series_id: str, timeout: int = 30) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URL)
        df = pd.read_csv(io.StringIO(resp.read().decode()))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fetch_fx_rates(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """Return ``{'rates': DataFrame, 'fx': DataFrame}`` of monthly short rates and USD FX spot, cache-first.

    Reads a cached ``greenback_fx.parquet`` if present; otherwise, only if ``fetch=True``, downloads short
    rates from **FRED** and FX spot from **yfinance**, aligns them to month-end, caches, and returns.

    **Pending-fetch note (Study 27 pattern).** FX spot from yfinance works in this sandbox, but the FRED
    rates download **times out** here — so without a pre-populated cache this returns ``{}`` and the
    study's real run is skipped. The offline synthetic core (:func:`synthetic_fx`) is the validated proof
    meanwhile; the real-tape verdict is PENDING one networked FRED fetch (see ``docs/results.md``).
    """
    cache = os.path.join(cache_dir, "greenback_fx.parquet")
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
    if not rates:
        return {}                                    # FRED timeout — the pending-fetch case in this sandbox
    try:
        import yfinance as yf
    except Exception:
        return {}
    fx = {}
    for ccy, tk in YF_FX.items():
        try:
            s = yf.download(tk, progress=False)["Close"].resample("ME").last()
            fx[ccy] = s.squeeze()
        except Exception:
            continue
    if not fx:
        return {}
    rates_df = pd.DataFrame(rates).sort_index()
    fx_df = pd.DataFrame(fx).sort_index()
    os.makedirs(cache_dir, exist_ok=True)
    out = pd.concat([rates_df.add_prefix("rate_"), fx_df.add_prefix("fx_")], axis=1).dropna(how="all")
    out.to_parquet(cache)
    return {"rates": rates_df, "fx": fx_df}
