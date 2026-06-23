"""Data layer for Study 364 — FX Carry Trade (FX spot + a carry proxy + synthetic control).

Two sources, both offline-friendly:

* **Real tape.** Daily FX spot for a fixed G10 basket vs USD (yfinance, no key),
  cached under ``_cache/fx_spot.csv`` (a wide CSV indexed by date, one column per
  pair). We normalise every pair to **USD-per-foreign-currency** units (so a *rise*
  means the foreign currency *appreciated*), then attach a **carry proxy**: a fixed,
  transparent long-run average short-rate differential vs USD for each currency
  (annualised, in percent). True overnight deposit rates are not on yfinance, so the
  carry is an explicit per-currency constant — a proxy, named as such everywhere.
  The monthly total return of a long position in currency *i* is the spot change
  plus 1/12 of its carry; a short position earns the negative.

* **Synthetic.** A deterministic, fixed-seed generator producing FX panels with a
  *known* carry premium and a *known* fat-tailed crash component injected by
  construction. It is the positive control: the engine must recover the planted
  carry Sharpe and the planted negative skew, and — with the edge knob set to zero —
  must NOT manufacture a significant premium out of noise.

Pure numpy + pandas + stdlib for the offline path. ``fetch_fx`` (network) is only
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "fx_spot.csv")

# A fixed G10 basket of currencies vs USD. yfinance serves these as ``EURUSD=X``
# (USD per EUR — already USD-per-foreign) or ``USDJPY=X`` (JPY per USD — inverted).
# We map every ticker to a (currency, invert) pair so the loader normalises all
# series to USD-per-foreign-currency (a rise => the foreign currency appreciated).
TICKERS = {
    "EURUSD=X": ("EUR", False),
    "GBPUSD=X": ("GBP", False),
    "AUDUSD=X": ("AUD", False),
    "NZDUSD=X": ("NZD", False),
    "USDJPY=X": ("JPY", True),
    "USDCHF=X": ("CHF", True),
    "USDCAD=X": ("CAD", True),
    "USDNOK=X": ("NOK", True),
    "USDSEK=X": ("SEK", True),
}

# Transparent, fixed carry-by-currency PROXY: a long-run average annualised
# short-rate differential vs USD (in percent), the carry a USD investor earns by
# holding that currency. Approximate, deliberately round numbers reflecting the
# 2004-2024 era ordering (commodity/high-yielders AUD/NZD on top, funding
# currencies JPY/CHF deeply negative). This is a PROXY for true deposit-rate
# differentials, named as such everywhere; the verdict does not hinge on the exact
# values, only on the high-minus-low ordering and the crash shape of the basket.
CARRY_PROXY = {
    "AUD": 1.6,
    "NZD": 1.8,
    "NOK": 0.6,
    "GBP": 0.4,
    "CAD": 0.2,
    "SEK": 0.0,
    "EUR": -0.4,
    "CHF": -1.4,
    "JPY": -1.6,
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_fx(start: str = "2004-01-01", end: str | None = None,
             path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the FX basket via yfinance and cache a wide USD-per-foreign CSV.

    Network-only; used once to build ``_cache/fx_spot.csv``. Never imported by the
    offline notebook cells. Each pair is normalised to USD-per-foreign-currency.
    """
    import yfinance as yf

    raw = yf.download(list(TICKERS.keys()), start=start, end=end,
                      auto_adjust=False, progress=False)["Close"]
    raw = raw.dropna(how="all")
    out = pd.DataFrame(index=raw.index)
    for tk, (ccy, invert) in TICKERS.items():
        if tk not in raw.columns:
            continue
        s = raw[tk].astype(float)
        out[ccy] = (1.0 / s) if invert else s
    # keep currencies present for >=60% of the window
    keep = [c for c in out.columns if out[c].notna().mean() >= 0.60]
    out = out[keep].sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_spot(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide USD-per-foreign spot frame (index = date, cols = ccy)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.dropna(how="all")


def monthly_returns(spot: pd.DataFrame,
                    carry: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-end spot returns and total (spot + carry) returns per currency.

    Returns ``(spot_ret, total_ret)`` monthly frames. ``total_ret`` = spot return +
    1/12 of the annualised carry proxy (the return of a *long* USD-funded position
    in that currency). A *short* position earns the negative of the total return.
    """
    carry = CARRY_PROXY if carry is None else carry
    me = spot.resample("ME").last()
    spot_ret = me.pct_change()
    carry_m = pd.Series({c: carry.get(c, 0.0) / 100.0 / 12.0 for c in me.columns})
    total_ret = spot_ret.add(carry_m, axis=1)
    return spot_ret.dropna(how="all"), total_ret.dropna(how="all")


def load_real(path: str = DEFAULT_CACHE,
              carry: dict[str, float] | None = None) -> dict:
    """Convenience bundle: cached spot -> monthly spot/total returns + the carry map."""
    spot = load_spot(path)
    spot_ret, total_ret = monthly_returns(spot, carry=carry)
    carry = CARRY_PROXY if carry is None else carry
    return {"spot": spot, "spot_ret": spot_ret, "total_ret": total_ret,
            "carry": {c: carry.get(c, 0.0) for c in spot.columns}}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_fx(n_months: int = 240, n_ccy: int = 8,
                 carry_spread: float = 0.07, carry_edge: float = 0.0,
                 crash_skew: float = 0.0, seed: int = 364) -> dict:
    """Deterministic FX panel with a PLANTED carry premium and crash skew.

    Builds ``n_ccy`` currencies with a fixed spread of carries (annualised, evenly
    spaced over ``[-carry_spread/2, +carry_spread/2]``). Each month, every currency's
    total return = its carry/12 + idiosyncratic FX noise + a *common* risk-off factor.
    The high-yielders load *positively* on the risk-off factor's downside, so when the
    factor crashes they crash hardest — the carry-crash mechanism, planted by
    construction.

    Knobs (the positive control turns each one independently)
    --------------------------------------------------------
    carry_spread : float
        Width of the planted annualised carry differential across currencies. **Set
        to 0 for the true null**: with no carry spread, no edge, and no crash, the
        long-high/short-low basket is pure noise and the engine must NOT manufacture
        significance (HAC ``|t| < 2``).
    carry_edge : float
        Extra annualised drift added to high-yielders on top of the mechanical carry
        — a planted *excess* the inference must recover (a large value must light up).
    crash_skew : float
        Magnitude (>= 0) of the planted fat-tailed left tail in the risk-off factor.
        ``0`` => roughly symmetric; large => deep negative skew in the basket return,
        which is the whole point: the engine must surface the planted crash shape.

    Returns the same bundle shape as :func:`load_real` so the strategy code is
    source-agnostic.
    """
    rng = np.random.default_rng(seed)
    # evenly spaced annualised carries spanning carry_spread (0 => flat => true null)
    if n_ccy > 1:
        carries = np.linspace(-carry_spread / 2.0, carry_spread / 2.0, n_ccy)
    else:
        carries = np.zeros(n_ccy)
    ccy = [f"C{i}" for i in range(n_ccy)]
    carry_map = {c: float(k * 100.0) for c, k in zip(ccy, carries)}

    # high-yielders load positively on the risk-off factor (so they fall in risk-off);
    # use a fixed unit spread of betas so the crash mechanism survives a flat carry.
    beta = np.linspace(-1.0, 1.0, n_ccy) if n_ccy > 1 else np.zeros(n_ccy)

    # common risk-off factor: mostly calm, with occasional fat-tailed down-spikes
    base = rng.normal(0.0, 0.015, size=n_months)
    if crash_skew > 0:
        # negative jumps: a few months get a large down-move scaled by crash_skew
        jump_when = rng.random(n_months) < 0.06
        jumps = -np.abs(rng.standard_t(3, size=n_months)) * crash_skew
        factor = base + np.where(jump_when, jumps, 0.0)
    else:
        factor = base

    spot_ret = np.empty((n_months, n_ccy))
    for j in range(n_ccy):
        idio = rng.normal(0.0, 0.020, size=n_months)
        spot_ret[:, j] = idio + beta[j] * factor + carry_edge * beta[j] / 12.0

    idx = pd.period_range("2004-01", periods=n_months, freq="M").to_timestamp("M")
    spot_ret_df = pd.DataFrame(spot_ret, index=idx, columns=ccy)
    carry_m = pd.Series({c: carry_map[c] / 100.0 / 12.0 for c in ccy})
    total_ret_df = spot_ret_df.add(carry_m, axis=1)
    # a notional spot level path (decorative; engine uses returns)
    spot_lvl = 1.0 * (1.0 + spot_ret_df).cumprod()
    return {"spot": spot_lvl, "spot_ret": spot_ret_df, "total_ret": total_ret_df,
            "carry": carry_map}
