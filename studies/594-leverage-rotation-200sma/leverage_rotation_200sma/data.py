"""Data layer for Study 594 — Leverage Rotation (TQQQ + 200SMA).

Two sources, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for **QQQ** (Nasdaq-100 ETF,
  listed 1999-03-10), **TQQQ** (ProShares UltraPro QQQ, 3x daily, listed 2010-02-11) and
  **^IRX** (13-week T-bill discount yield, the cash leg + financing rate), all from
  yfinance (no key), cached wide under ``_cache/lr200_prices.csv``.

* **Synthesised 3x.** TQQQ only exists since 2010, but the claim's whole point is the
  2000-02 crash. We synthesise a 3x daily-reset fund from QQQ's daily total return with
  the constant-leverage identity used (and validated at corr 0.999) by the desk's
  melting-ice study (../100-melting-ice/):

      r_3x(t) = 3 * r_QQQ(t) - 2 * rf(t) - fee/252

  i.e. 3x the underlying's daily return, minus financing on the 2x borrowed notional at
  the T-bill rate, minus an all-in expense/swap-spread fee. ``validate_sim3x`` scores the
  synthesis against the *real* TQQQ tape 2010-2026 (daily-return correlation, annualised
  residual drag, terminal-NAV ratio) so the pre-2010 extension is an audited instrument,
  not a guess.

* **Synthetic world.** A deterministic, fixed-seed regime generator with a **tunable
  planted effect**: ``effect = 0`` produces an i.i.d. tape (no exploitable trend
  structure — the null: a 200SMA rotation must NOT beat a matched random timer);
  ``effect > 0`` plants persistent bear regimes (negative drift, doubled volatility)
  that a 200SMA is built to dodge — the positive control must light up. Spans ~20
  business years (far under the 250-year ns-Timestamp ceiling).

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "lr200_prices.csv")

TICKERS = ["QQQ", "TQQQ", "^IRX"]
TRADING_DAYS = 252

# All-in synthesis fee: TQQQ expense ratio (~0.95%/yr) + swap-financing spread over
# T-bills. Calibrated ONCE against the real TQQQ tape 2010-2026: with financing at
# 2x T-bill, the measured residual drag of real TQQQ vs a 0.95%-fee synthesis is
# ~1.5%/yr, so the honest all-in figure is ~2.5%/yr. Using it (rather than the
# flattering bare expense ratio) means the pre-2010 extension does NOT understate
# the levered fund's true drag. validate_sim3x() audits what remains.
SIM3X_FEE = 0.025
LEVERAGE = 3.0

# Reproducibility pin: the last complete month before the study's as-of (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1999-01-01", end: str | None = None,
                 path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download QQQ + TQQQ (total-return closes) + ^IRX and cache them wide.

    Network-only; used once to build the cache. auto_adjust=True means QQQ/TQQQ closes
    are split- AND dividend-adjusted (total return). ^IRX is a yield index in percent
    (13-week T-bill discount yield), unaffected by adjustment.
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all").sort_index()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = PRICES_CACHE, as_of: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide frame (QQQ, TQQQ, ^IRX), pinned to the as-of date."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of is not None:
        px = px[px.index <= pd.Timestamp(as_of)]
    return px


# --------------------------------------------------------------------------- #
# Derived series
# --------------------------------------------------------------------------- #
def rf_daily(prices: pd.DataFrame) -> pd.Series:
    """Daily T-bill accrual from ^IRX (13-week discount yield, percent, ffilled)."""
    irx = prices["^IRX"].ffill() / 100.0
    return (irx / TRADING_DAYS).rename("rf")


def qqq_returns(prices: pd.DataFrame) -> pd.Series:
    """QQQ daily total return (auto-adjusted close-to-close)."""
    return prices["QQQ"].dropna().pct_change().dropna().rename("qqq")


def tqqq_returns(prices: pd.DataFrame) -> pd.Series:
    """Real TQQQ daily total return (2010-02-11 onward)."""
    return prices["TQQQ"].dropna().pct_change().dropna().rename("tqqq")


def sim3x_returns(qqq_ret: pd.Series, rf: pd.Series,
                  leverage: float = LEVERAGE, fee: float = SIM3X_FEE) -> pd.Series:
    """Synthesised 3x daily-reset returns: 3*r - 2*rf - fee/252 (see module doc)."""
    rf_al = rf.reindex(qqq_ret.index).ffill().fillna(0.0)
    r3 = leverage * qqq_ret - (leverage - 1.0) * rf_al - fee / TRADING_DAYS
    return r3.rename("sim3x")


def validate_sim3x(prices: pd.DataFrame) -> dict:
    """Score the synthesised 3x against the REAL TQQQ tape on their common span.

    Returns daily-return correlation, annualised residual drag (real minus sim mean,
    the fee misfit), RMS daily tracking error and the terminal-NAV ratio real/sim.
    """
    real = tqqq_returns(prices)
    sim = sim3x_returns(qqq_returns(prices), rf_daily(prices))
    common = real.index.intersection(sim.index)
    a, b = real.loc[common], sim.loc[common]
    resid = a - b
    nav_real = float(np.exp(np.log1p(a).sum()))
    nav_sim = float(np.exp(np.log1p(b).sum()))
    return {
        "start": str(common.min().date()), "end": str(common.max().date()),
        "n_days": int(len(common)),
        "corr": float(a.corr(b)),
        "resid_drag_ann_pct": float(resid.mean() * TRADING_DAYS * 100.0),
        "rms_te_bps": float(resid.std() * 1e4),
        "nav_ratio": nav_real / nav_sim,
    }


# --------------------------------------------------------------------------- #
# Synthetic world (deterministic positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 5040, effect: float = 0.0, seed: int = 594,
                    mu_bull: float = 0.00055, sig_bull: float = 0.012,
                    bear_sig_mult: float = 2.0) -> pd.Series:
    """Deterministic daily close series with a TUNABLE planted regime effect.

    * ``effect = 0`` — the **null**: i.i.d. Gaussian daily returns (drift ``mu_bull``,
      vol ``sig_bull``). There is no trend persistence to exploit, so a 200SMA rotation
      must NOT beat an exposure-matched random timer, however the noise falls.
    * ``effect > 0`` — the **planted effect**: a two-state Markov world where persistent
      bear regimes (stay-probability 0.99, expected length ~100 days) carry drift
      ``-effect`` per day and ``bear_sig_mult``x the bull volatility. Long negative-drift
      high-vol episodes are exactly what a 200SMA detects, so the rotation must beat the
      random timer decisively.

    ~5040 business days ≈ 20 years — far below the 250-year pandas ns-Timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    if effect == 0.0:
        ret = rng.normal(mu_bull, sig_bull, size=n_days)
    else:
        p_stay_bull, p_stay_bear = 0.995, 0.99
        state = np.zeros(n_days, dtype=int)  # 0 = bull, 1 = bear
        u = rng.random(n_days)
        for t in range(1, n_days):
            if state[t - 1] == 0:
                state[t] = 0 if u[t] < p_stay_bull else 1
            else:
                state[t] = 1 if u[t] < p_stay_bear else 0
        mu = np.where(state == 0, mu_bull, -effect)
        sig = np.where(state == 0, sig_bull, sig_bull * bear_sig_mult)
        ret = rng.normal(0.0, 1.0, size=n_days) * sig + mu
    close = 100.0 * np.exp(np.cumsum(np.log1p(ret)))
    return pd.Series(close, index=idx, name="synthetic_close")
