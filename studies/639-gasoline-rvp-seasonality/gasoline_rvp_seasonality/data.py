"""Data layer for Study 639 — Gasoline RVP Seasonality.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for the RBOB gasoline front-month chain (``RB=F``, trades since
  Oct 2005), the WTI crude front-month chain (``CL=F``, the control leg), and the United States
  Gasoline Fund ETF (``UGA``, Feb 2008+, ``auto_adjust=True`` so it is total-return) — all from
  yfinance, no key. Cached wide to ``_cache/rvp_prices.csv`` and always re-loaded **as-of**
  ``AS_OF`` (the last complete calendar month) so reruns cannot drift.

  A construction note that the whole study leans on: Yahoo's ``RB=F``/``CL=F`` are **spliced
  front-month chains** — at each roll the series jumps from the expiring contract's price to the
  next contract's price. Those roll jumps are *not* earned by any holder (a real roller sells old
  and buys new at equal value), so the spliced series tracks the **spot price level** seasonal,
  roll markup included. ``UGA`` is the **investable holder**: it owns the front month, pays every
  roll, and earns collateral interest. The Signal axis reads the spliced (spot-proxy) crack
  seasonal; the Tradability and third axes read UGA and the UGA-minus-splice gap.

* **Synthetic.** A deterministic, fixed-seed monthly world with a **planted RVP seasonal**
  (knob ``amp``: extra mean monthly gasoline-over-crude excess in Feb–Apr, mirrored as ``-amp``
  in Sep) and a **planted curve-pricing effect** (knob ``roll_drag``: a per-month drag on the
  *holder* series concentrated in the Feb–Apr window — the futures curve eating the seasonal).
  ``amp = 0`` and ``roll_drag = 0`` are the nulls: the detectors must NOT light up on them.
  Span is ~21 years — far below the 250-year ns-Timestamp ceiling.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "rvp_prices.csv")

TICKERS = ["RB=F", "CL=F", "UGA", "^IRX"]   # ^IRX = 13-week T-bill discount yield (% p.a.)

# Frozen as-of: the last complete calendar month at publication (2026-07-03). Bump deliberately.
AS_OF = "2026-06-30"

# The regulatory calendar under test (40 CFR 1090.215, ex-80.27): summer-RVP gasoline must be at
# refineries/terminals by MAY 1 (retail June 1); winter blend may return after SEPTEMBER 15.
RUNUP_MONTHS = (2, 3, 4)    # Feb-Apr: the run-up into the May-1 deadline
SWITCHBACK_MONTH = 9        # Sep: the switch-back to cheap winter blend


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2005-01-01", end: str | None = None,
                 path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the three daily close series and cache them wide. Network; runs once."""
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
    prices = raw.dropna(how="all").sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices.to_csv(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide daily closes, sliced to the frozen as-of (no sample creep)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px[px.index <= pd.Timestamp(asof)].copy()


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple returns (month-end to month-end); drops a partial final month."""
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


def tbill_monthly(prices: pd.DataFrame) -> pd.Series:
    """Monthly T-bill simple return from the ^IRX level (annualised % -> monthly).

    Used only to make the long-only UGA overlay an excess-vs-cash race (the desk's
    excess-vs-excess rule): sitting out of the window earns the bill, so the overlay must too.
    """
    irx = prices["^IRX"].resample("ME").last()
    return (irx / 100.0 / 12.0).rename("tbill")


# --------------------------------------------------------------------------- #
# Synthetic control world
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 21, amp: float = 0.0, roll_drag: float = 0.0,
                    seed: int = 639, sig_crude: float = 0.09,
                    sig_idio: float = 0.05, sig_track: float = 0.004) -> pd.DataFrame:
    """Deterministic monthly-returns world with a planted RVP seasonal + curve pricing.

    Columns mirror the real tape: ``RB=F`` (spot-proxy gasoline), ``CL=F`` (crude), ``UGA``
    (the holder). Gasoline = crude + idiosyncratic noise + a planted seasonal of ``+amp`` mean
    monthly log-excess in Feb-Apr and ``-amp`` in Sep. The holder equals gasoline minus
    ``roll_drag`` per month **inside Feb-Apr only** (the curve pricing the transition away).
    ``amp = roll_drag = 0`` is the null world. Returns a monthly simple-returns frame on a
    month-end index built from ``period_range`` (span ~21 years, well under the 250-year cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2005-01", periods=n_years * 12, freq="M")
    months = np.array([p.month for p in pidx])

    crude_lr = rng.normal(0.004, sig_crude, len(pidx))
    idio_lr = rng.normal(0.0, sig_idio, len(pidx))
    seas = np.where(np.isin(months, RUNUP_MONTHS), amp, 0.0)
    seas = seas + np.where(months == SWITCHBACK_MONTH, -amp, 0.0)
    gas_lr = crude_lr + idio_lr + seas
    # the holder pays the planted roll drag inside the window, plus a small tracking noise
    # (fees, roll-timing slippage) so the paired gap t behaves like a real statistic
    holder_lr = (gas_lr - np.where(np.isin(months, RUNUP_MONTHS), roll_drag, 0.0)
                 + rng.normal(0.0, sig_track, len(pidx)))

    idx = pidx.to_timestamp(how="end").normalize()
    return pd.DataFrame({
        "RB=F": np.expm1(gas_lr),
        "CL=F": np.expm1(crude_lr),
        "UGA": np.expm1(holder_lr),
    }, index=idx)
