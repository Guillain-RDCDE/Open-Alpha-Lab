"""Data layer for Study 895 — Defensive Momentum.

Two sources, both offline-friendly once cached.

* **Real tape (yfinance, no key).** Daily **total-return** (auto-adjusted, dividends
  reinvested, net of each fund's expense ratio) closes for the two sleeves under blend —
  **MTUM** (iShares MSCI USA Momentum, 2013-04) and **USMV** (iShares MSCI USA Min Vol,
  2011-10) — plus **QUAL** (a defensive-factor cross-check), **SPY** (the market
  benchmark) and **BIL** (1-3M T-bill ETF, the cash / risk-free leg — its own monthly
  total return IS the monthly risk-free rate, so ``excess = asset − BIL`` needs no
  yield-to-return conversion). Cached wide under the study's own ``_cache/`` as parquet.

* **Synthetic world.** A deterministic, seeded monthly generator (``synthetic_sleeves``)
  of a momentum sleeve, a min-vol sleeve, a market bench and a cash leg, with a TUNABLE
  ``edge`` knob. ``edge=0`` is the null: the two sleeves are the SAME series, so any blend
  is identical to each sleeve — no diversification benefit, zero Sharpe advantage, equal
  drawdown. ``edge>0`` plants the real story: momentum carries occasional crash shocks and
  a touch more market beta, min-vol is calmer and crash-free, so the blend earns a
  shallower drawdown and (if the vol/crash reduction beats the small return give-up) a
  Sharpe advantage. It proves the machinery is unbiased; it is NEVER cited for a stamp.

Pure numpy + pandas + stdlib on the offline path; ``fetch`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells. Synthetic indices are
built with ``pd.period_range(..., freq="M")`` kept as a PeriodIndex (spans stay far below
the pandas ns-Timestamp horizon — no overflow trap).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
TAPE_CACHE = os.path.join(CACHE_DIR, "defmom_prices.parquet")

# The two sleeves blended, plus cross-check factor, market bench and cash leg.
SLEEVES = ["MTUM", "USMV"]          # momentum + min-vol (the blend ingredients)
CROSS = "QUAL"                       # quality — a second defensive-factor cross-check
BENCH = "SPY"                        # the market
CASH = "BIL"                         # 1-3M T-bill ETF — the cash / risk-free leg
TICKERS = SLEEVES + [CROSS, BENCH, CASH]

# Frozen as-of: monthly stats never include a partial month (June 2026 = last complete).
AS_OF = "2026-06-30"

__all__ = [
    "SLEEVES", "CROSS", "BENCH", "CASH", "TICKERS", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_prices", "monthly_total_returns", "synthetic_sleeves",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2007-01-01", end: str | None = None,
          path: str = TAPE_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the tape once and cache it as parquet (network-only path).

    All columns are auto-adjusted closes = **total return** (dividends reinvested, net of
    each fund's expense ratio). BIL's total-return series doubles as the cash leg.
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
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the Defensive-Momentum tape.")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = TAPE_CACHE) -> pd.DataFrame:
    """Cached wide total-return price frame. OFFLINE — no yfinance import."""
    return pd.read_parquet(path).sort_index()


# --------------------------------------------------------------------------- #
# Monthly building blocks
# --------------------------------------------------------------------------- #
def monthly_total_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Month-end-to-month-end simple returns from a total-return price frame.

    Sliced to the frozen ``asof`` (the last complete calendar month) so a stamped run
    never contains a partial month and never drifts as new sessions arrive.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    return ret[ret.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof only)
# --------------------------------------------------------------------------- #
def synthetic_sleeves(edge: float = 0.0, seed: int = 895, n_months: int = 160,
                      drift_m: float = 0.0095, common_vol: float = 0.036,
                      idio_vol: float = 0.018, crash_prob: float = 0.06,
                      crash_size: float = 0.11, give_up: float = 0.0015,
                      beta_cut: float = 0.35, cash_m: float = 0.0022) -> pd.DataFrame:
    """Deterministic seeded monthly world: momentum, min-vol, bench and cash columns.

    A common market component ``g`` drives both sleeves. The TUNABLE ``edge`` scales three
    things at once, all zero at ``edge=0``:

    * momentum's **crash shocks** (rare large negative jumps min-vol does not share);
    * the sleeves' **independent idiosyncratic** noise (the raw diversification material);
    * min-vol's **calmer beta** (``1 − beta_cut·edge`` of the common move) and its small
      return **give-up** (``give_up·edge`` per month).

    At ``edge=0`` momentum == min-vol == ``drift_m + g`` exactly, so every blend equals
    each sleeve: zero Sharpe advantage, identical drawdown (the clean null). At ``edge>0``
    momentum takes the crashes and min-vol is the calmer, crash-free sleeve, so the blend
    posts a shallower drawdown and — when the vol/crash reduction beats the give-up — a
    positive excess-Sharpe advantage. Index is a PeriodIndex (no ns-Timestamp overflow).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2013-05", periods=n_months, freq="M")

    g = common_vol * rng.standard_normal(n_months)                 # common market move
    g_bench = 0.033 * rng.standard_normal(n_months)                # independent bench draw
    crash_hit = (rng.random(n_months) < crash_prob).astype(float)
    crash = -crash_size * crash_hit * edge                         # momentum-only crashes
    mom_idio = idio_vol * edge * rng.standard_normal(n_months)
    mv_idio = idio_vol * edge * rng.standard_normal(n_months)

    mom = drift_m + g + mom_idio + crash
    minvol = (drift_m - give_up * edge) + (1.0 - beta_cut * edge) * g + mv_idio
    bench = 0.0080 + g_bench
    cash = np.full(n_months, cash_m)

    return pd.DataFrame({"MTUM": mom, "USMV": minvol, "SPY": bench, "BIL": cash},
                        index=idx)
