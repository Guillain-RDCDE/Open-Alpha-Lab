"""Data layer for Study 888 — CLO AAA Carry.

The claim under test: a **AAA-rated CLO tranche** — the senior, first-loss-protected
slice of a collateralised loan obligation — pays a **spread over cash and over
same-rated corporate bonds** as compensation for *structural complexity* (a
securitisation few desks can underwrite), while sitting at the very top of the capital
stack where realized defaults are ~nil. Since late 2020 that slice has been packaged in
a liquid ETF (**JAAA**, Janus Henderson AAA CLO ETF; **ICLO**, Invesco AAA CLO ETF from
Dec-2022), floating-rate, so it carries almost no *duration*. The pitch: harvest
SOFR + an AAA-CLO spread with a tiny NAV wobble and no rate risk.

We test whether that carry is a **real, mechanical risk-adjusted pickup** against three
honest benchmarks, all excess-of-cash:

  * **LQD** — iShares IG corporate bonds (~A/BBB, ~8y duration). Same investment-grade
    rating bucket, but a very different risk (long duration + spread). The "vs same-rated
    corporates" leg of the story — confounded by duration, so we grade the *risk-adjusted*
    race, not the raw return.
  * **IEF** — 7-10y US Treasuries: pure duration, no credit. The risk-free-plus-duration
    yardstick.
  * **BKLN** — senior *leveraged* loans (Invesco): the SAME floating-rate collateral CLOs
    are built from, but held **un-tranched and below investment grade** (the risky whole,
    vs JAAA's protected top). The natural "is it just loan beta?" control (see study 340).

Cash / risk-free leg: **BIL** (1-3m T-bills) — the excess-of-cash subtraction that turns
every leg into an excess return, so the Sharpe race is excess-vs-excess.

**Short history — named on the Signal axis.** JAAA has traded only since 2020-10-19
(~5.7y to the as-of); ICLO since 2022-12. The sample spans one rate cycle (ZIRP → the
2022 hikes → a 5% plateau) but **no CLO credit-stress event** — the March-2020 AAA-CLO
mark-down (~-5-10% intraday) predates JAAA's inception. Every realized-Sharpe number
here is therefore measured on a calm tape and is an **upper bound** on the true
risk-adjusted edge; the caveat travels with the stamp.

Synthetic world: a deterministic seeded daily generator with a TUNABLE planted
carry spread over cash (knob ``carry_annual``, null at 0), plus a high-vol "duration"
decoy, so the machinery can be shown to fire on a real steady carry and stay silent on
the null. Index is a plain ``bdate_range`` (n well under the ns-Timestamp horizon).

Cache-first: ``fetch_tape`` (network, yfinance) runs once and writes
``_cache/clo_prices.csv``; everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "clo_prices.csv")

# One wide frame of total-return closes (auto_adjust=True): the AAA-CLO ETFs, the two
# credit/duration yardsticks, the leveraged-loan control, and the cash leg.
TICKERS = ["JAAA", "ICLO", "LQD", "IEF", "BKLN", "BIL"]

# The carry legs we race against cash, and a human label for each.
LEGS = {
    "JAAA": "AAA CLO ETF (Janus Henderson, 2020-10+; floating, senior tranche)",
    "ICLO": "AAA CLO ETF (Invesco, 2022-12+; floating, senior tranche)",
    "LQD":  "IG corporate bonds (~A/BBB, ~8y duration) — same rating bucket, long duration",
    "IEF":  "7-10y US Treasuries — pure duration, no credit",
    "BKLN": "senior leveraged loans (floating, below IG) — the un-tranched risky whole",
}
CASH = "BIL"  # 1-3m T-bills, the excess-of-cash subtraction

START = "2020-01-01"
AS_OF = "2026-06-30"  # last complete calendar month at build (2026-08); drop partial month

__all__ = [
    "TICKERS", "LEGS", "CASH", "START", "AS_OF", "CACHE_DIR", "PRICES_CACHE",
    "fetch_tape", "have_real", "load_prices", "daily_returns", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = START, end: str | None = None,
               path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for all tickers and cache them (network, run once)."""
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
        raise RuntimeError("yfinance returned no data for the CLO tape.")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame, cache-first, sliced to the as-of date (offline)."""
    if not os.path.exists(path):
        return fetch_tape(path=path)
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px[px.index <= pd.Timestamp(asof)]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple total-return returns for every column (index=date, columns=ticker)."""
    return prices.sort_index().pct_change()


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 1400, carry_annual: float = 0.012, seed: int = 888,
                    cash_annual: float = 0.03, carry_vol: float = 0.010,
                    dur_vol: float = 0.075, start: str = "2021-01-04") -> pd.DataFrame:
    """Deterministic daily world with a PLANTED carry spread over cash.

    Three total-return series, all built off a common cash path:

      * ``cash``  — a near-constant positive daily accrual (BIL-like), ``cash_annual``/252.
      * ``carry`` — cash + ``carry_annual``/252 + a low-vol idiosyncratic wobble
        (``carry_vol`` annualised): the AAA-CLO analogue. So ``carry - cash`` has mean
        ``carry_annual``/252 and a small standard deviation -> a *high* excess Sharpe when
        ``carry_annual > 0`` and a flat null when ``carry_annual = 0``.
      * ``dur``   — cash + a high-vol mean-zero "duration" factor (``dur_vol`` annualised):
        a decoy that earns *no* excess over cash on average but is far more volatile, so
        the excess-Sharpe detector must rank it BELOW the steady carry.

    ``carry_annual = 0`` is the null: the carry leg is then just cash + noise and its
    excess-of-cash mean/Sharpe/HAC-t must NOT fire. Business-day index, span well under
    the pandas ns-Timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    cash_d = cash_annual / 252.0
    cash = np.full(n_days, cash_d)
    carry = cash + carry_annual / 252.0 + rng.normal(0.0, carry_vol / np.sqrt(252), n_days)
    dur = cash + rng.normal(0.0, dur_vol / np.sqrt(252), n_days)
    return pd.DataFrame({"cash": cash, "carry": carry, "dur": dur}, index=idx)
