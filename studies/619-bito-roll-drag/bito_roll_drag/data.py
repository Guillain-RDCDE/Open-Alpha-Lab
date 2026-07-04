"""Data layer for Study 619 — BITO Roll Drag.

Two sources, both offline-friendly once cached:

* **Real tape (yfinance, no key).** Daily **total-return** closes (``auto_adjust=True`` — BITO
  pays very large monthly distributions, so a price-only tape would wildly overstate the drag)
  for four tickers:

  - ``BITO``    — ProShares Bitcoin Strategy ETF (CME front-month bitcoin *futures*, rolls monthly),
  - ``IBIT``    — iShares Bitcoin Trust (spot ETF, listed 2024-01-11),
  - ``BTC-USD`` — spot bitcoin (trades 7 days/week; weekend moves fold into Monday when aligned),
  - ``BTC=F``   — CME front-month bitcoin futures (used only to *classify* contango/backwardation).

  Cached wide as ``_cache/brd_prices.csv``; every analysis runs cache-first and offline.

* **Synthetic world.** A deterministic, seeded generator with a **tunable planted roll drag**: a
  GBM spot, a front-month futures curve carrying an annualized basis (the contango knob) plus a
  small AR(1) basis noise, and a futures ETF whose NAV rolls at each monthly expiry and pays a
  fee. The planted drag is ``basis_ann + fee_ann`` by construction; the **null** (``basis_ann=0,
  fee_ann=0``) must NOT manufacture a significant drag from the basis noise.

Timestamp caveat (named in results): Yahoo's BTC-USD daily close is ~00:00 UTC, BITO/IBIT close
16:00 ET — a few hours' offset that adds noise to *daily* gaps but washes out of the mean drag.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "brd_prices.csv")

TICKERS = ["BITO", "IBIT", "BTC-USD", "BTC=F"]

# BITO listed 2021-10-19; start a hair earlier so the first BITO row is its true first close.
START = "2021-10-01"

# Frozen as-of for every headline number (last complete month at publication).
ASOF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = START, end: str | None = None,
               path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the four-ticker total-return tape and cache it (network; run once)."""
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
    raw.index = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None).normalize()
                                  for t in raw.index])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide total-return close frame (index = calendar date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def align(prices: pd.DataFrame, asof: str | None = ASOF) -> pd.DataFrame:
    """The analysis frame: BITO trading days only, spot sampled on those same dates.

    Columns ``bito, ibit, spot, fut`` (``ibit`` is NaN before 2024-01-11). Because spot trades
    7 days/week, sampling it on BITO's trading days folds weekend moves into Monday — exactly
    the window BITO's own Friday-close -> Monday-close return spans. Sliced to ``asof`` so the
    published sample never creeps.
    """
    px = prices.copy()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    days = px.index[px["BITO"].notna()]
    out = pd.DataFrame({
        "bito": px.loc[days, "BITO"],
        "ibit": px.loc[days, "IBIT"],
        "spot": px.loc[days, "BTC-USD"],
        "fut": px.loc[days, "BTC=F"],
    })
    return out.dropna(subset=["bito", "spot"])


def load_real(path: str = PRICES_CACHE, asof: str | None = ASOF) -> pd.DataFrame:
    """Convenience: cached tape, aligned and as-of'd."""
    return align(load_prices(path), asof=asof)


# --------------------------------------------------------------------------- #
# Roll-window calendar (CME bitcoin futures expire the last Friday of the month)
# --------------------------------------------------------------------------- #
def last_friday(year: int, month: int) -> pd.Timestamp:
    """Last Friday of a calendar month (CME BTC futures termination day)."""
    end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return end - pd.Timedelta(days=(end.weekday() - 4) % 7)


def roll_window_flags(index: pd.DatetimeIndex, width: int = 5) -> np.ndarray:
    """Flag the ``width`` trading days ending on each month's expiry Friday.

    CME bitcoin futures terminate on the last Friday of the contract month, and BITO's
    prospectus rolls the position over the days *leading into* expiry — this window is the
    approximate "roll week". The mechanism prediction worth testing: the toll should NOT
    concentrate here (carry bleeds daily as the premium converges), so a diffuse drag is the
    signature of basis convergence, not of roll-day execution.
    """
    idx = pd.DatetimeIndex(index)
    flag = np.zeros(len(idx), dtype=bool)
    for (y, m) in sorted({(d.year, d.month) for d in idx}):
        lf = last_friday(y, m)
        pos = idx.searchsorted(lf, side="right") - 1  # last trading day <= expiry Friday
        if pos < 0:
            continue
        lo = max(0, pos - width + 1)
        flag[lo:pos + 1] = True
    return flag


# --------------------------------------------------------------------------- #
# Synthetic world — planted roll drag + null
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 8, basis_ann: float = 0.10, fee_ann: float = 0.0095,
                    seed: int = 619, mu_ann: float = 0.20, sig_ann: float = 0.60,
                    basis_sig: float = 0.03, ibit_fee_ann: float = 0.0025) -> pd.DataFrame:
    """Deterministic daily world with a PLANTED futures-roll drag.

    Spot is a GBM. The front-month future is ``F_t = S_t * exp(b_t * tau_t)`` where ``tau_t``
    is the year-fraction to the current front expiry (last Friday of each month) and
    ``b_t = basis_ann + AR(1) noise`` (the contango knob plus realistic basis wobble). The
    futures ETF holds the front contract, rolls at each expiry close, and pays ``fee_ann``:
    its planted annualized drag vs spot is ``basis_ann + fee_ann`` by construction. A spot ETF
    leg pays ``ibit_fee_ann``. The **null** is ``basis_ann=0, fee_ann=0``: basis noise alone
    must not fake a significant drag.

    Span is a few years of business days — far below the 250-year ns-Timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=int(n_years * 252))
    n = len(idx)

    # spot GBM
    z = rng.standard_normal(n)
    r_spot = mu_ann / 252.0 + sig_ann / np.sqrt(252.0) * z
    spot = 100.0 * np.exp(np.cumsum(r_spot))

    # AR(1) basis noise around the planted level
    eps = rng.standard_normal(n) * basis_sig * np.sqrt(1 - 0.97 ** 2)
    noise = np.empty(n)
    noise[0] = 0.0
    for i in range(1, n):
        noise[i] = 0.97 * noise[i - 1] + eps[i]
    b = basis_ann + noise

    # front expiry for each day (next trading day <= last Friday, strictly >= today;
    # on the expiry day itself the contract is still the front until its close)
    expiries = []
    for (y, m) in sorted({(d.year, d.month) for d in idx}):
        lf = last_friday(y, m)
        pos = idx.searchsorted(lf, side="right") - 1
        if pos >= 0:
            expiries.append(idx[pos])
    expiries = pd.DatetimeIndex(sorted(set(expiries)))

    exp_pos = np.searchsorted(expiries.values, idx.values, side="left")
    exp_pos = np.minimum(exp_pos, len(expiries) - 1)
    front_exp = expiries.values[exp_pos]
    tau = (front_exp - idx.values).astype("timedelta64[D]").astype(float) / 365.0

    fut = spot * np.exp(b * tau)

    # futures-ETF NAV: within a contract, NAV compounds F_t/F_{t-1}; across a roll, both
    # legs are valued on the NEW front contract (the roll itself is P&L-neutral); minus fee.
    nav = np.empty(n)
    nav[0] = 100.0
    for i in range(1, n):
        e = front_exp[i]
        tau_prev = (e - idx.values[i - 1]).astype("timedelta64[D]").astype(float) / 365.0
        f_prev = spot[i - 1] * np.exp(b[i - 1] * tau_prev)
        nav[i] = nav[i - 1] * (fut[i] / f_prev) * (1.0 - fee_ann / 252.0)

    ibit = 100.0 * np.exp(np.cumsum(r_spot - ibit_fee_ann / 252.0))

    return pd.DataFrame({"bito": nav, "ibit": ibit, "spot": spot, "fut": fut}, index=idx)
