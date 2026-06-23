"""Data layer for Study 398 — Entropy-Efficiency (SPY tape + synthetic control).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for SPY (yfinance, no key), cached under
  ``_cache/spy_prices.csv`` (a one-column CSV indexed by date). From it we build daily log
  returns; the entropy clock in :mod:`entropy_efficiency.strategy` consumes those returns.

* **Synthetic.** A deterministic, fixed-seed generator that toggles a return series between a
  **random** (high-entropy) regime and a **structured** (low-entropy, autocorrelated) regime.
  The low-entropy regime can carry a *planted forward edge* (``edge`` knob): with ``edge=0``
  the structured regime is more predictable but has **no** mean advantage, so the test must
  NOT manufacture significance; with a large ``edge`` the low-entropy regime really does pay,
  so the test must light up. It is the positive control for the whole "low entropy ⇒ tradable
  window" claim, demonstrated on data where we *know* the truth.

Pure numpy + pandas + stdlib for the offline path. ``fetch_spy`` (network) is only used once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "spy_prices.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_spy(start: str = "1995-01-01", end: str | None = None,
              path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY daily adjusted closes via yfinance and cache a one-column CSV.

    Network-only; used once to build ``_cache/spy_prices.csv``. Never imported by the offline
    notebook cells.
    """
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):          # yfinance may return a 1-col frame
        raw = raw.iloc[:, 0]
    out = raw.dropna().to_frame("SPY")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.Series:
    """Load the cached SPY adjusted-close series (index = date)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    col = "SPY" if "SPY" in df.columns else df.columns[0]
    return df[col].astype(float)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cached SPY -> frame with columns ``spy`` (price) and ``ret`` (daily log return)."""
    spy = load_prices(path)
    ret = np.log(spy / spy.shift(1))
    out = pd.DataFrame({"spy": spy, "ret": ret}).dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_returns(n_days: int = 3_000, block: int = 120, edge: float = 0.0,
                      seed: int = 398, mu_daily: float = 0.0003,
                      sig_daily: float = 0.010, period: int = 5,
                      window: int = 40) -> pd.DataFrame:
    """Deterministic return series toggling between RANDOM and STRUCTURED regimes.

    The tape alternates in ``block``-day stretches between:

      * a **random** (high-entropy) regime — iid Gaussian returns, drift ``mu_daily``; and
      * a **structured** (low-entropy) regime — a near-deterministic **zero-sum repeating
        cycle** of period ``period`` days (e.g. ``[+ + − − 0]``) plus a whisper of noise. A
        repeating, self-similar pattern has a **collapsed sample entropy** (highly predictable),
        and because each cycle sums to zero the structured regime carries the **same mean
        return** as the random one over any forward window that spans whole cycles. So *low
        entropy alone buys predictability of shape, not mean return*.

    The ``edge`` knob adds an extra daily drift **only inside the structured regime**:

      * ``edge = 0`` ⇒ the structured regime is far more predictable yet has **no** mean
        advantage — the honest null for the folklore. The regime-split test must NOT reject.
      * a large ``edge`` ⇒ the low-entropy regime genuinely pays and the test must light up.

    ``block`` is kept well above the entropy ``window`` so a window can sit inside one regime.
    (Permutation entropy barely moves even on this clean cycle — exactly the real-tape lesson:
    it is a near-useless regime gauge for returns; this control therefore keys on **sample
    entropy**.) ``regime`` (1 = structured / low-entropy, 0 = random) is returned so the
    notebooks can colour the ground truth. A decorative period index (``pd.period_range``)
    keeps very long series clear of pandas' datetime bounds; it is only a label.
    """
    rng = np.random.default_rng(seed)
    ret = np.empty(n_days)
    regime = np.zeros(n_days, dtype=int)
    pat = np.array([1.0, 1.0, -1.0, -1.0, 0.0]) * (sig_daily * 1.2)   # zero-sum cycle
    if period != 5:
        pat = np.resize(pat, period)
    phase = 0
    structured = False
    for t in range(n_days):
        if t % block == 0:
            structured = not structured            # flip regime every `block` days
            phase = 0
        if structured:
            ret[t] = mu_daily + edge + pat[phase % len(pat)] + rng.normal(0.0, sig_daily * 0.10)
            phase += 1
            regime[t] = 1
        else:
            ret[t] = rng.normal(mu_daily, sig_daily)
            regime[t] = 0

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01", periods=n_days, freq="D")
    return pd.DataFrame({"spy": price, "ret": ret, "regime": regime}, index=idx)
