"""Data layer for Study 655 — Ivy Portfolio.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for Faber's five-asset "Ivy"
  quintet — **VTI** (US total-market equity), **VEU** (all-world ex-US equity), **VNQ** (US
  REITs), **AGG** (US aggregate bonds) and **DBC** (broad commodities) — plus **BIL** (1-3
  month T-bill ETF), which does double duty as the cash leg the 10-month-SMA overlay parks an
  asset in when it's below its trend, AND as the risk-free rate for every excess-of-cash
  Sharpe. Everything from yfinance (public, no key), cached under the study's own ``_cache/``
  as CSV, and every computation runs cache-first.

* **Synthetic world.** A deterministic, fixed-seed monthly generator: five assets driven by a
  shared two-state (trending / mean-reverting) regime with a TUNABLE ``persistence`` knob.
  ``persistence = 0`` makes monthly moves i.i.d. — a 10-month SMA carries no information, so
  the timing overlay must NOT manufacture a drawdown/Sharpe edge over a matched-exposure random
  timer. High ``persistence`` plants genuine multi-month trends the SMA rule can ride. This is
  the machinery proof for the "risk-reduction vs alpha" question — never cited for the
  real-tape stamp.

Monthly construction (the honest bits):

* Month-end closes -> simple monthly total returns; the trailing **partial** month is dropped,
  and every headline run is sliced to the frozen ``AS_OF`` so the sample cannot creep.
* BIL's 2007-05-30 inception is the binding constraint on the joint window (later than VEU's
  2007-03-08 and DBC's 2006-02-06) — the real tape starts at BIL's first full monthly return,
  **2007-06-30**.
* The 10-month SMA needs 10 months of price history *before* the first signal, plus the
  mandatory one-month shift (signal at month t-1's close decides the position held during month
  t) — so the timed engine's first live position is ~11 months into the tape (2008-05).

Pure numpy + pandas for the offline path; ``fetch()`` (network) is used once to build the cache.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "ivy_prices.csv")

ASSETS = ["VTI", "VEU", "VNQ", "AGG", "DBC"]   # the five Ivy sleeves
CASH = "BIL"                                    # cash leg + risk-free proxy
TICKERS = ASSETS + [CASH]

SMA_WINDOW = 10           # Faber's 10-month SMA
AS_OF = "2026-06-30"      # last complete month at publication (2026-07-10 run date)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2003-01-01", path: str = PRICES_CACHE, retries: int = 3) -> None:
    """Download the 6 ETF closes once (auto-adjusted, i.e. total-return) and cache them.

    Network-only; never called from the offline/notebook path.
    """
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            px = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Close"]
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    px = px[TICKERS].dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_csv(path)


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()[TICKERS]


def monthly_panel(asof: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cache-first monthly panel every real-tape number is computed from.

    Returns ``(returns, closes)``:

    * ``returns`` — simple monthly total returns for VTI/VEU/VNQ/AGG/DBC/BIL, index = month-end.
      The trailing partial month is dropped; the sample is sliced to ``asof``.
    * ``closes`` — the month-end price levels for the same window (used to build the SMA
      signal); rows with any missing ticker are dropped, so the joint window starts at BIL's
      inception (2007-05), the binding constraint.
    """
    px = load_prices()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    if px.index.max() < m.index.max():         # drop a trailing partial month
        m = m.iloc[:-1]
    m = m.dropna(how="any")                     # joint window: all 6 tickers present
    ret = m.pct_change().dropna(how="all")
    return ret, m


# --------------------------------------------------------------------------- #
# Synthetic world — planted trend persistence (the machinery control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 480, persistence: float = 0.0,
                    seed: int = 655) -> pd.DataFrame:
    """Deterministic monthly 5-asset + cash panel with a TUNABLE planted trend.

    A single two-state Markov regime ("trend up" / "trend down") drives all five risky legs
    with different loadings and idiosyncratic noise — loosely mimicking equities/REITs moving
    together, bonds/commodities partially offsetting. ``persistence`` sets
    ``P(stay in regime) = 0.5 + 0.5 * persistence``:

    * ``persistence = 0.0`` -> i.i.d. months, no serial structure. A 10-month SMA carries zero
      information about the next month; the timing overlay must not out-Sharpe / out-drawdown a
      matched-exposure random timer here (the null).
    * ``persistence -> 1`` -> regimes last years; the SMA rule can genuinely ride the trend and
      dodge the down-regime (the planted positive control).

    Critically, each asset's regime means are built as ``mu_avg +/- boost``: the UNCONDITIONAL
    (persistence=0, i.e. a fair 50/50 coin flip each month) mean stays at a realistic positive
    asset-class premium, comfortably above the cash rate, in every world. So under the null,
    spending more time in cash can only ever *cost* expected return — the timing rule cannot
    win by mechanically hiding in cash from a negative-mean asset (the trap Study 592 also
    guards against); only genuine trend-following skill can win, and only when trends are
    persistent. 40 years monthly (well under the 250-year ns-Timestamp ceiling); a
    ``period_range`` avoids the Timestamp trap outright.
    """
    if n_months > 3_000:
        raise ValueError("keep the synthetic span under 250 years")
    rng = np.random.default_rng(seed)
    p_stay = 0.5 + 0.5 * float(persistence)

    state = np.zeros(n_months, dtype=int)          # 0 = trend up, 1 = trend down
    u = rng.random(n_months)
    for t in range(1, n_months):
        state[t] = state[t - 1] if u[t] < p_stay else 1 - state[t - 1]

    # per-asset (VTI, VEU, VNQ, AGG, DBC) unconditional monthly means (realistic annual
    # premia: 8/6/7/4/3%) +/- a regime boost/cut that AVERAGES OUT under a 50/50 null.
    mu_avg = np.array([0.08, 0.06, 0.07, 0.04, 0.03]) / 12.0
    boost = np.array([0.010, 0.010, 0.011, 0.003, 0.008])
    mu_up = mu_avg + boost
    mu_dn = mu_avg - boost
    vol = np.array([0.043, 0.048, 0.055, 0.013, 0.045])

    corr = np.array([
        [1.00, 0.75, 0.55, -0.10, 0.15],
        [0.75, 1.00, 0.50, -0.05, 0.20],
        [0.55, 0.50, 1.00, 0.05, 0.10],
        [-0.10, -0.05, 0.05, 1.00, 0.00],
        [0.15, 0.20, 0.10, 0.00, 1.00],
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_months, 5)) @ chol.T

    mu = np.where(state[:, None] == 0, mu_up[None, :], mu_dn[None, :])
    ret = mu + vol[None, :] * z
    cash_ret = np.full(n_months, 0.0015)            # flat ~1.8%/yr cash proxy

    prices = 100.0 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp(how="end")
    cols = ASSETS
    out = pd.DataFrame(prices, index=idx, columns=cols)
    out[CASH + "_lvl"] = 100.0 * np.exp(np.cumsum(np.log1p(cash_ret)))
    return out
