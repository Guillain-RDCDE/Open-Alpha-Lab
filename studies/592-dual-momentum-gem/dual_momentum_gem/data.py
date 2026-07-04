"""Data layer for Study 592 — Dual Momentum (GEM).

Two sources, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for the four ETFs of the classic
  Antonacci GEM implementation — **SPY** (US equities), **EFA** (developed ex-US equities),
  **AGG** (US aggregate bonds, the "safe" leg) with **IEF** (7-10y Treasuries) back-filling the
  bond leg before AGG's Sept-2003 inception — plus **^IRX** (13-week T-bill discount yield) as
  the risk-free / absolute-momentum hurdle. Everything from yfinance (public, no key), cached
  under the study's own ``_cache/`` as CSV, and every computation runs cache-first.

* **Synthetic world.** A deterministic, fixed-seed monthly generator with a TUNABLE planted
  effect: a two-state (bull/bear) Markov regime drives both equity legs, and the knob
  ``persistence`` controls how sticky regimes are. With ``persistence = 0`` regimes are i.i.d.
  coin flips — past 12-month returns carry **no** information, so GEM must NOT beat
  buy-and-hold (the null). With high persistence, regimes last years — trend-following is
  genuinely informative and GEM must dodge the planted bears (the positive control). The
  synthetic index is a monthly ``period_range`` spanning 80 years (well under the 250-year
  ns-Timestamp ceiling).

Monthly construction (the honest bits):

* Month-end closes -> simple monthly total returns; the final **partial** month is dropped, and
  every headline run is sliced to the frozen ``AS_OF`` so the sample cannot creep.
* The risk-free return for month *t* is the ^IRX yield observed at the **end of month t-1**
  (annual % / 100 / 12) — known in advance, no look-ahead.
* The bond leg is AGG's monthly return where available, IEF's before that (documented splice).

Pure numpy + pandas for the offline path; ``fetch()`` (network) is used once to build the cache.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "gem_prices.csv")
IRX_CACHE = os.path.join(CACHE_DIR, "gem_irx.csv")

TICKERS = ["SPY", "EFA", "AGG", "IEF"]

# Frozen as-of: June 2026 is the last complete month on a 2026-07-03 run.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1998-01-01", prices_path: str = PRICES_CACHE,
          irx_path: str = IRX_CACHE, retries: int = 3) -> None:
    """Download the ETF closes + ^IRX once and cache them. Network-only; never used offline."""
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            px = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Close"]
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    px = px.dropna(how="all")
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    px.to_csv(prices_path)

    irx = None
    for _ in range(retries):
        try:
            irx = yf.download("^IRX", start=start, auto_adjust=False, progress=False)["Close"]
            if irx is not None and len(irx) > 0:
                break
        except Exception:
            time.sleep(2.0)
    irx = irx.dropna(how="all")
    irx.columns = ["IRX"]
    irx.to_csv(irx_path)


def have_real(prices_path: str = PRICES_CACHE, irx_path: str = IRX_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(irx_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_irx(path: str = IRX_CACHE) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df["IRX"].astype(float)


def monthly_panel(asof: str | None = AS_OF) -> pd.DataFrame:
    """The cache-first monthly panel every real-tape number is computed from.

    Columns: ``SPY``, ``EFA`` (monthly total returns), ``BOND`` (AGG, IEF-spliced before AGG's
    first monthly return), ``RF`` (T-bill return for the month, from the **prior** month-end
    ^IRX yield — no look-ahead). Index = month-end timestamps. The final partial month is
    dropped and the sample is sliced to ``asof`` so headline runs never creep.
    """
    px = load_prices()
    irx = load_irx()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
        irx = irx[irx.index <= pd.Timestamp(asof)]

    m = px.resample("ME").last()
    # drop a trailing partial month (tape ends before that month-end)
    if px.index.max() < m.index.max():
        m = m.iloc[:-1]
    ret = m.pct_change()

    irx_m = irx.resample("ME").last().reindex(m.index).ffill()
    rf = (irx_m / 100.0 / 12.0).shift(1)          # yield known at END of prior month

    out = pd.DataFrame(index=ret.index)
    out["SPY"] = ret["SPY"]
    out["EFA"] = ret["EFA"]
    out["BOND"] = ret["AGG"].where(ret["AGG"].notna(), ret["IEF"])   # documented splice
    out["RF"] = rf
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted regime persistence (the machinery control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 960, persistence: float = 0.0,
                    seed: int = 592) -> pd.DataFrame:
    """Deterministic monthly panel with a TUNABLE planted trend effect.

    A two-state Markov regime (bull/bear) drives both equity legs:

    * bull: mean +2.5%/mo; bear: mean -1.7%/mo (equities), sigma 4%/mo, correlated 0.75. The
      UNCONDITIONAL equity mean (+0.4%/mo) stays above the bond mean (+0.25%/mo) in every
      world, so the null cannot reward hiding in bonds mechanically — only *timing* can win.
    * ``persistence`` in [0, 1): P(stay in current regime) = 0.5 + 0.5 * persistence.
      ``0.0`` -> i.i.d. regimes: **no** planted effect, past returns are uninformative and GEM
      must not beat buy-and-hold (it pays the timing drag instead). ``0.94`` -> regimes last
      ~2.8 years on average: a strong planted trend that 12-month momentum must exploit.
    * BOND: i.i.d. N(+0.25%, 1.2%) per month; RF constant 0.15%/mo.

    Index = 80 years of month-end stamps from a ``period_range`` (safe under the 250-year
    ns-Timestamp ceiling). Fully deterministic given ``seed``.
    """
    if n_months > 3_000:
        raise ValueError("keep the synthetic span under 250 years")
    rng = np.random.default_rng(seed)
    p_stay = 0.5 + 0.5 * float(persistence)

    state = np.zeros(n_months, dtype=int)          # 0 = bull, 1 = bear
    state[0] = 0
    u = rng.random(n_months)
    for t in range(1, n_months):
        state[t] = state[t - 1] if u[t] < p_stay else 1 - state[t - 1]

    mu = np.where(state == 0, 0.025, -0.017)
    z_common = rng.normal(0.0, 1.0, n_months)
    z_a = rng.normal(0.0, 1.0, n_months)
    z_b = rng.normal(0.0, 1.0, n_months)
    rho = 0.75
    sig = 0.04
    eq_a = mu + sig * (np.sqrt(rho) * z_common + np.sqrt(1 - rho) * z_a)
    eq_b = 0.9 * mu + sig * (np.sqrt(rho) * z_common + np.sqrt(1 - rho) * z_b)
    bond = rng.normal(0.0025, 0.012, n_months)

    pidx = pd.period_range("2000-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    return pd.DataFrame({"SPY": eq_a, "EFA": eq_b, "BOND": bond,
                         "RF": np.full(n_months, 0.0015)}, index=idx)
