"""Data layer for Study 604 — Month-End Rebalancing Flows.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily **total-return** (auto-adjusted) closes from yfinance for

    - ``SPY``   — the equity leg (1993-01-29 →), and
    - a **spliced Aggregate-bond leg**: ``AGG`` from its 2003-09-29 inception,
      ``VBMFX`` (Vanguard Total Bond Market Index, the *same* Bloomberg/Lehman
      Aggregate index, inception 1986) before that.

  The research plan sketched a TLT pre-2003 splice; TLT only reaches 2002-07 and is a
  20+year *long-duration* fund — a poor stand-in for the 60/40 bond leg institutions
  actually rebalance against. VBMFX tracks the identical Aggregate index AGG does and
  covers the whole SPY era, so the splice is **VBMFX → AGG at 2003-09-30**, done in
  *return* space (no level stitching): bond daily returns are VBMFX's through
  2003-09-29 and AGG's from 2003-09-30 on. Documented here and in results.md.

* **Synthetic.** A deterministic, fixed-seed two-asset daily world with a **planted
  rebalancing-flow reversal** (knob ``edge``): in months whose month-to-date
  equity-minus-bond gap (measured through the close 3 days before month-end, exactly
  like the detector) is *positive*, the equity-minus-bond spread is depressed by
  ``edge`` (spread across the last 3 trading days) and boosted by ``edge`` (across the
  first 3 days of the next month); gap-negative months get the mirror image. With
  ``edge = 0`` the world is a pure two-asset random walk: the detector must NOT fire.

Pure numpy + pandas + stdlib for the offline path. ``fetch_tape`` (network) is used
only once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "merf_prices.csv")

# Frozen as-of for the headline run — the sample never creeps as new sessions arrive.
# June 2026 is the last complete calendar month at publication.
AS_OF = "2026-06-30"

TICKERS = ["SPY", "AGG", "VBMFX"]
SPLICE_DATE = "2003-09-30"      # first day the bond return comes from AGG
WINDOW = 3                      # the last-3 / first-3 trading-day windows


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "1993-01-01", end: str | None = None,
               path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download SPY + AGG + VBMFX total-return closes and cache them wide.

    Network-only; used once to build the cache. Never imported by the offline cells.
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
    prices = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame (SPY, AGG, VBMFX), sliced to the frozen as-of."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


def daily_returns(prices: pd.DataFrame,
                  splice_date: str = SPLICE_DATE) -> pd.DataFrame:
    """Daily simple returns: ``eq`` (SPY) and the spliced ``bd`` Aggregate-bond leg.

    The splice is in RETURN space: bond returns are VBMFX's before ``splice_date``
    and AGG's on/after it — no price-level stitching, no fake jump on splice day.
    Rows are kept only where both legs have a return (SPY's 1993-02-01 first return
    starts the sample).
    """
    eq = prices["SPY"].dropna().pct_change()
    vb = prices["VBMFX"].dropna().pct_change()
    ag = prices["AGG"].dropna().pct_change()
    sd = pd.Timestamp(splice_date)
    bd = pd.concat([vb[vb.index < sd], ag[ag.index >= sd]]).sort_index()
    out = pd.DataFrame({"eq": eq, "bd": bd}).dropna()
    return out


def load_real() -> pd.DataFrame:
    """Convenience: cached prices → spliced daily-return frame (eq, bd)."""
    return daily_returns(load_prices())


# --------------------------------------------------------------------------- #
# Synthetic control — a two-asset world with a PLANTED flow reversal
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 360, edge: float = 0.0, seed: int = 604,
                    window: int = WINDOW,
                    mu_eq: float = 0.0004, sig_eq: float = 0.010,
                    mu_bd: float = 0.0002, sig_bd: float = 0.0030) -> pd.DataFrame:
    """Deterministic daily (eq, bd) return frame with a planted month-end reversal.

    The index is a real business-day calendar so the detector's calendar-month
    grouping applies unchanged (~30 years for the default — far below the 250-year
    ns-Timestamp cap). For each calendar month we compute the same conditioning gap
    the detector uses — cumulative eq-minus-bd from the month's first day through the
    close ``window`` days before month-end. If that gap is positive, the planted flow
    SELLS equity into month-end: the eq-minus-bd spread is lowered by ``edge`` total
    across the last ``window`` days and raised by ``edge`` total across the first
    ``window`` days of the next month. Gap-negative months get the mirror image.
    ``edge = 0`` is the null: a pure two-asset random walk the detector must NOT
    fire on.

    The planted effect is applied half to each leg (eq gets ∓edge/2, bd ±edge/2) so
    it is a pure *spread* distortion.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", periods=int(n_months * 21.6))
    eq = rng.normal(mu_eq, sig_eq, len(idx))
    bd = rng.normal(mu_bd, sig_bd, len(idx))

    if edge != 0.0:
        per_day = edge / window
        months = idx.to_period("M")
        uniq = months.unique()
        pos = {m: np.where(months == m)[0] for m in uniq}
        for i, m in enumerate(uniq):
            loc = pos[m]
            if len(loc) <= window + 2:
                continue
            pre = loc[:-window]
            tail = loc[-window:]
            gap = float(eq[pre].sum() - bd[pre].sum())
            sgn = 1.0 if gap > 0 else -1.0
            # last `window` days of month m: spread moved AGAINST the winner
            eq[tail] -= sgn * per_day / 2.0
            bd[tail] += sgn * per_day / 2.0
            # first `window` days of the next month: the reversal
            if i + 1 < len(uniq):
                head = pos[uniq[i + 1]][:window]
                eq[head] += sgn * per_day / 2.0
                bd[head] -= sgn * per_day / 2.0

    return pd.DataFrame({"eq": eq, "bd": bd}, index=idx)
