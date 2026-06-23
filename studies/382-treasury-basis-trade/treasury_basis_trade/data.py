"""Data layer for Study 382 — Treasury cash-futures Basis Trade.

Two sources, both offline-friendly:

* **Real tape.** Daily closes for three free yfinance series, cached under
  ``_cache/basis_inputs.csv`` (a wide CSV indexed by date):

    - ``^TNX`` — the 10-year Treasury yield (percent). Stands in for the **cash bond's
      yield** (the carry the long leg earns).
    - ``^IRX`` — the 13-week T-bill yield (percent). Stands in for the **repo / funding
      rate** the levered position pays on the short cash it borrows. (An explicit proxy:
      true GC repo is not free; the bill rate tracks it closely outside acute stress.)
    - ``IEF`` — the 7-10y Treasury ETF (total-return-ish). A sanity/illustration series
      for the cash leg's mark-to-market; the headline model does not depend on it.

  From these we build a **transparent carry / implied-repo model** of the basis trade:
  the daily net carry is ``(yield − funding)/252`` (earned with a 1-day lag — no
  look-ahead), and the residual mark-to-market is a small-duration *basis wobble* proxied
  by ``−DUR_RESID × Δ(yield − funding)``. This is a *model*, labelled as such everywhere;
  it captures the two forces that matter — a smooth positive carry and a fat-tailed
  convergence/mark-to-market — without pretending to be a live cash-vs-futures basis feed.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable
  carry-plus-jump return series with a **planted** unlevered Sharpe. It is the positive
  control: with the planted edge set to zero the harness must NOT manufacture a high
  Sharpe / significant HAC *t*, and with a large planted edge it must light up — proving
  the inference is faithful, and that the **leverage-invariance** of Sharpe (the whole
  lesson) is mechanical, not an artefact.

Pure numpy + pandas + stdlib for the offline path. ``fetch_inputs`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basis_inputs.csv")

# Free yfinance series used to build the carry model. ^TNX / ^IRX are quoted in PERCENT
# (e.g. 4.13 == 4.13%); IEF is a price level. This is a PROXY/MODEL for the cash-futures
# basis — named as such on the Signal axis (the carry is a term-premium proxy, the funding
# leg a repo proxy, the mark-to-market a small-duration residual).
TICKERS = ["^TNX", "^IRX", "IEF"]

# Residual duration of the (duration-hedged) cash-futures basis position, in years. The
# basis trade is long cash / short futures, so most duration nets out; a small residual
# remains and carries the mark-to-market wobble. 0.5y is a deliberately modest, transparent
# choice — the verdict does not hinge on it (it only scales the tail, not the carry).
DUR_RESID = 0.5


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_inputs(start: str = "2002-07-30", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download ^TNX, ^IRX, IEF via yfinance and cache a wide close CSV.

    Network-only; used once to build ``_cache/basis_inputs.csv``. Never imported by the
    offline notebook cells. Start is the earliest common date (IEF lists 2002-07-30).
    """
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    out = raw[[c for c in TICKERS if c in raw.columns]].dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_inputs(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide close frame (index = date, columns = ^TNX, ^IRX, IEF)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def basis_model(inputs: pd.DataFrame, dur_resid: float = DUR_RESID) -> pd.DataFrame:
    """Build the transparent basis-carry model from the raw yields.

    Returns a frame indexed by date with columns:
      * ``yield10``   — 10y yield (%, the cash-bond carry the long leg earns)
      * ``funding``   — 13w bill yield (%, the repo proxy the position pays)
      * ``net_carry`` — yield10 − funding (%, annual net carry before leverage)
      * ``carry``     — daily carry earned with a 1-day lag: net_carry(t−1)/100/252
      * ``mtm``       — residual mark-to-market: −dur_resid × Δ(net_carry/100)
      * ``ret``       — UNLEVERED daily basis P&L per unit notional = carry + mtm

    ``ret`` is the object everything else is computed on; leverage is applied downstream
    (Sharpe is leverage-invariant, which is the whole point). The 1-day lag on the carry
    means the signal known at the close of t earns t+1 — one shift, applied once.
    """
    y10 = inputs["^TNX"].astype(float)
    funding = inputs["^IRX"].astype(float)
    net_carry = y10 - funding
    carry = (net_carry / 100.0 / 252.0).shift(1)          # 1-day lag, no look-ahead
    mtm = -dur_resid * (net_carry / 100.0).diff()         # residual MtM (basis wobble)
    out = pd.DataFrame({
        "yield10": y10, "funding": funding, "net_carry": net_carry,
        "carry": carry, "mtm": mtm, "ret": carry + mtm,
    }).dropna()
    return out


def load_real(path: str = DEFAULT_CACHE, dur_resid: float = DUR_RESID) -> pd.DataFrame:
    """Convenience: cached inputs -> basis-model frame in one call."""
    return basis_model(load_inputs(path), dur_resid=dur_resid)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_basis(n_days: int = 6_000, edge_sharpe: float = 0.0, seed: int = 382,
                    carry_bp_day: float = 0.6, jump_prob: float = 0.01,
                    jump_scale: float = 0.0040) -> pd.DataFrame:
    """Deterministic unlevered basis-return series with a PLANTED unlevered Sharpe.

    Builds a daily return ``ret`` = a small steady **carry** (``carry_bp_day`` basis points
    per day, the smooth "free" income) + Gaussian noise + occasional **adverse jumps** (the
    funding-shock tail: with probability ``jump_prob`` per day, a fat *negative* shock of
    scale ``jump_scale``). The mean carry is then scaled so the *unlevered* annualised Sharpe
    equals ``edge_sharpe`` exactly (when ``edge_sharpe`` > 0); with ``edge_sharpe = 0`` the
    carry is removed and the series is pure mean-zero noise+jumps — the harness must then
    report a Sharpe ≈ 0 and an insignificant HAC *t* (no manufactured edge).

    The control proves two things: (1) the inference is faithful (recovers a planted Sharpe,
    finds nothing when there is nothing), and (2) the lesson — Sharpe is **leverage-invariant**
    — is mechanical: levering this series N× multiplies mean and vol identically, leaving the
    Sharpe fixed while the drawdown explodes. Returned columns mirror ``basis_model``'s
    ``ret`` (the rest are NaN-free decorative carry/mtm split).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.00030, size=n_days)              # diffusive basis noise
    jumps = np.where(rng.random(n_days) < jump_prob,
                     -np.abs(rng.normal(0.0, jump_scale, size=n_days)), 0.0)
    base = noise + jumps                                       # mean ~ 0, fat left tail
    if edge_sharpe and edge_sharpe != 0.0:
        sig = base.std()
        target_mean = edge_sharpe * sig / np.sqrt(252.0)      # daily mean for the Sharpe
        ret = base - base.mean() + target_mean
    else:
        ret = base - base.mean()                              # pure noise+jumps, Sharpe ~ 0
    # decorative carry / mtm split (carry = the steady part, mtm = the wobble)
    carry = np.full(n_days, max(ret.mean(), 0.0))
    mtm = ret - carry
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    return pd.DataFrame({"yield10": np.nan, "funding": np.nan, "net_carry": np.nan,
                         "carry": carry, "mtm": mtm, "ret": ret}, index=idx)
