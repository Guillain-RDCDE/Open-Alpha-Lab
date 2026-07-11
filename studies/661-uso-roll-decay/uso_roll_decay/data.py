"""Data layer for Study 661 — USO-Roll-Decay.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily **USO** (United States Oil Fund, split/dividend-adjusted close — USO
  executed a 1-for-8 reverse split on 2020-04-29, which ``auto_adjust=True`` folds back into a
  single continuous series) and daily **CL=F** (the continuously-rolled NYMEX WTI front-month
  futures contract — the number every news ticker prints as "the price of oil"), both from
  yfinance (no key), cached as CSV under the study's own ``_cache/``. CL=F is *not* physical
  cash spot (no free source exists for that); it is the front-month futures print, which is
  exactly what the folklore's "oil price" means and exactly what USO itself is built from — so
  this is the honest, like-for-like comparison the claim is actually making.

* **Hardcoded contango-stress windows.** Two widely-documented historical episodes of extreme
  WTI contango — the 2008-09 storage glut and the 2020 COVID demand collapse — as a fixed date
  table (facts, no network), used to test whether the roll-decay concentrates in known crisis
  regimes rather than bleeding uniformly. This is a **regime label**, not a curve-fitted split.

A single, well-documented data quirk: the front-month May-2020 WTI contract settled at
**-$37.63** on 2020-04-20 (CL=F prints this negative close) — the only negative-price day on
either tape. Log returns are undefined across a non-positive price, so the two adjacent daily
observations (2020-04-20, 2020-04-21) are automatically dropped from the continuous return
series (``np.log`` returns ``NaN`` for a non-positive price, and pandas' own diff/dropna do the
rest — no manual row deletion, so no adjacency artefact). That single day is instead treated as
its own named case study (see ``strategy.negative_oil_case``).

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
USO_CACHE = os.path.join(CACHE_DIR, "urd_uso.csv")
CLF_CACHE = os.path.join(CACHE_DIR, "urd_clf.csv")

START = "2006-04-10"        # USO inception
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
NEGATIVE_OIL_DAY = "2020-04-20"   # the May-2020 WTI contract's -$37.63 settlement
USO_RESPLIT_DATE = "2020-04-29"   # USO's 1-for-8 reverse split (auto-folded by auto_adjust=True)

# --------------------------------------------------------------------------- #
# Hardcoded contango-stress regime windows — widely-documented historical episodes of
# extreme WTI contango, used as a REGIME LABEL (not a fitted split). Source: EIA "This Week
# in Petroleum" storage reports and contemporaneous WTI term-structure commentary.
# --------------------------------------------------------------------------- #
CONTANGO_STRESS_WINDOWS = (
    ("2008-12-01", "2009-06-30",
     "2009 storage-glut super-contango: post-crash demand collapse filled Cushing storage "
     "near capacity, pushing the WTI curve into one of its steepest contangos on record."),
    ("2020-03-01", "2020-06-30",
     "2020 COVID demand-collapse super-contango: a global storage crisis drove the curve so "
     "deep into contango that the front WTI contract briefly settled negative (2020-04-20)."),
)


def stress_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series, True on dates inside any hardcoded contango-stress window."""
    m = pd.Series(False, index=index)
    for lo, hi, _ in CONTANGO_STRESS_WINDOWS:
        m |= (index >= lo) & (index <= hi)
    return m


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2006-01-01", end: str = "2026-07-01") -> None:
    """Download USO (split/div-adjusted close) and CL=F (raw front-month close); cache. Network."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    uso = yf.download("USO", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(uso.columns, pd.MultiIndex):
        uso.columns = uso.columns.get_level_values(0)
    uso[["Close"]].dropna().to_csv(USO_CACHE)

    # CL=F: a continuously-rolled futures print, no split/dividend concept — raw bars.
    clf = yf.download("CL=F", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(clf.columns, pd.MultiIndex):
        clf.columns = clf.columns.get_level_values(0)
    clf[["Close"]].dropna(how="all").to_csv(CLF_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (USO_CACHE, CLF_CACHE))


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (uso, clf) frames, sliced to [start, asof]."""
    out = []
    for path in (USO_CACHE, CLF_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Synthetic world — planted roll-drag (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seed: int = 2024, n: int = 5080, drag_daily_bps: float = 0.0,
                    stress_extra_bps: float = 0.0, sigma: float = 0.02, mu: float = 0.0002,
                    sigma_track: float = 0.004,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (spot, fund) daily log-return pair with a TUNABLE planted roll-drag.

    ``spot`` is a plain random walk (drift ``mu``, vol ``sigma`` — the "headline oil price").
    ``fund`` (the USO stand-in) equals spot minus a constant per-day drag (``drag_daily_bps``,
    the everyday roll/expense friction) minus an EXTRA drag on two synthetic "stress" blocks
    (``stress_extra_bps``, positioned like the real 2009/2020 windows) plus small independent
    idiosyncratic tracking noise (``sigma_track`` — two related-but-distinct instruments never
    move in perfect lockstep even with zero drag; without it a zero-drag null has an exactly
    zero gap and the t-stat machinery divides by zero). ``drag_daily_bps = stress_extra_bps =
    0`` is the null world: the detector must NOT fire. Returns ``(spot_ret, fund_ret,
    stress_mask)``, all length ``n`` — far below the ns-timestamp span trap (no date index
    needed here, the detector only consumes the return arrays).
    """
    rng = np.random.default_rng(seed)
    spot_r = rng.normal(mu, sigma, n)
    idio = rng.normal(0.0, sigma_track, n)
    stress = np.zeros(n, dtype=bool)
    stress[700:850] = True     # a ~150-day synthetic "2009-like" block
    stress[3500:3600] = True   # a ~100-day synthetic "2020-like" block
    drag = np.full(n, drag_daily_bps / 1e4)
    drag[stress] += stress_extra_bps / 1e4
    fund_r = spot_r - drag + idio
    return spot_r, fund_r, stress
