"""Data layer for Study 380 — Curve-Roll-Down (riding the yield curve).

Two sources, both offline-friendly and the same shape (a tz-naive daily frame of
constant-maturity Treasury **yields**, in percentage points):

* **Real tape.** Daily Treasury constant-maturity yield indices from Yahoo!
  (``^IRX`` 13-week bill, ``^FVX`` 5-year, ``^TNX`` 10-year, ``^TYX`` 30-year),
  cached under ``_cache/curve_yields.csv``. From these we read the curve **level**
  (the short rate = cash) and its **slope** (long minus short), and we approximate a
  smooth curve by interpolating across the four pillars. This is the canonical input
  for the "ride the curve" claim: a steep curve is exactly the setup believers say
  pays you to own duration and roll down.

* **Synthetic.** A deterministic, fixed-seed generator that builds a controllable
  yield curve whose **slope carries a planted amount of predictive power** about the
  forward excess return of a duration sleeve over cash (``edge`` knob). ``edge=0`` is
  the honest null — the curve's slope tells you nothing the term-premium baseline
  doesn't already — and a large ``edge`` MUST light the inference up. It is the
  positive control: the engine must recover a planted edge and must NOT manufacture
  significance when the true edge is zero.

Pure numpy + pandas + stdlib for the offline path. ``fetch_curve`` (network) is only
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "curve_yields.csv")

# Yahoo constant-maturity Treasury yield indices and their maturities (years).
# ^IRX is the 13-week bill (our "cash" / short rate); the rest are the curve pillars.
TICKERS = ["^IRX", "^FVX", "^TNX", "^TYX"]
PILLARS = {"IRX": 0.25, "FVX": 5.0, "TNX": 10.0, "TYX": 30.0}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_curve(start: str = "1990-01-01", end: str | None = None,
                path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the four CMT yield indices via yfinance and cache a wide CSV.

    Network-only; used once to build ``_cache/curve_yields.csv``. Never imported by the
    offline notebook cells. Columns are the de-prefixed ticker names (IRX/FVX/TNX/TYX),
    values are yields in percentage points.
    """
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    raw.columns = [str(c).replace("^", "") for c in raw.columns]
    raw.index.name = "date"
    raw = raw.dropna()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_yields(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide yield frame (index = date, columns = IRX/FVX/TNX/TYX, in %)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def curve_frame(yields: pd.DataFrame, sleeve_mat: float = 5.0,
                short_mat: float = 0.25) -> pd.DataFrame:
    """Build the roll-down working frame from CMT yields.

    Returns a frame indexed by date with columns:

      * ``short``  — the short rate (cash yield), the ``^IRX`` 13-week bill (% p.a.).
      * ``sleeve`` — the yield at the sleeve's target maturity ``sleeve_mat`` (% p.a.),
                     read off the curve (a pillar or log-interpolated between pillars).
      * ``rolled`` — the yield one year *down* the curve, i.e. at maturity
                     ``sleeve_mat - 1`` (% p.a.) — what the sleeve's bond will yield in a
                     year if the curve is unchanged. ``sleeve - rolled`` (>0 on a steep
                     curve) is the **roll-down** in yield terms.
      * ``slope``  — ``sleeve - short`` (% pts): the steepness believers say pays you.

    The curve is interpolated **linearly in maturity** across the four pillars
    (0.25y, 5y, 10y, 30y). Maturities below the lowest / above the highest pillar are
    clamped. This is a coarse 4-pillar curve, named as a proxy for the full CMT curve.
    """
    mats = np.array(sorted(PILLARS.values()))           # [0.25, 5, 10, 30]
    names = sorted(PILLARS, key=lambda k: PILLARS[k])   # ['IRX','FVX','TNX','TYX']
    Y = yields[names].to_numpy()                         # (T, 4) yields at the pillars

    def interp(m: float) -> np.ndarray:
        m = float(np.clip(m, mats[0], mats[-1]))
        # piecewise-linear in maturity across the pillar grid, per row
        return np.array([np.interp(m, mats, row) for row in Y])

    short = yields["IRX"].to_numpy()
    sleeve = interp(sleeve_mat)
    rolled = interp(max(sleeve_mat - 1.0, mats[0]))
    out = pd.DataFrame(
        {"short": short, "sleeve": sleeve, "rolled": rolled,
         "slope": sleeve - short},
        index=yields.index,
    )
    return out.dropna()


def load_real(path: str = DEFAULT_CACHE, sleeve_mat: float = 5.0) -> pd.DataFrame:
    """Convenience: cached yields -> roll-down working frame in one call."""
    return curve_frame(load_yields(path), sleeve_mat=sleeve_mat)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_curve(n_days: int = 6_000, edge: float = 0.0, seed: int = 380,
                    sleeve_mat: float = 5.0) -> pd.DataFrame:
    """Deterministic yield-curve tape with a PLANTED slope->forward-edge knob.

    Builds a mean-reverting short rate (AR(1) around ~2.5%) and an independent,
    mean-reverting curve **slope** (AR(1) around ~+1.2 pts, occasionally inverting).
    The sleeve yield is ``short + slope``; the one-year-rolled yield is
    ``short + slope * (sleeve_mat-1)/sleeve_mat`` (linear curve through the origin of
    the slope). Forward excess returns of the duration sleeve over cash are generated as::

        xs_ret_t = baseline_carry_t + edge * (slope_rank_{t-1} - 0.5) + noise_t

    where ``baseline_carry`` is the mechanical carry+roll term EVERY steep curve gives
    (the term-premium baseline that the null already contains), ``slope_rank`` is the
    out-of-sample rolling percentile of the slope, and ``noise`` is i.i.d. duration vol.

    ``edge = 0`` is the honest null: the slope carries **no** information beyond the
    baseline carry already in the curve, so the inference (excess of high-slope forward
    returns over the carry-matched baseline) must NOT reach significance. A large
    ``edge`` plants a genuine extra relationship the engine must light up. The control
    proves the machinery is unbiased *and* that a bare carry baseline is not an edge.
    """
    rng = np.random.default_rng(seed)

    # short rate: AR(1) around 2.5%, clamped to [0.05, 8]
    short = np.empty(n_days)
    short[0] = 2.5
    for t in range(1, n_days):
        short[t] = 2.5 + 0.995 * (short[t - 1] - 2.5) + rng.normal(0, 0.04)
    short = np.clip(short, 0.05, 8.0)

    # slope: AR(1) around +1.2 pts, can go negative (inversion)
    slope = np.empty(n_days)
    slope[0] = 1.2
    for t in range(1, n_days):
        slope[t] = 1.2 + 0.99 * (slope[t - 1] - 1.2) + rng.normal(0, 0.05)
    slope = np.clip(slope, -2.5, 4.0)

    sleeve = short + slope
    rolled = short + slope * (sleeve_mat - 1.0) / sleeve_mat

    # out-of-sample rolling percentile of the slope (lagged use only)
    idx = pd.bdate_range("1995-01-02", periods=n_days)
    slope_rank = pd.Series(slope, index=idx).rolling(252, min_periods=63).rank(pct=True)

    # forward 1y excess return of the duration sleeve over cash:
    #   mechanical carry+roll baseline (what ANY steep curve gives) + planted edge + noise
    dur = sleeve_mat - 0.5                                  # approx modified duration
    carry = slope / 100.0                                   # the yield pickup over cash
    rolldown = (slope / sleeve_mat) * dur / 100.0           # price gain if curve unchanged
    baseline = carry + rolldown
    noise = rng.normal(0.0, 0.06, n_days)                  # duration vol of the sleeve
    xs = np.empty(n_days)
    for t in range(1, n_days):
        r = slope_rank.iloc[t - 1]
        extra = 0.0 if np.isnan(r) else edge * (r - 0.5)
        xs[t] = baseline[t - 1] + extra + noise[t]
    xs[0] = noise[0]

    return pd.DataFrame(
        {"short": short, "sleeve": sleeve, "rolled": rolled, "slope": slope,
         "xs_fwd1y": xs},
        index=idx,
    )
