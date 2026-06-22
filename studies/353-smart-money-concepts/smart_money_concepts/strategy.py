"""Strategy + inference for Study 353 -- Smart-Money-Concepts (ICT order blocks & FVGs).

The viral "ICT / Smart Money Concepts" YouTube meme claims price leaves *footprints* of
institutional activity: it "respects order blocks" and "fills fair-value gaps". We test two
falsifiable claims against the proper null -- a matched random walk, which **also** produces
gaps and bounces:

* **(a) Fair-value gaps (FVG).** A 3-bar gap: a *bullish* FVG is a bar whose low sits above
  the high two bars back (an untouched void below price); a *bearish* FVG is the mirror. The
  claim: FVGs get "filled" (price returns into the void) more / faster than a random same-size
  price zone would. Test = fill-rate within H bars, and median time-to-fill, vs (i) a matched
  random-walk synthetic and (ii) a within-tape placebo of random same-width zones.

* **(b) Order blocks (OB).** The last opposite-colour candle before an impulsive move; ICT
  says forward returns from an OB zone beat returns from random levels. Test = mean forward
  H-day return entering when price *revisits* an OB zone, vs entering at random bars (the
  matched control), with a t-stat on the *difference*.

The honest spine: **a random walk has FVGs and bounces too.** So the Signal axis is never the
raw fill-rate (a high number proves nothing); it is the *excess* over the random-walk null, and
the t-stat is on that excess / on the OB-minus-random difference. Costs (one-way bps x NAV) hit
the Tradability axis. Pure numpy/pandas/scipy; deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Fair-value-gap detection (3-bar)
# --------------------------------------------------------------------------- #
def find_fvgs(df: pd.DataFrame, min_rel: float = 0.0) -> pd.DataFrame:
    """Detect 3-bar fair-value gaps.

    A *bullish* FVG completes at bar ``i`` when ``low[i] > high[i-2]`` -- the candle bodies/
    wicks of bars ``i-2`` and ``i`` do not overlap, leaving a void ``(high[i-2], low[i])``
    below current price. A *bearish* FVG is the mirror: ``high[i] < low[i-2]``, void
    ``(high[i], low[i-2])`` above.

    ``min_rel`` filters tiny gaps: keep only gaps whose width is >= ``min_rel`` x the bar-``i``
    close (e.g. 0.001 = 0.1%). Returns a frame with one row per gap: bar index ``i``, gap
    bounds ``lo``/``hi``, ``dir`` (+1 bull / -1 bear), and ``width_rel``.
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    rows = []
    for i in range(2, n):
        # bullish: gap between high[i-2] and low[i]
        if low[i] > h[i - 2]:
            lo, hi, d = h[i - 2], low[i], +1
        elif h[i] < low[i - 2]:
            lo, hi, d = h[i], low[i - 2], -1
        else:
            continue
        width_rel = (hi - lo) / c[i]
        if width_rel < min_rel:
            continue
        rows.append((i, lo, hi, d, width_rel))
    return pd.DataFrame(rows, columns=["i", "lo", "hi", "dir", "width_rel"])


def _fill_one(h: np.ndarray, low: np.ndarray, n: int, i: int,
              lo: float, hi: float, horizon: int) -> tuple[bool, int]:
    """Did the price range touch into ``[lo, hi]`` within ``horizon`` bars after ``i``?

    Returns ``(filled, bars_to_fill)`` -- ``bars_to_fill`` is the number of bars from ``i`` to
    first touch (``horizon + 1`` sentinel if never filled within the horizon). A "touch" is the
    bar's [low, high] range overlapping the gap zone (range_low <= hi and range_high >= lo).
    """
    end = min(n, i + 1 + horizon)
    for j in range(i + 1, end):
        if low[j] <= hi and h[j] >= lo:
            return True, j - i
    return False, horizon + 1


def fvg_fill_stats(df: pd.DataFrame, horizon: int = 20, min_rel: float = 0.001,
                   gaps: pd.DataFrame | None = None) -> dict:
    """Fill-rate and time-to-fill for the FVGs on a tape.

    For each gap, look forward up to ``horizon`` bars and record whether price touched the gap
    zone and how many bars it took. Returns ``n`` (gaps tested), ``fill_rate`` (fraction filled
    within ``horizon``), ``median_ttf`` (median bars-to-fill among *filled* gaps), and the
    per-gap boolean/time arrays for downstream inference.
    """
    if gaps is None:
        gaps = find_fvgs(df, min_rel=min_rel)
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    filled = np.zeros(len(gaps), dtype=bool)
    ttf = np.full(len(gaps), horizon + 1, dtype=float)
    # only count gaps with a full horizon of forward bars available
    valid = []
    for k, (i, lo, hi) in enumerate(zip(gaps["i"], gaps["lo"], gaps["hi"])):
        if i + horizon >= n:
            continue
        f, t = _fill_one(h, low, n, int(i), float(lo), float(hi), horizon)
        filled[k] = f
        ttf[k] = t
        valid.append(k)
    valid = np.array(valid, dtype=int)
    if len(valid) == 0:
        return {"n": 0, "fill_rate": float("nan"), "median_ttf": float("nan"),
                "filled": np.array([]), "ttf": np.array([])}
    fv = filled[valid]
    tv = ttf[valid]
    return {
        "n": int(len(valid)),
        "fill_rate": float(fv.mean()),
        "median_ttf": float(np.median(tv[fv])) if fv.any() else float("nan"),
        "filled": fv,
        "ttf": tv,
    }


def placebo_zone_fill(df: pd.DataFrame, gaps: pd.DataFrame, horizon: int = 20,
                      seed: int = 353) -> dict:
    """Within-tape placebo: random same-width zones at random anchor bars.

    For each detected gap we synthesise a *matched* control zone -- same width, same direction
    relative to the anchor close, but placed at a uniformly random bar of the tape. If the
    real FVG fill-rate is no higher than this placebo, "the gap fills" is just "any same-size
    void near price gets touched", i.e. geometry, not smart money.
    """
    rng = np.random.default_rng(seed)
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    filled = []
    for (i, lo, hi, d) in zip(gaps["i"], gaps["lo"], gaps["hi"], gaps["dir"]):
        i = int(i)
        if i + horizon >= n:
            continue
        width = float(hi - lo)
        # anchor at a random bar (leaving horizon room), zone offset on the same side
        j0 = int(rng.integers(2, n - horizon - 1))
        anchor = c[j0]
        if d > 0:      # bullish gap sits below the close -> void below anchor
            z_hi = anchor - 0.5 * width
            z_lo = z_hi - width
        else:
            z_lo = anchor + 0.5 * width
            z_hi = z_lo + width
        f, _ = _fill_one(h, low, n, j0, z_lo, z_hi, horizon)
        filled.append(f)
    filled = np.array(filled, dtype=bool)
    return {"n": int(len(filled)),
            "fill_rate": float(filled.mean()) if len(filled) else float("nan"),
            "filled": filled}


# --------------------------------------------------------------------------- #
# Order blocks
# --------------------------------------------------------------------------- #
def find_order_blocks(df: pd.DataFrame, impulse: int = 3,
                      impulse_rel: float = 0.015) -> pd.DataFrame:
    """Detect ICT-style order blocks.

    A *bullish* OB is the last **down** candle (close < open) immediately before an up-impulse:
    the next ``impulse`` bars rise by at least ``impulse_rel`` (cumulative close-to-close). The
    OB zone is that candle's ``[low, high]``. A *bearish* OB is the mirror (last up candle
    before a down-impulse). Returns bar index ``i`` of the OB candle, zone ``lo``/``hi``, and
    ``dir`` (+1 bull / -1 bear).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    rows = []
    for i in range(n - impulse - 1):
        fwd = (c[i + impulse] - c[i]) / c[i]
        down_candle = c[i] < o[i]
        up_candle = c[i] > o[i]
        if down_candle and fwd >= impulse_rel:
            rows.append((i, low[i], h[i], +1))
        elif up_candle and fwd <= -impulse_rel:
            rows.append((i, low[i], h[i], -1))
    return pd.DataFrame(rows, columns=["i", "lo", "hi", "dir"])


def ob_forward_returns(df: pd.DataFrame, obs: pd.DataFrame, horizon: int = 10,
                       lag: int = 1) -> dict:
    """Forward H-bar returns from *revisits* to an order-block zone (directional).

    For each OB, scan bars after the impulse; the first bar whose range touches the OB zone is
    a "revisit". We enter at the **next** bar's close (``lag = 1`` execution lag) in the OB's
    direction and hold ``horizon`` bars. Returns are signed by ``dir`` so a working bullish OB
    (price bounces up off the zone) yields a positive number. Returns the per-trade signed
    returns plus mean/t-stat.
    """
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    rets = []
    for (i, lo, hi, d) in zip(obs["i"], obs["lo"], obs["hi"], obs["dir"]):
        i = int(i)
        revisit = None
        for j in range(i + 2, n):           # after the impulse window
            if low[j] <= hi and h[j] >= lo:
                revisit = j
                break
        if revisit is None:
            continue
        entry = revisit + lag
        exit_ = entry + horizon
        if exit_ >= n:
            continue
        r = (c[exit_] - c[entry]) / c[entry] * d
        rets.append(r)
    rets = np.asarray(rets, dtype=float)
    return _summ(rets)


def random_entry_returns(df: pd.DataFrame, n_entries: int, horizon: int = 10,
                         lag: int = 1, seed: int = 353) -> dict:
    """Matched control: forward H-bar returns from ``n_entries`` random bars, signed +1/-1
    at random (so the control has the same long/short mix and the same holding period as the
    OB book but no level information). This is the null the OB edge must beat."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy()
    n = len(df)
    js = rng.integers(2, n - horizon - lag - 1, size=n_entries)
    signs = rng.choice([-1.0, 1.0], size=n_entries)
    entry = js + lag
    exit_ = entry + horizon
    r = (c[exit_] - c[entry]) / c[entry] * signs
    return _summ(np.asarray(r, dtype=float))


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def _summ(r: np.ndarray) -> dict:
    """Mean, std, t-stat vs 0 and n for a return array."""
    n = len(r)
    if n < 2:
        return {"n": n, "mean": float("nan"), "std": float("nan"), "t": float("nan"),
                "rets": r}
    m = float(r.mean())
    s = float(r.std(ddof=1))
    t = m / (s / np.sqrt(n)) if s > 0 else float("nan")
    return {"n": n, "mean": m, "std": s, "t": float(t), "rets": r}


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-proportion z-test (fill-rate vs control fill-rate). Pooled SE."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float((p1 - p2) / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t-test of mean(a) - mean(b) (OB book vs random control)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"diff": float(np.mean(a) - np.mean(b)), "t": float(t), "p": float(p),
            "na": len(a), "nb": len(b)}


def net_of_costs(rets: np.ndarray, cost_bps: float, legs: int = 2) -> dict:
    """Charge ``cost_bps`` one-way per leg (entry + exit = 2 legs) on NAV and re-summarise.

    A round-trip costs ``legs * cost_bps / 1e4`` of NAV; we subtract it from every trade's
    return (each trade is one entry + one exit). The Tradability axis lives here.
    """
    cost = legs * cost_bps / 1e4
    return _summ(rets - cost)
