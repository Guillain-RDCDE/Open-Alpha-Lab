"""Strategy + inference for Study 443 — Volume Profile POC.

The claim: the **point of control** (POC) — the price level with the most traded volume in a
session — acts like a *magnet*. If the next day opens away from yesterday's POC, price will be
"pulled back" to touch it. Traders fade the open toward the POC and target the level.

We test it on a per-session event table (one row per ticker-day) carrying the *prior* day's
POC and the current day's open / high / low / close (no look-ahead — the POC is known at the
prior close). Two questions, kept separate:

  * **Does price RESPECT the level?** Hit-rate of "the day's range touched the prior POC".
    The honest version asks whether that hit-rate beats a **distance-matched random control
    level** placed the same gap away from the open (the POC's magnetism vs. "any level gets
    touched a lot on a volatile day").

  * **Can you TRADE it?** The fade: at the open, if price is above the POC go short toward it
    (and vice versa); the trade wins if the POC is touched before the close, booking the gap to
    the POC, then we charge a realistic one-way spread on both legs. Intraday levels die to the
    spread — that is the whole capacity story, stated loudly.

Inference (the desk's shared spirit):
  * a one-sample / paired *t* of (POC-touch minus control-touch) per event against 0;
  * a **label-shuffle placebo** — randomize which side of the open the "level" sits on, or
    permute the gaps, and ask how often a random level matches the observed touch edge;
  * a **win-rate with a Wilson interval** vs the random-control base rate;
  * **costs** charged one-way × the fade's turnover (the level dies to the spread).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# default touch tolerance: a touch counts if the day's range comes within TOUCH_TOL of the POC,
# as a fraction of price (1 bp ~ a tick on SPY). 0 = exact range-straddle.
TOUCH_TOL = 0.0


# --------------------------------------------------------------------------- #
# Touch detection
# --------------------------------------------------------------------------- #
def touched(level: np.ndarray, low: np.ndarray, high: np.ndarray,
            tol: float = TOUCH_TOL) -> np.ndarray:
    """Boolean: did the [low, high] range reach ``level`` (within ``tol`` × level)?"""
    pad = tol * np.abs(level)
    return (low - pad <= level) & (level <= high + pad)


def control_level(open_px: np.ndarray, poc: np.ndarray, seed: int = 443) -> np.ndarray:
    """A distance-matched random control level: same |gap| from the open as the POC, but on a
    randomly chosen side. This is the honest benchmark — "a level this far from the open gets
    touched X% of the time by chance" — so the POC has to beat *it*, not beat zero.
    """
    rng = np.random.default_rng(seed)
    dist = np.abs(open_px - poc)
    sign = rng.choice([-1.0, 1.0], size=len(open_px))
    return open_px + sign * dist


# --------------------------------------------------------------------------- #
# Reversion hit-rate: POC vs control
# --------------------------------------------------------------------------- #
def reversion_frame(ev: pd.DataFrame, tol: float = TOUCH_TOL, seed: int = 443,
                    min_gap_bps: float = 5.0) -> pd.DataFrame:
    """Per-event touch outcomes for the POC and a distance-matched control level.

    Only events where the open is at least ``min_gap_bps`` away from the POC are kept (if the
    open is already on the POC the "did it come back?" question is vacuous). Returns the event
    frame with ``poc_touch`` and ``ctrl_touch`` boolean columns and the signed open gap.
    """
    e = ev.copy()
    gap = (e["open"].values - e["poc"].values)
    keep = np.abs(gap) / e["open"].values >= (min_gap_bps / 1e4)
    e = e.loc[keep].reset_index(drop=True)
    op, poc = e["open"].values, e["poc"].values
    lo, hi = e["low"].values, e["high"].values
    ctrl = control_level(op, poc, seed=seed)
    e["poc_touch"] = touched(poc, lo, hi, tol)
    e["ctrl_touch"] = touched(ctrl, lo, hi, tol)
    e["abs_gap_bps"] = np.abs(op - poc) / op * 1e4
    return e


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (centre - half, centre + half)


def paired_t(diff: np.ndarray) -> float:
    """One-sample t of a paired difference vector against 0. NaN if < 2 obs or zero variance."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return float("nan")
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    if se == 0:
        return float("nan")
    return float(diff.mean() / se)


def hac_t(diff: np.ndarray, lags: int = 5) -> float:
    """Newey-West (HAC) t of mean(diff) vs 0 — robust to the day-over-day autocorrelation of
    intraday touch outcomes. Falls back to the plain one-sample t if statsmodels is absent."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 3:
        return float("nan")
    try:
        import statsmodels.api as sm
        X = np.ones((len(diff), 1))
        res = sm.OLS(diff, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(res.params[0] / res.bse[0])
    except Exception:
        return paired_t(diff)


def reversion_stats(ev: pd.DataFrame, tol: float = TOUCH_TOL, seed: int = 443,
                    min_gap_bps: float = 5.0, hac_lags: int = 5) -> dict:
    """Headline reversion test: POC touch-rate vs distance-matched control touch-rate.

    The edge is the *paired* difference (poc_touch − ctrl_touch) per event; we report both
    touch-rates with Wilson intervals, the edge, its HAC t and plain t, and the win-rate of
    the POC vs the control coin flip.
    """
    fr = reversion_frame(ev, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    n = len(fr)
    pt = fr["poc_touch"].values.astype(float)
    ct = fr["ctrl_touch"].values.astype(float)
    diff = pt - ct
    poc_lo, poc_hi = wilson_interval(int(pt.sum()), n)
    ctrl_lo, ctrl_hi = wilson_interval(int(ct.sum()), n)
    return {
        "n": int(n),
        "poc_rate": float(pt.mean()) if n else float("nan"),
        "ctrl_rate": float(ct.mean()) if n else float("nan"),
        "poc_ci": (poc_lo, poc_hi), "ctrl_ci": (ctrl_lo, ctrl_hi),
        "edge": float(diff.mean()) if n else float("nan"),
        "t_paired": paired_t(diff),
        "t_hac": hac_t(diff, lags=hac_lags),
    }


def placebo_pvalue(ev: pd.DataFrame, n_draws: int = 5000, tol: float = TOUCH_TOL,
                   seed: int = 443, min_gap_bps: float = 5.0) -> dict:
    """Shuffle-the-POC placebo: break the link between each session's range and *its own* POC.

    The honest null is "the POC has no special relationship to this session's range — it is
    just a level sitting some distance from the open." We test it by reassigning, on each draw,
    a *shuffled* signed POC-offset (gap, in the open's units) to each session, then asking how
    often a randomly-paired level is touched as often as the real POC. p = P[shuffled touch-rate
    >= observed POC touch-rate]. A real magnet makes the *true* pairing stickier than a random
    one (small p); a non-magnet makes the true pairing indistinguishable from shuffled (p ≈ 0.5
    or worse).

    This respects the empirical joint distribution of intraday ranges and gap sizes — it only
    severs the (session ↔ its POC) bond that the magnet claim depends on.
    """
    fr = reversion_frame(ev, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    op, poc = fr["open"].values, fr["poc"].values
    lo, hi = fr["low"].values, fr["high"].values
    poc_rate = touched(poc, lo, hi, tol).mean()
    offset = poc - op                                 # signed POC offset from the open
    n = len(op)
    rng = np.random.default_rng(seed + 1)
    rates = np.empty(n_draws)
    for i in range(n_draws):
        lvl = op + offset[rng.permutation(n)]         # a randomly-paired level
        rates[i] = touched(lvl, lo, hi, tol).mean()
    # one-sided: is the TRUE POC stickier than a randomly-paired level?
    p = float((rates >= poc_rate).mean())
    return {"obs_edge": float(poc_rate - rates.mean()),
            "placebo_mean": float(rates.mean()),
            "p_value": p, "draws": rates, "poc_rate": float(poc_rate)}


# --------------------------------------------------------------------------- #
# The trade: fade the open toward the POC
# --------------------------------------------------------------------------- #
def fade_trade(ev: pd.DataFrame, cost_bps: float = 1.0, tol: float = TOUCH_TOL,
               seed: int = 443, min_gap_bps: float = 5.0) -> dict:
    """Fade the open toward the prior POC; exit at the POC if touched, else at the close.

    At the open, if the open is *above* the POC we short toward it (target = POC); if *below*,
    we go long toward it. The trade books:
       * if the POC is touched during the session → the move from open to POC (the target gap);
       * else → the move from open to close in the fade direction (let it run / stop at close).
    We charge ``cost_bps`` one-way on entry and exit (2 legs) — the spread the level dies to.

    Returns gross / net mean per-trade return and the touch (target-hit) rate.
    """
    fr = reversion_frame(ev, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    op = fr["open"].values
    poc = fr["poc"].values
    cl = fr["close"].values
    touch = fr["poc_touch"].values
    # fade direction: +1 if open below POC (go long), -1 if open above POC (go short)
    side = np.where(op < poc, 1.0, -1.0)
    target_ret = side * (poc - op) / op           # booked if POC touched
    close_ret = side * (cl - op) / op             # booked if not touched
    gross = np.where(touch, target_ret, close_ret)
    c = cost_bps / 1e4
    net = gross - 2.0 * c                          # entry + exit
    n = len(fr)
    return {
        "n": int(n),
        "hit_rate": float(touch.mean()) if n else float("nan"),
        "gross_mean": float(gross.mean()) if n else float("nan"),
        "net_mean": float(net.mean()) if n else float("nan"),
        "gross_t": paired_t(gross),
        "net_t": paired_t(net),
        "cost_bps": cost_bps,
        "breakeven_bps": float(gross.mean() / 2.0 * 1e4) if n else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(ev: pd.DataFrame, tol: float = TOUCH_TOL, seed: int = 443,
                   n_draws: int = 5000, min_gap_bps: float = 5.0) -> dict:
    """Full gauntlet: reversion stats, placebo, and the fade trade net of costs."""
    rs = reversion_stats(ev, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    pl = placebo_pvalue(ev, n_draws=n_draws, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    fd = fade_trade(ev, cost_bps=1.0, tol=tol, seed=seed, min_gap_bps=min_gap_bps)
    return {"reversion": rs, "placebo": pl, "fade": fd}
