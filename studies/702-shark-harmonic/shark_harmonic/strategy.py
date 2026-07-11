"""Strategy + inference for Study 702 — Shark-Harmonic.

The claim: **the Shark — Scott Carney's "5-0" pattern — predicts a reversal at a
completion ZONE, not a single point, and does so with NEITHER leg being a
retracement**, unlike every other pattern in the harmonic zoo. Chartists draw four
alternating swing pivots X, A, B, C, then read a fifth point D as a reversal zone:

1. **AB is an EXTENSION of XA, overshooting PAST point X** — B lands 1.13-1.618x the
   length of the XA leg beyond A, i.e. on the *opposite* side of X from where A sits.
   This is the Shark's first break from the rest of the zoo: Gartley/Bat/Crab/
   Butterfly all define B as a *retracement* of XA (B stays between X and A); the
   Shark's B never does. Structural (every well-formed Shark candidate needs it) —
   not itself the hypothesis under test.
2. **BC is a further EXTENSION of AB, overshooting PAST point A** — C lands
   1.618-2.24x the length of AB beyond B, continuing further in the XA direction.
   Also structural, also not under test.
3. **D is a completion ZONE on the ORIGINAL XA leg, 0.886-1.13x its length from X**
   — this is the Shark's defining signature and the specific range the brief asks us
   to test. Unlike the Crab's single 1.618 target or the Bat's single-point 0.886,
   the "5-0" name comes from D straddling both the ~0% extension of X (i.e. close to
   X's own level, scaled by AB — practitioners cite the near-50%-of-BC coincidence)
   and the 0.886-1.13 XA band tested here — a genuine PRICE ZONE, not a point plus
   tolerance. Believers read the B/C overshoot as exhaustion and expect price to
   resume the *original* XA direction once D is touched (``reversal_dir = sign(XA)``).

Pipeline, all real-time (no look-ahead):

1. **Pivot detection.** A percentage-threshold zigzag on daily closes. A pivot is
   only *known* at its **confirmation bar** — the bar where price reverses far
   enough to lock it in — never at the (earlier) extreme bar itself.
2. **XABCD candidates.** Every consecutive confirmed quadruple (X, A, B, C) whose
   AB/XA and BC/AB EXTENSION ratios sit inside the structural bands above, projecting
   a D-ZONE ``[X + 0.886*(A-X), X + 1.13*(A-X)]`` fully known the moment C confirms.
3. **D-zone scan.** From C's confirmation bar forward (a real trader's earliest
   possible knowledge), scan up to ``max_fwd_days`` sessions for the first bar whose
   high-low range overlaps the D zone.
4. **The fade.** Enter at the touch bar's close, in the *predicted reversal
   direction* (the direction that resumes the original XA trend), hold for a fixed
   horizon, exit at that close.

**Two independent honesty checks**, matching the brief:

* **Forward returns at D vs a random-day BASE RATE, Bonferroni-corrected.** The
  base rate redraws, per ticker and many seeds, the *same number* of random entry
  days with the *same* empirical mix of bullish/bearish reversal directions as the
  real Shark touches — the equity-drift confound a naive backtest would otherwise
  credit to the pattern. We test the pooled 5-day headline *and* all six per-ticker
  splits (7 tests total) and apply a Bonferroni correction across them.
* **A placebo arm (the third-axis myth-check).** Identical pivots, but each
  candidate's D-zone is re-centered on a deterministic, seeded location clear of
  the Shark's own 0.886-1.13 band — "does ANY nearby completion zone work, not
  specifically Carney's 0.886-1.13."
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 5, 10)


# --------------------------------------------------------------------------- #
# 1. Pivot detection — percentage zigzag, confirmed pivots only (no look-ahead)
# --------------------------------------------------------------------------- #
def zigzag_pivots(close: np.ndarray, pct: float) -> pd.DataFrame:
    """Confirmed alternating swing pivots (H/L) from a percentage-threshold zigzag.

    A pivot is appended to the result only once price has reversed by ``pct`` from
    the running extreme — i.e. at the **confirmation bar** ``conf_i``, which is
    always strictly after the extreme bar ``piv_i``. The final, still-forming leg
    (not yet confirmed) is dropped: nothing here uses information from the future
    relative to ``conf_i``.
    """
    n = len(close)
    rows: list[tuple[int, int, float, str]] = []
    if n < 2:
        return pd.DataFrame(rows, columns=["piv_i", "conf_i", "price", "kind"])

    ext_i, ext_p = 0, float(close[0])
    trend = 0  # 0 = undetermined, +1 = up (tracking a high), -1 = down (tracking a low)
    for i in range(1, n):
        p = float(close[i])
        if trend == 0:
            if p >= ext_p * (1 + pct):
                trend, ext_i, ext_p = 1, i, p
            elif p <= ext_p * (1 - pct):
                trend, ext_i, ext_p = -1, i, p
        elif trend == 1:
            if p > ext_p:
                ext_p, ext_i = p, i
            elif p <= ext_p * (1 - pct):
                rows.append((ext_i, i, ext_p, "H"))
                trend, ext_i, ext_p = -1, i, p
        else:  # trend == -1
            if p < ext_p:
                ext_p, ext_i = p, i
            elif p >= ext_p * (1 + pct):
                rows.append((ext_i, i, ext_p, "L"))
                trend, ext_i, ext_p = 1, i, p
    return pd.DataFrame(rows, columns=["piv_i", "conf_i", "price", "kind"])


# --------------------------------------------------------------------------- #
# 2. XABCD candidates — Shark arm and its seeded placebo
# --------------------------------------------------------------------------- #
def _seed_from(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary hashable parts (stable across runs
    and processes — avoids relying on Python's randomized str/bytes hashing)."""
    s = "-".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def find_shark(pivots: pd.DataFrame,
              ab_lo: float = 1.13, ab_hi: float = 1.618,
              bc_lo: float = 1.618, bc_hi: float = 2.24,
              d_lo: float = 0.886, d_hi: float = 1.13,
              placebo: bool = False, seed: int = 0) -> pd.DataFrame:
    """Scan consecutive confirmed pivot quadruples (X, A, B, C) for a Shark.

    Structural bands, identical in both arms (they define a well-formed XABCD
    shape, not the hypothesis under test): the AB leg must EXTEND the XA leg past
    point X by ``[ab_lo, ab_hi]`` = 1.13-1.618x (B overshoots beyond X — a
    retracement-free structure, unlike every other pattern in the zoo), and the BC
    leg must further EXTEND the AB leg past point A by ``[bc_lo, bc_hi]`` =
    1.618-2.24x. Both ratios are magnitudes (``|leg| / |prior leg|``); the zigzag's
    alternating H/L structure automatically alternates their signs.

    Shark arm (``placebo=False``): D is a PRICE ZONE on the *original* XA leg,
    ``[X + d_lo*(A-X), X + d_hi*(A-X)]`` with ``(d_lo, d_hi) = (0.886, 1.13)`` — the
    Shark's defining "5-0" completion band (a zone, not a single point-plus-
    tolerance like the rest of the zoo). ``reversal_dir`` is ``sign(XA)``: believers
    read the B/C overshoot as exhaustion and expect price to resume the *original*
    XA direction once D is touched.

    Placebo arm (``placebo=True``): identical structural pivots (same AB/BC bands,
    same candidate pool), but each candidate's D-zone is re-centered on a
    deterministic, seeded extension ratio drawn clear of [0.886, 1.13] (same
    half-width as the real zone) — "does ANY nearby completion zone work, not
    specifically Carney's 0.886-1.13."
    """
    half_width = (d_hi - d_lo) / 2.0    # 0.122 — same zone width reused by the placebo
    rows = []
    P = pivots.reset_index(drop=True)
    for k in range(len(P) - 3):
        X, A, B, C = P.iloc[k], P.iloc[k + 1], P.iloc[k + 2], P.iloc[k + 3]
        XA = float(A["price"] - X["price"])
        AB = float(B["price"] - A["price"])
        BC = float(C["price"] - B["price"])
        if XA == 0.0 or AB == 0.0:
            continue
        ext_ab = abs(AB / XA)
        ext_bc = abs(BC / AB)

        if not (ab_lo <= ext_ab <= ab_hi):
            continue
        if not (bc_lo <= ext_bc <= bc_hi):
            continue

        if placebo:
            rng = np.random.default_rng(_seed_from(
                seed, int(X["piv_i"]), int(A["piv_i"]), int(B["piv_i"]), int(C["piv_i"])))
            center = float(rng.uniform(0.35, 2.10))
            while abs(center - (d_lo + d_hi) / 2.0) < (half_width + 0.15):
                center = float(rng.uniform(0.35, 2.10))
        else:
            center = (d_lo + d_hi) / 2.0

        zone_lo = float(X["price"] + (center - half_width) * XA)
        zone_hi = float(X["price"] + (center + half_width) * XA)
        reversal_dir = 1.0 if XA > 0 else -1.0   # believers expect a turn back toward the XA trend

        rows.append({
            "X_i": int(X["piv_i"]), "A_i": int(A["piv_i"]), "B_i": int(B["piv_i"]),
            "C_i": int(C["piv_i"]), "conf_i": int(C["conf_i"]),
            "X_p": float(X["price"]), "A_p": float(A["price"]),
            "B_p": float(B["price"]), "C_p": float(C["price"]),
            "ext_ab": ext_ab, "ext_bc": ext_bc, "zone_center": center,
            "zone_lo": min(zone_lo, zone_hi), "zone_hi": max(zone_lo, zone_hi),
            "reversal_dir": reversal_dir,
        })
    cols = ["X_i", "A_i", "B_i", "C_i", "conf_i", "X_p", "A_p", "B_p", "C_p",
            "ext_ab", "ext_bc", "zone_center", "zone_lo", "zone_hi", "reversal_dir"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# 3+4. D-zone scan and the fade — vectorised forward window per candidate
# --------------------------------------------------------------------------- #
def scan_touches(bars: pd.DataFrame, candidates: pd.DataFrame, max_fwd_days: int = 120,
                 horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0) -> pd.DataFrame:
    """First touch of each candidate's projected D-ZONE, and the fade's forward return.

    Scans strictly *after* C's confirmation bar (real-time knowledge only) for the
    first bar whose high-low range overlaps ``[zone_lo, zone_hi]``. Enters the fade
    at that bar's close, in ``reversal_dir``; exits at the close ``h`` sessions later
    for each ``h`` in ``horizons``. One round trip = 2 x one-way ``cost_bps`` x NAV
    (house convention: costs one-way per leg, both legs charged).
    """
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(bars)
    h_max = max(horizons)

    rows = []
    for _, r in candidates.iterrows():
        t0 = int(r["conf_i"])
        if t0 + 1 >= n:
            continue
        win_end = min(t0 + max_fwd_days, n - h_max - 1)
        if win_end <= t0:
            continue

        w_hi = high[t0 + 1: win_end + 1]
        w_lo = low[t0 + 1: win_end + 1]
        zlo, zhi = float(r["zone_lo"]), float(r["zone_hi"])
        touched = (w_lo <= zhi) & (w_hi >= zlo)          # bar range overlaps the D zone
        first = int(np.argmax(touched)) if touched.any() else -1
        if first < 0:
            continue

        bar_i = t0 + 1 + first
        if bar_i + h_max >= n:
            continue
        entry_px = float(close[bar_i])
        d = float(r["reversal_dir"])
        rec = {
            "touch_date": bars.index[bar_i], "X_i": r["X_i"], "A_i": r["A_i"],
            "B_i": r["B_i"], "C_i": r["C_i"],
            "ext_ab": r["ext_ab"], "ext_bc": r["ext_bc"], "zone_center": r["zone_center"],
            "reversal_dir": d, "entry": entry_px, "entry_i": bar_i,
        }
        for hh in horizons:
            exit_px = float(close[bar_i + hh])
            g = d * (exit_px - entry_px) / entry_px
            rec[f"ret_gross_{hh}"] = float(g)
            rec[f"ret_net_{hh}"] = float(g - 2.0 * cost_bps / 1e4)
        rows.append(rec)

    cols = (["touch_date", "X_i", "A_i", "B_i", "C_i", "ext_ab", "ext_bc", "zone_center",
             "reversal_dir", "entry", "entry_i"]
            + [f"ret_gross_{hh}" for hh in horizons] + [f"ret_net_{hh}" for hh in horizons])
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def detect_and_scan(bars: pd.DataFrame, pct: float = 0.03, placebo: bool = False, seed: int = 0,
                    ab_lo: float = 1.13, ab_hi: float = 1.618,
                    bc_lo: float = 1.618, bc_hi: float = 2.24,
                    d_lo: float = 0.886, d_hi: float = 1.13,
                    max_fwd_days: int = 120, horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The full pipeline on one tape: pivots -> XABCD candidates -> D-zone ledger."""
    piv = zigzag_pivots(bars["close"].to_numpy(), pct)
    cand = find_shark(piv, ab_lo, ab_hi, bc_lo, bc_hi, d_lo, d_hi, placebo=placebo, seed=seed)
    ledger = scan_touches(bars, cand, max_fwd_days, horizons, cost_bps)
    return piv, cand, ledger


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_mean_t(x: np.ndarray) -> float:
    """Newey-West (Bartlett kernel) t of a sample mean against 0 — the
    autocorrelation-robust cross-check for a ledger whose events can cluster in
    time (overlapping Shark legs on the same tape)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def summarize(ledger: pd.DataFrame, col: str) -> dict:
    """Headline per-touch statistics for one ledger: n, hit rate (Wilson), mean
    return (bps) and its HAC t against 0."""
    if ledger.empty or col not in ledger.columns:
        return {"n": 0, "hit_rate": float("nan"), "hit_lo": float("nan"), "hit_hi": float("nan"),
                "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    k = int((r > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {
        "n": int(n), "hit_rate": float(k / n) if n else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "t": hac_mean_t(r),
    }


# --------------------------------------------------------------------------- #
# The base-rate control (Signal axis) — random-day entries, matched direction mix
# --------------------------------------------------------------------------- #
def base_rate_ledger(bars: pd.DataFrame, n_events: int, dir_mix: float,
                     horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0,
                     seed: int = 0, h_max_guard: int | None = None) -> pd.DataFrame:
    """``n_events`` random entry days on ``bars``, with the SAME bullish/bearish mix
    (``dir_mix`` = share of ``reversal_dir == +1``) as the real Shark touches — the
    equity-drift confound a naive backtest would otherwise credit to the pattern.
    One draw per (ticker, seed); pool many seeds for the null distribution.
    """
    n = len(bars)
    h_max = h_max_guard if h_max_guard is not None else max(horizons)
    if n_events <= 0 or n <= h_max + 1:
        cols = ["entry_i", "reversal_dir"] + [f"ret_gross_{hh}" for hh in horizons] \
            + [f"ret_net_{hh}" for hh in horizons]
        return pd.DataFrame(columns=cols)
    rng = np.random.default_rng(seed)
    close = bars["close"].to_numpy()
    hi = n - h_max - 1
    idx = rng.integers(low=0, high=max(hi, 1), size=n_events)
    dirs = np.where(rng.random(n_events) < dir_mix, 1.0, -1.0)

    rows = []
    for i, d in zip(idx, dirs):
        entry_px = float(close[i])
        rec = {"entry_i": int(i), "reversal_dir": float(d)}
        for hh in horizons:
            exit_px = float(close[i + hh])
            g = d * (exit_px - entry_px) / entry_px
            rec[f"ret_gross_{hh}"] = float(g)
            rec[f"ret_net_{hh}"] = float(g - 2.0 * cost_bps / 1e4)
        rows.append(rec)
    cols = ["entry_i", "reversal_dir"] + [f"ret_gross_{hh}" for hh in horizons] \
        + [f"ret_net_{hh}" for hh in horizons]
    return pd.DataFrame(rows, columns=cols)


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    """Two-sided normal-approximation critical |t| for a Bonferroni-corrected
    family-wise alpha across ``n_tests`` comparisons (large-sample ledgers here,
    so the normal approximation to the t-distribution is adequate)."""
    from scipy import stats
    a = alpha / (2.0 * max(n_tests, 1))
    return float(stats.norm.ppf(1.0 - a))
