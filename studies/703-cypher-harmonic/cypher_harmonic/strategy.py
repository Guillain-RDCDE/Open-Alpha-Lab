"""Strategy + inference for Study 703 — Cypher-Harmonic.

The claim: **the Cypher harmonic predicts the reversal at point D — and does so
with a uniquely "tight" 0.786 retracement of the XC leg**, the only pattern in the
Carney/Oglesbee harmonic-pattern zoo whose D ratio is measured off **XC** rather
than off XA (Crab, Butterfly) or a retracement of AB (Gartley, Bat). Chartists
draw four alternating swing pivots X, A, B, C, then project a fifth point D such
that:

1. **AB retraces XA by 0.382-0.618** — the same shallow structural band shared by
   the Crab and (loosely) the Bat; a well-formed B leg every XABCD candidate needs,
   not itself the hypothesis under test.
2. **C overshoots the original A swing** — unlike Gartley/Bat/Crab (where C sits
   *between* B and A, retracing AB) or the Butterfly (where D itself, not C, is the
   overshoot), the Cypher's C point extends **1.13x-1.414x of the XA leg PAST
   point A**, in the same direction as the original XA move. This is a structural
   band (it defines a well-formed "C beyond A" shape); it is not itself the
   hypothesis under test.
3. **D retraces the XC leg — not XA, not AB — by exactly 78.6%.** This is the
   Cypher's defining signature and the specific ratio the brief asks us to test:
   Darren Oglesbee's construction (popularized by Scott Carney's later Harmonic
   Trading volumes) measures the reversal zone off the *freshly extended* XC leg,
   landing D back inside the X-A range, close to X — a "retest of origin" read,
   distinct from every sibling pattern's D formula. Believers read the 0.786 XC
   retracement as an unusually precise support/resistance confluence and treat D
   as a reversal point ("potential reversal zone", PRZ).

Pipeline, all real-time (no look-ahead):

1. **Pivot detection.** A percentage-threshold zigzag on daily closes. A pivot is
   only *known* at its **confirmation bar** — the bar where price reverses far
   enough to lock it in — never at the (earlier) extreme bar itself.
2. **XABC candidates.** Every consecutive confirmed quadruple (X, A, B, C) whose
   AB/XA retracement and XC/XA extension sit inside the structural bands above,
   projecting **D = C - 0.786 x (C - X)** — a price target fully known the moment
   C confirms.
3. **D-touch scan.** From C's confirmation bar forward (a real trader's earliest
   possible knowledge), scan up to ``max_fwd_days`` sessions for the first bar
   whose high-low range brackets D (or closes within tolerance of it).
4. **The fade.** Enter at the touch bar's close, in the *predicted reversal
   direction* (the direction back toward A — that is what "the pattern completes
   and price turns" means), hold for a fixed horizon, exit at that close.

**Two independent honesty checks**, matching the brief:

* **Forward returns at D vs a random-day BASE RATE, Bonferroni-corrected.** The
  base rate redraws, per ticker and many seeds, the *same number* of random
  entry days with the *same* empirical mix of bullish/bearish reversal
  directions as the real Cypher touches — i.e. "what a random entry with the same
  directional tilt would have earned", the exact equity-drift confound a naive
  backtest would otherwise credit to the pattern. We test the pooled 5-day
  headline *and* all six per-ticker splits (7 tests total) and apply a
  Bonferroni correction across them — the honest price of looking at the basket
  six different ways.
* **A placebo arm (the third-axis myth-check).** Identical pivots, but each
  candidate's D-retracement *target* is replaced by a deterministic, seeded draw
  kept clear of the Cypher's own 0.786 band — "would ANY XC-retracement zone have
  worked", not specifically the precise 0.786 ratio the pattern is built on.
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
# 2. XABC candidates — Cypher arm and its seeded placebo
# --------------------------------------------------------------------------- #
def _seed_from(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary hashable parts (stable across runs
    and processes — avoids relying on Python's randomized str/bytes hashing)."""
    s = "-".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def find_cypher(pivots: pd.DataFrame,
                ab_lo: float = 0.382, ab_hi: float = 0.618,
                xc_lo: float = 1.13, xc_hi: float = 1.414,
                target_retrace: float = 0.786, retrace_tol: float = 0.10,
                placebo: bool = False, seed: int = 0) -> pd.DataFrame:
    """Scan consecutive confirmed pivot quadruples (X, A, B, C) for a Cypher.

    Structural bands, identical in both arms (they define a well-formed XABC
    shape, not the hypothesis under test): the AB retracement of XA must sit in
    ``[ab_lo, ab_hi]`` (0.382-0.618, the shallow band shared with the Crab) and C
    must **overshoot A**, in the *same direction* as the original XA leg, by
    ``XC/XA`` in ``[xc_lo, xc_hi]`` (1.13-1.414x) — unlike every sibling pattern,
    where C retraces *between* B and A.

    Cypher arm (``placebo=False``): D retraces the freshly-extended **XC leg** (not
    XA, not AB) back toward X by ``target_retrace`` (0.786) +/- ``retrace_tol`` —
    the pattern's defining, uniquely-XC-referenced signature.
    ``D = C - target_retrace x (C - X)``. ``reversal_dir`` is +1 (expect a bounce
    UP off D, back in the direction of the original move) when XA rose (bullish
    Cypher) and -1 otherwise.

    Placebo arm (``placebo=True``): identical structural pivots (same AB/XC
    bands, same candidate pool), but each candidate's D-retracement *target* is
    replaced by a deterministic, seeded draw kept clear of the Cypher's own 0.786
    band — "would ANY XC-retracement zone have worked", not specifically the
    precise 0.786 ratio the pattern is built on.
    """
    rows = []
    P = pivots.reset_index(drop=True)
    for k in range(len(P) - 3):
        X, A, B, C = P.iloc[k], P.iloc[k + 1], P.iloc[k + 2], P.iloc[k + 3]
        XA = float(A["price"] - X["price"])
        AB = float(B["price"] - A["price"])
        XC = float(C["price"] - X["price"])
        if XA == 0.0 or AB == 0.0:
            continue
        retrace_ab = abs(AB / XA)
        ext_xc = XC / XA  # signed: must be positive AND > 1 -> a genuine overshoot past A

        if not (ab_lo <= retrace_ab <= ab_hi):
            continue
        if not (xc_lo <= ext_xc <= xc_hi):
            continue

        if placebo:
            rng = np.random.default_rng(_seed_from(
                seed, int(X["piv_i"]), int(A["piv_i"]), int(B["piv_i"]), int(C["piv_i"])))
            t_ret = float(rng.uniform(0.30, 0.98))
            while abs(t_ret - target_retrace) < retrace_tol:
                t_ret = float(rng.uniform(0.30, 0.98))
        else:
            t_ret = target_retrace

        d_proj = float(C["price"] - t_ret * XC)          # D retraces the XC leg back toward X
        reversal_dir = 1.0 if XA > 0 else -1.0             # believers expect a turn AT D

        rows.append({
            "X_i": int(X["piv_i"]), "A_i": int(A["piv_i"]), "B_i": int(B["piv_i"]),
            "C_i": int(C["piv_i"]), "conf_i": int(C["conf_i"]),
            "X_p": float(X["price"]), "A_p": float(A["price"]),
            "B_p": float(B["price"]), "C_p": float(C["price"]),
            "retrace_ab": retrace_ab, "ext_xc": ext_xc,
            "d_retrace": t_ret, "D_proj": d_proj, "reversal_dir": reversal_dir,
        })
    cols = ["X_i", "A_i", "B_i", "C_i", "conf_i", "X_p", "A_p", "B_p", "C_p",
            "retrace_ab", "ext_xc", "d_retrace", "D_proj", "reversal_dir"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# 3+4. D-touch scan and the fade — vectorised forward window per candidate
# --------------------------------------------------------------------------- #
def scan_touches(bars: pd.DataFrame, candidates: pd.DataFrame, max_fwd_days: int = 120,
                 tolerance_pct: float = 0.0075, horizons=DEFAULT_HORIZONS,
                 cost_bps: float = 5.0) -> pd.DataFrame:
    """First touch of each candidate's projected D, and the fade's forward return.

    Scans strictly *after* C's confirmation bar (real-time knowledge only) for the
    first bar whose high-low range brackets ``D_proj`` (or whose close is within
    ``tolerance_pct``). Enters the fade at that bar's close, in ``reversal_dir``;
    exits at the close ``h`` sessions later for each ``h`` in ``horizons``. One
    round trip = 2 x one-way ``cost_bps`` x NAV (house convention: costs one-way
    per leg, both legs charged).
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
        w_cl = close[t0 + 1: win_end + 1]
        lv = float(r["D_proj"])
        tol = tolerance_pct * abs(lv)
        touched = ((w_lo <= lv) & (lv <= w_hi)) | (np.abs(w_cl - lv) <= tol)
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
            "retrace_ab": r["retrace_ab"], "ext_xc": r["ext_xc"],
            "d_retrace": r["d_retrace"], "reversal_dir": d, "entry": entry_px,
            "entry_i": bar_i,
        }
        for hh in horizons:
            exit_px = float(close[bar_i + hh])
            g = d * (exit_px - entry_px) / entry_px
            rec[f"ret_gross_{hh}"] = float(g)
            rec[f"ret_net_{hh}"] = float(g - 2.0 * cost_bps / 1e4)
        rows.append(rec)

    cols = (["touch_date", "X_i", "A_i", "B_i", "C_i", "retrace_ab", "ext_xc",
             "d_retrace", "reversal_dir", "entry", "entry_i"]
            + [f"ret_gross_{hh}" for hh in horizons] + [f"ret_net_{hh}" for hh in horizons])
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def detect_and_scan(bars: pd.DataFrame, pct: float = 0.03, placebo: bool = False, seed: int = 0,
                    ab_lo: float = 0.382, ab_hi: float = 0.618,
                    xc_lo: float = 1.13, xc_hi: float = 1.414,
                    target_retrace: float = 0.786, retrace_tol: float = 0.10,
                    max_fwd_days: int = 120, tolerance_pct: float = 0.0075,
                    horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The full pipeline on one tape: pivots -> XABC candidates -> D-touch ledger."""
    piv = zigzag_pivots(bars["close"].to_numpy(), pct)
    cand = find_cypher(piv, ab_lo, ab_hi, xc_lo, xc_hi, target_retrace, retrace_tol,
                       placebo=placebo, seed=seed)
    ledger = scan_touches(bars, cand, max_fwd_days, tolerance_pct, horizons, cost_bps)
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
    time (overlapping Cypher legs on the same tape)."""
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
    (``dir_mix`` = share of ``reversal_dir == +1``) as the real Cypher touches — the
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
