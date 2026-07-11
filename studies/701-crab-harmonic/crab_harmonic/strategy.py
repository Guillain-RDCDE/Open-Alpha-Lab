"""Strategy + inference for Study 701 — Crab-Harmonic.

The claim: **the Crab harmonic predicts the reversal at point D — and, per Scott
Carney, does so with the SHARPEST precision in the harmonic-pattern family.**
Chartists draw four alternating swing pivots X, A, B, C, then project a fifth
point D such that:

1. **AB retraces XA by 0.382-0.618** — a *shallower* band than the Butterfly's
   fixed 0.786, close to (but not identical to) the Bat's 0.382-0.50. This is a
   structural band (every XABCD pattern needs a well-formed B leg); it is not
   itself the hypothesis under test.
2. **BC retraces AB somewhere in 0.382-0.886** (the same loose structural band
   shared by every XABCD pattern in the Carney taxonomy — again not under test,
   just the geometry that keeps a candidate "well-formed").
3. **D extends the ORIGINAL XA leg by exactly 1.618x, PAST point X** — this is
   the Crab's defining signature and the specific number the brief asks us to
   test. Carney calls the Crab the pattern with the "most exact" and "sharpest"
   reversal zone in the entire family precisely because 1.618 — the golden
   ratio's own reciprocal-complement — is a single, tight target rather than a
   loose 1.27-1.618 range (contrast the Butterfly, whose D wanders anywhere in
   that band). An even deeper "Deep Crab" variant (AB retrace 0.886, D still
   1.618) exists in Carney's later books; this study tests the original, most
   commonly cited Crab grid. Believers read the 1.618 extension as an unusually
   precise support/resistance confluence and treat D as a reversal point
   ("potential reversal zone", PRZ).

Pipeline, all real-time (no look-ahead):

1. **Pivot detection.** A percentage-threshold zigzag on daily closes. A pivot is
   only *known* at its **confirmation bar** — the bar where price reverses far
   enough to lock it in — never at the (earlier) extreme bar itself.
2. **XABCD candidates.** Every consecutive confirmed quadruple (X, A, B, C) whose
   AB/XA and BC/AB retracements sit inside the structural bands above, projecting
   **D = X - 1.618 x (A - X)** — a price target fully known the moment C confirms.
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
  directions as the real Crab touches — i.e. "what a random entry with the same
  directional tilt would have earned", the exact equity-drift confound a naive
  backtest would otherwise credit to the pattern. We test the pooled 5-day
  headline *and* all six per-ticker splits (7 tests total) and apply a
  Bonferroni correction across them — the honest price of looking at the basket
  six different ways.
* **A placebo arm (the third-axis myth-check).** Identical pivots, but each
  candidate's AB-retrace and D-extension *targets* are replaced by a
  deterministic, seeded draw kept clear of the Crab's own bands — "would ANY
  extension-and-reversal projection near X have worked", not specifically the
  precise 1.618 ratio Carney calls "sharpest."
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
# 2. XABCD candidates — Crab arm and its seeded placebo
# --------------------------------------------------------------------------- #
def _seed_from(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary hashable parts (stable across runs
    and processes — avoids relying on Python's randomized str/bytes hashing)."""
    s = "-".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def find_crab(pivots: pd.DataFrame,
             ab_lo: float = 0.382, ab_hi: float = 0.618,
             bc_lo: float = 0.382, bc_hi: float = 0.886,
             target_ext: float = 1.618, ext_tol: float = 0.15,
             placebo: bool = False, seed: int = 0) -> pd.DataFrame:
    """Scan consecutive confirmed pivot quadruples (X, A, B, C) for a Crab.

    Structural bands, identical in both arms (they define a well-formed XABCD
    shape, not the hypothesis under test): the AB retracement of XA must sit in
    ``[ab_lo, ab_hi]`` (0.382-0.618 — shallower than the Butterfly's fixed 0.786)
    and the BC retracement of AB must sit in ``[bc_lo, bc_hi]`` (0.382-0.886, the
    shared XABCD band).

    Crab arm (``placebo=False``): D extends the *original* XA leg PAST point X,
    ``D = X - ext x (A - X)`` with ``ext`` targeted at ``target_ext`` (1.618) +/-
    ``ext_tol`` — the Crab's defining, most extreme extension in the zoo (deeper
    than the Butterfly's 1.27-1.618 *range*: Carney calls the Crab's 1.618 a
    single, "sharpest" target). ``reversal_dir`` is +1 (expect a bounce UP off D)
    when XA rose (bullish Crab, D projects below X) and -1 otherwise.

    Placebo arm (``placebo=True``): identical structural pivots (same AB/BC
    bands, same candidate pool), but each candidate's D-extension *target* is
    replaced by a deterministic, seeded draw kept clear of the Crab's own 1.618
    band — "would ANY extension-and-reversal projection near X have worked", not
    specifically the precise 1.618 ratio Carney calls "sharpest."
    """
    rows = []
    P = pivots.reset_index(drop=True)
    for k in range(len(P) - 3):
        X, A, B, C = P.iloc[k], P.iloc[k + 1], P.iloc[k + 2], P.iloc[k + 3]
        XA = float(A["price"] - X["price"])
        AB = float(B["price"] - A["price"])
        BC = float(C["price"] - B["price"])
        if XA == 0.0 or AB == 0.0:
            continue
        retrace_ab = abs(AB / XA)
        retrace_bc = abs(BC / AB)

        if not (ab_lo <= retrace_ab <= ab_hi):
            continue
        if not (bc_lo <= retrace_bc <= bc_hi):
            continue

        if placebo:
            rng = np.random.default_rng(_seed_from(
                seed, int(X["piv_i"]), int(A["piv_i"]), int(B["piv_i"]), int(C["piv_i"])))
            t_ext = float(rng.uniform(0.85, 2.20))
            while abs(t_ext - target_ext) < ext_tol:
                t_ext = float(rng.uniform(0.85, 2.20))
        else:
            t_ext = target_ext

        d_proj = float(X["price"] - t_ext * XA)          # D extends the ORIGINAL XA leg PAST X
        reversal_dir = 1.0 if XA > 0 else -1.0             # believers expect a turn AT D

        rows.append({
            "X_i": int(X["piv_i"]), "A_i": int(A["piv_i"]), "B_i": int(B["piv_i"]),
            "C_i": int(C["piv_i"]), "conf_i": int(C["conf_i"]),
            "X_p": float(X["price"]), "A_p": float(A["price"]),
            "B_p": float(B["price"]), "C_p": float(C["price"]),
            "retrace_ab": retrace_ab, "retrace_bc": retrace_bc,
            "ext_target": t_ext, "D_proj": d_proj, "reversal_dir": reversal_dir,
        })
    cols = ["X_i", "A_i", "B_i", "C_i", "conf_i", "X_p", "A_p", "B_p", "C_p",
            "retrace_ab", "retrace_bc", "ext_target", "D_proj", "reversal_dir"]
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
            "retrace_ab": r["retrace_ab"], "retrace_bc": r["retrace_bc"],
            "ext_target": r["ext_target"], "reversal_dir": d, "entry": entry_px,
            "entry_i": bar_i,
        }
        for hh in horizons:
            exit_px = float(close[bar_i + hh])
            g = d * (exit_px - entry_px) / entry_px
            rec[f"ret_gross_{hh}"] = float(g)
            rec[f"ret_net_{hh}"] = float(g - 2.0 * cost_bps / 1e4)
        rows.append(rec)

    cols = (["touch_date", "X_i", "A_i", "B_i", "C_i", "retrace_ab", "retrace_bc",
             "ext_target", "reversal_dir", "entry", "entry_i"]
            + [f"ret_gross_{hh}" for hh in horizons] + [f"ret_net_{hh}" for hh in horizons])
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def detect_and_scan(bars: pd.DataFrame, pct: float = 0.03, placebo: bool = False, seed: int = 0,
                    ab_lo: float = 0.382, ab_hi: float = 0.618,
                    bc_lo: float = 0.382, bc_hi: float = 0.886,
                    target_ext: float = 1.618, ext_tol: float = 0.15,
                    max_fwd_days: int = 120, tolerance_pct: float = 0.0075,
                    horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The full pipeline on one tape: pivots -> XABCD candidates -> D-touch ledger."""
    piv = zigzag_pivots(bars["close"].to_numpy(), pct)
    cand = find_crab(piv, ab_lo, ab_hi, bc_lo, bc_hi, target_ext, ext_tol,
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
    time (overlapping Crab legs on the same tape)."""
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
    (``dir_mix`` = share of ``reversal_dir == +1``) as the real Crab touches — the
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
