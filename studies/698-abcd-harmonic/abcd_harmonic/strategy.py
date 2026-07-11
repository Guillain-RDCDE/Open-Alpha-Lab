"""Strategy + inference for Study 698 — ABCD-Harmonic.

The claim: **the AB=CD harmonic predicts the reversal at point D.** Chartists draw
three alternating swing pivots A, B, C on a daily chart, then project a fourth point
D such that (a) the retracement leg BC pulls back **61.8%** of the impulse leg AB,
and (b) the projection leg CD is **equal in length to AB** ("AB=CD") — and expect
price to *reverse* the moment it reaches D, because the pattern is "complete".

This is the simplest member of the harmonic-pattern family — no X point, no
multi-leg Fibonacci confluence zone (that's Gartley/Butterfly/Bat territory, see
``docs/references.md``). Pipeline, all real-time (no look-ahead):

1. **Pivot detection.** A percentage-threshold zigzag on daily closes. A pivot is
   only *known* at its **confirmation bar** — the bar where price reverses far
   enough to lock it in — never at the (earlier) extreme bar itself.
2. **ABCD candidates.** Every consecutive confirmed triple (A, B, C) where the BC
   retracement of AB sits within tolerance of the target ratio (0.618 for the
   Fibonacci arm), projecting **D = C + ext_target × AB** — a price target that is
   fully known the moment C confirms.
3. **D-touch scan.** From C's confirmation bar forward (a real trader's earliest
   possible knowledge), scan up to ``max_fwd_days`` sessions for the first bar
   whose high-low range brackets D (or closes within tolerance of it).
4. **The fade.** Enter at the touch bar's close, in the *predicted reversal
   direction* (opposite the CD leg — that is what "the pattern completes and price
   turns" means), hold for a fixed horizon, exit at that close. One round trip =
   2 × one-way cost × NAV.

The **placebo arm** reruns the identical pipeline on the identical pivots, but
with the retracement/extension *targets* replaced by a deterministic, seeded,
per-candidate ratio drawn well away from the Fibonacci band — i.e. "any
equal-legged reversal projection", not specifically the 0.618/1.0 harmonic pair.
Only a Fibonacci arm that **beats** this placebo constitutes evidence that *these
particular ratios* matter, rather than "swings tend to retrace and reverse" in
general (the mean-reversion base rate any level-based rule would inherit).
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

    A pivot is appended to the result only once price has reversed by ``pct``
    from the running extreme — i.e. at the **confirmation bar** ``conf_i``, which
    is always strictly after the extreme bar ``piv_i``. The final, still-forming
    leg (not yet confirmed) is dropped: nothing here uses information from the
    future relative to ``conf_i``.
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
# 2. ABCD candidates — Fibonacci arm and its seeded placebo
# --------------------------------------------------------------------------- #
def _seed_from(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary hashable parts (stable across runs
    and processes — avoids relying on Python's randomized str/bytes hashing)."""
    s = "-".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def find_abcd(pivots: pd.DataFrame, target_retrace: float = 0.618, retrace_tol: float = 0.10,
              target_ext: float = 1.0, ext_tol: float = 0.15,
              placebo: bool = False, seed: int = 0) -> pd.DataFrame:
    """Scan consecutive confirmed pivot triples (A, B, C) for a completed AB=CD leg.

    Fibonacci arm (``placebo=False``): a triple qualifies when the BC retracement
    of AB is within ``retrace_tol`` of ``target_retrace`` (0.618). The projection
    target is ``D = C + target_ext × AB`` (``target_ext=1.0`` -> "AB=CD").

    Placebo arm (``placebo=True``): identical pivots, but each candidate draws its
    *own* deterministic retracement/extension targets, uniformly at random and
    kept away from the Fibonacci band under test — "would ANY equal-legged
    reversal projection have worked", not specifically 0.618/1.0.
    """
    rows = []
    P = pivots.reset_index(drop=True)
    for k in range(len(P) - 2):
        A, B, C = P.iloc[k], P.iloc[k + 1], P.iloc[k + 2]
        AB = float(B["price"] - A["price"])
        BC = float(C["price"] - B["price"])
        if AB == 0.0:
            continue
        retrace = abs(BC / AB)

        if placebo:
            rng = np.random.default_rng(_seed_from(seed, int(A["piv_i"]), int(B["piv_i"]), int(C["piv_i"])))
            t_retrace = float(rng.uniform(0.25, 0.90))
            while abs(t_retrace - target_retrace) < retrace_tol:
                t_retrace = float(rng.uniform(0.25, 0.90))
            t_ext = float(rng.uniform(0.5, 2.0))
            while abs(t_ext - target_ext) < ext_tol:
                t_ext = float(rng.uniform(0.5, 2.0))
        else:
            t_retrace, t_ext = target_retrace, target_ext

        if abs(retrace - t_retrace) > retrace_tol:
            continue

        d_proj = float(C["price"] + t_ext * AB)          # CD continues the AB direction
        reversal_dir = -1.0 if AB > 0 else 1.0            # believers expect a turn AT D

        rows.append({
            "A_i": int(A["piv_i"]), "B_i": int(B["piv_i"]), "C_i": int(C["piv_i"]),
            "conf_i": int(C["conf_i"]),
            "A_p": float(A["price"]), "B_p": float(B["price"]), "C_p": float(C["price"]),
            "retrace": retrace, "ext_target": t_ext, "retrace_target": t_retrace,
            "D_proj": d_proj, "reversal_dir": reversal_dir,
        })
    cols = ["A_i", "B_i", "C_i", "conf_i", "A_p", "B_p", "C_p",
            "retrace", "ext_target", "retrace_target", "D_proj", "reversal_dir"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# 3+4. D-touch scan and the fade — vectorised forward window per candidate
# --------------------------------------------------------------------------- #
def scan_touches(bars: pd.DataFrame, candidates: pd.DataFrame, max_fwd_days: int = 40,
                  tolerance_pct: float = 0.0075, horizons=DEFAULT_HORIZONS,
                  cost_bps: float = 5.0) -> pd.DataFrame:
    """First touch of each candidate's projected D, and the fade's forward return.

    Scans strictly *after* C's confirmation bar (real-time knowledge only) for the
    first bar whose high-low range brackets ``D_proj`` (or whose close is within
    ``tolerance_pct``). Enters the fade at that bar's close, in ``reversal_dir``;
    exits at the close ``h`` sessions later for each ``h`` in ``horizons``. One
    round trip = 2 × one-way ``cost_bps`` × NAV (house convention: costs one-way
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
            "touch_date": bars.index[bar_i], "A_i": r["A_i"], "B_i": r["B_i"], "C_i": r["C_i"],
            "retrace": r["retrace"], "ext_target": r["ext_target"], "retrace_target": r["retrace_target"],
            "reversal_dir": d, "entry": entry_px,
        }
        for hh in horizons:
            exit_px = float(close[bar_i + hh])
            g = d * (exit_px - entry_px) / entry_px
            rec[f"ret_gross_{hh}"] = float(g)
            rec[f"ret_net_{hh}"] = float(g - 2.0 * cost_bps / 1e4)
        rows.append(rec)

    cols = (["touch_date", "A_i", "B_i", "C_i", "retrace", "ext_target", "retrace_target",
             "reversal_dir", "entry"]
            + [f"ret_gross_{hh}" for hh in horizons] + [f"ret_net_{hh}" for hh in horizons])
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def detect_and_scan(bars: pd.DataFrame, pct: float = 0.03, placebo: bool = False, seed: int = 0,
                    target_retrace: float = 0.618, retrace_tol: float = 0.10,
                    target_ext: float = 1.0, ext_tol: float = 0.15,
                    max_fwd_days: int = 40, tolerance_pct: float = 0.0075,
                    horizons=DEFAULT_HORIZONS, cost_bps: float = 5.0
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The full pipeline on one tape: pivots -> ABCD candidates -> D-touch ledger."""
    piv = zigzag_pivots(bars["close"].to_numpy(), pct)
    cand = find_abcd(piv, target_retrace, retrace_tol, target_ext, ext_tol, placebo=placebo, seed=seed)
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
    autocorrelation-robust cross-check for a ledger whose events can cluster
    in time (overlapping ABCD legs on the same tape)."""
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
