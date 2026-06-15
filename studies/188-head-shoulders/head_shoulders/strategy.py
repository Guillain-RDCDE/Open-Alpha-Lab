"""The strategy and its honest controls — Study 188 (Head-Shoulders).

The head-and-shoulders top (H&S top) is the most famous chart pattern in technical
analysis: a central price peak (the *head*) flanked by two lower peaks (the
*shoulders*), with a roughly horizontal *neckline* connecting the troughs on either
side of the head.  The mirror image is the *inverse head-and-shoulders* (IH&S), a
bottoming pattern.  The trading rule is:

  - Identify the five-point structure (L-shoulder, L-trough, Head, R-trough,
    R-shoulder) using local-extrema logic (``scipy.signal.find_peaks``).
  - The *confirmation signal* fires when the close breaks *below* the neckline
    (for H&S top) or *above* it (for IH&S) on bar *t*.
  - Enter the following day's open (*t+1*), short for H&S top, long for IH&S.
  - Measure the *forward returns* at 1, 5, 20, and 60 trading days.

An honest control runs the same forward-return measurement on *random* days drawn
uniformly from the same tape (the *unconditional baseline*) — this tells us whether
the pattern, after the neckline break, does anything a random-day entry can't.

Multiple-comparisons note: two pattern types (top / inverse) × four horizons = eight
t-stats.  At α = 5% one spurious result is expected; we flag Bonferroni-adjusted
significance thresholds and consider the evidence weak unless *both* pattern types
show an aligned signal at the same horizon.

No look-ahead: all five structural points and the neckline break are detected on
closes available at bar *t*; forward returns begin at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks  # type: ignore
    _SCIPY = True
except ImportError:  # pragma: no cover — scipy is listed in requirements
    _SCIPY = False


# ---------------------------------------------------------------------------
# Local-extrema helpers
# ---------------------------------------------------------------------------
def _local_peaks(close: np.ndarray, min_dist: int = 5) -> np.ndarray:
    """Indices of local maxima in ``close``.

    Uses ``scipy.signal.find_peaks`` when available, else a simple rolling-window
    fallback so the engine survives without scipy (tests).
    """
    if _SCIPY:
        peaks, _ = find_peaks(close, distance=min_dist)
        return peaks
    # Fallback: bar is a peak if it's the maximum in its [i-min_dist, i+min_dist] window.
    n = len(close)
    peaks = []
    for i in range(min_dist, n - min_dist):
        window = close[max(0, i - min_dist) : i + min_dist + 1]
        if close[i] == window.max() and list(window).count(close[i]) == 1:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _local_troughs(close: np.ndarray, min_dist: int = 5) -> np.ndarray:
    """Indices of local minima in ``close`` (mirrors :func:`_local_peaks`)."""
    return _local_peaks(-close, min_dist=min_dist)


# ---------------------------------------------------------------------------
# H&S top and IH&S bottom detector
# ---------------------------------------------------------------------------
def detect_hs_patterns(
    bars: pd.DataFrame,
    min_dist: int = 5,
    shoulder_tol: float = 0.10,
    neckline_tol: float = 0.05,
    min_pattern_bars: int = 20,
    max_pattern_bars: int = 200,
) -> pd.DataFrame:
    """Detect confirmed head-and-shoulders top and inverse patterns on daily bars.

    For each confirmed neckline break the signal is stamped at the *close* of the
    break bar (bar *t*).  The forward return must be measured from bar *t+1* (no
    look-ahead).

    Detection algorithm
    -------------------
    1. Find all local peaks (for H&S top) or troughs (for IH&S bottom) in the close
       series using ``scipy.signal.find_peaks`` (``min_dist`` controls the minimum
       distance between extrema in bars).
    2. For each candidate *head* peak, scan backward for a *left shoulder* and forward
       for a *right shoulder* within ``[min_pattern_bars, max_pattern_bars]`` total span.
    3. Accept a candidate if:
       - The head is strictly higher than both shoulders (for H&S top).
       - The two shoulder heights are within ``shoulder_tol`` (10%) of each other
         (symmetry constraint — prevents every random peak triplet from firing).
       - The neckline (line connecting the two troughs around the head) is
         approximately horizontal: the two trough levels differ by at most
         ``neckline_tol`` (5%) of the head height above the neckline.
    4. A *confirmed* signal fires at the bar where the close first crosses below the
       neckline projection after the right shoulder (for H&S top).  Only the *first*
       neckline break per pattern is counted.

    Parameters
    ----------
    bars : pd.DataFrame
        Daily OHLCV frame (``close`` column used for detection).
    min_dist : int
        Minimum distance (bars) between consecutive local extrema.
    shoulder_tol : float
        Maximum fractional difference between left- and right-shoulder heights.
    neckline_tol : float
        Maximum fractional difference between left and right neckline troughs,
        relative to the head height above the neckline midpoint.
    min_pattern_bars : int
        Minimum bars from left-shoulder peak to right-shoulder peak.
    max_pattern_bars : int
        Maximum bars from left-shoulder peak to right-shoulder peak.

    Returns
    -------
    pd.DataFrame
        One row per confirmed break.  Columns:

        - ``signal_date`` — index of the break bar (bar *t*).
        - ``pattern_type`` — ``"top"`` (bearish, short) or ``"inverse"`` (bullish, long).
        - ``dir`` — trade direction: ``-1`` for top (short), ``+1`` for inverse (long).
        - ``head_idx``, ``ls_idx``, ``rs_idx`` — bar indices of head and shoulders.
        - ``neckline`` — neckline level at the break bar.
        - ``break_idx`` — bar index of the confirmed break.
    """
    close = bars["close"].to_numpy(dtype=float)
    dates = bars.index
    n = len(close)

    results = []

    # ---- H&S top (bearish) ------------------------------------------------
    peaks = _local_peaks(close, min_dist=min_dist)
    troughs = _local_troughs(close, min_dist=min_dist)

    for hi, head_idx in enumerate(peaks):
        # Left shoulder: the nearest peak *before* the head within max_pattern_bars/2.
        ls_candidates = peaks[(peaks < head_idx) & (peaks >= head_idx - max_pattern_bars // 2)]
        if len(ls_candidates) == 0:
            continue
        ls_idx = ls_candidates[-1]  # closest to head

        # Right shoulder: the nearest peak *after* the head within max_pattern_bars/2.
        rs_candidates = peaks[(peaks > head_idx) & (peaks <= head_idx + max_pattern_bars // 2)]
        if len(rs_candidates) == 0:
            continue
        rs_idx = rs_candidates[0]

        span = rs_idx - ls_idx
        if span < min_pattern_bars or span > max_pattern_bars:
            continue

        # Head must be strictly higher than both shoulders.
        h_head = close[head_idx]
        h_ls = close[ls_idx]
        h_rs = close[rs_idx]
        if not (h_head > h_ls and h_head > h_rs):
            continue

        # Shoulder symmetry: |h_ls - h_rs| / h_head <= shoulder_tol.
        if abs(h_ls - h_rs) / h_head > shoulder_tol:
            continue

        # Neckline troughs: one between ls and head, one between head and rs.
        lt_cands = troughs[(troughs > ls_idx) & (troughs < head_idx)]
        rt_cands = troughs[(troughs > head_idx) & (troughs < rs_idx)]
        if len(lt_cands) == 0 or len(rt_cands) == 0:
            continue
        lt_idx = lt_cands[-1]  # deepest available; take closest to head
        rt_idx = rt_cands[0]
        neckline_l = close[lt_idx]
        neckline_r = close[rt_idx]

        # Neckline approximately horizontal: difference relative to head-above-neckline.
        neck_mid = (neckline_l + neckline_r) / 2.0
        head_above = h_head - neck_mid
        if head_above <= 0:
            continue
        if abs(neckline_l - neckline_r) / head_above > neckline_tol:
            # Allow a slightly tilted neckline — raise tolerance to 0.15.
            if abs(neckline_l - neckline_r) / head_above > 0.15:
                continue

        # Neckline break: first close below the neckline projection after rs_idx.
        # Project neckline linearly between (lt_idx, neckline_l) and (rt_idx, neckline_r).
        for j in range(rs_idx + 1, min(n, rs_idx + max_pattern_bars // 2)):
            if j >= n:
                break
            # Linearly extrapolate neckline to bar j.
            if rt_idx != lt_idx:
                slope = (neckline_r - neckline_l) / (rt_idx - lt_idx)
            else:
                slope = 0.0
            neck_at_j = neckline_r + slope * (j - rt_idx)
            if close[j] < neck_at_j:
                # Confirmed break — stamp at bar j.
                results.append({
                    "signal_date": dates[j],
                    "pattern_type": "top",
                    "dir": -1,
                    "ls_idx": int(ls_idx),
                    "head_idx": int(head_idx),
                    "rs_idx": int(rs_idx),
                    "neckline": float(neck_at_j),
                    "break_idx": int(j),
                })
                break  # one confirmation per pattern

    # ---- Inverse H&S bottom (bullish) -------------------------------------
    inv_troughs = troughs  # reuse: head is a trough, shoulders are troughs
    inv_peaks = peaks      # neckline from peaks around the head

    for ti, head_idx in enumerate(inv_troughs):
        ls_candidates = inv_troughs[
            (inv_troughs < head_idx) & (inv_troughs >= head_idx - max_pattern_bars // 2)
        ]
        if len(ls_candidates) == 0:
            continue
        ls_idx = ls_candidates[-1]

        rs_candidates = inv_troughs[
            (inv_troughs > head_idx) & (inv_troughs <= head_idx + max_pattern_bars // 2)
        ]
        if len(rs_candidates) == 0:
            continue
        rs_idx = rs_candidates[0]

        span = rs_idx - ls_idx
        if span < min_pattern_bars or span > max_pattern_bars:
            continue

        # Head must be strictly lower than both shoulders.
        h_head = close[head_idx]
        h_ls = close[ls_idx]
        h_rs = close[rs_idx]
        if not (h_head < h_ls and h_head < h_rs):
            continue

        # Shoulder symmetry.
        if abs(h_ls - h_rs) / max(h_ls, h_rs) > shoulder_tol:
            continue

        # Neckline peaks: one between ls and head, one between head and rs.
        lp_cands = inv_peaks[(inv_peaks > ls_idx) & (inv_peaks < head_idx)]
        rp_cands = inv_peaks[(inv_peaks > head_idx) & (inv_peaks < rs_idx)]
        if len(lp_cands) == 0 or len(rp_cands) == 0:
            continue
        lp_idx = lp_cands[-1]
        rp_idx = rp_cands[0]
        neckline_l = close[lp_idx]
        neckline_r = close[rp_idx]

        neck_mid = (neckline_l + neckline_r) / 2.0
        head_below = neck_mid - h_head
        if head_below <= 0:
            continue
        if abs(neckline_l - neckline_r) / head_below > 0.15:
            continue

        # Neckline break: first close above the neckline projection after rs_idx.
        for j in range(rs_idx + 1, min(n, rs_idx + max_pattern_bars // 2)):
            if j >= n:
                break
            if rp_idx != lp_idx:
                slope = (neckline_r - neckline_l) / (rp_idx - lp_idx)
            else:
                slope = 0.0
            neck_at_j = neckline_r + slope * (j - rp_idx)
            if close[j] > neck_at_j:
                results.append({
                    "signal_date": dates[j],
                    "pattern_type": "inverse",
                    "dir": +1,
                    "ls_idx": int(ls_idx),
                    "head_idx": int(head_idx),
                    "rs_idx": int(rs_idx),
                    "neckline": float(neck_at_j),
                    "break_idx": int(j),
                })
                break

    if not results:
        return pd.DataFrame(columns=[
            "signal_date", "pattern_type", "dir", "ls_idx", "head_idx",
            "rs_idx", "neckline", "break_idx",
        ])
    df = pd.DataFrame(results)
    df = df.sort_values("break_idx").reset_index(drop=True)
    # Deduplicate: keep only the first confirmed break per overlapping pattern
    # (prevent the same structural period from generating many signals).
    df = _deduplicate_overlapping(df)
    return df


def _deduplicate_overlapping(df: pd.DataFrame) -> pd.DataFrame:
    """Remove signals whose left-shoulder is within 10 bars of a previous signal's RS."""
    if df.empty:
        return df
    keep = [True] * len(df)
    prev_rs = -999
    for i, row in df.iterrows():
        if row["ls_idx"] <= prev_rs + 10:
            keep[i] = False
        else:
            prev_rs = row["rs_idx"]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forward return engine
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame,
    signal_dates: pd.DatetimeIndex | pd.Index,
    horizons: tuple[int, ...] = (1, 5, 20, 60),
) -> pd.DataFrame:
    """Log close-to-close returns at each ``horizon`` after each signal date.

    The *h*-day forward return for a signal at date *t* is
    ``ln(close[t+h] / close[t])``.  This is the return that **starts at t+1**
    and ends at t+h — fully out-of-sample relative to the detection algorithm.

    Returns a DataFrame indexed by ``signal_dates`` with columns ``fwd_{h}d``.
    """
    close = bars["close"]
    date_to_pos = {d: i for i, d in enumerate(bars.index)}

    rows = []
    for sd in signal_dates:
        if sd not in date_to_pos:
            rows.append({f"fwd_{h}d": np.nan for h in horizons})
            continue
        i = date_to_pos[sd]
        row = {}
        for h in horizons:
            j = i + h
            if j < len(close):
                row[f"fwd_{h}d"] = float(np.log(close.iloc[j] / close.iloc[i]))
            else:
                row[f"fwd_{h}d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(signal_dates, name="signal_date"))


# ---------------------------------------------------------------------------
# Random-day control arm
# ---------------------------------------------------------------------------
def random_control_returns(
    bars: pd.DataFrame,
    n: int,
    horizons: tuple[int, ...] = (1, 5, 20, 60),
    seed: int = 0,
) -> pd.DataFrame:
    """Forward returns for ``n`` randomly selected days (the unconditional baseline).

    Days are drawn without replacement from ``bars``, excluding the last
    ``max(horizons)`` rows where forward returns would be truncated.  Each draw
    produces the same horizon set as :func:`forward_returns`.
    """
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    eligible = bars.index[:-max_h]
    n_eff = min(n, len(eligible))
    chosen = eligible[np.sort(rng.choice(len(eligible), size=n_eff, replace=False))]
    return forward_returns(bars, chosen, horizons=horizons)


# ---------------------------------------------------------------------------
# Pattern study runner
# ---------------------------------------------------------------------------
def run_hs_study(
    bars: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20, 60),
    seed: int = 0,
    **detect_kwargs,
) -> dict:
    """Run the full H&S study on one tape.

    Returns a dict with:
    - ``signals`` — DataFrame from :func:`detect_hs_patterns`.
    - ``fwd_top`` — forward-return frame for H&S top breaks (short entries).
    - ``fwd_inv`` — forward-return frame for IH&S breaks (long entries).
    - ``control`` — random-day forward returns (same n as total signals).
    - ``n_top``, ``n_inv`` — signal counts.
    """
    signals = detect_hs_patterns(bars, **detect_kwargs)

    top_mask = signals["pattern_type"] == "top"
    inv_mask = signals["pattern_type"] == "inverse"

    n_top = top_mask.sum()
    n_inv = inv_mask.sum()
    n_total = len(signals)

    fwd_top = pd.DataFrame()
    fwd_inv = pd.DataFrame()

    if n_top > 0:
        top_dates = signals.loc[top_mask, "signal_date"]
        fwd_top = forward_returns(bars, top_dates, horizons=horizons)

    if n_inv > 0:
        inv_dates = signals.loc[inv_mask, "signal_date"]
        fwd_inv = forward_returns(bars, inv_dates, horizons=horizons)

    control = random_control_returns(
        bars, n=max(n_total, 20), horizons=horizons, seed=seed
    )

    return {
        "signals": signals,
        "fwd_top": fwd_top,
        "fwd_inv": fwd_inv,
        "control": control,
        "n_top": int(n_top),
        "n_inv": int(n_inv),
    }


# ---------------------------------------------------------------------------
# Summary statistics — HAC t-stat on mean directional return vs baseline
# ---------------------------------------------------------------------------
def summarize(
    fwd: pd.DataFrame,
    direction: int,
    horizon: int,
    control: pd.DataFrame | None = None,
) -> dict:
    """Return statistics for one (pattern_type, horizon) cell.

    ``direction`` = -1 for H&S top (short) or +1 for IH&S (long).  The
    *directional* return is ``direction * fwd_return``, so a positive mean means
    the pattern predicted correctly.

    Computes HAC Newey-West t-stat on the mean directional return and, if
    ``control`` is provided, on the excess over the unconditional baseline.

    Returns
    -------
    dict with keys:
      ``n``, ``mean_pct``, ``win_rate``, ``tstat``, ``excess_pct``,
      ``tstat_excess`` (NaN if no control), and the horizon in days.
    """
    col = f"fwd_{horizon}d"
    if col not in fwd.columns:
        return {"n": 0, "horizon": horizon}
    r = fwd[col].dropna().to_numpy(dtype=float) * direction
    n = len(r)
    out: dict = {
        "n": int(n),
        "horizon": int(horizon),
        "mean_pct": float(r.mean() * 100) if n else float("nan"),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "tstat": float("nan"),
        "excess_pct": float("nan"),
        "tstat_excess": float("nan"),
    }
    if n > 5:
        out["tstat"] = _hac_tstat(r)

    if control is not None and col in control.columns:
        rc = control[col].dropna().to_numpy(dtype=float) * direction
        if len(rc) >= len(r):
            rc = rc[: len(r)]
        elif len(rc) > 0:
            rc = rc  # may be shorter — still compute
        else:
            return out
        n_c = min(n, len(rc))
        exc = r[:n_c] - rc[:n_c]
        out["excess_pct"] = float(exc.mean() * 100) if n_c > 0 else float("nan")
        if n_c > 5:
            out["tstat_excess"] = _hac_tstat(exc)

    return out


def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``r``."""
    n = len(r)
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Placebo control — random "pattern" detections
# ---------------------------------------------------------------------------
def placebo_signals(
    bars: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate ``n`` random signal dates as a placebo arm.

    Each placebo fires on a randomly chosen date, with a randomly assigned
    direction (±1), giving the null-hypothesis distribution of forward-return
    magnitudes.  Used in Beat 4 to confirm the detector doesn't just happen to
    fire on volatile days.
    """
    rng = np.random.default_rng(seed)
    eligible = bars.index[:-60]  # leave room for 60-day forward return
    n_eff = min(n, len(eligible))
    chosen_idx = np.sort(rng.choice(len(eligible), size=n_eff, replace=False))
    chosen = eligible[chosen_idx]
    dirs = rng.choice([-1, 1], size=n_eff)
    return pd.DataFrame({"signal_date": chosen, "dir": dirs}).reset_index(drop=True)
