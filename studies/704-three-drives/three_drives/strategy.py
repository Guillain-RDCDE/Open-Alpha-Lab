"""The mechanical Three-Drives detector + its honest controls — Study 704.

The "Three Drives" pattern (Larry Pesavento's harmonic-trading lineage, also taught alongside
Elliott Wave and Gartley material) reads five labelled swing pivots — point 1 through point 5,
with an implicit start "point 0" before the first drive — forming **three drives** (the legs
0->1, 2->3, 4->5) separated by **two corrections** (1->2, 3->4). The claim is Fibonacci
proportion on *both* legs: each correction retraces **0.382-0.886** of the drive before it, and
each drive extends the prior correction by **1.13-2.618x** (most-cited point estimate ~0.618 /
~1.27) — three "symmetric", Fibonacci-spaced pushes that exhaust the trend, so price is supposed
to reverse hard once drive 3 completes at point 5.

Pipeline (shares the ZigZag vocabulary with sibling 697-wolfe-waves, so the two Fibonacci
five-point patterns are directly comparable):

1. **ZigZag pivots** (percentage reversal filter) — the same swing-marking tool as 697/445.
2. **``three_drives_candidates``** — slide a window of 6 consecutive alternating pivots
   (point 0..point 5) and keep the ones whose leg ratios land on the Fibonacci grid (both
   "three drives up" expecting a reversal DOWN, and "three drives down" expecting a reversal UP),
   with each drive strictly extending the one before it (point 3 beyond point 1, point 5 beyond
   point 3 — the "extending drives" signature).
3. **The fade** — unlike Wolfe Waves' EPA line or Gartley's D-point, Three Drives has **no
   price target**: the claim is pure reversal. We enter *against* the three drives at point 5's
   confirmation, one bar later (one documented execution lag), and measure the forward return at
   fixed horizons vs a **random-time, random-direction placebo** on the same tape (the honest
   "does knowing 'three Fibonacci drives just finished' beat a coin flip?" null).
4. **Fibonacci-grid placebo** — swap the specific 0.382-0.886 / 1.13-2.618 bands for a *random*
   ratio grid, keeping the exact same ZigZag machinery and "extending drives" ordering rule. If the
   *particular* Fibonacci numbers are what marks a genuine reversal, the real grid should beat
   almost every random grid.
5. **Time-symmetry myth-check** — the folklore's own word is "symmetric": we score every detected
   pattern by how *evenly spaced in time* its three drives are (the coefficient of variation of
   the three drive-leg bar-counts) and test whether the more time-symmetric half of detections
   forward-returns better than the less time-symmetric half — the direct, falsifiable version of
   "symmetric" as a claim about *predictive* content, not just geometry.

No look-ahead: a pivot at bar p is only *confirmed* once price has reversed the ZigZag threshold
from it (bar q > p); every entry uses point 5's confirmation bar, and execution is the **next**
bar's close (one documented execution lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)

# The Three-Drives Fibonacci grid, the widest mechanical version a proponent would accept (the
# full retracement/extension bands most trading-education sources cite for this pattern): each
# correction retraces 0.382-0.886 of the prior drive (same band sibling 468-gartley-harmonic uses
# for its C_of_AB leg); each drive extends the prior correction by 1.13-2.618x.
THREE_DRIVES = {
    "corr_lo": 0.382, "corr_hi": 0.886,
    "ext_lo": 1.13, "ext_hi": 2.618,
}


# --------------------------------------------------------------------------- #
# ZigZag swing detector (shared vocabulary with 445-elliott-wave / 697-wolfe-waves)
# --------------------------------------------------------------------------- #
def zigzag(close: np.ndarray, pct: float = 0.04) -> pd.DataFrame:
    """Percentage ZigZag pivots on a close series.

    Walk forward tracking the running extreme since the last confirmed pivot. A new pivot is
    *confirmed* the first time price reverses by at least ``pct`` (e.g. 0.04 = 4%) from that
    extreme; the pivot is stamped at the bar of the extreme (``piv_idx``) but only *known* at
    the reversal bar (``conf_idx``) — the look-ahead-free confirmation bar.

    Returns a DataFrame of alternating pivots with columns:
      ``piv_idx`` (bar of the extreme), ``conf_idx`` (bar it was confirmed = known),
      ``price`` (extreme price), ``kind`` (+1 = swing high, -1 = swing low).
    """
    n = len(close)
    if n < 3:
        return pd.DataFrame(columns=["piv_idx", "conf_idx", "price", "kind"])

    pivots = []
    hi_idx = lo_idx = 0
    hi_px = lo_px = close[0]
    direction = 0  # 0 = undetermined, +1 = uptrend (seeking next high), -1 = downtrend

    for i in range(1, n):
        px = close[i]
        if direction >= 0 and px > hi_px:
            hi_px, hi_idx = px, i
        if direction <= 0 and px < lo_px:
            lo_px, lo_idx = px, i

        if direction == 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
            elif px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i
        elif direction > 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
        else:
            if px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i

    piv = pd.DataFrame(pivots, columns=["piv_idx", "conf_idx", "price", "kind"])
    return piv.drop_duplicates(subset=["piv_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Three-Drives geometry detector
# --------------------------------------------------------------------------- #
def three_drives_candidates(pivots: pd.DataFrame, grid: dict = THREE_DRIVES) -> pd.DataFrame:
    """Flag 6-pivot windows (point0..point5) matching the Three-Drives Fibonacci geometry.

    "Three drives UP" (kinds low,high,low,high,low,high -> reversal DOWN expected off point 5):
      drive1 = p1-p0 ; corr1 = p1-p2 ; drive2 = p3-p2 ; corr2 = p3-p4 ; drive3 = p5-p4  (all > 0)
      corr1/drive1 and corr2/drive2 land in [``corr_lo``, ``corr_hi``];
      drive2/corr1 and drive3/corr2 land in [``ext_lo``, ``ext_hi``];
      "extending drives": p3 > p1 and p5 > p3.

    "Three drives DOWN" is the exact mirror (reversal UP expected off point 5).

    Point 5's confirmation bar (``sig_idx``) is when the pattern becomes knowable; entry is the
    *next* close (one documented execution lag, applied downstream by :func:`forward_returns`).
    Also carries the three drive-leg bar-counts, for the time-symmetry myth-check.
    """
    if len(pivots) < 6:
        return pd.DataFrame()
    kinds = pivots["kind"].to_numpy()
    idxs = pivots["piv_idx"].to_numpy()
    confs = pivots["conf_idx"].to_numpy()
    prices = pivots["price"].to_numpy()

    clo, chi = grid["corr_lo"], grid["corr_hi"]
    elo, ehi = grid["ext_lo"], grid["ext_hi"]

    out = []
    for k in range(5, len(pivots)):
        kk = kinds[k - 5:k + 1]
        if np.array_equal(kk, [-1, 1, -1, 1, -1, 1]):
            direction = -1                     # three drives UP -> fade SHORT
        elif np.array_equal(kk, [1, -1, 1, -1, 1, -1]):
            direction = 1                      # three drives DOWN -> fade LONG
        else:
            continue

        i0, i1, i2, i3, i4, i5 = idxs[k - 5:k + 1]
        p0, p1, p2, p3, p4, p5 = prices[k - 5:k + 1]

        if direction == -1:
            drive1, corr1, drive2, corr2, drive3 = p1 - p0, p1 - p2, p3 - p2, p3 - p4, p5 - p4
            extending = p3 > p1 and p5 > p3
        else:
            drive1, corr1, drive2, corr2, drive3 = p0 - p1, p2 - p1, p2 - p3, p4 - p3, p4 - p5
            extending = p3 < p1 and p5 < p3

        legs = (drive1, corr1, drive2, corr2, drive3)
        if min(legs) <= 0 or not extending:
            continue

        r_corr1 = corr1 / drive1
        r_ext1 = drive2 / corr1
        r_corr2 = corr2 / drive2
        r_ext2 = drive3 / corr2

        if not (clo <= r_corr1 <= chi) or not (clo <= r_corr2 <= chi):
            continue
        if not (elo <= r_ext1 <= ehi) or not (elo <= r_ext2 <= ehi):
            continue

        sig_idx = int(confs[k])
        bars012 = int(i1 - i0)                 # drive-1 duration (bars)
        bars234 = int(i3 - i2)                 # drive-2 duration
        bars45 = int(i5 - i4)                  # drive-3 duration
        drive_bars = np.array([bars012, bars234, bars45], dtype=float)
        sym_cv = float(drive_bars.std(ddof=0) / drive_bars.mean()) if drive_bars.mean() > 0 \
            else float("nan")

        out.append({
            "i0": int(i0), "i1": int(i1), "i2": int(i2), "i3": int(i3), "i4": int(i4),
            "i5": int(i5), "dir": int(direction), "sig_idx": sig_idx,
            "r_corr1": float(r_corr1), "r_ext1": float(r_ext1),
            "r_corr2": float(r_corr2), "r_ext2": float(r_ext2),
            "drive1_bars": bars012, "drive2_bars": bars234, "drive3_bars": bars45,
            "sym_cv": sym_cv,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Forward-return engine — the fade
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series | np.ndarray, entries: pd.DataFrame, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Fixed-horizon return of the FADE trade: enter one bar after ``sig_idx``, exit ``horizon``
    bars later, signed by ``dir`` (the reversal direction). ``cost_bps`` is one-way, charged
    on both legs (2 x ``cost_bps``)."""
    p = close.to_numpy(dtype=float) if hasattr(close, "to_numpy") else np.asarray(close, float)
    n = p.size
    if len(entries) == 0:
        return np.asarray([], dtype=float)
    out = []
    for sig, d in zip(entries["sig_idx"].to_numpy(dtype=int), entries["dir"].to_numpy(dtype=int)):
        e = sig + 1
        x = e + horizon
        if e >= n or x >= n:
            continue
        gross = d * (p[x] / p[e] - 1.0)
        out.append(gross - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


def random_entries(n_bars: int, n: int, seed: int = 0, warmup: int = 50) -> pd.DataFrame:
    """``n`` random (time, direction) entries — the coin-flip base rate: same timing freedom,
    same fair-coin direction, no pattern knowledge at all."""
    rng = np.random.default_rng(seed)
    hi = max(n_bars - warmup - 1, warmup + 1)
    m = min(n, max(hi - warmup, 0))
    if m <= 0:
        return pd.DataFrame(columns=["sig_idx", "dir"])
    idx = rng.choice(np.arange(warmup, hi), size=m, replace=False)
    dirs = rng.choice([-1, 1], size=m)
    order = np.argsort(idx)
    return pd.DataFrame({"sig_idx": idx[order], "dir": dirs[order]})


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ttest_vs_zero(sample: np.ndarray) -> float:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
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


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate (+ Wilson), mean (bps), one-sample t, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n": 0, "win": float("nan"), "wilson_lo": float("nan"), "wilson_hi": float("nan"),
                "mean_bps": float("nan"), "t": float("nan"), "hac_t": float("nan")}
    wins = int((r > 0).sum())
    lo, hi = wilson_interval(wins, n)
    return {
        "n": int(n), "win": wins / n, "wilson_lo": lo, "wilson_hi": hi,
        "mean_bps": float(r.mean() * 1e4), "t": ttest_vs_zero(r), "hac_t": hac_t(r),
    }


# --------------------------------------------------------------------------- #
# Pooled detection + fade test across the basket
# --------------------------------------------------------------------------- #
def basket_pivots(basket: dict, pct: float = 0.04) -> dict:
    """ZigZag pivots for every ticker, computed once. Reuse across grid-placebo draws — the
    ZigZag itself doesn't depend on the Fibonacci grid, only ``three_drives_candidates`` does."""
    return {tk: zigzag(bars["close"].to_numpy(float), pct=pct) for tk, bars in basket.items()}


def pooled_fade_test(basket: dict, pct: float = 0.04, horizon: int = 20, cost_bps: float = 5.0,
                     grid: dict = THREE_DRIVES, pivots: dict | None = None) -> dict:
    """Detect + fade on every ticker in ``basket``; pool the per-trade returns.

    Pass a precomputed ``pivots`` dict (from :func:`basket_pivots`) to skip re-running the
    ZigZag — the expensive step — when only the Fibonacci ``grid`` changes (the ratio placebo).

    Returns gross/net returns, per-ticker (pivots, candidates) counts, and the pooled entries
    frame (for the symmetry myth-check and the placebo).
    """
    counts, entries_list, gross_list, net_list = {}, [], [], []
    for tk, bars in basket.items():
        piv = pivots[tk] if pivots is not None else zigzag(bars["close"].to_numpy(float), pct=pct)
        ent = three_drives_candidates(piv, grid=grid)
        counts[tk] = {"pivots": len(piv), "candidates": len(ent)}
        if len(ent) == 0:
            continue
        ent = ent.assign(ticker=tk)
        entries_list.append(ent)
        gross_list.append(forward_returns(bars["close"], ent, horizon, cost_bps=0.0))
        net_list.append(forward_returns(bars["close"], ent, horizon, cost_bps=cost_bps))
    entries = pd.concat(entries_list, ignore_index=True) if entries_list else pd.DataFrame()
    gross = np.concatenate(gross_list) if gross_list else np.asarray([], dtype=float)
    net = np.concatenate(net_list) if net_list else np.asarray([], dtype=float)
    return {"counts": counts, "entries": entries, "gross": gross, "net": net}


def pooled_random_baseline(basket: dict, per_ticker_n: dict, horizon: int, seed: int = 704,
                           n_reps: int = 1) -> np.ndarray:
    """The coin-flip (random time, random direction) baseline, matched per ticker to the number
    of real candidates found on that ticker, pooled across the basket."""
    out = []
    for tk_i, (tk, bars) in enumerate(sorted(basket.items())):
        n = per_ticker_n.get(tk, 0)
        if n <= 0:
            continue
        for r in range(n_reps):
            # deterministic per-ticker offset: NOT Python's hash() (randomized per-process,
            # PEP 456 — would silently break reproducibility across runs).
            re = random_entries(len(bars), n, seed=seed + tk_i * 97 + r)
            out.append(forward_returns(bars["close"], re, horizon))
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def coin_placebo_pvalue(basket: dict, per_ticker_n: dict, horizon: int, obs_mean: float,
                        n_draws: int = 1000, seed: int = 704) -> dict:
    """Draw ``n_draws`` full coin-flip replays (same per-ticker candidate counts), each producing
    a pooled mean fade return; ``p`` = share of draws whose mean >= the observed fade mean (the
    "would knowing 'three drives just finished' beat blind timing+direction?" null)."""
    means = np.empty(n_draws)
    for d in range(n_draws):
        r = pooled_random_baseline(basket, per_ticker_n, horizon, seed=seed * 1000 + d, n_reps=1)
        means[d] = np.nanmean(r) if r.size else np.nan
    means = means[np.isfinite(means)]
    p = float((means >= obs_mean).mean()) if means.size else float("nan")
    return {"obs": obs_mean, "placebo_mean": float(means.mean()) if means.size else float("nan"),
            "placebo_sd": float(means.std(ddof=1)) if means.size > 1 else float("nan"),
            "p_value": p, "n_draws": int(means.size)}


# --------------------------------------------------------------------------- #
# Fibonacci-grid placebo — are the specific ratios load-bearing?
# --------------------------------------------------------------------------- #
def ratio_grid_placebo(basket: dict, pct: float, horizon: int, n_draws: int = 500,
                       seed: int = 704) -> dict:
    """Placebo: replace the Fibonacci grid with a *random* ratio grid, keeping all machinery
    (ZigZag pivots, the 6-point window, the "extending drives" ordering rule). If price genuinely
    reverses at *Fibonacci*-proportioned three-drives, the real grid should beat almost every
    random grid's mean fade return, pooled across the basket. Returns the share of random grids
    whose mean fade return **beats** the real one."""
    piv = basket_pivots(basket, pct=pct)
    real = pooled_fade_test(basket, pct=pct, horizon=horizon, cost_bps=0.0, pivots=piv)
    obs = float(np.mean(real["gross"])) if real["gross"].size else float("nan")
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats, valid = 0, 0
    for _ in range(n_draws):
        clo = rng.uniform(0.15, 0.55)
        chi = clo + rng.uniform(0.20, 0.55)
        elo = rng.uniform(0.9, 1.35)
        ehi = elo + rng.uniform(0.5, 1.8)
        grid = {"corr_lo": clo, "corr_hi": chi, "ext_lo": elo, "ext_hi": ehi}
        rr = pooled_fade_test(basket, pct=pct, horizon=horizon, cost_bps=0.0, grid=grid, pivots=piv)
        if rr["gross"].size == 0:
            continue
        valid += 1
        if float(np.mean(rr["gross"])) >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Time-symmetry myth-check — does "symmetric" carry predictive content?
# --------------------------------------------------------------------------- #
def symmetry_split_test(entries: pd.DataFrame, returns: np.ndarray) -> dict:
    """Split detections at the median time-symmetry score (lower ``sym_cv`` = more evenly spaced
    in time) and Welch-t the fade returns of the "more symmetric" half vs the "less symmetric"
    half. Tests the folklore's own word — "symmetric" — as a claim about *predictive* content,
    not just geometry that the detector already enforces on price.
    """
    if len(entries) < 10 or len(returns) != len(entries):
        return {"n": len(entries), "median_cv": float("nan"), "mean_sym_bps": float("nan"),
                "mean_asym_bps": float("nan"), "welch_t": float("nan")}
    cv = entries["sym_cv"].to_numpy(float)
    med = float(np.nanmedian(cv))
    sym_mask = cv <= med
    a, b = returns[sym_mask], returns[~sym_mask]
    return {
        "n": len(entries), "median_cv": med,
        "mean_sym_bps": float(np.nanmean(a)) * 1e4 if a.size else float("nan"),
        "mean_asym_bps": float(np.nanmean(b)) * 1e4 if b.size else float("nan"),
        "welch_t": welch_t(a, b),
    }
