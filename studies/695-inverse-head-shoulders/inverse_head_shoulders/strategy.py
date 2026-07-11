"""Detector + timing rule + honest arbiters — Study 695 (Inverse Head-and-Shoulders).

The inverse head-and-shoulders (IH&S) is the textbook bullish **bottoming** figure (Edwards &
Magee, *Technical Analysis of Stock Trends*, 1948; Bulkowski, *Encyclopedia of Chart Patterns*,
2021 ed., reports it among the higher-scoring bottoms in his sample). The folk recipe:

* Price makes a trough (**left shoulder**), rallies to a **neckline**, falls to a *deeper* trough
  (the **head**), rallies to the neckline again, falls to a *third*, shallower trough (**right
  shoulder**, roughly as deep as the left one) — then **breaks and closes above the neckline**.
  The figure is *confirmed* on the neckline break: the downtrend is exhausted, you **buy**.
* The **measured-move target**: the head-to-neckline vertical distance, projected *up* from the
  neckline. "The pattern's height tells you the target" is the second half of the folklore, tested
  here directly (not just in siblings 188/415/416).

Chart figures are partly in the eye of the beholder, so — exactly like siblings 188 (H&S top/
inverse), 415 (triple top/bottom) and 416 (rounding bottom) — we test the closest **mechanical**
definition we can write down and say so loudly. Our detector, on daily closes:

* Find **swing pivots** (local highs/lows over a symmetric window ``w``).
* Candidate **head** = a swing low; **left/right shoulders** = the nearest swing lows before/
  after it, each within ``fig_max // 2`` bars. The head must be strictly the deepest of the three,
  and the two shoulders must sit within ``shoulder_tol`` of each other (symmetry).
* The **neckline** = the line through the two intervening swing highs (one between LS and head,
  one between head and RS); rejected if its tilt exceeds ``neckline_tol`` of the head's depth
  below the neckline (near-horizontal, as the textbook draws it).
* A **confirmed breakout** = the first close, within ``search_buf`` bars of the right shoulder,
  that clears the (linearly projected) neckline by ``breakout_buf``. That is the entry signal,
  stamped at that bar's close. Non-overlapping — a pattern used for one breakout does not spawn a
  second.

Timing: the breakout is known at its close; we **enter the next day's open** (one documented lag,
zero look-ahead) and measure forward returns at fixed horizons *and* a "long timer" that holds
until the measured-move target is touched or a timeout elapses. Arbiters: one-sample + HAC *t* of
the post-breakout excess over the name's own base rate, a same-tape random-date placebo, a
measured-move hit-rate against a magnitude-matched null, costs, and the synthetic positive
control. This study is the **bullish, bottom-only** twin — the bearish top is study 188's job (see
``docs/references.md`` for the full dedup map).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)        # trading-day forward horizons


# --------------------------------------------------------------------------- #
# Swing pivots
# --------------------------------------------------------------------------- #
def swing_pivots(close: np.ndarray, w: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices of local-high and local-low swing pivots on ``close`` (symmetric ``w``-bar window)."""
    n = len(close)
    highs, lows = [], []
    for i in range(w, n - w):
        seg = close[i - w:i + w + 1]
        if close[i] == seg.max() and np.argmax(seg) == w:
            highs.append(i)
        if close[i] == seg.min() and np.argmin(seg) == w:
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


# --------------------------------------------------------------------------- #
# Inverse head-and-shoulders detector
# --------------------------------------------------------------------------- #
def detect_inverse_hs(close: np.ndarray, w: int = 5, shoulder_tol: float = 0.12,
                      neckline_tol: float = 0.15, fig_min: int = 20, fig_max: int = 200,
                      breakout_buf: float = 0.0, search_buf: int = 40) -> list[dict]:
    """Detect confirmed inverse-H&S (bullish bottom) breakouts on a close series.

    Returns a list of dicts, one per confirmed breakout, with integer bar positions and levels:
    ``ls_idx``, ``head_idx``, ``rs_idx`` (the three troughs), ``neckline`` (the level at the
    breakout bar, linearly projected), ``breakout_idx`` (the signal bar), ``head_price``,
    ``target`` (the measured-move projection: neckline-midpoint + (neckline-midpoint − head)).
    """
    n = len(close)
    highs, lows = swing_pivots(close, w)
    out: list[dict] = []
    used_until = -1

    for head_idx in lows:
        if head_idx <= used_until:
            continue
        ls_cands = lows[(lows < head_idx) & (lows >= head_idx - fig_max // 2)]
        if len(ls_cands) == 0:
            continue
        ls_idx = ls_cands[-1]
        rs_cands = lows[(lows > head_idx) & (lows <= head_idx + fig_max // 2)]
        if len(rs_cands) == 0:
            continue
        rs_idx = rs_cands[0]

        span = rs_idx - ls_idx
        if span < fig_min or span > fig_max:
            continue

        h_head, h_ls, h_rs = close[head_idx], close[ls_idx], close[rs_idx]
        if not (h_head < h_ls and h_head < h_rs):
            continue
        if abs(h_ls - h_rs) / max(h_ls, h_rs) > shoulder_tol:
            continue

        lp_cands = highs[(highs > ls_idx) & (highs < head_idx)]
        rp_cands = highs[(highs > head_idx) & (highs < rs_idx)]
        if len(lp_cands) == 0 or len(rp_cands) == 0:
            continue
        lp_idx, rp_idx = lp_cands[-1], rp_cands[0]
        neck_l, neck_r = close[lp_idx], close[rp_idx]

        neck_mid = (neck_l + neck_r) / 2.0
        head_below = neck_mid - h_head
        if head_below <= 0:
            continue
        if abs(neck_l - neck_r) / head_below > neckline_tol:
            continue

        bo = None
        scan_end = min(rs_idx + search_buf, n)
        slope = (neck_r - neck_l) / (rp_idx - lp_idx) if rp_idx != lp_idx else 0.0
        for j in range(rs_idx + 1, scan_end):
            neck_at_j = neck_r + slope * (j - rp_idx)
            if close[j] > neck_at_j * (1.0 + breakout_buf):
                bo = j
                neckline_at_break = neck_at_j
                break
        if bo is None:
            continue

        height = neck_mid - h_head
        out.append({
            "ls_idx": int(ls_idx), "head_idx": int(head_idx), "rs_idx": int(rs_idx),
            "breakout_idx": int(bo), "neckline": float(neckline_at_break),
            "head_price": float(h_head), "height": float(height),
            "target": float(neck_mid + height),
        })
        used_until = bo
    return out


# --------------------------------------------------------------------------- #
# Forward returns after the breakout
# --------------------------------------------------------------------------- #
def forward_returns(close: np.ndarray, signal_idx: list[int], horizon: int,
                    lag: int = 1) -> np.ndarray:
    """Forward ``horizon``-day return, entered ``lag`` bars after each signal's close (long only).

    Signal known at ``signal_idx`` close; enter at ``signal_idx + lag`` close (one documented,
    zero-look-ahead lag: the pattern and its break are fully determined by bar ``signal_idx``);
    exit ``horizon`` bars later. Drops signals whose window overruns the tape.
    """
    out = []
    n = len(close)
    for si in signal_idx:
        entry = si + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= n:
            continue
        out.append(close[exit_] / close[entry] - 1.0)
    return np.asarray(out, dtype=float)


def base_rate(close: np.ndarray, horizon: int) -> float:
    """The name's own unconditional mean forward ``horizon``-day return (the long base rate)."""
    if len(close) <= horizon + 1:
        return float("nan")
    r = close[horizon:] / close[:-horizon] - 1.0
    return float(np.mean(r))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``sample`` mean against ``mu0``. NaN if fewer than 2 points."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def hac_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """Newey-West (HAC) t of the mean against ``mu0`` — autocorrelation-robust."""
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return one_sample_t(r, mu0)
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wgt * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    if se == 0:
        return float("nan")
    return float((mu - mu0) / se)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Panel-level orchestration — forward-return excess over base rate
# --------------------------------------------------------------------------- #
def collect_signals(panel: dict, names: list[str] | None = None,
                    **detect_kw) -> dict[str, list[dict]]:
    """Run :func:`detect_inverse_hs` on every name in the panel; ticker -> signal list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_inverse_hs(c, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 695, **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect confirmed inverse-H&S breakouts, take the forward-``horizon`` return
    after each (1-day lag), and subtract the name's own base rate (``excess=True``) so the test is
    "does the figure beat buy-and-hold *for that name*", not "is the market up". Pools the
    per-event excess across names; reports n, mean, win-rate, one-sample t, HAC t, a same-tape
    placebo p, and the net-of-cost mean (one round trip = 2 * ``cost_bps`` one-way).
    """
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    pooled, raw_pooled = [], []
    n_break = 0
    placebo_means = []
    rng = np.random.default_rng(seed)
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        sigs = detect_inverse_hs(c, **detect_kw)
        n_break += len(sigs)
        sig_idx = [d["breakout_idx"] for d in sigs]
        fr = forward_returns(c, sig_idx, horizon, lag=lag)
        if len(fr) == 0:
            continue
        br = base_rate(c, horizon)
        raw_pooled.extend(fr.tolist())
        pooled.extend((fr - (br if excess else 0.0)).tolist())
        n = len(c)
        hi = n - horizon - lag - 1
        if hi > 1 and len(fr) > 0:
            # Vectorized: draw all n_draws x len(fr) random start points at once instead of a
            # Python-level loop (identical distribution, ~100x faster for large baskets).
            si = rng.integers(1, hi, size=(n_draws, len(fr)))
            entry = si + lag
            exit_ = entry + horizon
            r = c[exit_] / c[entry] - 1.0
            draw_means = r.mean(axis=1) - (br if excess else 0.0)
            placebo_means.extend(draw_means.tolist())
    pooled = np.asarray(pooled, dtype=float)
    raw_pooled = np.asarray(raw_pooled, dtype=float)
    if len(pooled) == 0:
        return {"horizon": horizon, "n_breakouts": int(n_break), "n_events": 0,
                "mean": float("nan"), "raw_mean": float("nan"), "win": float("nan"),
                "t": float("nan"), "hac_t": float("nan"), "p_placebo": float("nan"),
                "net": float("nan")}
    obs = float(pooled.mean())
    cost = 2.0 * cost_bps / 1e4
    p = (float(np.mean(np.asarray(placebo_means) >= obs))
         if placebo_means else float("nan"))
    return {
        "horizon": horizon, "n_breakouts": int(n_break), "n_events": int(len(pooled)),
        "mean": obs, "raw_mean": float(raw_pooled.mean()),
        "win": float((pooled > 0).mean()),
        "t": one_sample_t(pooled), "hac_t": hac_t(pooled),
        "p_placebo": p, "net": float(obs - cost),
    }


# --------------------------------------------------------------------------- #
# The measured-move target — does the pattern's own height forecast anything?
# --------------------------------------------------------------------------- #
def measured_move_hits(panel: dict, names: list[str] | None = None, max_days: int = 126,
                       lag: int = 1, **detect_kw) -> dict:
    """Hit rate of the classic measured-move target (neckline + head-to-neckline height).

    For each confirmed breakout, enter at ``breakout_idx + lag`` and check whether the ``high``
    price touches ``target`` within ``max_days`` trading days. Reports the observed hit rate
    (Wilson 95% CI) and, as the honest control, a **magnitude-matched placebo**: at ``n`` random
    entry points on the same tape, does price travel the *same relative distance* (drawn from the
    observed target/entry ratios) within ``max_days``? A pattern-blind random walk of matched size
    should already hit a target of that magnitude some of the time — the placebo is what "hit rate"
    must beat, not zero.
    """
    closes, highs = panel["close"], panel["high"]
    if names is None:
        names = list(closes.columns)
    hits, days_to_hit, rel_moves = [], [], []
    n_sig = 0
    rng = np.random.default_rng(695)
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        h = highs[tk].reindex(closes[tk].dropna().index).to_numpy(dtype=float)
        sigs = detect_inverse_hs(c, **detect_kw)
        for d in sigs:
            entry_idx = d["breakout_idx"] + lag
            if entry_idx >= len(c):
                continue
            n_sig += 1
            entry_px = c[entry_idx]
            rel_moves.append(d["target"] / entry_px - 1.0)
            window_end = min(entry_idx + max_days, len(h))
            window = h[entry_idx:window_end]
            hit_pos = np.where(window >= d["target"])[0]
            if len(hit_pos) > 0:
                hits.append(True)
                days_to_hit.append(int(hit_pos[0]))
            else:
                hits.append(False)

    hits = np.asarray(hits, dtype=bool)
    n = len(hits)
    k = int(hits.sum())
    lo, hi_ci = wilson_interval(k, n) if n else (float("nan"), float("nan"))

    # Magnitude-matched placebo: random entries on the same tapes, target = entry * (1 + rel_move)
    # drawn from the OBSERVED distribution of relative move sizes (so the null asks for a move of
    # the same typical size, not an arbitrarily easy or hard one).
    placebo_hits = 0
    placebo_n = 0
    if n > 0:
        rel_arr = np.asarray(rel_moves)
        for tk in names:
            c = closes[tk].dropna().to_numpy(dtype=float)
            h = highs[tk].reindex(closes[tk].dropna().index).to_numpy(dtype=float)
            hi_bound = len(c) - max_days - lag - 1
            if hi_bound < 2:
                continue
            draws = min(200, hi_bound)
            idx = rng.choice(hi_bound, size=draws, replace=False) + 1
            rels = rng.choice(rel_arr, size=draws, replace=True)
            for si, rel in zip(idx, rels):
                entry_idx = si + lag
                entry_px = c[entry_idx]
                target = entry_px * (1.0 + rel)
                window = h[entry_idx:entry_idx + max_days]
                placebo_n += 1
                if len(window) and window.max() >= target:
                    placebo_hits += 1

    return {
        "n_signals": n, "n_hit": k, "hit_rate": (k / n) if n else float("nan"),
        "hit_lo": lo, "hit_hi": hi_ci,
        "median_days_to_hit": float(np.median(days_to_hit)) if days_to_hit else float("nan"),
        "mean_rel_move": float(np.mean(rel_moves)) if rel_moves else float("nan"),
        "placebo_n": placebo_n, "placebo_hit": placebo_hits,
        "placebo_rate": (placebo_hits / placebo_n) if placebo_n else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The long timer — hold to target-or-timeout, net of costs
# --------------------------------------------------------------------------- #
def timer_pnl(panel: dict, names: list[str] | None = None, max_days: int = 126,
             cost_bps: float = 5.0, lag: int = 1, **detect_kw) -> dict:
    """Trade-level P&L of the measured-move rule: long from entry to target-hit or timeout.

    Entry at ``breakout_idx + lag`` open-equivalent (close used as the tradable price, consistent
    with the rest of the study); exit at the **close** of the day the ``high`` first touches
    ``target`` (a same-day fill is optimistic — named), else exit at the ``max_days`` close
    (timeout). One round trip = 2 * ``cost_bps`` one-way. Reports gross/net mean per-trade return,
    one-sample + HAC t vs zero, and the excess over a **holding-period-matched base rate** (buy the
    same name for the same number of days, unconditionally) so the race is like-for-like.
    """
    closes, highs = panel["close"], panel["high"]
    if names is None:
        names = list(closes.columns)
    trades = []
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        h = highs[tk].reindex(closes[tk].dropna().index).to_numpy(dtype=float)
        sigs = detect_inverse_hs(c, **detect_kw)
        for d in sigs:
            entry_idx = d["breakout_idx"] + lag
            if entry_idx >= len(c):
                continue
            entry_px = c[entry_idx]
            window_end = min(entry_idx + max_days, len(c))
            hwin = h[entry_idx:window_end]
            hit_pos = np.where(hwin >= d["target"])[0]
            if len(hit_pos) > 0:
                exit_idx = entry_idx + int(hit_pos[0])
                hold_days = int(hit_pos[0])
                hit = True
            else:
                exit_idx = window_end - 1
                hold_days = exit_idx - entry_idx
                hit = False
            if exit_idx <= entry_idx or exit_idx >= len(c):
                continue
            exit_px = c[exit_idx]
            raw = exit_px / entry_px - 1.0
            # holding-period-matched base rate: unconditional mean return over `hold_days` days
            br = base_rate(c, max(hold_days, 1))
            trades.append({"raw": raw, "hold_days": hold_days, "hit": hit, "base": br})

    if not trades:
        return {"n": 0, "gross": float("nan"), "net": float("nan"), "t": float("nan"),
                "hac_t": float("nan"), "excess": float("nan"), "t_excess": float("nan"),
                "avg_hold": float("nan"), "hit_share": float("nan")}

    raw = np.array([t["raw"] for t in trades])
    base = np.array([t["base"] for t in trades])
    hold = np.array([t["hold_days"] for t in trades])
    hit = np.array([t["hit"] for t in trades])
    cost = 2.0 * cost_bps / 1e4
    net = raw - cost
    excess = raw - base
    return {
        "n": len(trades), "gross": float(raw.mean()), "net": float(net.mean()),
        "t": one_sample_t(net), "hac_t": hac_t(net),
        "excess": float(excess.mean()), "t_excess": hac_t(excess),
        "avg_hold": float(hold.mean()), "hit_share": float(hit.mean()),
    }
