"""Rising Wedge as a falsifiable mechanical rule — Study 462.

A **rising wedge** is two converging *up-sloping* trendlines:

* **support** — a rising line through recent swing *lows*;
* **resistance** — a rising line through recent swing *highs*;

with the channel **narrowing** (support climbing faster than resistance, so the two lines
converge toward an apex). Textbook technical analysis (Edwards & Magee, Bulkowski, every
chart-pattern site) calls it a **bearish** pattern: price is "supposed" to resolve to the
**downside**, breaking *down* through the rising support line. The folklore rule:

    Short on the first close below the rising support line. Enter at the next close.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower (higher)
   bars on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the
   confirmation lag), so the wedge is never fit with future bars.
2. **Rolling wedge** — at each bar we fit a support line through the last ``m`` confirmed
   swing *lows* and a resistance line through the last ``m`` confirmed swing *highs*. The
   pattern qualifies as a **rising wedge** iff both slopes are positive (rising), the
   support slope exceeds the resistance slope (converging), and the lines have not yet
   crossed (an apex still ahead).
3. **Support break-down** — a **short** entry fires when the close pierces *below* the
   rising support line, while a qualified rising wedge is active. Entry is at the **next**
   close (one documented lag); the forward H-day return is the **short** return
   (``-(P_{t+H}/P_t - 1)``), so a price fall is a positive return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold,
   same short sign) that captures the tape's drift, and (b) a **slope-scramble placebo**
   that keeps the break-detection cadence but randomizes the support line's slope/intercept
   (destroying the wedge geometry while keeping the marginal) — the honest "is the wedge's
   geometry doing anything?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the break is read on the close of
*t*, the short is entered at the close of *t+1*. A short's forward return is the negative of
the underlying forward move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed fractals)
# --------------------------------------------------------------------------- #
def find_pivots(close: pd.Series, k: int = 10) -> pd.DataFrame:
    """Confirmed swing pivots: local extrema with ``k`` strictly-beaten bars on each side.

    Returns a DataFrame indexed by the pivot bar with columns ``price`` and ``kind``
    (+1 = swing high, -1 = swing low). A pivot at position ``i`` is only *confirmed* (knowable)
    at bar ``i + k`` — :func:`build_wedges` enforces that lag, so no future data leaks in.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    rows = []
    for i in range(k, n - k):
        win = p[i - k:i + k + 1]
        c = p[i]
        if c == win.max() and (win[:k] < c).all() and (win[k + 1:] < c).all():
            rows.append((i, c, +1))
        elif c == win.min() and (win[:k] > c).all() and (win[k + 1:] > c).all():
            rows.append((i, c, -1))
    if not rows:
        return pd.DataFrame(columns=["pos", "price", "kind"]).set_index("pos")
    df = pd.DataFrame(rows, columns=["pos", "price", "kind"]).set_index("pos")
    return df


# --------------------------------------------------------------------------- #
# Wedge geometry
# --------------------------------------------------------------------------- #
def _line_through(pts):
    """Least-squares line ``price = slope*pos + intercept`` through (pos, price) points."""
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if xs.size < 2 or xs.std() == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def wedge_at(lows, highs, t):
    """Fit a rising-wedge support line at bar ``t`` from confirmed lows/highs available by ``t``.

    ``lows`` and ``highs`` are lists of (pos, price) confirmed pivots. We fit a support line
    through the last ``m`` lows and a resistance line through the last ``m`` highs and qualify
    a **rising wedge**: both slopes positive (rising), support slope > resistance slope
    (converging), lines not yet crossed at ``t`` (apex ahead). Returns ``(support_value_at_t,
    sup_slope, sup_b)`` if a rising wedge is active, else ``None``.
    """
    m = 3
    lo = [pt for pt in lows if pt[0] <= t][-m:]
    hi = [pt for pt in highs if pt[0] <= t][-m:]
    if len(lo) < 2 or len(hi) < 2:
        return None
    sup = _line_through(lo)
    res = _line_through(hi)
    if sup is None or res is None:
        return None
    s_sup, b_sup = sup
    s_res, b_res = res
    # rising wedge: both lines rising, support rising FASTER (converging), apex still ahead
    if not (s_sup > 0 and s_res > 0 and s_sup > s_res):
        return None
    sup_t = s_sup * t + b_sup
    res_t = s_res * t + b_res
    if sup_t >= res_t:                 # lines already crossed -> wedge complete, skip
        return None
    return sup_t, s_sup, b_sup


def build_wedges(close: pd.Series, k: int = 10):
    """For each bar, the *currently usable* rising-wedge support line (NaN if none active).

    Returns two aligned Series over ``close.index``: the support-line value and an active-flag
    (1.0 when a qualified rising wedge is present, NaN otherwise). A pivot at bar ``i`` becomes
    usable at ``i + k`` (confirmation lag) — so the wedge at bar ``t`` only uses pivots
    confirmed by ``t``.
    """
    piv = find_pivots(close, k=k)
    n = len(close)
    sup = np.full(n, np.nan)
    active = np.full(n, np.nan)
    if not piv.empty:
        lows, highs = [], []
        # (confirm_bar, pos, price) per pivot
        conf_lo = [(int(pos) + k, int(pos), float(pr))
                   for pos, pr, kd in zip(piv.index, piv["price"], piv["kind"]) if kd < 0]
        conf_hi = [(int(pos) + k, int(pos), float(pr))
                   for pos, pr, kd in zip(piv.index, piv["price"], piv["kind"]) if kd > 0]
        for t in range(n):
            lows = [(p, pr) for cb, p, pr in conf_lo if cb <= t]
            highs = [(p, pr) for cb, p, pr in conf_hi if cb <= t]
            w = wedge_at(lows, highs, t)
            if w is None:
                continue
            sup_t, _s, _b = w
            sup[t] = sup_t
            active[t] = 1.0
    idx = close.index
    return pd.Series(sup, index=idx), pd.Series(active, index=idx)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def wedge_break_entries(close: pd.Series, k: int = 10) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the rising support of an active rising wedge — SHORT.

    Only the *first* bar of each consecutive break run is kept (the break, not every day price
    stays below support). Entry is executed at the next close by :func:`forward_returns`.
    """
    sup, active = build_wedges(close, k=k)
    mask = (close < sup) & sup.notna() & active.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine  (SHORT: a price fall is a positive return)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    short: bool = True) -> np.ndarray:
    """Forward ``horizon``-day **short** return for each entry, entered at the *next* close.

    The rising-wedge rule is bearish, so we take a SHORT: the trade return is
    ``-(P_{e+H}/P_e - 1)`` (a price fall is a gain). One documented lag (enter at ``e=i+1``).
    ``cost_bps`` is a one-way cost charged twice (in + out). Trades whose window overruns the
    tape are dropped. Set ``short=False`` to recover the raw long forward return.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    sgn = -1.0 if short else 1.0
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = sgn * (p[e + horizon] / p[e] - 1.0)
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def slope_scramble_placebo(close: pd.Series, horizon: int, k: int = 10,
                           n_draws: int = 1000, seed: int = 462) -> dict:
    """Placebo: keep the break cadence but randomize the support line's slope & intercept.

    For each draw we replace the fitted support line with a random-but-plausible line (slope
    and level drawn from the empirical distribution of the real rising-wedge supports), detect
    "breaks below" against THAT nonsense line, and take the short forward return. This destroys
    the wedge geometry while preserving the price marginal and the rough break frequency.
    Returns the share of placebo runs whose mean break-down short return **beats** the real one
    — the honest "is the wedge's geometry adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, wedge_break_entries(close, k=k), horizon)))
    sup, active = build_wedges(close, k=k)
    valid_mask = sup.notna()
    if valid_mask.sum() < 10:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    cl = close.to_numpy(dtype=float)
    idx = close.index
    sup_vals = sup[valid_mask].to_numpy()
    cl_at_sup = close[valid_mask].to_numpy()
    # empirical gap of support below price (in fractional terms) to build plausible nonsense lines
    gaps = (cl_at_sup - sup_vals) / cl_at_sup
    gap_mu, gap_sd = float(np.nanmean(gaps)), float(np.nanstd(gaps) + 1e-9)
    warm = valid_mask.to_numpy()
    rng = np.random.default_rng(seed)
    n = cl.size
    beats = 0
    valid = 0
    for _ in range(n_draws):
        # a random support level under each bar's price (same marginal gap), only where a wedge
        # was active so the cadence matches; this scrambles WHICH geometry, not the activity.
        rand_gap = rng.normal(gap_mu, gap_sd, n)
        fake_sup = cl * (1.0 - rand_gap)
        fake_sup = np.where(warm, fake_sup, np.nan)
        fs = pd.Series(fake_sup, index=idx)
        mask = (close < fs) & fs.notna()
        first = mask & ~mask.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(close: pd.Series, k: int = 10, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: wedge-break short vs random-entry baseline, all H.

    Returns a dict keyed by horizon with the break-down short summary (gross + net), the
    drift-matched random-entry baseline (also shorted), and the break-minus-random delta.
    """
    ent = wedge_break_entries(close, k=k)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), k=k, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
