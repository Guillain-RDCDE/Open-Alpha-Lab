"""The engine and its honest controls — Study 562 (Block-Trade-Signal).

The claim, at full strength (microstructure folklore): a **giant block print** on the tape is an
informed institution crossing size, and the stock *drifts* in the direction of that informed flow.
The tradable expression is the **coattail** trade — infer the block's side (buyer- vs
seller-initiated), take the same side, and hold for the drift. The skeptic's view (the
microstructure literature's own finding, Holthausen-Leftwich-Mayers; Keim-Madhavan) is that most
blocks are *liquidity* events negotiated at a temporary concession that **reverts**, so the
coattail either earns nothing or *loses* after costs. Either way the sign of the coattail P&L is
the whole question.

This module measures that, honestly, on the deterministic synthetic event panel from ``data.py``:

1. **The coattail signal.** Each block-print event carries an inferred side (``sign``) and a size
   (``block_z``); the coattail trade takes side ``sign`` and its forward P&L is the ``fwd_ret``
   column (already in the block's direction). The headline statistic is a **one-sample t** on the
   cross-event series of coattail returns — does riding blocks earn a mean drift different from
   zero?

2. **The size sort.** Fold the "giant blocks are *more* informed" claim into a two-sample test:
   split events into a *giant*-block tail and a *small*-block tail by ``block_z`` and compare their
   coattail returns (the folklore says giant > small).

3. **The placebo null.** Shuffle the inferred *signs* against forward returns many times; the
   p-value is the fraction of shuffles whose |mean coattail return| >= the observed. A real
   informed-flow signal sits in the tail; noise does not.

4. **Costs + borrow.** The coattail book is a round-trip per event (enter on the block, exit at the
   window end); each event pays a one-way ``cost_bps`` twice, and the *short* side (seller-initiated
   coattails) pays a borrow, pro-rated to the (short) holding window. Block-driven trades cross the
   spread into a temporarily dislocated price, so the cost is real, not a footnote.

5. **A robustness sweep.** Re-run the coattail one-sample t at several *size thresholds* (trade
   only blocks above a rising ``block_z`` cut) and check the sign / t are stable.

6. **A synthetic positive control.** A deterministic world where the drift is planted
   (``drift_alpha != 0``); the engine must recover the right sign / a significant t and stay flat
   at the null — averaged over >= 20 seeds so no single lucky RNG seed can manufacture it.

Execution convention (one documented lag, no look-ahead): the coattail is entered *after* the block
prints (the signal, ``sign`` and ``block_z``, is fully observed at the print) and held over the
forward window that ``fwd_ret`` already measures. No future data enters the signal — the block is
public the instant it crosses the tape. This is the single execution lag; there is no same-print
fill at the block's own (dislocated) price beyond the cost charged in ``net_stats``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(n)))


# ---------------------------------------------------------------------------
# The coattail return series and its headline stats
# ---------------------------------------------------------------------------
def coattail_returns(panel: pd.DataFrame, min_block_z: float = 0.0) -> pd.Series:
    """Per-event coattail P&L: the forward return of copying each block's inferred side.

    ``min_block_z`` optionally restricts to the *giant* blocks (only trade prints at least this
    large relative to normal trade size). Returns a ``pd.Series`` of event-level coattail returns
    (before costs).
    """
    if panel.empty:
        return pd.Series(dtype=float)
    df = panel[panel["block_z"] >= min_block_z]
    return df["fwd_ret"].astype(float).reset_index(drop=True).rename("coattail")


def coattail_stats(panel: pd.DataFrame, min_block_z: float = 0.0) -> dict:
    """Headline stats of the block-coattail trade.

    Returns the mean per-event coattail return, its annualised-ish scaling is left to the caller
    (events are irregular), the one-sample t on the event series, the event count, and the hit rate
    (share of events with positive coattail P&L). A *positive* mean = the folklore (blocks drift
    the right way); *negative* = the reversal/liquidity view (the coattail loses).
    """
    s = coattail_returns(panel, min_block_z=min_block_z)
    if s.empty:
        return {"mean": float("nan"), "t": float("nan"), "n": 0, "hit": float("nan")}
    arr = s.to_numpy(dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "t": _one_sample_t(arr),
        "n": int(np.isfinite(arr).sum()),
        "hit": float(np.mean(arr > 0)),
    }


# ---------------------------------------------------------------------------
# The size sort — "giant blocks are more informed"
# ---------------------------------------------------------------------------
def size_sort(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Split events by block size and compare coattail returns (giant vs small).

    ``frac`` is the tail fraction in each bucket. The folklore says the *giant* blocks carry more
    information, so their coattail should beat the small blocks'. Returns bucket means, the spread
    (giant - small), and a two-sample (Welch) t.
    """
    if panel.empty or len(panel) < 6:
        return {"giant_mean": float("nan"), "small_mean": float("nan"),
                "spread": float("nan"), "t": float("nan"), "n": int(len(panel))}
    df = panel.sort_values("block_z")
    k = max(1, int(round(len(df) * frac)))
    small = df.head(k)["fwd_ret"].to_numpy(dtype=float)
    giant = df.tail(k)["fwd_ret"].to_numpy(dtype=float)
    na, nb = len(giant), len(small)
    va, vb = giant.var(ddof=1), small.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (giant.mean() - small.mean()) / se if se > 0 else float("nan")
    return {
        "giant_mean": float(giant.mean()),
        "small_mean": float(small.mean()),
        "spread": float(giant.mean() - small.mean()),
        "t": float(t),
        "n": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Firm-level pooled relation — the sign IS the claim
# ---------------------------------------------------------------------------
def signed_size_slope(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the standardised *signed* block size across all events.

    ``signed_z = sign * block_z`` is the informed-flow signal; the coattail P&L is already in the
    block's direction, so regressing ``fwd_ret`` on the *absolute* standardised size ``|signed_z|``
    tests "bigger informed blocks drift more". Slope > 0 = the folklore (bigger block, more drift).
    Returns the slope, its t-stat, and the correlation.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "t": float("nan"), "corr": float("nan"), "n": 0}
    x = np.abs(panel["signed_z"].to_numpy(dtype=float))
    x = (x - x.mean()) / (x.std(ddof=1) or 1.0)
    y = panel["fwd_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / sign-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 562) -> float:
    """Sign-shuffle placebo p-value for the mean coattail return.

    The coattail P&L is ``sign``-directional; if the block's *side* carries no information, then
    randomly re-signing the events should reproduce the observed |mean| just as often. We shuffle
    the sign of each event's directional component against the (undirected) return magnitude and
    read the tail probability. A real informed-flow signal sits in the tail; noise does not.

    Concretely: the directional part of ``fwd_ret`` is what flips with ``sign``. We reconstruct an
    undirected return ``u = fwd_ret * sign`` (the P&L had the block gone the *observed* way, stripped
    of side), then re-attach random +/-1 signs and recompute the mean. Under the null (side carries
    no drift) this mean is centred at zero with the observed spread.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(float(np.nanmean(panel["fwd_ret"].to_numpy(dtype=float))))
    u = (panel["fwd_ret"].to_numpy(dtype=float) * panel["sign"].to_numpy(dtype=float))
    n = len(u)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=n)
        m = abs(float(np.mean(u * s)))
        if m >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_stats(panel: pd.DataFrame, min_block_z: float = 0.0, cost_bps: float = 8.0,
              borrow_ann_bps: float = 100.0, holding_days: float = 5.0) -> dict:
    """Gross and net mean coattail return after per-event round-trip costs + short borrow.

    Each coattail event is a round trip (enter on the block, exit at the window end), so it pays a
    one-way ``cost_bps`` on entry and exit (2 crossings). The seller-initiated (short) coattails
    pay an annual ``borrow_ann_bps`` pro-rated to the ``holding_days`` window. Block-driven entries
    cross into a temporarily dislocated price, so ``cost_bps`` is deliberately not tiny.
    Returns ``{"gross", "net", "cost", "borrow"}`` in per-event return units.
    """
    s = coattail_stats(panel, min_block_z=min_block_z)
    gross = s["mean"]
    df = panel[panel["block_z"] >= min_block_z]
    short_share = float((df["sign"] < 0).mean()) if len(df) else 0.0
    cost = 2.0 * cost_bps * 1e-4
    borrow = short_share * borrow_ann_bps * 1e-4 * (holding_days / TRADING_DAYS)
    net = gross - cost - borrow
    return {"gross": float(gross), "net": float(net), "cost": float(cost),
            "borrow": float(borrow)}


# ---------------------------------------------------------------------------
# Robustness — size-threshold sweep
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, thresholds=(0.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """Coattail stats across rising block-size thresholds; the sign/t should be stable.

    The folklore is *strongest* for the biggest blocks, so if the signal is real the t should hold
    (or strengthen) as we restrict to giant prints. If it wanders/flips, the "informed block"
    reading is noise.
    """
    rows = []
    for z in thresholds:
        st = coattail_stats(panel, min_block_z=z)
        rows.append({"min_block_z": z, "mean": st["mean"], "t": st["t"], "n": st["n"],
                     "hit": st["hit"]})
    return pd.DataFrame(rows).set_index("min_block_z")


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, drift_alpha: float, n_seeds: int = 25,
                     base_seed: int = 562) -> float:
    """Average the coattail one-sample t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean coattail-t across seeds for a
    planted ``drift_alpha``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(drift_alpha=drift_alpha, seed=s)
        ts.append(coattail_stats(panel)["t"])
    return float(np.nanmean(ts))
