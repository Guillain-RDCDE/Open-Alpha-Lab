"""The engine and its honest controls — Study 574 (Penny-Beat).

Two questions, one module:

1. **Is there a discontinuity?** Does the surprise histogram show an *excess* of firm-quarters at
   exactly +$0.01 relative to a smooth counterfactual — the earnings-management fingerprint
   (Burgstahler & Dichev 1997; Degeorge-Patel-Zeckhauser 1999)? We measure it with a
   **discontinuity statistic**: the +1c bin's count minus the average of its two neighbours
   (the 0c and +2c bins), standardised by the Poisson-ish standard error of that local smooth,
   giving a z-score. A large positive z = a spike that a smooth distribution would not produce.

2. **Do penny-beaters underperform?** Split the population into *penny-beaters* (reported surprise
   exactly +1c) and *decisive beaters* (surprise ≥ +3c). Compare their forward returns with a
   two-sample (Welch) *t*; the claim predicts penny-beaters earn **less** (a negative
   penny-minus-decisive spread). We add a **label-shuffle placebo** null and costs.

Because the surprise histogram is the *observable* and "managed" is *latent*, the return test can
only condition on the +1c *bin* — exactly as a real analyst must. The synthetic panel plants the
truth so the positive control can confirm the engine recovers a real penalty and stays flat at the
null (seed-robust, ≥ 20 seeds).

Execution convention: the surprise (consensus vs actual) is known at the earnings release; the
position is entered at that release close and held over the forward window — one documented
execution lag, no look-ahead. The same convention runs the sweep and the synthetic control.

Data honesty: this is a **synthetic-only** study (no free consensus-vs-actual tape exists for a
no-key stack), so no cell here ever claims a real-tape number. The ceiling is `WEAK`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# ---------------------------------------------------------------------------
# 1. The discontinuity statistic
# ---------------------------------------------------------------------------
def discontinuity_z(panel: pd.DataFrame) -> dict:
    """Excess mass at the +$0.01 bin vs a local smooth of its neighbours, as a z-score.

    Let ``c(+1)`` be the count in the penny-beat bin and ``c(0)``, ``c(+2)`` its immediate
    neighbours. Under a *smooth* distribution the +1c count should sit near the average of its
    neighbours. The excess is ``c(+1) - (c(0)+c(+2))/2``; its standard error under a Poisson
    (count) model is ``sqrt(c(+1) + (c(0)+c(+2))/4)``. The ratio is a z-score: large positive =
    an earnings-management spike a smooth histogram would not produce.

    Also reports the "just-below-zero deficit": counts in the small-miss bins (−1..−3c) relative
    to a mirror of the small-beat bins (+2..+4c) — the other half of the fingerprint.

    Returns ``{c_penny, neighbour_avg, excess, z, deficit_ratio, n}``.
    """
    if panel.empty:
        return {"c_penny": 0, "neighbour_avg": 0.0, "excess": 0.0, "z": float("nan"),
                "deficit_ratio": float("nan"), "n": 0}
    h = _data.surprise_histogram(panel)
    c_penny = float(h.get(_data.PENNY_CENTS, 0))
    c_lo = float(h.get(0, 0))
    c_hi = float(h.get(2, 0))
    neigh = 0.5 * (c_lo + c_hi)
    excess = c_penny - neigh
    var = c_penny + 0.25 * (c_lo + c_hi)
    z = excess / np.sqrt(var) if var > 0 else float("nan")

    # just-below-zero deficit: small misses (−1..−3) vs small beats (+2..+4), a smooth mirror
    miss = sum(float(h.get(k, 0)) for k in (-1, -2, -3))
    beat = sum(float(h.get(k, 0)) for k in (2, 3, 4))
    deficit_ratio = (miss / beat) if beat > 0 else float("nan")
    return {
        "c_penny": c_penny,
        "neighbour_avg": neigh,
        "excess": excess,
        "z": float(z),
        "deficit_ratio": float(deficit_ratio),
        "n": int(len(panel)),
    }


def discontinuity_placebo_p(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 574) -> float:
    """Placebo p-value for the +1c spike: how often does a *smoothed* relabelling match it?

    We destroy any true spike by re-drawing each firm-quarter's bin from the *pooled* local shape
    (the counts in −3..+5 spread over their bins uniformly), then recompute the discontinuity z.
    The p-value is the fraction of relabellings whose z ≥ the observed z. A genuine spike sits in
    the tail; a histogram that only *looks* bumpy by chance does not.
    """
    if panel.empty or len(panel) < 30:
        return float("nan")
    obs = discontinuity_z(panel)["z"]
    if not np.isfinite(obs):
        return float("nan")
    h = _data.surprise_histogram(panel)
    # local window whose total mass we redistribute smoothly across its bins
    win = list(range(-3, 6))
    total = int(sum(int(h.get(k, 0)) for k in win))
    if total <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    bins = np.array(win)
    count = 0
    for _ in range(n_perm):
        draw = rng.integers(0, len(bins), size=total)  # uniform over the local bins = smooth
        c = np.bincount(draw, minlength=len(bins))
        cp = float(c[bins.tolist().index(_data.PENNY_CENTS)])
        c0 = float(c[bins.tolist().index(0)])
        c2 = float(c[bins.tolist().index(2)])
        neigh = 0.5 * (c0 + c2)
        var = cp + 0.25 * (c0 + c2)
        z = (cp - neigh) / np.sqrt(var) if var > 0 else 0.0
        if z >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# 2. The return test — penny-beaters vs decisive beaters
# ---------------------------------------------------------------------------
def penny_vs_decisive(panel: pd.DataFrame, decisive_min: int = 3) -> dict:
    """Forward-return means for penny-beaters (surprise == +1c) vs decisive beaters (>= +3c).

    The trading claim: penny-beaters, having managed the number, underperform the honest beaters.
    Returns bucket means, the spread (penny − decisive; the claim predicts negative), sizes and the
    per-name forward returns (for the two-sample t).
    """
    if panel.empty:
        return {}
    penny = panel.loc[panel["surprise"] == _data.PENNY_CENTS, "forward_ret"]
    decisive = panel.loc[panel["surprise"] >= decisive_min, "forward_ret"]
    return {
        "n_penny": int(len(penny)),
        "n_decisive": int(len(decisive)),
        "penny_mean": float(penny.mean()) if len(penny) else float("nan"),
        "decisive_mean": float(decisive.mean()) if len(decisive) else float("nan"),
        "spread": float(penny.mean() - decisive.mean()) if len(penny) and len(decisive)
        else float("nan"),
        "penny_ret": penny,
        "decisive_ret": decisive,
    }


def spread_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on penny − decisive forward returns. Negative t = the claim."""
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["penny_ret"].to_numpy(dtype=float)
    b = buckets["decisive_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


def within_bin_penalty(panel: pd.DataFrame) -> dict:
    """The *clean* management penalty: managed vs honest firms **inside the +1c bin**.

    The headline penny − decisive spread confounds two effects: (a) a mechanical **PEAD
    composition** gap (the +1c bin is a smaller honest surprise than a ≥+3c decisive beat, so it
    drifts less even absent any management), and (b) the true **management penalty**. This helper
    isolates (b) by comparing the *managed* and *honest* firms that both landed in the +1c bin —
    holding the reported surprise fixed. It relies on the latent ``managed`` truth column, so it is
    a **synthetic-control-only** diagnostic (a real analyst cannot see who managed the number);
    it is never applied to a real tape (there is none) and is used to show the engine measures the
    planted penalty cleanly once composition is netted out.

    Returns ``{managed_mean, honest_mean, penalty, t, n_managed, n_honest}``.
    """
    if panel.empty or "managed" not in panel.columns:
        return {"penalty": float("nan"), "t": float("nan"), "n_managed": 0, "n_honest": 0}
    p1 = panel.loc[panel["surprise"] == _data.PENNY_CENTS]
    man = p1.loc[p1["managed"] == 1, "forward_ret"].to_numpy(dtype=float)
    hon = p1.loc[p1["managed"] == 0, "forward_ret"].to_numpy(dtype=float)
    if len(man) < 2 or len(hon) < 2:
        return {"managed_mean": float("nan"), "honest_mean": float("nan"),
                "penalty": float("nan"), "t": float("nan"),
                "n_managed": len(man), "n_honest": len(hon)}
    se = np.sqrt(man.var(ddof=1) / len(man) + hon.var(ddof=1) / len(hon))
    t = (man.mean() - hon.mean()) / se if se > 0 else float("nan")
    return {"managed_mean": float(man.mean()), "honest_mean": float(hon.mean()),
            "penalty": float(man.mean() - hon.mean()), "t": float(t),
            "n_managed": int(len(man)), "n_honest": int(len(hon))}


def return_placebo_p(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 574,
                     decisive_min: int = 3) -> float:
    """Label-shuffle placebo for the penny − decisive spread.

    Pool penny-beaters and decisive beaters, shuffle the *bucket labels* against forward returns
    ``n_perm`` times, and read the fraction of shuffles whose spread ≤ the observed (signed) spread
    — a one-sided tail for the claim's *negative* prediction.
    """
    b = penny_vs_decisive(panel, decisive_min)
    if not b or b["n_penny"] < 2 or b["n_decisive"] < 2:
        return float("nan")
    obs = b["spread"]
    pooled = np.concatenate([b["penny_ret"].to_numpy(float), b["decisive_ret"].to_numpy(float)])
    k = b["n_penny"]
    rng = np.random.default_rng(seed)
    n = len(pooled)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        spread = pooled[idx[:k]].mean() - pooled[idx[k:]].mean()
        if spread <= obs + 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs — shorting the penny-beaters, long the decisive beaters
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 100.0,
               holding_years: float = 0.25) -> float:
    """Penny − decisive spread net of costs and a short borrow on the penny-beat (short) leg.

    The tradable expression is **short penny-beaters, long decisive beaters** (harvest the negative
    spread). The book is built once per quarter and held ``holding_years``: a round-trip one-way
    cost on each leg (entry + exit, both legs = 4 crossings) plus a pro-rated short borrow on the
    penny leg. Because we would be *short* the penny-beaters, a negative gross spread is the *win*,
    so costs push the realised (shortable) magnitude toward zero.
    """
    if not buckets or not np.isfinite(buckets.get("spread", float("nan"))):
        return float("nan")
    gross = buckets["spread"]                      # penny - decisive (negative = claim)
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    # the harvestable magnitude of a short-penny/long-decisive book, after frictions
    harvest = -gross - cost - borrow
    return float(harvest)


# ---------------------------------------------------------------------------
# Robustness sweep — does the return penalty survive across spike strengths?
# ---------------------------------------------------------------------------
def robustness_sweep(spikes=(0.2, 0.4, 0.55, 0.7), penny_penalty: float = -0.05,
                     seed: int = 574) -> pd.DataFrame:
    """Re-run the joint engine across earnings-management intensities (spike strengths).

    For each ``spike`` (holding the planted ``penny_penalty`` fixed) report the discontinuity z and
    the penny − decisive spread + t. The claim is *stable* if the penalty holds its sign as the
    spike grows.
    """
    rows = []
    for sp in spikes:
        panel, _ = _data.synthetic_panel(spike=sp, penny_penalty=penny_penalty, seed=seed)
        dz = discontinuity_z(panel)["z"]
        b = penny_vs_decisive(panel)
        t = spread_tstat(b)
        rows.append({"spike": sp, "disc_z": dz, "spread": b["spread"], "t": t["t"],
                     "n_penny": b["n_penny"]})
    return pd.DataFrame(rows).set_index("spike")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule)
# ---------------------------------------------------------------------------
def synthetic_control(penny_penalty: float, spike: float = 0.55, n_seeds: int = 25,
                      base_seed: int = 574) -> dict:
    """Average the discontinuity z and the penny−decisive t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages its stat over >= 20 seeds so no lucky RNG
    seed can manufacture significance. Returns mean discontinuity z and mean spread-t across seeds
    for a planted ``(spike, penny_penalty)``.
    """
    zs, ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = _data.synthetic_panel(spike=spike, penny_penalty=penny_penalty, seed=s)
        zs.append(discontinuity_z(panel)["z"])
        ts.append(spread_tstat(penny_vs_decisive(panel))["t"])
    return {"mean_disc_z": float(np.nanmean(zs)), "mean_spread_t": float(np.nanmean(ts)),
            "n_seeds": n_seeds}
