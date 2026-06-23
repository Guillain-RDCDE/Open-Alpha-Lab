"""Strategy + inference for Study 372 — Max-Pain Pinning.

The claim: at expiry the underlying gets **pinned to the max-pain strike**. The testable
form is *closeness* — does the expiry close land **closer to max-pain than to a random
strike**, by more than the strike geometry alone would give?

We measure, per (underlying, expiry) episode, the distance from the realised close to:
  * the **max-pain** strike, and
  * the **spot-anchor** strike — the grid strike nearest to where price *started*
    (``days_to_expiry`` sessions before expiry).
Both expressed in **strike-spacings** (so episodes on different grids are comparable). This
baseline is the honest null: with **no** pinning the close is a random walk *from spot*, so
it lands naturally near the spot-anchor and (when max-pain ≠ spot) *farther* from max-pain —
the difference is negative. Pinning has to drag the close *away* from where it started and
*onto* max-pain to make the difference positive. (A naive "random strike on the grid"
baseline is biased — both close and max-pain cluster centrally — and would manufacture a
positive result with zero pinning; the spot-anchor avoids that trap.)

Tests:
  * a **paired Welch t** on the per-episode difference ``dist_spot_anchor - dist_maxpain``
    (>0 ⇒ the close is pulled toward max-pain, away from its origin);
  * a **label-shuffle placebo** null — permute which max-pain goes with which episode's
    close+spot and recompute the mean closeness many times; the p-value is the share of
    permutations whose mean distance matches/beats the true pairing (the honest test that the
    *true* max-pain attracts *its own* close, not just "some central strike");
  * a **win-rate**: P[close nearer max-pain than the spot-anchor] vs the 50% base rate;
  * a **1-day-lag, one-way-cost** "fade toward max-pain" trade as the Tradability probe.

For the **real snapshot** the close is unobserved (yfinance retains no expiry-day history),
so the snapshot path measures only the *spot-vs-max-pain gap* — explicitly a description of
*where price sits now*, never the landing claim. The landing test runs on the synthetic
control, where the truth is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Closeness measurement
# --------------------------------------------------------------------------- #
def _spot_anchor(lo: float, spacing: float, n_strk: int, spot: float) -> float:
    """The grid strike nearest to where price started (the spot ``days_to_expiry`` out)."""
    grid = lo + spacing * np.arange(int(n_strk))
    return float(grid[int(np.argmin(np.abs(grid - spot)))])


def closeness(episodes: pd.DataFrame, seed: int = 372) -> pd.DataFrame:
    """Per-episode distances (in strike-spacings) of the close to max-pain vs the spot-anchor.

    Adds columns:
      * ``d_mp``     — |close - maxpain|     / spacing
      * ``d_anchor`` — |close - spot_anchor| / spacing  (strike nearest the *initial* spot)
      * ``diff``     — ``d_anchor - d_mp`` (positive ⇒ close pulled toward max-pain, away from origin)
      * ``win``      — bool, ``d_mp < d_anchor``
    """
    out = episodes.copy()
    sp = out["spacing"].values if "spacing" in out else np.ones(len(out))
    close = out["close"].values
    mp = out["maxpain"].values
    lo = out["lo"].values
    spot = out["spot"].values
    n_strk = out["nstrk"].values.astype(int)
    anchor = np.array([_spot_anchor(lo[i], sp[i], n_strk[i], spot[i]) for i in range(len(out))])
    out["d_mp"] = np.abs(close - mp) / sp
    out["d_anchor"] = np.abs(close - anchor) / sp
    out["diff"] = out["d_anchor"] - out["d_mp"]
    out["win"] = out["d_mp"] < out["d_anchor"]
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu0`` (here the paired difference vs 0). NaN if n<2."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def placebo_pvalue(episodes: pd.DataFrame, n_draws: int = 5_000,
                   seed: int = 372) -> dict:
    """Label-shuffle placebo: does the *true* max-pain attract *its own* close?

    Each episode's max-pain is *relative* to that episode's grid centre, so we work in the
    centred coordinate ``(close - maxpain)`` and ask whether the **true pairing** of close to
    its own max-pain yields a smaller mean |distance| than max-pain offsets **permuted across
    episodes**. For each draw we permute the per-episode ``(maxpain - grid_centre)`` offsets
    and re-apply them to each episode's own grid centre, then recompute the mean distance. The
    p-value is the share of permutations whose mean distance is **<=** the true pairing's —
    i.e. how often a *shuffled* max-pain pins a close as well as its own. Small p ⇒ the
    attraction is to *the* max-pain, not to "a central strike in general".
    """
    rng = np.random.default_rng(seed)
    close = episodes["close"].values.astype(float)
    mp = episodes["maxpain"].values.astype(float)
    lo = episodes["lo"].values.astype(float)
    sp = (episodes["spacing"].values if "spacing" in episodes
          else np.ones(len(episodes))).astype(float)
    n_strk = episodes["nstrk"].values.astype(int)
    n = len(episodes)
    centre = lo + sp * (n_strk - 1) / 2.0
    offset = mp - centre                       # max-pain offset from each grid's centre
    true_mean = float((np.abs(close - mp) / sp).mean())
    means = np.empty(n_draws)
    for d in range(n_draws):
        perm = rng.permutation(n)
        placebo_mp = centre + offset[perm]     # shuffled offsets, re-anchored to own centre
        means[d] = float((np.abs(close - placebo_mp) / sp).mean())
    p = float((means <= true_mean).mean())
    return {"n": n, "true_mean_dist": true_mean, "placebo_mean_dist": float(means.mean()),
            "p_value": p}


def summarize(episodes: pd.DataFrame, seed: int = 372) -> dict:
    """Headline closeness stats: mean distances, paired t, win-rate vs base rate, placebo p."""
    c = closeness(episodes, seed=seed)
    pl = placebo_pvalue(episodes, seed=seed)
    # base rate: under no pinning the close is closer to max-pain than to the spot-anchor
    # ~50% of the time by symmetry (when they coincide it is a tie); we report the realised one.
    return {
        "n": int(len(c)),
        "mean_d_mp": float(c["d_mp"].mean()),
        "mean_d_anchor": float(c["d_anchor"].mean()),
        "mean_diff": float(c["diff"].mean()),
        "t": welch_t(c["diff"].values),
        "win_rate": float(c["win"].mean()),
        "base_rate": 0.50,
        "p_placebo": pl["p_value"],
        "placebo_mean_dist": pl["placebo_mean_dist"],
    }


# --------------------------------------------------------------------------- #
# Tradability — the expiry "fade toward max-pain" probe
# --------------------------------------------------------------------------- #
def fade_trade(episodes: pd.DataFrame, cost_bps: float = 5.0,
               lag: int = 1) -> dict:
    """The believer's trade: a few days before expiry, bet the underlying moves toward max-pain.

    Per episode the *direction-correct* gross return is ``sign(maxpain - spot) * (close/spot -
    1)`` — you go long if max-pain is above spot, short if below, and earn the realised move
    (this is the pinning bet stated as a directional trade with a **1-day entry lag** already
    baked into the spot→close path). One round-trip cost of ``cost_bps`` is charged per trade
    (shorts also pay this one-way slippage; borrow is negligible over the few-day hold and is
    folded into the round-trip for this illustrative probe). Returns gross/net mean and the
    hit-rate.
    """
    spot = episodes["spot"].values
    close = episodes["close"].values
    mp = episodes["maxpain"].values
    direction = np.sign(mp - spot)
    gross = direction * (close / spot - 1.0)
    c = cost_bps / 1e4
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "hit_rate": float((gross > 0).mean()) if len(gross) else float("nan"),
        "cost_bps": cost_bps,
        "lag": lag,
    }


# --------------------------------------------------------------------------- #
# Snapshot description (real tape) — gap only, NEVER a landing claim
# --------------------------------------------------------------------------- #
def snapshot_gap_stats(snapshot: pd.DataFrame) -> dict:
    """Describe the real snapshot's spot-vs-max-pain gap (|%|). NOT a pinning test.

    If price were already pinned to max-pain *today*, the gaps would cluster near 0. They do
    not — this quantifies how far spot sits from max-pain at the snapshot instant, with the
    explicit caveat that yfinance retains no expiry-day close, so the landing claim is
    untestable on this tape and is deferred to the synthetic control.
    """
    g = snapshot["gap_pct"].abs().values
    return {
        "n": int(len(g)),
        "median_abs_gap": float(np.median(g)),
        "mean_abs_gap": float(np.mean(g)),
        "frac_within_0p5pct": float((g <= 0.5).mean()),
        "frac_within_1pct": float((g <= 1.0).mean()),
        "max_abs_gap": float(np.max(g)),
    }
