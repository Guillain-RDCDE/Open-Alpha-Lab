"""The engine and its honest controls — Study 557 (Borrow-Fee-Signal).

The claim, at full strength (Cohen, Diether & Malloy 2007; D'Avolio 2002; Drechsler & Drechsler
2016 on the borrow-fee premium): the securities-lending **fee** — the annualised cost to borrow a
stock — is the market-clearing price of shorting demand against lendable supply. Names that are
*special* / hard-to-borrow (high fee) attract informed short sellers whose demand drives the fee,
and those names go on to earn **negative** forward returns. And the fee is claimed to carry signal
*beyond* raw short interest: it prices the supply side that short-% of float cannot see.

This module tests that, honestly, on a synthetic cross-section (the free real fee tape does not
exist — see ``data.py``):

1. **The signal.** The standardized borrow fee across the cross-section — higher = more expensive
   / harder to borrow. (We also carry standardized short interest to test incrementality.)

2. **The quintile sort.** Rank by fee; form a *cheap-to-borrow* bucket (low fee) and an
   *expensive/hard-to-borrow* bucket (high fee), and compare forward returns. The claim predicts
   cheap > expensive (a positive long-cheap / short-expensive spread).

3. **The long-short.** The claim's tradable expression is **long cheap, short expensive-to-borrow**.
   We report the spread, a two-sample (Welch) *t*, a **label-shuffle placebo** null, and — the
   honest twist unique to *this* signal — the cost of the trade is the borrow fee **itself**: the
   short leg you want is literally the expensive-to-borrow tail, so the fee is both the alpha and
   the bill.

4. **The firm-level relation.** OLS of forward return on the standardized fee across all names —
   the sign of the slope *is* the claim (negative = expensive-to-borrow names underperform). Plus
   an **incremental** OLS on both fee and short interest, to test whether the fee survives
   controlling for raw short interest (the dedup-from-262 argument).

5. **A synthetic positive control.** A deterministic world where the anomaly is planted
   (``fee_alpha < 0``); the engine must recover a negative slope / positive long-short and stay
   flat at the null — averaged over >= 20 seeds so no lucky seed manufactures it. A second control
   plants signal in *short interest instead* (``si_alpha < 0``, ``fee_alpha = 0``) and shows the
   fee-sort stays quiet — proving the fee-sort reads the fee, not SI.

Execution convention: the fee and short interest are known *as-of the formation date* (public that
day); the position is entered at that close and held over the *forward* window measured strictly
after formation. No future data enters the signal (no look-ahead). The one execution lag — form on
the snapshot, enter at that close — is the single documented convention; a one-bar-later entry on a
multi-month hold moves the headline spread trivially and changes no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 557


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def fee_signal(panel: pd.DataFrame) -> pd.Series:
    """Standardized borrow fee across the cross-section; higher = more expensive to borrow."""
    if panel.empty:
        return pd.Series(dtype=float)
    return _z(panel["borrow_fee"]).rename("fee_z")


# ---------------------------------------------------------------------------
# The sort and the long-short
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.2) -> dict:
    """Sort by borrow fee, return cheap / expensive bucket forward-return means and the spread.

    ``frac`` is the tail fraction in each bucket (0.2 ≈ quintiles). The claim predicts the
    cheap-to-borrow names beat the expensive/hard-to-borrow ones, i.e. spread (cheap − expensive)
    > 0. Returns bucket means, the spread, sizes, per-name returns, and the mean fee of the
    expensive (short) leg (the borrow bill you'd pay).
    """
    if panel.empty:
        return {}
    d = fee_signal(panel)
    df = panel.assign(fee_z=d).sort_values("fee_z")
    k = max(1, int(round(len(df) * frac)))
    cheap = df.head(k)            # lowest fee (easy to borrow)
    expensive = df.tail(k)        # highest fee (hard to borrow — the short leg)
    cheap_ret = cheap["forward_ret"]
    exp_ret = expensive["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "cheap_mean": float(cheap_ret.mean()),
        "expensive_mean": float(exp_ret.mean()),
        "spread": float(cheap_ret.mean() - exp_ret.mean()),
        "cheap_ret": cheap_ret,
        "expensive_ret": exp_ret,
        "short_leg_fee": float(expensive["borrow_fee"].mean()),  # annualised %, the borrow bill
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on cheap − expensive forward returns.

    The claim's spread is a cross-sectional difference in means, so the honest test is a Welch
    two-sample t between the cheap and expensive buckets' forward returns. Returns the spread, the
    t-stat and n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["cheap_ret"].to_numpy(dtype=float)
    b = buckets["expensive_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The firm-level relation — the sign IS the claim
# ---------------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their standard errors (homoskedastic)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return coef, se


def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the standardized borrow fee across all names.

    Slope < 0 = the claim (more expensive to borrow → earns less). Returns the slope, its t-stat,
    and the Pearson correlation between fee and forward return.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    d = fee_signal(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(d), d])
    coef, se = _ols(X, y)
    se_slope = float(se[1])
    corr = float(np.corrcoef(d, y)[0, 1]) if d.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(len(y)),
    }


def incremental_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on BOTH standardized fee and standardized short interest.

    Tests whether the borrow fee carries signal *beyond* raw short interest (the dedup-from-262
    argument). Returns each slope and its t-stat. If the fee slope-t stays significant while the
    SI slope-t collapses, the fee has *incremental* information over short % of float.
    """
    if panel.empty or len(panel) < 5 or "short_interest" not in panel.columns:
        return {
            "fee_slope": float("nan"), "fee_t": float("nan"),
            "si_slope": float("nan"), "si_t": float("nan"), "n": 0,
        }
    fee = _z(panel["borrow_fee"]).to_numpy(dtype=float)
    si = _z(panel["short_interest"]).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(fee), fee, si])
    coef, se = _ols(X, y)
    return {
        "fee_slope": float(coef[1]),
        "fee_t": float(coef[1] / se[1]) if se[1] > 0 else float("nan"),
        "si_slope": float(coef[2]),
        "si_t": float(coef[2] / se[2]) if se[2] > 0 else float("nan"),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, frac: float = 0.2,
                   seed: int = SEED) -> float:
    """Label-shuffle placebo p-value for the long-short spread.

    Shuffle the fee *labels* against forward returns ``n_perm`` times; the p-value is the fraction
    of shuffles whose |spread| ≥ the observed |spread|. A real cross-sectional signal sits in the
    tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel, frac)["spread"])
    rng = np.random.default_rng(seed)
    d = fee_signal(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * frac)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(d[perm])           # sort returns by shuffled fee
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs — the borrow fee IS the cost (the signal's own trap)
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, holding_years: float = 0.25) -> float:
    """Long-cheap / short-expensive spread net of trading costs AND the actual borrow fee.

    The honest twist unique to this signal: the short leg is the *expensive-to-borrow* tail, so the
    borrow you pay is the observed ``short_leg_fee`` (annualised, pro-rated over the hold) — not a
    flat 100 bps placeholder. Trading cost is a round-trip one-way on each leg (entry + exit, two
    legs → 4 crossings). The alpha and the bill come from the *same* variable.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4
    borrow = (buckets.get("short_leg_fee", 0.0) / 100.0) * holding_years  # fee is in %, hold-prorated
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — bucket-fraction sweep
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, fracs=(0.10, 0.20, 0.30, 0.40)) -> list:
    """Re-run the sort at several tail fractions; report (frac, spread, welch_t) each.

    A signal that only appears at one arbitrary cut is fragile; a real one is monotone-ish across
    reasonable bucket widths.
    """
    out = []
    for f in fracs:
        b = bucket_returns(panel, f)
        ls = long_short_tstat(b)
        out.append((float(f), float(b["spread"]), float(ls["t"])))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic controls (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, fee_alpha: float, si_alpha: float = 0.0, n_seeds: int = 25,
                     base_seed: int = 557) -> float:
    """Average the fee-sort's firm-level slope-t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages a Welch-style stat over >= 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns the mean fee slope-t across
    seeds for a planted ``fee_alpha`` (and optional ``si_alpha`` for the SI-only control).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(fee_alpha=fee_alpha, si_alpha=si_alpha, seed=s)
        ts.append(cross_section_regression(panel)["slope_t"])
    return float(np.nanmean(ts))


def synthetic_mean_ls_t(data_mod, fee_alpha: float, si_alpha: float = 0.0, frac: float = 0.2,
                        n_seeds: int = 25, base_seed: int = 557) -> float:
    """Average the long-cheap/short-expensive Welch t over ``n_seeds`` synthetic worlds."""
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(fee_alpha=fee_alpha, si_alpha=si_alpha, seed=s)
        ts.append(long_short_tstat(bucket_returns(panel, frac))["t"])
    return float(np.nanmean(ts))
