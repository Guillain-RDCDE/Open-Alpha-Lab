"""Strategy + inference for Study 533 — Analyst-Dispersion (Diether-Malloy-Scherbina puzzle).

The claim: stocks with **high dispersion in analyst forecasts earn LOWER returns** (Diether,
Malloy & Scherbina 2002). We test it as a single cross-sectional sort:

    * **Dispersion** = the forecast spread ``(high − low) / |mean|`` of the current-fiscal-year
      consensus EPS estimate (the free DMS proxy).
    * **The DMS long-short** = go *long the LOW-dispersion* tercile and *short the HIGH-
      dispersion* tercile, and ask whether that **low-minus-high** spread is positive (the DMS
      prediction: low dispersion out-performs).
    * **The return measured** is the trailing H-day return realised into the snapshot date
      (the only thing a single dispersion snapshot lets us do honestly — a contemporaneous
      association, not a tradable forward panel).

Inference, the desk's shared spirit:

  * a **one-sample t** of the per-name long-short sample (low-tercile returns as longs,
    high-tercile returns as −shorts) against 0;
  * a **label-shuffle placebo** — permute the dispersion labels across names many times and ask
    how often a random tercile split yields as large a low-minus-high spread (the honest null
    for a tiny cross-section);
  * a **win-rate** of the long-short vs the 50% coin-flip base rate;
  * **one-way costs × turnover** (+ borrow on the short leg) on the implied trade.

The decisive number is the low-minus-high spread and its t. On a ~40-name single cross-section,
even a real effect is unlikely to clear the desk's t ≥ 2 bar; we report what the tape shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126, 252)            # trailing-return horizons (trading days)


# --------------------------------------------------------------------------- #
# Tercile bucketing
# --------------------------------------------------------------------------- #
def _bucketize(x: np.ndarray, n_buckets: int = 3) -> np.ndarray:
    """Rank ``x`` into ``n_buckets`` equal-frequency bins (0 = lowest dispersion)."""
    order = np.argsort(np.argsort(x))            # dense ranks 0..n-1
    return np.minimum((order * n_buckets // len(x)), n_buckets - 1)


def long_short(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion",
               n_buckets: int = 3) -> dict:
    """Low-minus-high dispersion long-short on a frame carrying ``ret_col`` and ``disp_col``.

    DMS sign: **long the LOW-dispersion** bucket, **short the HIGH-dispersion** bucket. The
    spread is mean(low-disp returns) − mean(high-disp returns); a positive value supports the
    DMS claim (low dispersion out-performs). Returns bucket means, the spread, a pooled
    per-name long-short sample (low-disp returns as +, high-disp returns as −) for the t-test,
    and the win-rate.
    """
    f = frame.dropna(subset=[ret_col, disp_col])
    s = f[disp_col].values
    r = f[ret_col].values
    b = _bucketize(s, n_buckets)
    low = r[b == 0]                              # LOW dispersion = long leg
    high = r[b == n_buckets - 1]                 # HIGH dispersion = short leg
    ls_mean = float(low.mean() - high.mean())
    ls_sample = np.concatenate([low, -high])     # longs +, shorts −
    win = float((ls_sample > 0).mean())
    return {
        "n": int(len(f)), "n_low": int(len(low)), "n_high": int(len(high)),
        "low_mean": float(low.mean()), "high_mean": float(high.mean()),
        "ls_mean": ls_mean, "ls_sample": ls_sample, "ls_win": win,
        "low": low, "high": high,
    }


def bucket_means(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion",
                 n_buckets: int = 3) -> np.ndarray:
    """Mean return in each dispersion bucket (low -> high) — the DMS monotonicity picture."""
    f = frame.dropna(subset=[ret_col, disp_col])
    s = f[disp_col].values
    r = f[ret_col].values
    b = _bucketize(s, n_buckets)
    return np.array([r[b == k].mean() if (b == k).any() else np.nan
                     for k in range(n_buckets)])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0 (the long-short edge test)."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def placebo_pvalue(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion",
                   n_draws: int = 20_000, n_buckets: int = 3, seed: int = 533) -> dict:
    """Label-shuffle placebo null for the low-minus-high dispersion spread.

    Permute the dispersion labels across names many times; each shuffle re-forms random
    terciles and recomputes the low-minus-high spread. ``p`` = P[|shuffled| ≥ |observed|] —
    two-sided, the honest answer to "could a random sort of the same returns have looked this
    good?" on a tiny cross-section.
    """
    f = frame.dropna(subset=[ret_col, disp_col])
    s = f[disp_col].values
    r = f[ret_col].values
    obs = abs(float(long_short(f, ret_col, disp_col, n_buckets)["ls_mean"]))
    rng = np.random.default_rng(seed)
    hi = n_buckets - 1
    means = np.empty(n_draws)
    for i in range(n_draws):
        b = _bucketize(rng.permutation(s), n_buckets)
        means[i] = r[b == 0].mean() - r[b == hi].mean()
    p = float((np.abs(means) >= obs).mean())
    return {"obs": float(long_short(f, ret_col, disp_col, n_buckets)["ls_mean"]),
            "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


def summarize(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion",
              n_buckets: int = 3, placebo: bool = True, n_draws: int = 20_000) -> dict:
    """Headline stats for one return horizon: bucket means, spread/t/win/placebo."""
    ls = long_short(frame, ret_col, disp_col, n_buckets)
    t = ttest_vs_zero(ls["ls_sample"])
    p = (placebo_pvalue(frame, ret_col, disp_col, n_draws=n_draws,
                        n_buckets=n_buckets)["p_value"] if placebo else float("nan"))
    return {
        "ret_col": ret_col, "n": ls["n"], "n_low": ls["n_low"], "n_high": ls["n_high"],
        "low_mean": ls["low_mean"], "high_mean": ls["high_mean"],
        "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"], "t": t, "p_placebo": p,
    }


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion",
                 cost_bps: float = 10.0, borrow_bps_ann: float = 100.0,
                 horizon_days: int = 252, n_buckets: int = 3) -> dict:
    """Low-minus-high spread net of one-way costs × turnover (+ borrow on the short leg).

    The implied trade is a fresh long-short establishment: ``cost_bps`` one-way per leg per
    side (2 legs long + 2 legs short = 4 × one-way on the long-short), plus ``borrow_bps_ann``
    annualised over the holding horizon on the short (high-dispersion) leg.
    """
    ls = long_short(frame, ret_col, disp_col, n_buckets)
    c = cost_bps / 1e4
    round_trip = 4.0 * c
    borrow = (borrow_bps_ann / 1e4) * (horizon_days / 252.0)
    gross = ls["ls_mean"]
    net = gross - round_trip - borrow
    return {"gross": float(gross), "net": float(net),
            "cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann}


# --------------------------------------------------------------------------- #
# Spearman correlation — dispersion vs return (the continuous view)
# --------------------------------------------------------------------------- #
def disp_return_corr(frame: pd.DataFrame, ret_col: str, disp_col: str = "dispersion") -> float:
    """Spearman rank correlation between dispersion and the return column.

    DMS predicts a **negative** correlation (high dispersion → low return). Reported alongside
    the tercile spread as the continuous-signal view that does not throw away the middle.
    """
    f = frame.dropna(subset=[ret_col, disp_col])
    if len(f) < 3:
        return float("nan")
    a = np.argsort(np.argsort(f[disp_col].values)).astype(float)
    b = np.argsort(np.argsort(f[ret_col].values)).astype(float)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / denom) if denom > 0 else float("nan")
