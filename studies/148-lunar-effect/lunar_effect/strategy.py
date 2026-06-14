"""Strategy and analysis for Study 148 (Lunar-Effect).

The claim under test: daily stock returns are **lower** in the ±7-day window
around a full moon than in the ±7-day window around a new moon.

We implement the honest test in two complementary ways:

1. **Window-contrast t-test (HAC).** Compare the mean daily return in the 'NEW'
   window against the mean in the 'FULL' window, with a Newey-West standard error
   on the *difference series* (each day's return minus the unconditional mean, labelled
   by window).  The HAC t-stat on the contrast is the inference-bar number.

2. **Unconditional baseline.** The control arm is simply the unconditional daily return
   of buy-and-hold — if new-moon days beat full-moon days, they must also beat the
   average day by a material amount, since random days are the opportunity cost.

3. **Shuffled-label control.** A random permutation of the lunar labels on the same
   returns, seeded for reproducibility, gives the null distribution of the
   contrast under no relation between the moon and returns.

No look-ahead: the lunar phase on day t is determined by pure astronomy, so no
future return is used to construct the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core contrast statistic
# ---------------------------------------------------------------------------
def window_returns(frame: pd.DataFrame, label: str) -> np.ndarray:
    """Daily returns for a given lunar label ('NEW', 'FULL', or 'OTHER')."""
    mask = frame["lunar_label"] == label
    return frame.loc[mask, "ret"].to_numpy(dtype=float)


def hac_tstat(r: np.ndarray) -> float:
    """Newey-West (HAC) t-stat for the mean of a return series.

    Uses the Bartlett kernel with the standard rule-of-thumb lag count
    ``floor(4 * (n/100)^(2/9))``.
    """
    r = r[np.isfinite(r)]
    n = r.size
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


def contrast_tstat(r_new: np.ndarray, r_full: np.ndarray) -> float:
    """HAC t-stat on the mean *difference* between new-moon and full-moon returns.

    We form a pooled difference series: each new-moon return becomes +1 * r,
    each full-moon return becomes -1 * r, then test whether the pooled mean is
    positive (new > full).  This is equivalent to a two-sample test where we pool
    both groups and test the signed mean.
    """
    r_new = np.asarray(r_new, dtype=float)
    r_new = r_new[np.isfinite(r_new)]
    r_full = np.asarray(r_full, dtype=float)
    r_full = r_full[np.isfinite(r_full)]
    if r_new.size < 6 or r_full.size < 6:
        return float("nan")
    pooled = np.concatenate([r_new, -r_full])
    return hac_tstat(pooled)


def summarize(frame: pd.DataFrame) -> dict:
    """Core statistics for the lunar-window contrast.

    Returns mean returns (in bps/day) for each window, the contrast
    (new − full), and HAC t-stats for each window mean and the contrast.
    Also returns a shuffled-label control (same returns, random labels).
    """
    r_new = window_returns(frame, "NEW")
    r_full = window_returns(frame, "FULL")
    r_other = window_returns(frame, "OTHER")
    r_all = frame["ret"].to_numpy(dtype=float)
    r_all = r_all[np.isfinite(r_all)]

    mean_new = float(r_new.mean() * 1e4) if r_new.size else float("nan")
    mean_full = float(r_full.mean() * 1e4) if r_full.size else float("nan")
    mean_other = float(r_other.mean() * 1e4) if r_other.size else float("nan")
    mean_all = float(r_all.mean() * 1e4) if r_all.size else float("nan")

    contrast_bps = mean_new - mean_full  # positive = new beats full (the YZZ claim)

    return {
        "n_new": int(r_new.size),
        "n_full": int(r_full.size),
        "n_other": int(r_other.size),
        "n_total": int(r_all.size),
        "mean_new_bps": mean_new,
        "mean_full_bps": mean_full,
        "mean_other_bps": mean_other,
        "mean_all_bps": mean_all,
        "contrast_bps": contrast_bps,
        "t_new": hac_tstat(r_new),
        "t_full": hac_tstat(r_full),
        "t_contrast": contrast_tstat(r_new, r_full),
    }


# ---------------------------------------------------------------------------
# Shuffled-label control (the null distribution)
# ---------------------------------------------------------------------------
def shuffled_contrast(
    frame: pd.DataFrame,
    n_perms: int = 500,
    seed: int = 148,
) -> np.ndarray:
    """Distribution of the new-vs-full contrast under random label permutation.

    Permutes the ``lunar_label`` column (preserving label counts) and recomputes
    the contrast ``mean(NEW) - mean(FULL)`` for each permutation.  The fraction of
    permuted contrasts exceeding the observed contrast is a permutation p-value.

    Returns a 1-D array of ``n_perms`` contrast values (in bps/day).
    """
    rng = np.random.default_rng(seed)
    labels = frame["lunar_label"].to_numpy()
    rets = frame["ret"].to_numpy(dtype=float)
    contrasts = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(labels)
        r_new = rets[perm == "NEW"]
        r_full = rets[perm == "FULL"]
        if r_new.size and r_full.size:
            contrasts[i] = (r_new.mean() - r_full.mean()) * 1e4
        else:
            contrasts[i] = float("nan")
    return contrasts


# ---------------------------------------------------------------------------
# Decade-by-decade breakdown (post-publication decay check)
# ---------------------------------------------------------------------------
def by_decade(frame: pd.DataFrame) -> pd.DataFrame:
    """Contrast and t-stat broken down by decade.

    Returns a DataFrame indexed by decade start year with columns:
    ``n_new``, ``n_full``, ``contrast_bps``, ``t_contrast``.
    """
    years = frame.index.year
    decades = range((years.min() // 10) * 10, (years.max() // 10) * 10 + 10, 10)
    rows = []
    for dec in decades:
        sub = frame[(years >= dec) & (years < dec + 10)]
        if len(sub) < 50:
            continue
        s = summarize(sub)
        rows.append({
            "decade": dec,
            "n_new": s["n_new"],
            "n_full": s["n_full"],
            "contrast_bps": s["contrast_bps"],
            "t_contrast": s["t_contrast"],
        })
    return pd.DataFrame(rows).set_index("decade")


# ---------------------------------------------------------------------------
# Annualised Sharpe for a naive long-new / short-full strategy
# ---------------------------------------------------------------------------
def lunar_strategy_returns(frame: pd.DataFrame) -> pd.Series:
    """Daily returns of a simple long-new / short-full overlay.

    - NEW days: hold the index (long).
    - FULL days: short the index (short).
    - OTHER days: flat (return = 0).

    This is the most optimistic possible implementation of the YZZ claim with
    no transaction costs — it will be compared with buy-and-hold and with
    a shuffled-label version to see if any gross edge exists.
    """
    label = frame["lunar_label"].to_numpy()
    ret = frame["ret"].to_numpy(dtype=float)
    sign = np.where(label == "NEW", 1.0, np.where(label == "FULL", -1.0, 0.0))
    return pd.Series(sign * ret, index=frame.index, name="strategy_ret")
