"""The strategy and its honest controls — Study 193 (Window-Dressing).

The folk claim (Carhart et al. 2002): mutual funds bid up their winners in the last
few days of each calendar quarter ("window dressing" / "portfolio pumping") to
present a flattering holdings snapshot at the quarter end. This inflates returns in
the *pump* window (last ``window`` trading days of Mar/Jun/Sep/Dec) and produces a
mechanical *reversal* in the first ``window`` days of the following quarter.

We run three honest tests:

1. **Pump vs baseline** — the mean daily log-return in the pump window vs the
   unconditional mean of all *other* days.  HAC Newey-West t-stat on each group;
   Welch t-test on the contrast.

2. **Reversal vs baseline** — the same comparison for the early-next-quarter window.
   The Carhart story predicts a *negative* return bias here.

3. **Pump − Reversal spread** — the "buy into the pump and fade the reversal" strategy:
   long in the pump window, flat or short in the reversal window. We compute the mean
   daily return of the long-pump / short-reversal strategy and compare it to a
   **random-day control** (random assignments of the same number of long/short days,
   seeded, to confirm the signal is in the *timing* and not in the equity premium).

No look-ahead: the calendar window is determined purely from the date, which is known
before the open. The return measured is the same-day close-to-close log-return. Using
the day *after* the signal would be a T+1 entry; for a daily study testing a calendar
label, the convention is to test the *labelled* day directly (the calendar is a pure
predictor, not a reaction to prices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {
            "tstat": float("nan"),
            "mean": float(r.mean()) if n else float("nan"),
            "se": float("nan"),
            "n": n,
        }
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "mean": float(mu),
        "se": float(se),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Test 1 & 2: pump and reversal windows vs the unconditional baseline
# ---------------------------------------------------------------------------
def window_comparison(df: pd.DataFrame, window_col: str) -> dict:
    """Compare mean return in a labelled window to all other trading days.

    Parameters
    ----------
    df : DataFrame with column ``ret`` and boolean column ``window_col``
         (``is_pump`` or ``is_reversal``).
    window_col : str
        The boolean label column to compare against its complement.

    Returns
    -------
    dict with keys: n_window, mean_window_bps, tstat_window,
    n_other, mean_other_bps, tstat_other, contrast_bps, tstat_welch, pval_welch.
    """
    from scipy.stats import ttest_ind

    win = df.loc[df[window_col], "ret"].to_numpy(dtype=float)
    other = df.loc[~df[window_col], "ret"].to_numpy(dtype=float)
    win = win[np.isfinite(win)]
    other = other[np.isfinite(other)]

    hw = _hac_tstat(win)
    ho = _hac_tstat(other)

    if len(win) > 1 and len(other) > 1:
        t_welch, pval_welch = ttest_ind(win, other, equal_var=False)
    else:
        t_welch, pval_welch = float("nan"), float("nan")

    label = window_col.replace("is_", "")
    return {
        f"n_{label}": int(len(win)),
        f"mean_{label}_bps": float(win.mean() * 1e4) if len(win) else float("nan"),
        f"tstat_{label}": hw["tstat"],
        "n_other": int(len(other)),
        "mean_other_bps": float(other.mean() * 1e4) if len(other) else float("nan"),
        "tstat_other": ho["tstat"],
        "contrast_bps": float((win.mean() - other.mean()) * 1e4) if len(win) and len(other) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval_welch),
    }


# ---------------------------------------------------------------------------
# Test 3: random-day control
# ---------------------------------------------------------------------------
def random_day_control(
    df: pd.DataFrame,
    n_pump: int,
    n_reversal: int,
    seed: int = 0,
) -> dict:
    """Placebo: long on n_pump randomly-selected days, short on n_reversal others.

    This isolates whether the pump/reversal *timing* (calendar label) carries
    information, or whether any long-short constructed from the same number of
    randomly drawn days earns the same return.

    Returns the mean long-minus-short daily return (in bps) and its HAC t-stat.
    """
    rng = np.random.default_rng(seed)
    ret = df["ret"].dropna().to_numpy(dtype=float)
    n = len(ret)
    idx_long = rng.choice(n, size=min(n_pump, n), replace=False)
    remaining = np.setdiff1d(np.arange(n), idx_long)
    idx_short = rng.choice(remaining, size=min(n_reversal, len(remaining)), replace=False)

    longs = ret[idx_long]
    shorts = ret[idx_short]
    spread = np.concatenate([longs, -shorts])

    h = _hac_tstat(spread)
    return {
        "mean_bps": float(spread.mean() * 1e4),
        "tstat": h["tstat"],
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
    }


# ---------------------------------------------------------------------------
# Pump-minus-reversal strategy
# ---------------------------------------------------------------------------
def pump_minus_reversal(df: pd.DataFrame) -> dict:
    """Long-pump / short-reversal spread return.

    Returns mean daily log-return of the combined strategy (long on pump days,
    short on reversal days), its HAC t-stat, and the annualised mean return.
    Neutral days are not included in the mean (they contribute 0 to the
    strategy; only active days are averaged — this gives the *per-active-day*
    edge, which is the right signal-quality metric).
    """
    pump_ret = df.loc[df["is_pump"], "ret"].to_numpy(dtype=float)
    rev_ret = df.loc[df["is_reversal"], "ret"].to_numpy(dtype=float)
    pump_ret = pump_ret[np.isfinite(pump_ret)]
    rev_ret = rev_ret[np.isfinite(rev_ret)]

    spread = np.concatenate([pump_ret, -rev_ret])
    h = _hac_tstat(spread)
    ann = float(spread.mean() * 252.0) if len(spread) else float("nan")

    return {
        "n_pump": int(len(pump_ret)),
        "n_reversal": int(len(rev_ret)),
        "n_active": int(len(spread)),
        "mean_pump_bps": float(pump_ret.mean() * 1e4) if len(pump_ret) else float("nan"),
        "mean_reversal_bps": float(rev_ret.mean() * 1e4) if len(rev_ret) else float("nan"),
        "mean_spread_bps": float(spread.mean() * 1e4) if len(spread) else float("nan"),
        "tstat": h["tstat"],
        "ann_ret": ann,
    }


# ---------------------------------------------------------------------------
# Seasonal window sweep: test windows of 1–10 days to check for data-snooping
# ---------------------------------------------------------------------------
def window_sweep(df: pd.DataFrame, max_window: int = 10) -> pd.DataFrame:
    """Sweep window sizes 1–max_window and report the Welch p for each.

    This is the multiple-comparisons check: if the result only appears at one
    window size and is absent everywhere else, it is likely a chance artefact.
    A real effect should be somewhat robust to the choice of window.

    Returns a DataFrame indexed by window size with columns: n_pump, contrast_pump_bps,
    pval_pump, n_reversal, contrast_reversal_bps, pval_reversal, pval_bonferroni.
    """
    from window_dressing.data import _quarter_labels

    rows = []
    n_tests = max_window  # one test per window size
    idx = df.index
    ret = df["ret"].to_numpy(dtype=float)
    base_df = pd.DataFrame({"ret": ret}, index=idx)

    for w in range(1, max_window + 1):
        labs = _quarter_labels(idx, window=w)
        tmp = base_df.copy()
        tmp["is_pump"] = labs["is_pump"].values
        tmp["is_reversal"] = labs["is_reversal"].values

        cp = window_comparison(tmp, "is_pump")
        cr = window_comparison(tmp, "is_reversal")

        raw_p = min(cp["pval_welch"], cr["pval_welch"])
        bonf_p = min(raw_p * n_tests, 1.0) if np.isfinite(raw_p) else float("nan")

        rows.append(
            {
                "window": w,
                "n_pump": cp["n_pump"],
                "contrast_pump_bps": cp["contrast_bps"],
                "pval_pump": cp["pval_welch"],
                "n_reversal": cr["n_reversal"],
                "contrast_reversal_bps": cr["contrast_bps"],
                "pval_reversal": cr["pval_welch"],
                "pval_bonferroni": bonf_p,
            }
        )
    return pd.DataFrame(rows).set_index("window")


# ---------------------------------------------------------------------------
# Summarize — all tests in one call
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for Window-Dressing: all tests combined.

    Returns a flat dict mirroring docs/results.md for use in notebooks and verify.py.
    """
    pump = window_comparison(df, "is_pump")
    rev = window_comparison(df, "is_reversal")
    pmr = pump_minus_reversal(df)
    n_pump = pmr["n_pump"]
    n_rev = pmr["n_reversal"]
    ctrl = random_day_control(df, n_pump=n_pump, n_reversal=n_rev, seed=193)

    return {
        # Pump window
        "n_pump": pump["n_pump"],
        "mean_pump_bps": pump["mean_pump_bps"],
        "tstat_pump": pump["tstat_pump"],
        "contrast_pump_bps": pump["contrast_bps"],
        "pval_pump": pump["pval_welch"],
        # Reversal window
        "n_reversal": rev["n_reversal"],
        "mean_reversal_bps": rev["mean_reversal_bps"],
        "tstat_reversal": rev["tstat_reversal"],
        "contrast_reversal_bps": rev["contrast_bps"],
        "pval_reversal": rev["pval_welch"],
        # Spread strategy
        "mean_spread_bps": pmr["mean_spread_bps"],
        "tstat_spread": pmr["tstat"],
        "ann_ret": pmr["ann_ret"],
        # Random-day control
        "ctrl_mean_bps": ctrl["mean_bps"],
        "ctrl_tstat": ctrl["tstat"],
        # Baseline (all other days)
        "mean_other_bps": pump["mean_other_bps"],
    }
