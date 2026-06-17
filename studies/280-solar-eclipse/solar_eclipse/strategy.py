"""Strategy and analysis layer for Study 280 (Solar-Eclipse).

The folklore: solar eclipses are bad omens, so the stock market should swoon
around eclipse days. We test this as a classic **event study**:

  - Align ^GSPC daily returns into a symmetric window of ``win`` trading days
    either side of each eclipse (event day 0 = first trading day on/after the
    eclipse, the soonest a trader could act — one explicit execution-lag day).
  - Compute the **average abnormal return (AAR)** per relative day and the
    **cumulative abnormal return (CAR)** across the window, where "abnormal" is
    measured against the full-sample daily mean (a flat market model — eclipses
    are too rare for a rolling estimation window to matter).
  - Test the event-day mean return with a **HAC (Newey-West) t-stat** and a
    **permutation test** (shuffle which calendar days are "eclipse days" and
    recompute the event-day mean), the honest small-n null.
  - Slice by ``kind`` (total / annular / hybrid) and by ``us_path`` (path crossed
    the contiguous US) — each slice is a separate test, the textbook
    multiple-comparisons problem that manufactures false discoveries.

The tiny-n reckoning: there are ~80 central solar eclipses in the ^GSPC era,
and only ~10 whose path crossed the US. With ~1% daily volatility, a single
eclipse-day mean is estimated to roughly ``1% / sqrt(80) ≈ 11 bps`` — so any
"effect" smaller than ~25 bps is statistical noise dressed as an omen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SEED = 280


# ---------------------------------------------------------------------------
# Event alignment
# ---------------------------------------------------------------------------
def event_day_returns(
    ret: pd.Series,
    eclipses: pd.DataFrame,
    lag: int = 0,
) -> pd.DataFrame:
    """Return one row per eclipse with the event-day return on ^GSPC.

    Event day 0 is the first trading day on/after the eclipse date; ``lag``
    shifts it forward by that many trading days (``lag=0`` = act on the first
    available close, the most generous-to-the-omen convention with a built-in
    one-day execution realism since the eclipse itself is mid-session/overnight).

    Returns columns: ``date`` (eclipse), ``event_date`` (trading day used),
    ``kind``, ``us_path``, ``ret`` (event-day simple return). Eclipses whose
    window falls outside the price series are dropped.
    """
    idx = ret.index
    rows = []
    for _, e in eclipses.iterrows():
        pos = idx.searchsorted(pd.Timestamp(e["date"])) + lag
        if 0 <= pos < len(idx):
            rows.append({
                "date": e["date"],
                "event_date": idx[pos],
                "kind": e["kind"],
                "us_path": bool(e["us_path"]),
                "ret": float(ret.iloc[pos]),
            })
    return pd.DataFrame(rows)


def event_window_matrix(
    ret: pd.Series,
    eclipses: pd.DataFrame,
    win: int = 5,
) -> pd.DataFrame:
    """Stack a symmetric ±``win`` trading-day return window around each eclipse.

    Returns a DataFrame indexed by eclipse date, columns = relative day
    ``-win .. +win`` (0 = first trading day on/after the eclipse). Eclipses
    without a full window in the price series are dropped.
    """
    idx = ret.index
    rel = list(range(-win, win + 1))
    rows = {}
    for _, e in eclipses.iterrows():
        pos = idx.searchsorted(pd.Timestamp(e["date"]))
        if pos - win < 0 or pos + win >= len(idx):
            continue
        rows[e["date"]] = [float(ret.iloc[pos + k]) for k in rel]
    mat = pd.DataFrame.from_dict(rows, orient="index", columns=rel)
    mat.index.name = "eclipse"
    return mat


# ---------------------------------------------------------------------------
# HAC inference + permutation test
# ---------------------------------------------------------------------------
def _nw_tstat(x: np.ndarray, lags: int = 3) -> tuple[float, float]:
    """Newey-West (HAC) t-stat for the mean of ``x`` against 0.

    Returns ``(t, se)``. With ``lags=0`` this is the ordinary t-stat; a few lags
    guard against the mild autocorrelation that creeps in when nearby eclipse
    windows overlap.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = (u @ u) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = (u[k:] @ u[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n) if var > 0 else float("nan")
    t = mu / se if se and se > 0 else float("nan")
    return float(t), float(se)


def event_stats(
    ret: pd.Series,
    eclipses: pd.DataFrame,
    lag: int = 0,
    nw_lags: int = 3,
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> dict:
    """Event-day return statistics with HAC t-stat and a permutation test.

    The event-day return for each eclipse is compared to the **unconditional**
    daily mean (the honest baseline — the market drifts up ~3 bps/day regardless).
    The 'abnormal' event-day return is ``event_ret - uncond_mean``.

    Permutation test: draw ``n_eclipses`` random trading days from the same price
    series 10,000 times and record the distribution of mean abnormal returns; the
    p-value is the (two-sided) fraction of shuffles at least as extreme as the
    observed mean.

    Returns a flat dict (means in bps) plus ``perm_dist`` (the full permutation
    distribution of mean abnormal returns, in bps, for plotting).
    """
    ev = event_day_returns(ret, eclipses, lag=lag)
    uncond_mean = float(ret.mean())
    uncond_bps = uncond_mean * 1e4

    ev_ret = ev["ret"].to_numpy(dtype=float)
    abn = ev_ret - uncond_mean
    n = len(abn)

    mean_event_bps = float(ev_ret.mean()) * 1e4
    mean_abn_bps = float(abn.mean()) * 1e4
    win_rate = float((ev_ret > 0).mean())

    t_hac, se = _nw_tstat(abn, lags=nw_lags)
    se_bps = se * 1e4 if se == se else float("nan")

    # Ordinary one-sample t for reference.
    if n >= 2:
        t_ord, p_ord = scipy_stats.ttest_1samp(abn, 0.0)
    else:
        t_ord, p_ord = float("nan"), float("nan")

    # Permutation test: random trading days, same n, same series.
    rng = np.random.default_rng(seed)
    all_ret = ret.to_numpy(dtype=float)
    n_all = len(all_ret)
    perm_means = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        pick = rng.choice(n_all, size=n, replace=False)
        perm_means[i] = (all_ret[pick].mean() - uncond_mean) * 1e4
    perm_p = float((np.abs(perm_means) >= abs(mean_abn_bps)).mean())

    return {
        "n_events": int(n),
        "uncond_bps": round(uncond_bps, 3),
        "mean_event_bps": round(mean_event_bps, 3),
        "mean_abn_bps": round(mean_abn_bps, 3),
        "se_abn_bps": round(se_bps, 3) if se_bps == se_bps else float("nan"),
        "win_rate": round(win_rate, 4),
        "t_hac": round(float(t_hac), 3),
        "t_ord": round(float(t_ord), 3),
        "p_ord": round(float(p_ord), 4),
        "perm_p": round(perm_p, 4),
        "perm_dist": perm_means,
    }


# ---------------------------------------------------------------------------
# CAR over the window
# ---------------------------------------------------------------------------
def car_profile(ret: pd.Series, eclipses: pd.DataFrame, win: int = 5) -> pd.DataFrame:
    """Average abnormal return (AAR) and cumulative AAR (CAR) by relative day.

    Returns a DataFrame indexed by relative day (-win..+win) with columns
    ``aar_bps`` (cross-eclipse mean abnormal return that day, in bps) and
    ``car_bps`` (running sum of aar_bps from -win). Abnormal = raw minus the
    full-sample daily mean.
    """
    mat = event_window_matrix(ret, eclipses, win=win)
    uncond = float(ret.mean())
    aar = (mat.mean(axis=0) - uncond) * 1e4
    out = pd.DataFrame({"aar_bps": aar})
    out["car_bps"] = out["aar_bps"].cumsum()
    out.index.name = "rel_day"
    return out


# ---------------------------------------------------------------------------
# Multiple-comparisons summary over slices
# ---------------------------------------------------------------------------
def multiple_comparisons_summary(
    ret: pd.Series,
    eclipses: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the event-day test on each slice and Bonferroni-correct the p-values.

    Slices: all eclipses, total only, annular only, US-path only. Each shares the
    same ^GSPC tape, so the Bonferroni factor is the number of slices tested.
    """
    slices = [
        ("all eclipses", eclipses),
        ("total only", eclipses[eclipses["kind"] == "total"]),
        ("annular only", eclipses[eclipses["kind"] == "annular"]),
        ("US-path only", eclipses[eclipses["us_path"]]),
    ]
    rows = []
    n_tests = len(slices)
    for label, sub in slices:
        if len(sub) < 2:
            continue
        st = event_stats(ret, sub, n_permutations=n_permutations, seed=seed)
        rows.append({
            "slice": label,
            "n": st["n_events"],
            "mean_abn_bps": st["mean_abn_bps"],
            "t_hac": st["t_hac"],
            "perm_p": st["perm_p"],
            "bonferroni_perm_p": min(1.0, st["perm_p"] * n_tests),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    ret: pd.Series,
    eclipses: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> dict:
    """One-stop summary dict mirroring R = dict(...) in build_notebooks.py."""
    st_all = event_stats(ret, eclipses, n_permutations=n_permutations, seed=seed)
    us = eclipses[eclipses["us_path"]]
    st_us = event_stats(ret, us, n_permutations=n_permutations, seed=seed)
    tot = eclipses[eclipses["kind"] == "total"]
    st_tot = event_stats(ret, tot, n_permutations=n_permutations, seed=seed)

    car = car_profile(ret, eclipses, win=5)
    car_full = float(car["car_bps"].iloc[-1])

    return {
        "n_all": st_all["n_events"],
        "n_us": st_us["n_events"],
        "n_total": st_tot["n_events"],
        "uncond_bps": st_all["uncond_bps"],
        "all_mean_abn_bps": st_all["mean_abn_bps"],
        "all_t_hac": st_all["t_hac"],
        "all_perm_p": st_all["perm_p"],
        "all_win_rate_pct": round(st_all["win_rate"] * 100, 1),
        "us_mean_abn_bps": st_us["mean_abn_bps"],
        "us_t_hac": st_us["t_hac"],
        "us_perm_p": st_us["perm_p"],
        "total_mean_abn_bps": st_tot["mean_abn_bps"],
        "total_t_hac": st_tot["t_hac"],
        "total_perm_p": st_tot["perm_p"],
        "car_window_bps": round(car_full, 2),
    }
