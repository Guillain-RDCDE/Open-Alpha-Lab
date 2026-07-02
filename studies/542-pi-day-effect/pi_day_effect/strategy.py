"""The engine and its honest controls — Study 542 (Pi-Day-Effect).

The (tongue-in-cheek) claim: the market behaves differently on dates that encode a
famous mathematical constant — Pi Day (3/14), Euler's day (2/7), Tau day (6/28), and
friends — than on ordinary trading days. This is a numerology anomaly, so the honest
question is not "is the mean nonzero?" but "is it more extreme than what a *random*
set of the same number of calendar slots would produce?"

The tests:

1. **Pi-Day vs the rest.** Welch two-sample t on Pi-Day daily returns vs all other
   trading days, plus a HAC (Newey-West) t on the Pi-Day mean itself.

2. **All constant days vs the rest.** The pooled constant-day set (Pi/e/tau/phi/sqrt2/
   Feigenbaum) vs ordinary days — same two-sample machinery.

3. **Random-date-set placebo (the kill shot).** Draw many random sets of K distinct
   (month, day) slots — the same size as the constant-day set — and measure how often a
   random set's |contrast| beats the constant-day |contrast|. A genuine numerology
   signal sits in the tail; noise does not. This is the multiple-testing-aware null: it
   embeds the fact that we *chose* these dates from the space of all calendar dates.

4. **Per-constant sweep with Bonferroni.** Test each constant date separately and
   Bonferroni-correct across the six — so cherry-picking the most extreme constant
   can't manufacture significance.

5. **Costs.** A naive "own the market only on constant days" timing rule pays a
   round-trip cost per event; report gross vs net so the tradability axis is honest.

6. **Robustness.** The Pi-Day contrast over rolling multi-decade sub-windows — a real
   effect keeps its sign; numerology does not.

7. **Synthetic positive control.** A deterministic world with a planted constant-day
   bump; the engine must recover it (placebo p in the tail) and stay flat at the null,
   averaged over >= 20 seeds so no single lucky seed fakes significance.

No look-ahead: the date label is known before the open; the return is that day's
close-to-close log-return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic on a sample mean
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {"tstat": float("nan"), "mean": float(r.mean()) if n else float("nan"),
                "se": float("nan"), "n": n}
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"tstat": float(mu / se) if se > 0 else float("nan"),
            "mean": float(mu), "se": float(se), "n": n}


# ---------------------------------------------------------------------------
# Two-sample comparisons
# ---------------------------------------------------------------------------
def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch (unequal-variance) two-sample t and p on means of ``a`` vs ``b``."""
    from scipy.stats import ttest_ind

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) > 1 and len(b) > 1:
        t, p = ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    return float("nan"), float("nan")


def group_comparison(df: pd.DataFrame, mask_col: str) -> dict:
    """Compare returns on ``df[mask_col]`` days vs all other days.

    Returns means (bps), event count, contrast (bps), the Welch two-sample t/p, and a
    HAC t on the event-group mean.
    """
    grp = df.loc[df[mask_col], "ret"].to_numpy(dtype=float)
    rest = df.loc[~df[mask_col], "ret"].to_numpy(dtype=float)
    grp = grp[np.isfinite(grp)]
    rest = rest[np.isfinite(rest)]
    t, p = _welch(grp, rest)
    h = _hac_tstat(grp)
    return {
        "n_event": int(len(grp)),
        "n_rest": int(len(rest)),
        "mean_event_bps": float(grp.mean() * 1e4) if len(grp) else float("nan"),
        "mean_rest_bps": float(rest.mean() * 1e4) if len(rest) else float("nan"),
        "contrast_bps": float((grp.mean() - rest.mean()) * 1e4)
        if len(grp) and len(rest) else float("nan"),
        "tstat_welch": t,
        "pval_welch": p,
        "tstat_hac": h["tstat"],
    }


def pi_day_comparison(df: pd.DataFrame) -> dict:
    """Pi-Day (March 14) returns vs all other trading days."""
    return group_comparison(df, "is_pi")


def constant_day_comparison(df: pd.DataFrame) -> dict:
    """All constant days (pooled) vs all other trading days."""
    return group_comparison(df, "is_constant")


# ---------------------------------------------------------------------------
# Random-date-set placebo — the multiple-testing-aware null
# ---------------------------------------------------------------------------
def _md_of(df: pd.DataFrame) -> list[tuple[int, int]]:
    """(month, day) tuples for each row, as a plain list for hashing/masking."""
    idx = df.index
    return list(zip(idx.month.tolist(), idx.day.tolist()))


def random_dateset_placebo(
    df: pd.DataFrame, k: int, n_perm: int = 5000, seed: int = 542
) -> dict:
    """Placebo p-value: how often does a *random* set of K calendar slots beat ours?

    We measure the observed |contrast| of the constant-day set (K = 6 slots), then draw
    ``n_perm`` random sets of K distinct (month, day) slots (from the calendar slots that
    actually occur in the tape) and record the fraction whose |contrast| >= observed.

    This is the honest numerology null: it prices in the fact that the constant dates
    were *chosen* from the space of all dates. A real effect sits in the tail (small p);
    a coincidence does not.
    """
    md_all = _md_of(df)
    ret_all = df["ret"].to_numpy(dtype=float)
    is_const_all = df["is_constant"].to_numpy()
    finite = np.isfinite(ret_all)
    ret = ret_all[finite]
    md = [m for m, keep in zip(md_all, finite) if keep]
    is_const = is_const_all[finite]

    # observed contrast on the constant-day set
    obs = abs(ret[is_const].mean() - ret[~is_const].mean())

    # map each row to an integer slot id (distinct calendar (month, day) present)
    slots = sorted(set(md))
    slot_id = {s: i for i, s in enumerate(slots)}
    row_slot = np.array([slot_id[m] for m in md], dtype=int)
    n_slots = len(slots)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        pick = rng.choice(n_slots, size=k, replace=False)
        sel = np.isin(row_slot, pick)
        if sel.sum() < 2 or (~sel).sum() < 2:
            continue
        contrast = abs(ret[sel].mean() - ret[~sel].mean())
        if contrast >= obs - 1e-18:
            count += 1
    return {"obs_contrast_bps": float(obs * 1e4), "k": int(k),
            "n_perm": int(n_perm), "pval": (count + 1) / (n_perm + 1)}


# ---------------------------------------------------------------------------
# Per-constant sweep with Bonferroni
# ---------------------------------------------------------------------------
def per_constant_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Test each constant date separately vs all other days; Bonferroni-correct.

    Returns a DataFrame (sorted by raw p) with n, mean_bps, contrast_bps, pval_raw and
    pval_bonferroni across the six constant dates. The most extreme constant sitting at
    the top with a corrected p far from significance is the multiple-testing kill shot.
    """
    names = _data.CONSTANT_NAMES
    n_tests = len(names)
    rows = []
    for nm in names:
        mask = _data.constant_mask(df.index, nm).to_numpy()
        grp = df.loc[mask, "ret"].to_numpy(dtype=float)
        rest = df.loc[~mask, "ret"].to_numpy(dtype=float)
        grp = grp[np.isfinite(grp)]
        rest = rest[np.isfinite(rest)]
        t, p = _welch(grp, rest)
        rows.append({
            "constant": nm,
            "n": int(len(grp)),
            "mean_bps": float(grp.mean() * 1e4) if len(grp) else float("nan"),
            "contrast_bps": float((grp.mean() - rest.mean()) * 1e4)
            if len(grp) and len(rest) else float("nan"),
            "tstat_welch": t,
            "pval_raw": p,
            "pval_bonferroni": float(min(p * n_tests, 1.0)) if np.isfinite(p) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("pval_raw").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Costs — a naive "own the market only on constant days" timing rule
# ---------------------------------------------------------------------------
def timing_rule_pnl(df: pd.DataFrame, mask_col: str = "is_pi",
                    cost_bps: float = 5.0) -> dict:
    """A toy rule: hold the market only on the event days, flat otherwise.

    Because each event is an isolated single day, every event is a full round trip
    (enter at prior close, exit at that close) -> two one-way crossings of ``cost_bps``
    per event. Reports total gross and net event PnL (bps) and per-event averages.
    """
    grp = df.loc[df[mask_col], "ret"].to_numpy(dtype=float)
    grp = grp[np.isfinite(grp)]
    n = len(grp)
    gross_total_bps = float(grp.sum() * 1e4)
    cost_total_bps = 2.0 * cost_bps * n  # 2 crossings per isolated event
    net_total_bps = gross_total_bps - cost_total_bps
    return {
        "n_event": int(n),
        "gross_total_bps": gross_total_bps,
        "net_total_bps": net_total_bps,
        "gross_per_event_bps": float(gross_total_bps / n) if n else float("nan"),
        "net_per_event_bps": float(net_total_bps / n) if n else float("nan"),
        "cost_bps_per_leg": float(cost_bps),
    }


# ---------------------------------------------------------------------------
# Robustness — Pi-Day contrast across sub-windows
# ---------------------------------------------------------------------------
def subwindow_sweep(df: pd.DataFrame, edges: list[str]) -> pd.DataFrame:
    """Pi-Day contrast (bps) and Welch t over consecutive sub-windows.

    ``edges`` is a list of date strings defining window boundaries (len n -> n-1
    windows). A real effect keeps its sign across windows; numerology flips.
    """
    rows = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        sub = df[(df.index >= pd.Timestamp(a)) & (df.index < pd.Timestamp(b))]
        if sub.empty:
            continue
        r = pi_day_comparison(sub)
        rows.append({
            "window": f"{a[:4]}-{b[:4]}",
            "n_pi": r["n_event"],
            "contrast_bps": r["contrast_bps"],
            "tstat_welch": r["tstat_welch"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule >= 20 seeds)
# ---------------------------------------------------------------------------
def synthetic_control(constant_effect: float, n_seeds: int = 25,
                      base_seed: int = 542, n_perm: int = 1500) -> dict:
    """Average the constant-day contrast and random-date-set placebo p over seeds.

    For each of ``n_seeds`` synthetic worlds with a planted ``constant_effect``, compute
    the pooled constant-day contrast (bps) and the random-date-set placebo p; return the
    means. At the null (effect = 0) the contrast is ~0 and p is ~uniform (~0.5); a
    planted effect drives the contrast up and the placebo p toward 0.
    """
    contrasts, pvals = [], []
    k = len(_data.CONSTANT_DATES)
    for s in range(base_seed, base_seed + n_seeds):
        dfx, _ = _data.synthetic_daily(constant_effect=constant_effect, seed=s)
        c = constant_day_comparison(dfx)["contrast_bps"]
        p = random_dateset_placebo(dfx, k=k, n_perm=n_perm, seed=s)["pval"]
        contrasts.append(c)
        pvals.append(p)
    return {
        "constant_effect": float(constant_effect),
        "n_seeds": int(n_seeds),
        "mean_contrast_bps": float(np.nanmean(contrasts)),
        "mean_placebo_p": float(np.nanmean(pvals)),
    }


# ---------------------------------------------------------------------------
# Summarize — the headline dict mirroring docs/results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_perm: int = 5000, seed: int = 542) -> dict:
    """All headline numbers for a real (or synthetic) tape in one flat dict."""
    pi = pi_day_comparison(df)
    const = constant_day_comparison(df)
    plac = random_dateset_placebo(df, k=len(_data.CONSTANT_DATES), n_perm=n_perm, seed=seed)
    sweep = per_constant_sweep(df)
    row_pi = sweep[sweep["constant"] == "pi"].iloc[0]
    return {
        "n_days": int(len(df)),
        "n_pi": pi["n_event"],
        "mean_pi_bps": pi["mean_event_bps"],
        "mean_rest_bps": pi["mean_rest_bps"],
        "pi_contrast_bps": pi["contrast_bps"],
        "pi_tstat_welch": pi["tstat_welch"],
        "pi_pval_welch": pi["pval_welch"],
        "pi_tstat_hac": pi["tstat_hac"],
        "n_constant": const["n_event"],
        "constant_contrast_bps": const["contrast_bps"],
        "constant_tstat_welch": const["tstat_welch"],
        "constant_pval_welch": const["pval_welch"],
        "placebo_p": plac["pval"],
        "placebo_obs_contrast_bps": plac["obs_contrast_bps"],
        "pi_pval_bonferroni": float(row_pi["pval_bonferroni"]),
        "top_constant": str(sweep.iloc[0]["constant"]),
        "top_pval_raw": float(sweep.iloc[0]["pval_raw"]),
        "top_pval_bonferroni": float(sweep.iloc[0]["pval_bonferroni"]),
        "sweep": sweep,
    }
