"""Strategy and analysis layer for Study 300 (Sports-Sentiment).

The Edmans, Garcia & Norli (2007) "loss effect": the mean next-trading-day return
after a national-team elimination loss should be significantly NEGATIVE. We test
this directly:

  - The **mean next-day return** across all elimination events, in bps.
  - A **HAC (Newey-West) t-statistic** for H0: mean = 0. With at most one event
    per day this is essentially the OLS-on-a-constant t-stat, but the
    Newey-West correction guards against any residual clustering (e.g. two
    countries eliminated on the same tournament day moving together).
  - A **permutation / sign-flip test**: the loss-effect hypothesis is one-sided
    (returns should be negative), so we also report the bootstrap fraction of
    resampled means that are >= 0.
  - **Sub-group cuts**: penalty-shootout losses (the most painful, where Edmans
    finds the effect strongest) vs non-shootout, and World-Cup vs other.

The honest bar (repo rubric): a REAL signal needs |HAC t| >= 2 on the REAL tape.
Literature support alone is only WEAK.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newey-West HAC standard error for the mean of a series
# ---------------------------------------------------------------------------
def hac_mean_tstat(x: np.ndarray, lags: int = 1) -> dict:
    """HAC (Newey-West) t-stat for H0: E[x] = 0.

    Regresses ``x`` on a constant; the OLS coefficient is the sample mean, and
    the Newey-West long-run variance gives a heteroskedasticity- and
    autocorrelation-consistent standard error. With ``lags=0`` this reduces to
    the ordinary (White) mean t-stat.

    Returns dict with ``mean``, ``se``, ``t``, ``n``, ``lags``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "n": int(n), "lags": int(lags)}
    mu = float(x.mean())
    e = x - mu  # residuals from regression on a constant
    # Long-run variance via Bartlett kernel.
    gamma0 = float(np.dot(e, e) / n)
    lrv = gamma0
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gk = float(np.dot(e[k:], e[:-k]) / n)
        lrv += 2.0 * w * gk
    lrv = max(lrv, 0.0)
    se = float(np.sqrt(lrv / n))
    t = mu / se if se > 0 else float("nan")
    return {"mean": mu, "se": se, "t": float(t), "n": int(n), "lags": int(L)}


# ---------------------------------------------------------------------------
# Core loss-effect test
# ---------------------------------------------------------------------------
def loss_effect_stats(
    df: pd.DataFrame,
    ret_col: str = "next_day_ret",
    hac_lags: int = 1,
    n_boot: int = 10_000,
    seed: int = 300,
) -> dict:
    """Test the next-day loss effect on a panel of elimination events.

    Parameters
    ----------
    df : DataFrame with a ``ret_col`` column of next-day simple returns.
    hac_lags : Newey-West lag truncation.
    n_boot : i.i.d. bootstrap resamples for the mean distribution.
    seed : RNG seed.

    Returns a dict with the mean (bps), HAC t-stat, plain (i.i.d.) t-stat, the
    one-sided bootstrap p (fraction of resampled means >= 0), the fraction of
    negative events, and the down/up count.
    """
    r = df[ret_col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size

    hac = hac_mean_tstat(r, lags=hac_lags)
    mean_bps = hac["mean"] * 1e4

    # Plain i.i.d. t-stat (for comparison).
    sd = r.std(ddof=1)
    plain_t = float(r.mean() / (sd / np.sqrt(n))) if sd > 0 and n > 1 else float("nan")

    # i.i.d. bootstrap of the mean -> one-sided p (loss effect: mean < 0).
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(r, size=(n_boot, n), replace=True).mean(axis=1)
    boot_p_ge0 = float((boot_means >= 0).mean())  # one-sided: chance mean is non-negative

    frac_neg = float((r < 0).mean())
    n_down = int((r < 0).sum())

    return {
        "n": int(n),
        "mean_bps": round(float(mean_bps), 2),
        "hac_t": round(float(hac["t"]), 3),
        "hac_se_bps": round(float(hac["se"] * 1e4), 2),
        "hac_lags": int(hac["lags"]),
        "plain_t": round(float(plain_t), 3),
        "boot_p_ge0": round(float(boot_p_ge0), 4),
        "frac_neg": round(frac_neg, 4),
        "n_down": n_down,
        "n_up": int(n - n_down),
    }


# ---------------------------------------------------------------------------
# Sub-group summary
# ---------------------------------------------------------------------------
def subgroup_summary(
    df: pd.DataFrame,
    ret_col: str = "next_day_ret",
    hac_lags: int = 1,
    seed: int = 300,
) -> pd.DataFrame:
    """Mean and HAC t for the full sample and key sub-groups.

    Sub-groups (only computed if the columns exist): shootout vs non-shootout,
    World-Cup vs non-World-Cup. Returns one row per group.
    """
    rows = []

    def _row(label, sub):
        rr = sub[ret_col].to_numpy(dtype=float)
        rr = rr[np.isfinite(rr)]
        if rr.size < 2:
            return None
        h = hac_mean_tstat(rr, lags=hac_lags)
        return {
            "group": label,
            "n": int(rr.size),
            "mean_bps": round(float(h["mean"] * 1e4), 2),
            "hac_t": round(float(h["t"]), 3),
            "frac_neg": round(float((rr < 0).mean()), 3),
        }

    rows.append(_row("All eliminations", df))
    if "shootout" in df.columns:
        rows.append(_row("Penalty-shootout losses", df[df["shootout"] == True]))   # noqa: E712
        rows.append(_row("Non-shootout losses", df[df["shootout"] == False]))      # noqa: E712
    if "tournament" in df.columns:
        rows.append(_row("World Cup only", df[df["tournament"] == "WC"]))
        rows.append(_row("Euro / Copa", df[df["tournament"] != "WC"]))
    return pd.DataFrame([r for r in rows if r is not None])


# ---------------------------------------------------------------------------
# Tradability: a naive short-on-loss strategy, gross and net of costs
# ---------------------------------------------------------------------------
def tradable_pnl(
    df: pd.DataFrame,
    ret_col: str = "next_day_ret",
    cost_bps_oneway: float = 5.0,
) -> dict:
    """PnL of going SHORT the country ETF for the single next session per loss.

    The 'strategy' is: at each elimination, short the country's ETF at the next
    open and cover at the next close (one session of exposure). PnL per event is
    ``-next_day_ret``. Round-trip cost = 2 * one-way (shorts pay borrow + spread;
    single-country ETFs are wide). We report gross and net mean PnL in bps and the
    annualised tally (events cluster in ~6-week tournament windows every 2 years,
    so the strategy is idle most of the time -- a capacity/utilisation problem).

    Returns gross/net mean bps, total events, and the net hit-rate.
    """
    r = df[ret_col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    pnl_gross = -r  # short
    rt_cost = 2.0 * cost_bps_oneway * 1e-4
    pnl_net = pnl_gross - rt_cost
    return {
        "n_events": int(r.size),
        "gross_mean_bps": round(float(pnl_gross.mean() * 1e4), 2),
        "net_mean_bps": round(float(pnl_net.mean() * 1e4), 2),
        "net_total_bps": round(float(pnl_net.sum() * 1e4), 2),
        "net_hit_rate": round(float((pnl_net > 0).mean()), 4),
        "cost_bps_oneway": cost_bps_oneway,
    }


# ---------------------------------------------------------------------------
# Summarize -- compact dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, ret_col: str = "next_day_ret",
              hac_lags: int = 1, n_boot: int = 10_000, seed: int = 300) -> dict:
    """One-stop summary mirroring the R dict in build_notebooks.py / results.md."""
    s = loss_effect_stats(df, ret_col=ret_col, hac_lags=hac_lags,
                          n_boot=n_boot, seed=seed)
    tr = tradable_pnl(df, ret_col=ret_col)
    out = dict(s)
    out.update({
        "gross_mean_bps": tr["gross_mean_bps"],
        "net_mean_bps": tr["net_mean_bps"],
        "net_hit_rate": tr["net_hit_rate"],
    })
    return out
