"""The analysis and its honest controls — Study 297 (March-Madness).

The claim: while America watches the NCAA basketball tournament, the market is
distracted — returns sag, volume thins, the index drifts lower for ~3 weeks every
March. The test mirrors the desk's standard distraction/seasonality teardown:

  (A) Mean daily return INSIDE the tournament window vs OUTSIDE (Newey-West HAC
      t-stat — the bar a REAL signal must clear is |t| >= 2 on the real tape).
  (B) Realised volatility inside vs outside (variance ratio + Levene test):
      the "madness causes chaos" corollary.
  (C) A tradable "avoidance" strategy: hold cash during the tournament window,
      stay long the index otherwise — with an explicit ONE-DAY execution lag and
      one-way transaction costs charged on every switch (entry into and exit out
      of cash). Gross vs net are both reported.
  (D) A block-permutation control: how often does a *randomly placed* ~6% slice
      of trading days (matching the tournament's calendar footprint) look as bad
      as the real tournament window?

Honesty notes baked in:
  - Returns are price-only (no dividends) — labelled on the Signal axis.
  - Survivorship is not an issue here (single broad index), but the index has an
    upward drift; sitting out ~6% of days mechanically costs ~6% of the drift,
    which the strategy must overcome to add value.
  - Execution lag: the strategy enters/exits cash one trading day AFTER the window
    boundary, so it never trades on information it would not yet have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics helper
# ---------------------------------------------------------------------------
def _hac_tstat(arr: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the sample mean."""
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 6:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# A: Mean return — in-tournament vs outside
# ---------------------------------------------------------------------------
def tournament_return_test(ret: pd.Series, mask: pd.Series) -> dict:
    """Compare mean daily return on in-tournament vs outside trading days.

    Returns a dict with: ``n_in``, ``n_out``, ``in_mean_bps``, ``out_mean_bps``,
    ``diff_bps`` (in minus out), ``in_t`` (HAC t on in-tournament days),
    ``out_t``, ``welch_t`` (two-sample Welch t on the difference of means).

    The folklore predicts ``diff_bps < 0`` with a meaningful |welch_t|.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)

    in_r = r[m].to_numpy()
    out_r = r[~m].to_numpy()
    n_in, n_out = len(in_r), len(out_r)

    mean_in = float(in_r.mean()) if n_in else float("nan")
    mean_out = float(out_r.mean()) if n_out else float("nan")
    diff = mean_in - mean_out
    se_welch = (
        np.sqrt(in_r.var(ddof=1) / n_in + out_r.var(ddof=1) / n_out)
        if n_in > 1 and n_out > 1
        else float("nan")
    )

    return {
        "n_in": int(n_in),
        "n_out": int(n_out),
        "in_mean_bps": float(mean_in * 1e4),
        "out_mean_bps": float(mean_out * 1e4),
        "diff_bps": float(diff * 1e4),
        "in_t": _hac_tstat(in_r),
        "out_t": _hac_tstat(out_r),
        "welch_t": float(diff / se_welch)
        if np.isfinite(se_welch) and se_welch > 0
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# B: Volatility test — variance ratio between regimes
# ---------------------------------------------------------------------------
def tournament_vol_test(ret: pd.Series, mask: pd.Series) -> dict:
    """Compare realised daily volatility inside vs outside the tournament window.

    Returns ``in_vol_ann``, ``out_vol_ann``, ``vol_ratio`` (in/out), and a Levene
    variance-equality test (stat + p). A vol_ratio > 1 means the tournament window
    is *more* volatile, which is the "madness = chaos" corollary.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)

    in_r = r[m].to_numpy()
    out_r = r[~m].to_numpy()
    n_in, n_out = len(in_r), len(out_r)

    var_in = float(np.var(in_r, ddof=1)) if n_in > 1 else float("nan")
    var_out = float(np.var(out_r, ddof=1)) if n_out > 1 else float("nan")
    vol_in = np.sqrt(var_in * 252)
    vol_out = np.sqrt(var_out * 252)
    ratio = var_in / var_out if var_out > 0 else float("nan")

    from scipy import stats as scipy_stats

    try:
        lev_stat, lev_p = scipy_stats.levene(in_r, out_r)
    except Exception:
        lev_stat, lev_p = float("nan"), float("nan")

    return {
        "n_in": int(n_in),
        "n_out": int(n_out),
        "in_vol_ann": float(vol_in),
        "out_vol_ann": float(vol_out),
        "vol_ratio": float(ratio),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
    }


# ---------------------------------------------------------------------------
# C: Strategy — tournament-avoidance vs buy-and-hold (lagged, with costs)
# ---------------------------------------------------------------------------
def tournament_avoidance(
    ret: pd.Series,
    mask: pd.Series,
    lag: int = 1,
    cost_bps: float = 1.0,
) -> dict:
    """Hold cash during the tournament window, long the index otherwise.

    The most generous framing of the distraction claim: if in-tournament days are
    truly bearish, stepping aside should beat buy-and-hold.

    Honesty:
      - ``lag`` trading days of execution delay: the position for day *t* is the
        tournament label of day *t - lag* (default 1), so we never trade on
        same-day information.
      - ``cost_bps`` one-way transaction cost (in bps of NAV) charged on every
        switch between long and cash — i.e. on entry to cash and exit from cash.
        This is the honest drag a real implementation pays.

    Returns gross and net CAGR for both legs, the excess (net) CAGR, the number of
    round-trips, and the HAC t on the daily (net) excess return.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)
    n = len(r)
    years = n / 252.0

    # Position: 0 (cash) on lagged in-tournament days, 1 (long) otherwise.
    in_lagged = m.shift(lag).fillna(False).astype(bool)
    pos = (~in_lagged).astype(float)  # 1.0 long, 0.0 cash

    # Switches: a transaction whenever the position changes (entry/exit cash).
    switches = pos.diff().abs().fillna(0.0)
    n_switches = int((switches > 0).sum())
    cost = switches * (cost_bps * 1e-4)  # one-way cost per switch, NAV fraction

    strat_gross = pos * r
    strat_net = strat_gross - cost

    bah_cum = float((1 + r).prod())
    sg_cum = float((1 + strat_gross).prod())
    sn_cum = float((1 + strat_net).prod())

    bah_cagr = float(bah_cum ** (1 / years) - 1)
    sg_cagr = float(sg_cum ** (1 / years) - 1)
    sn_cagr = float(sn_cum ** (1 / years) - 1)

    excess_net = (strat_net - r).to_numpy()

    return {
        "n": int(n),
        "years": float(years),
        "n_switches": n_switches,
        "bah_cagr": float(bah_cagr),
        "strat_gross_cagr": float(sg_cagr),
        "strat_net_cagr": float(sn_cagr),
        "excess_gross_cagr": float(sg_cagr - bah_cagr),
        "excess_net_cagr": float(sn_cagr - bah_cagr),
        "excess_net_hac_t": _hac_tstat(excess_net),
        "bah_ret": r,
        "strat_ret": strat_net,
    }


# ---------------------------------------------------------------------------
# D: Permutation control — random-period baseline
# ---------------------------------------------------------------------------
def permutation_null(
    ret: pd.Series,
    mask: pd.Series,
    n_perm: int = 5000,
    seed: int = 297,
) -> dict:
    """How often does a randomly placed ~6% of trading days look as bad as the window?

    We draw random subsets of the same size as the real in-tournament set and ask
    what fraction produce a mean daily return at least as low as the actual
    in-tournament mean. If the distraction claim were real, the observed mean
    should sit in the low tail of these random controls.

    Returns ``in_mean_bps``, ``pct_worse_or_equal`` (the one-sided p-value),
    ``n_perm``, ``n_in``, ``frac_in``.
    """
    r = pd.Series(ret).astype(float).dropna()
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)
    in_mean = float(r[m].mean())

    rng = np.random.default_rng(seed)
    n_total = len(r)
    n_in = int(m.sum())
    vals = r.to_numpy()

    count = 0
    for _ in range(n_perm):
        idx = rng.choice(n_total, size=n_in, replace=False)
        if float(vals[idx].mean()) <= in_mean:
            count += 1

    return {
        "in_mean_bps": float(in_mean * 1e4),
        "pct_worse_or_equal": float(count / n_perm),
        "n_perm": int(n_perm),
        "n_in": int(n_in),
        "frac_in": float(n_in / n_total) if n_total else float("nan"),
    }


# ---------------------------------------------------------------------------
# Summary: all tests in one call
# ---------------------------------------------------------------------------
def summarize(
    ret: pd.Series,
    mask: pd.Series,
    n_perm: int = 5000,
    seed: int = 297,
) -> dict:
    """Run all four tests and return a combined results dict — the study's spine."""
    rtest = tournament_return_test(ret, mask)
    vtest = tournament_vol_test(ret, mask)
    avoid = tournament_avoidance(ret, mask)
    perm = permutation_null(ret, mask, n_perm=n_perm, seed=seed)

    r = pd.Series(ret).astype(float).dropna()
    n = len(r)
    years = n / 252.0
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    cagr = float((1 + r).prod() ** (1 / years) - 1)

    return {
        # Full-tape headline
        "n": int(n),
        "years": float(years),
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(252)),
        "sharpe": float(mean / std * np.sqrt(252)) if std > 0 else float("nan"),
        # Return test
        **{f"ret_{k}": v for k, v in rtest.items()},
        # Vol test
        **{f"vol_{k}": v for k, v in vtest.items()},
        # Avoidance strategy
        "avoid_bah_cagr": avoid["bah_cagr"],
        "avoid_strat_gross_cagr": avoid["strat_gross_cagr"],
        "avoid_strat_net_cagr": avoid["strat_net_cagr"],
        "avoid_excess_net_cagr": avoid["excess_net_cagr"],
        "avoid_excess_net_t": avoid["excess_net_hac_t"],
        "avoid_n_switches": avoid["n_switches"],
        # Permutation control
        "perm_pct_worse_or_equal": perm["pct_worse_or_equal"],
        "perm_n_in": perm["n_in"],
        "perm_frac_in": perm["frac_in"],
    }
