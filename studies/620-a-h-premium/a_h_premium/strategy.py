"""Strategy + inference for Study 620 — A-H Share Premium.

The claim: **the same company costs ~30% more in Shanghai (A-share, CNY) than in Hong Kong
(H-share, HKD) — and no one can arbitrage it away**, because A and H lines are non-fungible
(you cannot convert one into the other) and mainland A-shares cannot be shorted by the
investors who see the spread. Four measurements:

    * **Level & persistence.** Per-pair FX-adjusted premium ``P_A / (P_H × HKD→CNY) − 1``
      from raw closes (the Hang Seng AH Premium index convention), equal-weight across pairs,
      monthly. Mean level with a Newey-West HAC *t*, share of months positive, per-pair means.

    * **Non-convergence.** AR(1) persistence of the equal-weight monthly premium: phi,
      half-life, and a Dickey-Fuller unit-root test (Δp on lagged p; 5% critical value −2.86)
      — is the premium statistically distinguishable from a random walk at all?

    * **Co-movement.** Correlation structure of monthly premium *changes* across pairs: mean
      pairwise correlation and the first principal component's variance share — is the A-H
      premium one market-wide factor rather than 14 idiosyncratic mispricings?

    * **The fade trade.** Does a WIDE premium predict the H line out-earning the A line?
      (a) cross-sectional: long H / short A in the widest-premium half of pairs, the reverse
      in the narrowest half, monthly, signal from month-end *t*, held month *t+1* (the ONE
      documented execution lag); random half-split baseline averaged over ≥ 20 seeds;
      (b) unconditional carry: always long H / short A, all pairs — this is also the third
      axis' H-vs-A total-return race. All relative returns use dividend-adjusted (total-return)
      closes with the H leg converted to CNY, so the H line's higher dividend yield is counted.

Inference: Newey-West HAC *t* everywhere a mean of a serially-correlated series is quoted;
the premium level uses lag 12 (months), return races lag 6. No single-seed baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

MONTHS = 12.0
DF_CRIT_5PCT = -2.86  # Dickey-Fuller 5% critical value, constant, large n (Fuller 1976 tab.)


# --------------------------------------------------------------------------- #
# HAC (Newey-West) t of a mean
# --------------------------------------------------------------------------- #
def nw_tstat(x: np.ndarray | pd.Series, lags: int = 6) -> tuple[float, float, float]:
    """(mean, HAC standard error, t) of the mean of ``x`` with Newey-West (Bartlett) weights."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    return float(mu), float(se), float(mu / se)


# --------------------------------------------------------------------------- #
# Premium construction
# --------------------------------------------------------------------------- #
def premium_panel(close: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """Daily FX-adjusted premium per pair: ``P_A / (P_H × HKD→CNY) − 1`` from RAW closes.

    A and H lines are 1:1-par ordinary shares with identical per-share dividends, so the raw
    price ratio (H converted into CNY) is the whole story — this is the Hang Seng AH Premium
    index construction, pair by pair. Rows where either leg is missing are NaN.
    """
    pairs = pairs or data.PAIRS
    fx = data.hkd_cny(close)
    out = {}
    for name, a, h in pairs:
        out[name] = close[a] / (close[h] * fx) - 1.0
    return pd.DataFrame(out)


def monthly_premium(prem_daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end premium per pair (last daily observation of each complete month)."""
    return prem_daily.resample("ME").last()


def level_stats(prem_m: pd.DataFrame, lags: int = 12) -> dict:
    """Equal-weight premium level: mean, HAC t, persistence, per-pair means."""
    ew = prem_m.mean(axis=1).dropna()
    mu, se, t = nw_tstat(ew, lags=lags)
    per_pair = prem_m.mean().sort_values(ascending=False)
    return {
        "ew": ew,
        "mean_pct": mu * 100.0,
        "se_pct": se * 100.0,
        "hac_t": t,
        "share_pos": float((ew > 0).mean()),
        "n_months": int(len(ew)),
        "per_pair_pct": per_pair * 100.0,
        "n_pairs_pos": int((per_pair > 0).sum()),
        "n_pairs": int(len(per_pair)),
        "last_pct": float(ew.iloc[-1] * 100.0),
        "min_pct": float(ew.min() * 100.0),
        "max_pct": float(ew.max() * 100.0),
    }


# --------------------------------------------------------------------------- #
# Non-convergence: AR(1) + Dickey-Fuller
# --------------------------------------------------------------------------- #
def ar1_stats(p: pd.Series) -> dict:
    """AR(1) fit of a monthly premium series + Dickey-Fuller unit-root test.

    OLS ``p_t = a + phi p_{t-1}``: phi, se(phi), half-life = ln(0.5)/ln(phi) in months.
    DF: OLS ``Δp_t = a + b p_{t-1}``; t(b) against the 5% critical value −2.86 (constant,
    large n). Failing to reject means the tape cannot distinguish the premium from a random
    walk — the statistical face of "no arbitrage channel".
    """
    p = p.dropna()
    y, x = p.values[1:], p.values[:-1]
    n = len(y)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    phi = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + phi * (x - xm))
    se_phi = float(np.sqrt((resid @ resid) / (n - 2) / sxx))
    hl = float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else float("inf")
    # Dickey-Fuller: delta on lagged level
    d = y - x
    b = float(((x - xm) * (d - d.mean())).sum() / sxx)
    resid_df = d - (d.mean() + b * (x - xm))
    se_b = float(np.sqrt((resid_df @ resid_df) / (n - 2) / sxx))
    return {"phi": phi, "se_phi": se_phi, "half_life_m": hl,
            "df_t": b / se_b, "df_crit": DF_CRIT_5PCT, "n": n}


def per_pair_half_lives(prem_m: pd.DataFrame, min_months: int = 60) -> pd.Series:
    """Half-life (months) of each pair's own monthly premium, AR(1)."""
    out = {}
    for c in prem_m.columns:
        s = prem_m[c].dropna()
        if len(s) < min_months:
            continue
        out[c] = ar1_stats(s)["half_life_m"]
    return pd.Series(out).sort_values()


# --------------------------------------------------------------------------- #
# Co-movement: one factor?
# --------------------------------------------------------------------------- #
def comovement(prem_m: pd.DataFrame) -> dict:
    """Mean pairwise correlation + PC1 variance share of monthly premium CHANGES."""
    chg = prem_m.diff().dropna(how="all")
    chg = chg.dropna(axis=0)                       # complete months across all pairs
    corr = chg.corr()
    k = corr.shape[0]
    off = (corr.values.sum() - k) / (k * (k - 1))
    evals = np.linalg.eigvalsh(corr.values)
    pc1 = float(evals[-1] / evals.sum())
    return {"mean_pair_corr": float(off), "pc1_share": pc1,
            "n_pairs": k, "n_months": int(len(chg))}


# --------------------------------------------------------------------------- #
# Relative returns (total-return, common currency) + the fade trade
# --------------------------------------------------------------------------- #
def monthly_rel_returns(adj: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """Monthly H-minus-A total-return difference per pair, both legs in CNY.

    H leg: adjusted HKD close × HKD→CNY (so the FX move is part of the H return, as it is
    for a mainland or global investor comparing the two lines in one currency). A leg:
    adjusted CNY close. Both dividend-adjusted — the H line's higher cash yield is counted.
    """
    pairs = pairs or data.PAIRS
    fx = data.hkd_cny(adj)
    out = {}
    for name, a, h in pairs:
        h_cny = (adj[h] * fx).resample("ME").last()
        a_cny = adj[a].resample("ME").last()
        out[name] = h_cny.pct_change() - a_cny.pct_change()
    return pd.DataFrame(out)


def monthly_rel_returns_price_only(close: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """Same H-minus-A monthly difference from RAW closes (price-only, no dividends) — used to
    decompose the fade into premium convergence vs the H line's higher cash yield."""
    pairs = pairs or data.PAIRS
    fx = data.hkd_cny(close)
    out = {}
    for name, a, h in pairs:
        h_cny = (close[h] * fx).resample("ME").last()
        a_cny = close[a].resample("ME").last()
        out[name] = h_cny.pct_change() - a_cny.pct_change()
    return pd.DataFrame(out)


def carry_stats(rel_m: pd.DataFrame, lags: int = 6) -> dict:
    """Unconditional always-long-H / short-A carry, equal-weight across pairs (also the
    third-axis H-vs-A race). Mean monthly diff + HAC t + cumulative wealth ratio."""
    ew = rel_m.mean(axis=1).dropna()
    mu, se, t = nw_tstat(ew, lags=lags)
    wealth = float((1.0 + ew).prod())
    yrs = len(ew) / MONTHS
    return {"mean_bps": mu * 1e4, "hac_t": t, "n_months": int(len(ew)),
            "ann_pct": ((1.0 + mu) ** MONTHS - 1.0) * 100.0,
            "wealth_ratio": wealth, "years": yrs,
            "share_pos": float((ew > 0).mean())}


def leg_race(adj: pd.DataFrame, pairs=None, lags: int = 6) -> dict:
    """Equal-weight H basket vs A basket, total-return, both in CNY: annualised legs +
    HAC t of the monthly difference (rebalanced monthly; pairs enter when both legs exist)."""
    pairs = pairs or data.PAIRS
    fx = data.hkd_cny(adj)
    h_rets, a_rets = {}, {}
    for name, a, h in pairs:
        h_rets[name] = (adj[h] * fx).resample("ME").last().pct_change()
        a_rets[name] = adj[a].resample("ME").last().pct_change()
    h_m = pd.DataFrame(h_rets).mean(axis=1).dropna()
    a_m = pd.DataFrame(a_rets).mean(axis=1).dropna()
    common = h_m.index.intersection(a_m.index)
    h_m, a_m = h_m[common], a_m[common]
    d = h_m - a_m
    mu, se, t = nw_tstat(d, lags=lags)
    yrs = len(common) / MONTHS
    h_ann = ((1.0 + h_m).prod()) ** (1.0 / yrs) - 1.0
    a_ann = ((1.0 + a_m).prod()) ** (1.0 / yrs) - 1.0
    return {"h_ann_pct": h_ann * 100.0, "a_ann_pct": a_ann * 100.0,
            "diff_bps_mo": mu * 1e4, "hac_t": t, "years": yrs,
            "n_months": int(len(common)),
            "h_wealth": float((1.0 + h_m).prod()), "a_wealth": float((1.0 + a_m).prod())}


def fade_cross_section(prem_m: pd.DataFrame, rel_m: pd.DataFrame,
                       cost_bps_oneway: float = 0.0, lags: int = 6) -> dict:
    """Cross-sectional fade: long H / short A in the WIDEST-premium half of pairs, the reverse
    in the narrowest half. Signal = month-end *t* premium; position held month *t+1* (one
    documented lag). Equal weight inside each half; portfolio return = mean(wide-half rel)
    − mean(narrow-half rel), halved (0.5 long spread, 0.5 short spread → 1× gross notional
    per side). Costs: ``cost_bps_oneway`` × monthly one-way turnover (bounded by 2 sides ×
    membership churn; charged as 4 legs × churn fraction, conservative).
    """
    common = prem_m.index.intersection(rel_m.index)
    prem, rel = prem_m.loc[common], rel_m.loc[common]
    rets, churns = [], []
    prev_wide: set = set()
    idx = []
    for i in range(len(common) - 1):
        row = prem.iloc[i].dropna()
        nxt = rel.iloc[i + 1]
        if len(row) < 6:
            continue
        med = row.median()
        wide = set(row[row > med].index)
        narrow = set(row[row <= med].index)
        rw = nxt[list(wide)].dropna()
        rn = nxt[list(narrow)].dropna()
        if len(rw) < 2 or len(rn) < 2:
            continue
        gross = 0.5 * (rw.mean() - rn.mean())
        churn = 1.0 if not prev_wide else (len(wide ^ prev_wide) / max(1, len(wide) * 2))
        prev_wide = wide
        # 4 legs move when a pair switches halves (close 2 spreads, open 2)
        cost = (cost_bps_oneway / 1e4) * 4.0 * churn
        rets.append(gross - cost)
        churns.append(churn)
        idx.append(common[i + 1])
    r = pd.Series(rets, index=idx)
    mu, se, t = nw_tstat(r, lags=lags)
    return {"ret": r, "mean_bps": mu * 1e4, "hac_t": t, "n_months": int(len(r)),
            "ann_pct": ((1.0 + mu) ** MONTHS - 1.0) * 100.0,
            "avg_churn": float(np.mean(churns)) if churns else float("nan")}


def fade_random_baseline(prem_m: pd.DataFrame, rel_m: pd.DataFrame,
                         n_seeds: int = 20, lags: int = 6) -> dict:
    """Random-half placebo for the cross-sectional fade, AVERAGED over ``n_seeds`` seeds
    (single-seed baselines are banned). Each seed re-draws the wide/narrow split at random
    every month; the mean monthly return and t are averaged across seeds."""
    common = prem_m.index.intersection(rel_m.index)
    prem, rel = prem_m.loc[common], rel_m.loc[common]
    means, ts = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(620 + seed)
        rets = []
        for i in range(len(common) - 1):
            row = prem.iloc[i].dropna()
            nxt = rel.iloc[i + 1]
            if len(row) < 6:
                continue
            names = list(row.index)
            rng.shuffle(names)
            half = len(names) // 2
            rw = nxt[names[:half]].dropna()
            rn = nxt[names[half:]].dropna()
            if len(rw) < 2 or len(rn) < 2:
                continue
            rets.append(0.5 * (rw.mean() - rn.mean()))
        mu, _se, t = nw_tstat(np.asarray(rets), lags=lags)
        means.append(mu * 1e4)
        ts.append(t)
    return {"mean_bps": float(np.mean(means)), "mean_t": float(np.mean(ts)),
            "n_seeds": n_seeds}


def fade_time_series(prem_m: pd.DataFrame, rel_m: pd.DataFrame,
                     window: int = 36, min_obs: int = 24, lags: int = 6) -> dict:
    """Time-series fade: for each pair, long H / short A ONLY in months following a month-end
    premium above the pair's own trailing ``window``-month median (wide vs own history);
    flat otherwise. Equal-weight across active pairs; one-month lag."""
    common = prem_m.index.intersection(rel_m.index)
    prem, rel = prem_m.loc[common], rel_m.loc[common]
    med = prem.rolling(window, min_periods=min_obs).median()
    sig = (prem > med)          # wide vs own history at month-end t
    pos = sig.shift(1)          # held during month t+1
    active = pos & rel.notna()
    port = rel.where(active).mean(axis=1).dropna()
    mu, se, t = nw_tstat(port, lags=lags)
    return {"mean_bps": mu * 1e4, "hac_t": t, "n_months": int(len(port)),
            "avg_active": float(active.sum(axis=1).reindex(port.index).mean())}


def feasible_leg(prem_m: pd.DataFrame, rel_m: pd.DataFrame,
                 cost_bps_oneway: float = 0.0, borrow_bps_yr: float = 0.0,
                 lags: int = 6) -> dict:
    """The only HALF of the fade an outside investor can actually execute: **long A / short H**
    in the NARROWEST-premium half of pairs (buy A northbound via Stock Connect, borrow + short
    the H line in Hong Kong). The other half — long H / *short A* — has no channel: A and H are
    non-fungible and mainland A-shares are effectively unborrowable for foreigners (and, since
    the 2024 CSRC relending suspension, for most locals). One-month lag as everywhere. Costs:
    ``cost_bps_oneway`` × 4 legs × monthly membership churn + ``borrow_bps_yr``/12 on the 1×
    short-H notional."""
    common = prem_m.index.intersection(rel_m.index)
    prem, rel = prem_m.loc[common], rel_m.loc[common]
    rets, churns, idx = [], [], []
    prev: set = set()
    for i in range(len(common) - 1):
        row = prem.iloc[i].dropna()
        nxt = rel.iloc[i + 1]
        if len(row) < 6:
            continue
        med = row.median()
        narrow = set(row[row <= med].index)
        rn = nxt[list(narrow)].dropna()
        if len(rn) < 2:
            continue
        churn = 1.0 if not prev else (len(narrow ^ prev) / max(1, len(narrow) * 2))
        prev = narrow
        gross = -rn.mean()                       # long A / short H = minus (H − A)
        cost = (cost_bps_oneway / 1e4) * 4.0 * churn + (borrow_bps_yr / 1e4) / MONTHS
        rets.append(gross - cost)
        churns.append(churn)
        idx.append(common[i + 1])
    r = pd.Series(rets, index=idx)
    mu, se, t = nw_tstat(r, lags=lags)
    return {"ret": r, "mean_bps": mu * 1e4, "hac_t": t, "n_months": int(len(r)),
            "ann_pct": ((1.0 + mu) ** MONTHS - 1.0) * 100.0,
            "avg_churn": float(np.mean(churns)) if churns else float("nan")}


# --------------------------------------------------------------------------- #
# Monte-Carlo null placebo for the LEVEL (the honest near-unit-root inference)
# --------------------------------------------------------------------------- #
def null_mc_level(ew: pd.Series, n_sims: int = 4000, seed: int = 6200,
                  phi_override: float | None = None, rw: bool = False) -> dict:
    """How often does a ZERO-mean wandering premium average as much as the observed one?

    HAC *t* on a level with AR(1) phi ≈ 0.98 badly understates the standard error (our own
    synthetic control shows 26/40 zero-mean seeds faking |HAC t| ≥ 2), so the level claim gets
    a Monte-Carlo placebo instead: simulate ``n_sims`` premium paths with **mean zero** and the
    tape's own persistence and innovation volatility (AR(1) fit on the equal-weight premium;
    ``phi_override`` for stress values; ``rw=True`` = driftless random walk anchored at the
    tape's actual 2010 starting level). p = share of null paths whose sample mean ≥ the
    observed mean, and whose share-of-positive-months ≥ the observed share.
    """
    ew = ew.dropna()
    n = len(ew)
    ar = ar1_stats(ew)
    phi = float(phi_override) if phi_override is not None else ar["phi"]
    y, x = ew.values[1:], ew.values[:-1]
    resid = (y - y.mean()) - ar["phi"] * (x - x.mean())
    sig = float(resid.std(ddof=2))
    obs_mean, obs_share = float(ew.mean()), float((ew > 0).mean())
    rng = np.random.default_rng(seed)
    hits_m = hits_s = 0
    for _ in range(n_sims):
        e = rng.normal(0.0, sig, n)
        p = np.empty(n)
        if rw:
            p[0] = float(ew.iloc[0])
            for t in range(1, n):
                p[t] = p[t - 1] + e[t]
        else:
            p[0] = e[0] / np.sqrt(1.0 - phi ** 2)
            for t in range(1, n):
                p[t] = phi * p[t - 1] + e[t]
        if p.mean() >= obs_mean:
            hits_m += 1
        if (p > 0).mean() >= obs_share:
            hits_s += 1
    return {"p_mean": hits_m / n_sims, "p_share": hits_s / n_sims,
            "phi": phi, "sig": sig, "n_sims": n_sims,
            "obs_mean_pct": obs_mean * 100.0, "obs_share": obs_share}


# --------------------------------------------------------------------------- #
# Synthetic control suites (machinery proof — never market evidence)
# --------------------------------------------------------------------------- #
def control_level_suite(plant: float, n_seeds: int = 40) -> dict:
    """Seed-averaged level recovery: EW premium mean across ``n_seeds`` synthetic worlds with
    a planted premium level, plus (the teaching point) how many ZERO-mean worlds fake an
    |HAC(12) t| ≥ 2 on the level."""
    means, fake_t = [], 0
    for s in range(n_seeds):
        w = data.synthetic_world(premium=plant, seed=620 + s)
        ew = w["prem"].mean(axis=1)
        m, _se, t = nw_tstat(ew, lags=12)
        means.append(m)
        if abs(t) >= 2.0:
            fake_t += 1
    means = np.asarray(means)
    return {"plant": plant, "avg_mean_pct": float(means.mean() * 100.0),
            "sd_pct": float(means.std() * 100.0),
            "max_abs_pct": float(np.abs(means).max() * 100.0),
            "n_ge_real": int((np.abs(means) >= 0.297).sum()),
            "n_hac_t_ge2": fake_t, "n_seeds": n_seeds}


def control_fade_suite(phi: float, n_seeds: int = 10) -> dict:
    """Seed-averaged fade detector: the cross-sectional fade must fire in a mean-reverting
    premium world (phi < 1) and stay flat in a random-walk premium world (phi = 1)."""
    ts, mus = [], []
    for s in range(n_seeds):
        w = data.synthetic_world(premium=0.30, phi=phi, seed=700 + s)
        f = fade_cross_section(w["prem"], w["rel"])
        ts.append(f["hac_t"])
        mus.append(f["mean_bps"])
    return {"phi": phi, "avg_bps": float(np.mean(mus)), "avg_t": float(np.mean(ts)),
            "min_t": float(np.min(ts)), "max_t": float(np.max(ts)), "n_seeds": n_seeds}


def predictive_slope(prem_m: pd.DataFrame, rel_m: pd.DataFrame, lags: int = 6) -> dict:
    """Pooled predictive check as a portfolio: monthly return = cross-pair mean of
    z_i(t) × rel_i(t+1), where z is the cross-sectional z-score of the premium at month-end t.
    A positive HAC t says wide premia predict H out-earning A (the fade 'should' work)."""
    common = prem_m.index.intersection(rel_m.index)
    prem, rel = prem_m.loc[common], rel_m.loc[common]
    rets, idx = [], []
    for i in range(len(common) - 1):
        row = prem.iloc[i].dropna()
        nxt = rel.iloc[i + 1]
        if len(row) < 6 or row.std() == 0:
            continue
        z = (row - row.mean()) / row.std()
        r = (z * nxt[row.index]).dropna()
        if len(r) < 4:
            continue
        rets.append(r.mean())
        idx.append(common[i + 1])
    r = pd.Series(rets, index=idx)
    mu, se, t = nw_tstat(r, lags=lags)
    return {"mean_bps": mu * 1e4, "hac_t": t, "n_months": int(len(r))}
