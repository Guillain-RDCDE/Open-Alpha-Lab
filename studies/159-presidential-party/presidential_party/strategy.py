"""Strategy and analysis layer for Study 159 (Presidential-Party).

The folk claim: stocks do substantially better under Democratic presidents than under
Republican ones. Santa-Clara & Valkanov (2003, *Journal of Finance*) found a gap of
roughly **1,200 bps/year** in *excess* real returns over 1927-1998 — one of the most
cited and most cited-in-disbelief results in the calendar-anomaly literature.

We test it with three honest tools:

1. **Party comparison** — mean monthly return (log) under D vs R, with a Welch t-test
   on the tiny sample and a HAC t-stat (Newey-West) on each party's own series.
2. **Baseline / unconditional control** — compare both party means against the *overall*
   unconditional mean (are we actually detecting something, or just finding that stocks
   go up?). The right question is "D minus R", not "D minus zero".
3. **Permutation test** — randomly permute the party labels (keeping their block
   structure intact) and ask: how often does a random party assignment produce a gap as
   large as the one we observe? This is the multiple-comparisons / small-n honest check.

The small-n problem is structural and non-negotiable: with ~17 presidential terms since
1927, we have roughly 8 "D" data-points at the *term* level (even though they span ~600
monthly observations). The HAC t-stat on monthly returns is useful, but the overlap
between administrations and business cycles means it does not certify a *causal* party
effect no matter how large. The study says so loudly and honestly.

Four analysis functions:
- :func:`party_returns` — mean monthly returns per party for a DataFrame with ``ret`` and
  ``party`` columns.
- :func:`dem_minus_rep` — the headline scalar: D mean minus R mean, with HAC t-stats and
  an annualised return differential.
- :func:`permutation_pvalue` — label-permutation p-value on the D-minus-R gap.
- :func:`summarize` — one-stop compact dict for the R-dict in notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HAC t-stat helper (Newey-West, Bartlett kernel)
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the sample mean.

    Uses the standard Bartlett-kernel long-run variance estimator with the
    rule-of-thumb lag count ``floor(4 * (n/100)^(2/9))``. For very small n
    (< 6) falls back to a plain OLS standard error.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if n < 6:
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Party-stratified statistics
# ---------------------------------------------------------------------------
def party_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per party ('D', 'R') from a monthly return DataFrame.

    ``df`` must have columns ``ret`` (monthly log-return) and ``party`` ('D' or 'R').
    Returns a DataFrame indexed by party with columns:
    ``n``, ``mean_monthly_bps``, ``std_monthly_bps``, ``mean_ann_pct``, ``hac_tstat``.
    """
    rows = []
    for party in ("D", "R"):
        vals = df.loc[df["party"] == party, "ret"].dropna().values.astype(float)
        n = len(vals)
        mean_m = float(vals.mean()) if n > 0 else float("nan")
        std_m = float(vals.std(ddof=1)) if n > 1 else float("nan")
        # Annualise: compound monthly log-return * 12 (approximate), then convert to pct
        mean_ann = float(mean_m * 12 * 100) if np.isfinite(mean_m) else float("nan")
        tstat = _hac_tstat(vals)
        rows.append({
            "party": party,
            "n": n,
            "mean_monthly_bps": round(mean_m * 1e4, 2),
            "std_monthly_bps": round(std_m * 1e4, 2),
            "mean_ann_pct": round(mean_ann, 2),
            "hac_tstat": round(tstat, 3),
        })
    return pd.DataFrame(rows).set_index("party")


# ---------------------------------------------------------------------------
# Headline: D-minus-R gap, with honest Welch t and annualised differential
# ---------------------------------------------------------------------------
def dem_minus_rep(df: pd.DataFrame) -> dict:
    """The headline D-minus-R gap in monthly returns, with inference.

    Uses a Welch t-test (unequal-variance) on the *monthly* return series — not
    annual aggregates — which gives the most power possible on this tiny dataset.
    The HAC t-stat on the *difference series* (obtained by subtracting the grand mean
    and computing the signed party indicator regression) is also reported; it is the
    most honest single number for autocorrelation-robust inference.

    Returns a dict with: gap_monthly_bps, gap_ann_pct, welch_t, hac_t, n_dem, n_rep,
    n_terms_dem, n_terms_rep.
    """
    d_vals = df.loc[df["party"] == "D", "ret"].dropna().values.astype(float)
    r_vals = df.loc[df["party"] == "R", "ret"].dropna().values.astype(float)
    n_d, n_r = len(d_vals), len(r_vals)

    gap_m = float(d_vals.mean() - r_vals.mean()) if (n_d > 0 and n_r > 0) else float("nan")
    gap_ann = float(gap_m * 12 * 100) if np.isfinite(gap_m) else float("nan")

    # Welch t on monthly observations
    if n_d >= 2 and n_r >= 2:
        se_welch = np.sqrt(d_vals.var(ddof=1) / n_d + r_vals.var(ddof=1) / n_r)
        welch_t = float(gap_m / se_welch) if se_welch > 0 else float("nan")
    else:
        welch_t = float("nan")

    # HAC t-stat on the party-coded return series:
    # x_t = ret_t * indicator(party=="D") - mean(D) is not the right thing;
    # instead construct a signed series: +ret under D, -ret under R,
    # so its mean = (mean_D - mean_R)/2, then scale.
    # A cleaner approach: regress ret on (party == "D") via OLS and use HAC SE.
    x = (df["party"] == "D").astype(float).values
    y = df["ret"].values.astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    x, y = x[mask], y[mask]
    n_ols = len(x)
    if n_ols >= 10:
        # OLS slope for party_D dummy: beta = Cov(x,y)/Var(x)
        xm = x - x.mean()
        ym = y - y.mean()
        beta = float(xm @ ym) / float(xm @ xm) if float(xm @ xm) > 0 else float("nan")
        resid = y - (x.mean() * beta + (y.mean() - x.mean() * beta))  # residuals
        # HAC SE for beta
        e = resid
        lags = int(np.floor(4.0 * (n_ols / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n_ols
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n_ols
        # SE for beta = sqrt(lrv / Var(x) / n)
        var_x = float(xm @ xm) / n_ols
        se_beta = np.sqrt(max(lrv, 0.0) / n_ols) / var_x if var_x > 0 else float("nan")
        hac_t = float(beta / se_beta) if np.isfinite(se_beta) and se_beta > 0 else float("nan")
    else:
        hac_t = float("nan")

    # Count distinct terms (not months)
    n_terms_d = df.loc[df["party"] == "D", "president"].nunique() if "president" in df.columns else float("nan")
    n_terms_r = df.loc[df["party"] == "R", "president"].nunique() if "president" in df.columns else float("nan")

    return {
        "gap_monthly_bps": round(gap_m * 1e4, 2),
        "gap_ann_pct": round(gap_ann, 2),
        "welch_t": round(welch_t, 3),
        "hac_t": round(hac_t, 3),
        "n_dem": int(n_d),
        "n_rep": int(n_r),
        "n_terms_dem": int(n_terms_d) if not isinstance(n_terms_d, float) else n_terms_d,
        "n_terms_rep": int(n_terms_r) if not isinstance(n_terms_r, float) else n_terms_r,
    }


# ---------------------------------------------------------------------------
# Permutation test — is the gap real or a lucky arrangement of the labels?
# ---------------------------------------------------------------------------
def permutation_pvalue(
    df: pd.DataFrame,
    n_perm: int = 10_000,
    seed: int = 159,
) -> dict:
    """Block-permutation p-value on the D-minus-R monthly return gap.

    We permute the party labels *block-by-block* (keeping each presidential term as an
    intact block) rather than observation-by-observation. This preserves the temporal
    clustering structure — business cycles, crises, booms all tend to be administration-
    length in duration, so permuting at the term level is the only honest randomisation.

    With ~17 terms, there are only 17! / (9! * 8!) ≈ 24,310 distinct label arrangements
    (choosing which 9 terms are "D"). We use 10,000 random draws with replacement — a
    Monte Carlo approximation — and report the two-sided p-value.

    Returns: observed_gap_bps, pvalue_two_sided, n_perm, n_terms.
    """
    rng = np.random.default_rng(seed)

    # Build a list of (party, returns) per term
    if "president" not in df.columns:
        # Fallback: group by contiguous party blocks
        blocks = []
        prev_party = None
        block_rets: list[float] = []
        for party, ret in zip(df["party"].values, df["ret"].values):
            if party != prev_party:
                if block_rets:
                    blocks.append((prev_party, np.array(block_rets, dtype=float)))
                block_rets = []
                prev_party = party
            if np.isfinite(ret):
                block_rets.append(ret)
        if block_rets:
            blocks.append((prev_party, np.array(block_rets, dtype=float)))
    else:
        # Group by president (a single named term)
        blocks = []
        for pres, grp in df.groupby("president", sort=False):
            party = grp["party"].iloc[0]
            rets = grp["ret"].dropna().values.astype(float)
            if len(rets) > 0:
                blocks.append((party, rets))

    # Filter out NaN-party blocks
    blocks = [(p, r) for p, r in blocks if p in ("D", "R")]
    n_terms = len(blocks)
    if n_terms < 4:
        return {
            "observed_gap_bps": float("nan"),
            "pvalue_two_sided": float("nan"),
            "n_perm": n_perm,
            "n_terms": n_terms,
        }

    labels = np.array([p for p, _ in blocks])
    rets_per_block = [r for _, r in blocks]

    # Observed gap
    d_rets = np.concatenate([r for p, r in zip(labels, rets_per_block) if p == "D"])
    r_rets = np.concatenate([r for p, r in zip(labels, rets_per_block) if p == "R"])
    obs_gap = float(d_rets.mean() - r_rets.mean()) if (len(d_rets) > 0 and len(r_rets) > 0) else float("nan")

    # Permutation distribution
    n_d = int((labels == "D").sum())
    perm_gaps = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        d_r = np.concatenate([r for p, r in zip(perm_labels, rets_per_block) if p == "D"])
        r_r = np.concatenate([r for p, r in zip(perm_labels, rets_per_block) if p == "R"])
        perm_gaps[i] = float(d_r.mean() - r_r.mean()) if (len(d_r) > 0 and len(r_r) > 0) else 0.0

    pval = float((np.abs(perm_gaps) >= abs(obs_gap)).mean())
    return {
        "observed_gap_bps": round(obs_gap * 1e4, 2),
        "pvalue_two_sided": round(pval, 4),
        "n_perm": int(n_perm),
        "n_terms": n_terms,
        "n_d_terms": int(n_d),
        "n_r_terms": int(n_terms - n_d),
    }


# ---------------------------------------------------------------------------
# One-stop summary for notebooks and docs/results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Compact summary dict for a party-labelled monthly return DataFrame.

    Computes per-party statistics, the D-minus-R headline gap, and the
    permutation p-value. Returns a flat dict suitable for the R-dict in notebooks.
    """
    pr = party_returns(df)
    gap = dem_minus_rep(df)
    perm = permutation_pvalue(df)

    overall_mean = float(df["ret"].dropna().mean() * 1e4)

    out = {
        "n_total": int(df["ret"].notna().sum()),
        "n_dem": gap["n_dem"],
        "n_rep": gap["n_rep"],
        "n_terms_dem": gap.get("n_terms_dem", float("nan")),
        "n_terms_rep": gap.get("n_terms_rep", float("nan")),
        # Democrat stats
        "dem_mean_bps": pr.loc["D", "mean_monthly_bps"],
        "dem_mean_ann": pr.loc["D", "mean_ann_pct"],
        "dem_hac_t": pr.loc["D", "hac_tstat"],
        # Republican stats
        "rep_mean_bps": pr.loc["R", "mean_monthly_bps"],
        "rep_mean_ann": pr.loc["R", "mean_ann_pct"],
        "rep_hac_t": pr.loc["R", "hac_tstat"],
        # Unconditional baseline
        "overall_mean_bps": round(overall_mean, 2),
        # Headline gap
        "gap_monthly_bps": gap["gap_monthly_bps"],
        "gap_ann_pct": gap["gap_ann_pct"],
        "welch_t": gap["welch_t"],
        "hac_t": gap["hac_t"],
        # Permutation test
        "perm_pvalue": perm["pvalue_two_sided"],
        "n_perm": perm["n_perm"],
        "n_terms": perm["n_terms"],
    }
    return out
