"""Measurement + inference for Study 948 — the income-fund capture scorecard.

The claim: an income fund (covered-call or high-dividend) "gives up a little upside to
avoid a lot of downside". Stated properly that is a claim about **capture ratios**,
measured on monthly total returns against the benchmark the fund is marketed against:

* **up-capture**   = mean(fund return | benchmark month > 0) / mean(benchmark return | > 0)
* **down-capture** = mean(fund return | benchmark month < 0) / mean(benchmark return | < 0)
* **capture spread** = up-capture − down-capture.

A fund that is simply a low-beta version of its index — beta ``b``, **no alpha** — has
up-capture = down-capture = ``b`` and a spread of **zero**. Low beta is not skill, it is a
smaller position. A positive spread is supposed to mean the fund is convex: it keeps more
of the good months than of the bad ones.

**The caveat the fact sheets never print:** that "spread = 0" holds only at zero alpha. A
*perfectly linear* fund with intercept ``a`` has spread ``a·(1/mean_up − 1/mean_down)``,
and since ``mean_down < 0`` the two terms **add** — so any positive alpha reads as
convexity, with no curvature anywhere in the payoff. The capture spread cannot separate
"bends upward" from "earns more"; only a fit with a free intercept can. That is why the
inference in this study rests on :func:`hm_regression`, not on the spread, and it is why
:func:`matched_null_spread` reports both an alpha-free and an alpha-carrying null.

Three layers of evidence, in increasing strictness:

1. **The descriptive scorecard** — capture ratios and the spread per fund, with a circular
   block bootstrap CI on the spread (the ratio is a ratio of means, so its sampling
   distribution is skewed and a textbook standard error would lie).
2. **The parametric twin** — a Henriksson-Merton piecewise regression on *excess-of-cash*
   returns, ``e_f = a + b_up·max(e_b,0) + b_dn·min(e_b,0) + u``. The convexity
   coefficient ``b_up − b_dn`` is the capture spread's regression twin, it is testable
   with a HAC (Newey-West) *t*, and — unlike the raw ratio — it does not blow up when a
   denominator approaches zero. Because fourteen funds are screened at once, the positive
   side is judged against a **Bonferroni** bar, not a nominal one.
3. **Does the scorecard travel?** An era cut and a Spearman rank-persistence test: if the
   capture spread is a property of a fund rather than of a sample, this era's ranking
   should predict the next era's.

And one tradability check: the **beta-neutral traded spread**. Long the fund, short
``beta_hat`` of the benchmark, the remainder in cash. This is the only arm that trades, so
it is the only arm with an execution lag and with costs:

* **ONE execution lag, and only here** — ``beta_hat`` is estimated on the rolling 36-month
  window ending at month ``t`` and held over month ``t+1``. Capture ratios themselves are
  a contemporaneous *attribution* of the same month's returns: nothing is traded on them,
  so no lag applies (and quoting one would be theatre).
* **Costs** are one-way basis points × NAV on the traded notional, and the **short
  benchmark leg pays borrow** — both are swept.

All returns are **monthly simple total returns** (``auto_adjust=True`` closes), never
price-only: an income fund pays most of its return out as distributions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
HAC_LAGS = 6          # ~half a year of monthly autocorrelation


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain iid t of mean(x) vs 0 (kept for the null-vs-HAC comparison)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int = HAC_LAGS) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def hac_ols(y, X, lags: int = HAC_LAGS):
    """OLS with a Newey-West covariance matrix. Returns ``(beta, cov)``.

    ``X`` must already carry its own intercept column if one is wanted.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    z = resid[:, None] * X
    s = z.T @ z
    n = len(y)
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        a = z[lag:].T @ z[:-lag]
        s = s + w * (a + a.T)
    return beta, xtx_inv @ s @ xtx_inv


def spearman(a, b) -> tuple:
    """Spearman rank correlation and its large-sample two-sided p (no scipy dependency).

    Ties get average ranks; the p-value uses the usual ``t = rho·sqrt((n-2)/(1-rho²))``
    approximation with a normal tail, which is plenty for the n ≈ 10 rank-persistence
    test this study runs (and is reported alongside n so nobody over-reads it).
    """
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1.0 - rho * rho))
    p = float(math.erfc(abs(t) / math.sqrt(2.0)))   # two-sided normal tail
    return rho, p


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni critical value (normal) for ``n_tests`` simultaneous tests.

    Screening fourteen funds at a nominal |t| ≥ 2 gives roughly a one-in-two chance of at
    least one false positive under the null. The family-wise bar is what a fund screen has
    to clear, and it is what this study quotes.
    """
    if n_tests < 1:
        return float("nan")
    # Solve Phi(z) = 1 - alpha/(2 n_tests) by bisection — exact enough and stdlib-only.
    target = 1.0 - alpha / (2.0 * n_tests)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# The descriptive scorecard
# --------------------------------------------------------------------------- #
def capture_ratios(r_fund: pd.Series, r_bench: pd.Series) -> dict:
    """Arithmetic up/down-capture and their spread, on monthly **total** returns.

    Months are partitioned by the *benchmark's* sign (a fund never gets to choose which
    months count as good ones). Flat benchmark months (exactly zero) are excluded from
    both legs. The arithmetic ratio-of-means convention is used throughout;
    :func:`geometric_capture_ratios` gives the compounding variant as a robustness check.

    Returns ``up``, ``down``, ``spread`` (= up − down), the two month counts, and the two
    denominators — the denominators matter, because a down-capture computed against a
    near-zero mean down month is an unstable number and the reader deserves to see it.
    """
    idx = r_fund.dropna().index.intersection(r_bench.dropna().index)
    f, b = r_fund.loc[idx], r_bench.loc[idx]
    up_m, dn_m = b > 0, b < 0
    den_up = float(b[up_m].mean()) if up_m.sum() else float("nan")
    den_dn = float(b[dn_m].mean()) if dn_m.sum() else float("nan")
    up = float(f[up_m].mean()) / den_up if up_m.sum() and den_up != 0 else float("nan")
    dn = float(f[dn_m].mean()) / den_dn if dn_m.sum() and den_dn != 0 else float("nan")
    return {
        "up": up, "down": dn, "spread": up - dn,
        "n_up": int(up_m.sum()), "n_down": int(dn_m.sum()), "n_months": int(len(idx)),
        "den_up": den_up, "den_down": den_dn,
    }


def geometric_capture_ratios(r_fund: pd.Series, r_bench: pd.Series) -> dict:
    """Morningstar's compounding variant: ratio of annualised up-month (down-month) returns.

    Same partition, but each leg is geometrically compounded and annualised before the
    ratio is taken. Reported as a convention check — if a headline flips between the
    arithmetic and geometric definitions it was never a finding.
    """
    idx = r_fund.dropna().index.intersection(r_bench.dropna().index)
    f, b = r_fund.loc[idx], r_bench.loc[idx]
    out = {}
    for tag, mask in (("up", b > 0), ("down", b < 0)):
        n = int(mask.sum())
        if n < 2:
            out[tag] = float("nan")
            continue
        gf = float((1.0 + f[mask]).prod() ** (MONTHS_PER_YEAR / n) - 1.0)
        gb = float((1.0 + b[mask]).prod() ** (MONTHS_PER_YEAR / n) - 1.0)
        out[tag] = gf / gb if gb != 0 else float("nan")
    out["spread"] = out["up"] - out["down"]
    return out


def bootstrap_capture_spread(
    r_fund: pd.Series,
    r_bench: pd.Series,
    n_boot: int = 2000,
    block: int = 3,
    seed: int = 948,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the capture **spread**.

    Fund and benchmark are resampled **jointly** in blocks of ``block`` consecutive months
    (default one quarter), so the pairing and the local vol clustering survive. Draws in
    which either leg has fewer than five months are discarded. Returns the point spread,
    the percentile interval, and ``frac_positive`` — the share of resamples with a
    positive spread, i.e. a blunt one-sided read on "could this fund really be convex?".
    """
    idx = r_fund.dropna().index.intersection(r_bench.dropna().index)
    f = r_fund.loc[idx].to_numpy(dtype=float)
    b = r_bench.loc[idx].to_numpy(dtype=float)
    n = len(f)
    point = capture_ratios(r_fund, r_bench)["spread"]
    if n < 3 * block:
        return {"spread": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_positive": float("nan"), "n_obs": n, "n_boot": 0, "block": block}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        take = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        fb, bb = f[take], b[take]
        up, dn = bb > 0, bb < 0
        if up.sum() < 5 or dn.sum() < 5:
            continue
        du, dd = bb[up].mean(), bb[dn].mean()
        if du == 0 or dd == 0:
            continue
        draws.append(fb[up].mean() / du - fb[dn].mean() / dd)
    arr = np.asarray(draws, dtype=float)
    if arr.size < 100:
        return {"spread": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_positive": float("nan"), "n_obs": n, "n_boot": int(arr.size),
                "block": block}
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"spread": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_positive": float((arr > 0).mean()), "n_obs": n,
            "n_boot": int(arr.size), "block": block}


# --------------------------------------------------------------------------- #
# The parametric twin — Henriksson-Merton convexity, and plain CAPM alpha
# --------------------------------------------------------------------------- #
def hm_regression(e_fund: pd.Series, e_bench: pd.Series, lags: int = HAC_LAGS) -> dict:
    """Henriksson-Merton piecewise fit on **excess-of-cash** monthly returns.

    ``e_f = a + b_up·max(e_b, 0) + b_dn·min(e_b, 0) + u``. The **convexity**
    ``b_up − b_dn`` is the capture spread's well-behaved regression twin, and its HAC *t*
    is the study's inference bar. The intercept ``a`` is the fund's excess return **at a
    flat benchmark month** — for an option-writer that is the premium it collects, and it
    is emphatically *not* a CAPM alpha (a kinked fit pushes the intercept up mechanically).
    Use :func:`capm_alpha` for the alpha question.
    """
    idx = e_fund.dropna().index.intersection(e_bench.dropna().index)
    ef, eb = e_fund.loc[idx].to_numpy(float), e_bench.loc[idx].to_numpy(float)
    n = len(ef)
    if n < 12:
        return {k: float("nan") for k in
                ("b_up", "b_dn", "convexity", "t_convexity", "kink_intercept_bps",
                 "t_kink_intercept")} | {"n_months": n}
    x = np.column_stack([np.ones(n), np.maximum(eb, 0.0), np.minimum(eb, 0.0)])
    beta, cov = hac_ols(ef, x, lags=lags)
    c = np.array([0.0, 1.0, -1.0])
    var = float(c @ cov @ c)
    se = np.sqrt(var) if var > 0 else float("nan")
    conv = float(beta[1] - beta[2])
    return {
        "b_up": float(beta[1]), "b_dn": float(beta[2]),
        "convexity": conv,
        "t_convexity": conv / se if se and np.isfinite(se) else float("nan"),
        "kink_intercept_bps": float(beta[0] * 1e4),
        "t_kink_intercept": float(beta[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else float("nan"),
        "n_months": n,
    }


def capm_alpha(e_fund: pd.Series, e_bench: pd.Series, lags: int = HAC_LAGS) -> dict:
    """Single-beta market model on excess-of-cash returns: alpha (bps/month) and beta.

    The question the capture scorecard is usually a proxy for — "is this fund adding
    anything beyond a smaller position in the index?" — answered directly, with a HAC *t*.
    """
    idx = e_fund.dropna().index.intersection(e_bench.dropna().index)
    ef, eb = e_fund.loc[idx].to_numpy(float), e_bench.loc[idx].to_numpy(float)
    n = len(ef)
    if n < 12:
        return {"alpha_bps": float("nan"), "t_alpha": float("nan"),
                "beta": float("nan"), "n_months": n}
    x = np.column_stack([np.ones(n), eb])
    beta, cov = hac_ols(ef, x, lags=lags)
    return {
        "alpha_bps": float(beta[0] * 1e4),
        "t_alpha": float(beta[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else float("nan"),
        "beta": float(beta[1]),
        "n_months": n,
    }


# --------------------------------------------------------------------------- #
# Excess-of-cash performance summary (monthly simple returns)
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised stats for a monthly simple-return series.

    Pass an **excess-of-cash** series to get the excess Sharpe — every Sharpe race in this
    study is excess-of-cash on both sides.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {k: float("nan") for k in
                ("cagr", "sharpe", "vol_ann", "max_drawdown", "mean_bps", "tstat")} | {"n_months": n}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_months": int(n),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The per-fund scorecard and the cross-sectional panel
# --------------------------------------------------------------------------- #
def score_fund(
    r_fund: pd.Series,
    r_bench: pd.Series,
    r_cash: pd.Series,
    n_boot: int = 2000,
    seed: int = 948,
) -> dict:
    """Everything the study measures for one fund against one benchmark.

    Capture ratios (arithmetic and geometric), the bootstrap CI on the spread, the HM
    convexity with its HAC *t*, the CAPM alpha/beta, and the excess-of-cash Sharpe race
    (fund vs benchmark) with a HAC *t* on the monthly return difference.
    """
    idx = (r_fund.dropna().index
           .intersection(r_bench.dropna().index)
           .intersection(r_cash.dropna().index))
    f, b, c = r_fund.loc[idx], r_bench.loc[idx], r_cash.loc[idx]
    ef, eb = (f - c).rename("e_fund"), (b - c).rename("e_bench")

    cap = capture_ratios(f, b)
    geo = geometric_capture_ratios(f, b)
    boot = bootstrap_capture_spread(f, b, n_boot=n_boot, seed=seed)
    hm = hm_regression(ef, eb)
    cm = capm_alpha(ef, eb)
    s_f, s_b = summary(ef), summary(eb)

    return {
        "n_months": int(len(idx)),
        "start": str(idx[0].date()) if len(idx) else "",
        "end": str(idx[-1].date()) if len(idx) else "",
        "up": cap["up"], "down": cap["down"], "spread": cap["spread"],
        "n_up": cap["n_up"], "n_down": cap["n_down"],
        "den_up": cap["den_up"], "den_down": cap["den_down"],
        "geo_up": geo["up"], "geo_down": geo["down"], "geo_spread": geo["spread"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "frac_positive": boot["frac_positive"],
        "convexity": hm["convexity"], "t_convexity": hm["t_convexity"],
        "kink_intercept_bps": hm["kink_intercept_bps"],
        "beta": cm["beta"], "alpha_bps": cm["alpha_bps"], "t_alpha": cm["t_alpha"],
        "sharpe_fund": s_f["sharpe"], "sharpe_bench": s_b["sharpe"],
        "sharpe_gap": s_f["sharpe"] - s_b["sharpe"],
        "t_sharpe_gap": newey_west_t((ef - eb).to_numpy()),
        "dd_fund": s_f["max_drawdown"], "dd_bench": s_b["max_drawdown"],
        "e_fund": ef, "e_bench": eb,
    }


def scorecard(
    returns: pd.DataFrame,
    bench_map: dict,
    cash_col: str = "BIL",
    n_boot: int = 2000,
    seed: int = 948,
    force_bench: str = None,
) -> pd.DataFrame:
    """Run :func:`score_fund` across the whole cross-section.

    ``returns`` is a frame of monthly simple total returns; ``bench_map`` names each
    fund's benchmark column. ``force_bench`` overrides the map with a single benchmark for
    every fund — that is the sweep of the benchmark-assignment ASSUMPTION.
    Returns one row per fund, ordered as ``bench_map`` is.
    """
    rows = {}
    keep = ["n_months", "start", "end", "up", "down", "spread", "n_up", "n_down",
            "den_up", "den_down", "geo_up", "geo_down", "geo_spread",
            "ci_low", "ci_high", "frac_positive", "convexity", "t_convexity",
            "kink_intercept_bps", "beta", "alpha_bps", "t_alpha",
            "sharpe_fund", "sharpe_bench", "sharpe_gap", "t_sharpe_gap",
            "dd_fund", "dd_bench"]
    for fund, bench in bench_map.items():
        b = force_bench or bench
        if fund not in returns.columns or b not in returns.columns:
            continue
        s = score_fund(returns[fund], returns[b], returns[cash_col],
                       n_boot=n_boot, seed=seed)
        row = {k: s[k] for k in keep}
        row["bench"] = b
        rows[fund] = row
    df = pd.DataFrame(rows).T
    df.index.name = "fund"
    num = [c for c in df.columns if c not in ("start", "end", "bench")]
    df[num] = df[num].astype(float)
    return df


# --------------------------------------------------------------------------- #
# The fund-matched null — the estimator's bias measured on the REAL tape
# --------------------------------------------------------------------------- #
def matched_null_spread(
    r_fund: pd.Series,
    r_bench: pd.Series,
    r_cash: pd.Series,
    n_sims: int = 1000,
    block: int = 3,
    seed: int = 948,
    include_alpha: bool = False,
) -> dict:
    """The capture spread's null distribution **matched to this fund's own sample**.

    The generic synthetic null (:func:`null_spread_distribution`) proves the estimator is
    biased, but the *size* of that bias belongs to the world it was measured in — months,
    benchmark vol, idiosyncratic vol, alpha. Subtracting a synthetic bias from a real-tape
    spread would be mixing tapes. This function measures the reference distribution **on
    the fund's own data**, conditioning on the benchmark path the spread was computed with:

    1. Fit the fund's *linear* twin on excess-of-cash returns, ``e_f = a + b·e_b + u``.
    2. Rebuild the fund ``n_sims`` times as ``cash + a·[include_alpha] + b·(bench − cash)
       + u*``, where ``u*`` is a circular block bootstrap of the fitted residuals — the
       **real** benchmark path, the **real** cash path, the fund's own beta and residual
       vol, and by construction **zero convexity**.
    3. Score each replicate with the same :func:`capture_ratios` the real table uses.

    Two different nulls, and the difference between them is the study's sharpest point:

    * ``include_alpha=False`` (**the default, and the test**) is the null the sales pitch
      implicitly claims to beat: *a plain scaled index position*, zero alpha, zero
      convexity. ``p_value`` is the share of such replicates whose spread reaches the
      observed one — a one-sided *p* for "more convex than simply owning less of the
      index". This null has power: on the planted synthetic panel the genuinely convex
      fund clears it (*p* = 0.000) and the concave ones are pinned at the far tail.
      **Its known limitation, stated plainly:** it also fires on a fund that is perfectly
      linear but carries a positive alpha, because alpha alone bends the spread. So a
      small *p* here means "beats a plain scaled index", **not** "is convex". Convexity
      as such is tested by :func:`hm_regression`, whose free intercept separates the two;
      this null is reported as the estimator diagnostic it is, and never as the bar.
    * ``include_alpha=True`` is a **decomposition, not a test**. A fund with a positive
      alpha has a mechanically positive capture spread — ``a`` is divided by a positive
      mean up-month and by a *negative* mean down-month, so it lands in the spread twice
      with the same sign — and an OLS line fitted to a genuinely convex fund answers with
      a large positive intercept. That means the alpha-carrying twin reproduces the
      observed spread almost exactly, for convex and flat funds alike. Reported precisely
      to show that the capture spread carries **no information beyond (alpha, beta)**: it
      is a re-encoding of the CAPM fit, not an independent measurement. Never read its
      ``p_value`` as evidence of anything.

    ``excess_spread`` is the observed spread minus ``null_mean``: the bias-corrected number.
    """
    idx = (r_fund.dropna().index
           .intersection(r_bench.dropna().index)
           .intersection(r_cash.dropna().index))
    f, b, c = r_fund.loc[idx], r_bench.loc[idx], r_cash.loc[idx]
    observed = capture_ratios(f, b)["spread"]
    n = len(idx)
    nan_out = {"observed": observed, "null_mean": float("nan"),
               "null_sd": float("nan"), "excess_spread": float("nan"),
               "p_value": float("nan"), "n_sims": 0, "n_months": n}
    if n < 24:
        return nan_out

    ef = (f - c).to_numpy(float)
    eb = (b - c).to_numpy(float)
    x = np.column_stack([np.ones(n), eb])
    coef, *_ = np.linalg.lstsq(x, ef, rcond=None)
    resid = ef - x @ coef
    resid = resid - resid.mean()

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    bench_np, cash_np = b.to_numpy(float), c.to_numpy(float)
    up_m, dn_m = bench_np > 0, bench_np < 0
    den_up = bench_np[up_m].mean() if up_m.sum() else np.nan
    den_dn = bench_np[dn_m].mean() if dn_m.sum() else np.nan
    if not (np.isfinite(den_up) and np.isfinite(den_dn)) or den_up == 0 or den_dn == 0:
        return nan_out

    draws = np.empty(n_sims, dtype=float)
    a = float(coef[0]) if include_alpha else 0.0
    linear = cash_np + a + coef[1] * (bench_np - cash_np)
    for i in range(n_sims):
        starts = rng.integers(0, n, n_blocks)
        take = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sim = linear + resid[take]
        draws[i] = sim[up_m].mean() / den_up - sim[dn_m].mean() / den_dn

    return {
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "excess_spread": float(observed - draws.mean()),
        "p_value": float((draws >= observed).mean()),
        "alpha_bps": float(coef[0] * 1e4),
        "beta": float(coef[1]),
        "n_sims": int(n_sims),
        "n_months": n,
    }


def matched_null_panel(
    returns: pd.DataFrame,
    bench_map: dict,
    cash_col: str = "BIL",
    n_sims: int = 1000,
    seed: int = 948,
    include_alpha: bool = False,
) -> pd.DataFrame:
    """:func:`matched_null_spread` across the cross-section, one row per fund."""
    rows = {}
    for fund, bench in bench_map.items():
        if fund not in returns.columns or bench not in returns.columns:
            continue
        rows[fund] = matched_null_spread(returns[fund], returns[bench],
                                         returns[cash_col], n_sims=n_sims, seed=seed,
                                         include_alpha=include_alpha)
    df = pd.DataFrame(rows).T
    df.index.name = "fund"
    return df.astype(float)


# --------------------------------------------------------------------------- #
# Does the scorecard travel? Era cut + rank persistence
# --------------------------------------------------------------------------- #
def era_spreads(
    returns: pd.DataFrame,
    bench_map: dict,
    split: str,
    min_months: int = 24,
) -> pd.DataFrame:
    """Capture spread computed separately on each side of ``split``.

    Funds without ``min_months`` of history on *both* sides are dropped rather than
    reported on a handful of months — a capture ratio computed on twelve months is a
    rounding error with a decimal point.
    """
    rows = {}
    cut = pd.Timestamp(split)
    for fund, bench in bench_map.items():
        if fund not in returns.columns or bench not in returns.columns:
            continue
        idx = returns[fund].dropna().index.intersection(returns[bench].dropna().index)
        early, late = idx[idx < cut], idx[idx >= cut]
        if len(early) < min_months or len(late) < min_months:
            continue
        a = capture_ratios(returns[fund].loc[early], returns[bench].loc[early])
        b = capture_ratios(returns[fund].loc[late], returns[bench].loc[late])
        rows[fund] = {"bench": bench,
                      "n_early": len(early), "n_late": len(late),
                      "spread_early": a["spread"], "spread_late": b["spread"],
                      "up_early": a["up"], "down_early": a["down"],
                      "up_late": b["up"], "down_late": b["down"]}
    df = pd.DataFrame(rows).T
    df.index.name = "fund"
    return df


def rank_persistence(era: pd.DataFrame) -> dict:
    """Spearman rho between the early-era and late-era capture spreads.

    The scorecard's whole selling point is that it *ranks* funds. If this era's ranking
    carries no information about the next era's, the scorecard is a description of the
    past and nothing more.
    """
    if era is None or len(era) < 3:
        return {"n": 0 if era is None else len(era), "rho": float("nan"), "p": float("nan")}
    rho, p = spearman(era["spread_early"].to_numpy(dtype=float),
                      era["spread_late"].to_numpy(dtype=float))
    sign_agree = float(np.mean(np.sign(era["spread_early"].to_numpy(dtype=float))
                               == np.sign(era["spread_late"].to_numpy(dtype=float))))
    return {"n": int(len(era)), "rho": rho, "p": p, "sign_agreement": sign_agree}


# --------------------------------------------------------------------------- #
# Tradability — the beta-neutral traded spread (the only arm that trades)
# --------------------------------------------------------------------------- #
def traded_beta_neutral(
    r_fund: pd.Series,
    r_bench: pd.Series,
    r_cash: pd.Series,
    window: int = 36,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 100.0,
) -> pd.Series:
    """Long the fund, short ``beta_hat`` of the benchmark, remainder in cash.

    **The one execution lag in this study.** ``beta_hat`` is the OLS slope of the fund's
    excess return on the benchmark's over the ``window``-month window **ending at month
    t**, and it is held over month ``t+1`` — signal formed through ``t``, acted at ``t+1``.

    Frictions: ``cost_bps`` one-way × NAV on the notional actually turned over (the change
    in ``beta_hat`` on the short leg, plus a small standing rebalance of the long leg), and
    ``borrow_bps_ann`` annual borrow on the **absolute short notional** — the short leg is
    a borrowed index position and it is charged as one. Returns the net monthly spread
    return, indexed by month.
    """
    idx = (r_fund.dropna().index
           .intersection(r_bench.dropna().index)
           .intersection(r_cash.dropna().index))
    ef = (r_fund.loc[idx] - r_cash.loc[idx])
    eb = (r_bench.loc[idx] - r_cash.loc[idx])
    cov = ef.rolling(window).cov(eb)
    var = eb.rolling(window).var()
    beta_hat = (cov / var).shift(1)          # <- the lag: through t, applied at t+1
    gross = (ef - beta_hat * eb).dropna()
    if gross.empty:
        return gross
    bh = beta_hat.reindex(gross.index)
    turnover = bh.diff().abs().fillna(bh.abs()) + 0.02   # short-leg drift + long-leg upkeep
    borrow = bh.abs() * (borrow_bps_ann * 1e-4) / MONTHS_PER_YEAR
    return (gross - turnover * cost_bps * 1e-4 - borrow).rename("traded_spread")


def traded_sweep(
    r_fund: pd.Series,
    r_bench: pd.Series,
    r_cash: pd.Series,
    window: int = 36,
    cost_grid=(0.0, 5.0, 25.0),
    borrow_grid=(0.0, 50.0, 100.0, 200.0),
) -> list:
    """Cost × borrow sweep of the beta-neutral traded spread. Gross is ``(0, 0)``."""
    out = []
    for cost in cost_grid:
        for borrow in borrow_grid:
            r = traded_beta_neutral(r_fund, r_bench, r_cash, window=window,
                                    cost_bps=cost, borrow_bps_ann=borrow)
            if len(r) < 12:
                continue
            out.append({"cost_bps": cost, "borrow_bps": borrow,
                        "n_months": int(len(r)),
                        "mean_bps": float(r.mean() * 1e4),
                        "t": newey_west_t(r.to_numpy()),
                        "sharpe": summary(r)["sharpe"]})
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def null_spread_distribution(panels) -> dict:
    """The sampling distribution of the capture spread when the truth is exactly zero.

    ``panels`` is an iterable of ``(returns, truth)`` pairs from
    ``data.synthetic_panel(signal_strength=0.0, ...)`` — worlds in which every fund is
    plain linear beta, so every fund's *true* capture spread is 0.000.

    This is not a formality. The capture spread is a difference of two **ratios of means**,
    and the down-leg denominator (the mean down month) is a small negative number, so the
    down-capture is the noisier of the two and the difference inherits a **positive
    small-sample bias**. Quantifying that bias is what tells a reader whether a mildly
    positive spread on the real tape is evidence of convexity or just the estimator
    breathing. The Henriksson-Merton convexity coefficient is reported alongside precisely
    because it does *not* carry this bias.
    """
    spreads, convex = [], []
    n_panels = 0
    for returns, truth in panels:
        sc = scorecard(returns, truth["bench_map"], cash_col="CASH", n_boot=1)
        spreads.extend(sc["spread"].tolist())
        convex.extend(sc["convexity"].tolist())
        n_panels += 1
    s = np.asarray(spreads, dtype=float)
    c = np.asarray(convex, dtype=float)
    return {
        "n_panels": n_panels, "n_obs": int(s.size),
        "spread_mean": float(s.mean()), "spread_median": float(np.median(s)),
        "spread_sd": float(s.std(ddof=1)), "spread_frac_positive": float((s > 0).mean()),
        "convexity_mean": float(c.mean()), "convexity_median": float(np.median(c)),
        "convexity_sd": float(c.std(ddof=1)),
    }


def synthetic_detect(returns: pd.DataFrame, truth: dict, n_boot: int = 400,
                     seed: int = 948) -> dict:
    """Score a planted synthetic panel and check the machinery recovers the truth.

    On the planted world (``signal_strength = 1``) the measured capture spread and the HM
    convexity must both track the planted half-spread — positive for ``CONVEX``, negative
    for the ``CONCAVE_*`` funds, ~0 for the ``FLAT_*`` ones. On the null
    (``signal_strength = 0``) every planted spread is exactly zero and nothing should fire.
    Returns the per-fund measured-vs-planted table plus the count of |t| ≥ 2 hits.
    """
    sc = scorecard(returns, truth["bench_map"], cash_col="CASH",
                   n_boot=n_boot, seed=seed)
    sc["true_spread"] = [truth["funds"][f]["true_spread"] for f in sc.index]
    sc["err"] = sc["spread"] - sc["true_spread"]
    hits_pos = int((sc["t_convexity"] >= 2.0).sum())
    hits_neg = int((sc["t_convexity"] <= -2.0).sum())
    corr = float(np.corrcoef(sc["spread"].to_numpy(float),
                             sc["true_spread"].to_numpy(float))[0, 1]) \
        if sc["true_spread"].std() > 0 else float("nan")
    return {
        "table": sc,
        "max_abs_err": float(sc["err"].abs().max()),
        "corr_measured_true": corr,
        "n_hits_positive": hits_pos,
        "n_hits_negative": hits_neg,
        "n_funds": int(len(sc)),
    }
