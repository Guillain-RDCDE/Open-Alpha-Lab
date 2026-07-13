"""Strategy + inference for Study 748 (CEO-Age-Effect).

The claim, at literature strength: young CEOs run riskier, more aggressive firms
(Serfling 2014; Yim 2013), so a long-young / short-old equity book earns a spread. We
build the tradable version and judge it the honest way:

1. **The long/short.** Equal-weight the young-CEO names, equal-weight the old-CEO names,
   go long young / short old — a dollar-neutral monthly time series ``ls = young - old``.

2. **HAC inference.** The headline is a **Newey-West (HAC) t** on the mean monthly L/S
   return — the desk's autocorrelation-robust bar. `REAL` needs |t| >= 2 here.

3. **Alpha vs beta — the decisive control.** Regress the L/S on the market (CAPM). If the
   whole "young premium" is just a high-beta growth tilt (young founders' firms ARE
   high-beta), the CAPM **alpha** collapses even when the raw spread looks big. We report
   the alpha, its HAC t, and the book's market beta.

4. **Label-shuffle placebo.** Shuffle the young/old labels across names and recompute the
   L/S HAC t; the p-value is the fraction of shuffles as extreme as observed. The honest
   null for a two-bucket split on ~40 curated names.

5. **Costs + borrow.** The book pays a one-way round-trip on turnover (low — age is nearly
   static) plus an annual borrow on the short (old) leg; gross AND net reported.

6. **A synthetic positive control.** A deterministic world where young firms carry a
   genuine ``age_alpha`` on top of their higher beta; the engine must light up the CAPM
   alpha t and stay flat at the null (alpha = 0, beta tilt only) — averaged over >= 20
   seeds (house rule) so no single lucky seed manufactures significance.

Execution convention: the bucket label (CEO age at the scoring date) is public in advance;
membership known at the close of month *t* earns month *t+1*'s return — one documented
lag, applied once by ``lag_returns``. Prices are dividend-adjusted (total return); the
market leg is SPY total return. The curated table is not survivorship-free and the young
bucket skews to recent IPOs — named on the SIGNAL axis, not buried here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


# ---------------------------------------------------------------------------
# Execution lag
# ---------------------------------------------------------------------------
def lag_returns(ls: pd.Series, lag: int = 1) -> pd.Series:
    """The conservative one-month formation lag: decide the book at the close of month t,
    earn month t+lag's return.

    The bucket label (CEO age at the scoring date) is *calendar-known in advance*, like a
    turn-of-month rule, so a static-membership book needs no lag at all — the contemporaneous
    monthly return IS tradable. This helper exists only to prove that a conservative one-month
    delay changes nothing material (``lag=1`` drops one month and re-aligns; the mean barely
    moves). ``lag=0`` is the identity.
    """
    if lag <= 0:
        return ls.dropna()
    return ls.iloc[lag:].dropna()


# ---------------------------------------------------------------------------
# Newey-West (HAC) inference on a mean
# ---------------------------------------------------------------------------
def _nw_lags(n: int) -> int:
    """Automatic HAC lag (Newey-West 1994 plug-in rule): floor(4 * (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean_t(x: np.ndarray, lags: int | None = None) -> dict:
    """HAC (Newey-West) t-stat that the mean of ``x`` differs from zero.

    Returns the mean, the HAC standard error, the t-stat, n, and the lag used. The HAC
    variance sums the sample variance plus Bartlett-weighted autocovariances, so serially
    correlated monthly returns don't inflate the t.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "n": n, "lags": 0}
    L = _nw_lags(n) if lags is None else lags
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        cov = np.dot(xc[k:], xc[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n) if var > 0 else float("nan")
    t = x.mean() / se if se and se > 0 else float("nan")
    return {"mean": float(x.mean()), "se": float(se), "t": float(t), "n": int(n), "lags": int(L)}


def annualize(mean_monthly: float, vol_monthly: float) -> dict:
    """Annualised return / vol / Sharpe from monthly moments (12x / sqrt(12)x)."""
    ann_ret = mean_monthly * TRADING_MONTHS
    ann_vol = vol_monthly * np.sqrt(TRADING_MONTHS)
    return {"ann_ret": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")}


# ---------------------------------------------------------------------------
# CAPM alpha (the alpha-vs-beta control) with HAC t on the alpha
# ---------------------------------------------------------------------------
def capm_alpha(ls: np.ndarray, mkt: np.ndarray, lags: int | None = None) -> dict:
    """Regress ``ls`` on ``mkt`` (with intercept); HAC t on the alpha and the beta.

    Returns monthly alpha, its HAC t, the market beta, the beta's HAC t, R^2 and n. A big
    raw spread with a small alpha t means the "young premium" is just market/growth beta.
    """
    ls = np.asarray(ls, dtype=float)
    mkt = np.asarray(mkt, dtype=float)
    m = np.isfinite(ls) & np.isfinite(mkt)
    ls, mkt = ls[m], mkt[m]
    n = len(ls)
    if n < 5:
        return {"alpha": float("nan"), "alpha_t": float("nan"), "beta": float("nan"),
                "beta_t": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), mkt])
    beta, *_ = np.linalg.lstsq(X, ls, rcond=None)
    resid = ls - X @ beta
    L = _nw_lags(n) if lags is None else lags
    # HAC covariance of OLS coefficients: (X'X)^-1 (X' Omega X) (X'X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag 0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        Xk = X[k:] * resid[k:, None]
        Xk0 = X[:-k] * resid[:-k, None]
        G = Xk.T @ Xk0
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = np.sum((ls - ls.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(beta[0]), "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else float("nan"),
            "beta": float(beta[1]), "beta_t": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "r2": float(r2), "n": int(n)}


# ---------------------------------------------------------------------------
# Bucket summary — young / old / L/S vs the market
# ---------------------------------------------------------------------------
def bucket_stats(returns: pd.DataFrame) -> dict:
    """Annualised return / vol / Sharpe for young, old, ls and mkt, plus the L/S HAC t."""
    out = {}
    for col in ("young", "old", "ls", "mkt"):
        if col not in returns:
            continue
        s = returns[col].dropna().to_numpy()
        ann = annualize(s.mean(), s.std(ddof=1))
        out[col] = {"mean_m": float(s.mean()), **ann, "n": int(len(s))}
    out["ls"]["hac"] = hac_mean_t(returns["ls"].dropna().to_numpy())
    return out


# ---------------------------------------------------------------------------
# Label-shuffle placebo — the two-bucket null
# ---------------------------------------------------------------------------
def placebo_pvalue(prices: pd.DataFrame, data_mod, n_perm: int = 2000, seed: int = 748,
                   score_date: str | None = None) -> dict:
    """Label-shuffle placebo p-value for the L/S HAC |t|.

    Shuffle the young/old bucket labels across the curated names, rebuild the equal-weight
    L/S each time, and recompute the HAC t. The p-value is the fraction of shuffles whose
    |t| is at least the observed |t|. This is the honest null for the two-bucket split.
    """
    from .data import SCORE_DATE, YOUNG_MAX_AGE
    sd = score_date or SCORE_DATE
    ages = data_mod.curated_ages(sd, YOUNG_MAX_AGE)
    rets = data_mod.monthly_returns(prices)
    names = [t for t in ages.index if t in rets.columns]
    n_young = int((ages.loc[names, "bucket"] == "young").sum())
    if n_young < 2 or len(names) - n_young < 2:
        return {"p": float("nan"), "obs_t": float("nan")}
    R = rets[names]
    mkt = rets[data_mod.BENCHMARK]
    # observed
    yl = ages.loc[names, "bucket"].eq("young").to_numpy()
    obs_ls = (R.loc[:, yl].mean(axis=1) - R.loc[:, ~yl].mean(axis=1)).dropna()
    obs_t = abs(hac_mean_t(obs_ls.to_numpy())["t"])
    rng = np.random.default_rng(seed)
    cnt = 0
    m = len(names)
    for _ in range(n_perm):
        perm = rng.permutation(m)
        mask = np.zeros(m, dtype=bool)
        mask[perm[:n_young]] = True
        ls = (R.loc[:, mask].mean(axis=1) - R.loc[:, ~mask].mean(axis=1)).dropna()
        t = abs(hac_mean_t(ls.to_numpy())["t"])
        if np.isfinite(t) and np.isfinite(obs_t) and t >= obs_t - 1e-12:
            cnt += 1
    return {"p": (cnt + 1) / (n_perm + 1), "obs_t": float(obs_t)}


# ---------------------------------------------------------------------------
# Costs + borrow on the L/S book
# ---------------------------------------------------------------------------
def net_of_costs(returns: pd.DataFrame, cost_bps: float = 5.0, borrow_ann_bps: float = 75.0,
                 annual_turnover: float = 0.30) -> dict:
    """Net the L/S book for one-way trading costs (turnover x NAV) and a short-leg borrow.

    ``annual_turnover`` is one-way turnover of the book (age is nearly static, so this is
    low — a name only moves buckets when a CEO changes). Costs are charged on the two legs;
    the borrow is charged on the short (old) leg. Returns annualised gross and net L/S
    return and the resulting Sharpe.
    """
    s = returns["ls"].dropna().to_numpy()
    gross_ann = s.mean() * TRADING_MONTHS
    ann_vol = s.std(ddof=1) * np.sqrt(TRADING_MONTHS)
    cost = 2.0 * (cost_bps * 1e-4) * annual_turnover     # two legs, one-way x NAV
    borrow = borrow_ann_bps * 1e-4                        # short leg financing
    net_ann = gross_ann - cost - borrow
    return {"gross_ann": float(gross_ann), "net_ann": float(net_ann),
            "gross_sharpe": float(gross_ann / ann_vol) if ann_vol > 0 else float("nan"),
            "net_sharpe": float(net_ann / ann_vol) if ann_vol > 0 else float("nan"),
            "cost_ann": float(cost + borrow)}


# ---------------------------------------------------------------------------
# Robustness — vary the age cutoff and the sub-period
# ---------------------------------------------------------------------------
def cutoff_sweep(data_mod, prices: pd.DataFrame, cutoffs: list[int]) -> pd.DataFrame:
    """Rebuild the L/S at several young/old age cutoffs; read the HAC t and CAPM alpha t."""
    rows = {}
    for c in cutoffs:
        r = data_mod.build_returns(prices, young_max=c)
        if r.empty or len(r) < 12:
            continue
        hac = hac_mean_t(r["ls"].to_numpy())
        ca = capm_alpha(r["ls"].to_numpy(), r["mkt"].to_numpy())
        rows[f"age<{c}"] = {
            "ls_ann": r["ls"].mean() * TRADING_MONTHS, "ls_t": hac["t"],
            "alpha_ann": ca["alpha"] * TRADING_MONTHS, "alpha_t": ca["alpha_t"],
            "beta": ca["beta"], "n": hac["n"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def subperiod_sweep(returns: pd.DataFrame, splits: list[tuple[str, str, str]]) -> pd.DataFrame:
    """HAC t + CAPM alpha t over labelled sub-periods ``(label, start, end)``."""
    rows = {}
    for label, a, b in splits:
        sub = returns.loc[a:b]
        if len(sub) < 8:
            continue
        hac = hac_mean_t(sub["ls"].to_numpy())
        ca = capm_alpha(sub["ls"].to_numpy(), sub["mkt"].to_numpy())
        rows[label] = {"ls_ann": sub["ls"].mean() * TRADING_MONTHS, "ls_t": hac["t"],
                       "alpha_t": ca["alpha_t"], "beta": ca["beta"], "n": hac["n"]}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule: >= 20 seeds)
# ---------------------------------------------------------------------------
def synthetic_ls(panel: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Turn a synthetic firm panel into the young/old/ls/mkt return frame the tests use."""
    buckets = panel.attrs["bucket"]
    young = [c for c in panel.columns if buckets[c] == "young"]
    old = [c for c in panel.columns if buckets[c] == "old"]
    y = panel[young].mean(axis=1)
    o = panel[old].mean(axis=1)
    return pd.DataFrame({"young": y, "old": o, "ls": y - o, "mkt": mkt})


def synthetic_mean_alpha_t(data_mod, age_alpha: float, n_seeds: int = 25,
                           base_seed: int = 748) -> dict:
    """Average the CAPM alpha t and the raw-spread HAC t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages its statistic over >= 20 seeds.
    Returns the mean CAPM alpha t (the honest test) and the mean raw-spread HAC t (which is
    high even at the null, because young firms carry a beta tilt) for a planted ``age_alpha``.
    """
    alpha_ts, raw_ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, mkt, _ = data_mod.synthetic_panel(age_alpha=age_alpha, seed=s)
        frame = synthetic_ls(panel, mkt)
        alpha_ts.append(capm_alpha(frame["ls"].to_numpy(), frame["mkt"].to_numpy())["alpha_t"])
        raw_ts.append(hac_mean_t(frame["ls"].to_numpy())["t"])
    return {"mean_alpha_t": float(np.nanmean(alpha_ts)),
            "mean_raw_t": float(np.nanmean(raw_ts))}
