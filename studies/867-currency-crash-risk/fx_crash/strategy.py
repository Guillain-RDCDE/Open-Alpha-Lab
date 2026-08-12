"""Strategy + inference for Study 867 — Currency Crash Risk (Brunnermeier-Nagel-Pedersen).

The claim (Brunnermeier, Nagel & Pedersen 2008): high-carry currencies are exposed to
**crash risk** — their returns are **negatively skewed**, and the higher the carry the
more negative the skew. A long-high / short-low carry basket therefore inherits a deep
negative skew: the carry premium is compensation for a sold-crash tail ("up by the
stairs, down by the elevator").

Two decisive real-tape tests, both directly about the *third moment*:

* **The basket crash skew (primary).** Build the dollar-neutral long-high / short-low
  carry basket (rank on the carry proxy). Its realized skewness is the headline number;
  we attach a **Newey-West (HAC) *t*** on that skewness via the standardised-cubed-
  residual series ``g_t = z_t**3`` (``mean(g) = skew``), so the "is the basket
  negatively skewed" question gets a robust, serial-correlation-aware *t*. A significant
  *negative* skew = the Brunnermeier-et-al crash signature.

* **The skew-carry cross-section (secondary).** Compute each currency's own realized
  skewness and regress it on the carry proxy across the cross-section. BNP predict a
  **negative slope** (higher carry -> more negative skew). We report the slope, its
  analytic *t*, and the rank correlation.

Around them: the carry premium itself (mean return, NW *t* — is there a premium at
all?), a crash-conditional split (how much of the premium is given back in the worst
weeks), a label-shuffle placebo (is the negative skew tied to the *carry ordering* or
generic?), a two-era robustness cut, a costed carry book, and a seeded synthetic
positive control.

Execution: one documented lag — the carry ranking is known at the close of week ``t``
and the basket is held over week ``t+1`` (the static proxy ranking makes this immaterial
but the convention is stated). This is distinct from:

* [364-fx-carry-trade](../../364-fx-carry-trade/) — tests whether the carry basket earns
  a **premium** (does UIP fail); this study tests the **crash skew** that is said to
  *justify* that premium — the third moment, not the mean.
* [828-fx-dollar-factor](../../828-fx-dollar-factor/) — the **dollar factor** DOL (common
  average currency move), not the **high-minus-low carry** cross-section.
* [27-steamroller](../../27-steamroller/) — the generic "picking up pennies in front of a
  steamroller" short-vol shape; here the specific FX carry-crash instance.
* [797-fx-value-ppp](../../797-fx-value-ppp/) — the **PPP value** anomaly, a different
  currency signal entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52


# --------------------------------------------------------------------------- #
# Carry basket construction
# --------------------------------------------------------------------------- #
def carry_ranks(carry: dict[str, float], columns) -> pd.Series:
    """Carry (annualised %, proxy) aligned to the panel columns, high to low."""
    return pd.Series({c: carry.get(c, 0.0) for c in columns}).sort_values(ascending=False)


def carry_weights(carry: dict[str, float], columns, k: int = 3) -> pd.Series:
    """Dollar-neutral weights: +1/k on the top-k carries, -1/k on the bottom-k.

    The carry map is static, so the weights are constant across weeks (a fixed
    long-high / short-low portfolio) — the canonical "borrow low-yield, hold high-yield".
    """
    ranked = carry_ranks(carry, columns)
    w = pd.Series(0.0, index=ranked.index)
    if len(ranked) < 2 * k:
        k = max(1, len(ranked) // 2)
    w.iloc[:k] = 1.0 / k
    w.iloc[-k:] = -1.0 / k
    return w.reindex(columns).fillna(0.0)


def basket_returns(total_ret: pd.DataFrame, carry: dict[str, float],
                   k: int = 3) -> pd.Series:
    """Weekly return of the long-high / short-low carry basket (dollar-neutral).

    ``total_ret`` is the per-currency weekly total return (spot + carry/52) of a *long*
    USD-funded position; the short leg earns the negative via the negative weights. The
    result is already an excess-of-cash return.
    """
    w = carry_weights(carry, total_ret.columns, k=k)
    return (total_ret * w).sum(axis=1).rename("carry_basket")


def leg_returns(total_ret: pd.DataFrame, carry: dict[str, float],
                k: int = 3) -> tuple[pd.Series, pd.Series]:
    """(high-carry long leg, low-carry short-funding leg) equal-weight weekly returns."""
    ranked = carry_ranks(carry, total_ret.columns)
    kk = min(k, len(ranked) // 2) or 1
    hi_names = list(ranked.index[:kk])
    lo_names = list(ranked.index[-kk:])
    hi = total_ret[hi_names].mean(axis=1).rename("high_carry")
    lo = total_ret[lo_names].mean(axis=1).rename("low_carry")
    return hi, lo


# --------------------------------------------------------------------------- #
# Inference primitives (shared house set — cf. study 803 / 828)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Realized skewness + a robust t on it
# --------------------------------------------------------------------------- #
def realized_skew(x: np.ndarray) -> float:
    """Sample skewness (population third standardised moment, ddof=0)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 3))


def skew_nw_t(x: np.ndarray, lags: int = 6) -> float:
    """Newey-West *t* for "is the skewness of ``x`` different from 0".

    The sample skewness equals the mean of the standardised-cubed residuals
    ``g_t = ((x_t - mean)/sd)**3`` (``mean(g) = skew``). We put a HAC (Newey-West) *t* on
    that ``g`` series, so a serially-correlated tail does not overstate significance. This
    is the moment-estimator *t* (it treats the standardising mean/sd as known); it is a
    deliberately conservative, transparent stand-in for a full delta-method skew SE and is
    documented as such. A *negative* significant value = the Brunnermeier-et-al crash skew.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return float("nan")
    g = ((x - m) / s) ** 3
    return newey_west_t(g, lags)


def annualised_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(x: np.ndarray) -> float:
    """Max drawdown of the cumulative (additive) return path of ``x`` (<= 0)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def crash_split(x: np.ndarray, q: float = 0.05) -> dict:
    """Split weekly returns into 'risk-off' (worst ``q`` quantile) vs 'calm'.

    Returns the mean in each bucket and the share of the *total* additive return given
    back in the worst-``q`` weeks — the carry-crash accounting.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return {"calm_mean": float("nan"), "off_mean": float("nan"),
                "tail_share": float("nan"), "worst": float("nan")}
    thr = np.quantile(x, q)
    off = x[x <= thr]
    calm = x[x > thr]
    total = x.sum()
    return {"calm_mean": float(calm.mean()), "off_mean": float(off.mean()),
            "tail_share": float(off.sum() / total) if total != 0 else float("nan"),
            "worst": float(x.min())}


# --------------------------------------------------------------------------- #
# Headline: the basket crash skew + premium
# --------------------------------------------------------------------------- #
def basket_stats(bundle: dict, k: int = 3, nw_lags: int = 6) -> dict:
    """Crash-skew + premium stats for the long-high / short-low carry basket."""
    total_ret = bundle["total_ret"]
    carry = bundle["carry"]
    b = basket_returns(total_ret, carry, k=k).to_numpy(dtype=float)
    b = b[~np.isnan(b)]
    n = len(b)
    mu = float(b.mean()) if n else float("nan")
    sd = float(b.std(ddof=1)) if n > 1 else float("nan")
    cs = crash_split(b)
    return {
        "n_weeks": n,
        "skew": realized_skew(b),
        "skew_t": skew_nw_t(b, nw_lags),
        "mean_bps": mu * 1e4,
        "ann_pct": mu * PERIODS_PER_YEAR * 100,
        "premium_t": newey_west_t(b, nw_lags),
        "sharpe": (mu / sd * np.sqrt(PERIODS_PER_YEAR)) if sd and sd > 0 else float("nan"),
        "worst_week_pct": cs["worst"] * 100 if cs["worst"] == cs["worst"] else float("nan"),
        "max_dd_pct": max_drawdown(b) * 100,
        "tail_share": cs["tail_share"],
    }


# --------------------------------------------------------------------------- #
# The skew-carry cross-section
# --------------------------------------------------------------------------- #
def per_currency_skew(bundle: dict) -> pd.Series:
    """Each currency's full-sample realized skewness of weekly spot returns."""
    ret = bundle["spot_ret"]
    return pd.Series({c: realized_skew(ret[c].to_numpy()) for c in ret.columns})


def skew_carry_regression(bundle: dict) -> dict:
    """OLS of per-currency realized skewness on carry across the cross-section.

    BNP predict a **negative slope**: higher carry -> more negative skew. Returns the
    slope, its analytic (small-sample) *t*, R², the Spearman rank correlation, and n.
    """
    skew = per_currency_skew(bundle)
    carry = pd.Series({c: bundle["carry"].get(c, 0.0) for c in skew.index})
    d = pd.DataFrame({"skew": skew, "carry": carry}).dropna()
    n = len(d)
    if n < 3:
        return {"slope": float("nan"), "t_slope": float("nan"), "r2": float("nan"),
                "spearman": float("nan"), "n": n}
    x = d["carry"].to_numpy(); y = d["skew"].to_numpy()
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    dof = n - 2
    ss = float((x - x.mean()) @ (x - x.mean()))
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    se_b = np.sqrt(sigma2 / ss) if ss > 0 and sigma2 == sigma2 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    # Spearman rank correlation (no scipy dependency for the core path)
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    sp = float(np.corrcoef(rx, ry)[0, 1]) if n > 1 else float("nan")
    return {"slope": float(b[1]), "t_slope": float(b[1] / se_b) if se_b and se_b > 0 else float("nan"),
            "r2": r2, "spearman": sp, "n": n}


def leg_skews(bundle: dict, k: int = 3, nw_lags: int = 6) -> dict:
    """Realized skew of the high-carry leg vs the low-carry leg (and their difference)."""
    hi, lo = leg_returns(bundle["total_ret"], bundle["carry"], k=k)
    hv, lv = hi.to_numpy(), lo.to_numpy()
    return {
        "hi_skew": realized_skew(hv), "hi_skew_t": skew_nw_t(hv, nw_lags),
        "lo_skew": realized_skew(lv), "lo_skew_t": skew_nw_t(lv, nw_lags),
        "diff": realized_skew(hv) - realized_skew(lv),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the negative skew tied to the CARRY ordering, or generic?
# --------------------------------------------------------------------------- #
def label_shuffle_placebo(bundle: dict, k: int = 3, n_perm: int = 1000,
                          seed: int = 867) -> dict:
    """Permute which currency owns which carry value, rebuild the basket, remeasure skew.

    Each shuffle reassigns the fixed set of carry values to the currencies at random and
    reconstructs the long-high / short-low basket. p = share of shuffles whose basket
    skew is <= observed (a **left-tail** test — the BNP prediction is a specifically
    *negative* skew tied to the true carry ordering).
    """
    total_ret = bundle["total_ret"]
    cols = list(total_ret.columns)
    carry_vals = np.array([bundle["carry"].get(c, 0.0) for c in cols], dtype=float)
    obs = realized_skew(basket_returns(total_ret, bundle["carry"], k=k).to_numpy())
    R = total_ret.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    skews = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(cols))
        cmap = {cols[j]: carry_vals[perm[j]] for j in range(len(cols))}
        ranked = pd.Series(cmap).sort_values(ascending=False)
        kk = min(k, len(ranked) // 2) or 1
        w = pd.Series(0.0, index=ranked.index)
        w.iloc[:kk] = 1.0 / kk
        w.iloc[-kk:] = -1.0 / kk
        w = w.reindex(cols).fillna(0.0).to_numpy()
        skews[i] = realized_skew(R @ w)
    return {"obs_skew": float(obs),
            "placebo_mean": float(np.nanmean(skews)),
            "placebo_sd": float(np.nanstd(skews, ddof=1)) if n_perm > 1 else float("nan"),
            "p_value": float((skews <= obs).mean()), "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# Era split
# --------------------------------------------------------------------------- #
def era_stats(bundle: dict, split: str = "2015-01-01", k: int = 3,
              nw_lags: int = 6) -> dict:
    """Basket crash-skew + premium in two eras (early < split <= late).

    Requires a DatetimeIndex on ``total_ret`` (the real tape); returns both halves.
    """
    tr = bundle["total_ret"]
    idx = tr.index
    early_mask = idx < pd.Timestamp(split)
    late_mask = ~early_mask

    def _sub(mask):
        sub = {"total_ret": tr[mask], "spot_ret": bundle["spot_ret"][mask],
               "carry": bundle["carry"]}
        return basket_stats(sub, k=k, nw_lags=nw_lags)

    return {"early": _sub(early_mask), "late": _sub(late_mask), "split": split}


# --------------------------------------------------------------------------- #
# The costed carry book
# --------------------------------------------------------------------------- #
def timer_stats(bundle: dict, k: int = 3, cost_bps: float = 2.0,
                borrow_bps_ann: float = 50.0) -> dict:
    """Cost the long-high / short-low carry book.

    The static ranking means week-to-week turnover is ~0 once on, so the frictions are a
    small rebalancing drag (one-way ``cost_bps`` on the 2.0-NAV gross book, per week) plus
    the ongoing **borrow** on the short leg (``borrow_bps_ann``/52). Returns gross/net
    annualised mean, net Sharpe, the net NW *t*, and the (unchanged) skew — costs shift
    the mean, not the shape.
    """
    gross = basket_returns(bundle["total_ret"], bundle["carry"], k=k).to_numpy(dtype=float)
    gross = gross[~np.isnan(gross)]
    weekly_drag = (cost_bps / 1e4) * 2.0 + (borrow_bps_ann / 1e4) / PERIODS_PER_YEAR
    net = gross - weekly_drag
    return {
        "n_weeks": len(gross),
        "gross_ann_pct": float(gross.mean() * PERIODS_PER_YEAR * 100),
        "net_ann_pct": float(net.mean() * PERIODS_PER_YEAR * 100),
        "cost_ann_pct": float(weekly_drag * PERIODS_PER_YEAR * 100),
        "sharpe_net": annualised_sharpe(net),
        "t_net": newey_west_t(net),
        "skew": realized_skew(gross),
    }


def cost_sweep(bundle: dict, k: int = 3, borrow_grid=(0.0, 25.0, 50.0, 100.0)) -> list[dict]:
    out = []
    for b in borrow_grid:
        tm = timer_stats(bundle, k=k, cost_bps=2.0, borrow_bps_ann=b)
        out.append({"borrow_bps": b, "net_ann_pct": tm["net_ann_pct"],
                    "sharpe_net": tm["sharpe_net"]})
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bundle: dict, k: int = 3) -> dict:
    """Run the headline crash-skew + skew-carry slope on a synthetic bundle."""
    bs = basket_stats(bundle, k=k)
    reg = skew_carry_regression(bundle)
    return {"basket_skew": bs["skew"], "skew_t": bs["skew_t"],
            "slope": reg["slope"], "spearman": reg["spearman"],
            "n_weeks": bs["n_weeks"]}
