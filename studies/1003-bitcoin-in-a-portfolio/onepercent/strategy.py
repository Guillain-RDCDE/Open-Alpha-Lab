"""How much bitcoin, and can the question be answered — Study 1003.

Mean-variance optimisation with a historical mean is a machine for converting noise into
confident recommendations, and bitcoin is the extreme case: its volatility is roughly ten times
an equity index's, so the standard error on its mean return is roughly ten times as large, so
the sample needed to estimate it is roughly a hundred times as long. Bitcoin has about a decade
of history. That is the whole study in three sentences, and the rest is measurement.

Three things are computed:

1. **The in-sample answer.** What weight would have been best, which is what most published
   allocations amount to. Reported because it is the number under discussion, not because it
   means anything.

2. **The out-of-sample answer.** Estimate the weight on data up to a date, hold it, measure what
   happened. ``walk_forward_weights`` does this properly, including the transaction costs of
   rebalancing an asset that moves 4% a day.

3. **The estimability question**, which comes first logically and last in every discussion.
   ``weight_standard_error`` and ``sample_needed`` ask how much data would be required for the
   optimiser's answer to have a confidence interval narrower than the range of allocations
   people argue about. If the interval spans 0% to 20%, then the difference between a 1% house
   view and a 5% house view is not a disagreement about bitcoin — it is two draws from the same
   distribution.

The calendar alignment in ``data`` matters more than it sounds. Bitcoin trades 365 days a year;
comparing its annualised volatility computed on 365 observations against an equity index's
computed on 252 inflates every Sharpe-like comparison in bitcoin's favour, and it is a common
error in exactly the material that recommends the 1%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The calendar, which decides every ratio that follows
# --------------------------------------------------------------------------- #
def align_to_equity_calendar(prices: pd.DataFrame, reference: str = "SPY") -> pd.DataFrame:
    """Reindex the whole panel onto the sessions on which ``reference`` traded.

    Bitcoin trades 365 days a year; SPY trades about 252. Left alone, a panel built from both
    has bitcoin rows on Saturdays where every other column is missing, and a naive annualisation
    then measures bitcoin's volatility over 365 observations and everything else's over 252.
    Since annualised volatility scales with the square root of the observation count, that
    inflates bitcoin's by about 20% — and *deflates* its Sharpe ratio by the same factor, which
    is why the error usually survives review: it looks conservative while making the
    diversification case stronger, because the correlation and covariance terms are computed on
    mismatched calendars too.

    Forward-filling onto the reference calendar folds each weekend's move into the following
    Monday, which is what an investor holding both actually experiences: the portfolio can only
    be rebalanced when the equity leg trades.
    """
    if reference not in prices.columns:
        return prices
    cal = prices[reference].dropna().index
    return prices.reindex(cal).ffill(limit=5)


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def sixty_forty(rets: pd.DataFrame, equity: str, bonds: str,
                w_equity: float = 0.6) -> pd.Series:
    """A daily-rebalanced 60/40. The base portfolio everything is measured against."""
    df = rets[[equity, bonds]].dropna()
    return (w_equity * df[equity] + (1 - w_equity) * df[bonds]).rename("60/40")


def sleeve(base: pd.Series, asset: pd.Series, weight: float) -> pd.Series:
    """Add a ``weight`` sleeve of ``asset``, funded pro rata from the base portfolio."""
    df = pd.concat([base.rename("base"), asset.rename("asset")], axis=1,
                   sort=False).dropna()
    return ((1 - weight) * df["base"] + weight * df["asset"]).rename(f"{weight:.0%}")


def stats(r: pd.Series, periods: int = TRADING_DAYS) -> dict:
    """Annualised summary statistics for a return series."""
    r = r.dropna()
    if len(r) < 30:
        return {}
    lg = np.log1p(r)
    cagr = float(np.expm1(lg.sum() * periods / len(r)))
    vol = float(r.std(ddof=1) * np.sqrt(periods))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    down = r[r < 0]
    return {"cagr": cagr, "vol": vol, "sharpe": cagr / vol if vol > 0 else np.nan,
            "max_drawdown": dd, "sortino": cagr / (down.std(ddof=1) * np.sqrt(periods))
            if len(down) > 2 and down.std(ddof=1) > 0 else np.nan,
            "calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "skew": float(r.skew()), "kurtosis": float(r.kurtosis()), "n": int(len(r))}


def weight_sweep(base: pd.Series, asset: pd.Series,
                 weights=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)) -> pd.DataFrame:
    """Every candidate allocation, scored the same way."""
    rows = []
    for w in weights:
        s = stats(sleeve(base, asset, w))
        if s:
            rows.append({"weight": w, **s})
    return pd.DataFrame(rows).set_index("weight")


# --------------------------------------------------------------------------- #
# The optimiser
# --------------------------------------------------------------------------- #
def _grid_stats(b: np.ndarray, a: np.ndarray, grid: np.ndarray) -> dict:
    """Annualised statistics for every weight in ``grid`` at once.

    The whole grid is one (n × w) matrix of portfolio returns, so the objective curve costs a
    couple of matrix operations instead of a Python loop rebuilding a DataFrame per weight.
    That is not a micro-optimisation: the bootstrap in ``weight_standard_error`` calls the
    optimiser several hundred times, and the loop version made a single test take three minutes.
    """
    n = len(b)
    if n < 30:
        return {}
    # R[t, j] = (1 - w_j) * b[t] + w_j * a[t]
    R = b[:, None] + np.outer(a - b, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log1p(R)
    lg = np.where(np.isfinite(lg), lg, -np.inf)
    total = lg.sum(axis=0)
    cagr = np.expm1(total * TRADING_DAYS / n)
    vol = R.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    eq = np.exp(np.cumsum(lg, axis=0))
    dd = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
    neg = np.where(R < 0, R, np.nan)
    with np.errstate(invalid="ignore"):
        down = np.nanstd(neg, axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"cagr": cagr, "vol": vol,
                "sharpe": np.where(vol > 0, cagr / vol, np.nan),
                "max_drawdown": dd,
                "sortino": np.where(down > 0, cagr / down, np.nan),
                "calmar": np.where(dd < 0, cagr / np.abs(dd), np.nan)}


def optimal_weight(base, asset, max_weight: float = 1.0,
                   objective: str = "sharpe", rf: float = 0.0) -> float:
    """The weight that maximises the chosen objective, on the data given.

    A grid search rather than a closed form, because the objectives are not all quadratic and
    because the grid makes the *shape* of the objective visible — which turns out to matter far
    more than its argmax. A flat optimum means the recommendation is arbitrary.
    """
    b, a = _paired(base, asset)
    if len(b) < 30:
        return 0.0
    grid = np.linspace(0.0, max_weight, 201)
    s = _grid_stats(b, a, grid)
    if not s:
        return 0.0
    v = s[objective].copy()
    if objective == "sharpe" and rf:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(s["vol"] > 0, (s["cagr"] - rf) / s["vol"], np.nan)
    if not np.isfinite(v).any():
        return 0.0
    return float(grid[np.nanargmax(v)])


def _paired(base, asset):
    """Align two return series and return them as plain float arrays."""
    if isinstance(base, np.ndarray) and isinstance(asset, np.ndarray):
        ok = np.isfinite(base) & np.isfinite(asset)
        return base[ok], asset[ok]
    df = pd.concat([pd.Series(base).rename("b"), pd.Series(asset).rename("a")],
                   axis=1, sort=False).dropna()
    return df["b"].to_numpy(dtype=float), df["a"].to_numpy(dtype=float)


def objective_curve(base, asset, max_weight: float = 0.30,
                    n: int = 61, objective: str = "sharpe") -> pd.DataFrame:
    """The objective as a function of weight — the shape, not just the peak."""
    b, a = _paired(base, asset)
    grid = np.linspace(0.0, max_weight, n)
    s = _grid_stats(b, a, grid)
    if not s:
        return pd.DataFrame(columns=["value", "cagr", "vol", "max_drawdown"],
                            index=pd.Index([], name="weight"))
    return pd.DataFrame({"value": s[objective], "cagr": s["cagr"], "vol": s["vol"],
                         "max_drawdown": s["max_drawdown"]},
                        index=pd.Index(grid, name="weight"))


def flatness(curve: pd.DataFrame, tol: float = 0.01) -> dict:
    """How wide is the range of weights that is within ``tol`` (relative) of the best?

    The single most useful number in the study. If every allocation from 0% to 15% is within one
    percent of optimal, then "the optimal allocation is 4%" is a statement about the third
    decimal place of an estimate, and the disagreement between a 1% house view and a 5% one is
    not a disagreement about anything measurable.
    """
    v = curve["value"].to_numpy()
    w = curve.index.to_numpy()
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {}
    best = float(np.nanmax(v[ok]))
    thresh = best - abs(best) * tol
    inside = w[ok][v[ok] >= thresh]
    return {"best_weight": float(w[ok][np.nanargmax(v[ok])]), "best_value": best,
            "plateau_lo": float(inside.min()), "plateau_hi": float(inside.max()),
            "plateau_width": float(inside.max() - inside.min()), "tol": tol}


# --------------------------------------------------------------------------- #
# Can the weight be estimated at all?
# --------------------------------------------------------------------------- #
def weight_standard_error(base: pd.Series, asset: pd.Series, max_weight: float = 0.50,
                          n_boot: int = 400, block: int = 21,
                          seed: int = 1003) -> dict:
    """A block bootstrap of the optimiser's answer.

    Resampling in blocks preserves the volatility clustering in both series, which matters:
    an i.i.d. bootstrap would understate the uncertainty in exactly the direction that makes
    the recommendation look more solid than it is.
    """
    b, a = _paired(base, asset)
    n = len(b)
    if n < 250:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    grid = np.linspace(0.0, max_weight, 201)
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        s = _grid_stats(b[idx], a[idx], grid)
        v = s.get("sharpe") if s else None
        out[i] = grid[np.nanargmax(v)] if v is not None and np.isfinite(v).any() else 0.0
    return {"mean": float(out.mean()), "median": float(np.median(out)),
            "sd": float(out.std(ddof=1)), "p05": float(np.percentile(out, 5)),
            "p95": float(np.percentile(out, 95)),
            "share_at_zero": float((out <= 1e-9).mean()),
            "share_at_cap": float((out >= max_weight - 1e-9).mean()),
            "draws": out}


def weight_vs_assumed_mean(base, asset, mus=None, max_weight: float = 0.50) -> pd.DataFrame:
    """The optimal weight as a function of the expected return you assume for the asset.

    This is the axis the debate actually turns on, and the one a backtest cannot inform. The
    asset's returns are recentred to each candidate mean — its volatility, its correlation and
    its whole path shape are preserved, only the drift changes — and the optimiser is rerun.

    It is the honest complement to ``weight_standard_error``: that function resamples the
    history and therefore inherits the realised mean as though it were known, while this one
    varies the single quantity nobody knows.
    """
    b, a = _paired(base, asset)
    if len(b) < 250:
        return pd.DataFrame()
    realised = float(np.expm1(np.log1p(a).sum() * TRADING_DAYS / len(a)))
    if mus is None:
        se = float(a.std(ddof=1) * np.sqrt(TRADING_DAYS)) / np.sqrt(len(a) / TRADING_DAYS)
        mus = np.linspace(realised - 2 * se, realised + 2 * se, 21)
    grid = np.linspace(0.0, max_weight, 201)
    daily_realised = np.log1p(a).mean()
    rows = []
    for mu in mus:
        shifted = np.expm1(np.log1p(a) - daily_realised + np.log1p(mu) / TRADING_DAYS)
        s = _grid_stats(b, shifted, grid)
        v = s["sharpe"] if s else None
        rows.append({"assumed_mean": float(mu),
                     "optimal_weight": float(grid[np.nanargmax(v)])
                     if v is not None and np.isfinite(v).any() else np.nan,
                     "realised_mean": realised})
    return pd.DataFrame(rows).set_index("assumed_mean")


def weight_with_mean_uncertainty(base, asset, max_weight: float = 0.50,
                                 n_draws: int = 400, seed: int = 1003) -> dict:
    """The optimiser's answer once uncertainty about the MEAN is admitted.

    ``weight_standard_error`` resamples the observed history, so every draw carries the realised
    mean and the resulting interval answers "how much would this recommendation move if the
    *ordering* of history had differed". That is not the question. The mean itself is estimated
    from the same {n} observations, with a standard error of σ/√years, and at bitcoin's
    volatility that standard error is enormous.

    Here each draw takes a mean from N(realised, SE), recentres the asset's returns onto it, and
    reruns the optimiser. The interval widens dramatically, and the widening is the finding:
    almost none of the uncertainty in an allocation recommendation comes from path noise, and
    almost all of it comes from not knowing the expected return.
    """
    b, a = _paired(base, asset)
    n = len(b)
    if n < 250:
        return {}
    years = n / TRADING_DAYS
    realised = float(np.expm1(np.log1p(a).sum() * TRADING_DAYS / n))
    sigma = float(a.std(ddof=1) * np.sqrt(TRADING_DAYS))
    se = sigma / np.sqrt(years)
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, max_weight, 201)
    daily_realised = np.log1p(a).mean()
    out = np.empty(n_draws)
    for i in range(n_draws):
        mu = rng.normal(realised, se)
        shifted = np.expm1(np.log1p(a) - daily_realised
                           + np.log1p(max(mu, -0.95)) / TRADING_DAYS)
        s = _grid_stats(b, shifted, grid)
        v = s["sharpe"] if s else None
        out[i] = grid[np.nanargmax(v)] if v is not None and np.isfinite(v).any() else 0.0
    return {"realised_mean": realised, "mean_se": se, "sigma": sigma,
            "median": float(np.median(out)), "sd": float(out.std(ddof=1)),
            "p05": float(np.percentile(out, 5)), "p95": float(np.percentile(out, 95)),
            "share_at_zero": float((out <= 1e-9).mean()),
            "share_at_cap": float((out >= max_weight - 1e-9).mean()),
            "draws": out}


def implied_mean_for_weight(base, asset, targets=(0.01, 0.02, 0.05, 0.10),
                            max_weight: float = 0.50, lo: float = -0.40,
                            hi: float = 0.90, n: int = 131) -> pd.DataFrame:
    """Invert the question: what expected return does each recommended weight imply?

    This is the most useful thing in the study, and it exists because the direct question turned
    out to have an uncomfortable answer. Fed bitcoin's actual history, a Sharpe optimiser does
    not recommend 1% or 2% — it recommends a large double-digit sleeve, because the realised
    return was enormous. So the published small allocations are not conservative *readings* of
    the record; they are the answers you get after overriding it. Inverting the mapping makes
    the override explicit and quantifies it: each weight is reported alongside the annual return
    an investor must expect for that weight to be optimal.
    """
    curve = weight_vs_assumed_mean(base, asset, np.linspace(lo, hi, n), max_weight)
    if curve.empty:
        return pd.DataFrame()
    mus = curve.index.to_numpy(dtype=float)
    ws = curve["optimal_weight"].to_numpy(dtype=float)
    rows = []
    for t in targets:
        j = int(np.nanargmin(np.abs(ws - t)))
        # refine by linear interpolation between the bracketing grid points
        implied = mus[j]
        if 0 < j < len(mus) - 1 and ws[j + 1] != ws[j - 1]:
            implied = float(np.interp(t, ws[j - 1:j + 2], mus[j - 1:j + 2]))
        rows.append({"weight": t, "implied_mean": float(implied),
                     "nearest_grid_weight": float(ws[j])})
    return pd.DataFrame(rows).set_index("weight")


def sample_needed(mu: float, sigma: float, target_se: float = 0.02,
                  periods: int = TRADING_DAYS) -> dict:
    """Years of data needed for the mean return's standard error to reach ``target_se``.

    The standard error of an annualised mean is sigma / sqrt(years). This is the calculation
    that should precede every allocation recommendation and follows almost none of them: it
    depends on volatility alone, so it can be answered before any data is collected.
    """
    years = (sigma / target_se) ** 2 if target_se > 0 else np.inf
    return {"mu": mu, "sigma": sigma, "target_se": target_se, "years_needed": float(years),
            "se_at_10y": float(sigma / np.sqrt(10)),
            "t_stat_at_10y": float(mu / (sigma / np.sqrt(10))) if sigma > 0 else np.nan}


def power_to_distinguish(base: pd.Series, asset: pd.Series, w_a: float = 0.01,
                         w_b: float = 0.05, n_boot: int = 600, block: int = 21,
                         seed: int = 1003) -> dict:
    """How often does a sample of this length prefer 5% over 1%, and by how much?

    Not "which is better" but "can this much data tell them apart". If the two allocations swap
    places from one bootstrap draw to the next roughly half the time, the argument between their
    advocates is unresolvable with the evidence available, whoever happens to be right.
    """
    b, a = _paired(base, asset)
    n = len(b)
    if n < 250:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    pair = np.array([w_a, w_b], dtype=float)
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        s = _grid_stats(b[idx], a[idx], pair)
        if s and np.isfinite(s["sharpe"]).all():
            diffs.append(float(s["sharpe"][1] - s["sharpe"][0]))
    diffs = np.array(diffs)
    return {"w_a": w_a, "w_b": w_b, "mean_diff": float(diffs.mean()),
            "sd_diff": float(diffs.std(ddof=1)),
            "share_b_wins": float((diffs > 0).mean()),
            "p05": float(np.percentile(diffs, 5)),
            "p95": float(np.percentile(diffs, 95)),
            "distinguishable": bool(np.percentile(diffs, 5) > 0
                                    or np.percentile(diffs, 95) < 0)}


# --------------------------------------------------------------------------- #
# Out of sample
# --------------------------------------------------------------------------- #
def walk_forward_weights(base: pd.Series, asset: pd.Series, lookback_years: float = 3.0,
                         rebalance: int = 63, max_weight: float = 0.20,
                         cost_bps: float = 30.0) -> pd.DataFrame:
    """Estimate the weight on the past, hold it forward, pay to rebalance.

    ``cost_bps`` is charged on turnover. It is set to 30 basis points by default rather than the
    5 or 10 an equity fund would pay, because the sleeve being traded is bitcoin: spread,
    slippage and (for many holders) a fund wrapper are all wider. Ignoring this is the second
    most common flattery in the genre, after the calendar.
    """
    df = pd.concat([base.rename("b"), asset.rename("a")], axis=1, sort=False).dropna()
    lb = int(lookback_years * TRADING_DAYS)
    if len(df) < lb + rebalance:
        return pd.DataFrame()
    rows = []
    w = 0.0
    for i in range(lb, len(df), rebalance):
        hist = df.iloc[max(0, i - lb):i]
        new_w = optimal_weight(hist["b"].to_numpy(dtype=float),
                               hist["a"].to_numpy(dtype=float), max_weight)
        fwd = df.iloc[i:i + rebalance]
        if len(fwd) == 0:
            break
        turnover = abs(new_w - w)
        cost = turnover * cost_bps / 10000.0
        realised = (1 - new_w) * fwd["b"] + new_w * fwd["a"]
        realised.iloc[0] -= cost
        rows.append({"date": df.index[i], "weight": new_w, "turnover": turnover,
                     "cost": cost, "n_days": len(fwd),
                     "realised": float(np.expm1(np.log1p(realised).sum())),
                     "base_realised": float(np.expm1(np.log1p(fwd["b"]).sum()))})
        w = new_w
    return pd.DataFrame(rows).set_index("date")


def walk_forward_series(base: pd.Series, asset: pd.Series, lookback_years: float = 3.0,
                        rebalance: int = 63, max_weight: float = 0.20,
                        cost_bps: float = 30.0) -> pd.DataFrame:
    """The daily return series of the walk-forward strategy, for comparison with fixed weights."""
    df = pd.concat([base.rename("b"), asset.rename("a")], axis=1, sort=False).dropna()
    lb = int(lookback_years * TRADING_DAYS)
    if len(df) < lb + rebalance:
        return pd.DataFrame()
    out = pd.Series(index=df.index[lb:], dtype=float)
    w = 0.0
    for i in range(lb, len(df), rebalance):
        hist = df.iloc[max(0, i - lb):i]
        new_w = optimal_weight(hist["b"].to_numpy(dtype=float),
                               hist["a"].to_numpy(dtype=float), max_weight)
        fwd = df.iloc[i:i + rebalance]
        if len(fwd) == 0:
            break
        seg = (1 - new_w) * fwd["b"] + new_w * fwd["a"]
        seg.iloc[0] -= abs(new_w - w) * cost_bps / 10000.0
        out.loc[seg.index] = seg.to_numpy()
        w = new_w
    return pd.DataFrame({"walk_forward": out.dropna(),
                         "base": df["b"].reindex(out.dropna().index)})


def rebalancing_matters(base: pd.Series, asset: pd.Series, weight: float = 0.02,
                        frequencies=(1, 21, 63, 252, 10_000)) -> pd.DataFrame:
    """A fixed sleeve, rebalanced at different frequencies.

    With an asset this volatile the difference between rebalancing monthly and never is not a
    detail: never rebalancing turns a 2% sleeve into a 20% position after a good run, so the
    "2% allocation" whose performance is being quoted may never have been a 2% allocation.
    """
    df = pd.concat([base.rename("b"), asset.rename("a")], axis=1, sort=False).dropna()
    b = df["b"].to_numpy()
    a = df["a"].to_numpy()
    rows = []
    for freq in frequencies:
        vb, va = 1 - weight, weight
        vals = np.empty(len(df))
        max_w = weight
        for t in range(len(df)):
            vb *= 1 + b[t]
            va *= 1 + a[t]
            tot = vb + va
            vals[t] = tot
            max_w = max(max_w, va / tot if tot > 0 else 0.0)
            if freq < 10_000 and (t + 1) % freq == 0:
                vb, va = (1 - weight) * tot, weight * tot
        r = pd.Series(vals, index=df.index).pct_change().dropna()
        s = stats(r)
        rows.append({"rebalance_days": freq, "max_weight_reached": max_w, **s})
    return pd.DataFrame(rows).set_index("rebalance_days")


def synthetic_pair(n: int = 2500, true_weight: float = 0.03, base_vol: float = 0.10,
                   base_mu: float = 0.06, asset_vol: float = 0.70,
                   corr: float = 0.15, seed: int = 1003) -> dict:
    """A base portfolio and a volatile asset with a KNOWN optimal weight.

    ``asset_mu`` is solved so that the Sharpe-maximising weight equals ``true_weight`` under the
    given volatilities and correlation. The optimiser can then be scored against a truth, which
    is the only way to separate "the estimate is biased" from "the estimate is noisy" — and the
    answer here is emphatically the second.
    """
    # For a two-asset Sharpe optimum with a risk-free rate of zero, the tangency weight is
    # proportional to the inverse-covariance times the mean vector; inverting for asset_mu:
    cov_ab = corr * base_vol * asset_vol
    w = true_weight
    # w* = (mu_a * var_b - mu_b * cov) / (mu_a * var_b + mu_b * var_a - (mu_a+mu_b) * cov)
    # solved numerically over a grid to avoid a fragile closed form
    grid = np.linspace(0.0, 3.0, 3001)
    var_b, var_a = base_vol ** 2, asset_vol ** 2
    num = grid * var_b - base_mu * cov_ab
    den = grid * var_b + base_mu * var_a - (grid + base_mu) * cov_ab
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(np.abs(den) > 1e-12, num / den, np.nan)
    asset_mu = float(grid[np.nanargmin(np.abs(implied - w))])
    rng = np.random.default_rng(seed)
    cov = np.array([[var_b, cov_ab], [cov_ab, var_a]]) / TRADING_DAYS
    mu = np.array([base_mu, asset_mu]) / TRADING_DAYS
    draws = rng.multivariate_normal(mu, cov, size=n)
    idx = pd.bdate_range("2014-10-01", periods=n)
    return {"base": pd.Series(draws[:, 0], index=idx, name="base"),
            "asset": pd.Series(draws[:, 1], index=idx, name="asset"),
            "true_weight": true_weight, "asset_mu": asset_mu, "asset_vol": asset_vol}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if a bitcoin sleeve improved the realised Sharpe ratio in-sample;
      **Weak** if it improved return but not the ratio; **None** otherwise. Backward-looking
      by construction, and the stamp claims nothing about the future.
    - **Tradability**: keyed to whether the *published* recommendations follow from the data.
      **Investable** if the optimiser's answer is close to what allocators actually recommend;
      **Fragile** if the gap is moderate; **Mirage** if the common recommendation requires
      assuming an expected return the record contradicts — because then the recommendation is
      a prior wearing a backtest's clothes, and the backtest is decoration.
    """
    signal = ("Real" if h["best_sharpe"] > h["base_sharpe"] + 0.05
              else ("Weak" if h["best_cagr"] > h["base_cagr"] else "None"))
    gap = h["best_weight"] - 0.02
    trad = ("Investable" if gap < 0.02 else ("Fragile" if gap < 0.06 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Emphatically, in sample. Over {h['years']:.1f} years to {h['as_of']}, a 60/40 "
            f"returned {h['base_cagr']:.2%} a year at {h['base_vol']:.1%} volatility, Sharpe "
            f"{h['base_sharpe']:.2f}. The Sharpe-maximising bitcoin sleeve was "
            f"**{h['best_weight']:.1%}** — not 1%, not 2% — lifting the ratio to "
            f"{h['best_sharpe']:.2f} on a {h['best_cagr']:.2%} return, at the cost of a "
            f"{h['best_dd']:.1%} maximum drawdown against the base portfolio's "
            f"{h['base_dd']:.1%}. That number is what the historical record says, and the gap "
            f"between it and every published recommendation is the whole subject of this study. "
            f"One correction along the way: bitcoin trades 365 days a year and the rest of a "
            f"portfolio does not, so everything here is aligned to the equity calendar; "
            f"skipping that step changes bitcoin's annualised volatility by "
            f"{h['calendar_inflation']:.0%} and every ratio built on it."),
        "trad_why": (
            f"So why does nobody recommend {h['best_weight']:.0%}? Because nobody believes the "
            f"realised mean. Bitcoin returned {h['realised_mean']:.0%} a year over this sample; "
            f"inverting the optimiser shows what each published allocation is really assuming. "
            f"A **1% sleeve is optimal if you expect bitcoin to return "
            f"{h['implied_1pct']:+.1%} a year**. A 2% sleeve implies {h['implied_2pct']:+.1%} — "
            f"essentially nothing. A 5% sleeve implies {h['implied_5pct']:+.1%}, about what "
            f"equities are expected to do. These are not cautious readings of the data; they "
            f"are the answers you get **after discarding it**, which is a defensible position "
            f"and an entirely different one from what the accompanying backtests imply. The "
            f"override is justified by the standard error: at {h['btc_vol']:.0%} volatility, "
            f"pinning bitcoin's expected return to ±2 percentage points takes "
            f"**{h['years_needed']:,.0f} years**, and {h['years']:.0f} years leaves a standard "
            f"error of {h['se_at_now']:.0%} — so the {h['realised_mean']:.0%} is not knowledge. "
            f"Out of sample the walk-forward allocator returned {h['wf_cagr']:.2%} against the "
            f"base portfolio's {h['wf_base_cagr']:.2%}, with its chosen weight swinging from "
            f"{h['wf_min_w']:.1%} to {h['wf_max_w']:.1%}. The honest form of a bitcoin "
            f"recommendation states the expected return it assumes; the weight then follows, "
            f"and can be argued with."),
        "trad": trad,
        "one_sentence": (
            f"Bitcoin's own history says hold {h['best_weight']:.1%}, so the industry's 1-2% is "
            f"not a reading of the data but an override of it — a 2% sleeve is what you get "
            f"from assuming bitcoin returns {h['implied_2pct']:+.1%} a year."),
    }
