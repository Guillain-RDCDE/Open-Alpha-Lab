"""Strategy + inference for Study 955 — ADR Catch-Up.

The claim: the home market closed hours before the ADR did, so the ADR should still be
*owing* part of the home move at its own US close — and the unpaid residue should be paid
the next session. Three nested questions, each with its own test:

1. **Is anything stale?** Pooled HAC regression of the ADR's day-``t`` return on the home
   dollar move of day ``t`` *and* day ``t-1``. If the ADR closes at fair value, the
   *lagged* home move carries no loading (``beta_lag`` ~ 0). That is the marquee test:
   it needs no trading rule, no cost model and no signal construction.

2. **Does the un-caught-up residue predict?** For each name we strip out the part of the
   ADR move the home market explains, using a **rolling** beta estimated on the previous
   ``window`` days only (never the full sample — that would be look-ahead), leaving a
   residual ``e_t``. Then we ask, Fama-MacBeth style, whether ``e_t`` predicts the ADR's
   next-day return. The cross-sectional slope is computed date by date and the *time
   series* of slopes carries the HAC ``t``, which is the honest way to handle eight names
   that all move together.

3. **Is that just plain one-day reversal?** ``e_t`` contains ``a_t``, so any negative
   loading could simply be the well-known short-horizon reversal of a single stock
   (bid-ask bounce, liquidity provision) wearing an ADR costume. The discriminating
   regression puts ``a_t`` and ``x_t`` on the right-hand side together: if catch-up is a
   *distinct* phenomenon, ``x_t`` must still load once ``a_t`` is controlled for.

4. **Is the loading a loading, or forty-eight enormous nights?** A HAC ``t`` corrects the
   standard error for dependence, not for *leverage*: a pooled slope is an average, and a
   handful of huge regressor values can own it outright. :func:`tail_sensitivity` re-runs
   the marquee regression with the biggest lagged moves trimmed and, separately,
   winsorized, calibrated against a synthetic panel where the planted lag is genuinely
   linear (there the same knife moves nothing). :func:`lag_bucket_table` asks the same
   question without a regression at all. This is the test that decides the stamp here.

Then the tradable form: a dollar-ish-neutral long/short book, long the ADRs whose
residual says they under-reacted and short the ones that over-reacted, gross exposure 1,
**one execution lag** (the residual is formed from closes through day ``t``; the book is
held over day ``t+1``), one-way cost x NAV charged on turnover, and an explicit borrow
charge on the short leg.

Because the book is self-financing (gross 1, near-zero net), its collateral earns the cash
rate and its **excess-of-cash return equals the spread itself** — so the Sharpe race
against a cash-only alternative is already excess-vs-excess. Where we race it against the
equal-weight ADR basket, that basket is put on the same footing by subtracting the cash
leg explicitly.

Returns are **simple** (arithmetic) throughout, so a weighted book return is exactly
``sum_i w_i r_i`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _hac_lags(n: int) -> int:
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = _hac_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def hac_ols(y, regressors, lags: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OLS of ``y`` on a constant plus ``regressors``, with Newey-West standard errors.

    ``regressors`` is a sequence of 1-D arrays. Returns ``(coefficients, t_stats)`` with
    the intercept first. The Bartlett bandwidth follows the usual ``4 (n/100)^(2/9)`` rule.
    """
    y = np.asarray(y, dtype=float)
    cols = [np.ones(len(y))] + [np.asarray(r, dtype=float) for r in regressors]
    X = np.column_stack(cols)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
    if n < X.shape[1] + 3:
        k = X.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)
    if lags is None:
        lags = _hac_lags(n)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = Xu[lag:].T @ Xu[:-lag]
        S += w * (A + A.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, b / se, np.nan)
    return b, t


def block_bootstrap_ci(
    r: pd.Series,
    stat: str = "sharpe",
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 955,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe (or mean) of a return series.

    Blocks of ``block`` consecutive days preserve vol clustering and the day-to-day
    autocorrelation the HAC ``t`` also guards against.
    """
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    n = x.size
    ann = np.sqrt(periods_per_year)

    def _stat(s):
        if stat == "mean":
            return s.mean() * periods_per_year
        sd = s.std(ddof=1)
        return s.mean() / sd * ann if sd > 0 else np.nan

    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    point = float(_stat(x))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = _stat(x[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and len(wealth) and wealth.iloc[-1] > 0 else float("nan"))
    dd = float((wealth / wealth.cummax() - 1.0).min()) if n else float("nan")
    return {
        "n_days": int(n),
        "ann_return": float(mu * periods_per_year),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# 1. Is anything stale? — the pooled catch-up regression
# --------------------------------------------------------------------------- #
def _lagged_frame(panel: pd.DataFrame, cols=("x",)) -> pd.DataFrame:
    """``panel`` plus a within-name one-observation lag of each of ``cols``, NaNs dropped.

    The lag is the previous *surviving* row for that name. After a home-market holiday the
    row is dropped on both sides (see :func:`adr_catchup.data.build_panel`), so on ~8% of
    Japanese rows "yesterday's home move" is two or three sessions old.
    :func:`consecutive_only` re-runs the test on the strictly adjacent rows.
    """
    df = panel.copy()
    g = df.groupby("adr", sort=False)
    for c in cols:
        df[f"{c}_lag"] = g[c].shift(1)
    need = ["a", *cols, *[f"{c}_lag" for c in cols]]
    return df.dropna(subset=need).sort_index()


def stale_catchup_test(panel: pd.DataFrame) -> dict:
    """Pooled HAC regression ``a_t = alpha + b0 x_t + b1 x_{t-1} + eps``.

    ``b0`` is how much of the home dollar move the ADR prices the *same* day (it need not
    be 1 — the ADR is a single stock and ``x`` is an *index*, so ``b0`` is the stock's
    beta to its home market, not a catch-up ratio). ``b1`` is the test that matters: a
    positive, significant loading on *yesterday's* home move means the ADR closed with
    part of it still unpaid. ``b1`` ~ 0 means the catch-up is already complete by the ADR's
    own close.

    Rows are pooled across names and sorted by date, so the Bartlett kernel absorbs both
    serial correlation and the same-day cross-sectional correlation of neighbouring rows.
    """
    df = panel.copy()
    df["x_lag"] = df.groupby("adr", sort=False)["x"].shift(1)
    df = df.dropna(subset=["a", "x", "x_lag"]).sort_index()
    b, t = hac_ols(df["a"].to_numpy(), [df["x"].to_numpy(), df["x_lag"].to_numpy()])
    return {
        "alpha_bps": float(b[0] * 1e4), "t_alpha": float(t[0]),
        "beta_same_day": float(b[1]), "t_same_day": float(t[1]),
        "beta_lag": float(b[2]), "t_lag": float(t[2]),
        "n_obs": int(len(df)),
    }


def leg_decomposition(panel: pd.DataFrame) -> dict:
    """Split the home dollar move into its index leg and its FX leg, same-day and lagged.

    ``a_t = alpha + c0 h_t + c1 h_{t-1} + d0 f_t + d1 f_{t-1}``. If a lagged loading is a
    genuine stale-price effect it should show in the *index* leg — the leg that actually
    stopped trading hours ago. FX trades around the clock and Yahoo stamps its close in
    New York hours, so a large lagged FX loading would point at a stamping artefact rather
    than a stale market, and is worth knowing about either way.
    """
    df = panel.copy()
    g = df.groupby("adr", sort=False)
    df["h_lag"] = g["h"].shift(1)
    df["f_lag"] = g["f"].shift(1)
    df = df.dropna(subset=["a", "h", "f", "h_lag", "f_lag"]).sort_index()
    b, t = hac_ols(df["a"].to_numpy(),
                   [df["h"].to_numpy(), df["h_lag"].to_numpy(),
                    df["f"].to_numpy(), df["f_lag"].to_numpy()])
    return {"n_obs": int(len(df)),
            "h": float(b[1]), "t_h": float(t[1]),
            "h_lag": float(b[2]), "t_h_lag": float(t[2]),
            "f": float(b[3]), "t_f": float(t[3]),
            "f_lag": float(b[4]), "t_f_lag": float(t[4])}


def tradable_gamma(panel: pd.DataFrame) -> dict:
    """The *univariate* predictive coefficient — the only one a trader can actually bank.

    ``a_{t+1} = alpha + gamma x_t``, with no control for tomorrow's home move (a trader
    does not know it). This is deliberately *not* the same statistic as ``beta_lag`` from
    :func:`stale_catchup_test`, and the gap between them is informative: because the home
    dollar move is itself negatively autocorrelated (see :func:`home_autocorrelation`), the
    multivariate ``beta_lag`` is mechanically inflated relative to the univariate
    ``gamma``. A study that quoted only ``beta_lag`` would overstate what is on the table.
    """
    df = panel.dropna(subset=["a_next", "x"]).sort_index()
    b, t = hac_ols(df["a_next"].to_numpy(), [df["x"].to_numpy()])
    return {"n_obs": int(len(df)), "gamma": float(b[1]), "t_gamma": float(t[1])}


def home_autocorrelation(panel: pd.DataFrame) -> dict:
    """First-order autocorrelation of the home dollar move ``x`` (pooled within name).

    A negative value is why ``beta_lag`` exceeds ``gamma``: in the two-regressor stale
    test, part of the lagged loading is the estimator undoing the same-day loading applied
    to a mean-reverting regressor.
    """
    df = panel.copy()
    df["x_lag"] = df.groupby("adr", sort=False)["x"].shift(1)
    d = df.dropna(subset=["x", "x_lag"])
    rho = float(np.corrcoef(d["x"].to_numpy(), d["x_lag"].to_numpy())[0, 1])
    return {"rho1": rho, "n_obs": int(len(d))}


def tail_sensitivity(panel: pd.DataFrame,
                     quantiles=(0.995, 0.99, 0.975, 0.95)) -> pd.DataFrame:
    """Is ``beta_lag`` a *loading*, or the average of a handful of enormous days?

    A loading is linear: if the ADR really repays a fixed fraction of yesterday's home
    move, the coefficient should barely move when the largest lagged moves are dropped
    (selection on a regressor does not bias OLS under linearity) and should not move at all
    when they are merely capped. This function runs both knives:

    - **trim** — delete every row whose ``|x_lag|`` exceeds the ``q``-quantile;
    - **winsor** — keep every row but clip ``x`` and ``x_lag`` at that quantile.

    A genuine linear lag survives both (pinned down by
    ``test_tail_sensitivity_spares_a_linear_planted_lag``, which shows a planted lag moves
    by a few percent even at a 5% trim). A coefficient that collapses under the trim is
    telling you the effect lives in the tail of the distribution, not in the body — i.e.
    there is *something* there on huge overnight moves, and nothing on an ordinary day.
    Either way the reader gets the number rather than the adjective.
    """
    df = _lagged_frame(panel)
    rows = []
    b, t = hac_ols(df["a"].to_numpy(), [df["x"].to_numpy(), df["x_lag"].to_numpy()])
    rows.append({"mode": "full", "q": 1.0, "n": int(len(df)),
                 "beta_lag": float(b[2]), "t_lag": float(t[2])})
    for q in quantiles:
        thr = float(df["x_lag"].abs().quantile(q))
        d = df[df["x_lag"].abs() <= thr]
        b, t = hac_ols(d["a"].to_numpy(), [d["x"].to_numpy(), d["x_lag"].to_numpy()])
        rows.append({"mode": "trim", "q": float(q), "n": int(len(d)),
                     "beta_lag": float(b[2]), "t_lag": float(t[2])})
        w = df.copy()
        w["x"] = w["x"].clip(-thr, thr)
        w["x_lag"] = w["x_lag"].clip(-thr, thr)
        b, t = hac_ols(w["a"].to_numpy(), [w["x"].to_numpy(), w["x_lag"].to_numpy()])
        rows.append({"mode": "winsor", "q": float(q), "n": int(len(w)),
                     "beta_lag": float(b[2]), "t_lag": float(t[2])})
    return pd.DataFrame(rows)


def lag_bucket_table(panel: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Mean ADR return by bucket of *yesterday's* home dollar move — the bettable sort.

    This is what a trader would actually see: sort every name-day by the home move that
    already happened, and look at what the ADR did next. It is the non-parametric twin of
    :func:`tradable_gamma`, and it is immune to the leverage a single 10% day exerts on a
    regression slope. The ``t`` column is the **naive i.i.d.** t of the bucket mean and is
    printed for scale only — rows inside a bucket share dates and home markets, so it
    overstates the evidence; read the basis points, not the stars.
    """
    df = _lagged_frame(panel)
    q = pd.qcut(df["x_lag"], n_buckets, labels=False, duplicates="drop").to_numpy()
    rows = []
    for k in range(int(np.nanmax(q)) + 1):
        s = df[q == k]
        rows.append({"bucket": k + 1, "n": int(len(s)),
                     "x_lag_mean": float(s["x_lag"].mean()),
                     "a_mean_bps": float(s["a"].mean() * 1e4),
                     "naive_t": one_sample_t(s["a"].to_numpy())})
    out = pd.DataFrame(rows).set_index("bucket")
    return out


def consecutive_only(panel: pd.DataFrame, max_gap_days: int = 3) -> dict:
    """Re-run the stale test using only rows whose lag really is *yesterday*.

    Because holiday rows drop out on both sides, the within-name shift sometimes reaches
    back two or three sessions. Restricting to rows at most ``max_gap_days`` calendar days
    after the previous surviving row (3 covers a weekend) removes that ambiguity.
    """
    parts = []
    for _adr, g in panel.groupby("adr", sort=False):
        g = g.copy()
        g["x_lag"] = g["x"].shift(1)
        g["_gap"] = pd.Series(g.index, index=range(len(g))).diff().dt.days.to_numpy()
        parts.append(g)
    df = pd.concat(parts).dropna(subset=["a", "x", "x_lag", "_gap"]).sort_index()
    d = df[df["_gap"] <= max_gap_days]
    b, t = hac_ols(d["a"].to_numpy(), [d["x"].to_numpy(), d["x_lag"].to_numpy()])
    return {"n_obs": int(len(d)), "share_kept": float(len(d) / max(len(df), 1)),
            "beta_lag": float(b[2]), "t_lag": float(t[2])}


def beta_lag_bootstrap(panel: pd.DataFrame, n_boot: int = 2000, block: int = 21,
                       seed: int = 955, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for ``beta_lag``, resampling *dates* in blocks.

    Rows are collapsed to one equal-weighted observation per date first, so a block draw
    keeps a whole day's cross-section together and never splits it. The CI it produces
    answers "how much of this is a few lucky months", which the HAC ``t`` alone does not.

    **Run it on a homogeneous group** (one region, or one name). Collapsing a *mixed* panel
    averages ADRs with different home betas against an average home move and estimates
    something else entirely — on the eight-name panel the collapsed point is −0.060 against
    a pooled +0.021. The returned ``point_pooled`` is the pooled
    :func:`stale_catchup_test` coefficient precisely so that this disagreement is visible
    instead of silent: if the two do not match, the CI is not about the pooled number.
    """
    df = _lagged_frame(panel).groupby(level=0)[["a", "x", "x_lag"]].mean().sort_index()
    n = len(df)
    y = df["a"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), df["x"].to_numpy(dtype=float),
                         df["x_lag"].to_numpy(dtype=float)])
    pooled = stale_catchup_test(panel)["beta_lag"]
    if n < block + 3:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_le_zero": float("nan"), "n_dates": int(n),
                "point_pooled": float(pooled)}
    point = float(np.linalg.lstsq(X, y, rcond=None)[0][2])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][2]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "point_pooled": float(pooled),
            "ci_low": float(lo), "ci_high": float(hi),
            "frac_le_zero": float((boots <= 0).mean()), "n_dates": int(n),
            "n_boot": int(n_boot), "block": int(block)}


def per_name_betas(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ADR same-day and lagged loadings on the home dollar move (HAC ``t``)."""
    rows = []
    for adr, g in panel.groupby("adr", sort=True):
        g = g.copy()
        g["x_lag"] = g["x"].shift(1)
        g = g.dropna(subset=["a", "x", "x_lag"])
        b, t = hac_ols(g["a"].to_numpy(), [g["x"].to_numpy(), g["x_lag"].to_numpy()])
        rows.append({"adr": adr, "region": g["region"].iloc[0], "n": len(g),
                     "beta_same_day": b[1], "t_same_day": t[1],
                     "beta_lag": b[2], "t_lag": t[2]})
    return pd.DataFrame(rows).set_index("adr")


# --------------------------------------------------------------------------- #
# 2. The residual — rolling beta, never full-sample
# --------------------------------------------------------------------------- #
def add_residual(panel: pd.DataFrame, window: int = 252,
                 beta_clip: tuple[float, float] = (0.0, 3.0)) -> pd.DataFrame:
    """Add the out-of-sample catch-up residual ``e`` (and the rolling ``beta``) per name.

    The beta on day ``t`` is estimated on the ``window`` days ending at ``t-1`` — strictly
    past data — and applied to day ``t``'s home move, so ``e_t = a_t - beta_t x_t`` is
    formed entirely from information available at the close of day ``t``. A negative ``e``
    means the ADR moved *less* than its home market's dollar move said it should: the
    under-reaction the catch-up story predicts will be repaid tomorrow.
    """
    out = []
    for adr, g in panel.groupby("adr", sort=False):
        g = g.copy()
        cov = g["a"].rolling(window).cov(g["x"]).shift(1)
        var = g["x"].rolling(window).var().shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (cov / var).clip(*beta_clip)
        g["beta"] = beta
        g["e"] = g["a"] - beta * g["x"]
        g["a_next"] = g["a"].shift(-1)
        out.append(g)
    res = pd.concat(out).sort_values(["date", "adr"])
    return res


def _cross_sectional_rank(s: pd.Series) -> pd.Series:
    """Demeaned within-date rank in roughly [-0.5, +0.5] — a bounded, outlier-proof score.

    A raw cross-sectional slope divides by the day's dispersion, which with a handful of
    names can be near zero and blow the slope up (we saw |slope| in the hundreds on a
    two-name group). Ranking fixes the denominator by construction, so the slope becomes a
    stable long-short spread return rather than a leverage-on-a-quiet-day artefact.
    """
    g = s.groupby(level=0)
    n = g.transform("size")
    return (g.rank(method="average") - (n + 1) / 2.0) / n.clip(lower=2)


def fama_macbeth(panel: pd.DataFrame, xcols=("e",), ycol: str = "a_next",
                 min_names: int = 4, standardize: str = "none") -> dict:
    """Fama-MacBeth: a cross-sectional slope each date, HAC ``t`` on the slope series.

    With eight names that all load on the same handful of home markets, a pooled ``t``
    over 40,000 rows badly overstates the evidence — the rows are not independent. Running
    the regression *within* each date and testing the time series of slopes removes the
    cross-sectional correlation by construction. Dates with fewer than ``min_names``
    usable names are skipped.

    The single-regressor case (the one this study leans on) takes a closed-form
    covariance-over-variance path, which is both exact and fast enough to run inside the
    era, region and synthetic-control loops.

    ``standardize="rank"`` replaces each regressor with its demeaned within-date rank, so
    the slope is a bounded long-short spread return in units of daily return. Use it
    whenever the group is small — a raw slope on two or three names is not interpretable.
    """
    df = panel.dropna(subset=[ycol, *xcols])
    if standardize == "rank":
        df = df.copy()
        for c in xcols:
            df[c] = _cross_sectional_rank(df[c])
    elif standardize != "none":
        raise ValueError(f"unknown standardize {standardize!r}")
    if len(xcols) == 1:
        c = xcols[0]
        g = df.groupby(level=0, sort=True)
        n = g[ycol].transform("size")
        keep = n >= min_names
        d = df[keep]
        g = d.groupby(level=0, sort=True)
        ybar = g[ycol].transform("mean")
        xbar = g[c].transform("mean")
        num = ((d[ycol] - ybar) * (d[c] - xbar)).groupby(level=0).sum()
        den = ((d[c] - xbar) ** 2).groupby(level=0).sum()
        slope = (num / den.replace(0.0, np.nan)).dropna()
        s = slope.to_numpy(dtype=float)
        return {"n_dates": int(s.size),
                f"slope_{c}": float(s.mean()) if s.size else float("nan"),
                f"t_{c}": newey_west_t(s),
                f"series_{c}": slope.rename(f"slope_{c}")}

    slopes = {c: [] for c in xcols}
    dates = []
    for dt, g in df.groupby(level=0, sort=True):
        if len(g) < min_names:
            continue
        b, _ = hac_ols(g[ycol].to_numpy(), [g[c].to_numpy() for c in xcols], lags=1)
        if not np.isfinite(b).all():
            continue
        dates.append(dt)
        for i, c in enumerate(xcols):
            slopes[c].append(b[i + 1])
    out = {"n_dates": len(dates)}
    for c in xcols:
        s = np.asarray(slopes[c], dtype=float)
        out[f"slope_{c}"] = float(np.nanmean(s)) if s.size else float("nan")
        out[f"t_{c}"] = newey_west_t(s)
        out[f"series_{c}"] = pd.Series(s, index=pd.DatetimeIndex(dates), name=f"slope_{c}")
    return out


def pooled_predictive(panel: pd.DataFrame, xcols=("a", "x")) -> dict:
    """Pooled HAC regression of ``a_{t+1}`` on ``xcols`` — the discriminating test.

    Running it with ``xcols=("a", "x")`` asks whether the home move ``x_t`` still predicts
    tomorrow once the ADR's *own* move ``a_t`` is controlled for. If it does not, whatever
    reversal the residual picks up is plain one-day stock reversal, not ADR catch-up.

    One caveat, pinned down by ``test_bounce_biases_the_own_move_control_upward``: pure
    bid-ask bounce manufactures a spuriously **positive** ``x_t`` loading here, because the
    negative own-move coefficient lands on the ``beta * x_t`` slice inside ``a_t`` and
    ``x_t`` enters to undo it. The test is therefore biased *towards* the catch-up story,
    which makes a zero reading strong evidence against it. The bounce-immune test is
    :func:`stale_catchup_test`, where yesterday's home move is independent of today's
    pricing error by construction.
    """
    df = panel.dropna(subset=["a_next", *xcols]).sort_index()
    b, t = hac_ols(df["a_next"].to_numpy(), [df[c].to_numpy() for c in xcols])
    out = {"n_obs": int(len(df)), "alpha_bps": float(b[0] * 1e4), "t_alpha": float(t[0])}
    for i, c in enumerate(xcols):
        out[f"coef_{c}"] = float(b[i + 1])
        out[f"t_{c}"] = float(t[i + 1])
    return out


# --------------------------------------------------------------------------- #
# 3. The costed rule
# --------------------------------------------------------------------------- #
def catchup_portfolio(
    panel: pd.DataFrame,
    signal_col: str = "x",
    direction: int = +1,
    weighting: str = "sign",
    cost_bps: float = 5.0,
    borrow_bps: float = 50.0,
    dollar_neutral: bool = False,
) -> pd.DataFrame:
    """A daily-rebalanced long/short ADR book driven by one per-name signal.

    Two rules share this engine, and the study runs both:

    - **the catch-up book** (``signal_col="x"``, ``direction=+1``) — long the ADRs whose
      home market rose on day ``t``, short those whose home market fell. This is the
      claim in its purest form: if part of the home move is still unpaid at the ADR's
      close, buying the ADR after a strong home session should pay tomorrow.
    - **the residual-reversal book** (``signal_col="e"``, ``direction=-1``) — long the
      ADRs that moved *less* than their home market said they should (a negative
      residual), short those that overshot. This is the claim's second half: the
      un-caught-up residue mean-reverts.

    Parameters
    ----------
    signal_col:
        The per-name signal, formed entirely from closes through day ``t``.
    direction:
        ``+1`` to go *with* the signal (continuation), ``-1`` to fade it (reversal).
    weighting:
        ``"sign"`` for equal-weight ±1 buckets, ``"linear"`` to weight by the signal's
        own magnitude. Either way the book is rescaled so **gross exposure is exactly 1**.
    cost_bps:
        One-way transaction cost in bps **of NAV**, charged on the day's turnover
        ``sum_i |w_i,t - w_i,t-1|``. The book rebalances every session, so turnover is
        close to 1 per day and cost is the dominant term — deliberately so, because that is
        the honest picture for a daily-rebalanced ADR spread.
    borrow_bps:
        Annualised stock-borrow rate charged on the short leg, accrued daily
        (``short_exposure * borrow_bps/1e4/252``). This is a **PROXY/ASSUMPTION** — there
        is no borrow tape in this study — so it is swept in :func:`borrow_sweep`.
    dollar_neutral:
        If True, cross-sectionally demean the weights so net exposure is exactly zero
        (removing the common ADR-market factor); otherwise net exposure floats near zero.

    The single execution lag is explicit: weights formed at the close of ``t`` are shifted
    one day and earn day ``t+1``'s returns. Being self-financing at gross 1, the book's
    return *is* its excess-of-cash return.
    """
    sig = panel.pivot_table(index=panel.index, columns="adr", values=signal_col)
    ret = panel.pivot_table(index=panel.index, columns="adr", values="a")
    ret = ret.reindex(columns=sig.columns)

    if direction not in (1, -1):
        raise ValueError("direction must be +1 (follow) or -1 (fade)")
    if weighting == "sign":
        w = float(direction) * np.sign(sig)
    elif weighting == "linear":
        w = float(direction) * sig
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    if dollar_neutral:
        w = w.sub(w.mean(axis=1), axis=0)
    gross_exp = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(gross_exp, axis=0).fillna(0.0)

    w_held = w.shift(1)                     # ONE lag: formed at close t, held over t+1
    gross = (w_held * ret).sum(axis=1, min_count=1)
    turnover = (w_held - w_held.shift(1)).abs().sum(axis=1)
    short_exposure = w_held.clip(upper=0.0).abs().sum(axis=1)
    cost = turnover * cost_bps * 1e-4
    borrow = short_exposure * borrow_bps * 1e-4 / TRADING_DAYS
    out = pd.DataFrame({
        "gross": gross,
        "cost": cost,
        "borrow": borrow,
        "net": gross - cost - borrow,
        "turnover": turnover,
        "net_exposure": w_held.sum(axis=1),
        "short_exposure": short_exposure,
    })
    return out.dropna(subset=["gross", "turnover"])


#: The two named rules, as (signal_col, direction) pairs.
BOOKS = {
    "catchup": ("x", +1),      # follow the home move
    "residual": ("e", -1),     # fade the un-caught-up residue
    "reversal": ("a", -1),     # the control: fade the ADR's own move, no home data at all
}


def book(panel: pd.DataFrame, name: str = "catchup", **kwargs) -> pd.DataFrame:
    """Run one of the named books in :data:`BOOKS` through :func:`catchup_portfolio`."""
    sig, direction = BOOKS[name]
    return catchup_portfolio(panel, signal_col=sig, direction=direction, **kwargs)


def breakeven_cost_bps(panel: pd.DataFrame, name: str = "catchup",
                       weighting: str = "sign", dollar_neutral: bool = False) -> float:
    """The one-way cost (bps) at which the gross edge is exactly eaten by friction."""
    bt = book(panel, name, weighting=weighting, cost_bps=0.0, borrow_bps=0.0,
              dollar_neutral=dollar_neutral)
    turn = bt["turnover"].mean()
    if turn <= 0:
        return float("nan")
    return float(bt["gross"].mean() / turn * 1e4)


def cost_sweep(panel: pd.DataFrame, grid=(0.0, 1.0, 2.0, 5.0, 10.0, 25.0),
               name: str = "catchup", borrow_bps: float = 50.0) -> pd.DataFrame:
    """Net Sharpe / annual return of a book across one-way costs."""
    rows = []
    for c in grid:
        s = summary(book(panel, name, cost_bps=c, borrow_bps=borrow_bps)["net"])
        rows.append({"cost_bps": c, "ann_return": s["ann_return"],
                     "sharpe": s["sharpe"], "tstat": s["tstat"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def borrow_sweep(panel: pd.DataFrame, grid=(0.0, 25.0, 50.0, 100.0, 300.0),
                 name: str = "catchup", cost_bps: float = 5.0) -> pd.DataFrame:
    """Net Sharpe across assumed annual borrow rates (the study's one non-tape input)."""
    rows = []
    for b in grid:
        s = summary(book(panel, name, cost_bps=cost_bps, borrow_bps=b)["net"])
        rows.append({"borrow_bps": b, "ann_return": s["ann_return"],
                     "sharpe": s["sharpe"], "tstat": s["tstat"]})
    return pd.DataFrame(rows).set_index("borrow_bps")


# --------------------------------------------------------------------------- #
# Robustness cuts
# --------------------------------------------------------------------------- #
def era_cut(panel: pd.DataFrame, split: str = "2015-01-01") -> dict:
    """Split the sample and re-run the stale test, the FM slope and the gross book."""
    out = {}
    for tag, mask in [("early", panel.index < pd.Timestamp(split)),
                      ("late", panel.index >= pd.Timestamp(split))]:
        p = panel[mask]
        if len(p) < 500:
            out[tag] = None
            continue
        stale = stale_catchup_test(p)
        fm = fama_macbeth(p, xcols=("e",), standardize="rank")
        disc = pooled_predictive(p, xcols=("a", "x"))
        trad = tradable_gamma(p)
        s = summary(book(p, "catchup", cost_bps=0.0, borrow_bps=0.0)["gross"])
        sr = summary(book(p, "residual", cost_bps=0.0, borrow_bps=0.0)["gross"])
        out[tag] = {
            "n_obs": stale["n_obs"],
            "beta_lag": stale["beta_lag"], "t_lag": stale["t_lag"],
            "gamma": trad["gamma"], "t_gamma": trad["t_gamma"],
            "fm_slope_e": fm["slope_e"], "fm_t_e": fm["t_e"],
            "coef_x_ctrl": disc["coef_x"], "t_x_ctrl": disc["t_x"],
            "gross_sharpe": s["sharpe"], "gross_t": s["tstat"],
            "gross_ann": s["ann_return"],
            "resid_sharpe": sr["sharpe"], "resid_t": sr["tstat"],
        }
    return out


def region_cut(panel: pd.DataFrame) -> pd.DataFrame:
    """The same battery, region by region — Tokyo's overnight window is the long one.

    Tokyo closes at 02:00 New York time, so a Japanese ADR has a full US session in which
    to catch up. London and Frankfurt close at 11:30 ET — *inside* the US session — so
    most of their move is already inside the ADR's close-to-close window and only a stub
    remains. If catch-up exists anywhere, it should be strongest in Japan.
    """
    rows = []
    for region, g in panel.groupby("region", sort=True):
        stale = stale_catchup_test(g)
        fm = fama_macbeth(g, xcols=("e",), min_names=2, standardize="rank")
        disc = pooled_predictive(g, xcols=("a", "x"))
        trad = tradable_gamma(g)
        rho = home_autocorrelation(g)
        rows.append({"region": region, "n_obs": stale["n_obs"],
                     "beta_same_day": stale["beta_same_day"],
                     "beta_lag": stale["beta_lag"], "t_lag": stale["t_lag"],
                     "gamma": trad["gamma"], "t_gamma": trad["t_gamma"],
                     "rho1_x": rho["rho1"],
                     "fm_slope_e": fm["slope_e"], "fm_t_e": fm["t_e"],
                     "coef_x_ctrl": disc["coef_x"], "t_x_ctrl": disc["t_x"]})
    return pd.DataFrame(rows).set_index("region")


def sub_era_table(panel: pd.DataFrame,
                  edges=("2004-01-01", "2010-01-01", "2015-01-01",
                         "2021-01-01", "2027-01-01")) -> pd.DataFrame:
    """The stale coefficient and the tradable gamma block by block, for era-robustness."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (panel.index >= pd.Timestamp(lo)) & (panel.index < pd.Timestamp(hi))
        p = panel[m]
        if len(p) < 400:
            continue
        s = stale_catchup_test(p)
        t = tradable_gamma(p)
        rows.append({"from": lo[:4], "to": str(int(hi[:4]) - 1), "n_obs": s["n_obs"],
                     "beta_same_day": s["beta_same_day"],
                     "beta_lag": s["beta_lag"], "t_lag": s["t_lag"],
                     "gamma": t["gamma"], "t_gamma": t["t_gamma"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, window: int = 252,
                     cost_bps: float = 0.0, borrow_bps: float = 0.0) -> dict:
    """Run the full battery on a synthetic panel with a known planted catch-up lag.

    On the planted world the stale-catchup ``beta_lag`` must be clearly positive, the home
    move must survive the own-move control, and the gross catch-up book must earn. On the
    null all three must be quiet. This proves the machinery is unbiased; it never supports
    a real-tape stamp.
    """
    p = add_residual(panel, window=window)
    stale = stale_catchup_test(p)
    fm = fama_macbeth(p, xcols=("e",))
    disc = pooled_predictive(p, xcols=("a", "x"))
    bt = book(p, "catchup", cost_bps=cost_bps, borrow_bps=borrow_bps)
    s = summary(bt["gross"])
    sr = summary(book(p, "residual", cost_bps=cost_bps, borrow_bps=borrow_bps)["gross"])
    return {
        "beta_same_day": stale["beta_same_day"],
        "beta_lag": stale["beta_lag"], "t_lag": stale["t_lag"],
        "fm_slope_e": fm["slope_e"], "fm_t_e": fm["t_e"],
        "coef_x_ctrl": disc["coef_x"], "t_x_ctrl": disc["t_x"],
        "gross_sharpe": s["sharpe"], "gross_t": s["tstat"],
        "gross_ann": s["ann_return"], "turnover": float(bt["turnover"].mean()),
        "resid_sharpe": sr["sharpe"], "resid_t": sr["tstat"],
        "n_obs": stale["n_obs"],
    }
