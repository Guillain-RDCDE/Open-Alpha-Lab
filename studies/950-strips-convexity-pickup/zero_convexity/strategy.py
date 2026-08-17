"""Strategy + inference for Study 950 — Zero-Coupon Convexity.

The race. **Arm A** is 100% of a zero-coupon Treasury fund (EDV, 20-30y STRIPS; ZROZ as
the cross-check). **Arm B** is the *duration-matched* coupon mix: ``L`` units of TLT plus
``(1 - L)`` units of BIL, with ``L > 1`` (the coupon long bond is shorter in duration than
the STRIPS, so matching it upward means financing a levered TLT position at bills). Both
arms are measured **excess-of-cash** — the BIL total return is subtracted from each — so
the cash leg cancels and the difference ``diff = e_zero - e_mix`` is a clean, roughly
duration-neutral spread.

The hedge ratio is solved from the **realised beta to the same rate factor**: a rolling
252-day OLS of each leg's daily excess return on the daily change in the **30-year
constant-maturity yield** (``^TYX``), giving each leg's empirical ``dr/dy``. ``L`` is their
ratio. It is estimated on data through the **last session of month m** and traded for every
session of **month m+1** — exactly one execution lag, and the only one in the study.

The headline is *not* the average of ``diff``. If convexity is genuinely being paid for,
the zero leg should **win in large-move months and lose slightly in quiet ones**, because
the second-order term ``+0.5 * C * dy**2`` only bites when ``dy`` is large. So the test is
an explicit asymmetry regression on monthly data::

    diff_m = a + b1 * dy_m + b2 * dy_m**2 + e

- ``b2 > 0`` is the convexity pickup (the payoff smile);
- ``a < 0`` is its price (the carry you give up in quiet months);
- ``b1`` should be near zero if the duration match holds at the monthly horizon — a
  significant ``b1`` means the spread is still trading the *curve* (20s vs 30s), not
  convexity, and is reported rather than hidden.

Costs and financing. Rebalancing ``L`` monthly costs ``cost_bps`` one-way x NAV on the
turnover ``|dL|``; the levered part of the mix pays a financing spread over bills. **That
spread is a PROXY** (25 bp/yr by default, swept 0-100 bp) — it is not in the tape. Both
frictions fall on Arm B only (the zero leg is buy-and-hold), so the race is deliberately
tilted *towards* the claim being tested.

Returns are **simple** (arithmetic) throughout so the mix return is exactly
``L * r_TLT + (1 - L) * r_BIL`` and wealth paths compound correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (float(u[l:] @ u[:-l]) / n)
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def hac_ols(y, X, lags: int = 6) -> dict:
    """OLS with Newey-West (Bartlett) standard errors. ``X`` must include its own intercept.

    Returns ``{"beta", "se", "t", "r2", "n"}``. Used for the asymmetry regression, where
    the monthly residuals are mildly autocorrelated (overlapping rate regimes) and a naive
    OLS *t* would be optimistic.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        A = Xu[l:].T @ Xu[:-l] / n
        S += w * (A + A.T)
    V = XtXi @ (S * n) @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "beta": beta, "se": se, "t": t,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(n),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The rate factor and the duration match
# --------------------------------------------------------------------------- #
def rate_factor(yield_pp: pd.Series) -> pd.Series:
    """Daily change in the yield level, converted from percentage points to decimals.

    ``^TYX`` quotes the 30-year yield as e.g. 4.85 (percent). A 1 bp move is 0.0001 in
    decimals — the unit in which ``-D*dy`` and ``0.5*C*dy**2`` are dimensionally correct.
    """
    return (pd.Series(yield_pp).astype(float).diff() / 100.0).rename("dy")


def rolling_rate_beta(excess: pd.Series, dy: pd.Series, window: int = 252) -> pd.Series:
    """Rolling OLS slope of a leg's daily excess return on the daily rate factor.

    The slope is the leg's realised ``dr/dy`` — a *negative* number whose magnitude is
    (approximately) its effective duration. Requires a full ``window`` (no partial
    windows), so the first year of any sample is warmup.
    """
    e = pd.Series(excess).astype(float)
    d = pd.Series(dy).astype(float)
    cov = e.rolling(window, min_periods=window).cov(d)
    var = d.rolling(window, min_periods=window).var()
    return (cov / var).rename("rate_beta")


def monthly_hedge_ratio(
    beta_zero: pd.Series,
    beta_coupon: pd.Series,
    lo: float = 0.5,
    hi: float = 3.0,
) -> pd.Series:
    """Duration-matching leverage ``L`` on the coupon leg, set monthly with a one-month lag.

    ``L = beta_zero / beta_coupon`` (a ratio of two negative slopes, hence positive), read
    at the **last session of month m** and held for every session of **month m+1**. Values
    outside ``[lo, hi]`` are treated as an unusable estimate and dropped (they only occur
    in the warmup, when the denominator is near zero).
    """
    ratio = (beta_zero / beta_coupon)
    idx = ratio.index
    per = idx.to_period("M")
    month_end = ratio.groupby(per).last()          # value at the last session of month m
    applied = month_end.shift(1)                   # ... traded through month m+1
    out = pd.Series(applied.reindex(per).to_numpy(dtype=float), index=idx, name="L")
    out[~np.isfinite(out)] = np.nan
    out[(out < lo) | (out > hi)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def run_race(
    zero: pd.Series,
    coupon: pd.Series,
    cash: pd.Series,
    yield_pp: pd.Series,
    window: int = 252,
    cost_bps: float = 3.0,
    finance_bps: float = 25.0,
) -> pd.DataFrame:
    """Race a zero-coupon fund against a duration-matched levered coupon mix.

    Parameters
    ----------
    zero, coupon, cash:
        Daily **total-return** close levels of the zero-coupon fund (EDV/ZROZ), the coupon
        long bond (TLT) and the cash leg (BIL).
    yield_pp:
        The 30-year constant-maturity yield **level in percentage points** (``^TYX``) —
        the shared rate factor both legs are matched on.
    window:
        Lookback (trading days) for the rolling rate betas. 252 = one year.
    cost_bps:
        One-way transaction cost in bps x NAV charged on the mix's monthly turnover
        ``|dL|``. The zero arm is buy-and-hold and pays nothing.
    finance_bps:
        **PROXY/ASSUMPTION** — annual financing spread over bills paid on the levered part
        ``(L - 1)`` of the mix. Not in the tape; swept in :func:`sweep_finance`.

    Returns a daily frame with ``e_zero`` (arm A excess-of-cash), ``e_mix`` (arm B,
    net of financing and turnover), ``diff`` (A - B), ``L``, ``dy`` and ``turnover``.
    """
    idx = zero.index.intersection(coupon.index).intersection(cash.index)
    idx = idx.intersection(yield_pp.index).sort_values()
    z, c, k, y = zero.loc[idx], coupon.loc[idx], cash.loc[idx], yield_pp.loc[idx]

    r_cash = k.pct_change()
    e_zero_raw = (z.pct_change() - r_cash).rename("e_zero")
    e_coupon = (c.pct_change() - r_cash).rename("e_coupon")
    dy = rate_factor(y)

    frame = pd.DataFrame({"e_zero": e_zero_raw, "e_coupon": e_coupon, "dy": dy}).dropna()
    b_zero = rolling_rate_beta(frame["e_zero"], frame["dy"], window=window)
    b_coupon = rolling_rate_beta(frame["e_coupon"], frame["dy"], window=window)
    L = monthly_hedge_ratio(b_zero, b_coupon)

    out = frame.join(L).dropna()
    turnover = out["L"].diff().abs().fillna(0.0)
    fin_daily = (finance_bps * 1e-4) / TRADING_DAYS
    e_mix = (
        out["L"] * out["e_coupon"]
        - (out["L"] - 1.0) * fin_daily
        - turnover * (cost_bps * 1e-4)
    ).rename("e_mix")
    res = pd.DataFrame({
        "e_zero": out["e_zero"], "e_mix": e_mix,
        "diff": out["e_zero"] - e_mix,
        "L": out["L"], "dy": out["dy"], "turnover": turnover,
    })
    return res


def to_monthly(race: pd.DataFrame, yield_pp: pd.Series) -> pd.DataFrame:
    """Compound the daily race to calendar months and attach the monthly yield change.

    The first month drops out on its own (a month-on-month yield change needs a prior
    month-end mark), and the study-wide as-of trims the tape at a month end, so the last
    month is whole by construction — no partial period ever reaches the regression.
    """
    m = pd.DataFrame({
        "e_zero": (1.0 + race["e_zero"]).resample("ME").prod() - 1.0,
        "e_mix": (1.0 + race["e_mix"]).resample("ME").prod() - 1.0,
        "L": race["L"].resample("ME").last(),
    })
    m["diff"] = m["e_zero"] - m["e_mix"]
    y_m = pd.Series(yield_pp).astype(float).resample("ME").last()
    m["dy"] = (y_m.reindex(m.index).diff() / 100.0)
    # Realised variance of the daily rate factor inside the month. Convexity (gamma) P&L
    # on a daily-marked fund accrues with the *path*, not only with the net move; the two
    # regressors answer slightly different questions and are both reported.
    m["rv"] = (race["dy"] ** 2).resample("ME").sum().reindex(m.index)
    return m.dropna()


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a simple-return series (pass an *excess* series)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n": int(n),
        "mean_ann": float(mu * periods_per_year),
        "vol_ann": float(sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The headline — the asymmetry (convexity) regression
# --------------------------------------------------------------------------- #
def convexity_regression(monthly: pd.DataFrame, hac_lags: int = 6,
                         regressor: str = "dy2") -> dict:
    """Regress the monthly spread on the rate move and a quadratic rate term.

    ``diff_m = a + b1 * dy_m + b2 * Q_m + e``, with ``dy`` in decimals and ``Q`` either

    - ``regressor="dy2"`` — the **squared net monthly move** ``dy_m**2``: the textbook
      second-order term for a bond repriced from one month-end yield to the next; or
    - ``regressor="rv"`` — the **realised variance** ``sum(dy_t**2)`` of the daily rate
      factor inside the month: the gamma P&L a *daily-marked, daily-compounded* fund
      actually accrues along the path.

    Coefficients are reported with ``a`` in **basis points per month**, ``b1`` as the
    residual ``dr/dy`` (so ``-b1`` is a residual duration in years) and, for readability,
    ``b2_per_25bp`` / ``b2_per_50bp`` — the extra bp/month the zero leg earns at
    ``Q = 0.0025**2`` and ``Q = 0.0050**2``.

    The convexity claim predicts ``b2 > 0`` (the smile) and ``a < 0`` (its price).
    """
    col = "dy" if regressor == "dy2" else "rv"
    m = monthly.dropna(subset=["diff", "dy", col])
    y = m["diff"].to_numpy(dtype=float)
    dy = m["dy"].to_numpy(dtype=float)
    q = dy ** 2 if regressor == "dy2" else m["rv"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(m)), dy, q])
    fit = hac_ols(y, X, lags=hac_lags)
    b = fit["beta"]
    return {
        "n": fit["n"],
        "regressor": regressor,
        "a_bp_mo": float(b[0] * 1e4),
        "a_t": float(fit["t"][0]),
        "b1": float(b[1]),
        "b1_t": float(fit["t"][1]),
        "b1_resid_duration_yrs": float(-b[1]),
        "b2": float(b[2]),
        "b2_t": float(fit["t"][2]),
        "b2_per_25bp": float(b[2] * (0.0025 ** 2) * 1e4),
        "b2_per_50bp": float(b[2] * (0.0050 ** 2) * 1e4),
        # The move a month must deliver for the convexity term to repay the intercept.
        # Only defined when convexity is positive and it is actually being paid for.
        "breakeven_move_bp": float(np.sqrt(-b[0] / b[2]) * 1e4)
        if (b[2] > 0 and b[0] < 0) else float("nan"),
        "r2": fit["r2"],
    }


def linearly_hedged_spread(monthly: pd.DataFrame) -> pd.Series:
    """The monthly spread with its residual *linear* rate exposure regressed out.

    The monthly duration match is never perfect — the two legs sit at different points of
    the long end (20-30y STRIPS versus a 20y+ coupon fund), so the spread keeps a small
    residual ``dr/dy``. This strips it out with a full-sample OLS on ``[1, dy]``. It is an
    **in-sample, descriptive** adjustment (it uses the whole sample to fit one slope) — it
    is used to look at the *shape* of the payoff, never to claim a tradable return.
    """
    m = monthly.dropna(subset=["diff", "dy"])
    X = np.column_stack([np.ones(len(m)), m["dy"].to_numpy(dtype=float)])
    b = np.linalg.pinv(X.T @ X) @ X.T @ m["diff"].to_numpy(dtype=float)
    return (m["diff"] - b[1] * m["dy"]).rename("diff_hedged")


def move_buckets(monthly: pd.DataFrame, n_buckets: int = 3, col: str = "diff") -> pd.DataFrame:
    """Mean monthly spread inside terciles of the *absolute* monthly yield move.

    The convexity story in its rawest form: the zero leg should be paid in the large-move
    bucket and pay away in the quiet bucket. Reports the mean spread, its HAC *t* and the
    hit rate per bucket. Pass ``col="diff_hedged"`` (see :func:`linearly_hedged_spread`)
    to look at the shape after the residual linear rate exposure is removed.
    """
    if col == "diff_hedged" and "diff_hedged" not in monthly.columns:
        series = linearly_hedged_spread(monthly)
    else:
        series = monthly[col]
    m = pd.DataFrame({"diff": series, "dy": monthly["dy"]}).dropna()
    m["absdy"] = m["dy"].abs()
    qs = np.array(m["absdy"].quantile(np.linspace(0, 1, n_buckets + 1)).to_numpy(), dtype=float)
    qs[0], qs[-1] = -np.inf, np.inf
    labels = ["quiet", "middling", "large"] if n_buckets == 3 else [f"q{i+1}" for i in range(n_buckets)]
    m["bucket"] = pd.cut(m["absdy"], bins=qs, labels=labels, include_lowest=True)
    rows = []
    for lab in labels:
        g = m[m["bucket"] == lab]
        if len(g) == 0:
            continue
        rows.append({
            "bucket": lab,
            "n": int(len(g)),
            "mean_absdy_bp": float(g["absdy"].mean() * 1e4),
            "mean_diff_bp_mo": float(g["diff"].mean() * 1e4),
            "t_hac": newey_west_t(g["diff"].to_numpy(), lags=3),
            "hit_rate": float((g["diff"] > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("bucket")


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(
    x: pd.Series,
    stat: str = "mean",
    n_boot: int = 2000,
    block: int = 6,
    seed: int = 950,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the mean (or annualised Sharpe) of a monthly series.

    Blocks of ``block`` consecutive months preserve the rate-regime persistence that makes
    a naive i.i.d. bootstrap over-confident.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}

    def _stat(v):
        if stat == "sharpe":
            sd = v.std(ddof=1)
            return v.mean() / sd * np.sqrt(MONTHS_PER_YEAR) if sd > 0 else np.nan
        return v.mean()

    point = float(_stat(r))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[b] = _stat(r[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


def bootstrap_b2_ci(
    monthly: pd.DataFrame,
    n_boot: int = 2000,
    block: int = 6,
    seed: int = 950,
    alpha: float = 0.05,
    regressor: str = "dy2",
) -> dict:
    """Block-bootstrap CI for the convexity coefficient ``b2`` of the asymmetry regression.

    Resamples whole blocks of consecutive months (rows of the design matrix together with
    their outcome), refits, and reports the percentile interval and the share of resamples
    with a negative ``b2``.
    """
    col = "dy" if regressor == "dy2" else "rv"
    m = monthly.dropna(subset=["diff", "dy", col])
    y = m["diff"].to_numpy(dtype=float)
    dy = m["dy"].to_numpy(dtype=float)
    q = dy ** 2 if regressor == "dy2" else m["rv"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(m)), dy, q])
    n = len(m)
    if n < block + 4:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    point = float((np.linalg.pinv(X.T @ X) @ X.T @ y)[2])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        Xb, yb = X[idx], y[idx]
        try:
            boots[b] = (np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ yb)[2]
        except np.linalg.LinAlgError:
            continue
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


# --------------------------------------------------------------------------- #
# Robustness — eras, costs, financing
# --------------------------------------------------------------------------- #
def era_cut(race: pd.DataFrame, yield_pp: pd.Series, split: str = "2018-01-01") -> dict:
    """Re-run the monthly spread and the asymmetry regression on each half of the sample."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = race.loc[sl]
        if len(sub) < 300:
            out[tag] = None
            continue
        m = to_monthly(sub, yield_pp)
        if len(m) < 24:
            out[tag] = None
            continue
        reg = convexity_regression(m)
        out[tag] = {
            "n_months": int(len(m)),
            "mean_diff_bp_mo": float(m["diff"].mean() * 1e4),
            "t_diff": newey_west_t(m["diff"].to_numpy(), lags=6),
            "b2": reg["b2"], "b2_t": reg["b2_t"],
            "b2_per_25bp": reg["b2_per_25bp"],
            "a_bp_mo": reg["a_bp_mo"], "a_t": reg["a_t"],
            "b1_t": reg["b1_t"],
        }
    return out


def cut_grid(
    races: dict,
    yield_pp: pd.Series,
    split: str = "2018-01-01",
    specs=("dy2", "rv"),
) -> pd.DataFrame:
    """Census of *every* fund x quadratic-specification x era cut, in one frame.

    ``races`` maps a label (the zero-coupon fund) to its daily race frame. For each label
    the asymmetry regression is refit on the full sample and on both halves of the ``split``,
    under both quadratic regressors — so a two-fund study yields 2 x 3 x 2 = 12 rows.

    This exists to keep the robustness prose honest. Claims of the form "the sign is the
    same in every cut" or "the best *t* anywhere is X" are only checkable if every cut that
    the design implies is actually run and printed; quoting a sign-uniformity count over a
    hand-picked subset of the grid is selection by omission, even when it is accidental.
    Returns one row per (fund, era, spec) with ``a``, ``b1``, ``b2`` and their HAC *t*'s.
    """
    eras = [("full", slice(None, None)),
            ("early", slice(None, split)),
            ("late", slice(split, None))]
    rows = []
    for fund, race in races.items():
        for era, sl in eras:
            sub = race.loc[sl]
            if len(sub) < 300:
                continue
            m = to_monthly(sub, yield_pp)
            if len(m) < 24:
                continue
            for spec in specs:
                r = convexity_regression(m, regressor=spec)
                rows.append({
                    "fund": fund, "era": era, "spec": spec, "n_months": r["n"],
                    "a_bp_mo": r["a_bp_mo"], "a_t": r["a_t"],
                    "b1_t": r["b1_t"], "b2": r["b2"], "b2_t": r["b2_t"],
                })
    return pd.DataFrame(rows)


def grid_census(grid: pd.DataFrame) -> dict:
    """Summarise a :func:`cut_grid`: sign counts, the extreme *t*, and the |t| >= 2 tally.

    ``max_abs_b2_t`` is the largest |*t*| **anywhere in the grid** — the number a study must
    quote when it says "the best we find is ...", rather than the best of the cuts it
    happened to write up.
    """
    n = len(grid)
    if n == 0:
        return {"n_cuts": 0}
    i = int(grid["b2_t"].abs().idxmax())
    return {
        "n_cuts": n,
        "b2_positive": int((grid["b2"] > 0).sum()),
        "a_negative": int((grid["a_bp_mo"] < 0).sum()),
        "sign_flips": int(((grid["b2"] <= 0) | (grid["a_bp_mo"] >= 0)).sum()),
        "n_b2_t_ge_2": int((grid["b2_t"].abs() >= 2.0).sum()),
        "max_abs_b2_t": float(grid["b2_t"].abs().max()),
        "max_cut": f"{grid.loc[i, 'fund']} {grid.loc[i, 'era']} {grid.loc[i, 'spec']}",
        "max_b2_t": float(grid.loc[i, "b2_t"]),
    }


def sweep_costs(
    zero: pd.Series, coupon: pd.Series, cash: pd.Series, yield_pp: pd.Series,
    grid=(0.0, 1.0, 3.0, 10.0, 25.0), finance_bps: float = 25.0, window: int = 252,
) -> list[dict]:
    """One-way transaction-cost sweep on the mix's monthly rebalance turnover."""
    rows = []
    for c in grid:
        race = run_race(zero, coupon, cash, yield_pp, window=window,
                        cost_bps=c, finance_bps=finance_bps)
        m = to_monthly(race, yield_pp)
        reg = convexity_regression(m)
        rows.append({
            "cost_bps": float(c),
            "mean_diff_bp_mo": float(m["diff"].mean() * 1e4),
            "t_diff": newey_west_t(m["diff"].to_numpy(), lags=6),
            "b2": reg["b2"], "b2_t": reg["b2_t"], "a_bp_mo": reg["a_bp_mo"],
        })
    return rows


def sweep_finance(
    zero: pd.Series, coupon: pd.Series, cash: pd.Series, yield_pp: pd.Series,
    grid=(0.0, 25.0, 50.0, 100.0), cost_bps: float = 3.0, window: int = 252,
) -> list[dict]:
    """Sweep the **assumed** financing spread paid on the levered part of the mix.

    The spread is the study's only non-tape input. A wider spread makes the mix worse and
    therefore flatters the zero leg — so the sweep bounds how much of any result could be
    an artefact of the assumption.
    """
    rows = []
    for f in grid:
        race = run_race(zero, coupon, cash, yield_pp, window=window,
                        cost_bps=cost_bps, finance_bps=f)
        m = to_monthly(race, yield_pp)
        reg = convexity_regression(m)
        rows.append({
            "finance_bps": float(f),
            "mean_diff_bp_mo": float(m["diff"].mean() * 1e4),
            "t_diff": newey_west_t(m["diff"].to_numpy(), lags=6),
            "b2": reg["b2"], "b2_t": reg["b2_t"], "a_bp_mo": reg["a_bp_mo"],
        })
    return rows


def sweep_window(
    zero: pd.Series, coupon: pd.Series, cash: pd.Series, yield_pp: pd.Series,
    grid=(126, 252, 504), cost_bps: float = 3.0, finance_bps: float = 25.0,
) -> list[dict]:
    """Sweep the beta-estimation lookback (how the duration match is solved)."""
    rows = []
    for w in grid:
        race = run_race(zero, coupon, cash, yield_pp, window=int(w),
                        cost_bps=cost_bps, finance_bps=finance_bps)
        m = to_monthly(race, yield_pp)
        reg = convexity_regression(m)
        rows.append({
            "window": int(w), "n_months": int(len(m)),
            "L_mean": float(race["L"].mean()),
            "mean_diff_bp_mo": float(m["diff"].mean() * 1e4),
            "t_diff": newey_west_t(m["diff"].to_numpy(), lags=6),
            "b2": reg["b2"], "b2_t": reg["b2_t"],
        })
    return rows


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery proof (never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    panel: pd.DataFrame, window: int = 252, cost_bps: float = 0.0, finance_bps: float = 0.0,
) -> dict:
    """Run the whole race + asymmetry regression on a synthetic panel.

    On the planted world (``signal_strength=1``) the harness must recover a **positive**
    ``b2`` with a convincing *t* and a **negative** intercept (the convexity is paid for);
    on the null (``signal_strength=0``) both must be indistinguishable from zero. Proves
    the detector is unbiased — it never supports a real-tape verdict.
    """
    race = run_race(panel["zero"], panel["coupon"], panel["cash"], panel["yield_pp"],
                    window=window, cost_bps=cost_bps, finance_bps=finance_bps)
    m = to_monthly(race, panel["yield_pp"])
    reg = convexity_regression(m, regressor="rv")
    reg_dy2 = convexity_regression(m, regressor="dy2")
    return {
        "n_months": int(len(m)),
        "L_mean": float(race["L"].mean()),
        "mean_diff_bp_mo": float(m["diff"].mean() * 1e4),
        "t_diff": newey_west_t(m["diff"].to_numpy(), lags=6),
        "vol_ratio": float(race["e_zero"].std() / race["e_mix"].std()),
        "b2_dy2": reg_dy2["b2"], "b2_dy2_t": reg_dy2["b2_t"],
        **{k: reg[k] for k in ("a_bp_mo", "a_t", "b1_t", "b2", "b2_t", "b2_per_25bp", "r2")},
    }
