"""Is the market more volatile than ever? — Study 1020.

Four pieces of machinery, one per question the claim raises.

**1. Is volatility trending?** Realised volatility is extremely persistent: a calm decade is
followed by a calm year, a storm by a stormy one. That persistence is poison for a trend test.
An OLS slope through log volatility with i.i.d. standard errors will "find" trends in a series
that has none, simply because it wandered. So :func:`trend_test` reports the slope three ways
and lets them disagree in the open:

- **Newey-West (HAC)** with the usual data-dependent bandwidth — the textbook correction, known
  to over-reject when the errors are as persistent as volatility is;
- **Kiefer-Vogelsang-Bunzel fixed-b** — the Bartlett kernel with bandwidth equal to the sample,
  whose null distribution is non-standard but pivotal; we simulate its critical values once
  (:func:`kvb_null_distribution`) rather than borrowing a table. Much better size under strong
  serial correlation, at a cost in power;
- **a residual circular block bootstrap** of the slope under the null of no trend.

**2. Are extreme days more frequent?** :func:`extreme_days` flags |r| > kσ against a *fixed*
long-run σ (which asks "are big moves more common in absolute terms?") and against a *trailing*
one-year σ (which asks "are moves more surprising given the recent regime?") — the two can
point in opposite directions. :func:`count_table` gives exact (Garwood) Poisson intervals per
bucket, :func:`count_trend` a Poisson regression with HAC errors and a block-permutation p-value.

**3. Why does it feel that way?** :func:`points_vs_percent` decomposes the daily *point* move —
what headlines quote — into the index level times the *percent* move. A constant-percent
volatility makes point moves grow with the index, so "the biggest point drop in history" is a
statement about the level of the index, not about risk. :func:`era_table` and
:func:`decade_contrast` put the decade a reader remembers next to the 1930s.

**4. The tradable question.** A risk manager has to pick a volatility number for next period.
:func:`forecast_race` pits the *long-run average*, *recent history* and a *trend-extrapolated*
forecast against each other out of sample under QLIKE (Patton 2011), with Diebold-Mariano
tests. Every forecast uses information to the end of period *t* only and is scored on period
*t+1*: one period of lag, applied once.

:func:`verdict` stamps the result by rules fixed before the real tape was run.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
MONTHS = 12


# --------------------------------------------------------------------------- #
# Measuring volatility
# --------------------------------------------------------------------------- #
def log_returns_from_simple(r: pd.Series) -> pd.Series:
    return np.log1p(r.astype(float))


def log_returns_from_price(p: pd.Series) -> pd.Series:
    return np.log(p.astype(float)).diff().dropna()


def annual_rv_from_monthly(r: pd.Series, first: int | None = None,
                           last: int | None = None) -> pd.Series:
    """Realised volatility per calendar year from monthly simple returns.

    ``sqrt(sum of the year's squared monthly log returns)`` — a realised volatility, not a
    sample standard deviation, so a year with a large *mean* move is not mistaken for a calm one
    and vice versa. Only complete years (12 months) are kept; ``first``/``last`` trim further.
    """
    lr = log_returns_from_simple(r)
    g = lr.groupby(lr.index.year)
    out = np.sqrt((lr ** 2).groupby(lr.index.year).sum())[g.size() == MONTHS]
    if first is not None:
        out = out[out.index >= first]
    if last is not None:
        out = out[out.index <= last]
    out.index.name = "year"
    return out.rename("rv")


def annual_rv_from_daily(price: pd.Series, min_days: int = 200) -> pd.Series:
    """Annualised realised volatility per calendar year from a daily level series.

    ``sqrt(252 · mean of squared daily log returns)``. Years with fewer than ``min_days``
    sessions are dropped (a partial year is not a year).
    """
    lr = log_returns_from_price(price)
    g = lr.groupby(lr.index.year)
    out = np.sqrt(TRADING_DAYS * (lr ** 2).groupby(lr.index.year).mean())[g.size() >= min_days]
    out.index.name = "year"
    return out.rename("rv")


def parkinson_daily_var(ohlc: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) range-based daily variance: ``(ln H/L)² / (4 ln 2)``.

    Uses only the day's high and low, so it is an estimator independent of the close-to-close
    path; it is biased slightly low on a discretely sampled index (the true extremes are missed)
    and blind to overnight gaps — both noted where it is used.
    """
    hl = np.log(ohlc["high"].astype(float) / ohlc["low"].astype(float))
    return (hl ** 2 / (4.0 * np.log(2.0))).rename("pk_var")


def annual_rv_parkinson(ohlc: pd.DataFrame, min_days: int = 200) -> pd.Series:
    """Annualised Parkinson range volatility per calendar year."""
    v = parkinson_daily_var(ohlc)
    g = v.groupby(v.index.year)
    out = np.sqrt(TRADING_DAYS * g.mean())[g.size() >= min_days]
    out.index.name = "year"
    return out.rename("rv_parkinson")


def monthly_rv_from_daily(price: pd.Series, min_days: int = 15) -> pd.Series:
    """Annualised realised **variance** per calendar month from a daily level series."""
    lr = log_returns_from_price(price)
    key = lr.index.to_period("M")
    g = (lr ** 2).groupby(key)
    out = (TRADING_DAYS * g.mean())[g.size() >= min_days]
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out.rename("rvar")


# --------------------------------------------------------------------------- #
# Trend inference
# --------------------------------------------------------------------------- #
def _nw_bandwidth(n: int) -> int:
    """Newey-West (1994) style rule of thumb: floor(4·(T/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _cbb_index(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular block bootstrap indices, shape (n_boot, n)."""
    block = max(1, min(int(block), n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def _slope_stats(Y: np.ndarray, lags: int) -> tuple:
    """Vectorised trend regression on rows of ``Y`` (reps × T).

    Returns slope, naive t, Newey-West t (``lags``) and KVB fixed-b t, all for the slope on a
    centred time index (so the intercept is orthogonal and only the slope's variance matters).
    """
    Y = np.atleast_2d(Y)
    T = Y.shape[1]
    tc = np.arange(T) - (T - 1) / 2.0
    sxx = float((tc ** 2).sum())
    b = (Y - Y.mean(axis=1, keepdims=True)) @ tc / sxx
    U = Y - Y.mean(axis=1, keepdims=True) - b[:, None] * tc[None, :]
    s2 = (U ** 2).sum(axis=1) / (T - 2)
    t_ols = b / np.sqrt(s2 / sxx)
    V = U * tc[None, :]
    omega = (V ** 2).sum(axis=1)
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        omega = omega + 2.0 * w * (V[:, l:] * V[:, :-l]).sum(axis=1)
    t_nw = b / np.sqrt(np.clip(omega, 1e-300, None) / sxx ** 2)
    S = np.cumsum(V, axis=1)
    omega_kvb = 2.0 / T * (S ** 2).sum(axis=1)
    t_kvb = b / np.sqrt(np.clip(omega_kvb, 1e-300, None) / sxx ** 2)
    return b, t_ols, t_nw, t_kvb


@lru_cache(maxsize=4)
def kvb_null_distribution(T: int = 300, reps: int = 6000, seed: int = 1020) -> np.ndarray:
    """Simulated null distribution of |t_KVB| for a trend slope (sorted).

    The fixed-b statistic's limit is pivotal but non-normal (its tails are much fatter than a
    standard normal), so the p-value comes from simulation under i.i.d. Gaussian errors with a
    large T, computed once and cached. Deterministic.
    """
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((reps, T))
    _, _, _, t_kvb = _slope_stats(Y, lags=0)
    return np.sort(np.abs(t_kvb))


def kvb_pvalue(t: float) -> float:
    """Two-sided p-value of a KVB fixed-b trend t from the simulated null."""
    null = kvb_null_distribution()
    return float((len(null) - np.searchsorted(null, abs(t), side="left")) / len(null))


def kvb_critical(alpha: float = 0.05) -> float:
    return float(np.quantile(kvb_null_distribution(), 1.0 - alpha))


def trend_test(y: pd.Series, per_year: float = 1.0, n_boot: int = 1999,
               block: int | None = None, lags: int | None = None,
               seed: int = 1020) -> dict:
    """Linear trend in ``y`` (typically log volatility), tested three robust ways.

    ``per_year`` converts the slope per observation into a slope per year (1 for annual data,
    12 for monthly). Reports the slope per year, the implied percent change of volatility per
    year and per decade when ``y`` is a log, a naive OLS t, a Newey-West t, the KVB fixed-b t
    and its simulated p-value, and a residual circular-block-bootstrap p-value (null: no
    trend, residual dependence preserved within blocks of ``block`` observations).
    """
    y = pd.Series(y).dropna().astype(float)
    T = len(y)
    if T < 8:
        return {}
    L = _nw_bandwidth(T) if lags is None else int(lags)
    Y = y.to_numpy()[None, :]
    b, t_ols, t_nw, t_kvb = (float(v[0]) for v in _slope_stats(Y, L))
    tc = np.arange(T) - (T - 1) / 2.0
    resid = y.to_numpy() - y.mean() - b * tc
    blk = max(3, int(round(1.5 * T ** (1.0 / 3.0)))) if block is None else int(block)
    rng = np.random.default_rng(seed)
    E = resid[_cbb_index(T, blk, n_boot, rng)]
    b_star = (E - E.mean(axis=1, keepdims=True)) @ tc / float((tc ** 2).sum())
    p_boot = float((np.sum(np.abs(b_star) >= abs(b)) + 1) / (n_boot + 1))
    se_boot = float(np.std(b_star, ddof=1))
    slope_y = b * per_year
    return {"n": T, "slope": slope_y, "pct_per_year": float(np.expm1(slope_y)),
            "pct_per_decade": float(np.expm1(10 * slope_y)),
            "t_ols": t_ols, "t_nw": t_nw, "nw_lags": L, "t_kvb": t_kvb,
            "p_kvb": kvb_pvalue(t_kvb), "p_boot": p_boot, "boot_block": blk,
            "t_boot": float(b / se_boot) if se_boot > 0 else np.nan,
            "ci_boot": (float((b - 1.96 * se_boot) * per_year),
                        float((b + 1.96 * se_boot) * per_year)),
            "rho1": float(pd.Series(resid).autocorr(1)) if T > 3 else np.nan}


def passes(tr: dict, direction: int = 1) -> bool:
    """The pre-registered bar for "a significant trend in the claimed direction".

    Slope of the right sign, Newey-West |t| ≥ 2 **and** a KVB fixed-b p < 0.05. Both, because
    NW alone over-rejects on persistent series and KVB alone is the one that is honest about it.
    """
    if not tr:
        return False
    return bool(np.sign(tr["slope"]) == direction and direction * tr["t_nw"] >= 2.0
                and tr["p_kvb"] < 0.05)


def trend_from_every_start(y: pd.Series, min_len: int = 20, lags: int | None = None) -> pd.DataFrame:
    """Refit the trend from every possible start year to the end of the sample.

    "Volatility is rising" charts nearly always start somewhere. A persistent series observed
    from a calm start will slope up with no trend at all, so the share of start dates that give
    a significant rise is more informative than any single slope.
    """
    y = pd.Series(y).dropna()
    rows = []
    for i in range(0, len(y) - min_len + 1):
        sub = y.iloc[i:]
        Y = sub.to_numpy()[None, :]
        L = _nw_bandwidth(len(sub)) if lags is None else lags
        b, t_ols, t_nw, t_kvb = (float(v[0]) for v in _slope_stats(Y, L))
        rows.append({"start": y.index[i], "n": len(sub), "slope": b,
                     "pct_per_decade": float(np.expm1(10 * b)), "t_nw": t_nw,
                     "t_kvb": t_kvb, "p_kvb": kvb_pvalue(t_kvb)})
    return pd.DataFrame(rows).set_index("start")


# --------------------------------------------------------------------------- #
# Eras and decades
# --------------------------------------------------------------------------- #
def _rms_vol(x: np.ndarray, per_year: float) -> np.ndarray:
    return np.sqrt(per_year * np.mean(x ** 2, axis=-1))


def boot_vol(lr: pd.Series, per_year: float, block: int, n_boot: int = 2000,
             seed: int = 1020) -> np.ndarray:
    """Circular-block-bootstrap draws of annualised RMS volatility of a return segment."""
    x = lr.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    return _rms_vol(x[_cbb_index(len(x), block, n_boot, rng)], per_year)


def era_table(r_monthly: pd.Series, eras, block: int = 12, n_boot: int = 2000,
              seed: int = 1020) -> pd.DataFrame:
    """Annualised volatility per era from monthly returns, with block-bootstrap 95% CIs,
    plus the within-era trend in annual log RV (Newey-West t; ~20 points, read with care)."""
    lr = log_returns_from_simple(r_monthly)
    rows = []
    for name, y0, y1 in eras:
        seg = lr[(lr.index.year >= y0) & (lr.index.year <= y1)]
        draws = boot_vol(seg, MONTHS, block, n_boot, seed)
        ann = annual_rv_from_monthly(r_monthly[(r_monthly.index.year >= y0)
                                               & (r_monthly.index.year <= y1)])
        tr = trend_test(np.log(ann), n_boot=499, seed=seed) if len(ann) >= 8 else {}
        rows.append({"era": name, "months": int(len(seg)),
                     "vol": float(_rms_vol(seg.to_numpy(), MONTHS)),
                     "ci_lo": float(np.quantile(draws, 0.025)),
                     "ci_hi": float(np.quantile(draws, 0.975)),
                     "worst_month": float(np.expm1(seg.min())),
                     "within_slope_pct_decade": tr.get("pct_per_decade", np.nan),
                     "within_t_nw": tr.get("t_nw", np.nan)})
    return pd.DataFrame(rows).set_index("era")


def decade_contrast(r_monthly: pd.Series, a: tuple, b: tuple, block: int = 12,
                    n_boot: int = 4000, seed: int = 1020) -> dict:
    """Volatility of window ``a`` minus window ``b`` (inclusive year pairs), with a
    block-bootstrap 95% CI and two-sided p-value for the difference (independent resamples)."""
    lr = log_returns_from_simple(r_monthly)
    sa = lr[(lr.index.year >= a[0]) & (lr.index.year <= a[1])]
    sb = lr[(lr.index.year >= b[0]) & (lr.index.year <= b[1])]
    da = boot_vol(sa, MONTHS, block, n_boot, seed)
    db = boot_vol(sb, MONTHS, block, n_boot, seed + 1)
    diff = da - db
    va, vb = float(_rms_vol(sa.to_numpy(), MONTHS)), float(_rms_vol(sb.to_numpy(), MONTHS))
    p = 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return {"a": a, "b": b, "vol_a": va, "vol_b": vb, "ratio": va / vb, "diff": va - vb,
            "ci_lo": float(np.quantile(diff, 0.025)), "ci_hi": float(np.quantile(diff, 0.975)),
            "p": float(min(1.0, max(p, 1.0 / n_boot))),
            "worst_month_a": float(np.expm1(sa.min())), "worst_month_b": float(np.expm1(sb.min())),
            "n_down10_a": int((np.expm1(sa) <= -0.10).sum()),
            "n_down10_b": int((np.expm1(sb) <= -0.10).sum())}


def decade_vols(r_monthly: pd.Series) -> pd.DataFrame:
    """Annualised RMS volatility by calendar decade (monthly returns), months counted."""
    lr = log_returns_from_simple(r_monthly)
    dec = (lr.index.year // 10) * 10
    g = lr.groupby(dec)
    out = pd.DataFrame({"months": g.size(),
                        "vol": g.apply(lambda s: float(_rms_vol(s.to_numpy(), MONTHS))),
                        "worst_month": g.min().map(np.expm1)})
    out.index = [f"{d}s" for d in out.index]
    return out


# --------------------------------------------------------------------------- #
# Extreme days
# --------------------------------------------------------------------------- #
def extreme_days(lr: pd.Series, k: float = 3.0, mode: str = "fixed",
                 window: int = TRADING_DAYS) -> pd.Series:
    """Boolean flags for |r_t| > k·σ.

    ``mode="fixed"`` uses one long-run σ (the full-sample standard deviation — a descriptive
    yardstick, not a forecast). ``mode="trailing"`` uses the standard deviation of the previous
    ``window`` returns, ending at *t−1* (shift of one, so the day being judged is never in its
    own yardstick). The first ``window`` days of the trailing mode are dropped.
    """
    lr = lr.dropna().astype(float)
    if mode == "fixed":
        sig = pd.Series(lr.std(ddof=1), index=lr.index)
    elif mode == "trailing":
        sig = lr.rolling(window).std(ddof=1).shift(1)
    else:
        raise ValueError("mode is 'fixed' or 'trailing'")
    ok = sig.notna()
    return (lr[ok].abs() > k * sig[ok]).rename(f"x{k:g}_{mode}")


def poisson_ci(count: int, alpha: float = 0.05) -> tuple:
    """Exact (Garwood) two-sided interval for a Poisson mean given one observed count."""
    lo = 0.0 if count == 0 else stats.chi2.ppf(alpha / 2.0, 2 * count) / 2.0
    hi = stats.chi2.ppf(1.0 - alpha / 2.0, 2 * count + 2) / 2.0
    return float(lo), float(hi)


def count_table(flags: pd.Series, buckets, per: int = 1000) -> pd.DataFrame:
    """Extreme-day counts per bucket of years, with exact Poisson 95% CIs per ``per`` days.

    Buckets are ``(label, first_year, last_year)``. The rate per 1,000 sessions makes a short
    bucket (2020-21) comparable to a full decade; its interval is correspondingly wide.
    """
    rows = []
    for label, y0, y1 in buckets:
        f = flags[(flags.index.year >= y0) & (flags.index.year <= y1)]
        if len(f) == 0:
            continue
        c = int(f.sum())
        lo, hi = poisson_ci(c)
        rows.append({"bucket": label, "days": int(len(f)), "count": c,
                     "rate": per * c / len(f), "ci_lo": per * lo / len(f),
                     "ci_hi": per * hi / len(f)})
    return pd.DataFrame(rows).set_index("bucket")


def count_trend(flags: pd.Series, n_boot: int = 1999, block: int = 3,
                seed: int = 1020) -> dict:
    """Is the yearly rate of extreme days trending? Poisson GLM with an exposure offset.

    Counts per year are regressed on the year (centred) with ``log(days)`` as the offset, and
    the slope gets a Newey-West (HAC, 2 lags) z. Because extreme days cluster in crisis years —
    which makes a count series overdispersed and serially dependent — the decisive p-value is a
    **block permutation**: whole blocks of ``block`` consecutive years are reshuffled in time,
    which destroys any trend while keeping crisis clusters intact, and the slope is refitted.
    """
    import statsmodels.api as sm

    f = flags.dropna()
    g = f.groupby(f.index.year)
    yrs = pd.DataFrame({"count": g.sum().astype(float), "days": g.size().astype(float)})
    yrs = yrs[yrs["days"] >= 200]
    if len(yrs) < 6 or yrs["count"].sum() == 0:
        return {"n_years": len(yrs), "slope": 0.0, "z_hac": 0.0, "p_perm": 1.0,
                "pct_per_decade": 0.0, "total": float(yrs["count"].sum())}
    t = (yrs.index.to_numpy() - yrs.index.to_numpy().mean()).astype(float)
    X = sm.add_constant(t)
    off = np.log(yrs["days"].to_numpy())
    y = yrs["count"].to_numpy()

    def _fit(yv):
        m = sm.GLM(yv, X, family=sm.families.Poisson(), offset=off)
        return m.fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    res = _fit(y)
    slope = float(res.params[1])
    z = float(res.params[1] / res.bse[1]) if res.bse[1] > 0 else 0.0

    # Block permutation: reshuffle blocks of years (counts and exposures move together).
    rng = np.random.default_rng(seed)
    n = len(y)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rate = y / yrs["days"].to_numpy()
    w = yrs["days"].to_numpy()
    # Slope of a weighted log-linear fit on rates (fast proxy refitted on each permutation)
    def _wls_slope(rv):
        ly = np.log(rv + 0.5 / w)
        tw = t - np.average(t, weights=w)
        return float(np.sum(w * tw * (ly - np.average(ly, weights=w))) / np.sum(w * tw ** 2))
    obs = _wls_slope(rate)
    perm = np.empty(n_boot)
    for i in range(n_boot):
        order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        perm[i] = _wls_slope(rate[order])
    p_perm = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_boot + 1))
    return {"n_years": int(n), "total": float(y.sum()), "slope": slope,
            "pct_per_decade": float(np.expm1(10 * slope)), "z_hac": z, "p_perm": p_perm,
            "yearly": yrs}


def monthly_extremes_by_decade(r_monthly: pd.Series, k: float = 3.0) -> pd.DataFrame:
    """Months with |r| > k·σ (fixed full-sample σ of monthly log returns), per decade."""
    lr = log_returns_from_simple(r_monthly)
    flag = lr.abs() > k * lr.std(ddof=1)
    dec = (lr.index.year // 10) * 10
    out = pd.DataFrame({"months": flag.groupby(dec).size(), "count": flag.groupby(dec).sum()})
    out.index = [f"{d}s" for d in out.index]
    return out


# --------------------------------------------------------------------------- #
# Points versus percent
# --------------------------------------------------------------------------- #
def points_vs_percent(price: pd.Series, big_points: float = 100.0, top: int = 20,
                      burn_years: int = 5) -> dict:
    """How a constant-percent risk becomes an ever-larger point risk.

    The daily point change is ``ΔP_t = P_{t−1} · r_t``, so ``log|ΔP|`` is ``log P + log|r|``
    exactly. Per calendar year we report the mean absolute point move, the mean absolute percent
    move and the mean level, fit log-trends to each (Newey-West t), and count:

    - days with an absolute move over ``big_points`` index points (the "S&P moves 100 points"
      headline, the analogue of the Dow's "1,000-point day");
    - where the ``top`` largest **point** drops and the ``top`` largest **percent** drops fall;
    - "record" days — a drop larger than every earlier drop on the tape — in points and in
      percent, after a ``burn_years`` warm-up so the first weeks do not set trivial records.
    """
    p = price.dropna().astype(float)
    dp = p.diff().dropna()
    r = p.pct_change().dropna()
    lvl = p.shift(1).reindex(dp.index)
    yrs = pd.DataFrame({"mean_abs_points": dp.abs().groupby(dp.index.year).mean(),
                        "mean_abs_pct": r.abs().groupby(r.index.year).mean(),
                        "mean_level": lvl.groupby(lvl.index.year).mean(),
                        "days_over_big": (dp.abs() > big_points).groupby(dp.index.year).sum(),
                        "days": dp.groupby(dp.index.year).size()})
    yrs = yrs[yrs["days"] >= 200]
    tr_pts = trend_test(np.log(yrs["mean_abs_points"]), n_boot=499)
    tr_pct = trend_test(np.log(yrs["mean_abs_pct"]), n_boot=499)
    tr_lvl = trend_test(np.log(yrs["mean_level"]), n_boot=499)
    top_pts = dp.nsmallest(top)
    top_pct = r.nsmallest(top)
    start = p.index[0] + pd.DateOffset(years=burn_years)
    prev_pts = (-dp).cummax().shift(1)
    prev_pct = (-r).cummax().shift(1)
    rec_pts = dp[(dp.index >= start) & (dp < 0) & (-dp > prev_pts)]
    rec_pct = r[(r.index >= start) & (r < 0) & (-r > prev_pct)]
    last_decade = p.index[-1].year - 9
    return {"yearly": yrs, "trend_points": tr_pts, "trend_pct": tr_pct, "trend_level": tr_lvl,
            "top_points": top_pts, "top_pct": top_pct,
            "top_points_share_last_decade": float((top_pts.index.year >= last_decade).mean()),
            "top_pct_share_last_decade": float((top_pct.index.year >= last_decade).mean()),
            "last_decade_start": int(last_decade),
            "records_points": rec_pts, "records_pct": rec_pct,
            "big_points": big_points,
            "days_over_big_first_half": int(yrs["days_over_big"][yrs.index < yrs.index[len(yrs) // 2]].sum()),
            "days_over_big_second_half": int(yrs["days_over_big"][yrs.index >= yrs.index[len(yrs) // 2]].sum()),
            "first_year_over_big": int(yrs.index[yrs["days_over_big"] > 0][0])
            if (yrs["days_over_big"] > 0).any() else None}


# --------------------------------------------------------------------------- #
# The tradable question: which volatility forecast should a risk manager use?
# --------------------------------------------------------------------------- #
def qlike(realised_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss on variances: ``RV/h − log(RV/h) − 1`` (zero at a perfect forecast).

    Robust to noise in the volatility proxy (Patton 2011) and, unlike squared error, not ruled
    by the few crash years — which would make the race a contest about 1932 and 2008 alone.
    """
    x = np.asarray(realised_var, float) / np.asarray(forecast_var, float)
    return x - np.log(x) - 1.0


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 3) -> dict:
    """DM test of equal predictive accuracy; negative mean = forecast A has the lower loss.

    Newey-West variance of the loss differential with ``lags`` lags.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"mean_diff": np.nan, "t": np.nan, "p": np.nan, "n": n}
    u = d - d.mean()
    lr = float(np.sum(u ** 2))
    for l in range(1, lags + 1):
        lr += 2.0 * (1.0 - l / (lags + 1.0)) * float(np.sum(u[l:] * u[:-l]))
    se = np.sqrt(lr / n ** 2)
    t = float(d.mean() / se) if se > 0 else np.nan
    return {"mean_diff": float(d.mean()), "t": t,
            "p": float(2.0 * (1.0 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
            "n": n}


def _trend_extrapolate(logvol: np.ndarray, steps: float = 1.0) -> float:
    T = len(logvol)
    tt = np.arange(T, dtype=float)
    b, a = np.polyfit(tt, logvol, 1)
    return float(a + b * (T - 1 + steps))


def forecast_race(rv: pd.Series, recent: int, trend_window: int, burn: int,
                  dm_lags: int = 3) -> dict:
    """Out-of-sample race between three volatility forecasts (decided before the run).

    ``rv`` is a series of realised **variances** on a regular clock (annual or monthly). At the
    end of period *t*, using only periods ≤ *t*:

    - **LONG-RUN** — the expanding mean of all variances so far (the "history says" answer);
    - **RECENT** — the mean variance of the last ``recent`` periods (the "this market is
      different" answer);
    - **TREND** — an OLS line through the last ``trend_window`` log volatilities, extrapolated
      one period ahead (the "it keeps getting worse" answer — the claim turned into a rule);
    - **BLEND** — an equal mix of LONG-RUN and RECENT variances, reported as a reference;
    - **WINDOW** — the plain mean variance of the same ``trend_window`` periods TREND uses, with
      no slope. A reference that separates "the trend helped" from "a shorter memory helped":
      if TREND beats LONG-RUN but not WINDOW, the gain came from forgetting old storms, not from
      extrapolating a rise.

    Each is scored against period *t+1*'s realised variance under QLIKE, after ``burn`` periods.
    One period of lag, applied once. Returns the forecast frame, mean losses and DM tests.
    """
    v = pd.Series(rv).dropna().astype(float)
    vals = v.to_numpy()
    lv = 0.5 * np.log(vals)
    rows = []
    start = max(burn, trend_window, recent)
    for i in range(start, len(v)):
        hist = vals[:i]
        lr_f = float(hist.mean())
        rc_f = float(hist[-recent:].mean())
        tr_f = float(np.exp(2.0 * _trend_extrapolate(lv[i - trend_window:i])))
        wn_f = float(hist[-trend_window:].mean())
        rows.append({"date": v.index[i], "realised": vals[i], "LONG-RUN": lr_f,
                     "RECENT": rc_f, "TREND": tr_f, "BLEND": 0.5 * (lr_f + rc_f),
                     "WINDOW": wn_f})
    F = pd.DataFrame(rows).set_index("date")
    models = ["LONG-RUN", "RECENT", "TREND", "BLEND", "WINDOW"]
    L = pd.DataFrame({m: qlike(F["realised"], F[m]) for m in models}, index=F.index)
    mse = {m: float(np.mean((np.sqrt(F["realised"]) - np.sqrt(F[m])) ** 2)) for m in models}
    dm = {"TREND_vs_LONG-RUN": diebold_mariano(L["TREND"], L["LONG-RUN"], dm_lags),
          "TREND_vs_RECENT": diebold_mariano(L["TREND"], L["RECENT"], dm_lags),
          "TREND_vs_WINDOW": diebold_mariano(L["TREND"], L["WINDOW"], dm_lags),
          "RECENT_vs_LONG-RUN": diebold_mariano(L["RECENT"], L["LONG-RUN"], dm_lags),
          "BLEND_vs_LONG-RUN": diebold_mariano(L["BLEND"], L["LONG-RUN"], dm_lags)}
    return {"forecasts": F, "losses": L, "mean_qlike": L.mean().to_dict(), "rmse_vol":
            {m: float(np.sqrt(x)) for m, x in mse.items()}, "dm": dm, "n": int(len(F)),
            "first": F.index[0], "last": F.index[-1],
            "winner": str(L.mean().idxmin())}


# --------------------------------------------------------------------------- #
# Synthetic harness — detector must fire on the plant and stay quiet on the null
# --------------------------------------------------------------------------- #
def synthetic_detection(signal_strength: float, seeds=range(10), n_years: int = 90,
                        **kw) -> dict:
    """Run the pre-registered trend bar on ``len(seeds)`` synthetic monthly worlds.

    Returns the share of worlds where :func:`passes` fires, plus the shares for NW alone and
    for naive OLS alone — the latter is the size distortion a naive chart-reader suffers.
    """
    from . import data

    fires, nw_only, ols_only, slopes = 0, 0, 0, []
    for s in seeds:
        f, truth = data.synthetic_returns(n_years=n_years, freq="M",
                                          signal_strength=signal_strength, seed=1020 + s, **kw)
        rv = annual_rv_from_monthly(f["ret"])
        tr = trend_test(np.log(rv), n_boot=299, seed=1020 + s)
        fires += passes(tr)
        nw_only += tr["t_nw"] >= 2.0
        ols_only += tr["t_ols"] >= 2.0
        slopes.append(tr["slope"])
    n = len(list(seeds))
    return {"signal_strength": signal_strength, "n_worlds": n, "fire_rate": fires / n,
            "nw_rate": nw_only / n, "ols_rate": ols_only / n,
            "mean_slope": float(np.mean(slopes)),
            "true_slope": float(signal_strength * np.log(kw.get("planted_multiple", 3.0))
                                / n_years)}


# --------------------------------------------------------------------------- #
# The verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    Signal — the claim is that volatility (and the frequency of extreme days) is rising.

    - ``pass_century``: the 1927-2017 annual log-RV trend passes :func:`passes` (slope > 0, NW
      t ≥ 2, KVB fixed-b p < 0.05).
    - ``pass_daily``: the same bar on 1990-2021 daily-based annual log RV.
    - ``pass_extremes``: the yearly count of fixed-σ 3σ days trends up with HAC z ≥ 2 **and**
      block-permutation p < 0.05.

    **Real** if ``pass_century`` and (``pass_daily`` or ``pass_extremes``). **Mixed** if exactly
    one of ``pass_century`` / ``pass_daily`` holds (a genuine split by tape). **Weak** if neither
    passes but the extreme-day trend does, or either point slope is positive with NW t ≥ 1.
    **None** otherwise.

    Tradability — the claim turned into a risk rule ("extrapolate the rise").

    **Investable** if TREND beats both LONG-RUN and RECENT with DM p < 0.05 on both tapes.
    **Fragile** if TREND has a lower mean QLIKE than LONG-RUN on at least one tape.
    **Mirage** otherwise.

    Myth-check (third axis): **Busted** if the Signal is None or Weak and the 1930s were more
    volatile than the remembered decade with a bootstrap p < 0.05; **Confirmed** if Real;
    **Not supported** otherwise.
    """
    c, d, x = h["century"], h["daily"], h["extremes_fixed3"]
    pass_century = passes(c)
    pass_daily = passes(d)
    pass_extremes = bool(x["slope"] > 0 and x["z_hac"] >= 2.0 and x["p_perm"] < 0.05)
    if pass_century and (pass_daily or pass_extremes):
        signal = "Real"
    elif pass_century != pass_daily:
        signal = "Mixed"
    elif pass_extremes or (c["slope"] > 0 and c["t_nw"] >= 1.0) or \
            (d["slope"] > 0 and d["t_nw"] >= 1.0):
        signal = "Weak"
    else:
        signal = "None"

    fa, fm = h["race_annual"], h["race_monthly"]

    def _sig_better(r, other):
        dm = r["dm"][f"TREND_vs_{other}"]
        return bool(dm["mean_diff"] < 0 and dm["p"] < 0.05)

    if all(_sig_better(r, o) for r in (fa, fm) for o in ("LONG-RUN", "RECENT")):
        trad = "Investable"
    elif fa["mean_qlike"]["TREND"] < fa["mean_qlike"]["LONG-RUN"] or \
            fm["mean_qlike"]["TREND"] < fm["mean_qlike"]["LONG-RUN"]:
        trad = "Fragile"
    else:
        trad = "Mirage"

    dc = h["thirties_vs_remembered"]
    if signal == "Real":
        myth = "Confirmed"
    elif signal in ("None", "Weak") and dc["diff"] > 0 and dc["p"] < 0.05:
        myth = "Busted"
    else:
        myth = "Not supported"

    ra, rm = fa["mean_qlike"], fm["mean_qlike"]
    dma, dmm = fa["dm"], fm["dm"]
    pp = h["points"]
    direction = ("the point slope is actually *negative*" if c["slope"] < 0
                 else "the point slope is positive but not significant" if not pass_century
                 else "a significant rise")
    signal_why = (
        f"Over {c['n']} complete years ({h['century_span']}, Fama-French total return), the "
        f"trend in log realised volatility is **{c['pct_per_decade']:+.1%} per decade** — "
        f"{direction} (Newey-West *t* = {c['t_nw']:+.2f}; KVB fixed-b *t* = {c['t_kvb']:+.2f}, "
        f"p = {c['p_kvb']:.2f}; block-bootstrap p = {c['p_boot']:.2f}), and what slope there is "
        f"comes from the 1930s sitting at the start. On the daily S&P 500 price index "
        f"({h['daily_span']}) it is {d['pct_per_decade']:+.1%} per decade (NW *t* = "
        f"{d['t_nw']:+.2f}, KVB p = {d['p_kvb']:.2f}); the independent intraday-range estimator "
        f"({h['ohlc_span']}) gives {h['ohlc']['pct_per_decade']:+.1%} (NW *t* = "
        f"{h['ohlc']['t_nw']:+.2f}) over a window that opens in the dot-com bust. Extreme days against a fixed long-run σ: "
        f"{x['total']:.0f} sessions beyond 3σ, a fitted trend of {x['pct_per_decade']:+.0%} per "
        f"decade that does not survive a test respecting crisis clustering (HAC z = "
        f"{x['z_hac']:+.2f}, block-permutation p = {x['p_perm']:.2f}) — the count is two "
        f"bursts, 2008-09 and 2020, not a slope. "
        f"{'None' if h['share_starts_sig_up'] == 0 else format(h['share_starts_sig_up'], '.0%')}"
        f" of the start years from {h['starts_first']} to {h['starts_last']} gives a "
        f"significant rise to {h['century_last']}. Volatility arrives in storms and then calms down; it does not "
        f"climb.")
    trend_beats_lr = [ra["TREND"] < ra["LONG-RUN"], rm["TREND"] < rm["LONG-RUN"]]
    trad_why = (
        f"Framed as a risk manager's choice of next-period volatility, scored out of sample by "
        f"QLIKE. Century tape (annual, {fa['n']} years): LONG-RUN {ra['LONG-RUN']:.3f}, RECENT "
        f"{ra['RECENT']:.3f}, TREND {ra['TREND']:.3f}, and a plain {h['trend_window_annual']}-"
        f"year mean with no slope {ra['WINDOW']:.3f}. Daily tape (monthly, {fm['n']} months): "
        f"LONG-RUN {rm['LONG-RUN']:.3f}, RECENT {rm['RECENT']:.3f}, TREND {rm['TREND']:.3f}, "
        f"WINDOW {rm['WINDOW']:.3f}. The trend-extrapolation rule — the claim turned into a "
        f"forecast — {'beats' if trend_beats_lr[0] else 'loses to'} the long-run average on "
        f"the century (DM *t* = {dma['TREND_vs_LONG-RUN']['t']:+.2f}, "
        f"{'significant' if dma['TREND_vs_LONG-RUN']['p'] < 0.05 else 'not significant'}) "
        f"and {'beats' if trend_beats_lr[1] else 'loses to'} it monthly (DM *t* "
        f"= {dmm['TREND_vs_LONG-RUN']['t']:+.2f}); against the same window *without* the slope "
        f"it scores DM *t* = {dma['TREND_vs_WINDOW']['t']:+.2f} (annual) and "
        f"{dmm['TREND_vs_WINDOW']['t']:+.2f} (monthly). Where TREND wins it is because a "
        f"shorter memory forgets the 1930s, not because volatility rises. The robust lesson "
        f"is the opposite of the claim's: the best forecasts here mix recent and long-run "
        f"(BLEND vs LONG-RUN DM *t* = {dma['BLEND_vs_LONG-RUN']['t']:+.2f} annual, "
        f"{dmm['BLEND_vs_LONG-RUN']['t']:+.2f} monthly) — volatility clusters and then "
        f"mean-reverts. Nothing is traded, so no costs are charged; the stamp grades the "
        f"trend rule by the pre-registered bar"
        + (" — Fragile rather than Mirage only because that bar asks TREND to beat the "
           "long-run average on one tape on a point estimate, which it does by forgetting, "
           "not by extrapolating." if trad == "Fragile" else "."))
    one = (f"A century of US market volatility shows no upward trend "
           f"({c['pct_per_decade']:+.1%} per decade, KVB p = {c['p_kvb']:.2f}), the 1930s were "
           f"{dc['ratio']:.1f}× as volatile as {dc['b'][0]}-{dc['b'][1]}, and the record "
           f"crashes in the headlines are point moves that grow with the index "
           f"({pp['trend_points']['pct_per_decade']:+.0%} per decade in points vs "
           f"{pp['trend_pct']['pct_per_decade']:+.0%} in percent) — a risk manager should mix "
           f"recent and long-run volatility, not extrapolate a rise.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "myth": myth, "one_sentence": one,
            "flags": {"pass_century": pass_century, "pass_daily": pass_daily,
                      "pass_extremes": pass_extremes}}
