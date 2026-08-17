"""Strategy + inference for Study 943 — Reset Frequency.

The construction. A leveraged position is a margin account: exposure ``w`` units of the
underlying, funded by ``w - 1`` units of borrowed cash. Its daily return is

    r_p(t) = w(t) * r_asset(t) - (w(t) - 1) * f(t) - cost(t)

where ``f`` is the daily financing rate (``^IRX`` + a spread — a labelled PROXY) and
``cost`` is a one-way transaction cost charged on the **change in exposure**, times NAV.
The position is long-only, so there is no stock borrow to pay; the only borrow here is
the *cash* leg, and its spread is swept.

The two arms differ only in **when the leverage is reset to its target ``L``**:

- **daily reset** — ``w(t) = L`` every day. This is exactly what SSO (2x) and UPRO (3x)
  do, and it is what retail folklore blames for "volatility decay".
- **monthly reset** — ``w`` is reset to ``L`` once a month and then left to *drift* with
  the position for the rest of the month: if the underlying falls, equity falls faster,
  so effective leverage rises (and vice versa). The reset trade is decided at the close
  of the last trading day of month *M* and is in force from the first day of month *M+1*
  — **exactly one execution lag**, and the only one in the study.

Between resets a monthly-reset account is *not* leverage-controlled, so it carries two
hazards a daily-reset fund cannot have: exposure drift to arbitrary multiples, and an
actual **margin call**. Both are modelled: ``maintenance`` is the maintenance-margin
ratio (equity / exposure) below which the broker liquidates to cash — a Reg-T-flavoured
PROXY, default 25%, swept — and equity that goes negative intraperiod is a wipeout.

We race, all excess-of-cash (minus BIL's total return):

1. **fund** — the actual SSO / UPRO tape, which already carries its ~0.9% expense ratio,
   its own financing, and its tracking error.
2. **daily-synth** — our own daily-reset replication at the same ``L`` and financing.
   Fund minus daily-synth isolates the *fee and tracking* leg of the gap.
3. **monthly-synth** — the monthly-reset replication. Monthly-synth minus daily-synth
   isolates the *reset-frequency* leg — the thing this study is actually about.

The headline questions: does the monthly reset earn a better excess-of-cash Sharpe than
the daily-reset fund, does the return difference clear an HAC *t* >= 2, does the sign
flip between trending and choppy months, and is any of it forecastable or survivable?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# --- Non-tape PROXIES / ASSUMPTIONS (all swept somewhere in the study) ------- #
DEFAULT_SPREAD_BPS = 50.0     # financing spread over ^IRX, annualised bps (PROXY)
DEFAULT_COST_BPS = 2.0        # one-way transaction cost on exposure turnover (PROXY)
DEFAULT_MAINTENANCE = 0.25    # maintenance-margin ratio triggering liquidation (PROXY)


# --------------------------------------------------------------------------- #
# Financing
# --------------------------------------------------------------------------- #
def financing_rate(irx: pd.Series, spread_bps: float = DEFAULT_SPREAD_BPS) -> pd.Series:
    """Daily financing rate from the ``^IRX`` 13-week T-bill **discount yield** (percent).

    ``irx`` is quoted as an annualised percentage (5.25 = 5.25%). We take the value known
    at the *previous* close (``shift(1)``, so no same-day information leaks into the day's
    financing), add ``spread_bps`` and divide by 252. Two labelled approximations: the
    discount yield is used directly rather than converted to a bond-equivalent ACT/360
    rate (a few bps at 5% levels), and the spread over the bill rate is an **assumption**,
    not a tape — it is swept from 0 to 200 bps in the results.
    """
    return (irx.shift(1) / 100.0 + spread_bps * 1e-4) / TRADING_DAYS


# --------------------------------------------------------------------------- #
# The replication engine
# --------------------------------------------------------------------------- #
def replicate(
    r_asset: pd.Series,
    fin: pd.Series,
    leverage: float = 2.0,
    reset: str = "daily",
    cost_bps: float = DEFAULT_COST_BPS,
    maintenance: float = DEFAULT_MAINTENANCE,
    cash_r: pd.Series | None = None,
) -> pd.DataFrame:
    """Path of a levered margin account with a ``daily`` or ``monthly`` leverage reset.

    Parameters
    ----------
    r_asset, fin:
        Daily simple returns of the underlying and the daily financing rate, aligned.
    leverage:
        Target exposure ``L`` (2.0 or 3.0 here).
    reset:
        ``"daily"`` (exposure pinned at ``L`` every day, what SSO/UPRO do) or ``"monthly"``
        (reset at the month-end close, in force from the next day, drifting in between).
    cost_bps:
        One-way cost in bps charged on ``|change in exposure| x NAV`` — so the daily-reset
        arm pays a small toll every day and the monthly arm pays once a month.
    maintenance:
        Maintenance-margin ratio (equity / exposure). When drifting exposure pushes the
        ratio below it the account is liquidated to cash at that close and stays there.
        ``0.0`` disables the check (an infinitely patient lender). A PROXY, swept.
    cash_r:
        Daily return **credited to the liquidated account** once it has been called — the
        proceeds sit in the cash leg, so the honest post-call series is the cash return,
        not zero. Pass the same cash leg the race is measured excess of (BIL) and a called
        arm then contributes exactly **zero excess return** afterwards rather than a
        spurious −cash drift for the rest of the sample. ``None`` = 0% (a mattress); a
        wiped-out account (equity gone) earns nothing either way, since there is nothing
        left to deposit.

    Returns a frame indexed like ``r_asset`` with ``r`` (daily simple return of the
    account, net of financing and cost), ``w`` (exposure in force that day) and ``alive``.
    """
    if reset not in ("daily", "monthly"):
        raise ValueError("reset must be 'daily' or 'monthly'")
    idx = r_asset.index
    ra = np.asarray(r_asset, dtype=float)
    f = np.asarray(fin, dtype=float)
    n = len(ra)
    # Pre-compute the reset calendar once (indexing a PeriodIndex inside the loop is slow).
    # Plain arithmetic on year/month keeps this portable across pandas 2.x and 3.x.
    mcode = np.asarray(idx.year) * 12 + np.asarray(idx.month)
    is_reset_day = np.zeros(n, dtype=bool)
    if n > 1:
        is_reset_day[:-1] = mcode[1:] != mcode[:-1]     # last session of each month

    if cash_r is None:
        cr = np.zeros(n)
    else:
        cr = np.nan_to_num(np.asarray(pd.Series(cash_r).reindex(idx), dtype=float))

    w = np.zeros(n)
    r = np.zeros(n)
    alive = np.zeros(n, dtype=bool)
    cur = float(leverage)
    live = True
    wiped = False
    cost = cost_bps * 1e-4
    liq_date = None
    lev = float(leverage)
    daily_reset = reset == "daily"
    max_w = (1.0 / maintenance) if maintenance > 0.0 else np.inf

    for i in range(n):
        if not live:
            # Called (not wiped out): the proceeds sit in the cash leg from the next
            # session on. Wiped out: nothing left to deposit, so nothing accrues.
            r[i] = 0.0 if wiped else cr[i]
            continue
        alive[i] = True
        w[i] = cur
        gross = cur * ra[i] - (cur - 1.0) * f[i]
        equity = 1.0 + gross
        if equity <= 0.0:
            # Wipeout: equity gone (and then some). Floor the loss at -100%.
            r[i] = -1.0
            live = False
            wiped = True
            liq_date = idx[i]
            continue
        drift_w = cur * (1.0 + ra[i]) / equity          # exposure after the day's move
        if drift_w > max_w:
            # Margin call: liquidate the whole position to cash at this close.
            r[i] = gross - drift_w * cost
            live = False
            liq_date = idx[i]
            continue
        target = lev if (daily_reset or is_reset_day[i]) else drift_w
        r[i] = gross - abs(target - drift_w) * cost
        cur = target

    out = pd.DataFrame({"r": r, "w": w, "alive": alive}, index=idx)
    out.attrs["liquidation_date"] = liq_date
    out.attrs["wiped_out"] = bool(wiped)
    out.attrs["leverage"] = float(leverage)
    out.attrs["reset"] = reset
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912)
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


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
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
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 3) -> dict:
    """Univariate OLS ``y = a + b x`` with a Newey-West HAC *t* on the slope."""
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    m = ~(np.isnan(y) | np.isnan(x))
    y, x = y[m], x[m]
    n = len(y)
    if n < 5:
        return {"slope": float("nan"), "intercept": float("nan"), "t": float("nan"), "n": n}
    xb, yb = x.mean(), y.mean()
    sxx = float(((x - xb) ** 2).sum())
    if sxx <= 0:
        return {"slope": float("nan"), "intercept": float("nan"), "t": float("nan"), "n": n}
    b = float(((x - xb) * (y - yb)).sum() / sxx)
    a = float(yb - b * xb)
    resid = y - (a + b * x)
    h = (x - xb) * resid                                   # score for the slope
    s = float(h @ h) / n
    for l in range(1, min(lags, n - 1) + 1):
        wgt = 1.0 - l / (lags + 1.0)
        s += 2.0 * wgt * float(h[l:] @ h[:-l]) / n
    var_b = n * s / (sxx ** 2)
    se = np.sqrt(var_b) if var_b > 0 else float("nan")
    return {"slope": b, "intercept": a, "t": float(b / se) if se and np.isfinite(se) else float("nan"),
            "n": int(n), "se": float(se)}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to get the excess Sharpe (that is what every race in
    this study does); ``max_drawdown`` and ``cagr`` are computed on whatever is passed, so
    the absolute (lived) figures are taken from the raw arm.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    end = float(wealth.iloc[-1]) if n else float("nan")
    cagr = float(end ** (1.0 / years) - 1.0) if years > 0 and end > 0 else float("nan")
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "terminal_wealth": end,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return **difference** r1 - r2 (Jobson-Korkie in NW form).

    Both arms are measured excess-of-cash, so the cash leg cancels in the difference and
    ``r1 - r2`` is simply monthly-reset minus daily-reset. |*t*| >= 2 = a reliably higher
    mean return; note that a higher mean return with proportionally higher risk is *not*
    a Sharpe advantage, which is why both are reported.
    """
    diff = (r1 - r2).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy())


def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 943,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdb = s.std(ddof=1)
        if sdb > 0:
            boots[b] = s.mean() / sdb * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


def bootstrap_diff_ci(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 21,
    seed: int = 943, periods_per_year: int = TRADING_DAYS, alpha: float = 0.05,
) -> dict:
    """Block-bootstrap CI for the **Sharpe advantage** (Sharpe of ``a`` minus Sharpe of ``b``).

    Blocks are drawn jointly so the two arms stay paired day by day — the advantage is a
    paired statistic and resampling them independently would inflate its variance.
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    x = df.to_numpy(dtype=float)
    n = len(df)
    if n < block + 2:
        return {"adv": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(periods_per_year)

    def adv(arr):
        sa, sb = arr[:, 0].std(ddof=1), arr[:, 1].std(ddof=1)
        if sa <= 0 or sb <= 0:
            return np.nan
        return (arr[:, 0].mean() / sa - arr[:, 1].mean() / sb) * ann

    point = adv(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = adv(x[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"adv": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n)}


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def build_frame(
    prices: pd.DataFrame,
    fund: str,
    asset: str = "SPY",
    cash: str = "BIL",
    rate: str = "IRX",
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> pd.DataFrame:
    """Align one leveraged fund with its underlying, the cash leg and the financing rate.

    Returns a frame of daily simple returns (``asset``, ``fund``, ``cash``) plus the daily
    financing rate ``fin``, restricted to the common window where all four exist — which
    is what gates the samples here (BIL starts 2007-05, UPRO 2009-06).
    """
    sub = prices.dropna(subset=[asset, fund, cash, rate])
    out = pd.DataFrame({
        "asset": sub[asset].pct_change(),
        "fund": sub[fund].pct_change(),
        "cash": sub[cash].pct_change(),
        "fin": financing_rate(sub[rate], spread_bps=spread_bps),
    }).dropna()
    return out


def race(
    frame: pd.DataFrame,
    leverage: float,
    cost_bps: float = DEFAULT_COST_BPS,
    maintenance: float = DEFAULT_MAINTENANCE,
) -> dict:
    """Race the fund, the daily-reset replication and the monthly-reset replication.

    All three are reported **excess-of-cash** (minus BIL's total return). Returns the
    per-arm summaries, the monthly-minus-daily Sharpe advantage and HAC *t*, the
    fee/tracking leg (daily-synth minus fund), the exposure path statistics and any
    liquidation date.

    A margin-called arm is credited the **cash leg** afterwards (``cash_r=frame["cash"]``),
    so a call costs it exactly what a call costs — the equity destroyed on the day — and
    not an extra, invented −cash drift for the rest of the sample.
    """
    daily = replicate(frame["asset"], frame["fin"], leverage, "daily",
                      cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])
    monthly = replicate(frame["asset"], frame["fin"], leverage, "monthly",
                        cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])

    e_fund = (frame["fund"] - frame["cash"]).rename("e_fund")
    e_daily = (daily["r"] - frame["cash"]).rename("e_daily")
    e_monthly = (monthly["r"] - frame["cash"]).rename("e_monthly")

    s_fund, s_daily, s_monthly = summary(e_fund), summary(e_daily), summary(e_monthly)
    fee_leg = (e_daily - e_fund).dropna()

    return {
        "leverage": float(leverage),
        "fund": s_fund, "daily": s_daily, "monthly": s_monthly,
        "abs_fund": summary(frame["fund"]), "abs_daily": summary(daily["r"]),
        "abs_monthly": summary(monthly["r"]), "abs_asset": summary(frame["asset"]),
        "sharpe_adv": s_monthly["sharpe"] - s_daily["sharpe"],
        "t_diff": sharpe_diff_tstat(e_monthly, e_daily),
        "mean_diff_bps": float((e_monthly - e_daily).mean() * 1e4),
        "sharpe_adv_vs_fund": s_monthly["sharpe"] - s_fund["sharpe"],
        "t_diff_vs_fund": sharpe_diff_tstat(e_monthly, e_fund),
        "fee_leg_ann_pct": float(fee_leg.mean() * TRADING_DAYS * 100),
        "t_fee_leg": newey_west_t(fee_leg.to_numpy()),
        "w_mean": float(monthly["w"][monthly["alive"]].mean()),
        "w_max": float(monthly["w"].max()),
        "w_min": float(monthly["w"][monthly["alive"]].min()),
        "liquidation_date": monthly.attrs["liquidation_date"],
        "n_days": int(len(frame)),
        "start": frame.index[0], "end": frame.index[-1],
        "e_fund": e_fund, "e_daily": e_daily, "e_monthly": e_monthly,
        "daily_path": daily, "monthly_path": monthly,
    }


# --------------------------------------------------------------------------- #
# Trending vs choppy — the decomposition
# --------------------------------------------------------------------------- #
def efficiency_ratio(log_returns: pd.Series) -> float:
    """Kaufman's efficiency ratio: |sum of log returns| / sum of |log returns|.

    1.0 = a perfectly straight-line month (pure trend); near 0 = a month that went
    nowhere the long way round (pure chop). Path shape only — it says nothing about
    direction or size.
    """
    s = pd.Series(log_returns).dropna()
    denom = float(s.abs().sum())
    return float(abs(s.sum()) / denom) if denom > 0 else float("nan")


def monthly_gap_table(
    frame: pd.DataFrame,
    leverage: float,
    cost_bps: float = DEFAULT_COST_BPS,
    maintenance: float = 0.0,
    min_days: int = 15,
) -> pd.DataFrame:
    """Per-calendar-month reset-frequency gap against the month's path shape.

    Columns: ``gap_pp`` (monthly-reset minus daily-reset log return, in percentage
    points), ``er`` (the month's efficiency ratio on the underlying), ``mret_pp`` (the
    underlying's own log return) and ``n``. Months with fewer than ``min_days`` sessions
    are dropped (partial first/last months). ``maintenance`` defaults to 0 here: the
    decomposition asks what the *path shape* does to the gap, so it must not be truncated
    by a liquidation — the liquidation risk is reported separately.
    """
    daily = replicate(frame["asset"], frame["fin"], leverage, "daily",
                      cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])
    monthly = replicate(frame["asset"], frame["fin"], leverage, "monthly",
                        cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])
    df = pd.DataFrame({
        "lm": np.log1p(monthly["r"].clip(lower=-0.999999)),
        "ld": np.log1p(daily["r"].clip(lower=-0.999999)),
        "la": np.log1p(frame["asset"]),
    })
    df["abs_la"] = df["la"].abs()
    g = df.groupby(frame.index.to_period("M"))
    agg = g.sum()
    tbl = pd.DataFrame({
        "gap_pp": (agg["lm"] - agg["ld"]) * 100.0,
        # Kaufman's efficiency ratio, month by month (see `efficiency_ratio`).
        "er": agg["la"].abs() / agg["abs_la"].where(agg["abs_la"] > 0),
        "mret_pp": agg["la"] * 100.0,
        "n": g["la"].count(),
    })
    return tbl[tbl["n"] >= min_days]


def chop_regression(tbl: pd.DataFrame, lags: int = 3) -> dict:
    """Regress the monthly reset-frequency gap on the month's efficiency ratio (HAC *t*).

    A negative slope means the monthly reset wins in *choppy* months and loses in
    *trending* ones — the sign the theory predicts and the opposite of the folklore that
    blames the daily reset unconditionally. Contemporaneous by construction: the
    regressor is measured over the *same* month as the gap, so this is a **decomposition,
    not a forecast** (see :func:`predictive_regression`).
    """
    fit = ols_hac(tbl["gap_pp"].to_numpy(), tbl["er"].to_numpy(), lags=lags)
    q1, q2 = tbl["er"].quantile([1.0 / 3, 2.0 / 3])
    terciles = {}
    for tag, mask in [("choppy", tbl["er"] <= q1),
                      ("mid", (tbl["er"] > q1) & (tbl["er"] < q2)),
                      ("trending", tbl["er"] >= q2)]:
        g = tbl.loc[mask, "gap_pp"]
        terciles[tag] = {"n": int(len(g)), "mean_pp": float(g.mean()),
                         "t": one_sample_t(g.to_numpy())}
    fit["terciles"] = terciles
    fit["er_q1"] = float(q1)
    fit["er_q2"] = float(q2)
    return fit


def predictive_regression(tbl: pd.DataFrame, lags: int = 3) -> dict:
    """Same regression with the efficiency ratio **lagged one month** — the tradable form.

    If last month's path shape predicted this month's gap, you could choose the reset
    frequency in advance. Also reports a naive switching rule: take the monthly reset only
    after a below-median-ER (choppy) month, else the daily reset.

    **The switch rule is deliberately generous.** Its threshold is the **full-sample**
    median efficiency ratio, which no one knew in 1998 — it is a hindsight cut, reported
    only because even with that free gift the rule earns a few bps a month on a 40%-vol
    sleeve. Read it as an upper bound on the tradable version, never as a backtest.
    """
    y = tbl["gap_pp"].to_numpy()[1:]
    x = tbl["er"].to_numpy()[:-1]
    fit = ols_hac(y, x, lags=lags)
    med = float(np.median(x))       # FULL-SAMPLE median: an in-sample (hindsight) threshold
    fit["threshold_is_in_sample"] = True
    picked = np.where(x < med, y, 0.0)
    fit["switch_mean_pp"] = float(picked.mean())
    fit["switch_t"] = one_sample_t(picked)
    fit["switch_n"] = int(len(picked))
    return fit


# --------------------------------------------------------------------------- #
# Robustness sweeps
# --------------------------------------------------------------------------- #
def era_cut(frame: pd.DataFrame, leverage: float, split: str = "2017-01-01",
            cost_bps: float = DEFAULT_COST_BPS,
            maintenance: float = DEFAULT_MAINTENANCE) -> dict:
    """Split the sample and re-run the excess-of-cash race on each half."""
    out: dict[str, dict | None] = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = frame.loc[sl]
        if len(sub) < 260:
            out[tag] = None
            continue
        r = race(sub, leverage, cost_bps=cost_bps, maintenance=maintenance)
        out[tag] = {
            "n_days": r["n_days"],
            "sharpe_daily": r["daily"]["sharpe"],
            "sharpe_monthly": r["monthly"]["sharpe"],
            "sharpe_adv": r["sharpe_adv"],
            "t_diff": r["t_diff"],
            "w_max": r["w_max"],
        }
    return out


def era_difference_test(frame: pd.DataFrame, leverage: float, split: str = "2017-01-01",
                        cost_bps: float = DEFAULT_COST_BPS,
                        maintenance: float = DEFAULT_MAINTENANCE) -> dict:
    """Is the era *decay* itself significant? HAC test of late-minus-early mean gap.

    Reporting "+0.047 before 2017, +0.004 after" is a contrast, and the house bar says a
    contrast needs a test of the **difference**, not two point estimates side by side. We
    regress the daily monthly-minus-daily gap on an era dummy: the slope is (late mean −
    early mean) in bps/day and its HAC *t* says whether the two halves are distinguishable
    at all. The split is the sample midpoint, chosen before looking, not snooped.
    """
    daily = replicate(frame["asset"], frame["fin"], leverage, "daily",
                      cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])
    monthly = replicate(frame["asset"], frame["fin"], leverage, "monthly",
                        cost_bps=cost_bps, maintenance=maintenance, cash_r=frame["cash"])
    gap = (monthly["r"] - daily["r"]).dropna()
    late = (gap.index >= pd.Timestamp(split)).astype(float)
    fit = ols_hac(gap.to_numpy() * 1e4, late, lags=10)
    return {"late_minus_early_bps": fit["slope"], "t": fit["t"],
            "early_mean_bps": float(gap[late == 0].mean() * 1e4),
            "late_mean_bps": float(gap[late == 1].mean() * 1e4),
            "n": int(len(gap)), "split": split}


def spread_sweep(prices: pd.DataFrame, fund: str, leverage: float,
                 grid=(0.0, 25.0, 50.0, 100.0, 200.0),
                 cost_bps: float = DEFAULT_COST_BPS,
                 maintenance: float = DEFAULT_MAINTENANCE) -> list[dict]:
    """Sweep the financing spread over ``^IRX`` — the study's main non-tape assumption."""
    rows = []
    for sp in grid:
        fr = build_frame(prices, fund, spread_bps=sp)
        r = race(fr, leverage, cost_bps=cost_bps, maintenance=maintenance)
        rows.append({"spread_bps": sp, "sharpe_monthly": r["monthly"]["sharpe"],
                     "sharpe_daily": r["daily"]["sharpe"], "sharpe_fund": r["fund"]["sharpe"],
                     "sharpe_adv": r["sharpe_adv"], "t_diff": r["t_diff"],
                     "adv_vs_fund": r["sharpe_adv_vs_fund"]})
    return rows


def cost_sweep(frame: pd.DataFrame, leverage: float, grid=(0.0, 1.0, 2.0, 5.0, 10.0),
               maintenance: float = DEFAULT_MAINTENANCE) -> list[dict]:
    """Sweep the one-way transaction cost charged on exposure turnover."""
    rows = []
    for c in grid:
        r = race(frame, leverage, cost_bps=c, maintenance=maintenance)
        rows.append({"cost_bps": c, "sharpe_monthly": r["monthly"]["sharpe"],
                     "sharpe_daily": r["daily"]["sharpe"], "sharpe_adv": r["sharpe_adv"],
                     "t_diff": r["t_diff"]})
    return rows


def maintenance_sweep(frame: pd.DataFrame, leverage: float,
                      grid=(0.0, 0.15, 0.25, 0.30),
                      cost_bps: float = DEFAULT_COST_BPS) -> list[dict]:
    """Sweep the maintenance-margin PROXY — does the monthly-reset account get called?"""
    rows = []
    for m in grid:
        r = race(frame, leverage, cost_bps=cost_bps, maintenance=m)
        liq = r["liquidation_date"]
        rows.append({"maintenance": m, "sharpe_monthly": r["monthly"]["sharpe"],
                     "sharpe_adv": r["sharpe_adv"], "w_max": r["w_max"],
                     "terminal_monthly": r["abs_monthly"]["terminal_wealth"],
                     "terminal_daily": r["abs_daily"]["terminal_wealth"],
                     "liquidation": None if liq is None else str(pd.Timestamp(liq).date())})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_frame(prices: pd.DataFrame, spread_bps: float = 0.0) -> pd.DataFrame:
    """Turn a synthetic ``(asset, cash)`` price frame into the race's return frame.

    The synthetic cash index doubles as the financing benchmark (its daily accrual plus
    the spread), so the planted comparison is clean: the two arms differ *only* in reset
    frequency.
    """
    cash = prices["cash"].pct_change()
    return pd.DataFrame({
        "asset": prices["asset"].pct_change(),
        "fund": prices["asset"].pct_change(),   # placeholder; unused by the reset race
        "cash": cash,
        "fin": cash + spread_bps * 1e-4 / TRADING_DAYS,
    }).dropna()


def synthetic_detect(prices: pd.DataFrame, leverage: float = 2.0,
                     cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """Measure the monthly-minus-daily reset gap on a synthetic tape.

    Liquidation is disabled (``maintenance=0``) so the planted path-shape effect is
    measured cleanly. On a *choppy* (mean-reverting) world the monthly reset must win; on
    a *trending* world it must lose; on the iid null the gap must be centred at zero. This
    proves the machinery is unbiased — it never supports a real-tape stamp.
    """
    frame = synthetic_frame(prices)
    daily = replicate(frame["asset"], frame["fin"], leverage, "daily",
                      cost_bps=cost_bps, maintenance=0.0, cash_r=frame["cash"])
    monthly = replicate(frame["asset"], frame["fin"], leverage, "monthly",
                        cost_bps=cost_bps, maintenance=0.0, cash_r=frame["cash"])
    e_d = daily["r"] - frame["cash"]
    e_m = monthly["r"] - frame["cash"]
    diff = (e_m - e_d).dropna()
    tbl = monthly_gap_table(frame, leverage, cost_bps=cost_bps, maintenance=0.0)
    fit = chop_regression(tbl)
    return {
        "mean_gap_bps": float(diff.mean() * 1e4),
        "t_gap": newey_west_t(diff.to_numpy()),
        "sharpe_adv": float(summary(e_m)["sharpe"] - summary(e_d)["sharpe"]),
        "chop_slope": fit["slope"],
        "chop_t": fit["t"],
        "n_days": int(len(frame)),
        "n_months": int(len(tbl)),
    }
