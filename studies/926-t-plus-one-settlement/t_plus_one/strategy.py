"""Measurement, inference and the one tradable arm for Study 926 — T+1.

The event is a plumbing change, not a price signal: on **28 May 2024** US cash equities,
ETFs and corporate bonds moved from T+2 to **T+1** settlement, while Europe and Asia
stayed on T+2. So the question is not "what does the signal predict" but "**did the tape
change**, and did it change *more* for the funds that now carry a settlement mismatch?".

That makes this a **difference-in-difference**. For each daily outcome ``y`` we build the
treated-minus-control difference ``d_t = y_treated,t − y_control,t`` and regress it on a
post-switch dummy with Newey-West standard errors:

    d_t = a + b * Post_t + e_t          b = the DiD estimate, HAC t on b

Because the market factor is common to both legs, ``d_t`` already nets out the global
shocks that dominate 2024-2026 (the cutting cycle, the April-2025 tariff shock, the
dollar). What it cannot net out is everything that makes EFA *structurally* different
from SPY — above all that **EFA's "overnight" window contains the entire European and
Asian cash session**. That mechanical fact is the study's biggest confound and it is
named on every number.

Outcomes measured (all daily, all from the exact night/day identity):

* ``r_on``  — overnight return, previous close → today's open, in bps;
* ``abs_on`` — |overnight return|, a realised overnight-volatility proxy, in bps;
* ``on_var_share`` — ``r_on² / (r_on² + r_id²)``, the overnight share of the day's
  realised variance, bounded in [0, 1];
* ``abs_cc`` — |close-to-close return|, a total realised-volatility proxy, in bps.

The tradable arm is deliberately the *cheapest* instantiation of the month-end story: a
dollar-neutral **long EFA / short SPY** spread whose calendar signal is the turn of the
month (last trading day of a month plus the first three of the next). The position for
day ``t+1`` is decided from the calendar known through day ``t`` — **one execution lag**,
the same convention as every other study on the desk. That lag has a consequence worth
stating rather than hiding: because the signal fires *on* the month-end day, the days
actually **held** are the **first four trading days of each month**, and the classic
month-end session itself is never in the book. Costs are one-way × NAV on each leg at
every change of position, and the short leg pays a **borrow** rate that is swept.

Returns are **simple** (arithmetic) throughout so the spread return is exactly
``w * (r_long − r_short)`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The night/day decomposition (an identity, not a model)
# --------------------------------------------------------------------------- #
def decompose(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Split daily OHLC into overnight / intraday / close-to-close simple returns.

    ``r_on = open_t / close_{t-1} − 1``, ``r_id = close_t / open_t − 1`` and
    ``r_cc = close_t / close_{t-1} − 1``; by construction
    ``(1 + r_on)(1 + r_id) = (1 + r_cc)`` exactly. Column matching is case-insensitive.
    The first row is dropped (no previous close).
    """
    df = ohlc.rename(columns=str.lower)
    if not {"open", "close"}.issubset(df.columns):
        raise KeyError("decompose needs 'open' and 'close' columns")
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    out = pd.DataFrame(
        {
            "r_on": o / prev - 1.0,
            "r_id": c / o - 1.0,
            "r_cc": c / prev - 1.0,
        }
    )
    return out.dropna()


def daily_outcomes(ohlc: pd.DataFrame) -> pd.DataFrame:
    """The four daily DiD outcomes for one ticker, from its OHLC tape.

    ``r_on`` and ``abs_cc`` are in **bps**; ``on_var_share`` is a unit-free fraction of
    the day's realised variance attributable to the overnight leg. Days on which the
    close-to-close return is exactly zero (a frozen print) get a NaN share rather than a
    0/0.
    """
    d = decompose(ohlc)
    on2 = d["r_on"] ** 2
    id2 = d["r_id"] ** 2
    denom = on2 + id2
    share = (on2 / denom).where(denom > 0)
    return pd.DataFrame(
        {
            "r_on": d["r_on"] * 1e4,
            "r_id": d["r_id"] * 1e4,
            "abs_on": d["r_on"].abs() * 1e4,
            "abs_cc": d["r_cc"].abs() * 1e4,
            "on_var_share": share,
        }
    )


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #
def month_ranks(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Position of each trading day within its calendar month, forwards and backwards.

    Returns ``(rank_fwd, rank_bwd)``: ``rank_fwd = 0`` on the first trading day of a
    month, ``rank_bwd = 0`` on the last. Both are computed from the *realised* trading
    days in ``index``, so no exchange calendar is needed and no price is consulted.
    """
    idx = pd.DatetimeIndex(index)
    m = np.asarray(idx.to_period("M").astype(str))
    n = len(idx)
    rank_fwd = np.zeros(n, dtype=int)
    counter = 0
    for i in range(n):
        if i > 0 and m[i] != m[i - 1]:
            counter = 0
        rank_fwd[i] = counter
        counter += 1
    rank_bwd = np.zeros(n, dtype=int)
    counter = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and m[i] != m[i + 1]:
            counter = 0
        rank_bwd[i] = counter
        counter += 1
    return rank_fwd, rank_bwd


def turn_of_month_mask(index: pd.DatetimeIndex, n_last: int = 1, n_first: int = 3) -> pd.Series:
    """Boolean mask: the last ``n_last`` trading days of a month + the first ``n_first``."""
    idx = pd.DatetimeIndex(index)
    rank_fwd, rank_bwd = month_ranks(idx)
    return pd.Series((rank_bwd < n_last) | (rank_fwd < n_first), index=idx, name="tom")


def quarter_turn_mask(index: pd.DatetimeIndex, n_last: int = 1, n_first: int = 3) -> pd.Series:
    """Turn-of-month mask restricted to the four quarter boundaries.

    That is the last ``n_last`` trading days of March, June, September and December, plus
    the first ``n_first`` of January, April, July and October — the days on which index
    reconstitution and pension rebalancing flows are heaviest.
    """
    idx = pd.DatetimeIndex(index)
    rank_fwd, rank_bwd = month_ranks(idx)
    month = np.asarray(idx.month)
    ends = np.isin(month, [3, 6, 9, 12]) & (rank_bwd < n_last)
    starts = np.isin(month, [1, 4, 7, 10]) & (rank_fwd < n_first)
    return pd.Series(ends | starts, index=idx, name="qturn")


# --------------------------------------------------------------------------- #
# Inference primitives (desk-standard, mirrored from Study 912)
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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
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
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float(u[lag:] @ u[:-lag]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def auto_lags(n: int) -> int:
    """Newey-West bandwidth rule of thumb, ``floor(4 (n/100)^(2/9))``."""
    return max(1, int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with Newey-West (Bartlett) standard errors.

    ``X`` must already contain an intercept column. Returns the coefficient vector, the
    HAC standard errors and the HAC t-statistics. This is the workhorse behind every DiD
    number in the study: regressing the treated-minus-control difference on a post-switch
    dummy makes the dummy's coefficient the difference-in-difference estimate, and its
    HAC t the honest test of it.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k + 2:
        nan = np.full(k, np.nan)
        return {"beta": nan, "se": nan, "t": nan, "n": int(n), "lags": 0}
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = auto_lags(n)
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": t, "n": int(n), "lags": int(lags)}


def block_bootstrap_ci(
    x: pd.Series | np.ndarray,
    stat,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 926,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap percentile CI for ``stat`` applied to a 1-D series.

    Blocks of ``block`` consecutive observations preserve the autocorrelation and
    volatility clustering that make naive i.i.d. bootstraps far too confident.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    point = float(stat(r))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = stat(r[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


# --------------------------------------------------------------------------- #
# The difference-in-difference
# --------------------------------------------------------------------------- #
def symmetric_windows(index: pd.DatetimeIndex, switch: str, asof: str) -> dict:
    """Pre / post windows of the SAME length in trading days, straddling ``switch``.

    The post window runs from the switch through ``asof``; the pre window is the equal
    number of trading days immediately before the switch. Equal lengths matter: a DiD
    whose pre-period is five times longer than its post-period is really a statement
    about the pre-period.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    idx = idx[idx <= pd.Timestamp(asof)]
    sw = pd.Timestamp(switch)
    post = idx[idx >= sw]
    pre_all = idx[idx < sw]
    n = min(len(post), len(pre_all))
    pre = pre_all[-n:]
    post = post[:n]
    return {
        "pre": pre, "post": post, "n_each": int(n),
        "pre_start": pre[0] if n else None, "pre_end": pre[-1] if n else None,
        "post_start": post[0] if n else None, "post_end": post[-1] if n else None,
    }


def did_estimate(
    treated: pd.Series,
    control: pd.Series,
    switch: str,
    windows: dict | None = None,
    asof: str | None = None,
) -> dict:
    """Difference-in-difference on one daily outcome, with HAC inference.

    ``treated`` and ``control`` are the *same* outcome measured on the two legs. We form
    ``d_t = treated_t − control_t`` on their common dates, restrict to the symmetric
    pre/post windows, and regress ``d_t`` on ``[1, Post_t]``. The ``Post`` coefficient is
    the DiD; its Newey-West t is the test. We also report the raw pre/post means of both
    legs so the reader can see whether the difference comes from the treated leg moving
    or the control leg moving.
    """
    common = treated.dropna().index.intersection(control.dropna().index)
    tr = treated.reindex(common).astype(float)
    ct = control.reindex(common).astype(float)
    if windows is None:
        windows = symmetric_windows(common, switch, asof or str(common.max().date()))
    keep = windows["pre"].union(windows["post"])
    tr, ct = tr.reindex(keep).dropna(), ct.reindex(keep).dropna()
    keep = tr.index.intersection(ct.index)
    tr, ct = tr.loc[keep], ct.loc[keep]
    d = (tr - ct)
    post = pd.Series(keep >= pd.Timestamp(switch), index=keep).astype(float)

    X = np.column_stack([np.ones(len(d)), post.to_numpy()])
    fit = hac_ols(d.to_numpy(), X)

    pre_m = keep < pd.Timestamp(switch)
    post_m = ~pre_m
    return {
        "did": float(fit["beta"][1]),
        "did_se": float(fit["se"][1]),
        "did_t": float(fit["t"][1]),
        "n": int(fit["n"]),
        "lags": int(fit["lags"]),
        "n_pre": int(pre_m.sum()),
        "n_post": int(post_m.sum()),
        "treated_pre": float(tr[pre_m].mean()),
        "treated_post": float(tr[post_m].mean()),
        "control_pre": float(ct[pre_m].mean()),
        "control_post": float(ct[post_m].mean()),
        "diff_pre": float(d[pre_m].mean()),
        "diff_post": float(d[post_m].mean()),
        "d_series": d,
        "post_dummy": post,
    }


def did_table(
    panel: dict[str, pd.DataFrame],
    treated: str,
    control: str,
    switch: str,
    asof: str,
    outcomes=("r_on", "abs_on", "on_var_share", "abs_cc"),
) -> dict:
    """Run :func:`did_estimate` for every outcome on one treated/control pair."""
    y_tr = daily_outcomes(panel[treated])
    y_ct = daily_outcomes(panel[control])
    common = y_tr.index.intersection(y_ct.index)
    common = common[common <= pd.Timestamp(asof)]
    win = symmetric_windows(common, switch, asof)
    out = {}
    for col in outcomes:
        out[col] = did_estimate(y_tr[col], y_ct[col], switch, windows=win)
    out["_windows"] = win
    return out


def did_bootstrap_ci(
    est: dict, n_boot: int = 2000, block: int = 21, seed: int = 926, alpha: float = 0.05
) -> dict:
    """Circular block-bootstrap CI for a DiD estimate from :func:`did_estimate`.

    The pre and post halves of the difference series are resampled **separately** in
    blocks of ``block`` days, and the statistic recomputed as ``mean(post) − mean(pre)``.
    Blocking preserves the strong autocorrelation in a volatility proxy, which an i.i.d.
    bootstrap would blow straight through.
    """
    d = est["d_series"].dropna()
    post = est["post_dummy"].reindex(d.index).fillna(0.0) > 0.5
    a = d[~post].to_numpy(dtype=float)
    b = d[post].to_numpy(dtype=float)
    if min(a.size, b.size) < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan")}
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)

    def draw(x):
        nb = int(np.ceil(x.size / block))
        starts = rng.integers(0, x.size, nb)
        idx = ((starts[:, None] + offsets[None, :]) % x.size).ravel()[: x.size]
        return x[idx].mean()

    boots = np.array([draw(b) - draw(a) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(b.mean() - a.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_boot": int(boots.size),
            "block": int(block)}


def window_length_sweep(
    panel: dict[str, pd.DataFrame],
    treated: str,
    control: str,
    switch: str,
    asof: str,
    outcome: str = "on_var_share",
    half_windows=(63, 126, 252, 524),
) -> list[dict]:
    """Re-run the DiD with progressively longer symmetric windows.

    An event study has no "eras" in the buy-and-hold sense, so this is the honest
    substitute: if the break is real it should survive being measured over one quarter,
    six months, one year and two years either side. If it only appears at one window
    length, it is a window artefact.
    """
    y_tr = daily_outcomes(panel[treated])
    y_ct = daily_outcomes(panel[control])
    common = y_tr.index.intersection(y_ct.index)
    common = common[common <= pd.Timestamp(asof)]
    sw = pd.Timestamp(switch)
    pos = int(np.searchsorted(common.to_numpy(), np.datetime64(sw)))
    rows = []
    for h in half_windows:
        lo = max(0, pos - h)
        hi = min(len(common), pos + h)
        idx = common[lo:hi]
        win = {"pre": idx[idx < sw], "post": idx[idx >= sw]}
        win["n_each"] = int(min(len(win["pre"]), len(win["post"])))
        if win["n_each"] < 40:
            continue
        est = did_estimate(y_tr[outcome], y_ct[outcome], switch, windows=win)
        rows.append({"half_window": int(h), "n_pre": est["n_pre"], "n_post": est["n_post"],
                     "did": est["did"], "did_t": est["did_t"]})
    return rows


# --------------------------------------------------------------------------- #
# Placebo switch dates — the falsification test
# --------------------------------------------------------------------------- #
def placebo_switches(
    panel: dict[str, pd.DataFrame],
    treated: str,
    control: str,
    switch: str,
    asof: str,
    outcome: str = "on_var_share",
    step_days: int = 63,
    exclude_days: int = 252,
) -> dict:
    """Re-run the DiD at fake switch dates and rank the real one against them.

    Every ``step_days`` trading days we pretend the settlement change happened there
    instead, with the same symmetric-window construction, and record |t|. Dates within
    ``exclude_days`` of the true switch are skipped so the real break cannot leak into
    its own placebo distribution. If the true |t| is unremarkable inside this
    distribution, the "structural break" is just what a two-year window of this tape
    looks like at any date you pick.
    """
    y_tr = daily_outcomes(panel[treated])
    y_ct = daily_outcomes(panel[control])
    common = y_tr.index.intersection(y_ct.index)
    common = common[common <= pd.Timestamp(asof)]
    sw = pd.Timestamp(switch)
    real = did_estimate(y_tr[outcome], y_ct[outcome], switch,
                        windows=symmetric_windows(common, switch, asof))
    pos = int(np.searchsorted(common.to_numpy(), np.datetime64(sw)))
    ts, dates = [], []
    for i in range(step_days, len(common) - step_days, step_days):
        if abs(i - pos) <= exclude_days:
            continue
        fake = str(common[i].date())
        win = symmetric_windows(common, fake, asof)
        if win["n_each"] < 120:
            continue
        est = did_estimate(y_tr[outcome], y_ct[outcome], fake, windows=win)
        if np.isfinite(est["did_t"]):
            ts.append(abs(est["did_t"]))
            dates.append(fake)
    ts_arr = np.asarray(ts, dtype=float)
    real_t = abs(real["did_t"])
    pct = float((ts_arr < real_t).mean()) if ts_arr.size else float("nan")
    return {
        "outcome": outcome,
        "real_t": float(real["did_t"]),
        "real_did": float(real["did"]),
        "placebo_abs_t": ts_arr,
        "placebo_dates": dates,
        "n_placebo": int(ts_arr.size),
        "pct_below": pct,
        "placebo_median_abs_t": float(np.median(ts_arr)) if ts_arr.size else float("nan"),
        "placebo_max_abs_t": float(ts_arr.max()) if ts_arr.size else float("nan"),
        "n_placebo_exceeding": int((ts_arr >= real_t).sum()) if ts_arr.size else 0,
    }


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Sharpe is of the series **as given** — pass an excess-of-cash series for the excess
    Sharpe. CAGR is geometric; the drawdown is on the compounded wealth path; the t is
    Newey-West on the mean.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": int(n), "cagr": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "max_drawdown": float("nan"),
                "mean_daily_bps": float("nan"), "tstat": float("nan")}
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# The tradable arm — a turn-of-month long/short spread, one execution lag
# --------------------------------------------------------------------------- #
def tom_spread_backtest(
    long_close: pd.Series,
    short_close: pd.Series,
    cost_bps: float = 3.0,
    borrow_bps_ann: float = 50.0,
    n_last: int = 1,
    n_first: int = 3,
) -> pd.DataFrame:
    """Dollar-neutral long/short spread on a turn-of-month calendar signal.

    The position for day ``t+1`` is set from the calendar known through day ``t`` — the
    single execution lag. The signal window is ``n_last`` month-end days plus ``n_first``
    month-start days; after the lag the days **held** are shifted one forward, so with the
    defaults (1, 3) the book is on for the **first four trading days of each month** and
    flat on the month-end session itself. The spread is 1 unit long / 1 unit short of NAV,
    so its gross return is ``w_t (r_long,t − r_short,t)``. Frictions:

    * ``cost_bps`` one-way × NAV on **each** leg whenever the position changes (so a full
      entry, both legs, costs ``2 × cost_bps``, and the round trip ``4 × cost_bps``);
    * ``borrow_bps_ann`` annualised borrow on the short leg, charged pro-rata on the days
      the short is actually on. Shorting a large-cap ETF is cheap but never free, and the
      rate is swept in :func:`borrow_sweep`.

    Returns a frame with the two leg returns, the (lagged) weight and the net spread
    return.
    """
    common = long_close.dropna().index.intersection(short_close.dropna().index)
    lc = long_close.reindex(common).sort_index()
    sc = short_close.reindex(common).sort_index()
    r_long = lc.pct_change(fill_method=None).rename("r_long")
    r_short = sc.pct_change(fill_method=None).rename("r_short")

    tom = turn_of_month_mask(common, n_last=n_last, n_first=n_first).astype(float)
    w = tom.shift(1).rename("w")            # decided at t, held over t+1

    turnover = w.diff().abs().fillna(0.0)
    cost = turnover * 2.0 * cost_bps * 1e-4          # both legs cross the spread
    borrow = w.abs() * (borrow_bps_ann * 1e-4) / TRADING_DAYS

    gross = (w * (r_long - r_short)).rename("r_gross")
    net = (gross - cost - borrow).rename("r_net")
    out = pd.concat([r_long, r_short, w, gross, net], axis=1)
    return out.dropna()


def tom_pre_post(
    long_close: pd.Series,
    short_close: pd.Series,
    switch: str,
    asof: str,
    cost_bps: float = 3.0,
    borrow_bps_ann: float = 50.0,
) -> dict:
    """The turn-of-month spread, pre-switch vs post-switch, gross and net.

    The DiD here is on the *strategy*: did the month-end spread pay differently once
    settlement shortened? Reported with a HAC t on the on-days return difference, over
    the same symmetric windows as the measurement DiD.
    """
    bt = tom_spread_backtest(long_close, short_close, cost_bps=cost_bps,
                             borrow_bps_ann=borrow_bps_ann)
    bt = bt[bt.index <= pd.Timestamp(asof)]
    win = symmetric_windows(bt.index, switch, asof)
    keep = win["pre"].union(win["post"])
    bt = bt.reindex(keep).dropna()
    on = bt[bt["w"] > 0]
    pre_m = on.index < pd.Timestamp(switch)
    post_m = ~pre_m

    X = np.column_stack([np.ones(len(on)), post_m.astype(float)])
    fit_g = hac_ols(on["r_gross"].to_numpy() * 1e4, X)
    fit_n = hac_ols(on["r_net"].to_numpy() * 1e4, X)

    return {
        "n_on_days_pre": int(pre_m.sum()),
        "n_on_days_post": int(post_m.sum()),
        "gross_pre_bps": float(on["r_gross"][pre_m].mean() * 1e4),
        "gross_post_bps": float(on["r_gross"][post_m].mean() * 1e4),
        "net_pre_bps": float(on["r_net"][pre_m].mean() * 1e4),
        "net_post_bps": float(on["r_net"][post_m].mean() * 1e4),
        "did_gross_bps": float(fit_g["beta"][1]),
        "did_gross_t": float(fit_g["t"][1]),
        "did_net_bps": float(fit_n["beta"][1]),
        "did_net_t": float(fit_n["t"][1]),
        "t_post_net": newey_west_t(on["r_net"][post_m].to_numpy() * 1e4,
                                   lags=auto_lags(int(post_m.sum()))),
        "t_pre_net": newey_west_t(on["r_net"][pre_m].to_numpy() * 1e4,
                                  lags=auto_lags(int(pre_m.sum()))),
        "summary_post_net": summary(bt["r_net"][bt.index >= pd.Timestamp(switch)]),
        "summary_pre_net": summary(bt["r_net"][bt.index < pd.Timestamp(switch)]),
        "bt": bt,
        "on_days": on,
    }


def borrow_sweep(
    long_close: pd.Series,
    short_close: pd.Series,
    switch: str,
    asof: str,
    cost_grid=(0.0, 1.0, 3.0, 10.0),
    borrow_grid=(0.0, 25.0, 50.0, 100.0),
) -> list[dict]:
    """Sweep the two non-tape assumptions: the one-way cost and the short borrow rate.

    Neither is observable in the price file, so both are **assumptions**, and the honest
    thing is to show the whole surface rather than one flattering corner.
    """
    rows = []
    for c in cost_grid:
        for b in borrow_grid:
            r = tom_pre_post(long_close, short_close, switch, asof,
                             cost_bps=c, borrow_bps_ann=b)
            rows.append({
                "cost_bps": c, "borrow_bps_ann": b,
                "net_post_bps": r["net_post_bps"],
                "t_post_net": r["t_post_net"],
                "did_net_bps": r["did_net_bps"],
                "did_net_t": r["did_net_t"],
            })
    return rows


# --------------------------------------------------------------------------- #
# Excess-of-cash Sharpe race, pre vs post
# --------------------------------------------------------------------------- #
def excess_sharpe_pre_post(
    close: pd.Series, cash_close: pd.Series, switch: str, asof: str
) -> dict:
    """Excess-of-cash Sharpe for one ETF, pre-switch vs post-switch.

    Both arms are excess of the **same** cash leg (BIL's realised total return), so the
    comparison is excess-of-cash against excess-of-cash — which matters a lot here,
    because the pre-window sits at ~5% short rates and the post-window walks them down.
    """
    common = close.dropna().index.intersection(cash_close.dropna().index)
    r = close.reindex(common).pct_change(fill_method=None)
    rc = cash_close.reindex(common).pct_change(fill_method=None)
    e = (r - rc).dropna()
    e = e[e.index <= pd.Timestamp(asof)]
    win = symmetric_windows(e.index, switch, asof)
    pre = e.reindex(win["pre"]).dropna()
    post = e.reindex(win["post"]).dropna()
    return {"pre": summary(pre), "post": summary(post),
            "welch_t": welch_t(post.to_numpy(), pre.to_numpy())}


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], truth: dict) -> dict:
    """Run the DiD machinery on a synthetic panel with a known planted break.

    On the planted world the estimator must recover the break (positive DiD on the
    overnight drift, on the overnight-volatility proxy and on the overnight variance
    share, with |t| well clear of 2); on the null it must stay quiet. This proves the
    estimator is unbiased — it never supports a real-tape stamp.
    """
    switch = truth["switch_date"]
    asof = str(panel["treated"].index[-1].date())
    tab = did_table(panel, "treated", "control", switch, asof)
    tom = tom_pre_post(panel["treated"]["close"], panel["control"]["close"],
                       switch, asof, cost_bps=0.0, borrow_bps_ann=0.0)
    return {
        "did_r_on_bps": tab["r_on"]["did"],
        "t_r_on": tab["r_on"]["did_t"],
        "did_abs_on_bps": tab["abs_on"]["did"],
        "t_abs_on": tab["abs_on"]["did_t"],
        "did_on_var_share": tab["on_var_share"]["did"],
        "t_on_var_share": tab["on_var_share"]["did_t"],
        "did_abs_cc_bps": tab["abs_cc"]["did"],
        "t_abs_cc": tab["abs_cc"]["did_t"],
        "did_tom_gross_bps": tom["did_gross_bps"],
        "t_tom_gross": tom["did_gross_t"],
        "planted_on_drift_bps": truth["planted_on_drift_bps"],
        "planted_tom_bps": truth["planted_tom_bps"],
        "n_each": tab["_windows"]["n_each"],
    }
