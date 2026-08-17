"""Strategy + inference for Study 941 — Short Both Legs.

The trade: per $1 of equity, sell short $0.50 of the **3x long** fund (TQQQ) and $0.50 of
the **−3x inverse** fund (SQQQ) on the same index, so the gross short book is 1.0x NAV and
the *net* index exposure is (near) zero. Both funds reset their leverage daily, so both
bleed the rebalancing decay ``~0.5 * L * (L-1) * sigma^2`` per year. The folklore says
shorting both harvests that decay as a market-neutral carry.

Accounting (the honest version):

- The short proceeds (1.0x NAV) plus the posted equity (1.0x NAV) sit in cash and earn the
  **cash rate** — that is the standard institutional short rebate, and it is why every
  number here is reported **excess-of-cash**: an excess return of zero means the trade
  added nothing over simply holding T-bills.
- **Borrow** is charged as an annual rate on the *gross short notional*, accrued daily.
  It is an **ASSUMPTION**, not tape — there is no free borrow-fee history — so it is swept
  from 0 to 15%/yr. A retail account that earns *no* interest on the short proceeds is the
  same arithmetic as paying borrow equal to (cash rate + the true fee), which the sweep
  covers explicitly.
- **Trading cost** is one-way bps × the traded notional, charged on every rebalance.
- **One execution lag:** the rebalance is *decided* at the close of day ``t`` (from the
  calendar alone — the first trading day of a week/month) and the reset weights only affect
  the return of day ``t+1``. There is no other lag anywhere in this study.

The position is **short gamma on a rebalance schedule**: when the index trends, the leg
that is losing grows into the book and the rebalance sells low / shorts high. So the
rebalance frequency is itself a risk parameter, and we race daily / weekly / monthly / never.

Returns are **simple** (arithmetic) so the equity path compounds exactly.

**One identity to keep honest.** On the *daily* reset with a full rebate, zero borrow and
zero cost, the book's excess-of-cash return collapses algebraically to

    e_nav[t] = r_cash[t] - 0.5 * (r_long[t] + r_short[t])

— the same expression as the "implied average all-in fund load" printed in
``examples/verify.py``. The two therefore agree to machine precision **by construction**;
that agreement is an identity, *not* an independent confirmation, and nothing in this
study should be read as if it were. What the simulator adds beyond the identity is the
part that is *not* algebra: the non-daily schedules, the borrow and rebate terms, the
trading cost, the compounding of the equity path, and the ruin check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Rebalance calendars — decided from the calendar only, executed with one lag
# --------------------------------------------------------------------------- #
def rebalance_flags(index: pd.DatetimeIndex, freq: str = "daily") -> np.ndarray:
    """Boolean array: True on days whose *close* triggers a reset to equal dollars.

    ``daily`` resets every close; ``weekly`` / ``monthly`` reset on the first trading day
    of each ISO week / calendar month (knowable at that day's close — no look-ahead, no
    price input); ``none`` never resets (the buy-and-hold short book).
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if freq == "daily":
        return np.ones(n, dtype=bool)
    if freq == "none":
        return np.zeros(n, dtype=bool)
    if freq == "weekly":
        # ``isocalendar()`` returns nullable UInt32 columns, whose ``.to_numpy()`` is
        # object-dtype on pandas 2.x — cast explicitly so the comparison below is a plain
        # integer compare on every supported pandas.
        key = np.asarray(idx.isocalendar().week.to_numpy(), dtype=np.int64)
    elif freq == "monthly":
        key = np.asarray(idx.year, dtype=np.int64) * 12 + np.asarray(idx.month, dtype=np.int64)
    else:
        raise ValueError(f"unknown rebalance frequency: {freq}")
    flags = np.zeros(n, dtype=bool)
    flags[1:] = key[1:] != key[:-1]
    return flags


# --------------------------------------------------------------------------- #
# The short-pair book
# --------------------------------------------------------------------------- #
def run_short_pair(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
) -> pd.DataFrame:
    """Simulate the equal-dollar short of both levered legs, mark to market daily.

    Parameters
    ----------
    lev_long, lev_short, cash:
        Daily total-return **close levels** of the 3x long fund, the −3x inverse fund and
        the cash (T-bill) leg, aligned on a common index.
    rebalance:
        ``daily`` / ``weekly`` / ``monthly`` / ``none`` — how often the two shorts are
        reset to equal dollars at ``gross/2`` of NAV each.
    gross:
        Gross short notional as a multiple of NAV at each reset (1.0 = $0.50 short in each
        leg per $1 of equity).
    borrow_ann:
        Annual stock-borrow fee on the gross short notional, accrued daily. **ASSUMPTION** —
        swept, never asserted.
    cost_bps:
        One-way transaction cost in bps of traded notional, charged on every reset.
    rebate:
        Fraction of the cash rate actually earned on the **short proceeds**. ``1.0`` is the
        institutional full-rebate account; ``0.0`` is the ordinary retail account that earns
        nothing on its short proceeds. This is an account-terms **ASSUMPTION**, so both ends
        are reported rather than one being asserted.

    Returns a frame with ``r_nav`` (equity return), ``r_cash``, ``e_nav`` (excess-of-cash),
    ``gross_short`` (the live short notional / NAV) and ``turnover`` (traded notional / NAV).
    The book is stopped (equity pinned at zero, returns NaN afterwards) if equity is ever
    wiped out — a short book *can* be destroyed, and we say so rather than hiding it.
    """
    common = lev_long.index.intersection(lev_short.index).intersection(cash.index)
    a = lev_long.reindex(common).sort_index().astype(float)
    b = lev_short.reindex(common).sort_index().astype(float)
    c = cash.reindex(common).sort_index().astype(float)
    r_a = a.pct_change().to_numpy()
    r_b = b.pct_change().to_numpy()
    r_c = c.pct_change().to_numpy()
    flags = rebalance_flags(common, rebalance)
    n = len(common)

    equity = 1.0
    l_a = 0.5 * gross          # short liabilities, as a fraction of the initial $1 equity
    l_b = 0.5 * gross
    cash_bal = equity + l_a + l_b
    borrow_daily = borrow_ann / TRADING_DAYS
    cost = cost_bps * 1e-4

    r_nav = np.full(n, np.nan)
    gross_short = np.full(n, np.nan)
    turnover = np.zeros(n)
    ruined = False

    for i in range(1, n):
        if ruined:
            continue
        prev_equity = equity
        notional = l_a + l_b
        l_a *= (1.0 + r_a[i])
        l_b *= (1.0 + r_b[i])
        # Interest: the posted equity always earns cash; the short proceeds earn cash only
        # to the extent of the rebate. Borrow is charged on the gross short notional.
        cash_bal += (prev_equity + rebate * notional) * r_c[i]
        cash_bal -= notional * borrow_daily
        equity = cash_bal - l_a - l_b
        gross_short[i] = (l_a + l_b) / equity if equity > 0 else np.nan
        if equity <= 0:
            r_nav[i] = -1.0
            equity = 0.0
            ruined = True
            continue
        # The reset is decided at this close and its *cost* is borne today; the reset
        # weights themselves only drive tomorrow's return (the single execution lag).
        if flags[i]:
            target = 0.5 * gross * equity
            traded = abs(l_a - target) + abs(l_b - target)
            turnover[i] = traded / equity
            l_a = target
            l_b = target
            equity -= traded * cost
        r_nav[i] = equity / prev_equity - 1.0
        cash_bal = equity + l_a + l_b

    out = pd.DataFrame(
        {
            "r_nav": r_nav,
            "r_cash": r_c,
            "e_nav": r_nav - r_c,
            "gross_short": gross_short,
            "turnover": turnover,
        },
        index=common,
    ).dropna(subset=["r_nav", "r_cash"])
    # Stamp AFTER the dropna: ``DataFrame.attrs`` propagation through pandas methods has
    # never been guaranteed across versions, and a silently-lost ruin flag would read as
    # "the book survived". Setting it on the object we actually return removes that risk.
    out.attrs["ruined"] = bool(ruined)
    return out


# --------------------------------------------------------------------------- #
# Where the harvest comes from, leg by leg
# --------------------------------------------------------------------------- #
def implied_leg_shortfall(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    lev: float = 3.0,
    bench: pd.Series | None = None,
) -> dict:
    """Each leg's annual **shortfall** against a costless ±``lev`` replication financed at cash.

    A costless daily-reset ``+L`` fund would return ``L*r_index - (L-1)*r_cash`` on an
    average day; a costless ``-L`` fund would return ``-L*r_index + (L+1)*r_cash``. What the
    real funds fail to deliver against that idealisation is each leg's own **residual load**,
    and the equal-dollar pair short collects the *average* of the two.

    This is a decomposition of the same residual, not a new measurement: it inherits every
    caveat attached to it (it lumps the expense ratio, the financing spread over the funds'
    true funding rate, their internal end-of-day trading, tracking error and the cash-proxy
    basis into one number, and separates none of them). Its use is directional: it says
    which leg the harvest actually comes from, and therefore which leg's borrow fee the
    trade can least afford.
    """
    if bench is None:
        raise ValueError("implied_leg_shortfall needs the unlevered index leg as `bench`")
    idx = (lev_long.index.intersection(lev_short.index)
           .intersection(cash.index).intersection(bench.index))
    rl = lev_long.reindex(idx).sort_index().pct_change().dropna()
    rs = lev_short.reindex(idx).sort_index().pct_change().dropna()
    rc = cash.reindex(idx).sort_index().pct_change().dropna()
    rq = bench.reindex(idx).sort_index().pct_change().dropna()
    m_l, m_s = rl.mean() * TRADING_DAYS, rs.mean() * TRADING_DAYS
    m_c, m_q = rc.mean() * TRADING_DAYS, rq.mean() * TRADING_DAYS
    short_long = lev * m_q - (lev - 1.0) * m_c - m_l
    short_short = -lev * m_q + (lev + 1.0) * m_c - m_s
    return {
        "shortfall_long": float(short_long),
        "shortfall_short": float(short_short),
        "average": float(0.5 * (short_long + short_short)),
        "share_from_long": float(short_long / (short_long + short_short))
        if (short_long + short_short) != 0 else float("nan"),
        "n_days": int(len(rl)),
    }


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk lineage)
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
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def hac_t(x) -> float:
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    return newey_west_t(arr, lags=auto_lags(arr.size))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Performance & risk summaries
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an **excess-of-cash** series to get the excess Sharpe. Also reports the tail that
    matters for a short-gamma book: skew, excess kurtosis, worst day, worst month and the
    max drawdown of the equity path.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))

    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    final = float(wealth.iloc[-1]) if n else float("nan")
    cagr = float(final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else float("nan")
    dd = float((wealth / wealth.cummax() - 1.0).min()) if n else float("nan")
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0

    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "ann_mean": float(mu * periods_per_year),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "worst_day": float(r.min()),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "tstat": hac_t(r),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return *difference* r1 − r2 (Jobson-Korkie, Newey-West form).

    Both arms are excess-of-cash, so the cash leg cancels in the difference.
    """
    diff = (pd.Series(r1) - pd.Series(r2)).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy(), lags=auto_lags(len(diff)))


def beta_to(excess: pd.Series, bench_excess: pd.Series) -> dict:
    """OLS beta and alpha of an excess series on a benchmark excess series, HAC *t* on alpha.

    The point of shorting *both* legs is that the index exposure cancels. This measures how
    much residual beta survives — a residual beta is not a harvest, it is a directional bet.
    """
    df = pd.concat([pd.Series(excess).rename("y"), pd.Series(bench_excess).rename("x")],
                   axis=1).dropna()
    if len(df) < 30:
        return {"beta": float("nan"), "alpha_ann": float("nan"), "t_alpha": float("nan"),
                "r2": float("nan"), "n": int(len(df))}
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    xc = x - x.mean()
    beta = float((xc @ (y - y.mean())) / (xc @ xc)) if (xc @ xc) > 0 else float("nan")
    alpha_daily = float(y.mean() - beta * x.mean())
    resid = y - (alpha_daily + beta * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    # HAC t on alpha: the residual + alpha series is the alpha-carrying part of the return.
    t_alpha = newey_west_t(resid + alpha_daily, lags=auto_lags(len(df)))
    return {"beta": beta, "alpha_ann": alpha_daily * TRADING_DAYS, "t_alpha": t_alpha,
            "r2": r2, "n": int(len(df))}


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe CI (circular block bootstrap on the excess series)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 941,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series.

    Blocks of ``block`` consecutive days preserve the vol clustering that drives both the
    decay harvest and its tail. Returns the point Sharpe, the (1-alpha) percentile interval
    and the share of resamples with a negative Sharpe.
    """
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for bdx in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdev = s.std(ddof=1)
        if sdev > 0:
            boots[bdx] = s.mean() / sdev * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": n,
            "n_boot": int(valid.size), "block": block}


def bootstrap_mean_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 941,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the *annualised mean* excess return (the harvest)."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean_ann": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for bdx in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[bdx] = r[idx].mean() * TRADING_DAYS
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_ann": float(r.mean() * TRADING_DAYS), "ci_low": float(lo),
            "ci_high": float(hi), "frac_negative": float((boots < 0).mean()),
            "n_obs": n, "n_boot": n_boot, "block": block}


# --------------------------------------------------------------------------- #
# The headline race
# --------------------------------------------------------------------------- #
def compare(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    bench: pd.Series | None = None,
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
) -> dict:
    """Run the short-pair book and summarise it excess-of-cash, with residual beta.

    ``bench`` (the unlevered index ETF) is optional; when given, its excess-of-cash return
    is the benchmark for the residual-beta regression and for the Sharpe race.
    """
    bt = run_short_pair(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                        borrow_ann=borrow_ann, cost_bps=cost_bps, rebate=rebate)
    s = summary(bt["e_nav"])
    out = {
        "bt": bt,
        "e_nav": bt["e_nav"],
        "strategy": s,
        "sharpe": s["sharpe"],
        "ann_mean": s["ann_mean"],
        "t_hac": s["tstat"],
        "max_drawdown": max_drawdown(bt["r_nav"]),
        "turnover_ann": float(bt["turnover"].mean() * TRADING_DAYS),
        "max_gross": float(bt["gross_short"].max()),
        "ruined": bool(bt.attrs.get("ruined", False)),
        "rebalance": rebalance,
        "borrow_ann": borrow_ann,
        "cost_bps": cost_bps,
        "rebate": rebate,
    }
    if bench is not None:
        e_bench = (bench.pct_change() - cash.pct_change()).reindex(bt.index).dropna()
        s_b = summary(e_bench)
        out["bench"] = s_b
        out["e_bench"] = e_bench
        out["beta"] = beta_to(bt["e_nav"], e_bench)
        out["sharpe_adv_vs_bench"] = s["sharpe"] - s_b["sharpe"]
        out["t_diff_vs_bench"] = sharpe_diff_tstat(bt["e_nav"], e_bench)
    return out


# --------------------------------------------------------------------------- #
# Sweeps — borrow (the crux), cost, rebalance frequency, eras
# --------------------------------------------------------------------------- #
def borrow_sweep(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    rebalance: str = "daily",
    gross: float = 1.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
    grid=(0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0),
) -> list[dict]:
    """Annualised excess return / Sharpe / HAC *t* across assumed borrow rates (%/yr).

    Borrow is the crux of this trade and it is an **ASSUMPTION**, so the sweep is the
    headline, not a footnote: the breakeven borrow rate is the number that decides the
    verdict.
    """
    rows = []
    for b in grid:
        cmp = compare(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                      borrow_ann=b / 100.0, cost_bps=cost_bps, rebate=rebate)
        rows.append({
            "borrow_pct": float(b),
            "ann_mean": cmp["ann_mean"],
            "sharpe": cmp["sharpe"],
            "t_hac": cmp["t_hac"],
            "max_drawdown": cmp["max_drawdown"],
        })
    return rows


def breakeven_borrow(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    rebalance: str = "daily",
    gross: float = 1.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
) -> float:
    """The borrow rate (%/yr) at which the excess-of-cash harvest reaches zero.

    Borrow is charged on the gross short notional, which is ``gross`` × NAV at every reset,
    so to a very good approximation the annual excess return is linear in the borrow rate.
    We solve it exactly by bisecting the simulator (which also picks up the compounding).
    """
    lo, hi = 0.0, 100.0
    m_lo = compare(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                   borrow_ann=lo / 100.0, cost_bps=cost_bps, rebate=rebate)["ann_mean"]
    if not np.isfinite(m_lo) or m_lo <= 0:
        return 0.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        m = compare(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                    borrow_ann=mid / 100.0, cost_bps=cost_bps, rebate=rebate)["ann_mean"]
        if m > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cost_sweep(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    rebate: float = 1.0,
    grid=(0.0, 1.0, 2.0, 5.0, 10.0),
) -> list[dict]:
    """One-way cost sweep at a fixed borrow assumption."""
    rows = []
    for c in grid:
        cmp = compare(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                      borrow_ann=borrow_ann, cost_bps=c, rebate=rebate)
        rows.append({
            "cost_bps": float(c),
            "ann_mean": cmp["ann_mean"],
            "sharpe": cmp["sharpe"],
            "t_hac": cmp["t_hac"],
            "turnover_ann": cmp["turnover_ann"],
        })
    return rows


def frequency_race(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    bench: pd.Series | None = None,
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
    freqs=("daily", "weekly", "monthly", "none"),
) -> list[dict]:
    """Race the rebalance schedules — the short-gamma dial of this trade."""
    rows = []
    for f in freqs:
        cmp = compare(lev_long, lev_short, cash, bench=bench, rebalance=f, gross=gross,
                      borrow_ann=borrow_ann, cost_bps=cost_bps, rebate=rebate)
        rows.append({
            "rebalance": f,
            "ann_mean": cmp["ann_mean"],
            "sharpe": cmp["sharpe"],
            "vol_ann": cmp["strategy"]["vol_ann"],
            "t_hac": cmp["t_hac"],
            "max_drawdown": cmp["max_drawdown"],
            "worst_day": cmp["strategy"]["worst_day"],
            "worst_month": cmp["strategy"]["worst_month"],
            "skew": cmp["strategy"]["skew"],
            "excess_kurtosis": cmp["strategy"]["excess_kurtosis"],
            "turnover_ann": cmp["turnover_ann"],
            "max_gross": cmp["max_gross"],
            "ruined": cmp["ruined"],
            "beta": cmp["beta"]["beta"] if bench is not None else float("nan"),
        })
    return rows


def era_cut(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    split: str = "2018-01-01",
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
) -> dict:
    """Split the sample at ``split`` and re-run the harvest on each half."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        a, b, c = lev_long.loc[sl], lev_short.loc[sl], cash.loc[sl]
        if len(a) < 120:
            out[tag] = None
            continue
        cmp = compare(a, b, c, rebalance=rebalance, gross=gross,
                      borrow_ann=borrow_ann, cost_bps=cost_bps, rebate=rebate)
        out[tag] = {
            "n_days": cmp["strategy"]["n_days"],
            "ann_mean": cmp["ann_mean"],
            "sharpe": cmp["sharpe"],
            "t_hac": cmp["t_hac"],
            "max_drawdown": cmp["max_drawdown"],
            "worst_month": cmp["strategy"]["worst_month"],
        }
    return out


def calendar_year_table(
    lev_long: pd.Series,
    lev_short: pd.Series,
    cash: pd.Series,
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
    rebate: float = 1.0,
) -> pd.DataFrame:
    """Per-calendar-year excess-of-cash return (%) of the short-pair book, with turnover."""
    bt = run_short_pair(lev_long, lev_short, cash, rebalance=rebalance, gross=gross,
                        borrow_ann=borrow_ann, cost_bps=cost_bps, rebate=rebate)
    rows = []
    for y in sorted(set(bt.index.year)):
        m = bt.index.year == y
        nav = float((1.0 + bt["r_nav"][m]).prod() - 1.0)
        csh = float((1.0 + bt["r_cash"][m]).prod() - 1.0)
        rows.append({"year": int(y), "nav_pct": nav * 100, "cash_pct": csh * 100,
                     "excess_pct": (nav - csh) * 100,
                     "worst_day_pct": float(bt["r_nav"][m].min() * 100)})
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices: pd.DataFrame,
    rebalance: str = "daily",
    gross: float = 1.0,
    borrow_ann: float = 0.0,
    cost_bps: float = 2.0,
) -> dict:
    """Run the short-pair harvest on a synthetic (lev_long, lev_short, cash) tape.

    On the planted high-vol world the harvest must be clearly positive and (near)
    market-neutral; on the vol-free null it must be ~0. Proves the machinery is unbiased —
    it never supports a real-tape stamp.
    """
    cmp = compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                  bench=prices["under"], rebalance=rebalance, gross=gross,
                  borrow_ann=borrow_ann, cost_bps=cost_bps)
    return {
        "ann_mean": cmp["ann_mean"],
        "sharpe": cmp["sharpe"],
        "t_hac": cmp["t_hac"],
        "beta": cmp["beta"]["beta"],
        "max_drawdown": cmp["max_drawdown"],
        "n_days": cmp["strategy"]["n_days"],
    }
