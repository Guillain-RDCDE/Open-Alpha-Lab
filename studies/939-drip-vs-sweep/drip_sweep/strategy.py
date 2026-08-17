"""Strategy + inference for Study 939 — DRIP or Sweep.

Two investors hold **the same number of shares** of the same ETF from day one and
differ in exactly one respect: what happens to a distribution once it lands.

1. **DRIP** — the cash is used to buy more shares the moment it arrives (pay date),
   at the ``drip_cost_bps`` a broker's automatic reinvestment charges (zero at every
   large US broker).
2. **SWEEP** — the cash goes into T-bills (the BIL total-return leg) and is
   reinvested into the ETF on a calendar schedule: at each quarter end (``"Q"``) or
   each year end (``"A"``), paying ``sweep_cost_bps`` on the ticket.

Both arms hold the distribution in a non-interest-bearing float between the ex-date
and the pay date (``pay_lag_days``, an ASSUMPTION — the tape has no pay dates), so
the *only* economic difference is the window between the pay date and the sweep arm's
next calendar reinvestment: over that window the swept cash earns the bill rate
instead of the fund's return. DRIP therefore wins whenever the fund out-earns cash
over the delay, and loses when it does not. That is the whole mechanism, and it is
arithmetically tiny: a ~2% yield idle for ~half a quarter at a ~5 pp equity-cash
spread is worth single-digit basis points a year.

**Execution lag.** Exactly one, everywhere: a purchase *decided* at the close of day
``t`` (a distribution landing, or a calendar sweep date arriving) is *executed* at the
close of day ``t+1``. Nothing in this engine can see day ``t+1``'s price on day ``t``.

**Costs.** One-way, charged on the NAV actually traded (the reinvested amount), never
on the whole portfolio. Neither arm ever shorts, so no borrow applies.

**Price-only vs total-return.** The simulation runs on the **price-only** close (the
one that drops on the ex-date) plus the reconstructed per-share distribution. The
total-return close is used only to *build* that distribution stream, and as a
sanity benchmark: a zero-lag, zero-cost DRIP must reproduce it.

The headline is the **terminal-wealth gap**, reported as annualised basis points of
log wealth, with a HAC *t* on the daily log-return difference and a block-bootstrap
CI. The Sharpe race is excess-of-cash on both arms, gross and net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BPS = 1e4


# --------------------------------------------------------------------------- #
# Inference primitives (house standard, mirrored from Study 912)
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
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
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to get the excess Sharpe.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, std = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "sharpe": float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * BPS),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def _pay_index(dates: pd.DatetimeIndex, ex_positions: np.ndarray,
               pay_lag_days: int) -> dict:
    """Map each ex-date position to the first trading position >= ex + lag days.

    ``pay_lag_days`` is calendar days (an ASSUMPTION; Yahoo publishes ex-dates, not
    pay dates). An ex-date whose pay date falls past the end of the sample never pays
    — the cash simply stays in the float, which both arms treat identically.
    """
    out = {}
    for i in ex_positions:
        target = dates[i] + pd.Timedelta(days=int(pay_lag_days))
        j = int(dates.searchsorted(target, side="left"))
        if j < len(dates):
            out[i] = max(j, int(i))
    return out


def sweep_dates(dates: pd.DatetimeIndex, freq: str = "Q") -> np.ndarray:
    """Positions of the last trading day of each quarter (``"Q"``) or year (``"A"``).

    These are the sweep arm's reinvestment decision days; execution is the next
    trading day (the single execution lag).
    """
    s = pd.Series(np.arange(len(dates)), index=dates)
    key = dates.to_period("Q") if freq.upper().startswith("Q") else dates.to_period("Y")
    return np.asarray(s.groupby(key).max().to_numpy(), dtype=int)


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
def simulate(
    price: pd.Series,
    dividend_ps: pd.Series,
    cash_index: pd.Series,
    policy: str = "drip",
    pay_lag_days: int = 30,
    cost_bps: float = 0.0,
    sweep_freq: str = "Q",
    capital: float = 10_000.0,
) -> pd.DataFrame:
    """Run one accounting policy over a price / dividend / cash tape.

    Parameters
    ----------
    price:
        Daily **price-only** close (drops on the ex-date).
    dividend_ps:
        Per-share cash going ex on each date (0.0 on non-ex days) — normally the
        output of ``data.reconstruct_dividends``.
    cash_index:
        Daily **total-return** close of the cash leg (BIL). Its daily return is what
        swept cash earns.
    policy:
        ``"drip"`` — buy shares on the pay date (executed t+1).
        ``"sweep"`` — park the cash in T-bills, buy shares at the next ``sweep_freq``
        boundary (executed t+1).
    pay_lag_days:
        Calendar days from ex-date to pay date. ASSUMPTION, swept.
    cost_bps:
        One-way cost in bps of the amount reinvested.
    sweep_freq:
        ``"Q"`` (quarter end) or ``"A"`` (year end). Ignored by the DRIP arm.

    Returns a daily frame with ``wealth`` (shares × price + every cash bucket),
    ``shares``, ``cash_interest`` (in T-bills), ``float_cash`` (in transit, no
    interest), ``n_trades`` (cumulative) and ``cost_paid`` (cumulative currency).
    """
    if policy not in ("drip", "sweep"):
        raise ValueError("policy must be 'drip' or 'sweep'")

    common = price.index.intersection(cash_index.index)
    p = price.reindex(common).astype(float).sort_index()
    d = dividend_ps.reindex(common).fillna(0.0).astype(float).sort_index()
    c = cash_index.reindex(common).astype(float).sort_index()
    ok = p.notna() & c.notna() & (p > 0)
    p, d, c = p[ok], d[ok], c[ok]

    dates = pd.DatetimeIndex(p.index)
    n = len(dates)
    if n < 3:
        raise ValueError("need at least 3 aligned observations")

    p_arr = p.to_numpy()
    d_arr = d.to_numpy()
    r_cash = c.pct_change().fillna(0.0).to_numpy()

    ex_positions = np.flatnonzero(d_arr > 0)
    pay_of = _pay_index(dates, ex_positions, pay_lag_days)
    pay_on: dict[int, list[int]] = {}
    for i, j in pay_of.items():
        pay_on.setdefault(j, []).append(i)
    sweep_on = set(int(x) for x in sweep_dates(dates, sweep_freq)) if policy == "sweep" else set()

    cost = cost_bps * 1e-4

    shares = capital / p_arr[0]
    cash_int = 0.0        # interest-bearing (T-bills) — sweep arm only
    float_cash = 0.0      # in transit ex -> pay, no interest — both arms
    pending_exec = 0.0    # decided yesterday, executes at today's close
    n_trades = 0
    cost_paid = 0.0

    wealth = np.empty(n)
    shr = np.empty(n)
    ci = np.empty(n)
    fc = np.empty(n)
    nt = np.empty(n)
    cp = np.empty(n)

    # Per-ex-date float amounts, keyed by ex position.
    float_by_ex: dict[int, float] = {}

    for i in range(n):
        # 1. the interest-bearing bucket accrues the bill rate.
        cash_int *= (1.0 + r_cash[i])

        # 2. execute what was decided at yesterday's close (the single lag).
        if pending_exec > 0.0:
            fee = pending_exec * cost
            shares += (pending_exec - fee) / p_arr[i]
            cost_paid += fee
            n_trades += 1
            pending_exec = 0.0

        # 3. a distribution goes ex today: cash leaves the share price, enters float.
        if d_arr[i] > 0.0:
            amt = shares * d_arr[i]
            float_by_ex[i] = amt
            float_cash += amt

        # 4. distributions whose pay date is today land in the account.
        for ex_i in pay_on.get(i, ()):
            amt = float_by_ex.pop(ex_i, 0.0)
            float_cash -= amt
            if policy == "drip":
                pending_exec += amt      # decided now, bought at tomorrow's close
            else:
                cash_int += amt          # parked in T-bills

        # 5. the sweep arm's calendar reinvestment decision.
        if i in sweep_on and cash_int > 0.0:
            pending_exec += cash_int
            cash_int = 0.0

        wealth[i] = shares * p_arr[i] + cash_int + float_cash + pending_exec
        shr[i] = shares
        ci[i] = cash_int
        fc[i] = float_cash
        nt[i] = n_trades
        cp[i] = cost_paid

    return pd.DataFrame(
        {"wealth": wealth, "shares": shr, "cash_interest": ci,
         "float_cash": fc, "n_trades": nt, "cost_paid": cp},
        index=dates,
    )


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def race(
    price: pd.Series,
    dividend_ps: pd.Series,
    cash_index: pd.Series,
    pay_lag_days: int = 30,
    drip_cost_bps: float = 0.0,
    sweep_cost_bps: float = 2.0,
    sweep_freq: str = "Q",
    capital: float = 10_000.0,
) -> dict:
    """Race the DRIP arm against the sweep arm on one tape.

    Returns per-arm summaries (excess-of-cash Sharpe, CAGR, vol, drawdown), the
    terminal-wealth gap in annualised basis points of log wealth, the HAC *t* on the
    daily log-return difference, the trade counts and the currency cost paid.
    """
    a = simulate(price, dividend_ps, cash_index, policy="drip",
                 pay_lag_days=pay_lag_days, cost_bps=drip_cost_bps, capital=capital)
    b = simulate(price, dividend_ps, cash_index, policy="sweep",
                 pay_lag_days=pay_lag_days, cost_bps=sweep_cost_bps,
                 sweep_freq=sweep_freq, capital=capital)

    common = a.index.intersection(b.index)
    wa, wb = a["wealth"].reindex(common), b["wealth"].reindex(common)
    ra, rb = wa.pct_change().dropna(), wb.pct_change().dropna()
    rc = cash_index.reindex(common).pct_change().reindex(ra.index).fillna(0.0)

    e_drip = (ra - rc).rename("e_drip")
    e_sweep = (rb - rc).rename("e_sweep")

    la = np.log(wa / wa.shift(1)).dropna()
    lb = np.log(wb / wb.shift(1)).dropna()
    dlog = (la - lb).dropna()
    years = len(dlog) / TRADING_DAYS

    gap_total_log = float(np.log(wa.iloc[-1] / wb.iloc[-1]))
    gap_bps_yr = float(gap_total_log / years * BPS) if years > 0 else float("nan")

    return {
        "drip": summary(e_drip),
        "sweep": summary(e_sweep),
        "terminal_drip": float(wa.iloc[-1]),
        "terminal_sweep": float(wb.iloc[-1]),
        "terminal_ratio": float(wa.iloc[-1] / wb.iloc[-1]),
        "gap_total_log": gap_total_log,
        "gap_bps_per_year": gap_bps_yr,
        "t_hac_dlog": newey_west_t(dlog.to_numpy()),
        "sharpe_gap": summary(e_drip)["sharpe"] - summary(e_sweep)["sharpe"],
        "n_trades_drip": int(a["n_trades"].iloc[-1]),
        "n_trades_sweep": int(b["n_trades"].iloc[-1]),
        "cost_drip": float(a["cost_paid"].iloc[-1]),
        "cost_sweep": float(b["cost_paid"].iloc[-1]),
        "years": float(years),
        "start": common[0], "end": common[-1],
        "dlog": dlog, "e_drip": e_drip, "e_sweep": e_sweep,
        "wealth_drip": wa, "wealth_sweep": wb,
    }


# --------------------------------------------------------------------------- #
# Block bootstrap on the annualised gap
# --------------------------------------------------------------------------- #
def bootstrap_gap_ci(
    dlog: pd.Series,
    n_boot: int = 2000,
    block: int = 63,
    seed: int = 939,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised log-wealth gap (bps/yr).

    Blocks of ``block`` consecutive days (a quarter by default) preserve the
    distribution cycle: the gap is generated in bursts around pay dates, so a
    day-level i.i.d. bootstrap would badly understate its variance.
    """
    x = np.asarray(pd.Series(dlog).dropna(), dtype=float)
    n = x.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    point = float(x.mean() * TRADING_DAYS * BPS)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = x[idx].mean() * TRADING_DAYS * BPS
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": n,
            "n_boot": int(n_boot), "block": int(block)}


# --------------------------------------------------------------------------- #
# Sweeps: the assumptions that are not tape
# --------------------------------------------------------------------------- #
def pay_lag_sweep(price, dividend_ps, cash_index,
                  lags=(0, 15, 30, 45), **kw) -> list[dict]:
    """The pay-lag ASSUMPTION swept. Longer lag = more idle float for *both* arms."""
    out = []
    for lag in lags:
        r = race(price, dividend_ps, cash_index, pay_lag_days=lag, **kw)
        out.append({"pay_lag_days": lag, "gap_bps_per_year": r["gap_bps_per_year"],
                    "t_hac_dlog": r["t_hac_dlog"], "terminal_ratio": r["terminal_ratio"]})
    return out


def cost_sweep(price, dividend_ps, cash_index,
               grid=((0.0, 0.0), (0.0, 2.0), (0.0, 5.0), (0.0, 10.0), (5.0, 5.0)),
               **kw) -> list[dict]:
    """Cost sweep over (drip_cost_bps, sweep_cost_bps) pairs, one-way × amount traded.

    The last pair is the symmetric case (a broker that charges for DRIP too), which
    removes the sweep arm's cost handicap entirely.
    """
    out = []
    for dc, sc in grid:
        r = race(price, dividend_ps, cash_index,
                 drip_cost_bps=dc, sweep_cost_bps=sc, **kw)
        out.append({"drip_cost_bps": dc, "sweep_cost_bps": sc,
                    "gap_bps_per_year": r["gap_bps_per_year"],
                    "t_hac_dlog": r["t_hac_dlog"],
                    "n_trades_drip": r["n_trades_drip"],
                    "n_trades_sweep": r["n_trades_sweep"]})
    return out


def frequency_sweep(price, dividend_ps, cash_index, freqs=("Q", "A"), **kw) -> list[dict]:
    """Quarterly vs annual sweeping — how much does waiting longer actually cost?"""
    out = []
    for f in freqs:
        r = race(price, dividend_ps, cash_index, sweep_freq=f, **kw)
        out.append({"sweep_freq": f, "gap_bps_per_year": r["gap_bps_per_year"],
                    "t_hac_dlog": r["t_hac_dlog"],
                    "terminal_ratio": r["terminal_ratio"],
                    "n_trades_sweep": r["n_trades_sweep"]})
    return out


def era_cut(price, dividend_ps, cash_index, split: str = "2016-01-01", **kw) -> dict:
    """Split the sample and re-run the race on each half.

    The natural economic cut: the ZIRP years (swept cash earns nothing, so DRIP should
    win by the full equity premium) against the post-2016 / post-2022 years (swept cash
    earns 4-5%, so the gap should shrink toward zero).
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        p = price.loc[sl]
        d = dividend_ps.reindex(p.index).fillna(0.0)
        c = cash_index.reindex(price.index).loc[sl]
        if len(p) < 260:
            out[tag] = None
            continue
        r = race(p, d, c, **kw)
        out[tag] = {
            "n_days": r["drip"]["n_days"],
            "years": r["years"],
            "gap_bps_per_year": r["gap_bps_per_year"],
            "t_hac_dlog": r["t_hac_dlog"],
            "sharpe_drip": r["drip"]["sharpe"],
            "sharpe_sweep": r["sweep"]["sharpe"],
            "terminal_ratio": r["terminal_ratio"],
        }
    return out


def rate_regime_cut(price, dividend_ps, cash_index, threshold: float = 0.02,
                    **kw) -> dict:
    """Race separately over calendar years where the cash leg yielded above/below
    ``threshold`` (annualised, realised from the BIL total-return leg).

    The naive story says the DRIP advantage should be *larger* when cash pays nothing,
    because the swept money then earns zero. The correct statement is that the
    advantage is the realised **equity-minus-cash** spread over the delay window, and
    the level of the cash rate says nothing about that spread on its own. Note the two
    buckets are unions of non-contiguous calendar years, so the wealth paths inside
    each are spliced — read the sign, not the third decimal.
    """
    c = cash_index.dropna()
    # groupby(year).last() rather than resample: the "YE"/"A" alias moved between
    # pandas 2.0 and 2.2 and CI spans both.
    year_end = c.groupby(c.index.year).last()
    yearly = year_end.pct_change().dropna()
    lo_years = set(int(y) for y in yearly[yearly < threshold].index)
    hi_years = set(int(y) for y in yearly[yearly >= threshold].index)
    out = {}
    for tag, years in [("low_rate", lo_years), ("high_rate", hi_years)]:
        mask = price.index.year.isin(sorted(years))
        p = price[mask]
        if len(p) < 260:
            out[tag] = None
            continue
        d = dividend_ps.reindex(p.index).fillna(0.0)
        cc = cash_index.reindex(p.index)
        r = race(p, d, cc, **kw)
        out[tag] = {"n_years_calendar": len(years), "n_days": r["drip"]["n_days"],
                    "gap_bps_per_year": r["gap_bps_per_year"],
                    "t_hac_dlog": r["t_hac_dlog"]}
    return out


# --------------------------------------------------------------------------- #
# Benchmarks & audits
# --------------------------------------------------------------------------- #
def drip_tracks_total_return(price: pd.Series, dividend_ps: pd.Series,
                             total_return: pd.Series, capital: float = 10_000.0) -> dict:
    """Audit: a zero-lag, zero-cost DRIP must reproduce the total-return index.

    Run with ``pay_lag_days=0`` and no cost, the DRIP arm is (bar the one-day
    execution lag) the definition of the adjusted-close series. A large residual means
    the dividend reconstruction is wrong — this is the study's single most important
    data check.
    """
    flat_cash = pd.Series(1.0, index=price.index)
    sim = simulate(price, dividend_ps, flat_cash, policy="drip",
                   pay_lag_days=0, cost_bps=0.0, capital=capital)
    tr = total_return.reindex(sim.index).dropna()
    sim = sim.reindex(tr.index)
    bench = capital * tr / tr.iloc[0]
    rel = (sim["wealth"] / bench)
    return {
        "terminal_sim": float(sim["wealth"].iloc[-1]),
        "terminal_tr": float(bench.iloc[-1]),
        "terminal_ratio": float(rel.iloc[-1]),
        "max_abs_dev_pct": float((rel - 1.0).abs().max() * 100.0),
        "ann_tracking_bps": float(np.log(rel.iloc[-1]) / (len(rel) / TRADING_DAYS) * BPS),
        "n_days": int(len(rel)),
    }


def seed_sweep(gen, signal_strength: float, n_seeds: int = 8, base_seed: int = 939,
               gen_kw: dict | None = None, **kw) -> dict:
    """Run ``synthetic_detect`` over ``n_seeds`` draws and summarise the gap.

    ``gen`` is a generator callable with the ``synthetic_daily`` signature. Returns the
    mean, sd and standard error of the annualised gap (bps/yr) — the honest way to read
    a detector whose per-seed noise is comparable to the effect it hunts.
    """
    gen_kw = dict(gen_kw or {})
    gaps = np.array([
        synthetic_detect(
            gen(signal_strength=signal_strength, seed=base_seed + s, **gen_kw)[0], **kw
        )["gap_bps_per_year"]
        for s in range(n_seeds)
    ])
    sd = float(gaps.std(ddof=1)) if n_seeds > 1 else float("nan")
    return {"signal_strength": signal_strength, "n_seeds": int(n_seeds),
            "mean": float(gaps.mean()), "sd": sd,
            "se": sd / np.sqrt(n_seeds) if n_seeds > 1 else float("nan"),
            "gaps": gaps}


def synthetic_detect(frame: pd.DataFrame, **kw) -> dict:
    """Run the race on a synthetic ``(close, dividend, cash)`` frame.

    On the planted world (a real equity premium over cash) the DRIP arm must win by a
    positive, detectable margin; on the null (price drifts at the cash rate) the gap
    must be ~0. Proves the machinery is unbiased — it never supports a real-tape stamp.
    """
    r = race(frame["close"], frame["dividend"], frame["cash"], **kw)
    return {
        "gap_bps_per_year": r["gap_bps_per_year"],
        "t_hac_dlog": r["t_hac_dlog"],
        "terminal_ratio": r["terminal_ratio"],
        "sharpe_drip": r["drip"]["sharpe"],
        "sharpe_sweep": r["sweep"]["sharpe"],
        "n_trades_drip": r["n_trades_drip"],
        "n_trades_sweep": r["n_trades_sweep"],
        "years": r["years"],
    }
