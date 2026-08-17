"""Strategy + inference for Study 935 — Value Averaging vs Dollar-Cost Averaging.

**The rule.** Edleson's value averaging (VA) replaces "invest the same amount every
month" with "own the same *value* every month". A value path ``V_t`` is fixed in
advance; at each monthly decision date the saver buys or sells whatever it takes to
drag the equity sleeve back onto that path. Because the path grows smoothly while
the market does not, VA mechanically buys more after falls and sells after rallies.
That is the whole claim: a contrarian schedule should beat a flat one.

**The catch this study is built around.** VA is not self-funding. When the market
falls, the required purchase can exceed the money the saver has actually put aside,
and when it rallies, VA hands cash *back*. So a VA programme only exists alongside a
**cash buffer**, and any comparison that ignores that buffer is comparing two
different amounts of invested capital. Here both arms are handed exactly the same
committed capital — a pre-funded buffer plus the identical monthly contribution
schedule — idle money earns **BIL's actual total return**, and the accounts are
compared on their **whole-programme terminal wealth**. The buffer is finite: when VA
demands more than is there, the purchase is **capped** and the shortfall recorded.

**Conventions.**

- Value path: ``V_t = C * ((1+g)^t - 1) / g`` — the future value at month ``t`` of the
  same monthly contribution ``C`` compounding at an assumed rate ``g`` (``V_t = C*t``
  when ``g = 0``, Edleson's basic linear path). ``g`` is an **ASSUMPTION**, not tape:
  it is swept.
- **One execution lag, and only one:** the order is sized from the equity value at
  the decision month-end close ``t`` and filled at the **next trading day's close**.
  The contribution lands in cash at the decision close, so it earns one day of
  T-bill before it is spent.
- Costs: ``cost_bps`` one-way on traded notional (x NAV), charged on every buy *and*
  every sell. No shorting anywhere — the sell leg is capped at the shares held — so
  there is **no borrow cost** in this study.
- Both arms are measured against a **pure-cash** benchmark programme with the
  identical flows, so every headline can be quoted excess-of-cash.

Returns are levels, not rates: the primitive quantity is terminal wealth per dollar
contributed, which is what a saver actually ends up with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
# A purchase counts as "capped" only when the unfunded shortfall exceeds 1% of one
# monthly contribution; below that it is just the transaction cost being netted out
# of the contribution, which is what a real saver does.
BIND_TOL = 0.01


# --------------------------------------------------------------------------- #
# The value path
# --------------------------------------------------------------------------- #
def value_path(n_months: int, contrib: float = 1.0, growth_ann: float = 0.0) -> np.ndarray:
    """Edleson value path: target equity value after each of ``n_months`` months.

    ``growth_ann`` is the assumed annual growth of the path (an **ASSUMPTION**, swept
    in ``growth_sweep``). At ``growth_ann = 0`` the path is linear, ``V_t = C*t``.
    Otherwise it is the future value of the same contribution stream compounding at
    ``growth_ann``, so a VA saver whose market delivered exactly ``growth_ann`` would
    place no trades beyond the contributions themselves.
    """
    g = (1.0 + growth_ann) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    t = np.arange(1, int(n_months) + 1, dtype=float)
    if abs(g) < 1e-12:
        return contrib * t
    return contrib * ((1.0 + g) ** t - 1.0) / g


# --------------------------------------------------------------------------- #
# The accumulation engine (one window, one arm)
# --------------------------------------------------------------------------- #
def _as_map(s):
    """Accept a Series or an already-built ``{date: level}`` mapping.

    Date-keyed dict lookups are ~50x faster than ``Series.loc`` in the innermost
    loop, and the rolling race calls that loop a few hundred thousand times.
    """
    return s if isinstance(s, dict) else {k: float(v) for k, v in s.items()}


def run_plan(
    asset,
    cash,
    dec_dates,
    exe_dates,
    mode: str = "va",
    contrib: float = 1.0,
    buffer_mult: float = 6.0,
    growth_ann: float = 0.0,
    cost_bps: float = 1.0,
    dca_scale: float = 1.0,
) -> dict:
    """Run one accumulation programme over one window.

    Parameters
    ----------
    asset, cash:
        Daily total-return **close levels** of the equity sleeve and the cash leg.
    dec_dates:
        ``T+1`` month-end decision dates. ``dec_dates[0]`` is inception (buffer is
        deposited, no contribution, no trade); ``dec_dates[1..T]`` are the ``T``
        contribution/rebalance months.
    exe_dates:
        The matching fill dates — ``exe_dates[i]`` is the trading day **after**
        ``dec_dates[i]`` (the single execution lag).
    mode:
        ``"dca"`` — buy ``dca_scale * contrib`` of the sleeve every month (the scale
        is 1.0 for the headline arm and is dialled below 1.0 only for the
        exposure-matched control).
        ``"va"``  — trade to the value path (buy the shortfall, sell the excess).
        ``"cash"`` — the benchmark: never buy; everything sits in the cash leg.
    buffer_mult:
        Pre-funded cash buffer, in multiples of ``contrib`` (an **ASSUMPTION**,
        swept). It is deposited at inception into the cash leg and is committed
        capital for **both** arms, so the comparison is like-for-like.

    Returns a dict of terminal values, flows, trade statistics and cap-binding
    counts. All money is in units of ``contrib``.
    """
    if mode not in ("va", "dca", "cash"):
        raise ValueError(f"unknown mode {mode!r}")
    n = len(dec_dates) - 1
    if n < 1 or len(exe_dates) != len(dec_dates):
        raise ValueError("need at least one contribution month and matching fill dates")

    pa = _as_map(asset)
    pc = _as_map(cash)
    c = cost_bps * 1e-4
    target = value_path(n, contrib=contrib, growth_ann=growth_ann)

    t0 = dec_dates[0]
    cash_units = (buffer_mult * contrib) / pc[t0]
    shares = 0.0
    committed = buffer_mult * contrib

    equity_flows: list[tuple[pd.Timestamp, float]] = []
    prog_flows: list[tuple[pd.Timestamp, float]] = [(t0, -buffer_mult * contrib)]

    n_trades = 0
    traded_notional = 0.0
    total_cost = 0.0
    bind_months = 0
    shortfall_total = 0.0
    shortfall_max_month = 0.0
    invested_frac: list[float] = []

    for i in range(1, n + 1):
        d = dec_dates[i]
        e = exe_dates[i]
        cash_units += contrib / pc[d]
        committed += contrib
        prog_flows.append((d, -contrib))

        if mode == "dca":
            order = dca_scale * contrib
        elif mode == "va":
            order = float(target[i - 1]) - shares * pa[d]
        else:
            order = 0.0

        avail = cash_units * pc[e]
        if order > 0:
            want = order
            amt = min(want, max(avail / (1.0 + c), 0.0))
            if want - amt > BIND_TOL * contrib:
                bind_months += 1
                shortfall_total += want - amt
                shortfall_max_month = max(shortfall_max_month, want - amt)
            if amt > 0:
                cost = amt * c
                cash_units -= (amt + cost) / pc[e]
                shares += amt / pa[e]
                equity_flows.append((e, -(amt + cost)))
                n_trades += 1
                traded_notional += amt
                total_cost += cost
        elif order < 0:
            want = -order
            amt = min(want, shares * pa[e])
            if amt > 0:
                cost = amt * c
                shares -= amt / pa[e]
                cash_units += (amt - cost) / pc[e]
                equity_flows.append((e, amt - cost))
                n_trades += 1
                traded_notional += amt
                total_cost += cost

        eq = shares * pa[e]
        cv = cash_units * pc[e]
        invested_frac.append(eq / (eq + cv) if eq + cv > 0 else 0.0)

    e_end = exe_dates[n]
    equity_end = shares * pa[e_end]
    cash_end = cash_units * pc[e_end]
    total_end = equity_end + cash_end
    prog_flows.append((e_end, total_end))

    return {
        "mode": mode,
        "terminal_total": total_end,
        "terminal_equity": equity_end,
        "terminal_cash": cash_end,
        "committed": committed,
        "contributions": contrib * n,
        "n_trades": n_trades,
        "traded_notional": traded_notional,
        "total_cost": total_cost,
        "bind_months": bind_months,
        # ``shortfall_total`` is the sum over the WHOLE programme; ``shortfall_max_month``
        # is the largest single-month unfunded call. They are different numbers and the
        # write-up must not swap them: a programme can bind in several months.
        "shortfall_total": shortfall_total,
        "shortfall_max_month": shortfall_max_month,
        "mean_invested_frac": float(np.mean(invested_frac)) if invested_frac else 0.0,
        "equity_flows": equity_flows,
        "prog_flows": prog_flows,
        "terminal_date": e_end,
        "shares": shares,
    }


# --------------------------------------------------------------------------- #
# IRR (money-weighted return)
# --------------------------------------------------------------------------- #
def irr_annual(flows, lo: float = -0.95, hi: float = 5.0, tol: float = 1e-8) -> float:
    """Annualised money-weighted IRR of dated cashflows (bisection, actual/365.25).

    ``flows`` is a list of ``(date, amount)`` with negative amounts paid in. Returns
    NaN when the flows do not bracket a root (e.g. total wipe-out).
    """
    if len(flows) < 2:
        return float("nan")
    t0 = pd.Timestamp(flows[0][0])
    yrs = np.array([(pd.Timestamp(d) - t0).days / 365.25 for d, _ in flows])
    amt = np.array([float(a) for _, a in flows])

    def npv(r):
        return float(np.sum(amt / (1.0 + r) ** yrs))

    f_lo, f_hi = npv(lo), npv(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def equity_irr(res: dict) -> float:
    """Edleson's own metric: IRR of the **equity-side** flows only.

    This is the number the value-averaging literature quotes, and it is flattering
    by construction: it ignores the buffer that had to be sitting there to make the
    purchases possible, and it credits VA for selling into strength without charging
    it for the idle proceeds afterwards.
    """
    flows = list(res["equity_flows"]) + [(res["terminal_date"], res["terminal_equity"])]
    return irr_annual(flows)


def programme_irr(res: dict) -> float:
    """The honest metric: IRR of the **whole programme** — buffer in, contributions
    in, total account (equity + cash) out."""
    return irr_annual(res["prog_flows"])


# --------------------------------------------------------------------------- #
# One window, both arms + the cash benchmark
# --------------------------------------------------------------------------- #
def compare_window(
    asset,
    cash,
    dec_dates,
    exe_dates,
    contrib: float = 1.0,
    buffer_mult: float = 6.0,
    growth_ann: float = 0.0,
    cost_bps: float = 1.0,
    dca_scale: float = 1.0,
) -> dict:
    """Race VA against DCA over one window on identical committed capital.

    Both arms receive the same pre-funded buffer and the same monthly contributions;
    idle money earns the cash leg in both. The headline is ``gap_cents`` — the VA
    terminal-wealth advantage in **cents per dollar contributed**.
    """
    asset = _as_map(asset)
    cash = _as_map(cash)
    kw = dict(contrib=contrib, buffer_mult=buffer_mult,
              growth_ann=growth_ann, cost_bps=cost_bps)
    va = run_plan(asset, cash, dec_dates, exe_dates, mode="va", **kw)
    dca = run_plan(asset, cash, dec_dates, exe_dates, mode="dca", dca_scale=dca_scale, **kw)
    cb = run_plan(asset, cash, dec_dates, exe_dates, mode="cash", **kw)

    contributions = va["contributions"]
    return {
        "start": pd.Timestamp(dec_dates[0]),
        "end": pd.Timestamp(exe_dates[-1]),
        "w_va": va["terminal_total"],
        "w_dca": dca["terminal_total"],
        "w_cash": cb["terminal_total"],
        "committed": va["committed"],
        "contributions": contributions,
        "gap_cents": (va["terminal_total"] - dca["terminal_total"]) / contributions * 100.0,
        "excess_va_cents": (va["terminal_total"] - cb["terminal_total"]) / contributions * 100.0,
        "excess_dca_cents": (dca["terminal_total"] - cb["terminal_total"]) / contributions * 100.0,
        "va_wins": float(va["terminal_total"] > dca["terminal_total"]),
        "irr_prog_va": programme_irr(va),
        "irr_prog_dca": programme_irr(dca),
        "irr_eq_va": equity_irr(va),
        "irr_eq_dca": equity_irr(dca),
        "va_bind_months": va["bind_months"],
        "va_shortfall": va["shortfall_total"],
        "va_shortfall_max_month": va["shortfall_max_month"],
        "va_trades": va["n_trades"],
        "dca_trades": dca["n_trades"],
        "va_notional": va["traded_notional"],
        "dca_notional": dca["traded_notional"],
        "va_cost": va["total_cost"],
        "dca_cost": dca["total_cost"],
        "va_invested_frac": va["mean_invested_frac"],
        "dca_invested_frac": dca["mean_invested_frac"],
    }


# --------------------------------------------------------------------------- #
# Rolling race over every start month
# --------------------------------------------------------------------------- #
def build_windows(index: pd.DatetimeIndex, month_end_dates, horizon_months: int = 36):
    """Yield ``(dec_dates, exe_dates)`` for every start month that fits the horizon.

    ``exe_dates[i]`` is the trading day after ``dec_dates[i]``; a window is dropped
    if its final decision month-end is the last bar on the tape (no fill day exists),
    which is the mechanical form of "never slice into the future".
    """
    idx = pd.DatetimeIndex(index).sort_values()
    pos = {d: i for i, d in enumerate(idx)}
    me = list(pd.DatetimeIndex(month_end_dates))
    out = []
    for s in range(len(me) - horizon_months):
        dec = me[s: s + horizon_months + 1]
        exe = []
        ok = True
        for d in dec:
            p = pos[d] + 1
            if p >= len(idx):
                ok = False
                break
            exe.append(idx[p])
        if ok:
            out.append((dec, exe))
    return out


def rolling_race(
    asset: pd.Series,
    cash: pd.Series,
    horizon_months: int = 36,
    contrib: float = 1.0,
    buffer_mult: float = 6.0,
    growth_ann: float = 0.0,
    cost_bps: float = 1.0,
    dca_scale: float = 1.0,
    month_end_dates=None,
) -> pd.DataFrame:
    """Run the VA-vs-DCA race over **every** start month and return one row per window."""
    common = asset.dropna().index.intersection(cash.dropna().index)
    a = asset.reindex(common).sort_index()
    c = cash.reindex(common).sort_index()
    if month_end_dates is None:
        s = pd.Series(common, index=common)
        month_end_dates = pd.DatetimeIndex(
            sorted(s.groupby([common.year, common.month]).max().values)
        )
    amap, cmap = _as_map(a), _as_map(c)
    rows = [
        compare_window(amap, cmap, dec, exe, contrib=contrib, buffer_mult=buffer_mult,
                       growth_ann=growth_ann, cost_bps=cost_bps, dca_scale=dca_scale)
        for dec, exe in build_windows(common, month_end_dates, horizon_months)
    ]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.set_index("start").sort_index()
    return df


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


def newey_west_t(x, lags: int = 36) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    With ``horizon_months``-long overlapping windows the natural lag truncation is
    the horizon itself, which is the default used throughout.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_ci(
    x,
    n_boot: int = 2000,
    block: int = 36,
    seed: int = 935,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the mean of an overlapping-window series.

    The block length defaults to the horizon, which is the span over which two
    windows share tape and are therefore mechanically correlated.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n), "block": int(block)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
            "block": int(block), "n_boot": int(n_boot)}


def non_overlapping_t(df: pd.DataFrame, horizon_months: int = 36,
                      col: str = "gap_cents") -> dict:
    """One-sample t on a **non-overlapping** subset (every ``horizon_months``-th window).

    Overlapping windows share tape; this throws most of the data away in exchange
    for genuinely independent observations. If the sign survives here it is not an
    artefact of the overlap.
    """
    sub = df[col].to_numpy()[::horizon_months]
    return {"n": int(len(sub)), "mean": float(np.mean(sub)) if len(sub) else float("nan"),
            "t": one_sample_t(sub)}


# --------------------------------------------------------------------------- #
# Summaries, sweeps and cuts
# --------------------------------------------------------------------------- #
def summarise(df: pd.DataFrame, horizon_months: int = 36, seed: int = 935) -> dict:
    """Headline statistics of one rolling race."""
    gap = df["gap_cents"].to_numpy()
    k = int(df["va_wins"].sum())
    n = int(len(df))
    lo, hi = wilson_interval(k, n)
    boot = block_bootstrap_ci(gap, block=horizon_months, seed=seed)
    nov = non_overlapping_t(df, horizon_months=horizon_months)
    return {
        "n_windows": n,
        "gap_mean_cents": float(gap.mean()),
        "gap_median_cents": float(np.median(gap)),
        "gap_sd_cents": float(gap.std(ddof=1)) if n > 1 else float("nan"),
        "t_hac": newey_west_t(gap, lags=horizon_months),
        "t_nonoverlap": nov["t"],
        "n_nonoverlap": nov["n"],
        "boot_lo": boot["ci_low"],
        "boot_hi": boot["ci_high"],
        "va_win_rate": k / n if n else float("nan"),
        "win_lo": lo, "win_hi": hi,
        "excess_va_cents": float(df["excess_va_cents"].mean()),
        "excess_dca_cents": float(df["excess_dca_cents"].mean()),
        "irr_prog_va": float(df["irr_prog_va"].mean()),
        "irr_prog_dca": float(df["irr_prog_dca"].mean()),
        "irr_eq_va": float(df["irr_eq_va"].mean()),
        "irr_eq_dca": float(df["irr_eq_dca"].mean()),
        "bind_window_rate": float((df["va_bind_months"] > 0).mean()),
        "bind_month_rate": float(df["va_bind_months"].sum() / (n * horizon_months)) if n else float("nan"),
        "mean_shortfall": float(df["va_shortfall"].mean()),
        "worst_prog_shortfall": float(df["va_shortfall"].max()),
        "worst_month_shortfall": float(df["va_shortfall_max_month"].max()),
        "bind_months_total": int(df["va_bind_months"].sum()),
        "va_trades": float(df["va_trades"].mean()),
        "dca_trades": float(df["dca_trades"].mean()),
        "va_notional": float(df["va_notional"].mean()),
        "dca_notional": float(df["dca_notional"].mean()),
        "va_cost_cents": float(df["va_cost"].mean() / df["contributions"].mean() * 100.0),
        "dca_cost_cents": float(df["dca_cost"].mean() / df["contributions"].mean() * 100.0),
        "va_invested_frac": float(df["va_invested_frac"].mean()),
        "dca_invested_frac": float(df["dca_invested_frac"].mean()),
        "gap_dispersion_ratio": float(df["excess_va_cents"].std(ddof=1) /
                                      df["excess_dca_cents"].std(ddof=1)) if n > 1 else float("nan"),
    }


def sweep(
    asset: pd.Series,
    cash: pd.Series,
    key: str,
    values,
    horizon_months: int = 36,
    **base,
) -> pd.DataFrame:
    """Re-run the whole race varying one knob (``growth_ann``, ``buffer_mult``,
    ``cost_bps`` or ``horizon_months``) and tabulate the headline."""
    rows = []
    for v in values:
        kw = dict(base)
        h = horizon_months
        if key == "horizon_months":
            h = int(v)
        else:
            kw[key] = v
        df = rolling_race(asset, cash, horizon_months=h, **kw)
        s = summarise(df, horizon_months=h)
        s[key] = v
        rows.append(s)
    return pd.DataFrame(rows).set_index(key)


def era_cut(df: pd.DataFrame, split: str = "2016-01-01", horizon_months: int = 36) -> dict:
    """Split the windows by *start* date and summarise each half."""
    out = {}
    for tag, sub in [("early", df[df.index < pd.Timestamp(split)]),
                     ("late", df[df.index >= pd.Timestamp(split)])]:
        out[tag] = summarise(sub, horizon_months=horizon_months) if len(sub) > 2 else None
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices: pd.DataFrame,
    horizon_months: int = 36,
    buffer_mult: float = 6.0,
    growth_ann: float = 0.0,
    cost_bps: float = 1.0,
) -> dict:
    """Run the rolling race on a synthetic ``(asset, cash)`` tape.

    On a planted mean-reverting wobble VA should finish reliably ahead of DCA; on the
    pure random-walk null it should not. This proves the harness is unbiased — it
    never supports a real-tape stamp.
    """
    df = rolling_race(prices["asset"], prices["cash"], horizon_months=horizon_months,
                      buffer_mult=buffer_mult, growth_ann=growth_ann, cost_bps=cost_bps)
    s = summarise(df, horizon_months=horizon_months)
    s["n_windows"] = int(len(df))
    return s


# --------------------------------------------------------------------------- #
# The exposure-matched control (the decisive cut)
# --------------------------------------------------------------------------- #
def exposure_matched_race(
    asset: pd.Series,
    cash: pd.Series,
    horizon_months: int = 36,
    lo: float = 0.5,
    hi: float = 1.0,
    tol: float = 0.002,
    max_iter: int = 12,
    **kw,
) -> dict:
    """Re-race VA against a DCA arm dialled down to VA's own average equity weight.

    Value averaging does not just time the market; it also changes *how much* of the
    programme is in the market, because a value path that grows more slowly than the
    tape parks the difference in cash. That is a beta choice masquerading as a
    strategy. Here the DCA arm's monthly purchase is scaled by ``lambda`` (the rest
    stays in the cash leg) and ``lambda`` is bisected until the two arms share the
    same mean invested fraction. Whatever gap survives is the contrarian *timing*,
    with the exposure difference taken out.
    """
    lam_lo, lam_hi = lo, hi
    best = None
    for _ in range(max_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        df = rolling_race(asset, cash, horizon_months=horizon_months,
                          dca_scale=lam, **kw)
        s = summarise(df, horizon_months=horizon_months)
        s["dca_scale"] = lam
        s["exposure_gap"] = s["va_invested_frac"] - s["dca_invested_frac"]
        best = s
        if abs(s["exposure_gap"]) < tol:
            break
        if s["exposure_gap"] > 0:      # VA still holds more -> DCA needs more
            lam_lo = lam
        else:
            lam_hi = lam
    return best
