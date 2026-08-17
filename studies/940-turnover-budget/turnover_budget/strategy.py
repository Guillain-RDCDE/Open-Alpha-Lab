"""Strategy + inference for Study 940 — The Turnover Budget.

One sleeve, four speeds. The sleeve is a plain cross-sectional momentum book on the
eleven Select Sector SPDRs: rank each available sector on its **12-1** total return
(the trailing 252-day return, skipping the most recent 21 days), go long the top
``top_k`` and short the bottom ``top_k``, equal-weighted within each leg, 0.5 gross
per side so the book is **dollar-neutral with 1.0 gross**. The only thing that changes
between arms is *how often the book is re-ranked*: **daily / weekly / monthly /
quarterly**.

Execution lag — exactly one, and it is the only one. The signal is formed from closes
through day ``t``; the resulting weights are effective for day ``t+1``'s return and the
trade is charged on day ``t+1``. Between rebalances weights **drift** with realised
returns (no free intra-period rebalancing) and are renormalised by the book's own return,
so gross exposure stays where the mandate put it and the turnover we measure is the
turnover a real book would pay.

Costs — one-way and multiplicative on NAV. Traded notional on a rebalance day is
``sum_i |w_new_i - w_drifted_i|`` (a fraction of NAV); the cost charged is
``cost_bps * traded_notional``. The short leg pays a borrow fee of ``borrow_bps`` per
annum on the short notional, accrued daily. Both the cost per unit turnover and the
borrow rate are **assumptions, not tape** — they are swept, and the sweep is the point
of the study rather than a footnote to it.

Excess-of-cash — a dollar-neutral book is funded by its own collateral, which earns
the T-bill rate; the long-minus-short spread *is* the excess-of-cash return, so no
further subtraction is required (and doing it twice would be a double count). The
long-only cross-check arm holds real exposure and therefore has BIL's total return
subtracted explicitly. Both conventions are implemented in ``run_backtest`` and
labelled in its output.

Reuse warning — the engine reads a missing daily return as **zero**, which is safe on
this panel (all eleven SPDRs still trade, so a NaN only ever means "not listed yet") but
would silently forgive a delisting on a universe that has them. Point it at a dead-name
panel and you must model the terminal loss yourself.

The headline the study is built to produce: for each frequency, the **break-even cost
per unit of traded notional** — the level of ``cost_bps`` at which that arm's net
excess return is exactly zero. That single number is the turnover budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FREQS = ("D", "W", "M", "Q")
FREQ_LABEL = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"}


# --------------------------------------------------------------------------- #
# Signal & rebalance calendar
# --------------------------------------------------------------------------- #
def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 total return: the return from ``t-lookback`` to ``t-skip``, per column.

    Uses only closes through day ``t``; the skip month removes the well-documented
    short-term reversal that otherwise contaminates a raw 12-month sort. NaN wherever a
    sector has no full history yet (XLRE before 2015, XLC before 2018), which the
    weighting layer reads as "not investable".
    """
    px = prices.astype(float)
    return px.shift(skip) / px.shift(lookback) - 1.0


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Trading dates on which the book is re-ranked, for ``freq`` in D / W / M / Q.

    Period-based (``to_period``) rather than a long ``date_range``, so nothing can
    overflow pandas' nanosecond Timestamp horizon on any supported version.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    if freq == "D":
        return idx
    if freq not in ("W", "M", "Q"):
        raise ValueError(f"unknown rebalance frequency {freq!r}; expected one of {FREQS}")
    s = pd.Series(idx, index=idx)
    per = idx.to_period({"W": "W", "M": "M", "Q": "Q"}[freq])
    return pd.DatetimeIndex(s.groupby(per).last().to_numpy())


def target_weights(
    signal: pd.DataFrame,
    top_k: int = 3,
    long_only: bool = False,
    min_breadth: int = 6,
) -> pd.DataFrame:
    """Turn a cross-sectional signal frame into dollar-neutral (or long-only) weights.

    Long the ``top_k`` highest-signal available names and short the ``top_k`` lowest,
    equal-weighted inside each leg at 0.5 gross per side (so gross = 1.0, net = 0). With
    ``long_only=True`` the book is simply 1.0 spread over the top ``top_k``. Rows with
    fewer than ``min_breadth`` investable names are left all-NaN (the early sample, when
    a sector has yet to list) and are skipped by the backtest.
    """
    vals = signal.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan)
    need = max(min_breadth, top_k if long_only else 2 * top_k)
    for i in range(vals.shape[0]):
        row = vals[i]
        ok = np.flatnonzero(~np.isnan(row))
        if ok.size < need:
            continue
        order = ok[np.argsort(row[ok], kind="stable")]
        w = np.zeros(vals.shape[1])
        if long_only:
            w[order[-top_k:]] = 1.0 / top_k
        else:
            w[order[-top_k:]] = 0.5 / top_k
            w[order[:top_k]] = -0.5 / top_k
        out[i] = w
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


# --------------------------------------------------------------------------- #
# Backtest engine — drifting weights, one-way costs, borrow on the short leg
# --------------------------------------------------------------------------- #
def run_backtest(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    freq: str = "M",
    lookback: int = 252,
    skip: int = 21,
    top_k: int = 3,
    cost_bps: float = 5.0,
    borrow_bps: float = 40.0,
    long_only: bool = False,
    min_breadth: int = 6,
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the sleeve at one rebalance frequency and return the daily audit trail.

    Parameters
    ----------
    prices:
        Daily total-return close panel (date x asset). NaN before a name's inception.
    cash:
        Daily total-return close level of the cash leg (BIL). Required only for the
        ``long_only`` arm, whose excess-of-cash return subtracts it explicitly.
    freq:
        ``"D"`` / ``"W"`` / ``"M"`` / ``"Q"`` — the *only* thing that varies across arms.
    cost_bps:
        One-way transaction cost, in bps, charged on the traded notional as a fraction
        of NAV. An assumption, not tape — swept in ``cost_sweep``.
    borrow_bps:
        Annualised borrow fee on the short notional, accrued daily. An assumption, not
        tape — swept in ``borrow_sweep``. Ignored when ``long_only``.
    weights:
        Optional pre-computed target-weight frame (from :func:`target_weights`) — purely
        a speed hatch for the sweeps, which reuse one weight frame across every arm. The
        weights depend only on the signal, never on the rebalance clock.

    Returns a frame indexed by date with ``r_gross`` (before trading cost and borrow),
    ``r_net``, ``excess_gross`` / ``excess_net`` (the excess-of-cash convention for the
    arm), ``traded`` (notional traded that day, in units of NAV), ``short_notional`` and
    ``n_names``.
    """
    px = prices.sort_index().astype(float)
    rets = px.pct_change()
    if weights is None:
        sig = momentum_signal(px, lookback=lookback, skip=skip)
        weights = target_weights(sig, top_k=top_k, long_only=long_only,
                                 min_breadth=min_breadth)
    tgt = weights.reindex(px.index)
    rb_dates = rebalance_dates(px.index, freq)

    dates = px.index
    n = len(dates)
    tgt_arr = tgt.to_numpy(dtype=float)
    ret_arr = np.nan_to_num(rets.to_numpy(dtype=float), nan=0.0)
    is_rb = np.isin(dates.to_numpy(), rb_dates.to_numpy())
    tgt_ok = ~np.all(np.isnan(tgt_arr), axis=1)

    w = np.zeros(px.shape[1])
    live = False
    r_gross = np.full(n, np.nan)
    traded = np.zeros(n)
    traded_churn = np.zeros(n)
    traded_drift = np.zeros(n)
    short_n = np.zeros(n)
    n_names = np.zeros(n)

    for i in range(1, n):
        j = i - 1
        if is_rb[j] and tgt_ok[j]:
            new_w = np.nan_to_num(tgt_arr[j], nan=0.0)
            delta = np.abs(new_w - w)
            traded[i] = float(delta.sum())
            # Split the bill: notional traded because a name entered or left the book
            # (rank CHURN — the signal genuinely changed its mind) versus notional
            # traded merely to snap a still-held name back to its equal weight after it
            # DRIFTED. Only the first kind buys new information.
            churn_mask = (np.sign(new_w) != np.sign(w))
            traded_churn[i] = float(delta[churn_mask].sum())
            traded_drift[i] = float(delta[~churn_mask].sum())
            w = new_w
            live = True
        if not live:
            continue
        r_i = ret_arr[i]
        rp = float(w @ r_i)
        r_gross[i] = rp
        short_n[i] = float(-np.minimum(w, 0.0).sum())
        n_names[i] = float((np.abs(w) > 1e-12).sum())
        # Weights drift between rebalances — self-financing, so a position's share of the
        # book grows by its own return DIVIDED BY the book's. Without that denominator the
        # weights stay denominated in the *initial* NAV and the book silently levers up in
        # a rally and de-levers in a drawdown (measured: a long-only quarterly arm wandered
        # between 0.65x and 1.20x gross). Renormalising keeps gross where the mandate put it.
        denom = 1.0 + rp
        w = w * (1.0 + r_i) / denom if denom > 1e-9 else w * (1.0 + r_i)

    cost = traded * cost_bps * 1e-4
    borrow = 0.0 if long_only else short_n * (borrow_bps * 1e-4) / TRADING_DAYS
    out = pd.DataFrame(
        {
            "r_gross": r_gross,
            "r_net": r_gross - cost - borrow,
            "traded": traded,
            "traded_churn": traded_churn,
            "traded_drift": traded_drift,
            "short_notional": short_n,
            "n_names": n_names,
        },
        index=dates,
    ).dropna(subset=["r_gross"])

    if long_only:
        if cash is None:
            raise ValueError("the long_only arm needs a cash leg to measure excess-of-cash")
        r_cash = cash.sort_index().astype(float).pct_change().reindex(out.index)
        out["r_cash"] = r_cash
        out["excess_gross"] = out["r_gross"] - out["r_cash"]
        out["excess_net"] = out["r_net"] - out["r_cash"]
        out = out.dropna(subset=["excess_net"])
    else:
        # Dollar-neutral: the collateral earns cash, so the spread IS the excess return.
        out["excess_gross"] = out["r_gross"]
        out["excess_net"] = out["r_net"]
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


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
        wgt = 1.0 - l / (lags + 1.0)
        var += 2.0 * wgt * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return *difference* r1 - r2 (Jobson-Korkie, NW form).

    Both arms here are excess-of-cash by construction, so the cash leg cancels and the
    difference is a pure arm-vs-arm comparison.
    """
    diff = (pd.Series(r1) - pd.Series(r2)).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (
        float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    )
    return {
        "n_days": int(n),
        "sharpe": sharpe,
        "ann_return": float(mu * periods_per_year),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": cagr,
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 940,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series."""
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
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        ssd = s.std(ddof=1)
        if ssd > 0:
            boots[b] = s.mean() / ssd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": block}


# --------------------------------------------------------------------------- #
# The turnover budget — the study's headline object
# --------------------------------------------------------------------------- #
def turnover_stats(bt: pd.DataFrame) -> dict:
    """Annualised traded notional and per-rebalance turnover from a backtest frame."""
    traded = bt["traded"]
    n_rb = int((traded > 0).sum())
    years = len(bt) / TRADING_DAYS
    total = float(traded.sum())
    return {
        "ann_turnover": total / years if years > 0 else float("nan"),
        "n_rebalances": n_rb,
        "turnover_per_rebalance": float(traded[traded > 0].mean()) if n_rb else 0.0,
        "years": float(years),
        "churn_share": float(bt["traded_churn"].sum() / total) if total > 0 else float("nan"),
        "drift_share": float(bt["traded_drift"].sum() / total) if total > 0 else float("nan"),
    }


def _ensure_weights(prices: pd.DataFrame, kw: dict, long_only: bool = False) -> dict:
    """Compute the target-weight frame once and thread it through a sweep.

    The weights depend only on the signal, never on the rebalance clock, so every arm of
    a sweep shares one frame. Pure plumbing — it changes no number.
    """
    if kw.get("weights") is None:
        sig = momentum_signal(prices, lookback=kw.get("lookback", 252),
                              skip=kw.get("skip", 21))
        kw["weights"] = target_weights(sig, top_k=kw.get("top_k", 3),
                                       long_only=long_only,
                                       min_breadth=kw.get("min_breadth", 6))
    return kw


def run_breakeven(
    prices: pd.DataFrame,
    cash: pd.Series | None,
    freq: str,
    borrow_bps: float = 40.0,
    long_only: bool = False,
    **kw,
) -> dict:
    """Zero-trading-cost run for ``freq``, plus the break-even cost per unit turnover.

    Borrow is still charged (it is a holding cost, not a trading cost), so the break-even
    is the honest "how many bps per unit traded notional can this arm afford".
    """
    bt = run_backtest(prices, cash, freq=freq, cost_bps=0.0, borrow_bps=borrow_bps,
                      long_only=long_only, **kw)
    mu = float(bt["excess_net"].mean())
    mean_traded = float(bt["traded"].mean())
    be = float(mu / (mean_traded * 1e-4)) if mean_traded > 0 else float("nan")
    ts = turnover_stats(bt)
    return {"freq": freq, "breakeven_bps": be, "mean_daily_excess": mu, **ts, "bt": bt}


def frequency_table(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    freqs=FREQS,
    cost_bps: float = 5.0,
    borrow_bps: float = 40.0,
    long_only: bool = False,
    **kw,
) -> pd.DataFrame:
    """The core table: gross vs net excess-of-cash Sharpe, turnover and break-even cost.

    One row per rebalance frequency. ``sharpe_gross`` charges neither trading cost nor
    borrow; ``sharpe_net`` charges both at (``cost_bps``, ``borrow_bps``). ``breakeven_bps``
    is the cost per unit traded notional that would take the net excess return to zero
    with borrow still paid.
    """
    kw = _ensure_weights(prices, dict(kw), long_only=long_only)
    rows = []
    for f in freqs:
        bt = run_backtest(prices, cash, freq=f, cost_bps=cost_bps, borrow_bps=borrow_bps,
                          long_only=long_only, **kw)
        be = run_breakeven(prices, cash, f, borrow_bps=borrow_bps, long_only=long_only, **kw)
        s_gross = summary(bt["excess_gross"])
        s_net = summary(bt["excess_net"])
        ts = turnover_stats(bt)
        rows.append({
            "freq": f,
            "label": FREQ_LABEL[f],
            "n_days": s_net["n_days"],
            "n_rebalances": ts["n_rebalances"],
            "ann_turnover": ts["ann_turnover"],
            "turnover_per_rebalance": ts["turnover_per_rebalance"],
            "churn_share": ts["churn_share"],
            "sharpe_gross": s_gross["sharpe"],
            "sharpe_net": s_net["sharpe"],
            "ann_return_gross": s_gross["ann_return"],
            "ann_return_net": s_net["ann_return"],
            "vol_ann": s_net["vol_ann"],
            "max_drawdown": s_net["max_drawdown"],
            "t_gross": s_gross["tstat"],
            "t_net": s_net["tstat"],
            "breakeven_bps": be["breakeven_bps"],
        })
    return pd.DataFrame(rows).set_index("freq")


def cost_sweep(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    freqs=FREQS,
    cost_grid=(0.0, 1.0, 2.5, 5.0, 10.0, 25.0),
    borrow_bps: float = 40.0,
    long_only: bool = False,
    **kw,
) -> pd.DataFrame:
    """Net excess-of-cash Sharpe for every (frequency, cost) pair — the headline surface.

    The cost per unit turnover is an **assumption**, so it is swept rather than asserted:
    1 bp is an aggressive institutional fill on a penny-spread sector ETF, 5 bps a
    realistic retail all-in, 25 bps a punitive one.
    """
    kw = _ensure_weights(prices, dict(kw), long_only=long_only)
    rows = []
    for f in freqs:
        for c in cost_grid:
            bt = run_backtest(prices, cash, freq=f, cost_bps=c, borrow_bps=borrow_bps,
                              long_only=long_only, **kw)
            s = summary(bt["excess_net"])
            rows.append({"freq": f, "label": FREQ_LABEL[f], "cost_bps": c,
                         "sharpe_net": s["sharpe"], "ann_return_net": s["ann_return"],
                         "t_net": s["tstat"]})
    return pd.DataFrame(rows)


def borrow_sweep(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    freqs=FREQS,
    borrow_grid=(0.0, 25.0, 50.0, 100.0),
    cost_bps: float = 5.0,
    **kw,
) -> pd.DataFrame:
    """Net Sharpe across borrow-rate assumptions — the second non-tape input, swept.

    Sector SPDRs are general-collateral easy-to-borrow, so 25-50 bps/yr is the realistic
    band; 100 bps is a deliberately harsh stress. Borrow is a *holding* cost, so it hits
    every frequency almost equally — which is exactly why it cannot rescue or condemn a
    speed on its own.
    """
    kw = _ensure_weights(prices, dict(kw))
    rows = []
    for f in freqs:
        for b in borrow_grid:
            bt = run_backtest(prices, cash, freq=f, cost_bps=cost_bps, borrow_bps=b, **kw)
            s = summary(bt["excess_net"])
            rows.append({"freq": f, "label": FREQ_LABEL[f], "borrow_bps": b,
                         "sharpe_net": s["sharpe"], "ann_return_net": s["ann_return"]})
    return pd.DataFrame(rows)


def era_cut(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    split: str = "2013-01-01",
    freqs=FREQS,
    cost_bps: float = 5.0,
    borrow_bps: float = 40.0,
    **kw,
) -> dict:
    """Split the sample at ``split`` and rebuild the frequency table on each half.

    A real turnover budget should keep the same *shape* (which speed wins, and roughly
    how much friction it can afford) in both eras; a lucky one re-orders.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        p = prices.loc[sl]
        c = cash.loc[sl] if cash is not None else None
        if len(p) < 400:
            out[tag] = None
            continue
        out[tag] = frequency_table(p, c, freqs=freqs, cost_bps=cost_bps,
                                   borrow_bps=borrow_bps, **kw)
    return out


def frequency_race(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    fast: str = "D",
    slow: str = "Q",
    cost_bps: float = 5.0,
    borrow_bps: float = 40.0,
    **kw,
) -> dict:
    """Head-to-head between two speeds: net excess Sharpe gap and the HAC *t* on it.

    Both arms are excess-of-cash, so the daily difference is a clean paired comparison —
    the same signal, the same tape, only the rebalance clock differs.
    """
    bt_f = run_backtest(prices, cash, freq=fast, cost_bps=cost_bps, borrow_bps=borrow_bps, **kw)
    bt_s = run_backtest(prices, cash, freq=slow, cost_bps=cost_bps, borrow_bps=borrow_bps, **kw)
    common = bt_f.index.intersection(bt_s.index)
    ef = bt_f["excess_net"].reindex(common)
    es = bt_s["excess_net"].reindex(common)
    return {
        "fast": fast, "slow": slow, "n_days": int(len(common)),
        "sharpe_fast": summary(ef)["sharpe"],
        "sharpe_slow": summary(es)["sharpe"],
        "sharpe_gap": summary(ef)["sharpe"] - summary(es)["sharpe"],
        "t_diff": sharpe_diff_tstat(ef, es),
        "e_fast": ef, "e_slow": es,
    }


def equal_weight_targets(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    min_breadth: int = 6,
) -> pd.DataFrame:
    """Equal weight over exactly the names the momentum sleeve could have picked.

    Built from the *same* investability mask as :func:`momentum_signal` (a sector counts
    only once its own 12-1 window has filled), so the benchmark and the strategy share a
    universe, a start date and a breadth path. It carries no cross-sectional view — it is
    the "same clock, same costs, no signal" control.
    """
    sig = momentum_signal(prices, lookback=lookback, skip=skip)
    ok = sig.notna()
    n = ok.sum(axis=1)
    w = ok.astype(float).div(n.where(n > 0), axis=0)
    w.loc[n < min_breadth] = np.nan
    return w


def long_only_vs_equal_weight(
    prices: pd.DataFrame,
    cash: pd.Series,
    freqs=FREQS,
    cost_bps: float = 5.0,
    **kw,
) -> pd.DataFrame:
    """Long-only cross-check, raced against a **cost-matched** equal-weight book.

    A long-only top-``k`` momentum sleeve is *always fully invested in equities*, so its
    excess-of-cash Sharpe is dominated by plain equity beta. Racing it against an
    equal-weight book of the same sectors strips that beta out; what is left is the part
    momentum *selection* contributed. Reporting the raw excess-of-cash *t* for this arm —
    which clears 2 — without this race would be the classic beta-dressed-as-alpha error.

    The race is run **symmetrically**: the equal-weight leg goes through the same engine,
    on the same rebalance clock, with the same one-day lag, and pays the same ``cost_bps``
    on *its* own traded notional. An earlier cut of this study benchmarked against a
    frictionless daily-rebalanced equal-weight average while charging the momentum arm 5
    bps on 28x NAV a year; that one-sided ledger manufactured a −0.7%/yr "shortfall" at
    the daily clock out of nothing but the strategy's own commission. Both the net
    (``alpha_ann`` / ``t_vs_ew``) and the cost-free (``alpha_ann_gross`` / ``t_vs_ew_gross``)
    versions are reported so the reader can see how much of any gap is selection and how
    much is friction.
    """
    kw = _ensure_weights(prices, dict(kw), long_only=True)
    ew_w = equal_weight_targets(
        prices,
        lookback=kw.get("lookback", 252),
        skip=kw.get("skip", 21),
        min_breadth=kw.get("min_breadth", 6),
    )
    ew_kw = {k: v for k, v in kw.items() if k != "weights"}
    rows = []
    for f in freqs:
        bt = run_backtest(prices, cash, freq=f, cost_bps=cost_bps, long_only=True, **kw)
        btg = run_backtest(prices, cash, freq=f, cost_bps=0.0, long_only=True, **kw)
        ew = run_backtest(prices, cash, freq=f, cost_bps=cost_bps, long_only=True,
                          weights=ew_w, **ew_kw)
        ewg = run_backtest(prices, cash, freq=f, cost_bps=0.0, long_only=True,
                           weights=ew_w, **ew_kw)
        common = bt.index.intersection(ew.index)
        diff = (bt["excess_net"].reindex(common) - ew["excess_net"].reindex(common)).dropna()
        diff_g = (btg["excess_net"].reindex(common)
                  - ewg["excess_net"].reindex(common)).dropna()
        s = summary(bt["excess_net"])
        rows.append({
            "freq": f, "label": FREQ_LABEL[f],
            "sharpe_net": s["sharpe"], "t_excess_of_cash": s["tstat"],
            "sharpe_ew": summary(ew["excess_net"])["sharpe"],
            "alpha_ann": float(diff.mean() * TRADING_DAYS),
            "t_vs_ew": newey_west_t(diff.to_numpy()),
            "alpha_ann_gross": float(diff_g.mean() * TRADING_DAYS),
            "t_vs_ew_gross": newey_west_t(diff_g.to_numpy()),
            "ann_turnover": turnover_stats(bt)["ann_turnover"],
            "ann_turnover_ew": turnover_stats(ew)["ann_turnover"],
        })
    return pd.DataFrame(rows).set_index("freq")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices: pd.DataFrame,
    cash: pd.Series | None = None,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
    top_k: int = 3,
    min_breadth: int = 6,
) -> dict:
    """Run the frequency table on a synthetic panel and report the planted-effect reads.

    On a planted panel (persistent expected returns) the **gross** excess return must be
    positive and should decay as the rebalance slows — the mechanism this study prices.
    On the null panel every frequency must sit at zero. Proves the harness is unbiased;
    it never supports a real-tape stamp.
    """
    tbl = frequency_table(prices, cash, cost_bps=cost_bps, borrow_bps=borrow_bps,
                          top_k=top_k, min_breadth=min_breadth)
    return {
        "sharpe_gross": {f: float(tbl.loc[f, "sharpe_gross"]) for f in tbl.index},
        "ann_return_gross": {f: float(tbl.loc[f, "ann_return_gross"]) for f in tbl.index},
        "t_gross": {f: float(tbl.loc[f, "t_gross"]) for f in tbl.index},
        "ann_turnover": {f: float(tbl.loc[f, "ann_turnover"]) for f in tbl.index},
        "daily_minus_quarterly_gross": float(
            tbl.loc["D", "ann_return_gross"] - tbl.loc["Q", "ann_return_gross"]
        ),
        "table": tbl,
    }
