"""Strategy + inference for Study 938 — Open or Close.

**One rule, two fills.** The rule is the plain Faber/time-series-momentum filter: hold the
ETF when its period-end close is above its ``lookback``-period moving average of period-end
closes, else hold BIL (cash). Binary, long-only, no shorting — so there is **no short leg
and no borrow cost** anywhere in this study. Two flavours of the same rule are run: a
**10-month** version rebalanced at month ends and a **20-week** version rebalanced at week
ends (more trades, more power).

**Exactly one execution lag, and it is the only thing that differs.** The signal is formed
from data through the close of the rebalance day ``t``; both arms fill on the **next**
session ``t+1``. They differ only in *where in that session* the fill happens:

* ``open`` arm — fills at the **open of ``t+1``**. It therefore holds the *old* position
  across the overnight leg (close ``t`` → open ``t+1``) and the *new* position across the
  intraday leg (open ``t+1`` → close ``t+1``).
* ``close`` arm — fills at the **close of ``t+1``**. It holds the *old* position for the
  whole of day ``t+1``, and the new one from that close onwards.

Every non-trade day is identical in the two arms, so the entire difference between them is
a thin sliver: on each trade day the arms differ by ``Δweight × (intraday return of the
asset − intraday return of cash)``. That sliver is the **implicit slippage of choosing a
venue**, and this study asks whether it is systematic (HAC *t*) or noise, across four tapes
and two eras.

**Which return is which.** With ``auto_adjust=True`` the intraday leg (open → close) is a
*price-only* return and the overnight leg carries the dividend; their product is the
familiar close-to-close *total* return. Legs are labelled accordingly throughout.

**Cash within the day.** BIL's own printed open is sub-penny noise, so the cash leg's daily
total return is split between night and day by an explicit **assumption**:
``cash_overnight_frac`` (default 0.73 ≈ 17.5h of 24h). It is a PROXY and it is swept — at
~1 bp/day of cash return it cannot move the answer, and the sweep shows exactly that.

Returns are **simple** (arithmetic) within each leg; the open arm's two legs are compounded
within the day, so the wealth path is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# PROXY / ASSUMPTION: share of a day's cash (T-bill) accrual booked to the overnight leg.
# 17.5 of 24 hours sit between the 16:00 close and the 09:30 open. Swept in `cash_split_sweep`.
CASH_OVERNIGHT_FRAC = 0.73


# --------------------------------------------------------------------------- #
# The rule: a period-end moving-average filter -> a daily target weight
# --------------------------------------------------------------------------- #
def period_end_mask(index: pd.DatetimeIndex, freq: str = "M") -> pd.Series:
    """Boolean series marking the last trading day of each month (``M``) or week (``W``)."""
    idx = pd.DatetimeIndex(index)
    if freq.upper().startswith("M"):
        key = idx.to_period("M")
    elif freq.upper().startswith("W"):
        key = idx.to_period("W")
    else:
        raise ValueError("freq must be 'M' (month-end) or 'W' (week-end)")
    s = pd.Series(key, index=idx)
    return s != s.shift(-1)


def rule_target(close: pd.Series, freq: str = "M", lookback: int = 10) -> pd.Series:
    """Daily target weight {0, 1} formed from data **through the close of day t**.

    At each period end the rule compares that period-end close to the mean of the last
    ``lookback`` period-end closes (inclusive). The resulting weight is carried forward on
    every subsequent day until the next period end — so ``target[t]`` is always "what the
    rule wants, knowing everything up to the close of ``t``". Filling it is ``strategy``'s
    job, and both arms fill on ``t+1``.

    NaN through the warm-up (before ``lookback`` period ends exist).
    """
    close = pd.Series(close).astype(float).dropna().sort_index()
    ends = period_end_mask(close.index, freq=freq)
    pe = close[ends]
    ma = pe.rolling(lookback, min_periods=lookback).mean()
    sig = (pe > ma).astype(float)
    sig[ma.isna()] = np.nan
    target = sig.reindex(close.index).ffill()
    return target.rename("target")


# --------------------------------------------------------------------------- #
# Return legs
# --------------------------------------------------------------------------- #
def return_legs(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Split adjusted OHLC into ``r_cc`` (total), ``r_on`` (overnight), ``r_id`` (intraday).

    ``(1 + r_on)(1 + r_id) = 1 + r_cc`` exactly. ``r_id`` is price-only (the daily
    adjustment factor cancels); ``r_on`` carries the dividend step.
    """
    o = pd.Series(ohlc["open"]).astype(float)
    c = pd.Series(ohlc["close"]).astype(float)
    r_cc = c.pct_change().rename("r_cc")
    r_on = (o / c.shift(1) - 1.0).rename("r_on")
    r_id = (c / o - 1.0).rename("r_id")
    return pd.concat([r_cc, r_on, r_id], axis=1)


# --------------------------------------------------------------------------- #
# The two execution arms
# --------------------------------------------------------------------------- #
def run_arms(
    ohlc: pd.DataFrame,
    cash_close: pd.Series,
    target: pd.Series,
    cost_bps: float = 2.0,
    open_extra_bps: float = 0.0,
    cash_overnight_frac: float = CASH_OVERNIGHT_FRAC,
) -> pd.DataFrame:
    """Run the same target weights filled at the next **open** vs the next **close**.

    Parameters
    ----------
    ohlc:
        Adjusted daily ``open``/``close`` of the risk asset.
    cash_close:
        Daily total-return close of the cash leg (BIL).
    target:
        Daily {0, 1} target weight formed through the close of each day (``rule_target``).
    cost_bps:
        One-way transaction cost in bps of NAV, charged on ``|Δweight|`` on the trade day.
        Both arms trade the same amount on the same day, so this is symmetric.
    open_extra_bps:
        PROXY / ASSUMPTION — the *extra* one-way cost of filling in the opening auction
        rather than the closing auction (wider spread, thinner book). Swept, not assumed
        away. 0 = the two venues cost the same.
    cash_overnight_frac:
        PROXY — share of the cash leg's daily accrual booked overnight.

    Returns a frame with the leg returns, the two arms' gross and net daily returns, the
    trade indicator, and the excess-of-cash versions of both arms.
    """
    legs = return_legs(ohlc)
    cash = pd.Series(cash_close).astype(float)
    r_cash = cash.pct_change().rename("r_cash")

    df = pd.concat([legs, r_cash], axis=1).dropna()
    tgt = pd.Series(target).astype(float).reindex(df.index)

    # Weights. target[t] is filled on t+1: intraday of t+1 for the open arm, and from the
    # close of t+1 for the close arm (i.e. it governs day t+2 onwards).
    w_id = tgt.shift(1)     # open arm, intraday leg of day u
    w_on = tgt.shift(2)     # open arm, overnight leg of day u (position carried into it)
    w_cc = tgt.shift(2)     # close arm, the whole of day u

    df = df.assign(w_id=w_id, w_on=w_on, w_cc=w_cc).dropna()

    f = float(cash_overnight_frac)
    r_cash_on = f * df["r_cash"]
    r_cash_id = (1.0 - f) * df["r_cash"]

    leg_on = df["w_on"] * df["r_on"] + (1.0 - df["w_on"]) * r_cash_on
    leg_id = df["w_id"] * df["r_id"] + (1.0 - df["w_id"]) * r_cash_id
    r_open_gross = ((1.0 + leg_on) * (1.0 + leg_id) - 1.0).rename("r_open_gross")
    r_close_gross = (df["w_cc"] * df["r_cc"] + (1.0 - df["w_cc"]) * df["r_cash"]).rename(
        "r_close_gross"
    )

    trade = (df["w_id"] - df["w_on"]).abs().rename("trade")
    c_close = cost_bps * 1e-4
    c_open = (cost_bps + open_extra_bps) * 1e-4
    r_open = (r_open_gross - trade * c_open).rename("r_open")
    r_close = (r_close_gross - trade * c_close).rename("r_close")

    out = pd.concat(
        [df, r_open_gross, r_close_gross, r_open, r_close, trade], axis=1
    )
    out["e_open"] = out["r_open"] - out["r_cash"]
    out["e_close"] = out["r_close"] - out["r_cash"]
    out["diff"] = out["r_open"] - out["r_close"]
    out["diff_gross"] = out["r_open_gross"] - out["r_close_gross"]
    return out.dropna()


# --------------------------------------------------------------------------- #
# Inference primitives (desk-standard)
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
    return float(mu / np.sqrt(var / n))


def cluster_t(x, dates) -> float:
    """*t* of mean(x) vs 0 with a cluster-robust SE, clustering on the **trade date**.

    The pooled fill sample is not 538 independent draws: the four ETF tapes are highly
    correlated and their timing signals flip on the same days, and the two rules are run on
    the same window, so several fills share a calendar date and a common shock. Clustering
    on the date is the minimum honest correction; it is what the naive one-sample *t*
    silently assumes away.
    """
    x = np.asarray(pd.Series(x).astype(float))
    d = np.asarray(pd.Series(dates))
    ok = ~np.isnan(x)
    x, d = x[ok], d[ok]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = pd.Series(x - mu).groupby(pd.Series(d)).sum().to_numpy()
    g = e.size
    if g < 2:
        return float("nan")
    se = np.sqrt(float((e ** 2).sum())) / n * np.sqrt(g / (g - 1.0))
    return float(mu / se) if se > 0 else float("nan")


def cluster_bootstrap_trades(x, dates, n_boot: int = 4000, seed: int = 938,
                             alpha: float = 0.05) -> dict:
    """Resample whole **trade dates** (not individual fills) — CI on the mean and win rate.

    Same correction as :func:`cluster_t`, applied to the two headline statistics of the
    pooled fill sample. Deterministic given ``seed``.
    """
    s = pd.Series(np.asarray(pd.Series(x).astype(float)))
    d = pd.Series(np.asarray(pd.Series(dates)))
    keys = np.array(sorted(d.unique()))
    groups = [s[(d == k).to_numpy()].to_numpy() for k in keys]
    rng = np.random.default_rng(seed)
    m = np.empty(n_boot)
    w = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        m[b] = v.mean() * 1e4
        w[b] = (v > 0).mean()
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "n_fills": int(s.size), "n_dates": int(len(keys)),
        "mean_bps": float(s.mean() * 1e4),
        "mean_ci": tuple(float(v) for v in np.percentile(m, q)),
        "win_rate": float((s > 0).mean()),
        "win_ci": tuple(float(v) for v in np.percentile(w, q)),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on ``k``/``n``.

    Assumes ``n`` **independent** Bernoulli draws. On the pooled fill sample that assumption
    is false (correlated tapes, shared dates), so this interval is optimistically narrow —
    :func:`cluster_bootstrap_trades` gives the honest, wider one and both are reported.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_ci(
    x,
    stat: str = "mean_ann_bps",
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 938,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the mean (annualised, bps) or Sharpe of a series.

    Blocks of ``block`` consecutive days preserve the clustering of trade days and vol.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}

    def _stat(a: np.ndarray) -> float:
        if stat == "sharpe":
            sd = a.std(ddof=1)
            return float(a.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        return float(a.mean() * TRADING_DAYS * 1e4)

    point = _stat(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = _stat(r[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block), "stat": stat}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summary(returns, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def venue_gap(arms: pd.DataFrame, net: bool = True) -> dict:
    """The headline: is the open-minus-close execution gap systematic or noise?

    Reports the annualised gap in bps, its HAC *t*, the Sharpe of each arm
    (excess-of-cash), the per-trade slippage, and the trade count.

    Both growth rates are reported and both are labelled: ``cagr_*`` is the **excess-of-cash**
    compounded growth of the arm (the series the Sharpe is built on), ``cagr_*_total`` is the
    raw, cash-inclusive CAGR of the same book. They differ by roughly the T-bill return, and
    conflating the two is the easiest way to overstate a strategy.
    """
    col = "diff" if net else "diff_gross"
    d = arms[col].dropna()
    trades = arms.loc[arms["trade"] > 0, col].dropna()
    s_open = summary(arms["e_open"])
    s_close = summary(arms["e_close"])
    t_open = summary(arms["r_open"])
    t_close = summary(arms["r_close"])
    return {
        "n_days": int(len(d)),
        "n_trades": int((arms["trade"] > 0).sum()),
        "gap_ann_bps": float(d.mean() * TRADING_DAYS * 1e4),
        "gap_t_hac": newey_west_t(d.to_numpy()),
        "gap_per_trade_bps": float(trades.mean() * 1e4) if len(trades) else float("nan"),
        "gap_per_trade_t": one_sample_t(trades.to_numpy()) if len(trades) else float("nan"),
        "win_rate_trades": float((trades > 0).mean()) if len(trades) else float("nan"),
        "sharpe_open": s_open["sharpe"],
        "sharpe_close": s_close["sharpe"],
        "sharpe_gap": s_open["sharpe"] - s_close["sharpe"],
        "cagr_open": s_open["cagr"],            # EXCESS of cash
        "cagr_close": s_close["cagr"],          # EXCESS of cash
        "cagr_open_total": t_open["cagr"],      # raw, cash-inclusive
        "cagr_close_total": t_close["cagr"],    # raw, cash-inclusive
        "dd_open": s_open["max_drawdown"],
        "dd_close": s_close["max_drawdown"],
    }


# --------------------------------------------------------------------------- #
# One tape, one rule
# --------------------------------------------------------------------------- #
def run_tape(
    ohlc: pd.DataFrame,
    cash_close: pd.Series,
    freq: str = "M",
    lookback: int = 10,
    cost_bps: float = 2.0,
    open_extra_bps: float = 0.0,
    cash_overnight_frac: float = CASH_OVERNIGHT_FRAC,
) -> pd.DataFrame:
    """Align one risk tape with the cash leg, build the rule, and run both arms."""
    common = ohlc.index.intersection(pd.Series(cash_close).index)
    o = ohlc.loc[common]
    c = pd.Series(cash_close).loc[common]
    tgt = rule_target(o["close"], freq=freq, lookback=lookback)
    return run_arms(o, c, tgt, cost_bps=cost_bps, open_extra_bps=open_extra_bps,
                    cash_overnight_frac=cash_overnight_frac)


def run_panel(
    frames: dict,
    cash_close: pd.Series,
    tickers=None,
    **kwargs,
) -> dict:
    """Run every risk tape and build the equal-weight pooled book.

    The pooled book holds 1/N of each tape's arm; the pooled gap is the equal-weight
    average of the four per-tape gaps, on the common trading calendar.
    """
    tickers = list(tickers) if tickers is not None else [t for t in frames]
    per = {tk: run_tape(frames[tk], cash_close, **kwargs) for tk in tickers}
    cols = ["r_open", "r_close", "r_cash", "e_open", "e_close", "diff", "diff_gross", "trade"]
    idx = None
    for tk in tickers:
        idx = per[tk].index if idx is None else idx.intersection(per[tk].index)
    pooled = sum(per[tk].loc[idx, cols] for tk in tickers) / float(len(tickers))
    return {"per_tape": per, "pooled": pooled, "tickers": tickers}


# --------------------------------------------------------------------------- #
# Robustness: eras, cost sweep, cash-split sweep
# --------------------------------------------------------------------------- #
def era_cut(arms: pd.DataFrame, split: str = "2017-01-01") -> dict:
    """Split the daily arm frame at ``split`` and re-measure the gap on each half."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = arms.loc[sl]
        out[tag] = venue_gap(sub) if len(sub) > 60 else None
    return out


def cost_sweep(
    ohlc: pd.DataFrame,
    cash_close: pd.Series,
    extra_grid=(0.0, 1.0, 2.0, 5.0, 10.0),
    **kwargs,
) -> list[dict]:
    """Sweep the PROXY opening-auction spread penalty charged to the open arm."""
    rows = []
    for x in extra_grid:
        arms = run_tape(ohlc, cash_close, open_extra_bps=x, **kwargs)
        g = venue_gap(arms)
        rows.append({"open_extra_bps": float(x), **{k: g[k] for k in
                    ("gap_ann_bps", "gap_t_hac", "gap_per_trade_bps",
                     "sharpe_open", "sharpe_close", "n_trades")}})
    return rows


def cash_split_sweep(
    ohlc: pd.DataFrame,
    cash_close: pd.Series,
    frac_grid=(0.0, 0.5, 0.73, 1.0),
    **kwargs,
) -> list[dict]:
    """Sweep the PROXY overnight share of the cash accrual — it must not matter."""
    rows = []
    for f in frac_grid:
        arms = run_tape(ohlc, cash_close, cash_overnight_frac=f, **kwargs)
        g = venue_gap(arms)
        rows.append({"cash_overnight_frac": float(f),
                     "gap_ann_bps": g["gap_ann_bps"], "gap_t_hac": g["gap_t_hac"]})
    return rows


# --------------------------------------------------------------------------- #
# Mechanism: what the sliver is made of
# --------------------------------------------------------------------------- #
def trade_day_decomposition(arms: pd.DataFrame) -> dict:
    """Mean intraday (price-only) asset return on entry days vs exit days.

    The gap is ``Δweight × (asset intraday − cash intraday)`` summed over trade days. If
    intraday returns had a constant mean, entries (Δw = +1) and exits (Δw = −1) would
    cancel over a full cycle. A non-zero gap therefore needs the *conditional* means to
    differ — daily continuation right after a monthly/weekly signal flip.
    """
    tr = arms[arms["trade"] > 0]
    entries = tr[tr["w_id"] > tr["w_on"]]["r_id"]
    exits = tr[tr["w_id"] < tr["w_on"]]["r_id"]
    allday = arms["r_id"]
    return {
        "n_entries": int(len(entries)),
        "n_exits": int(len(exits)),
        "entry_intraday_bps": float(entries.mean() * 1e4) if len(entries) else float("nan"),
        "exit_intraday_bps": float(exits.mean() * 1e4) if len(exits) else float("nan"),
        "all_intraday_bps": float(allday.mean() * 1e4),
        "entry_t": one_sample_t(entries.to_numpy()) if len(entries) > 1 else float("nan"),
        "exit_t": one_sample_t(exits.to_numpy()) if len(exits) > 1 else float("nan"),
        "spread_bps": (float(entries.mean() * 1e4) - float(exits.mean() * 1e4))
        if len(entries) and len(exits) else float("nan"),
        "spread_t": welch_t(entries.to_numpy(), exits.to_numpy())
        if len(entries) > 1 and len(exits) > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, freq: str = "W", lookback: int = 20,
                     cost_bps: float = 0.0, **kwargs) -> dict:
    """Run the venue race on a synthetic ``(open, close, cash)`` tape.

    On the planted world (intraday continuation after strong/weak months) filling at the
    open must win with a clear HAC *t*; on the null (``signal_strength=0``) the gap must be
    centred on zero. Proves the harness is unbiased.
    """
    arms = run_arms(frame[["open", "close"]], frame["cash"],
                    rule_target(frame["close"], freq=freq, lookback=lookback),
                    cost_bps=cost_bps, **kwargs)
    g = venue_gap(arms)
    g["decomp"] = trade_day_decomposition(arms)
    return g
