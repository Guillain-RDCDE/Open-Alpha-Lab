"""Experiment + inference for Study 934 — Lump Sum vs DCA.

The experiment. Take $1 of new money. Two ways to put it to work over the next year:

1. **Lump sum** — the whole dollar into the asset at the first execution date, held
   for twelve months.
2. **DCA** — twelve equal tranches of 1/12, one at each month-end over the year. The
   balance that has not been invested yet sits in **BIL** and earns its *actual*
   total return, so the cash leg is credited the real path of short rates (0% in
   2009-2015, ~5% in 2023-2026) instead of the 0% that flatters the folk story.

Both arms are valued on the **same terminal date**, so the comparison is a clean
terminal-wealth race with no horizon mismatch. We roll the experiment over **every
start month** of the sample and keep the whole distribution: the win rate, the mean
and median gap, the dispersion, and the tails.

**One execution lag, documented.** The decision belongs to a month-end close; it is
executed at the **next trading day's close**. Every purchase — the single lump-sum
buy and each of the twelve DCA tranches — obeys the same lag, so neither arm can peek.

**Costs.** One-way, charged on NAV at each purchase. The lump sum pays once; DCA pays
twelve times. Neither arm ever shorts, so there is no borrow leg. The cost sweep is
in :func:`cost_sweep`.

**Inference.** The windows overlap (monthly starts, twelve-month horizons), so the
gap series is autocorrelated by construction: the *t* on the mean gap is Newey-West
with 12 lags, backed by a circular block bootstrap (12-month blocks) and by a
non-overlapping check that keeps only every twelfth start.

**The exposure confound, and the test that settles it.** DCA is not just *later* into
the market, it is *less* in it: its time-weighted exposure is only ``(n+1)/(2n)``
(13/24 for twelve tranches). A raw terminal-wealth race therefore compares a
full-beta portfolio with a half-beta one, and any positive risk premium hands the
lump sum a win it did not earn by *timing*. :func:`exposure_matched_race` removes
that confound by racing DCA against a **static** portfolio holding the same average
exposure in the same asset and the same cash leg for the whole window — and the
risk-adjusted reads (:func:`exposure_matched_race`) are excess of the same cash leg
on **both** arms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
N_TRANCHES = 12


# --------------------------------------------------------------------------- #
# The monthly grid
# --------------------------------------------------------------------------- #
def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().to_numpy())


def execution_dates(index: pd.DatetimeIndex, decision_dates: pd.DatetimeIndex) -> list:
    """Map each decision date to the NEXT trading day in ``index`` (the one-day lag).

    Returns a list the same length as ``decision_dates``; entries whose successor
    falls outside the sample are ``None``.
    """
    pos = index.get_indexer(decision_dates)
    out = []
    for p in pos:
        nxt = p + 1
        out.append(index[nxt] if (p >= 0 and nxt < len(index)) else None)
    return out


# --------------------------------------------------------------------------- #
# The head-to-head
# --------------------------------------------------------------------------- #
def run_experiment(
    asset: pd.Series,
    cash: pd.Series,
    n_tranches: int = N_TRANCHES,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Roll the lump-sum vs DCA race over every start month of the common sample.

    Parameters
    ----------
    asset, cash:
        Daily **total-return close levels** of the risk asset and of the cash leg
        (BIL), aligned on a common index.
    n_tranches:
        How many equal monthly tranches DCA uses (12 = the classic "average in over
        a year").
    cost_bps:
        One-way cost in bps of NAV, charged at every purchase. The lump sum pays it
        once; DCA pays it ``n_tranches`` times.

    Returns one row per start month, with the terminal wealth of each arm per $1
    invested, the gap in cents, the asset's own return over the window, and the two
    conditioning variables measured **at the start date only** (no look-ahead).
    """
    common = asset.index.intersection(cash.index)
    a = asset.reindex(common).sort_index().dropna()
    c = cash.reindex(common).sort_index().dropna()
    common = a.index.intersection(c.index)
    a, c = a.loc[common], c.loc[common]

    me = month_end_dates(common)
    ex = execution_dates(common, me)
    keep = [i for i, d in enumerate(ex) if d is not None]
    me = me[keep]
    ex = [ex[i] for i in keep]

    cost = cost_bps * 1e-4
    # Conditioning variables, computed on the asset tape and read only at t0.
    stretch = (a / a.rolling(756, min_periods=756).mean() - 1.0)   # PROXY for "stretched"
    drawdown = (a / a.cummax() - 1.0)

    rows = []
    n_starts = len(ex) - n_tranches
    for i in range(max(n_starts, 0)):
        buys = ex[i:i + n_tranches]
        d_end = ex[i + n_tranches]
        p0, pT = a.loc[buys[0]], a.loc[d_end]
        c0 = c.loc[buys[0]]

        w_lump = (1.0 - cost) * pT / p0

        w_dca = 0.0
        for d in buys:
            # The tranche waits in cash from d0 to its own buy date, then rides the asset.
            w_dca += (1.0 / n_tranches) * (c.loc[d] / c0) * (1.0 - cost) * (pT / a.loc[d])

        rows.append({
            "start": buys[0],
            "end": d_end,
            "w_lump": float(w_lump),
            "w_dca": float(w_dca),
            "gap_cents": float((w_lump - w_dca) * 100.0),
            "asset_ret": float(pT / p0 - 1.0),
            "cash_ret": float(c.loc[d_end] / c0 - 1.0),
            "stretch0": float(stretch.loc[buys[0]]) if np.isfinite(stretch.loc[buys[0]]) else np.nan,
            "dd0": float(drawdown.loc[buys[0]]),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index("start")
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of the desk's shared machinery)
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


def newey_west_t(x, lags: int = 12) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    With monthly starts and twelve-month horizons, consecutive windows share up to
    eleven months of tape; ``lags=12`` is the natural overlap correction.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_mean_ci(
    x,
    n_boot: int = 4000,
    block: int = 12,
    seed: int = 934,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the mean of an overlapping-window series.

    Twelve-month blocks keep a whole horizon's worth of overlap intact, so the
    interval does not pretend the windows are independent.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
        "n_boot": int(n_boot), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# The headline read of one experiment frame
# --------------------------------------------------------------------------- #
def summarise(exp: pd.DataFrame, n_tranches: int = N_TRANCHES,
              boot_seed: int = 934, n_boot: int = 4000) -> dict:
    """Win rate, gap moments, dispersion, and the overlap-corrected inference."""
    if len(exp) == 0:
        return {"n_windows": 0}
    gap = exp["gap_cents"].to_numpy(dtype=float)
    wins = int((gap > 0).sum())
    n = len(gap)
    lo, hi = wilson_interval(wins, n)
    boot = block_bootstrap_mean_ci(gap, n_boot=n_boot, block=n_tranches, seed=boot_seed)

    # Non-overlapping check: keep every n_tranches-th start (disjoint windows), then
    # average the plain t over the n_tranches possible phases so the answer does not
    # depend on which month you happen to start counting from.
    ts = []
    for phase in range(n_tranches):
        sub = gap[phase::n_tranches]
        if len(sub) >= 5:
            ts.append(one_sample_t(sub))
    t_nonoverlap = float(np.nanmean(ts)) if ts else float("nan")

    return {
        "n_windows": n,
        "win_rate": wins / n,
        "win_ci_low": float(lo), "win_ci_high": float(hi),
        "mean_gap_cents": float(gap.mean()),
        "median_gap_cents": float(np.median(gap)),
        "sd_gap_cents": float(gap.std(ddof=1)),
        "p05_gap_cents": float(np.percentile(gap, 5)),
        "p95_gap_cents": float(np.percentile(gap, 95)),
        "worst_gap_cents": float(gap.min()),
        "best_gap_cents": float(gap.max()),
        "t_hac": newey_west_t(gap, lags=n_tranches),
        "t_nonoverlap": t_nonoverlap,
        "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"],
        "boot_frac_negative": boot["frac_negative"],
        # Dispersion of where each arm LANDS — the risk claim, not the return claim.
        "sd_w_lump": float(exp["w_lump"].std(ddof=1)),
        "sd_w_dca": float(exp["w_dca"].std(ddof=1)),
        "dispersion_ratio": float(exp["w_dca"].std(ddof=1) / exp["w_lump"].std(ddof=1)),
        "worst_w_lump": float(exp["w_lump"].min() - 1.0),
        "worst_w_dca": float(exp["w_dca"].min() - 1.0),
        "mean_w_lump": float(exp["w_lump"].mean() - 1.0),
        "mean_w_dca": float(exp["w_dca"].mean() - 1.0),
    }


def race(asset: pd.Series, cash: pd.Series, n_tranches: int = N_TRANCHES,
         cost_bps: float = 1.0, boot_seed: int = 934, n_boot: int = 4000) -> dict:
    """Run the experiment and summarise it in one call."""
    exp = run_experiment(asset, cash, n_tranches=n_tranches, cost_bps=cost_bps)
    out = summarise(exp, n_tranches=n_tranches, boot_seed=boot_seed, n_boot=n_boot)
    out["exp"] = exp
    return out


# --------------------------------------------------------------------------- #
# What is the gap actually MADE of? (exposure, or timing?)
# --------------------------------------------------------------------------- #
def average_dca_exposure(n_tranches: int = N_TRANCHES) -> float:
    """Time-weighted average asset exposure of the DCA arm — analytic, no fitting.

    Tranche *j* (0-indexed) is invested for ``(n - j) / n`` of the window, so the
    average exposure over the horizon is ``(1/n) * sum_j (n - j)/n = (n + 1) / (2n)``
    — 13/24 = 54.2% for the twelve-tranche schedule. Nothing is estimated from the
    tape: it falls out of the *schedule*, so using it involves no hindsight.
    """
    return (n_tranches + 1) / (2.0 * n_tranches)


def exposure_matched_race(exp: pd.DataFrame, n_tranches: int = N_TRANCHES,
                          weight: float | None = None, n_boot: int = 4000,
                          seed: int = 934) -> dict:
    """Race DCA against a lump sum **sized to DCA's own average exposure**.

    The headline gap says the lump sum finishes richer. It does not say *why*. The
    DCA arm is not only later into the market, it is **less in** the market: its
    time-weighted exposure is :func:`average_dca_exposure` (13/24 for twelve
    tranches), so it is a lower-beta portfolio and must be expected to earn less.

    This function removes that confound the honest way: build a **static** portfolio
    that holds ``weight`` of the asset and ``1 - weight`` in the same cash leg for the
    whole twelve months — same exposure as DCA on average, same terminal date, no
    schedule — and race *it* against DCA. What survives is the part of the gap that is
    about **when** you bought rather than **how much** you owned.

    ``weight`` defaults to the analytic exposure (no hindsight). Passing the
    dispersion-matching weight instead is a fitted, in-sample calibration and must be
    labelled as such wherever it is quoted.

    Also returns each arm's **excess-of-cash** return per unit of dispersion, so the
    two legs are compared on the same risk-adjusted footing (both legs net of the same
    cash return, never one against a raw total).
    """
    if len(exp) == 0:
        return {"n_windows": 0}
    w = average_dca_exposure(n_tranches) if weight is None else float(weight)
    lump_ret = exp["w_lump"].to_numpy(dtype=float) - 1.0
    dca_ret = exp["w_dca"].to_numpy(dtype=float) - 1.0
    cash_ret = exp["cash_ret"].to_numpy(dtype=float)

    matched_ret = w * lump_ret + (1.0 - w) * cash_ret
    gap = (matched_ret - dca_ret) * 100.0
    boot = block_bootstrap_mean_ci(gap, n_boot=n_boot, block=n_tranches, seed=seed)

    ts = []
    for phase in range(n_tranches):
        sub = gap[phase::n_tranches]
        if len(sub) >= 5:
            ts.append(one_sample_t(sub))

    x_lump, x_dca = lump_ret - cash_ret, dca_ret - cash_ret
    def _rd(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd) if sd > 0 else float("nan")

    return {
        "weight": w,
        "n_windows": int(len(gap)),
        "mean_gap_cents": float(gap.mean()),
        "win_rate": float((gap > 0).mean()),
        "t_hac": newey_west_t(gap, lags=n_tranches),
        "t_nonoverlap": float(np.nanmean(ts)) if ts else float("nan"),
        "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"],
        "sd_matched": float(matched_ret.std(ddof=1)),
        "sd_dca": float(dca_ret.std(ddof=1)),
        # Risk-adjusted read, excess of the SAME cash leg on both sides.
        "reward_disp_lump": _rd(x_lump),
        "reward_disp_dca": _rd(x_dca),
        "mean_excess_lump": float(x_lump.mean()),
        "mean_excess_dca": float(x_dca.mean()),
    }


def dispersion_matched_weight(exp: pd.DataFrame) -> float:
    """The **in-sample** weight that equalises the two arms' dispersion (a fitted number).

    Hindsight: it is read off the realised standard deviations of this very sample, so
    it is quoted only as a sensitivity around the analytic exposure weight, never as a
    rule anyone could have set in advance.
    """
    x_lump = (exp["w_lump"] - 1.0) - exp["cash_ret"]
    x_dca = (exp["w_dca"] - 1.0) - exp["cash_ret"]
    return float(x_dca.std(ddof=1) / x_lump.std(ddof=1))


# --------------------------------------------------------------------------- #
# The conditional cuts (does the advice come good when you need it?)
# --------------------------------------------------------------------------- #
def conditional_cuts(exp: pd.DataFrame, dd_threshold: float = -0.10,
                     n_tranches: int = N_TRANCHES) -> dict:
    """Split the start months by starting valuation stretch and by drawdown state.

    ``stretch0`` is a **price-based PROXY** for "the market is expensive": the asset's
    level relative to its own trailing three-year mean, measured at the start date. It
    is not CAPE — no earnings data is used anywhere in this study — so it is labelled
    a proxy wherever it is quoted. Terciles are cut on the *in-sample* distribution of
    starts, which is a hindsight cut: it answers "with the benefit of hindsight, was
    there a state where DCA won?", not "can you use this live". The trailing mean needs a
    three-year runway inside the sample, so the stretch terciles cover fewer starts than
    the drawdown cut — the returned ``n`` per bucket says how many.

    ``dd0`` is the asset's drawdown from its running high at the start date, a
    genuinely observable state.
    """
    out = {}
    e = exp.dropna(subset=["stretch0"])
    if len(e) >= 30:
        q = e["stretch0"].quantile([1 / 3, 2 / 3]).to_numpy()
        labels = ["cheap", "middle", "stretched"]
        bucket = np.where(e["stretch0"] <= q[0], 0, np.where(e["stretch0"] <= q[1], 1, 2))
        for k, lab in enumerate(labels):
            sub = e[bucket == k]
            out[f"stretch_{lab}"] = _cut_row(sub, n_tranches)
    for lab, mask in [("after_drawdown", exp["dd0"] <= dd_threshold),
                      ("near_highs", exp["dd0"] > dd_threshold)]:
        out[lab] = _cut_row(exp[mask], n_tranches)
    return out


def _cut_row(sub: pd.DataFrame, n_tranches: int) -> dict:
    if len(sub) < 5:
        return {"n": int(len(sub))}
    gap = sub["gap_cents"].to_numpy(dtype=float)
    return {
        "n": int(len(sub)),
        "win_rate": float((gap > 0).mean()),
        "mean_gap_cents": float(gap.mean()),
        "median_gap_cents": float(np.median(gap)),
        "t_hac": newey_west_t(gap, lags=min(n_tranches, max(len(gap) - 2, 1))),
        "mean_asset_ret": float(sub["asset_ret"].mean()),
    }


def era_cut(exp: pd.DataFrame, split: str = "2017-01-01",
            n_tranches: int = N_TRANCHES) -> dict:
    """Split the start months at ``split`` and re-read the race on each half."""
    out = {}
    for tag, mask in [("early", exp.index < pd.Timestamp(split)),
                      ("late", exp.index >= pd.Timestamp(split))]:
        out[tag] = _cut_row(exp[mask], n_tranches)
    return out


def cost_sweep(asset: pd.Series, cash: pd.Series,
               cost_grid=(0.0, 1.0, 5.0, 25.0), n_tranches: int = N_TRANCHES) -> list:
    """Gross-to-net sweep on a **proportional** one-way cost (bps of NAV).

    Both arms put the same $1 to work in total, so a proportional cost is charged on
    the same total notional in each and very nearly cancels in the gap. That is a real
    property of this comparison, not a bug — and it is why the honest cost question
    here is the *fixed ticket*, swept in :func:`ticket_sweep`.
    """
    rows = []
    for cbps in cost_grid:
        exp = run_experiment(asset, cash, n_tranches=n_tranches, cost_bps=cbps)
        s = summarise(exp, n_tranches=n_tranches, n_boot=500)
        rows.append({
            "cost_bps": cbps, "win_rate": s["win_rate"],
            "mean_gap_cents": s["mean_gap_cents"], "t_hac": s["t_hac"],
        })
    return rows


def ticket_sweep(asset: pd.Series, cash: pd.Series, portfolio_usd: float = 10_000.0,
                 ticket_grid=(0.0, 1.0, 5.0, 10.0), n_tranches: int = N_TRANCHES) -> list:
    """Sweep a **fixed per-trade ticket charge** — the cost that does *not* cancel.

    ``portfolio_usd`` and the ticket amounts are ASSUMPTIONS, not tape: a $10,000
    windfall and a $0-$10 commission bracket the retail range from a zero-commission
    broker to an old-fashioned one. DCA pays the ticket ``n_tranches`` times, the lump
    sum once, so every dollar of ticket widens the gap by
    ``(n_tranches - 1) x ticket / portfolio``. Reported so the reader can re-price it
    for their own windfall size rather than take ours on faith.
    """
    rows = []
    base = run_experiment(asset, cash, n_tranches=n_tranches, cost_bps=0.0)
    for ticket in ticket_grid:
        extra_cents = 100.0 * (n_tranches - 1) * ticket / portfolio_usd
        gap = base["gap_cents"] + extra_cents
        rows.append({
            "ticket_usd": ticket,
            "portfolio_usd": portfolio_usd,
            "extra_gap_cents": float(extra_cents),
            "win_rate": float((gap > 0).mean()),
            "mean_gap_cents": float(gap.mean()),
            "t_hac": newey_west_t(gap.to_numpy(), lags=n_tranches),
        })
    return rows


def tranche_sweep(asset: pd.Series, cash: pd.Series,
                  grid=(3, 6, 12, 24), cost_bps: float = 1.0) -> list:
    """How the gap scales with how long you take to average in."""
    rows = []
    for k in grid:
        exp = run_experiment(asset, cash, n_tranches=k, cost_bps=cost_bps)
        s = summarise(exp, n_tranches=k, n_boot=500)
        rows.append({
            "n_tranches": k, "n_windows": s["n_windows"], "win_rate": s["win_rate"],
            "mean_gap_cents": s["mean_gap_cents"], "t_hac": s["t_hac"],
            "dispersion_ratio": s["dispersion_ratio"],
        })
    return rows


def zero_cash_variant(asset: pd.Series, cash: pd.Series,
                      n_tranches: int = N_TRANCHES, cost_bps: float = 1.0) -> dict:
    """The folk version of the experiment: uninvested cash earns **0%** (an ASSUMPTION).

    A flat cash index is what the popular write-ups (and this desk's own Study 101)
    assume. It *penalises* DCA, because it denies the waiting tranches the T-bill
    yield they would really have earned. Re-running the race against it isolates how
    much of the headline gap is the equity premium the cash misses and how much is
    simply the short-rate path of 2007-2026 flowing back to DCA.
    """
    flat = pd.Series(1.0, index=cash.index)
    exp = run_experiment(asset, flat, n_tranches=n_tranches, cost_bps=cost_bps)
    return summarise(exp, n_tranches=n_tranches, n_boot=500)


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, n_tranches: int = N_TRANCHES,
                     cost_bps: float = 1.0, n_boot: int = 500) -> dict:
    """Run the race on a synthetic ``(asset, cash)`` tape.

    With a fat planted excess drift the lump sum must win (positive mean gap, win rate
    well above 50%); at ``signal_strength=0`` the gap must sit on zero and the win
    rate on one half; with a negative drift DCA must win. Proves the harness has no
    thumb on the scale — it never supports a real-tape stamp.
    """
    s = race(prices["asset"], prices["cash"], n_tranches=n_tranches,
             cost_bps=cost_bps, n_boot=n_boot)
    s.pop("exp", None)
    return s


def synthetic_control(signal_strength: float, seeds=range(934, 946),
                      n_tranches: int = N_TRANCHES, cost_bps: float = 1.0,
                      synth=None) -> dict:
    """Monte-Carlo version of :func:`synthetic_detect` — average over ``seeds``.

    One 25-year path of a 16%-vol asset is a *very* noisy estimate of its own drift
    (the standard error of the realised annual drift is ~3 pp), so a single seed can
    and does land the wrong side of a 7 pp planted premium. Averaging over a dozen
    independent paths is what makes the control read the *planted* world rather than
    one draw of it. ``synth`` defaults to ``data.synthetic_daily``.
    """
    if synth is None:
        from . import data as _data
        synth = _data.synthetic_daily
    gaps, wins = [], []
    for sd in seeds:
        prices, _ = synth(signal_strength=signal_strength, seed=int(sd))
        d = synthetic_detect(prices, n_tranches=n_tranches, cost_bps=cost_bps, n_boot=200)
        gaps.append(d["mean_gap_cents"])
        wins.append(d["win_rate"])
    gaps = np.asarray(gaps, dtype=float)
    wins = np.asarray(wins, dtype=float)
    return {
        "signal_strength": signal_strength,
        "n_seeds": int(len(gaps)),
        "mean_gap_cents": float(gaps.mean()),
        "sd_gap_cents": float(gaps.std(ddof=1)),
        "se_gap_cents": float(gaps.std(ddof=1) / np.sqrt(len(gaps))),
        "mean_win_rate": float(wins.mean()),
        "frac_seeds_lump_wins": float((gaps > 0).mean()),
    }
