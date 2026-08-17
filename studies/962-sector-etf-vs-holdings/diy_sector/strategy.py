"""Strategy + inference for Study 962 — Do It Yourself.

The rule under test is the one every retail forum eventually proposes: *a sector ETF is
just a handful of mega-caps in a wrapper, so buy the top N names yourself, skip the
expense ratio, and keep the difference.*

We build that basket honestly:

- **Weights.** Either the fund's published **cap-ish weights as of 2026-06-30**
  (renormalised inside the basket — the deliberate, named **look-ahead** basket), or the
  fund's **January-2011 top-10 held equal weight, fixed from the first day of the
  sample** (the contemporaneous control for that look-ahead). ``N`` ∈ {3, 5, 10}.
- **One execution lag.** Target weights are read at a month-end close ``t`` and traded at
  ``t+1``; the basket drifts with the tape in between. That is the study's single lag.
- **Costs.** ``cost_bps`` one-way × NAV, charged on the traded turnover
  ``sum |w_target − w_drifted|`` on each rebalance day, plus a full one-way charge to
  build the basket on day one. Long-only throughout — **no short leg, so no borrow cost
  applies**; the cost sweep is over the one-way rate instead.
- **The fund side pays nothing extra.** Its published total-return close already carries
  its expense ratio, so the race is fee-inclusive on the fund side and fee-free (but
  commission-paying) on the DIY side. That asymmetry is the whole point of the claim.

The headline is a difference, not a level: the annualised return gap (DIY − fund), its
**annualised tracking error**, an HAC (Newey-West) *t* on the mean daily gap, a
block-bootstrap CI on the annualised gap, an era cut, a cost sweep, and the
excess-of-cash Sharpe race (gross and net) that says whether the gap is *paid for* in
risk. Concentration risk is measured directly: the worst rolling 12-month shortfall of
the basket against the fund it is meant to replace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
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
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def hac_t(x) -> float:
    """HAC *t* with the standard automatic Bartlett bandwidth."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    return newey_west_t(a, lags=auto_lags(len(a)))


# --------------------------------------------------------------------------- #
# The do-it-yourself basket
# --------------------------------------------------------------------------- #
def rebalance_flags(index: pd.DatetimeIndex, freq: str = "M") -> np.ndarray:
    """Boolean array: True on days the basket is *traded* (the day after a period end).

    ``freq`` is ``"M"`` (month), ``"Q"`` (quarter), ``"A"`` (year) or ``"N"`` (never —
    buy once and let it drift). The period end is read at its close (day ``t``) and the
    trade happens on the next trading day (``t+1``) — the study's single execution lag.
    """
    n = len(index)
    flags = np.zeros(n, dtype=bool)
    if freq == "N" or n < 2:
        return flags
    if freq == "M":
        key = index.year * 12 + index.month
    elif freq == "Q":
        key = index.year * 4 + index.quarter
    elif freq == "A":
        key = index.year.to_numpy()
    else:
        raise ValueError(f"unknown rebalance frequency {freq!r}")
    key = np.asarray(key)
    flags[1:] = key[1:] != key[:-1]     # first day of a new period = the day after t
    return flags


def basket_returns(
    closes: pd.DataFrame,
    weights: pd.Series,
    cost_bps: float = 5.0,
    freq: str = "M",
) -> pd.DataFrame:
    """Daily net simple returns of a fixed-target-weight basket of single names.

    Parameters
    ----------
    closes:
        Daily **total-return** close levels, one column per name (NaN before a name
        lists, or after its tape ends).
    weights:
        Target weights, indexed by name, summing to 1. On every rebalance they are
        renormalised over the names that are *live* that day, so a name that has not
        listed yet (PSX in 2011) simply does not enter until it exists.
    cost_bps:
        One-way cost in bps × NAV, charged on traded turnover. A full one-way charge is
        also paid on day one to build the basket.
    freq:
        Rebalance frequency, see :func:`rebalance_flags`.

    Returns a frame with ``r_basket`` (net), ``r_basket_gross``, ``turnover`` and
    ``n_live``, indexed by date.
    """
    names = [c for c in weights.index if c in closes.columns]
    if not names:
        raise ValueError("none of the basket's names are present in the price frame")
    sub = closes[names].astype(float)
    w_target = weights.reindex(names).astype(float).to_numpy()

    px = sub.to_numpy()
    n_days, n_names = px.shape
    prev = np.vstack([np.full((1, n_names), np.nan), px[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = px / prev - 1.0
    tradable = np.isfinite(px)                 # the name has a price today
    earning = np.isfinite(rets)                # ... and had one yesterday
    # A held name whose tape is dead earns 0% that day until the next rebalance drops it.
    r_clean = np.where(earning, np.nan_to_num(rets), 0.0)
    any_live = tradable.any(axis=1)

    flags = rebalance_flags(sub.index, freq=freq)
    cost = cost_bps * 1e-4

    h = np.zeros(n_names)
    started = False
    pending_turn = 0.0
    out_net = np.full(n_days, np.nan)
    out_gross = np.full(n_days, np.nan)
    out_turn = np.zeros(n_days)
    out_live = np.zeros(n_days, dtype=int)

    for i in range(n_days):
        if not any_live[i]:
            continue
        live = tradable[i]
        if not started:
            # Build the basket on the first day any name is tradable. No return is
            # earned on the build day (it is dropped); the one-way charge on the whole
            # notional is carried into the first day the basket is actually invested,
            # so the gross and net legs stay aligned to the same days as the fund.
            tgt = np.where(live, w_target, 0.0)
            s = tgt.sum()
            if s <= 0:
                continue
            h = tgt / s
            pending_turn = 1.0
            started = True
            continue

        turn = 0.0
        if flags[i]:
            tgt = np.where(live, w_target, 0.0)
            s = tgt.sum()
            if s > 0:
                tgt = tgt / s
                turn = float(np.abs(tgt - h).sum())
                h = tgt

        r = r_clean[i]
        gross = float(h @ r)
        h = h * (1.0 + r)
        tot = h.sum()
        if tot > 0:
            h = h / tot
        turn += pending_turn
        pending_turn = 0.0
        out_gross[i] = gross
        out_net[i] = gross - cost * turn
        out_turn[i] = turn
        out_live[i] = int(live.sum())

    return pd.DataFrame(
        {"r_basket": out_net, "r_basket_gross": out_gross,
         "turnover": out_turn, "n_live": out_live},
        index=sub.index,
    ).dropna(subset=["r_basket"])


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an *excess* series for
    the excess-of-cash Sharpe)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": n, "cagr": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "max_drawdown": float("nan"),
                "mean_daily_bps": float("nan"), "tstat": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": hac_t(r),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def worst_rolling_shortfall(diff: pd.Series, window: int = TRADING_DAYS) -> float:
    """Worst rolling ``window``-day *compounded* shortfall of the difference series.

    This is the concentration-risk number that matters to someone who replaced a fund
    with five names: not the average gap, but the worst year of being wrong.
    """
    d = pd.Series(diff).astype(float).dropna()
    if len(d) < window + 1:
        return float("nan")
    log_d = np.log1p(d.clip(lower=-0.99))
    roll = log_d.rolling(window).sum()
    return float(np.expm1(roll.min()))


# --------------------------------------------------------------------------- #
# The race: DIY basket vs the fund it replaces
# --------------------------------------------------------------------------- #
def race(
    closes: pd.DataFrame,
    fund: str,
    weights: pd.Series,
    cash: pd.Series,
    cost_bps: float = 5.0,
    freq: str = "M",
) -> dict:
    """Race a DIY basket against its fund, both excess-of-cash, gross and net.

    ``closes`` must contain the fund column and the basket's names; ``cash`` is the
    daily total-return close of the cash leg (BIL). Returns per-arm summaries, the
    return gap and its inference, the tracking error, and the difference series.
    """
    bt = basket_returns(closes, weights, cost_bps=cost_bps, freq=freq)
    idx = bt.index.intersection(closes[fund].dropna().index).intersection(cash.dropna().index)
    bt = bt.loc[idx]
    r_fund = closes[fund].reindex(idx).pct_change()
    r_cash = cash.reindex(idx).pct_change()

    df = pd.concat(
        [bt["r_basket"].rename("r_diy"), bt["r_basket_gross"].rename("r_diy_gross"),
         r_fund.rename("r_fund"), r_cash.rename("r_cash"), bt["turnover"]],
        axis=1,
    ).dropna()

    e_diy = (df["r_diy"] - df["r_cash"]).rename("e_diy")
    e_diy_g = (df["r_diy_gross"] - df["r_cash"]).rename("e_diy_gross")
    e_fund = (df["r_fund"] - df["r_cash"]).rename("e_fund")
    diff = (df["r_diy"] - df["r_fund"]).rename("diff")
    diff_g = (df["r_diy_gross"] - df["r_fund"]).rename("diff_gross")

    s_diy, s_diy_g, s_fund = summary(e_diy), summary(e_diy_g), summary(e_fund)
    n = len(diff)
    te = float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann_gap = float((1.0 + diff.mean()) ** TRADING_DAYS - 1.0)
    ann_gap_g = float((1.0 + diff_g.mean()) ** TRADING_DAYS - 1.0)

    return {
        "fund": fund,
        "n_days": n,
        "start": str(df.index[0].date()) if n else None,
        "end": str(df.index[-1].date()) if n else None,
        "diy": s_diy, "diy_gross": s_diy_g, "fund_stats": s_fund,
        "sharpe_adv": s_diy["sharpe"] - s_fund["sharpe"],
        "sharpe_adv_gross": s_diy_g["sharpe"] - s_fund["sharpe"],
        "ann_gap": ann_gap,
        "ann_gap_gross": ann_gap_g,
        "mean_diff_bps": float(diff.mean() * 1e4),
        "t_diff": hac_t(diff),
        "t_diff_gross": hac_t(diff_g),
        "t_sharpe_race": hac_t(e_diy - e_fund),
        "tracking_error": te,
        "corr": float(df["r_diy"].corr(df["r_fund"])),
        "beta": float(np.polyfit(df["r_fund"], df["r_diy"], 1)[0]) if n > 3 else float("nan"),
        "worst_12m_shortfall": worst_rolling_shortfall(diff),
        "dd_diy": max_drawdown(df["r_diy"]),
        "dd_fund": max_drawdown(df["r_fund"]),
        "turnover_ann": float(df["turnover"].sum() / (n / TRADING_DAYS)),
        "diff": diff, "diff_gross": diff_g,
        "e_diy": e_diy, "e_fund": e_fund, "frame": df,
    }


# --------------------------------------------------------------------------- #
# Block bootstrap on the annualised gap
# --------------------------------------------------------------------------- #
def bootstrap_gap_ci(
    diff: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 962,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the **annualised** mean of a daily gap series.

    Blocks of ``block`` consecutive days preserve the volatility clustering that makes a
    naive i.i.d. interval far too tight on a tracking-difference series.
    """
    r = np.asarray(pd.Series(diff).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"ann_gap": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_negative": float("nan"), "n_obs": n}
    point = float((1.0 + r.mean()) ** TRADING_DAYS - 1.0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = (1.0 + r[idx].mean()) ** TRADING_DAYS - 1.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ann_gap": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": n,
            "n_boot": int(n_boot), "block": block}


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def depth_sweep(
    closes: pd.DataFrame,
    fund: str,
    cash: pd.Series,
    weight_fn,
    depths=(3, 5, 10),
    schemes=("cap2026", "eq2011"),
    cost_bps: float = 5.0,
    freq: str = "M",
) -> list[dict]:
    """Run the race at several basket depths and under both weighting schemes.

    ``weight_fn(fund, n, scheme)`` returns the target weights (``data.basket_weights``).
    """
    rows = []
    for scheme in schemes:
        for n in depths:
            w = weight_fn(fund, n, scheme)
            res = race(closes, fund, w, cash, cost_bps=cost_bps, freq=freq)
            rows.append({
                "fund": fund, "scheme": scheme, "n": n,
                "n_days": res["n_days"], "start": res["start"],
                "ann_gap": res["ann_gap"], "ann_gap_gross": res["ann_gap_gross"],
                "t_diff": res["t_diff"], "tracking_error": res["tracking_error"],
                "sharpe_diy": res["diy"]["sharpe"], "sharpe_fund": res["fund_stats"]["sharpe"],
                "sharpe_adv": res["sharpe_adv"], "t_sharpe_race": res["t_sharpe_race"],
                "worst_12m_shortfall": res["worst_12m_shortfall"],
                "dd_diy": res["dd_diy"], "dd_fund": res["dd_fund"],
                "turnover_ann": res["turnover_ann"], "corr": res["corr"],
            })
    return rows


def cost_sweep(
    closes: pd.DataFrame,
    fund: str,
    cash: pd.Series,
    weights: pd.Series,
    cost_grid=(0.0, 1.0, 5.0, 10.0, 25.0, 50.0),
    freq: str = "M",
) -> list[dict]:
    """Sweep the one-way cost rate. No short leg here, so there is no borrow to sweep."""
    rows = []
    for c in cost_grid:
        res = race(closes, fund, weights, cash, cost_bps=c, freq=freq)
        rows.append({"cost_bps": c, "ann_gap": res["ann_gap"], "t_diff": res["t_diff"],
                     "sharpe_adv": res["sharpe_adv"],
                     "tracking_error": res["tracking_error"],
                     "turnover_ann": res["turnover_ann"]})
    return rows


def freq_sweep(
    closes: pd.DataFrame,
    fund: str,
    cash: pd.Series,
    weights: pd.Series,
    freqs=("M", "Q", "A", "N"),
    cost_bps: float = 5.0,
) -> list[dict]:
    """Sweep the rebalance frequency, including never rebalancing at all (``"N"``)."""
    rows = []
    for f in freqs:
        res = race(closes, fund, weights, cash, cost_bps=cost_bps, freq=f)
        rows.append({"freq": f, "ann_gap": res["ann_gap"], "t_diff": res["t_diff"],
                     "tracking_error": res["tracking_error"],
                     "turnover_ann": res["turnover_ann"],
                     "worst_12m_shortfall": res["worst_12m_shortfall"]})
    return rows


def era_cut(
    closes: pd.DataFrame,
    fund: str,
    cash: pd.Series,
    weights: pd.Series,
    split: str = "2019-01-01",
    cost_bps: float = 5.0,
    freq: str = "M",
) -> dict:
    """Split the sample at ``split`` and re-run the race on each half."""
    out: dict[str, dict | None] = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = closes.loc[sl]
        c = cash.loc[sl]
        if len(sub) < 300:
            out[tag] = None
            continue
        res = race(sub, fund, weights, c, cost_bps=cost_bps, freq=freq)
        out[tag] = {
            "n_days": res["n_days"], "start": res["start"], "end": res["end"],
            "ann_gap": res["ann_gap"], "t_diff": res["t_diff"],
            "tracking_error": res["tracking_error"],
            "sharpe_diy": res["diy"]["sharpe"], "sharpe_fund": res["fund_stats"]["sharpe"],
            "sharpe_adv": res["sharpe_adv"],
            "worst_12m_shortfall": res["worst_12m_shortfall"],
        }
    return out


def calendar_year_table(closes: pd.DataFrame, fund: str, cash: pd.Series,
                        weights: pd.Series, cost_bps: float = 5.0,
                        freq: str = "M") -> pd.DataFrame:
    """Per-calendar-year total return (%) of the DIY basket vs its fund, and the gap."""
    res = race(closes, fund, weights, cash, cost_bps=cost_bps, freq=freq)
    df = res["frame"]
    rows = []
    for y in sorted(set(df.index.year)):
        m = df.index.year == y
        diy = float((1.0 + df["r_diy"][m]).prod() - 1.0)
        fnd = float((1.0 + df["r_fund"][m]).prod() - 1.0)
        rows.append({"year": int(y), "diy_pct": diy * 100, "fund_pct": fnd * 100,
                     "gap_pp": (diy - fnd) * 100})
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, cost_bps: float = 0.0,
                     freq: str = "M") -> dict:
    """Run the DIY-vs-fund race on a synthetic panel with a planted fee.

    On a panel where the fund charges ``fee_ann`` the basket of big names should recover
    approximately that fee (positive annualised gap, HAC *t* comfortably above 2 at zero
    cost); on the zero-fee null the gap must be indistinguishable from zero. Proves the
    detector is unbiased — it never supports a real-tape stamp.
    """
    res = race(prices, "fund", truth["weights"], prices["cash"],
               cost_bps=cost_bps, freq=freq)
    return {
        "ann_gap": res["ann_gap"],
        "ann_gap_gross": res["ann_gap_gross"],
        "t_diff": res["t_diff"],
        "tracking_error": res["tracking_error"],
        "planted_fee_ann": truth["planted_fee_ann"],
        "recovery_ratio": (res["ann_gap_gross"] / truth["planted_fee_ann"]
                           if truth["planted_fee_ann"] > 0 else float("nan")),
        "n_days": res["n_days"],
        "corr": res["corr"],
    }
