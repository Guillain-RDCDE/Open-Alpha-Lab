"""Strategy + inference for Study 617 — Crash-Insurance-Cost.

The claim under test: **TAIL and put-buying strategies bleed money every year — and
that bleed is the volatility risk premium seen from the buyer's side.** The desk has
already tested the SELLER of this premium (studies 92 and 63, both Real); here we sit on
the other side of the trade, on a live buyable product, and measure what the insurance
*costs*.

Four complementary measurements:

    * **Raw drift.** TAIL total-return CAGR since inception and the HAC/Newey-West t of
      its daily mean return. The claim predicts a negative drift.
    * **Decomposition vs its own design.** TAIL is ~90% intermediate Treasuries + an OTM
      SPX put ladder, so regress TAIL daily returns on IEF (its collateral proxy):
      ``tail = alpha + beta*ief``. The **alpha** is the put sleeve's net drag — the
      insurance premium actually paid, isolated from bond duration. HAC t on alpha.
    * **The 2020 payoff, quantified.** The one episode where the insurance paid:
      TAIL's gain from the SPY peak (2020-02-19) to TAIL's own COVID high, SPY's fall
      over the same window, how long the jackpot took to bleed away, and what an
      at-inception holder had at the COVID peak vs today.
    * **The premium named — a model put-cost series from real ^VIX + SPY.** Variance-
      notional arithmetic: each month the insurance buyer pays implied variance
      ``IV = (VIX_(m-1 end)/100)^2 / 12`` (known at month-end, one clean one-month lag)
      and receives realized variance ``RV = sum of daily squared SPY log returns over
      month m``. ``mean(RV - IV)`` < 0 is the variance risk premium *paid by the buyer*
      — the same premium studies 92/63 harvested from the short side. This is a
      **model-derived** series (variance-swap arithmetic, not an option backtest) built
      on real tapes; it names the mechanism, the TAIL alpha is the headline evidence.

Inference: HAC/Newey-West t on every serially-correlated mean (daily fund returns,
monthly variance-premium series). Execution conventions, documented: TAIL buy-and-hold
is a STATIC position — there is no timing signal, hence no signal lag to apply; the
variance-premium series uses the prior month-end VIX against the following month's
realized variance (exactly one month of lag, no look-ahead). TAIL's 0.59%/yr expense
ratio is inside its NAV (all TAIL numbers are net of it, labeled); the hedged-portfolio
overlay charges explicit one-way rebalancing costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TDAYS = 252.0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def nw_ols(y: np.ndarray, X: np.ndarray | None = None, lags: int | None = None) -> dict:
    """OLS with Newey-West (HAC) standard errors. X excludes the constant (added here).

    ``X = None`` runs the constant-only regression — the HAC t of the mean. ``lags``
    defaults to the standard ``floor(4 (n/100)^(2/9))`` plug-in.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    if X is None:
        Z = np.ones((n, 1))
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Z = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    u = y - Z @ beta
    Zu = Z * u[:, None]
    S = Zu.T @ Zu
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Zu[L:].T @ Zu[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return {"beta": beta, "t": t, "se": se, "n": n, "lags": lags,
            "resid": u}


def hac_mean(x: np.ndarray) -> dict:
    """Mean of a serially-correlated series with its HAC/Newey-West t."""
    r = nw_ols(np.asarray(x, dtype=float))
    return {"mean": float(r["beta"][0]), "t": float(r["t"][0]),
            "n": r["n"], "lags": r["lags"]}


# --------------------------------------------------------------------------- #
# Raw drift — what holding the insurance costs
# --------------------------------------------------------------------------- #
def to_returns(tape: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns of tail/ief/spy on their joint TAIL-era sample."""
    px = tape[["tail", "ief", "spy"]].dropna()
    return px.pct_change().dropna()


def drift_stats(tape: pd.DataFrame) -> dict:
    """TAIL's total-return record since inception: CAGR, total, vol, HAC t of the drift."""
    px = tape["tail"].dropna()
    ret = px.pct_change().dropna()
    years = (px.index[-1] - px.index[0]).days / 365.25
    total = float(px.iloc[-1] / px.iloc[0] - 1.0)
    cagr = float((px.iloc[-1] / px.iloc[0]) ** (1.0 / years) - 1.0)
    h = hac_mean(ret.values)
    m = px.resample("ME").last().pct_change().dropna()
    hm = hac_mean(m.values)
    return {
        "start": str(px.index[0].date()), "end": str(px.index[-1].date()),
        "years": years, "total_pct": total * 100, "cagr_pct": cagr * 100,
        "ann_vol_pct": float(ret.std(ddof=1)) * np.sqrt(TDAYS) * 100,
        "daily_mean_bps": h["mean"] * 1e4, "daily_t": h["t"],
        "n_days": h["n"], "nw_lags": h["lags"],
        "monthly_mean_bps": hm["mean"] * 1e4, "monthly_t": hm["t"], "n_months": hm["n"],
    }


# --------------------------------------------------------------------------- #
# Decomposition — the put sleeve's drag, isolated from bond duration
# --------------------------------------------------------------------------- #
def decompose_vs_ief(tape: pd.DataFrame) -> dict:
    """Regress TAIL daily returns on IEF (its collateral proxy), HAC/NW t on alpha.

    ``alpha`` is the net daily drag of everything TAIL does that IEF doesn't — i.e. the
    put ladder plus fees: the price actually paid for the crash insurance.
    """
    rets = to_returns(tape)
    r = nw_ols(rets["tail"].values, rets["ief"].values)
    alpha_d, beta = float(r["beta"][0]), float(r["beta"][1])
    y = rets["tail"].values
    r2 = 1.0 - float(r["resid"] @ r["resid"]) / float(((y - y.mean()) ** 2).sum())
    return {
        "alpha_daily_bps": alpha_d * 1e4, "alpha_ann_pct": alpha_d * TDAYS * 100,
        "t_alpha": float(r["t"][0]), "beta_ief": beta, "t_beta": float(r["t"][1]),
        "r2": r2, "n": r["n"], "lags": r["lags"],
    }


def decompose_monthly(tape: pd.DataFrame) -> dict:
    """Same regression at MONTHLY frequency (complete months only — the partial
    inception month is dropped). Monthly compounding soaks up the microstructure
    noise that dilutes the daily t; this is the headline decomposition stat."""
    rets = to_returns(tape)
    m = (1.0 + rets).resample("ME").prod() - 1.0
    m = m.iloc[1:]                                           # drop partial inception month
    r = nw_ols(m["tail"].values, m["ief"].values)
    alpha_m, beta = float(r["beta"][0]), float(r["beta"][1])
    return {
        "alpha_monthly_bps": alpha_m * 1e4, "alpha_ann_pct": alpha_m * 12.0 * 100,
        "t_alpha": float(r["t"][0]), "beta_ief": beta, "t_beta": float(r["t"][1]),
        "n_months": r["n"], "lags": r["lags"],
    }


def alpha_subperiods(tape: pd.DataFrame, pre_end: str = "2020-02-19",
                     post_start: str = "2020-07-01") -> dict:
    """Daily alpha-vs-IEF on the pre-COVID and post-COVID sub-samples (the crash
    itself excluded, so neither leg owes its sign to the jackpot)."""
    rets = to_returns(tape)
    out = {}
    for lab, sl in (("pre", rets.loc[:pre_end]), ("post", rets.loc[post_start:])):
        r = nw_ols(sl["tail"].values, sl["ief"].values)
        out[lab] = {"alpha_ann_pct": float(r["beta"][0]) * TDAYS * 100,
                    "t_alpha": float(r["t"][0]), "n": r["n"]}
    return out


# --------------------------------------------------------------------------- #
# The 2020 payoff — the one time the insurance paid (and what happened next)
# --------------------------------------------------------------------------- #
def covid_episode(tape: pd.DataFrame, peak: str = "2020-02-19",
                  trough: str = "2020-03-23") -> dict:
    """Quantify the COVID jackpot and how long holders kept it.

    From SPY's pre-crash peak we find TAIL's own COVID high, the gains/losses over the
    crash window, the date TAIL first closed back BELOW its pre-crash (2020-02-19) level
    (the jackpot fully bled away), and the at-inception holder's wealth at the COVID
    peak vs at the end of tape.
    """
    px = tape[["tail", "spy"]].dropna()
    p0 = px.loc[:peak].iloc[-1]
    window = px.loc[peak:"2020-06-30"]
    tail_peak_date = window["tail"].idxmax()
    tail_peak = float(window["tail"].max())
    tail_gain = tail_peak / float(p0["tail"]) - 1.0
    spy_at_trough = float(px.loc[:trough, "spy"].iloc[-1])
    spy_fall = spy_at_trough / float(p0["spy"]) - 1.0
    after = px.loc[tail_peak_date:, "tail"]
    below = after[after < float(p0["tail"])]
    gone_date = str(below.index[0].date()) if len(below) else None
    days_to_give_back = (below.index[0] - tail_peak_date).days if len(below) else None
    inception = px["tail"].iloc[0]
    holder_at_peak = tail_peak / float(inception) - 1.0
    holder_now = float(px["tail"].iloc[-1]) / float(inception) - 1.0
    return {
        "tail_gain_pct": tail_gain * 100, "tail_peak_date": str(tail_peak_date.date()),
        "spy_fall_pct": spy_fall * 100,
        "giveback_date": gone_date, "giveback_days": days_to_give_back,
        "holder_at_peak_pct": holder_at_peak * 100, "holder_now_pct": holder_now * 100,
    }


# --------------------------------------------------------------------------- #
# The premium named — variance-notional put-cost arithmetic from real ^VIX + SPY
# --------------------------------------------------------------------------- #
def variance_premium(tape: pd.DataFrame, start: str = "1993-02-01") -> dict:
    """Monthly buyer's P&L in variance points: realized minus implied variance.

    IV for month m is the prior month-end VIX squared, de-annualized ((VIX/100)^2 / 12)
    — known before the month starts, one clean one-month lag. RV is the sum of daily
    squared SPY log returns within month m. ``mean(RV - IV) < 0`` = the buyer of
    variance systematically pays more than variance delivers: the volatility risk
    premium, buyer's side. Model arithmetic (variance-swap payoff) on real tapes.
    """
    vix = tape["vix"].dropna()
    spy = tape["spy"].dropna()
    lr = np.log(spy / spy.shift(1)).dropna()
    rv = (lr ** 2).groupby(lr.index.to_period("M")).sum()
    iv_next = ((vix.resample("ME").last() / 100.0) ** 2 / 12.0)
    iv_next.index = iv_next.index.to_period("M") + 1        # IV quotes FOR the next month
    df = pd.DataFrame({"rv": rv, "iv": iv_next}).dropna()
    df = df[df.index >= pd.Period(start, freq="M")]
    pnl = (df["rv"] - df["iv"]).values                       # buyer receives RV, pays IV
    h = hac_mean(pnl)
    vol_pts = (np.sqrt(df["rv"] * 12.0) - np.sqrt(df["iv"] * 12.0)).values * 100
    hv = hac_mean(vol_pts)
    return {
        "start": str(df.index[0]), "end": str(df.index[-1]), "n_months": len(df),
        "mean_var_ann": h["mean"] * 12.0, "t_var": h["t"], "lags": h["lags"],
        "share_buyer_wins": float((pnl > 0).mean() * 100),
        "mean_vol_pts": hv["mean"], "t_vol": hv["t"],
        "iv_ann_vol": float(np.sqrt(df["iv"].mean() * 12.0)) * 100,
        "rv_ann_vol": float(np.sqrt(df["rv"].mean() * 12.0)) * 100,
    }


# --------------------------------------------------------------------------- #
# Cohorts — has ANY buy-and-hold TAIL buyer ever come out ahead?
# --------------------------------------------------------------------------- #
def cohort_table(tape: pd.DataFrame) -> dict:
    """Every month-end entry cohort of TAIL, held to the end of tape (total-return).

    For each complete month-end m in TAIL's life: the buy-at-m-hold-to-asof total
    return, plus whether that cohort was EVER ahead at any month-end along the way
    (the COVID spike counts). The third-axis question: net of the 2020 win, did any
    cohort come out ahead — and did anyone who held keep the win?
    """
    m = tape["tail"].dropna().resample("ME").last().dropna()
    final = float(m.iloc[-1])
    entries = m.iloc[:-1]
    tot = final / entries - 1.0
    running_max_fwd = m[::-1].cummax()[::-1]                 # max future month-end level
    ever_ahead = (running_max_fwd.reindex(entries.index) > entries)
    return {
        "n_cohorts": len(entries),
        "share_ahead_now": float((tot > 0).mean() * 100),
        "n_ahead_now": int((tot > 0).sum()),
        "best_pct": float(tot.max() * 100), "best_date": str(tot.idxmax().date()),
        "worst_pct": float(tot.min() * 100), "median_pct": float(tot.median() * 100),
        "share_ever_ahead": float(ever_ahead.mean() * 100),
        "cohort_returns": tot,
    }


# --------------------------------------------------------------------------- #
# Tradability — insurance inside a portfolio: 90/10 SPY/TAIL vs 100% SPY
# --------------------------------------------------------------------------- #
def hedged_portfolio(tape: pd.DataFrame, w_tail: float = 0.10,
                     cost_bps: float = 5.0) -> dict:
    """Monthly-rebalanced (1-w) SPY + w TAIL vs 100% SPY over TAIL's life.

    One-way ``cost_bps`` x turnover charged at each monthly rebalance (the drift back to
    target weights); TAIL's 0.59%/yr expense ratio is already inside its NAV. Reports
    CAGR, max drawdown and the COVID drawdown for both — what the insurance actually
    bought, and what it cost.
    """
    rets = to_returns(tape)[["spy", "tail"]]
    my = rets.copy()
    my["month"] = my.index.to_period("M")
    w = np.array([1.0 - w_tail, w_tail])
    c = cost_bps / 1e4
    port_curve, spy_curve = [1.0], [1.0]
    dates = [rets.index[0]]
    for _, g in my.groupby("month"):
        gr = (1.0 + g[["spy", "tail"]]).prod() - 1.0
        wealth = w * (1.0 + gr[["spy", "tail"]].values)
        gross = wealth.sum()
        drift = np.abs(wealth / gross - w).sum()             # turnover back to target
        net = gross * (1.0 - c * drift)
        port_curve.append(port_curve[-1] * net)
        spy_curve.append(spy_curve[-1] * (1.0 + float(gr["spy"])))
        dates.append(g.index[-1])
    port = pd.Series(port_curve, index=pd.DatetimeIndex(dates))
    spyc = pd.Series(spy_curve, index=pd.DatetimeIndex(dates))
    years = (port.index[-1] - port.index[0]).days / 365.25

    def _cagr(s): return float(s.iloc[-1] ** (1.0 / years) - 1.0) * 100

    def _mdd(s): return float((s / s.cummax() - 1.0).min()) * 100

    def _covid_dd(s):
        w20 = s.loc["2020-01-31":"2020-04-30"]
        return float(w20.min() / s.loc[:"2020-02-19"].iloc[-1] - 1.0) * 100

    return {
        "w_tail": w_tail, "cost_bps": cost_bps, "years": years,
        "port_cagr_pct": _cagr(port), "spy_cagr_pct": _cagr(spyc),
        "port_mdd_pct": _mdd(port), "spy_mdd_pct": _mdd(spyc),
        "port_covid_dd_pct": _covid_dd(port), "spy_covid_dd_pct": _covid_dd(spyc),
        "cagr_drag_pct": _cagr(spyc) - _cagr(port),
        "port_curve": port, "spy_curve": spyc,
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control — multi-seed, never market evidence
# --------------------------------------------------------------------------- #
def control_summary(synthetic_world, bleeds=(0.0, 0.05), n_seeds: int = 20,
                    seed0: int = 617) -> list[dict]:
    """Run the alpha-vs-bonds decomposition on the synthetic world across
    ``n_seeds`` seeds per planted ``bleed`` (>= 20 seeds — single-seed baselines
    are banned). ``bleed = 0`` is the fair-insurance null: the detector must stay
    quiet (~5% false positives at |t|>=2). A planted bleed must be recovered in
    size and lit up in t. A machinery proof only — never cited for the stamp."""
    rows = []
    for bleed in bleeds:
        a, t = [], []
        for s in range(seed0, seed0 + n_seeds):
            d = decompose_vs_ief(synthetic_world(bleed=bleed, seed=s))
            a.append(d["alpha_ann_pct"])
            t.append(d["t_alpha"])
        a, t = np.array(a), np.array(t)
        hit = float((t <= -2.0).mean()) if bleed > 0 else float((np.abs(t) >= 2.0).mean())
        rows.append({"bleed_pct": bleed * 100, "mean_alpha_ann_pct": float(a.mean()),
                     "mean_t": float(t.mean()), "share_flagged": hit * 100,
                     "n_seeds": n_seeds})
    return rows
