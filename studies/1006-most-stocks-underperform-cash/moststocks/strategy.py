"""Why most stocks lose while the index wins — Study 1006.

The two facts are not in tension; they are the same fact seen from two ends.

Compound a return series and you multiply. Multiplying a distribution turns it right-skewed even
if the single-period returns are symmetric, because a sequence of good years multiplies up
without bound while a sequence of bad ones is floored at −100%. The **mean** terminal wealth
tracks the arithmetic average return; the **median** tracks the geometric one, which is lower by
roughly half the variance. The gap between them grows with horizon and with volatility, and it
grows without anything unusual happening.

That is ``variance_drag``, and it is the whole engine. Everything else measures its
consequences:

- ``outcome_distribution`` computes what fraction of names beat cash over each horizon, and how
  the survivors' returns are distributed.
- ``wealth_concentration`` asks how few names produced the aggregate gain — the Bessembinder
  headline, on this basket.
- ``horizon_sweep`` shows the share beating cash *falling* with horizon even though the expected
  return per name rises with it. That combination is the counterintuitive core and it follows
  directly from the drag.
- ``index_reconciliation`` closes the loop: an index is a rebalanced *average*, so it earns the
  mean rather than the median. Rebalancing is not a detail here — it is precisely the mechanism
  by which an index escapes the fate of its own constituents, and ``buy_and_hold_index`` shows
  what happens without it.

The survivorship caveat runs the *right* way for once, and is measured rather than asserted:
this basket contains only firms that lasted, so every statistic here understates the effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The engine: variance drag
# --------------------------------------------------------------------------- #
def variance_drag(arith_mean: float, vol: float) -> float:
    """The gap between arithmetic and geometric mean return, ≈ σ²/2.

    Exact for lognormal returns and a very good approximation otherwise. This single quantity
    explains the entire paradox: the average stock earns ``arith_mean``, the *typical* stock
    compounds at ``arith_mean − σ²/2``, and a portfolio that rebalances captures the former
    while a buy-and-hold single name gets the latter.
    """
    return float(vol ** 2 / 2)


def log_growth(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised **log** growth rate: mean(log(1+r)) × periods.

    The drag identity lives in log space and nowhere else. Comparing a simple-scaled
    arithmetic mean (252·mean(r)) against a *compounded* geometric return conflates the drag
    with ordinary compounding, and for a riskless series it even gets the sign wrong — a
    constant 0.04% daily return has a compounded annual figure **above** 252× the daily one.
    Both quantities used in `drag_table` are therefore annualised the same way, linearly, and
    the difference between them is σ²/2 exactly as the theory says.
    """
    r = rets.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.log1p(r).mean() * periods)


def geometric_mean(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised compounded return — the display figure, not the one used for the drag."""
    r = rets.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.expm1(np.log1p(r).sum() * periods / len(r)))


def arithmetic_mean(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.mean() * periods)


def breakeven_volatility(drift: float, cash_rate: float = 0.0) -> float:
    """The volatility at which the *median* outcome stops beating cash.

    The median compounds at the log growth rate, roughly ``drift − σ²/2``. Setting that equal
    to the cash rate and solving:

        σ* = sqrt(2 × (drift − cash_rate))

    This is the study's deliverable. "Most stocks lose to bills" is not a universal law about
    equities; it is what happens when a cross-section's volatility exceeds this threshold. A
    name with an 11% drift against 2% bills needs σ above roughly 42% before its median holder
    loses — which surviving large caps do not reach, and which small caps and the pre-delisting
    population comfortably do.
    """
    excess = drift - cash_rate
    return float(np.sqrt(2 * excess)) if excess > 0 else 0.0


def median_beats_cash_condition(rets: pd.DataFrame, cash: pd.Series) -> dict:
    """Apply the breakeven condition to a real cross-section, name by name.

    Reports each name's drift, its volatility, the volatility it *could* have tolerated, and
    how much headroom it had. A basket where every name sits far below its threshold is a
    basket where the median cannot lose, and saying so is more useful than reporting that it
    did not.
    """
    c = log_growth(cash)
    rows = []
    for name in rets.columns:
        s = rets[name].dropna()
        if len(s) < 250:
            continue
        drift = arithmetic_mean(s)
        vol = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS))
        thresh = breakeven_volatility(drift, c)
        rows.append({"name": name, "drift": drift, "vol": vol,
                     "breakeven_vol": thresh, "headroom": thresh - vol,
                     "log_growth": log_growth(s), "cash_log_growth": c,
                     "median_beats_cash": bool(log_growth(s) > c)})
    d = pd.DataFrame(rows).set_index("name")
    return {"table": d, "cash_log_growth": c,
            "share_above_threshold": float((d["headroom"] < 0).mean()),
            "median_headroom": float(d["headroom"].median()),
            "mean_drift": float(d["drift"].mean()),
            "mean_vol": float(d["vol"].mean()),
            "mean_breakeven": float(d["breakeven_vol"].mean())}


def drag_table(rets: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic mean, geometric mean and the theoretical drag, name by name.

    The check that matters is the last column: measured drag against σ²/2. If they agree, the
    mechanism is identified and everything downstream is arithmetic rather than conjecture.
    """
    rows = []
    for c in rets.columns:
        s = rets[c].dropna()
        if len(s) < 250:
            continue
        a = arithmetic_mean(s)
        lg = log_growth(s)
        vol = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append({"name": c, "arithmetic": a, "log_growth": lg,
                     "geometric": geometric_mean(s), "vol": vol,
                     "measured_drag": a - lg, "predicted_drag": variance_drag(a, vol),
                     "n": int(len(s))})
    return pd.DataFrame(rows).set_index("name")


# --------------------------------------------------------------------------- #
# The distribution of outcomes
# --------------------------------------------------------------------------- #
def holding_outcomes(rets: pd.DataFrame, cash: pd.Series, horizon_days: int,
                     step: int = 63) -> pd.DataFrame:
    """Every overlapping ``horizon_days`` holding period, for every name, against cash.

    Overlapping windows are used deliberately here and stepped by a quarter: the object of
    interest is the *distribution* of investor experiences, not a hypothesis test, so
    overlapping periods are the right sample. Nothing downstream computes a t-statistic on
    these, which is where overlap would be a problem.
    """
    idx = rets.dropna(how="all").index
    c = cash.reindex(idx).fillna(0.0)
    lc = np.log1p(c.to_numpy(dtype=float))
    rows = []
    for name in rets.columns:
        s = rets[name].reindex(idx)
        lr = np.log1p(s.to_numpy(dtype=float))
        n = len(lr)
        for start in range(0, n - horizon_days, step):
            sl = slice(start, start + horizon_days)
            seg = lr[sl]
            if not np.isfinite(seg).all():
                continue
            stock = float(np.expm1(seg.sum()))
            bench = float(np.expm1(lc[sl].sum()))
            rows.append({"name": name, "start": idx[start], "stock": stock,
                         "cash": bench, "excess": stock - bench,
                         "beat_cash": stock > bench, "lost_money": stock < 0.0})
    return pd.DataFrame(rows)


def outcome_distribution(out: pd.DataFrame) -> dict:
    """Summary of a holding-outcome frame: how many won, and how the winners did."""
    if out.empty:
        return {}
    return {"n": int(len(out)), "share_beat_cash": float(out["beat_cash"].mean()),
            "share_lost_money": float(out["lost_money"].mean()),
            "median_return": float(out["stock"].median()),
            "mean_return": float(out["stock"].mean()),
            "median_excess": float(out["excess"].median()),
            "mean_excess": float(out["excess"].mean()),
            "p10": float(out["stock"].quantile(0.10)),
            "p90": float(out["stock"].quantile(0.90)),
            "skew": float(np.log1p(out["stock"].clip(lower=-0.99)).skew())}


def horizon_sweep(rets: pd.DataFrame, cash: pd.Series,
                  horizons_years=(0.25, 1, 3, 5, 10, 15), step: int = 63) -> pd.DataFrame:
    """The share of holdings beating cash, against how long they were held.

    The counterintuitive result lives here. Expected return per name *rises* with horizon —
    more compounding of a positive drift — and the share of names beating cash *falls*, because
    the distribution's right skew grows faster than its centre moves. Both columns are reported
    side by side so the tension is visible rather than argued.
    """
    rows = []
    for hy in horizons_years:
        hd = max(int(hy * TRADING_DAYS), 21)
        out = holding_outcomes(rets, cash, hd, step)
        d = outcome_distribution(out)
        if not d:
            continue
        rows.append({"years": hy, **d})
    return pd.DataFrame(rows).set_index("years")


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def wealth_concentration(rets: pd.DataFrame, cash: pd.Series) -> dict:
    """How few names produced the basket's total gain over cash.

    Measured on **dollar wealth creation** from an equal starting stake, which is the
    Bessembinder construction: a name that turned $1 into $60 contributes fifty-nine dollars,
    and a name that halved contributes minus fifty cents. Ranking by percentage return would
    give a different and much less meaningful answer.
    """
    idx = rets.dropna(how="all").index
    c = float(np.expm1(np.log1p(cash.reindex(idx).fillna(0.0)).sum()))
    final = {}
    for name in rets.columns:
        s = rets[name].reindex(idx).dropna()
        if len(s) < 250:
            continue
        final[name] = float(np.expm1(np.log1p(s).sum()))
    if not final:
        return {}
    w = pd.Series(final)
    excess = w - c
    ranked = excess.sort_values(ascending=False)
    total = float(ranked.sum())
    cum = ranked.cumsum()
    n = len(ranked)
    out = {"n_names": n, "cash_return": c, "total_excess": total,
           "share_beat_cash": float((excess > 0).mean()),
           "share_lost_money": float((w < 0).mean()),
           "median_terminal": float(w.median()), "mean_terminal": float(w.mean()),
           "best_name": str(ranked.index[0]), "best_excess": float(ranked.iloc[0])}
    for frac in (0.02, 0.05, 0.10, 0.25, 0.50):
        k = max(int(np.ceil(n * frac)), 1)
        out[f"top_{int(frac * 100)}pct_share"] = (float(cum.iloc[k - 1] / total)
                                                  if total != 0 else np.nan)
    # how few names account for ALL of it
    positive = int((cum <= total).sum()) if total > 0 else n
    out["n_names_for_all"] = int(np.argmax(cum.to_numpy() >= total * 0.999) + 1) \
        if total > 0 else n
    out["share_of_names_for_all"] = out["n_names_for_all"] / n
    out["ranked"] = ranked
    return out


# --------------------------------------------------------------------------- #
# Reconciling with the index
# --------------------------------------------------------------------------- #
def index_reconciliation(rets: pd.DataFrame, cash: pd.Series) -> dict:
    """Why the index wins when its median member does not.

    Three portfolios of the *same fifty names*:

    - **Median name**: buy one at random and hold. Earns the geometric mean, i.e. the median
      outcome.
    - **Buy and hold the basket**: equal money at the start, never touched. The winners grow
      into most of the portfolio, so this earns close to the *mean* terminal wealth.
    - **Rebalanced equal weight**: reset to equal weights, which earns the arithmetic average
      return of the constituents, less the drag on the *portfolio's* own volatility — and a
      portfolio's volatility is far below a single stock's.

    That last line is the resolution. The index does not escape variance drag; it escapes *most*
    of it, because diversification cuts the σ² that the drag is proportional to.
    """
    idx = rets.dropna(how="all").index
    R = rets.reindex(idx)
    c = float(np.expm1(np.log1p(cash.reindex(idx).fillna(0.0)).sum()))
    finals = np.array([float(np.expm1(np.log1p(R[n].dropna()).sum()))
                       for n in R.columns if R[n].dropna().shape[0] > 250])
    eq = R.mean(axis=1).dropna()                      # rebalanced daily
    eq_final = float(np.expm1(np.log1p(eq).sum()))
    single_vol = float(np.nanmean([R[n].std(ddof=1) for n in R.columns])
                       * np.sqrt(TRADING_DAYS))
    port_vol = float(eq.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {"cash": c, "median_single": float(np.median(finals)),
            "mean_single": float(finals.mean()),
            "buy_and_hold_basket": float(finals.mean()),
            "rebalanced_equal_weight": eq_final,
            "single_vol": single_vol, "portfolio_vol": port_vol,
            "single_drag": variance_drag(0.0, single_vol),
            "portfolio_drag": variance_drag(0.0, port_vol),
            "drag_saved": variance_drag(0.0, single_vol) - variance_drag(0.0, port_vol),
            "n_names": int(len(finals))}


def buy_and_hold_index(rets: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight rebalanced against buy-and-hold, and how concentrated the latter becomes.

    Shows the mechanism at work over time: without rebalancing the basket turns into a bet on
    its winners, which raises the terminal value and destroys the diversification that produced
    the low portfolio volatility in the first place.
    """
    R = rets.dropna()
    if R.empty:
        return pd.DataFrame()
    paths = (1 + R).cumprod()
    bh_value = paths.mean(axis=1)
    weights = paths.div(paths.sum(axis=1), axis=0)
    reb = (1 + R.mean(axis=1)).cumprod()
    return pd.DataFrame({"buy_and_hold": bh_value, "rebalanced": reb,
                         "max_weight": weights.max(axis=1),
                         "top5_weight": weights.apply(
                             lambda r: r.nlargest(min(5, len(r))).sum(), axis=1),
                         "effective_n": 1.0 / (weights ** 2).sum(axis=1)})


def concentrated_portfolio_odds(rets: pd.DataFrame, cash: pd.Series, sizes=(1, 3, 5, 10, 20),
                                horizon_years: float = 10.0, n_draws: int = 500,
                                seed: int = 1006) -> pd.DataFrame:
    """The practical question: what are the odds a concentrated book beats cash, and the index?

    This is what the Bessembinder result actually implies for a person. Holding five names is
    not "slightly less diversified than holding fifty" — it is a materially different bet on
    the right tail, and the probability of underperforming is the number that should be
    quoted.
    """
    R = rets.dropna()
    hd = max(int(horizon_years * TRADING_DAYS), 21)
    if len(R) < hd + 10:
        return pd.DataFrame()
    idx = R.index
    c = np.log1p(cash.reindex(idx).fillna(0.0).to_numpy(dtype=float))
    L = np.log1p(R.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    n_cols = L.shape[1]
    starts = np.arange(0, len(L) - hd, 21)
    full = np.expm1(L.mean(axis=1)[None, :].repeat(1, axis=0))
    rows = []
    for k in sizes:
        beat_cash, beat_index, finals = [], [], []
        for _ in range(n_draws):
            pick = rng.choice(n_cols, size=min(k, n_cols), replace=False)
            s0 = int(rng.choice(starts))
            sl = slice(s0, s0 + hd)
            port = np.expm1(np.log1p(np.expm1(L[sl][:, pick]).mean(axis=1)).sum())
            bench = float(np.expm1(c[sl].sum()))
            index = float(np.expm1(np.log1p(np.expm1(L[sl]).mean(axis=1)).sum()))
            beat_cash.append(port > bench)
            beat_index.append(port > index)
            finals.append(port)
        rows.append({"n_stocks": k, "share_beat_cash": float(np.mean(beat_cash)),
                     "share_beat_index": float(np.mean(beat_index)),
                     "median_return": float(np.median(finals)),
                     "mean_return": float(np.mean(finals)),
                     "p10": float(np.percentile(finals, 10)),
                     "p90": float(np.percentile(finals, 90))})
    return pd.DataFrame(rows).set_index("n_stocks")


# --------------------------------------------------------------------------- #
# The synthetic control
# --------------------------------------------------------------------------- #
def synthetic_cross_section(n_stocks: int = 200, n_days: int = 5000,
                            drift: float = 0.08, vol: float = 0.35,
                            seed: int = 1006) -> pd.DataFrame:
    """Independent lognormal stocks with an identical expected return.

    Every name has exactly the same arithmetic expected return, so any dispersion in outcomes
    is pure compounding luck. Raising ``vol`` widens the gap between the median and the mean
    without changing the mean at all — which is the study's mechanism in a single knob, and
    lets the claim be checked against theory rather than merely observed.
    """
    rng = np.random.default_rng(seed)
    dv = vol / np.sqrt(TRADING_DAYS)
    # set the daily log-drift so the ARITHMETIC annual mean is exactly `drift`
    mu_log = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
    R = np.expm1(rng.normal(mu_log, dv, (n_days, n_stocks)))
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.DataFrame(R, index=idx,
                        columns=[f"S{i:03d}" for i in range(n_stocks)])


def survivorship_experiment(n_stocks: int = 300, n_days: int = 5000, drift: float = 0.08,
                            vol: float = 0.40, delist_at: float = -0.80,
                            seed: int = 1006) -> dict:
    """Quantify which way survivorship bias runs, instead of merely gesturing at it.

    Names that fall below ``delist_at`` are removed from the "survivor" sample, exactly as the
    real basket in this study excludes companies that failed. Comparing the full cross-section
    with the survivors measures the bias directly — and it runs *against* the finding, meaning
    every real-data number here is a conservative one.
    """
    R = synthetic_cross_section(n_stocks, n_days, drift, vol, seed)
    finals = np.expm1(np.log1p(R).sum())
    survived = finals[finals > delist_at]
    return {"n_all": int(len(finals)), "n_survived": int(len(survived)),
            "delist_rate": float(1 - len(survived) / len(finals)),
            "median_all": float(finals.median()),
            "median_survivors": float(survived.median()),
            "share_negative_all": float((finals < 0).mean()),
            "share_negative_survivors": float((survived < 0).mean()),
            "bias": float(survived.median() - finals.median())}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: about the claim *as tested here*, on surviving large caps. **Confirmed** if
      the median holding underperforms cash at long horizons; **Partial** if marginally;
      **Busted** if the median comfortably beats cash throughout. The stamp is deliberately
      about this basket, not about Bessembinder's universe, which this data cannot reach.
    - **Tradability**: **Useful** if the study still yields a quantified, actionable statement
      — the breakeven condition and the concentration penalty both survive a Busted signal;
      **Partial** if only directional; **Mirage** if nothing actionable remains.
    """
    signal = ("Confirmed" if h["share_beat_cash_long"] < 0.50
              else ("Partial" if h["share_beat_cash_long"] < 0.60 else "Busted"))
    gap = h["odds_index_50"] - h["odds_index_5"]
    trad = ("Useful" if gap > 0.10 else ("Partial" if gap > 0.03 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"**Not on this basket, and the failure is the finding.** These {h['n_names']} "
            f"names all survived to 2026 — the exact opposite of Bessembinder's universe, which "
            f"is every firm that ever listed and mostly delisted. Here "
            f"**{h['share_beat_cash_long']:.0%}** of {h['long_years']:.0f}-year holding periods "
            f"beat Treasury bills, against {h['share_beat_cash_short']:.0%} at one year: the "
            f"share beating cash **rises** with horizon, the reverse of the headline. So the "
            f"claim is not a universal law about equities, and the useful question becomes what "
            f"it takes to produce it. The mechanism is real and verified — measured drag across "
            f"these names averaged {h['mean_drag']:.2%} a year against a theoretical "
            f"{h['predicted_drag']:.2%}, a correlation of {h['drag_corr']:.2f} name by name — "
            f"but the median compounds at roughly drift minus σ²/2, and these names' drift is "
            f"large enough to absorb it. The condition is exact: the median beats cash while "
            f"σ < sqrt(2·(drift − cash)). At an average drift of {h['mean_drift']:.1%} against "
            f"cash's {h['cash_log_growth']:.1%}, that threshold is **{h['mean_breakeven']:.0%} "
            f"volatility**, and these names average {h['mean_vol']:.0%} — a headroom of "
            f"{h['median_headroom']:.0%}, with {h['share_above_threshold']:.0%} of them over "
            f"the line. Bessembinder's population — small caps and firms on their way to "
            f"delisting — sits on the other side of that threshold. The synthetic control "
            f"quantifies the survivorship half of the gap: removing simulated failures raises "
            f"the median terminal wealth by {h['surv_bias']:.2f}×."),
        "trad_why": (
            f"The implication is not \"don't buy stocks\", it is \"don't buy few stocks\", and "
            f"the reconciliation makes that precise. Over the full sample the median single "
            f"name returned {h['median_single']:.1f}× against cash's {h['cash']:.1f}×, while "
            f"the **rebalanced equal-weight basket of those same names** returned "
            f"{h['rebalanced']:.1f}×. The index does not escape variance drag; it escapes most "
            f"of it, because drag is proportional to σ² and diversification cuts the average "
            f"single-name volatility of {h['single_vol']:.0%} to a portfolio "
            f"{h['portfolio_vol']:.0%} — worth {h['drag_saved']:.2%} a year, compounding. "
            f"Rebalancing is doing real work rather than housekeeping. Priced as odds over "
            f"{h['odds_horizon']:.0f} years: a five-stock portfolio beat the index "
            f"{h['odds_index_5']:.0%} of the time, a twenty-stock "
            f"{h['odds_index_20']:.0%}, the full basket {h['odds_index_50']:.0%}. A "
            f"concentrated book is not a slightly worse diversified one — it is a different "
            f"bet, on the right tail, that most often loses."),
        "trad": trad,
        "one_sentence": (
            f"On surviving large caps the famous result reverses — "
            f"{h['share_beat_cash_long']:.0%} of {h['long_years']:.0f}-year holdings beat bills "
            f"— because the median only loses once volatility exceeds "
            f"sqrt(2·(drift−cash)) ≈ {h['mean_breakeven']:.0%}, and these names run at "
            f"{h['mean_vol']:.0%}."),
    }
