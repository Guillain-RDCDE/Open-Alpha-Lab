"""The best-days statistic, and the three things it leaves out — Study 1002.

The claim is arithmetically true and rhetorically dishonest, which is an unusual combination and
worth taking apart carefully.

**What is true.** Compound returns are a product. Remove the largest factors from a product and
the product falls a great deal. Ten days out of roughly eight thousand really can be half the
total return, because daily returns are fat-tailed and the biggest ones are enormous.

**What is left out.**

1. **The symmetric statistic.** Missing the ten *worst* days improves the outcome by more than
   missing the ten best days damages it — because the worst days are larger in magnitude than
   the best ones. This number is never quoted, and it is not a technicality: it is exactly the
   same calculation with the sign flipped, and it points the opposite way.

2. **The best and worst days are neighbours.** They are not scattered through history; they
   cluster in the same few weeks, during crashes. October 2008 contains several of each. So the
   scenario "you were out of the market for the ten best days but present for all the worst" is
   not a thing that can happen to an investor following any rule. ``clustering_stats`` measures
   the distance between them, against a shuffled benchmark that destroys the clustering while
   keeping every return.

3. **Nobody misses days at random.** The brochure's implied counterfactual is an investor who
   steps out on precisely the ten worst possible days, which is a straw man. ``timing_frontier``
   asks the honest question instead: given that a switching rule that avoids bad days will also
   avoid some good ones, **what accuracy would it need** to beat buy-and-hold? That number turns
   out to be a genuine and demanding threshold, which is a much better argument for staying
   invested than the one on the brochure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The statistic itself
# --------------------------------------------------------------------------- #
def total_return(rets: pd.Series) -> float:
    """Compound total return over the whole series."""
    return float(np.expm1(np.log1p(rets.dropna()).sum()))


def annualised(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.expm1(np.log1p(r).sum() * periods / len(r)))


def drop_extremes(rets: pd.Series, n: int, which: str = "best",
                  replace_with: float = 0.0) -> pd.Series:
    """Return the series with the ``n`` best (or worst) days replaced by ``replace_with``.

    Replacing rather than deleting keeps the number of periods constant, which matters for the
    annualised figure. The brochure version deletes, which quietly shortens the horizon and
    flatters the effect a little further.
    """
    r = rets.dropna().copy()
    if n <= 0 or n >= len(r):
        return r
    order = np.argsort(r.to_numpy())
    idx = order[-n:] if which == "best" else order[:n]
    out = r.copy()
    out.iloc[idx] = replace_with
    return out


def missed_days_table(rets: pd.Series, counts=(0, 5, 10, 20, 30, 50, 100),
                      cash_rate: float = 0.0) -> pd.DataFrame:
    """The brochure table — and beside it, the one the brochure omits.

    ``replace_with`` is the cash rate, because an investor who is out of the market holds cash
    rather than a zero-return void. That detail moves the answer and is almost always skipped.
    """
    r = rets.dropna()
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    rows = []
    for n in counts:
        best = drop_extremes(r, n, "best", daily_cash)
        worst = drop_extremes(r, n, "worst", daily_cash)
        both = drop_extremes(drop_extremes(r, n, "best", daily_cash), n, "worst", daily_cash)
        rows.append({"days": n,
                     "miss_best_total": total_return(best),
                     "miss_best_cagr": annualised(best),
                     "miss_worst_total": total_return(worst),
                     "miss_worst_cagr": annualised(worst),
                     "miss_both_total": total_return(both),
                     "miss_both_cagr": annualised(both)})
    return pd.DataFrame(rows).set_index("days")


def asymmetry(rets: pd.Series, n: int = 10) -> dict:
    """How the two halves of the statistic compare, and why.

    The brochure prints the first number and not the second. The ratio between them is the
    single most informative summary of the rhetorical trick.

    The **mechanism** is worth stating precisely, because the obvious explanation is wrong.
    One would guess the omitted half is bigger because crashes are bigger than rallies. On the
    S&P 500 that is false in percentage terms: the ten best days average roughly +8.7% and the
    ten worst roughly −8.4%, so the *best* days are the larger ones. The asymmetry survives
    anyway, and it survives because compounding is multiplicative. Removing a day of return
    *x* from the product multiplies the result by 1/(1+*x*), so the quantity that matters is
    **log(1+x)**, not *x* — and in log terms a −8.4% day (−0.0873) is larger than a +8.7% day
    (+0.0838). Both ``mean_best``/``mean_worst`` and their log counterparts are returned so the
    two framings can be compared directly; ``worst_bigger_than_best`` uses the log scale,
    because that is the scale the compounding actually happens on.
    """
    r = rets.dropna()
    base = annualised(r)
    lose = base - annualised(drop_extremes(r, n, "best"))
    gain = annualised(drop_extremes(r, n, "worst")) - base
    s = np.sort(r.to_numpy())
    log_best = float(np.log1p(s[-n:]).mean())
    log_worst = float(np.log1p(s[:n]).mean())
    return {"n": n, "base_cagr": base, "cost_of_missing_best": lose,
            "benefit_of_missing_worst": gain,
            "ratio": float(gain / lose) if lose != 0 else np.nan,
            "mean_best": float(s[-n:].mean()), "mean_worst": float(s[:n].mean()),
            "log_best": log_best, "log_worst": log_worst,
            "best_bigger_in_percent": bool(s[-n:].mean() > abs(s[:n].mean())),
            "worst_bigger_than_best": bool(abs(log_worst) > log_best)}


# --------------------------------------------------------------------------- #
# Where the extreme days actually are
# --------------------------------------------------------------------------- #
def extreme_days(rets: pd.Series, n: int = 10) -> pd.DataFrame:
    """The n best and n worst days, dated, with their positions in the series."""
    r = rets.dropna()
    pos = pd.Series(np.arange(len(r)), index=r.index)
    s = r.sort_values()
    rows = [{"date": d, "ret": float(v), "kind": "worst", "pos": int(pos[d])}
            for d, v in s.head(n).items()]
    rows += [{"date": d, "ret": float(v), "kind": "best", "pos": int(pos[d])}
             for d, v in s.tail(n).items()]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def clustering_stats(rets: pd.Series, n: int = 10, n_shuffles: int = 400,
                     seed: int = 1002) -> dict:
    """Are the best days near the worst days, or is that an impression?

    The statistic is the median number of sessions from each best day to the nearest worst day.
    The benchmark is the same returns shuffled — which preserves every value, and the fat tail
    with it, while destroying the clustering. If the real gap is far smaller than the shuffled
    gap, the two sets of days genuinely arrive together and the brochure's counterfactual is
    not merely unlikely but structurally impossible.
    """
    r = rets.dropna()
    v = r.to_numpy()

    def gap(x):
        order = np.argsort(x)
        b = np.sort(order[-n:])
        w = np.sort(order[:n])
        d = np.abs(b[:, None] - w[None, :])
        return float(np.median(d.min(axis=1)))

    actual = gap(v)
    rng = np.random.default_rng(seed)
    null = np.array([gap(rng.permutation(v)) for _ in range(n_shuffles)])
    return {"n": n, "median_gap": actual, "shuffled_median_gap": float(np.median(null)),
            "shuffled_p05": float(np.percentile(null, 5)),
            "p_value": float((null <= actual).mean()),
            "ratio": float(actual / np.median(null)) if np.median(null) > 0 else np.nan}


def volatility_context(rets: pd.Series, n: int = 10, window: int = 21) -> dict:
    """What the market looked like around the extreme days.

    Both sets arrive in high-volatility regimes. That is the mechanism behind the clustering,
    and it is also why any rule that reduces exposure in turbulent markets necessarily gives up
    some of the best days along with some of the worst.
    """
    r = rets.dropna()
    vol = r.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    ex = extreme_days(r, n)
    out = {"typical_vol": float(vol.median())}
    for kind in ("best", "worst"):
        d = ex[ex["kind"] == kind]["date"]
        out[f"{kind}_vol"] = float(vol.reindex(d).dropna().median())
        out[f"{kind}_vol_ratio"] = out[f"{kind}_vol"] / out["typical_vol"]
    return out


def drawdown_context(rets: pd.Series, n: int = 10) -> dict:
    """How deep in a drawdown the best days occurred.

    The best days do not happen at market peaks. They happen part-way down, when an investor
    who has just bailed out is least likely to be present — which is a real argument for
    staying invested, and a different one from the brochure's.
    """
    r = rets.dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ex = extreme_days(r, n)
    out = {}
    for kind in ("best", "worst"):
        d = ex[ex["kind"] == kind]["date"]
        out[f"{kind}_median_drawdown"] = float(dd.reindex(d).dropna().median())
    out["typical_drawdown"] = float(dd.median())
    return out


# --------------------------------------------------------------------------- #
# What accuracy would a real timer need?
# --------------------------------------------------------------------------- #
def down_day_share(rets: pd.Series) -> float:
    """The share of sessions with a negative return — the hit rate of a coin-flip timer."""
    r = rets.dropna()
    return float((r < 0).mean())


def timing_frontier(rets: pd.Series, out_fraction: float = 0.20, hit_rates=None,
                    n_draws: int = 200, cash_rate: float = 0.0,
                    seed: int = 1002) -> pd.DataFrame:
    """The honest question: how accurate must a market timer be?

    **How "accuracy" is defined matters enormously, and the obvious definition is useless.**
    A first attempt at this function scored a "hit" as landing on one of the forty worst days
    in thirty-three years. Under that definition the break-even accuracy came out near 1%,
    which is not a finding about market timing — it is a finding about the definition. Hitting
    the forty worst days of a lifetime is not skill a coin flip has any access to, so a "50%
    hit rate" in that sense already describes a clairvoyant.

    The definition used here is the one a real rule can be scored against: a timer sits out
    ``out_fraction`` of all sessions, and its **hit rate is the share of those sit-out days
    that turn out to be down days**. A timer choosing at random achieves the unconditional
    down-day share — about 45% — so *that*, not 50%, is the coin-flip benchmark. The
    remaining sit-out days are drawn from the up days.

    This makes the frontier answerable and comparable: it is directional accuracy on days the
    rule is actually out of the market, which is a quantity a moving-average cross or a
    volatility target can be measured on.
    """
    if hit_rates is None:
        hit_rates = (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0)
    r = rets.dropna()
    v = r.to_numpy()
    n = len(v)
    k = max(int(round(out_fraction * n)), 1)
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    down = np.flatnonzero(v < 0)
    up = np.flatnonzero(v >= 0)
    base = annualised(r)
    rng = np.random.default_rng(seed)
    rows = []
    for hr in hit_rates:
        n_hit = min(int(round(k * hr)), len(down))
        n_miss = min(k - n_hit, len(up))
        outs = []
        for _ in range(n_draws):
            pick = np.concatenate([rng.choice(down, size=n_hit, replace=False),
                                   rng.choice(up, size=n_miss, replace=False)])
            w = v.copy()
            w[pick] = daily_cash
            outs.append(float(np.expm1(np.log1p(w).sum() * TRADING_DAYS / n)))
        outs = np.array(outs)
        rows.append({"hit_rate": hr, "days_out": k, "median_cagr": float(np.median(outs)),
                     "p10": float(np.percentile(outs, 10)),
                     "p90": float(np.percentile(outs, 90)),
                     "beats_hold": float(np.mean(outs > base)),
                     "buy_and_hold": base})
    return pd.DataFrame(rows).set_index("hit_rate")


def breakeven_hit_rate(rets: pd.Series, out_fraction: float = 0.20,
                       cash_rate: float = 0.0, seed: int = 1002) -> float:
    """The hit rate at which timing matches buy-and-hold, by linear interpolation.

    Compare it against ``down_day_share`` — a random timer's accuracy — rather than against
    50%. The gap between the two is what a rule has to earn.
    """
    grid = np.linspace(0.35, 1.0, 14)
    f = timing_frontier(rets, out_fraction, grid, 200, cash_rate, seed)
    base = float(f["buy_and_hold"].iloc[0])
    x = f.index.to_numpy(dtype=float)
    y = f["median_cagr"].to_numpy() - base
    for i in range(1, len(y)):
        if y[i - 1] < 0 <= y[i]:
            return float(x[i - 1] + (x[i] - x[i - 1]) * (-y[i - 1]) / (y[i] - y[i - 1]))
    return float("nan") if y[-1] < 0 else float(x[0])


def out_of_market_cost(rets: pd.Series, fractions=(0.01, 0.05, 0.10, 0.25),
                       n_draws: int = 300, cash_rate: float = 0.0,
                       seed: int = 1002) -> pd.DataFrame:
    """The cost of missing days chosen AT RANDOM — the correct null for the brochure.

    Missing ten days chosen at random costs almost nothing. The brochure's number comes from
    missing the ten *specifically best* days, which is a selection no process produces. Putting
    the two side by side is the cleanest way to show what the statistic is doing.
    """
    r = rets.dropna()
    v = r.to_numpy()
    n = len(v)
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    base = annualised(r)
    rng = np.random.default_rng(seed)
    rows = []
    for f in fractions:
        k = max(int(round(f * n)), 1)
        outs = []
        for _ in range(n_draws):
            w = v.copy()
            w[rng.choice(n, size=k, replace=False)] = daily_cash
            outs.append(float(np.expm1(np.log1p(w).sum() * TRADING_DAYS / n)))
        worst_case = annualised(drop_extremes(r, k, "best", daily_cash))
        rows.append({"fraction": f, "days": k, "random_median_cagr": float(np.median(outs)),
                     "random_p05": float(np.percentile(outs, 5)),
                     "worst_case_cagr": worst_case, "buy_and_hold": base,
                     "random_cost": base - float(np.median(outs)),
                     "worst_case_cost": base - worst_case})
    return pd.DataFrame(rows).set_index("fraction")


def synthetic_returns(n: int = 8000, clustered: bool = True, mu: float = 0.0003,
                      sigma: float = 0.011, persistence: float = 0.94,
                      vol_of_vol: float = 0.25, seed: int = 1002) -> pd.Series:
    """Returns with or without volatility clustering, same unconditional fat tail.

    The comparison is the experiment. Without clustering, extreme days are scattered at random
    and the best-days statistic must be entirely a consequence of the fat tail. With clustering,
    the best and worst days arrive together — and only then should the measured gap between them
    collapse the way it does on the real tape.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1993-02-01", periods=n)
    if clustered:
        log_v = np.zeros(n)
        e = rng.normal(0, vol_of_vol, n)
        for t in range(1, n):
            log_v[t] = persistence * log_v[t - 1] + e[t]
        vol = sigma * np.exp(log_v - log_v.var() / 2)
        r = mu + vol * rng.standard_t(6, n) / np.sqrt(6 / 4)
    else:
        r = mu + sigma * rng.standard_t(6, n) / np.sqrt(6 / 4)
    return pd.Series(r, index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the arithmetic of the best-days claim checks out *and* the omitted
      symmetric statistic is at least as large — i.e. the number is true and the argument built
      on it is not; **Mixed** if only one holds; **Busted** if the arithmetic itself fails.
    - **Tradability**: keyed to the accuracy a timer must earn **above what random selection
      already gives**, which is the only version of the question that means anything.
      **Investable** if the required edge is under a point; **Fragile** if it is a few points
      but must be sustained; **Mirage** if it is large.
    """
    arith = h["cost_of_missing_best"] > 0.01
    symmetric = h["asym_ratio"] >= 1.0
    signal = ("Real" if (arith and symmetric)
              else ("Mixed" if arith else "Busted"))
    be = h["breakeven_hit_rate"]
    edge = h["timing_edge_needed"]
    trad = ("Investable" if edge <= 0.01 else ("Fragile" if edge <= 0.05 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The arithmetic is correct. On {h['asset']} over {h['years']:.0f} years, missing "
            f"the ten best sessions cuts the annualised return from {h['base_cagr']:.2%} to "
            f"{h['base_cagr'] - h['cost_of_missing_best']:.2%} — a loss of "
            f"**{h['cost_of_missing_best']:.2%} a year** from ten days out of {h['n_days']:,}. "
            f"The brochure stops there. Missing the ten **worst** sessions raises it to "
            f"{h['base_cagr'] + h['benefit_of_missing_worst']:.2%}, a gain of "
            f"**{h['benefit_of_missing_worst']:.2%} a year** — "
            f"**{h['asym_ratio']:.2f}× the size of the loss**. The reason is not the one you "
            f"would guess. In percentage terms the ten best days are the *bigger* ones "
            f"({h['mean_best']:+.2%} against {h['mean_worst']:.2%}), so \"crashes are larger "
            f"than rallies\" is simply false here. The asymmetry survives because compounding "
            f"is multiplicative: removing a day multiplies the result by 1/(1+x), so the "
            f"quantity that matters is log(1+x) — and in log terms the worst days are the "
            f"larger ({h['log_worst']:+.4f} against {h['log_best']:+.4f}). "
            f"And the two sets are not scattered through history: the median distance from a "
            f"best day to the nearest worst day is **{h['median_gap']:.0f} sessions**, against "
            f"{h['shuffled_gap']:.0f} when the same returns are shuffled (p = "
            f"{h['cluster_p']:.3f}). They arrive in the same storms — the best days occurred at "
            f"a median drawdown of {h['best_drawdown']:.1%} and in volatility "
            f"{h['best_vol_ratio']:.1f}× normal. Being absent for all ten best days while "
            f"present for all ten worst is not an outcome any rule can produce."),
        "trad_why": (
            f"Which makes the interesting question the one nobody asks: how accurate would a "
            f"timer need to be? Take a rule that sits out {h['out_fraction']:.0%} of all "
            f"sessions — {h['days_out']:,} days — and score it on the share of those days that "
            f"turn out to be down days. **Choosing at random already achieves "
            f"{h['coin_flip_rate']:.1%}**, the unconditional down-day frequency, so that and "
            f"not 50% is the benchmark. Break-even against buy-and-hold arrives at "
            f"**{be:.1%}**, an edge of only **{h['timing_edge_needed'] * 100:.1f} percentage "
            f"points** over random. That sounds modest, and the modesty is the trap: the "
            f"frontier is savagely steep on both sides. Random selection — sitting out "
            f"{h['out_fraction']:.0%} of days with no skill whatsoever — returns "
            f"{h['coin_flip_cagr']:.2%} against buy-and-hold's {h['base_cagr']:.2%}, and a "
            f"timer {h['below_gap']:.0f} points *below* random gets {h['below_cagr']:.2%}. "
            f"The edge must be small, positive, and sustained across every one of "
            f"{h['days_out']:,} decisions; being slightly wrong is far more expensive than "
            f"being slightly right is profitable. For scale, missing {h['random_days']:,} days "
            f"chosen at random costs {h['random_cost']:.2%} a year while missing the "
            f"{h['random_days']:,} *best* costs {h['worst_case_cost']:.2%} — the brochure "
            f"quotes the worst case of a selection no process generates."),
        "trad": trad,
        "one_sentence": (
            f"Missing the ten best days costs {h['cost_of_missing_best']:.2%} a year and "
            f"missing the ten worst gains {h['benefit_of_missing_worst']:.2%} — they are "
            f"{h['median_gap']:.0f} sessions apart in the same storms, and a timer needs "
            f"{h['timing_edge_needed'] * 100:.1f} points of accuracy above random just to "
            f"break even."),
    }
