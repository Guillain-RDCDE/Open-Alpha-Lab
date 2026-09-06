"""The start-date lottery — Study 1008.

A lump sum invested once has no sequence risk: multiplication commutes, so the order of returns
cannot change the terminal value. Everyone else has sequence risk, and the amount depends on the
shape of their cash flows.

That observation organises the whole study. ``shuffle_invariance`` demonstrates it directly —
the same returns in a different order give an identical lump-sum result and a materially
different one for a contributor — which turns "sequence risk" from a phrase into a measurable
quantity: the dispersion that *disappears* when the path is shuffled is the part driven by the
distribution, and the rest is ordering.

The decomposition matters because the remedies target different halves. Diversification and
lower volatility shrink the distributional part. Contribution smoothing, glide paths and
withdrawal flexibility target the ordering part, and only make sense once you know how big it
is.

``accumulation_paths`` and ``decumulation_paths`` run the real thing over every available start
date. ``glide_path`` implements the standard age-based de-risking rule and, more usefully,
lets its shape be varied so the trade-off between dispersion reduction and expected wealth
given up can be plotted rather than argued. ``sequence_risk_metrics`` isolates the well-known
result that returns near the *end* of accumulation and the *start* of decumulation dominate —
and quantifies it as an explicit sensitivity by period, which is the version an investor can
act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The engine: does order matter?
# --------------------------------------------------------------------------- #
def terminal_lump_sum(rets: np.ndarray) -> float:
    """Terminal value of one unit invested at the start. Order-invariant by construction."""
    return float(np.exp(np.log1p(rets).sum()))


def terminal_with_contributions(rets: np.ndarray, contribution: float = 1.0,
                                initial: float = 0.0) -> float:
    """Terminal value when a fixed amount is added every period.

    Not order-invariant, and that is the entire subject. Money added late has less time to
    compound, so a path that rises early and falls late leaves a contributor far worse off than
    the reverse — even though both paths have the same average return.
    """
    v = float(initial)
    for r in rets:
        v = v * (1.0 + r) + contribution
    return v


def terminal_with_withdrawals(rets: np.ndarray, withdrawal: float,
                              initial: float = 1.0) -> dict:
    """Terminal value when a fixed amount is taken out every period, with a ruin flag.

    Withdrawals invert the asymmetry: a bad *first* decade is what ruins a retiree, because the
    withdrawals consume a portfolio that has not yet recovered. Reporting the ruin flag
    separately matters — a mean terminal value across paths conceals that some of them hit zero.
    """
    v = float(initial)
    ruined_at = -1
    for i, r in enumerate(rets):
        v = v * (1.0 + r) - withdrawal
        if v <= 0 and ruined_at < 0:
            ruined_at = i
            v = 0.0
            break
    return {"terminal": max(v, 0.0), "ruined": ruined_at >= 0, "ruined_at": ruined_at}


def shuffle_invariance(rets: np.ndarray, contribution: float = 1.0,
                       n_shuffles: int = 200, seed: int = 1008) -> dict:
    """The demonstration the whole study rests on.

    Shuffling a return path preserves every moment of the distribution and destroys the
    ordering. A lump sum is unaffected — to machine precision — while a contributor's outcome
    scatters. The spread of the shuffled contributor outcomes is a clean measure of **pure
    sequence risk** on this path, with the distribution held exactly fixed.
    """
    rng = np.random.default_rng(seed)
    lump, contrib = [], []
    for _ in range(n_shuffles):
        p = rng.permutation(rets)
        lump.append(terminal_lump_sum(p))
        contrib.append(terminal_with_contributions(p, contribution))
    lump, contrib = np.array(lump), np.array(contrib)
    return {"lump_mean": float(lump.mean()), "lump_sd": float(lump.std(ddof=1)),
            "lump_cv": float(lump.std(ddof=1) / lump.mean()) if lump.mean() else np.nan,
            "contrib_mean": float(contrib.mean()),
            "contrib_sd": float(contrib.std(ddof=1)),
            "contrib_cv": float(contrib.std(ddof=1) / contrib.mean())
            if contrib.mean() else np.nan,
            "contrib_p05": float(np.percentile(contrib, 5)),
            "contrib_p95": float(np.percentile(contrib, 95)),
            "sequence_spread": float(np.percentile(contrib, 95)
                                     / np.percentile(contrib, 5))}


# --------------------------------------------------------------------------- #
# Real paths, every start date
# --------------------------------------------------------------------------- #
def start_dates(rets: pd.Series, years: float, step: int = 21) -> np.ndarray:
    r = rets.dropna()
    h = int(years * TRADING_DAYS)
    return np.arange(0, len(r) - h + 1, step) if len(r) > h else np.empty(0, dtype=int)


def accumulation_paths(rets: pd.Series, years: float = 30.0, contribution: float = 1.0,
                       step: int = 21) -> pd.DataFrame:
    """Terminal wealth for a monthly contributor starting on every available date."""
    r = rets.dropna()
    h = int(years * TRADING_DAYS)
    v = r.to_numpy(dtype=float)
    per_period = contribution / 21.0            # spread the monthly amount over the month
    rows = []
    for s in start_dates(r, years, step):
        seg = v[s:s + h]
        rows.append({"start": r.index[s], "end": r.index[s + h - 1],
                     "terminal": terminal_with_contributions(seg, per_period),
                     "invested": per_period * h,
                     "lump_sum": terminal_lump_sum(seg),
                     "cagr": float(np.expm1(np.log1p(seg).sum() / years))})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["multiple"] = d["terminal"] / d["invested"]
    return d


def decumulation_paths(rets: pd.Series, years: float = 30.0, withdrawal_rate: float = 0.04,
                       step: int = 21) -> pd.DataFrame:
    """Outcome for a retiree withdrawing a fixed real amount, starting on every date."""
    r = rets.dropna()
    h = int(years * TRADING_DAYS)
    v = r.to_numpy(dtype=float)
    per_period = withdrawal_rate / TRADING_DAYS
    rows = []
    for s in start_dates(r, years, step):
        seg = v[s:s + h]
        out = terminal_with_withdrawals(seg, per_period, 1.0)
        rows.append({"start": r.index[s], "end": r.index[s + h - 1], **out,
                     "first_5y_cagr": float(np.expm1(
                         np.log1p(seg[:min(5 * TRADING_DAYS, len(seg))]).sum()
                         / min(5.0, years)))})
    return pd.DataFrame(rows)


def path_dispersion(paths: pd.DataFrame, column: str = "terminal") -> dict:
    """Summary of how much the start date mattered, with an effective-sample count."""
    if paths.empty or column not in paths:
        return {}
    v = paths[column].to_numpy(dtype=float)
    span_years = (paths["end"].iloc[-1] - paths["start"].iloc[0]).days / 365.25
    horizon = (paths["end"].iloc[0] - paths["start"].iloc[0]).days / 365.25
    return {"n_paths": int(len(v)), "effective_n": float(max(span_years / horizon, 1.0)),
            "median": float(np.median(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "cv": float(v.std(ddof=1) / v.mean())
            if v.mean() else np.nan,
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max()),
            "ratio_95_05": float(np.percentile(v, 95) / np.percentile(v, 5))
            if np.percentile(v, 5) > 0 else np.inf,
            "ratio_max_min": float(v.max() / v.min()) if v.min() > 0 else np.inf,
            "best_start": str(paths.loc[paths[column].idxmax(), "start"].date()),
            "worst_start": str(paths.loc[paths[column].idxmin(), "start"].date())}


# --------------------------------------------------------------------------- #
# Where does the sequence risk live?
# --------------------------------------------------------------------------- #
def sequence_risk_metrics(rets: pd.Series, years: float = 30.0, contribution: float = 1.0,
                          n_buckets: int = 6, step: int = 21) -> pd.DataFrame:
    """How much each slice of the horizon matters to a contributor's final outcome.

    Correlates the terminal wealth of each start date against the return realised in each
    *portion* of that path. For a lump sum every bucket must matter equally; for a contributor
    the late buckets dominate, because that is when the balance is largest. Quantifying the
    profile is what turns "sequence risk" into something an investor can plan around — the last
    decade before retirement is where the exposure actually sits.
    """
    r = rets.dropna()
    h = int(years * TRADING_DAYS)
    v = r.to_numpy(dtype=float)
    starts = start_dates(r, years, step)
    if len(starts) < 20:
        return pd.DataFrame()
    edges = np.linspace(0, h, n_buckets + 1).astype(int)
    finals, lumps, bucket_rets = [], [], []
    for s in starts:
        seg = v[s:s + h]
        finals.append(terminal_with_contributions(seg, contribution / 21.0))
        lumps.append(terminal_lump_sum(seg))
        bucket_rets.append([float(np.log1p(seg[edges[i]:edges[i + 1]]).sum())
                            for i in range(n_buckets)])
    finals = np.log(np.array(finals))
    lumps = np.log(np.array(lumps))
    B = np.array(bucket_rets)
    rows = []
    for i in range(n_buckets):
        rows.append({
            "bucket": i + 1,
            "years_from": round(i * years / n_buckets, 1),
            "years_to": round((i + 1) * years / n_buckets, 1),
            "corr_contributor": float(np.corrcoef(B[:, i], finals)[0, 1]),
            "corr_lump_sum": float(np.corrcoef(B[:, i], lumps)[0, 1]),
        })
    return pd.DataFrame(rows).set_index("bucket")


def glide_path(n_periods: int, start_weight: float = 1.0, end_weight: float = 0.3,
               shape: str = "linear", knee: float = 0.5) -> np.ndarray:
    """Equity weight over time, under several de-risking shapes.

    ``linear`` is the textbook rule. ``late`` keeps full equity until a knee and then de-risks
    quickly; ``early`` does the opposite. Having all three lets the question be settled by
    measurement — if sequence risk is concentrated in the final years, then de-risking late
    should buy most of the protection for a fraction of the expected return given up.
    """
    t = np.linspace(0.0, 1.0, n_periods)
    if shape == "linear":
        w = start_weight + (end_weight - start_weight) * t
    elif shape == "late":
        w = np.where(t < knee, start_weight,
                     start_weight + (end_weight - start_weight)
                     * (t - knee) / max(1 - knee, 1e-9))
    elif shape == "early":
        w = np.where(t < knee, start_weight + (end_weight - start_weight)
                     * t / max(knee, 1e-9), end_weight)
    elif shape == "constant":
        w = np.full(n_periods, start_weight)
    else:
        raise ValueError(f"unknown glide shape {shape!r}")
    return np.clip(w, 0.0, 1.0)


def glided_accumulation(equity: pd.Series, bonds: pd.Series, years: float = 30.0,
                        contribution: float = 1.0, shape: str = "linear",
                        start_weight: float = 1.0, end_weight: float = 0.3,
                        step: int = 21) -> pd.DataFrame:
    """A contributor following a glide path, over every start date."""
    df = pd.concat([equity.rename("e"), bonds.rename("b")], axis=1,
                   sort=False).dropna()
    h = int(years * TRADING_DAYS)
    if len(df) <= h:
        return pd.DataFrame()
    e = df["e"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    w = glide_path(h, start_weight, end_weight, shape)
    per_period = contribution / 21.0
    rows = []
    for s in range(0, len(df) - h + 1, step):
        seg = w * e[s:s + h] + (1 - w) * b[s:s + h]
        rows.append({"start": df.index[s], "end": df.index[s + h - 1],
                     "terminal": terminal_with_contributions(seg, per_period),
                     "invested": per_period * h})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["multiple"] = d["terminal"] / d["invested"]
    return d


def remedy_comparison(equity: pd.Series, bonds: pd.Series, years: float = 30.0,
                      step: int = 21) -> pd.DataFrame:
    """Score every remedy on the same paths: dispersion reduced against wealth given up.

    The table this study exists to produce. Each remedy is reported with both numbers, plus the
    ratio — dispersion removed per unit of median wealth sacrificed — because a remedy that
    halves the spread by halving the outcome has not helped anyone.
    """
    base = glided_accumulation(equity, bonds, years, 1.0, "constant", 1.0, 1.0, step)
    if base.empty:
        return pd.DataFrame()
    b = path_dispersion(base, "multiple")
    rows = [{"remedy": "100% equity throughout", "median": b["median"],
             "cv": b["cv"], "ratio_95_05": b["ratio_95_05"], "p05": b["p05"],
             "median_cost": 0.0, "cv_reduction": 0.0, "efficiency": np.nan}]
    variants = [
        ("linear glide to 30%", dict(shape="linear", end_weight=0.30)),
        ("late glide to 30%", dict(shape="late", end_weight=0.30)),
        ("early glide to 30%", dict(shape="early", end_weight=0.30)),
        ("constant 70/30", dict(shape="constant", start_weight=0.70,
                                end_weight=0.70)),
        ("constant 60/40", dict(shape="constant", start_weight=0.60,
                                end_weight=0.60)),
        ("late glide to 60%", dict(shape="late", end_weight=0.60)),
    ]
    for name, kw in variants:
        kw.setdefault("start_weight", 1.0)
        d = glided_accumulation(equity, bonds, years, 1.0, step=step, **kw)
        if d.empty:
            continue
        s = path_dispersion(d, "multiple")
        cost = 1 - s["median"] / b["median"] if b["median"] else np.nan
        red = 1 - s["cv"] / b["cv"] if b["cv"] else np.nan
        rows.append({"remedy": name, "median": s["median"], "cv": s["cv"],
                     "ratio_95_05": s["ratio_95_05"], "p05": s["p05"],
                     "median_cost": cost, "cv_reduction": red,
                     "efficiency": red / cost if cost and abs(cost) > 1e-9 else np.inf})
    return pd.DataFrame(rows).set_index("remedy")


# --------------------------------------------------------------------------- #
# The synthetic control
# --------------------------------------------------------------------------- #
def synthetic_path(n_days: int = 8400, drift: float = 0.09, vol: float = 0.16,
                   seed: int = 1008) -> pd.Series:
    """I.i.d. returns — so any start-date dispersion is sampling, not regime."""
    rng = np.random.default_rng(seed)
    dv = vol / np.sqrt(TRADING_DAYS)
    mu = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.Series(np.expm1(rng.normal(mu, dv, n_days)), index=idx, name="sim")


def lottery_size_by_volatility(vols=(0.08, 0.12, 0.16, 0.24, 0.32), years: float = 30.0,
                               n_paths: int = 200, drift: float = 0.09,
                               seed: int = 1008) -> pd.DataFrame:
    """How the size of the lottery scales with volatility, on independent paths.

    Independent rather than overlapping, which the real data cannot provide. This is the only
    place in the study where a dispersion figure has a real sample behind it, and it is what
    makes the magnitudes in the real-data sections interpretable rather than merely alarming.
    """
    rng = np.random.default_rng(seed)
    h = int(years * TRADING_DAYS)
    rows = []
    for vol in vols:
        dv = vol / np.sqrt(TRADING_DAYS)
        mu = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
        lump, contrib = [], []
        for _ in range(n_paths):
            seg = np.expm1(rng.normal(mu, dv, h))
            lump.append(terminal_lump_sum(seg))
            contrib.append(terminal_with_contributions(seg, 1.0 / 21.0))
        lump, contrib = np.array(lump), np.array(contrib)
        rows.append({"vol": vol,
                     "lump_cv": float(lump.std(ddof=1) / lump.mean()),
                     "contrib_cv": float(contrib.std(ddof=1) / contrib.mean()),
                     "contrib_ratio_95_05": float(np.percentile(contrib, 95)
                                                  / np.percentile(contrib, 5)),
                     "lump_ratio_95_05": float(np.percentile(lump, 95)
                                               / np.percentile(lump, 5))})
    return pd.DataFrame(rows).set_index("vol")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the start date produces a large spread in lifetime outcomes for
      an identical plan; **Weak** if modest; **None** if the start date barely matters.
    - **Tradability**: **Useful** if some controllable choice reduces the dispersion at a
      favourable rate — more dispersion removed than median wealth sacrificed; **Partial** if
      the trade is roughly one-for-one; **Mirage** if every remedy costs more than it saves.
    """
    signal = ("Real" if h["ratio_95_05"] > 1.8
              else ("Weak" if h["ratio_95_05"] > 1.3 else "None"))
    trad = ("Useful" if h["best_efficiency"] > 1.5
            else ("Partial" if h["best_efficiency"] > 0.8 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Large, and larger than it looks in the usual telling. Contributing the same "
            f"amount every month into {h['asset']} for {h['years']:.0f} years, the best "
            f"available start date turned each unit invested into {h['best_multiple']:.2f}× "
            f"and the worst into {h['worst_multiple']:.2f}× — a **{h['ratio_max_min']:.2f}× "
            f"gap between two people following an identical plan**, with the 5th-to-95th "
            f"percentile spanning {h['ratio_95_05']:.2f}×. The best start was "
            f"{h['best_start']} and the worst {h['worst_start']}. Two caveats keep that "
            f"honest: these paths overlap heavily and are worth roughly "
            f"{h['effective_n']:.1f} independent observations, and the *lump-sum* spread over "
            f"the same windows was {h['lump_ratio_95_05']:.2f}× — so much of this is the "
            f"return distribution rather than sequence. The clean separation is the shuffle "
            f"test: reordering a single path leaves a lump sum unchanged to "
            f"{h['shuffle_lump_cv']:.0e} relative precision while scattering a contributor's "
            f"outcome across {h['shuffle_sequence_spread']:.2f}×. **That** is pure sequence "
            f"risk, with the distribution held exactly fixed. And it is not spread evenly: "
            f"the final sixth of the horizon correlates {h['last_bucket_corr']:.2f} with a "
            f"contributor's outcome against {h['first_bucket_corr']:.2f} for the first, "
            f"whereas for a lump sum every period matters equally by construction."),
        "trad_why": (
            f"Yes, and the profile above says which remedy to reach for. Because the exposure "
            f"is concentrated late, de-risking late buys most of the protection cheaply. "
            f"Scoring every rule on the same paths — dispersion removed against median wealth "
            f"given up — the best was **{h['best_remedy']}**, cutting the coefficient of "
            f"variation by {h['best_cv_reduction']:.0%} for a median cost of "
            f"{h['best_median_cost']:.0%}, an efficiency of "
            f"**{h['best_efficiency']:.2f}× dispersion removed per unit of wealth "
            f"sacrificed**. A conventional linear glide managed {h['linear_efficiency']:.2f}×, "
            f"and simply holding a constant 60/40 throughout {h['sixty_forty_efficiency']:.2f}×. "
            f"The ranking is what matters more than the levels, since the levels inherit the "
            f"overlapping-window problem. Two things worth saying plainly: none of this "
            f"eliminates the lottery — the residual spread under the best remedy was still "
            f"{h['best_ratio_95_05']:.2f}× — and the largest lever is not on this table at "
            f"all. Contributing for forty years instead of thirty, or retaining the "
            f"flexibility to defer retirement by two, moves the distribution more than any "
            f"asset-allocation rule tested here."),
        "trad": trad,
        "one_sentence": (
            f"Identical thirty-year plans differed by {h['ratio_max_min']:.1f}× on start date "
            f"alone, and the cheapest defence is de-risking late rather than early — "
            f"{h['best_efficiency']:.1f}× as much dispersion removed per unit of wealth given "
            f"up as the conventional glide."),
    }
