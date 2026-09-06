"""Real-tape verification — Study 1002 (The Ten Best Days). Regenerates docs/results.md.

Rebuilds the brochure table on the real tape, prints the symmetric statistic
beside it, measures how far the best days sit from the worst against a shuffled benchmark that
keeps every return, places both sets in their volatility and drawdown context, contrasts the
cost of missing days chosen at random with the cost of missing the specifically best ones, and
computes the timing accuracy needed to break even.

    python studies/1002-best-days-missed/examples/verify.py            # cache-only
    python studies/1002-best-days-missed/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bestdays import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


N_EXTREME = 10
CASH_RATE = 0.0
OUT_FRACTION = 0.20


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "n_extreme": N_EXTREME, "asset": data.EQUITY,
               "fingerprint": data.fingerprint(px)}

    r = px[data.EQUITY].dropna().pct_change().dropna()
    h["n_days"] = int(len(r))
    h["years"] = float(len(r) / 252)
    h["start"] = str(r.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {data.EQUITY}: {len(r):,} sessions from {h['start']} "
          f"({h['years']:.1f} years)")

    print("\n=== 1. the brochure table, and the column it omits ===")
    t = st.missed_days_table(r, counts=(0, 5, 10, 20, 30, 50, 100), cash_rate=CASH_RATE)
    print(t[["miss_best_cagr", "miss_worst_cagr", "miss_both_cagr"]].round(4).to_string())
    h["table"] = t.reset_index().to_dict("records")
    a = st.asymmetry(r, N_EXTREME)
    h.update({"base_cagr": a["base_cagr"],
              "cost_of_missing_best": a["cost_of_missing_best"],
              "benefit_of_missing_worst": a["benefit_of_missing_worst"],
              "asym_ratio": a["ratio"], "mean_best": a["mean_best"],
              "mean_worst": a["mean_worst"], "log_best": a["log_best"],
              "log_worst": a["log_worst"],
              "best_bigger_in_percent": a["best_bigger_in_percent"]})
    print(f"  buy and hold:            {a['base_cagr']:.2%} a year")
    print(f"  miss the {N_EXTREME} best:        "
          f"{a['base_cagr'] - a['cost_of_missing_best']:.2%}  "
          f"(-{a['cost_of_missing_best']:.2%})   <- the brochure prints this")
    print(f"  miss the {N_EXTREME} worst:       "
          f"{a['base_cagr'] + a['benefit_of_missing_worst']:.2%}  "
          f"(+{a['benefit_of_missing_worst']:.2%})   <- and not this")
    print(f"  the omitted half is {a['ratio']:.2f}x the size of the quoted one")
    print(f"  WHY? Not because crashes are bigger. In percent the best days are LARGER here:")
    print(f"    best {a['mean_best']:+.2%}  vs  worst {a['mean_worst']:.2%}")
    print(f"  Compounding multiplies by 1/(1+x), so the scale that matters is log(1+x):")
    print(f"    best {a['log_best']:+.4f}  vs  worst {a['log_worst']:+.4f}   <- worst is bigger")
    both = t.loc[N_EXTREME, "miss_both_cagr"] if N_EXTREME in t.index else np.nan
    h["miss_both_cagr"] = float(both)
    print(f"  miss both sets:          {both:.2%} — close to buy-and-hold, which is the "
          f"real shape of the thing")

    print("\n=== 2. where the extreme days actually are ===")
    ex = st.extreme_days(r, N_EXTREME)
    h["extreme_days"] = [{"date": str(d.date()), "ret": v, "kind": k}
                         for d, v, k in zip(ex["date"], ex["ret"], ex["kind"])]
    for _, row in ex.iterrows():
        print(f"  {row['date'].date()}  {row['ret']:+7.2%}  {row['kind']}")
    years = pd.Series([d.year for d in ex["date"]]).value_counts().sort_index()
    h["extreme_years"] = {int(k): int(v) for k, v in years.items()}
    print(f"  those {len(ex)} days fall in only {len(years)} calendar years: "
          f"{', '.join(str(int(y)) for y in years.index)}")

    print("\n=== 3. are the best days near the worst days? ===")
    c = st.clustering_stats(r, N_EXTREME, n_shuffles=600)
    h.update({"median_gap": c["median_gap"], "shuffled_gap": c["shuffled_median_gap"],
              "cluster_p": c["p_value"], "cluster_ratio": c["ratio"]})
    print(f"  median sessions from a best day to the nearest worst day: "
          f"{c['median_gap']:.0f}")
    print(f"  the same returns shuffled:                                "
          f"{c['shuffled_median_gap']:.0f}  (5th pct {c['shuffled_p05']:.0f})")
    print(f"  p = {c['p_value']:.4f}")
    print("  the shuffle keeps every return and every fat tail — only the ORDER changes.")
    print("  So this is clustering, not a property of the distribution.")

    print("\n=== 4. what the market looked like around them ===")
    v = st.volatility_context(r, N_EXTREME)
    d = st.drawdown_context(r, N_EXTREME)
    h.update({"best_vol_ratio": v["best_vol_ratio"], "worst_vol_ratio": v["worst_vol_ratio"],
              "best_drawdown": d["best_median_drawdown"],
              "worst_drawdown": d["worst_median_drawdown"],
              "typical_drawdown": d["typical_drawdown"]})
    print(f"  trailing volatility on a typical day: {v['typical_vol']:.1%}")
    print(f"    around the best days:  {v['best_vol']:.1%}  "
          f"({v['best_vol_ratio']:.1f}x normal)")
    print(f"    around the worst days: {v['worst_vol']:.1%}  "
          f"({v['worst_vol_ratio']:.1f}x normal)")
    print(f"  median drawdown at the time: best days {d['best_median_drawdown']:.1%}, "
          f"worst days {d['worst_median_drawdown']:.1%}, typical day "
          f"{d['typical_drawdown']:.1%}")
    print("  the best days happen part-way down a crash, not at the peak")

    print("\n=== 5. missing days at random — the correct null ===")
    oc = st.out_of_market_cost(r, fractions=(0.001, 0.005, 0.01, 0.05, 0.10),
                               n_draws=400, cash_rate=CASH_RATE)
    print(oc.round(4).to_string())
    h["random_cost_table"] = oc.reset_index().to_dict("records")
    row = oc.loc[0.01]
    h.update({"random_fraction": 0.01, "random_days": int(row["days"]),
              "random_cost": float(row["random_cost"]),
              "worst_case_cost": float(row["worst_case_cost"])})
    print(f"  missing {int(row['days']):,} days at random costs "
          f"{row['random_cost']:.2%} a year")
    print(f"  missing the {int(row['days']):,} BEST days costs "
          f"{row['worst_case_cost']:.2%} a year")
    print(f"  ratio: {row['worst_case_cost'] / max(row['random_cost'], 1e-9):.0f}x. The "
          f"brochure quotes the second and implies the first.")

    print("\n=== 6. how accurate would a timer need to be? ===")
    rate = st.down_day_share(r)
    grid = np.round(np.concatenate([[0.35, 0.40, rate], np.arange(0.50, 1.01, 0.10)]), 4)
    f = st.timing_frontier(r, OUT_FRACTION, sorted(set(grid)), n_draws=400,
                           cash_rate=CASH_RATE)
    print(f.round(4).to_string())
    h["frontier"] = f.reset_index().to_dict("records")
    h["out_fraction"] = OUT_FRACTION
    h["days_out"] = int(f["days_out"].iloc[0])
    h["coin_flip_rate"] = float(rate)
    h["coin_flip_cagr"] = float(f.loc[round(rate, 4), "median_cagr"])
    h["breakeven_hit_rate"] = st.breakeven_hit_rate(r, OUT_FRACTION, CASH_RATE)
    h["timing_edge_needed"] = float(h["breakeven_hit_rate"] - rate)
    below = st.timing_frontier(r, OUT_FRACTION, (rate - 0.05,), n_draws=300,
                               cash_rate=CASH_RATE)
    h["below_gap"] = 5.0
    h["below_cagr"] = float(below.iloc[0]["median_cagr"])
    print(f"  a timer sitting out {OUT_FRACTION:.0%} of sessions = {h['days_out']:,} days")
    print(f"  picking those days AT RANDOM already gets {rate:.2%} of them right, because")
    print(f"  that is simply the down-day frequency. 45%, not 50%, is the coin flip.")
    print(f"  random selection returns {h['coin_flip_cagr']:.2%} vs buy-and-hold "
          f"{a['base_cagr']:.2%} — it loses the premium it sat out")
    print(f"  break-even hit rate: {h['breakeven_hit_rate']:.2%}")
    print(f"  required EDGE over random: {h['timing_edge_needed'] * 100:.1f} percentage points")
    print(f"  and 5 points BELOW random returns {h['below_cagr']:.2%} — the frontier is steep,")
    print(f"  so a small required edge is not the same as an easy one")

    print("\n=== 7. every market ===")
    cross = []
    for tk in data.TICKERS:
        if tk in (data.CASH,) or tk not in px.columns:
            continue
        s = px[tk].dropna().pct_change().dropna()
        if len(s) < 1500:
            continue
        aa = st.asymmetry(s, N_EXTREME)
        cc = st.clustering_stats(s, N_EXTREME, 300)
        be = st.breakeven_hit_rate(s, OUT_FRACTION, CASH_RATE)
        rr = st.down_day_share(s)
        cross.append({"asset": tk, "n": int(len(s)), "cagr": aa["base_cagr"],
                      "cost_best": aa["cost_of_missing_best"],
                      "gain_worst": aa["benefit_of_missing_worst"],
                      "ratio": aa["ratio"], "median_gap": cc["median_gap"],
                      "shuffled_gap": cc["shuffled_median_gap"], "breakeven": be,
                      "random_rate": rr, "edge": be - rr})
        print(f"  {tk:6s} cagr {aa['base_cagr']:6.2%}  miss-best -{aa['cost_of_missing_best']:.2%}"
              f"  miss-worst +{aa['benefit_of_missing_worst']:.2%}  ratio {aa['ratio']:.2f}"
              f"  gap {cc['median_gap']:.0f} vs {cc['shuffled_median_gap']:.0f}"
              f"  breakeven {be:.1%} (random {rr:.1%}, edge {(be - rr) * 100:+.1f}pp)")
    h["cross_asset"] = cross

    print("\n=== 8. does clustering cause the proximity? (synthetic control) ===")
    ctrl = []
    for clustered in (True, False):
        gaps, ratios = [], []
        for k in range(6):
            sim = st.synthetic_returns(n=8000, clustered=clustered, seed=1002 + k)
            cc = st.clustering_stats(sim, N_EXTREME, 200, seed=1002 + k)
            gaps.append(cc["median_gap"])
            ratios.append(cc["ratio"])
        ctrl.append({"clustered": clustered, "median_gap": float(np.mean(gaps)),
                     "gap_ratio_vs_shuffle": float(np.mean(ratios))})
        print(f"  volatility clustering {'ON ' if clustered else 'OFF'}: median gap "
              f"{np.mean(gaps):7.1f} sessions, {np.mean(ratios):.3f}x its own shuffle")
    h["clustering_control"] = ctrl
    print("  the fat tail is present in both worlds. Only clustering brings the best and")
    print("  worst days together, which is what the real tape shows.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    tbl = "\n".join(
        f"| {int(r['days'])} | {r['miss_best_cagr']:.2%} | {r['miss_worst_cagr']:.2%} | "
        f"{r['miss_both_cagr']:.2%} |" for r in h["table"])
    ex = "\n".join(f"| {r['date']} | {r['ret']:+.2%} | {r['kind']} |"
                   for r in h["extreme_days"])
    oc = "\n".join(
        f"| {r['fraction']:.1%} | {int(r['days']):,} | {r['random_cost']:.2%} | "
        f"{r['worst_case_cost']:.2%} | {r['worst_case_cost'] / max(r['random_cost'], 1e-9):.0f}× |"
        for r in h["random_cost_table"])
    fr = "\n".join(
        f"| {r['hit_rate']:.2%}{' ← random' if abs(r['hit_rate'] - h['coin_flip_rate']) < 1e-6 else ''} "
        f"| {r['median_cagr']:.2%} | {r['p10']:.2%} | "
        f"{r['p90']:.2%} | {r['beats_hold']:.0%} |" for r in h["frontier"])
    cross = "\n".join(
        f"| {r['asset']} | {r['n']:,} | {r['cagr']:.2%} | −{r['cost_best']:.2%} | "
        f"+{r['gain_worst']:.2%} | **{r['ratio']:.2f}** | {r['median_gap']:.0f} | "
        f"{r['shuffled_gap']:.0f} | {r['random_rate']:.1%} | {r['breakeven']:.1%} | "
        f"**{r['edge'] * 100:+.1f}pp** |" for r in h["cross_asset"])
    ctrl = "\n".join(
        f"| {'yes' if r['clustered'] else 'no'} | {r['median_gap']:.1f} | "
        f"{r['gap_ratio_vs_shuffle']:.3f} |" for r in h["clustering_control"])
    return f"""# Results — Study 1002 (The Ten Best Days) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['asset']}, {h['n_days']:,}
sessions from {h['start']} ({h['years']:.1f} years). As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## 1. The brochure table, and the column it omits

| Days missed | Miss the best | Miss the worst | Miss both |
|---|--:|--:|--:|
{tbl}

Buy and hold returned **{h['base_cagr']:.2%}** a year. Missing the ten best sessions takes that
to {h['base_cagr'] - h['cost_of_missing_best']:.2%} — the brochure's number, and it is correct.
Missing the ten **worst** takes it to
{h['base_cagr'] + h['benefit_of_missing_worst']:.2%}: a gain of
**{h['benefit_of_missing_worst']:.2%}**, which is **{h['asym_ratio']:.2f}×** the quoted loss.

**Why, and it is not the obvious reason.** The natural guess is that crashes are bigger than
rallies. Here that is false: in percentage terms the ten best days average
{h['mean_best']:+.2%} and the ten worst {h['mean_worst']:.2%}, so the *best* days are the
larger ones. The asymmetry survives because compounding is multiplicative — removing a day
multiplies the result by 1/(1+*x*), so the operative scale is log(1+*x*). On that scale the
worst days are larger: {h['log_worst']:+.4f} against {h['log_best']:+.4f}. The asymmetry is a
property of compounding, not of crashes, which also makes it more robust than it first appears.

Missing **both** sets returns {h['miss_both_cagr']:.2%} — near buy-and-hold. The extremes
largely cancel, which is the honest shape of the phenomenon.

## 2. Where those days actually are

| Date | Return | |
|---|--:|---|
{ex}

Twenty days, falling in only {len(h['extreme_years'])} calendar years.

## 3. They are neighbours, and that is not a coincidence

| | Sessions |
|---|--:|
| Median distance, best day to nearest worst day | **{h['median_gap']:.0f}** |
| The same returns, shuffled | {h['shuffled_gap']:.0f} |
| p-value | {h['cluster_p']:.4f} |

The shuffle preserves **every return** — the fat tail is untouched — and changes only the order.
So the proximity is clustering in time, not a property of the distribution. The scenario the
brochure asks you to fear, being absent for all ten best days while present for all ten worst,
requires the two to be separable. They are {h['median_gap']:.0f} sessions apart.

## 4. What the market looked like

Volatility around the best days ran {h['best_vol_ratio']:.1f}× normal, and around the worst
{h['worst_vol_ratio']:.1f}×. The best days occurred at a median drawdown of
**{h['best_drawdown']:.1%}** — part-way down a crash, not at a peak. That is a real argument for
staying invested, and a different one from the brochure's: the good days arrive when an investor
who has just capitulated is least likely to be there.

## 5. Missing days at random — the correct null

| Fraction missed | Days | Cost if random | Cost if the *best* | Ratio |
|---|--:|--:|--:|--:|
{oc}

Missing {h['random_days']:,} days chosen at random — {h['random_fraction']:.0%} of the whole
history — costs {h['random_cost']:.2%} a year. Missing the {h['random_days']:,} *best* costs
{h['worst_case_cost']:.2%}. The brochure quotes the second while implying the first is what
happens to people who step out of the market.

## 6. The accuracy a timer would actually need

A rule that sits out {h['out_fraction']:.0%} of sessions — {h['days_out']:,} days — scored on
the share of those days that turn out to be down days.

**The benchmark is {h['coin_flip_rate']:.2%}, not 50%.** That is the unconditional down-day
frequency, so a timer choosing at random already achieves it. An earlier version of this study
defined a "hit" as landing on one of the forty worst days in thirty-three years and produced a
break-even near 1% — a finding about the definition, not about market timing. Hitting the forty
worst days of a lifetime is not something a coin flip has any access to.

| Hit rate | Median CAGR | 10th pct | 90th pct | Beats hold |
|---|--:|--:|--:|--:|
{fr}

Break-even arrives at **{h['breakeven_hit_rate']:.2%}** — an edge of
**{h['timing_edge_needed'] * 100:.1f} percentage points** over random selection. Small, and that
is the trap. Random selection returns {h['coin_flip_cagr']:.2%} against buy-and-hold's
{h['base_cagr']:.2%}, because sitting out {h['out_fraction']:.0%} of sessions forfeits that much
of the equity premium; five points *below* random returns {h['below_cagr']:.2%}. The edge has to
be small, positive and sustained across every one of {h['days_out']:,} decisions, and being
slightly wrong costs far more than being slightly right pays.

## 7. Every market

| Asset | Sessions | CAGR | Miss best | Miss worst | Ratio | Gap | Shuffled | Random | Break-even | Edge |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{cross}

## 8. Does volatility clustering cause the proximity?

| Volatility clustering | Median gap | Ratio vs its own shuffle |
|---|--:|--:|
{ctrl}

Both synthetic worlds have the same fat tail. Only the clustered one brings the best and worst
days together — which is what the real tape shows, and it identifies the mechanism.

## Caveats

- **Cash is assumed to earn nothing** in the headline (`CASH_RATE = 0`). A real investor out of
  the market earns something, which makes the brochure's number *more* overstated, not less;
  the table in `missed_days_table` takes a rate if you want to see it.
- **No taxes or costs** on the switching strategies in section 6. Including them raises the
  break-even hit rate further — again in the direction of the conclusion.
- **The break-even is a hit rate on sit-out days, not a forecast accuracy.** A rule with 60%
  directional accuracy does not have a 60% hit rate in this sense; the mapping between the two
  depends on how often it trades.
- **Survivorship in the index itself.** All of these are index products, so the constituent
  survivorship is the index provider's, not this study's — but it is not zero.
- **The asymmetry ratio is sample-dependent.** It exceeds one in every market tested here, but a
  sample without a 2008 would look different. The clustering result is the more robust of the
  two findings.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1002-best-days-missed](../README.md). Not investment advice.*
"""

def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(h, default=float))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
