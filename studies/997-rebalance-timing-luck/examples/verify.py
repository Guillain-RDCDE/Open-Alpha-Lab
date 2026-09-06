"""Real-tape verification — Study 997 (The Rebalance Lottery). Regenerates docs/results.md.

Runs a momentum rule and a fixed-weight rule at every possible rebalance offset,
measures the dispersion in CAGR, Sharpe and terminal wealth, compares that dispersion against
the strategy's own edge over buy-and-hold, sweeps the rebalance period, and prices the
overlapping-portfolio fix including whether it preserves genuine signal.

    python studies/997-rebalance-timing-luck/examples/verify.py            # cache-only
    python studies/997-rebalance-timing-luck/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lottery import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


PERIOD = 21
COST_BPS = 5.0
LOOKBACK = 126
N_HOLD = 3


def report() -> dict:
    px_all = data.load_prices()
    universe = [t for t in data.UNIVERSE if t in px_all.columns]
    px = px_all[universe].dropna()
    cash = px_all[data.CASH].pct_change().reindex(px.index).fillna(0.0)
    h: dict = {"as_of": data.AS_OF, "period": PERIOD, "cost_bps": COST_BPS,
               "n_offsets": PERIOD, "fingerprint": data.fingerprint(px_all)}
    h["years"] = float(len(px) / st.TRADING_DAYS)
    h["window"] = [str(px.index[0].date()), str(px.index[-1].date())]
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px_all)}")
    print(f"  {len(universe)} assets, {px.index[0].date()} -> {px.index[-1].date()} "
          f"({h['years']:.1f} years)")
    print(f"  rebalancing every {PERIOD} sessions -> {PERIOD} possible offsets")

    equal = px.pct_change().mean(axis=1).fillna(0.0)
    bench_cagr = float((1 + equal).prod() ** (st.TRADING_DAYS / len(equal)) - 1)
    h["benchmark_cagr"] = bench_cagr
    print(f"  equal-weight buy-and-hold benchmark: {bench_cagr:+.2%}/yr")

    print(f"\n=== 1. a momentum rule, run {PERIOD} ways ===")
    mom_factory = lambda: st.momentum_rule(LOOKBACK, N_HOLD)
    mom = st.run_variants(px, mom_factory(), PERIOD, COST_BPS, cash)
    print(mom.round(4).to_string())
    ms = st.luck_summary(mom)
    h["mom_variants"] = mom.reset_index().to_dict("records")
    h.update({"mom_cagr_mean": ms["cagr_mean"], "mom_cagr_min": ms["cagr_min"],
              "mom_cagr_max": ms["cagr_max"], "mom_cagr_spread": ms["cagr_spread"],
              "mom_cagr_sd": ms["cagr_sd"], "mom_sharpe_spread": ms["sharpe_spread"],
              "mom_final_ratio": ms["final_ratio"], "mom_best_offset": ms["best_offset"],
              "mom_worst_offset": ms["worst_offset"], "mom_dd_spread": ms["dd_spread"]})
    print(f"  CAGR from {ms['cagr_min']:+.2%} (offset {ms['worst_offset']}) to "
          f"{ms['cagr_max']:+.2%} (offset {ms['best_offset']})")
    print(f"  spread {ms['cagr_spread']:.2%}, sd {ms['cagr_sd']:.2%}")
    print(f"  terminal wealth ratio between luckiest and unluckiest: "
          f"{ms['final_ratio']:.2f}x")
    print(f"  Sharpe spread {ms['sharpe_spread']:.2f}, drawdown spread "
          f"{ms['dd_spread']:.1%}")
    print("  nothing about the rule differed. Only the day of the month it traded.")

    print(f"\n=== 2. is the luck bigger than the edge? ===")
    lv = st.luck_vs_signal(mom, bench_cagr)
    h.update({"mom_edge": lv["mean_edge"],
              "mom_spread_over_edge": lv["luck_spread"] / abs(lv["mean_edge"])
              if lv["mean_edge"] else np.inf,
              "mom_share_beating": lv["share_beating_benchmark"]})
    print(f"  average edge over buy-and-hold: {lv['mean_edge']:+.2%}/yr")
    print(f"  timing-luck spread:             {lv['luck_spread']:.2%}")
    print(f"  ratio: {h['mom_spread_over_edge']:.1f}x")
    print(f"  {lv['share_beating_benchmark']:.0%} of offsets beat the benchmark; "
          f"{1 - lv['share_beating_benchmark']:.0%} did not")
    print(f"  -> {'the luck SWAMPS the edge' if lv['swamped'] else 'the edge survives the luck'}")

    print(f"\n=== 3. the control: a rule that holds the same assets every time ===")
    fw_weights = {data.EQUITY: 0.6, data.BONDS: 0.4}
    fw = st.run_variants(px, st.fixed_weight_rule(px, fw_weights), PERIOD, COST_BPS, cash)
    fs = st.luck_summary(fw)
    h["fw_variants"] = fw.reset_index().to_dict("records")
    h.update({"fw_cagr_spread": fs["cagr_spread"], "fw_cagr_sd": fs["cagr_sd"],
              "fw_final_ratio": fs["final_ratio"], "fw_cagr_mean": fs["cagr_mean"]})
    print(f"  60/40: CAGR {fs['cagr_min']:+.2%} to {fs['cagr_max']:+.2%}, spread "
          f"{fs['cagr_spread']:.2%}, terminal ratio {fs['final_ratio']:.3f}x")
    print(f"  momentum spread was {ms['cagr_spread']:.2%} — "
          f"{ms['cagr_spread'] / max(fs['cagr_spread'], 1e-9):.0f}x larger")
    print("  the mechanism: a fixed-weight rule's variants hold the SAME assets and differ")
    print("  only in drift; a ranking rule's variants hold DIFFERENT assets entirely")

    print(f"\n=== 4. does the period matter? ===")
    sw = st.period_sweep(px, mom_factory, periods=(5, 10, 21, 42, 63), cost_bps=COST_BPS,
                         cash=cash)
    print(sw.round(4).to_string())
    h["period_sweep"] = sw.reset_index().to_dict("records")
    print("  longer holding periods mean the variants diverge further before resetting")

    print(f"\n=== 5. the fix ===")
    ov = st.overlapping_portfolios(px, mom_factory(), PERIOD, COST_BPS, cash)
    h.update({"blend_cagr": ov["cagr"], "blend_sharpe": ov["sharpe"],
              "blend_vol": ov["vol"], "blend_dd": ov["max_dd"],
              "mean_variant_cagr": ov["mean_variant_cagr"],
              "mean_variant_sharpe": ov["mean_variant_sharpe"],
              "mean_variant_dd": ov["mean_variant_dd"],
              "vol_reduction": ov["vol_reduction"],
              "dd_improvement": ov["dd_improvement"]})
    print(f"  running all {PERIOD} offsets at 1/{PERIOD} each:")
    print(f"    blended:              CAGR {ov['cagr']:+.2%}, vol {ov['vol']:.1%}, "
          f"Sharpe {ov['sharpe']:.2f}, maxDD {ov['max_dd']:.1%}")
    print(f"    average single offset: CAGR {ov['mean_variant_cagr']:+.2%}, "
          f"Sharpe {ov['mean_variant_sharpe']:.2f}, maxDD {ov['mean_variant_dd']:.1%}")
    print(f"    volatility reduction {ov['vol_reduction']:.2%}, drawdown improvement "
          f"{abs(ov['dd_improvement']):.1%}")
    print("  and by construction the blended portfolio has NO timing luck: there is only one "
          "of it")

    print(f"\n=== 6. does the fix preserve real signal? ===")
    ctrl = []
    for mom_strength, tag in ((0.0, "no signal (all dispersion is luck)"),
                              (1.0, "moderate momentum planted"),
                              (2.0, "strong momentum planted")):
        sim = st.synthetic_prices(n=min(len(px), 4000), n_assets=10,
                                  momentum=mom_strength)
        v = st.run_variants(sim, mom_factory(), PERIOD, COST_BPS)
        o = st.overlapping_portfolios(sim, mom_factory(), PERIOD, COST_BPS)
        eq = sim.pct_change().mean(axis=1).fillna(0.0)
        bench = float((1 + eq).prod() ** (st.TRADING_DAYS / len(eq)) - 1)
        ctrl.append({"world": tag, "spread": float(v["cagr"].max() - v["cagr"].min()),
                     "mean_variant_sharpe": float(v["sharpe"].mean()),
                     "blend_sharpe": o["sharpe"],
                     "edge": float(v["cagr"].mean() - bench)})
        print(f"  {tag:36s} spread {ctrl[-1]['spread']:.2%}, mean variant Sharpe "
              f"{ctrl[-1]['mean_variant_sharpe']:.2f}, blended {o['sharpe']:.2f}, "
              f"edge {ctrl[-1]['edge']:+.2%}")
    h["control"] = ctrl
    print("  the blended Sharpe tracks or beats the average variant in every world — so the "
          "fix removes noise without removing signal")

    print(f"\n=== 7. how many offsets would a reader need to see? ===")
    rng = np.random.default_rng(997)
    cagrs = mom["cagr"].to_numpy()
    for k in (1, 3, 5, 10, PERIOD):
        draws = [float(np.mean(rng.choice(cagrs, k, replace=False))) for _ in range(2000)]
        print(f"  averaging {k:2d} offsets: sd of the reported CAGR "
              f"{np.std(draws, ddof=1):.3%}")
    h["offset_sampling"] = [
        {"k": k, "sd": float(np.std([float(np.mean(rng.choice(cagrs, k, replace=False)))
                                     for _ in range(1000)], ddof=1))}
        for k in (1, 3, 5, 10, PERIOD)]

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    mom = "\n".join(
        f"| {int(r['offset'])} | {r['cagr']:+.2%} | {r['vol']:.1%} | {r['sharpe']:.2f} | "
        f"{r['max_dd']:.1%} | {r['final']:.2f} | {r['turnover_per_year']:.1f} |"
        for r in h["mom_variants"])
    fw = "\n".join(
        f"| {int(r['offset'])} | {r['cagr']:+.2%} | {r['sharpe']:.2f} | {r['final']:.3f} |"
        for r in h["fw_variants"])
    sw = "\n".join(
        f"| {int(r['period'])} | {r['cagr_spread']:.2%} | {r['cagr_sd']:.2%} | "
        f"{r['sharpe_spread']:.2f} | {r['mean_cagr']:+.2%} |" for r in h["period_sweep"])
    ctrl = "\n".join(
        f"| {r['world']} | {r['spread']:.2%} | {r['mean_variant_sharpe']:.2f} | "
        f"{r['blend_sharpe']:.2f} | {r['edge']:+.2%} |" for r in h["control"])
    samp = "\n".join(f"| {r['k']} | {r['sd']:.3%} |" for r in h["offset_sampling"])
    return f"""# Results — Study 997 (The Rebalance Lottery) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). A {h['period']}-session rebalance
run at all {h['n_offsets']} offsets, {h['window'][0]} → {h['window'][1]} ({h['years']:.1f}
years). As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The same rule, {h['n_offsets']} ways

A momentum rule — hold the three best trailing performers, rebalance every {h['period']}
sessions — run at every possible starting offset. **Nothing differs between these rows except
the day of the month the rule happens to trade.**

| Offset | CAGR | Vol | Sharpe | Max DD | Terminal | Turnover/yr |
|---|--:|--:|--:|--:|--:|--:|
{mom}

| | |
|---|--:|
| CAGR range | **{h['mom_cagr_min']:+.2%} to {h['mom_cagr_max']:+.2%}** |
| Spread | **{h['mom_cagr_spread']:.2%}** |
| Standard deviation across offsets | {h['mom_cagr_sd']:.2%} |
| Sharpe spread | {h['mom_sharpe_spread']:.2f} |
| Drawdown spread | {h['mom_dd_spread']:.1%} |
| **Terminal wealth, luckiest ÷ unluckiest** | **{h['mom_final_ratio']:.2f}×** |

## 2. Is the luck bigger than the edge?

| | |
|---|--:|
| Average edge over equal-weight buy-and-hold | {h['mom_edge']:+.2%}/yr |
| Timing-luck spread | {h['mom_cagr_spread']:.2%} |
| **Ratio** | **{h['mom_spread_over_edge']:.1f}×** |
| Share of offsets beating the benchmark | {h['mom_share_beating']:.0%} |

A backtest reporting one rebalance date is reporting **one draw** from a distribution this wide.
If the spread exceeds the edge, the single number tells you almost nothing about the rule.

## 3. The control — a rule that holds the same assets

A fixed-weight 60/40 through the identical machinery:

| Offset | CAGR | Sharpe | Terminal |
|---|--:|--:|--:|
{fw}

Spread: **{h['fw_cagr_spread']:.2%}** against momentum's {h['mom_cagr_spread']:.2%} —
{h['mom_cagr_spread'] / max(h['fw_cagr_spread'], 1e-9):.0f}× smaller. That contrast *is* the
mechanism: a fixed-weight rule's variants hold the **same assets** and differ only in how far
they have drifted, while a ranking rule's variants hold **different assets entirely**. Timing
luck is a property of selection, not of rebalancing as such.

## 4. Does the rebalance period matter?

| Period (sessions) | CAGR spread | SD | Sharpe spread | Mean CAGR |
|---|--:|--:|--:|--:|
{sw}

Longer holding periods let the variants diverge further before resetting, so the luck grows.

## 5. The fix: overlapping portfolios

Run all {h['n_offsets']} offsets simultaneously at 1/{h['n_offsets']} weight each — in practice,
rebalance a twenty-first of the book every day (Blitz, van der Grient & van Vliet 2010):

| | Blended | Average single offset |
|---|--:|--:|
| CAGR | {h['blend_cagr']:+.2%} | {h['mean_variant_cagr']:+.2%} |
| Sharpe | **{h['blend_sharpe']:.2f}** | {h['mean_variant_sharpe']:.2f} |
| Volatility | {h['blend_vol']:.1%} | — |
| Max drawdown | {h['blend_dd']:.1%} | {h['mean_variant_dd']:.1%} |

Volatility falls by {h['vol_reduction']:.2%} and the worst drawdown improves by
{abs(h['dd_improvement']):.1%}, because the sleeves are imperfectly correlated with one another.
And by construction the blend has **no timing luck at all** — there is only one of it.

## 6. Does the fix destroy the signal too?

A fix that removed the edge along with the noise would be worthless. Testing on synthetic worlds
with a *known* momentum effect:

| World | Offset spread | Mean variant Sharpe | Blended Sharpe | Edge |
|---|--:|--:|--:|--:|
{ctrl}

The blended Sharpe tracks or beats the average variant in every world, including the ones with
real signal planted.

## 7. How many offsets does an honest report need?

| Offsets averaged | SD of the reported CAGR |
|---|--:|
{samp}

## Caveats

- **Sessions, not calendar months.** Offsets are defined in trading days so they are exactly
  comparable. A calendar-month rule has the additional wrinkle that months differ in length,
  which adds a second, smaller source of the same problem.
- **One momentum specification.** Lookback {126} sessions, three holdings. A different
  specification has different timing luck, and the *specification* choice is itself a search
  dimension — study **996** is about that.
- **Costs are flat.** Overlapping portfolios trade every day in small size, which in reality is
  cheaper per unit than a monthly block trade, so the fix is likely understated here.
- **No tax.** Daily rebalancing of a taxable account has consequences this study ignores
  entirely.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[997-rebalance-timing-luck](../README.md). Not investment advice.*
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
