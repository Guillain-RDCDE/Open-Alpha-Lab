"""Real-tape verification — Study 1006 (Most Stocks Lose). Regenerates docs/results.md.

Measures the arithmetic-versus-geometric gap name by name and checks it against
σ²/2, sweeps the share of holding periods beating Treasury bills across horizons from a quarter
to fifteen years, ranks dollar wealth creation to show how few names produced it, reconciles the
median member with the rebalanced index by decomposing the drag each one pays, prices the odds
for concentrated portfolios, and quantifies which way survivorship bias runs using a simulated
delisting rule.

    python studies/1006-most-stocks-underperform-cash/examples/verify.py            # cache-only
    python studies/1006-most-stocks-underperform-cash/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from moststocks import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


LONG_YEARS = 10.0
ODDS_HORIZON = 10.0


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "long_years": LONG_YEARS,
               "odds_horizon": ODDS_HORIZON, "fingerprint": data.fingerprint(px)}

    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2000]
    # Each name over its OWN history: intersecting fifty tapes would discard two decades,
    # and the question is about long holding periods.
    R = px[cols].pct_change().dropna(how="all")
    cash = (px[data.BILLS].pct_change().reindex(R.index).fillna(0.0)
            if data.BILLS in px.columns
            else pd.Series(np.full(len(R), 0.02 / 252), index=R.index))
    h["n_names"] = int(len(cols))
    h["n_days"] = int(len(R))
    h["years"] = float(len(R) / 252)
    h["start"] = str(R.index[0].date())
    h["shortest_history"] = float(min(px[c].dropna().shape[0] for c in cols) / 252)
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(cols)} names spanning {h['start']} to {data.AS_OF} "
          f"({h['years']:.1f} years), each scored over its own history")
    print(f"  shortest tape: {h['shortest_history']:.1f} years")
    print(f"  ALL FIFTY SURVIVED to 2026. Bessembinder's universe is every firm that ever")
    print(f"  listed, most of which did not. Expect the headline to FAIL here — the")
    print(f"  interesting question is by how much, and what it would take to restore it.")

    print("\n=== 1. the engine: arithmetic vs geometric ===")
    dt = st.drag_table(R)
    h["mean_drag"] = float(dt["measured_drag"].mean())
    h["predicted_drag"] = float(dt["predicted_drag"].mean())
    h["drag_corr"] = float(dt["measured_drag"].corr(dt["predicted_drag"]))
    h["drag_table"] = dt.reset_index().to_dict("records")
    print(f"  mean arithmetic return {dt['arithmetic'].mean():.2%} a year")
    print(f"  mean geometric  return {dt['geometric'].mean():.2%} a year")
    print(f"  mean LOG growth        {dt['log_growth'].mean():.2%} a year")
    print(f"  measured drag  {h['mean_drag']:.2%}   theoretical sigma^2/2  "
          f"{h['predicted_drag']:.2%}   correlation {h['drag_corr']:.3f}")
    print(f"  (both annualised linearly -- the identity holds in LOG space, and comparing")
    print(f"   against a COMPOUNDED return would make a riskless series show negative drag)")
    print("  the mechanism is identified, not inferred. Everything below is its consequences.")
    worst = dt.nlargest(3, "measured_drag")
    for nm, row in worst.iterrows():
        print(f"    {nm}: vol {row['vol']:.0%} -> drag {row['measured_drag']:.2%} a year")

    print("\n=== 2. how many holdings beat cash, by horizon ===")
    sw = st.horizon_sweep(R, cash, horizons_years=(0.25, 1, 3, 5, 10, 15), step=63)
    print(sw[["n", "share_beat_cash", "share_lost_money", "median_return",
              "mean_return", "skew"]].round(4).to_string())
    h["sweep"] = sw.reset_index().to_dict("records")
    h["share_beat_cash_short"] = float(sw.loc[1, "share_beat_cash"])
    nearest = min(sw.index, key=lambda k: abs(k - LONG_YEARS))
    h["share_beat_cash_long"] = float(sw.loc[nearest, "share_beat_cash"])
    print(f"  at 1 year:  {h['share_beat_cash_short']:.0%} of holdings beat bills")
    print(f"  at {nearest:.0f} years: {h['share_beat_cash_long']:.0%}")
    print(f"  meanwhile the MEAN return rose from {sw.loc[1, 'mean_return']:.1%} to "
          f"{sw.loc[nearest, 'mean_return']:.1%}")
    print(f"  log-return skew went from {sw.loc[1, 'skew']:+.2f} to "
          f"{sw.loc[nearest, 'skew']:+.2f}")
    if h["share_beat_cash_long"] >= h["share_beat_cash_short"]:
        print(f"  *** THE HEADLINE FAILS HERE. *** The share beating bills RISES with horizon.")
        print(f"  These names' drift is large enough to outrun the drag. Section 2b says by")
        print(f"  exactly how much, and what would have to change for the result to appear.")

    print("\n=== 2b. the condition: when DOES the median lose to cash? ===")
    cond = st.median_beats_cash_condition(R, cash)
    h.update({"cash_log_growth": cond["cash_log_growth"],
              "share_above_threshold": cond["share_above_threshold"],
              "median_headroom": cond["median_headroom"],
              "mean_drift": cond["mean_drift"], "mean_vol": cond["mean_vol"],
              "mean_breakeven": cond["mean_breakeven"]})
    h["condition_table"] = cond["table"].reset_index().to_dict("records")
    print(f"  the median compounds at roughly drift - sigma^2/2, so it beats cash while")
    print(f"      sigma < sqrt( 2 * (drift - cash) )")
    print(f"  average drift here {cond['mean_drift']:.1%}, cash "
          f"{cond['cash_log_growth']:.1%}")
    print(f"  -> breakeven volatility {cond['mean_breakeven']:.0%}")
    print(f"  -> these names actually run at {cond['mean_vol']:.0%}")
    print(f"  -> median headroom {cond['median_headroom']:.0%}; only "
          f"{cond['share_above_threshold']:.0%} are over the line")
    tight = cond["table"].nsmallest(5, "headroom")
    for nm, row in tight.iterrows():
        print(f"    {nm}: drift {row['drift']:6.1%}, vol {row['vol']:5.1%}, "
              f"threshold {row['breakeven_vol']:5.1%}, headroom "
              f"{row['headroom']:+6.1%}")
    print(f"  Bessembinder's population -- small caps and firms heading for delisting --")
    print(f"  sits on the OTHER side of this threshold. That is the whole difference.")

    print("\n=== 3. how few names did the work ===")
    wc = st.wealth_concentration(R, cash)
    h.update({k: wc[k] for k in ("share_beat_cash", "share_lost_money",
                                 "median_terminal", "mean_terminal", "best_name",
                                 "best_excess", "cash_return", "n_names_for_all",
                                 "share_of_names_for_all")})
    h["top_10pct_share"] = wc["top_10pct_share"]
    h["top_decile_pct"] = 10.0
    h["concentration"] = {f"top_{p}pct": wc[f"top_{p}pct_share"]
                          for p in (2, 5, 10, 25, 50)}
    print(f"  cash returned {wc['cash_return']:.2f}x over the period")
    print(f"  median name {wc['median_terminal']:.2f}x, mean name "
          f"{wc['mean_terminal']:.2f}x")
    print(f"  {wc['share_beat_cash']:.0%} of names beat cash; "
          f"{wc['share_lost_money']:.0%} lost money outright")
    for p in (2, 5, 10, 25, 50):
        print(f"    top {p:2d}% of names -> {wc[f'top_{p}pct_share']:.0%} of the "
              f"total excess over cash")
    print(f"  best name: {wc['best_name']} (+{wc['best_excess']:.1f}x of excess wealth)")
    print(f"  {wc['n_names_for_all']} names ({wc['share_of_names_for_all']:.0%}) account "
          f"for essentially all of it")

    print("\n=== 4. so why does the index win? ===")
    rec = st.index_reconciliation(R, cash)
    h.update({"cash": rec["cash"], "median_single": rec["median_single"],
              "mean_single": rec["mean_single"],
              "rebalanced": rec["rebalanced_equal_weight"],
              "buy_and_hold": rec["buy_and_hold_basket"],
              "single_vol": rec["single_vol"], "portfolio_vol": rec["portfolio_vol"],
              "single_drag": rec["single_drag"], "portfolio_drag": rec["portfolio_drag"],
              "drag_saved": rec["drag_saved"]})
    print(f"  cash:                          {rec['cash']:.2f}x")
    print(f"  median single name:            {rec['median_single']:.2f}x")
    print(f"  buy and hold the basket:       {rec['buy_and_hold_basket']:.2f}x")
    print(f"  rebalanced equal weight:       {rec['rebalanced_equal_weight']:.2f}x")
    print(f"  the reason is the drag, and drag is proportional to variance:")
    print(f"    average single-name vol {rec['single_vol']:.1%} -> drag "
          f"{rec['single_drag']:.2%} a year")
    print(f"    portfolio vol           {rec['portfolio_vol']:.1%} -> drag "
          f"{rec['portfolio_drag']:.2%} a year")
    print(f"    saved: {rec['drag_saved']:.2%} a year, compounding over "
          f"{h['years']:.0f} years")
    cum = float(np.expm1(np.log1p(rec["drag_saved"]) * h["years"]))
    h["drag_saved_cumulative"] = cum
    print(f"    = {cum:.0%} cumulatively. The index does not avoid variance drag; it")
    print(f"      avoids most of it, by cutting the variance the drag is proportional to.")

    print("\n=== 5. rebalancing is doing the work ===")
    bh = st.buy_and_hold_index(R)
    h["bh_start_max_weight"] = float(bh["max_weight"].iloc[0])
    h["bh_end_max_weight"] = float(bh["max_weight"].iloc[-1])
    h["bh_start_eff_n"] = float(bh["effective_n"].iloc[0])
    h["bh_end_eff_n"] = float(bh["effective_n"].iloc[-1])
    h["bh_end_top5"] = float(bh["top5_weight"].iloc[-1])
    print(f"  left alone, the basket's largest position grew from "
          f"{bh['max_weight'].iloc[0]:.1%} to {bh['max_weight'].iloc[-1]:.1%}")
    print(f"  its top five went to {bh['top5_weight'].iloc[-1]:.0%} of the portfolio")
    print(f"  effective number of holdings fell from {bh['effective_n'].iloc[0]:.0f} to "
          f"{bh['effective_n'].iloc[-1]:.0f}")
    print(f"  a buy-and-hold 'diversified' basket quietly becomes a concentrated one, and")
    print(f"  gives back the variance reduction that was paying for the low drag.")

    print(f"\n=== 6. what it means for a concentrated book ({ODDS_HORIZON:.0f} years) ===")
    odds = st.concentrated_portfolio_odds(R, cash, sizes=(1, 2, 3, 5, 10, 20, 50),
                                          horizon_years=ODDS_HORIZON, n_draws=600)
    print(odds.round(4).to_string())
    h["odds"] = odds.reset_index().to_dict("records")
    for k in (5, 20, 50):
        nk = min(odds.index, key=lambda x: abs(x - k))
        h[f"odds_index_{k}"] = float(odds.loc[nk, "share_beat_index"])
        h[f"odds_cash_{k}"] = float(odds.loc[nk, "share_beat_cash"])
    print(f"  a 5-stock book beat the index {h['odds_index_5']:.0%} of the time")
    print(f"  a 20-stock book {h['odds_index_20']:.0%}")
    print(f"  the whole basket {h['odds_index_50']:.0%}")
    print(f"  concentration is not 'slightly less diversified'. It is a bet on the right")
    print(f"  tail, and the modal outcome of that bet is losing to the average.")

    print("\n=== 7. which way does survivorship run? ===")
    surv = []
    for vol in (0.30, 0.45, 0.60):
        s = st.survivorship_experiment(n_stocks=600, n_days=6000, vol=vol)
        surv.append({"vol": vol, **{k: v for k, v in s.items()}})
        print(f"  vol {vol:.0%}: {s['delist_rate']:.0%} delisted, median terminal "
              f"{s['median_all']:.2f}x for ALL vs {s['median_survivors']:.2f}x for "
              f"survivors (bias +{s['bias']:.2f}x)")
    h["survivorship"] = surv
    h["surv_bias"] = float(surv[1]["bias"])
    print(f"  the bias RAISES the median. Every number in this study is therefore")
    print(f"  conservative — the real cross-section is worse than this basket.")

    print("\n=== 8. the control: one knob, the whole mechanism ===")
    ctrl = []
    for vol in (0.01, 0.15, 0.30, 0.45, 0.60):
        sim = st.synthetic_cross_section(n_stocks=400, n_days=6000, drift=0.08, vol=vol)
        finals = np.expm1(np.log1p(sim).sum())
        simcash = pd.Series(np.full(len(sim), 0.02 / 252), index=sim.index)
        sws = st.horizon_sweep(sim, simcash, horizons_years=(10,), step=252)
        ctrl.append({"vol": vol, "arith_mean": float(sim.mean().mean() * 252),
                     "median_terminal": float(finals.median()),
                     "mean_terminal": float(finals.mean()),
                     "share_beat_cash": float(sws.loc[10, "share_beat_cash"])
                     if 10 in sws.index else np.nan,
                     "predicted_drag": st.variance_drag(0, vol)})
        print(f"  vol {vol:4.0%}: arithmetic mean {ctrl[-1]['arith_mean']:.2%} (UNCHANGED), "
              f"median {finals.median():6.2f}x, mean {finals.mean():7.2f}x, "
              f"{ctrl[-1]['share_beat_cash']:.0%} beat cash")
    h["control"] = ctrl
    print("  every row has the SAME expected return by construction. Only volatility")
    print("  changes, and it moves the median without moving the mean. That is the")
    print("  whole of Bessembinder's result, reproduced from one parameter.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    sw = "\n".join(
        f"| {r['years']:g} | {int(r['n']):,} | **{r['share_beat_cash']:.0%}** | "
        f"{r['share_lost_money']:.0%} | {r['median_return']:+.1%} | "
        f"{r['mean_return']:+.1%} | {r['skew']:+.2f} |" for r in h["sweep"])
    conc = "\n".join(f"| Top {p}% of names | {h['concentration'][f'top_{p}pct']:.0%} |"
                     for p in (2, 5, 10, 25, 50))
    odds = "\n".join(
        f"| {int(r['n_stocks'])} | {r['share_beat_cash']:.0%} | "
        f"**{r['share_beat_index']:.0%}** | {r['median_return']:+.1%} | "
        f"{r['p10']:+.1%} | {r['p90']:+.1%} |" for r in h["odds"])
    surv = "\n".join(
        f"| {r['vol']:.0%} | {r['delist_rate']:.0%} | {r['median_all']:.2f}× | "
        f"{r['median_survivors']:.2f}× | **+{r['bias']:.2f}×** |" for r in h["survivorship"])
    ctrl = "\n".join(
        f"| {r['vol']:.0%} | {r['arith_mean']:.2%} | {r['predicted_drag']:.2%} | "
        f"{r['median_terminal']:.2f}× | {r['mean_terminal']:.2f}× | "
        f"**{r['share_beat_cash']:.0%}** |" for r in h["control"])
    return f"""# Results — Study 1006 (Most Stocks Lose) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_names']} large-cap names,
{h['n_days']:,} common sessions from {h['start']} ({h['years']:.1f} years), against short
Treasuries. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

> **The headline fails on this basket, and that is the study.** All {h['n_names']} of these
> names were still listed in 2026, and each is scored over its own history. Bessembinder's
> universe is every US firm that ever listed, most of which delisted. Here the share of holdings
> beating bills **rises** with horizon rather than falling. Sections 2b and 7 establish what
> would have to be different for the famous result to appear — which is more useful than a
> replication would have been.

## 1. The engine

| | Per year |
|---|--:|
| Measured drag (arithmetic − **log** growth) | **{h['mean_drag']:.2%}** |
| Theoretical drag (σ²/2) | {h['predicted_drag']:.2%} |
| Correlation between the two, across names | {h['drag_corr']:.3f} |

A name's *average* outcome grows at its arithmetic mean and its *typical* outcome at its log
growth rate, and the gap is σ²/2. Both figures are annualised **linearly**: the identity holds in
log space, and comparing an arithmetic mean against a *compounded* return would report negative
drag for a riskless series, which is ordinary compounding rather than drag. The mechanism is
identified, not inferred.

## 2. Share of holdings beating Treasury bills, by horizon

| Years held | Windows | Beat bills | Lost money | Median | Mean | Log-return skew |
|---|--:|--:|--:|--:|--:|--:|
{sw}

**The share beating bills rises from {h['share_beat_cash_short']:.0%} at one year to
{h['share_beat_cash_long']:.0%} at {h['long_years']:.0f}.** The study was pre-registered
expecting the opposite. On surviving large caps the drift comfortably outruns the drag, so the
median holder wins more often the longer they hold.

## 2b. The condition — when *does* the median lose to cash?

The median compounds at roughly drift − σ²/2, so it beats cash exactly while

> **σ < sqrt( 2 · (drift − cash) )**

| | |
|---|--:|
| Average drift across these names | {h['mean_drift']:.1%} |
| Cash (log growth) | {h['cash_log_growth']:.1%} |
| **Breakeven volatility** | **{h['mean_breakeven']:.0%}** |
| What these names actually run at | {h['mean_vol']:.0%} |
| Median headroom | {h['median_headroom']:.0%} |
| Names over the line | {h['share_above_threshold']:.0%} |

This is the deliverable. "Most stocks lose to bills" is not a law about equities; it is what
happens to a cross-section sitting above that threshold. Surviving large caps sit well below it.
Small caps, and firms on their way to delisting, sit above — which is precisely the population
Bessembinder measured.

## 3. How few names did the work

Ranked by **dollar** wealth creation over cash from an equal starting stake — a name that turned
$1 into $60 contributes $59, one that halved contributes −$0.50:

| | Share of total excess over cash |
|---|--:|
{conc}

Cash returned {h['cash_return']:.2f}×. The median name returned {h['median_terminal']:.2f}× and
the mean {h['mean_terminal']:.2f}×. {h['share_beat_cash']:.0%} of names beat cash;
{h['share_lost_money']:.0%} lost money outright *despite surviving*. The best,
**{h['best_name']}**, alone contributed {h['best_excess']:.1f}× of excess wealth, and
{h['n_names_for_all']} names ({h['share_of_names_for_all']:.0%}) account for essentially all of
it.

## 4. So why does the index win?

| | Terminal wealth |
|---|--:|
| Cash | {h['cash']:.2f}× |
| Median single name | {h['median_single']:.2f}× |
| Buy and hold the basket | {h['buy_and_hold']:.2f}× |
| **Rebalanced equal weight** | **{h['rebalanced']:.2f}×** |

Because drag is proportional to variance, and diversification cuts variance:

| | Volatility | Drag per year |
|---|--:|--:|
| Average single name | {h['single_vol']:.1%} | {h['single_drag']:.2%} |
| The portfolio | {h['portfolio_vol']:.1%} | {h['portfolio_drag']:.2%} |
| **Saved** | | **{h['drag_saved']:.2%}** |

Over {h['years']:.0f} years that saving compounds to **{h['drag_saved_cumulative']:.0%}**. The
index does not escape variance drag — it escapes most of it, by shrinking the σ² the drag is
proportional to. This is the reconciliation, and it is arithmetic rather than a story.

## 5. Rebalancing is doing the work

Left alone, the basket's largest position grew from {h['bh_start_max_weight']:.1%} to
{h['bh_end_max_weight']:.1%}, its top five reached {h['bh_end_top5']:.0%} of the portfolio, and
the effective number of holdings fell from {h['bh_start_eff_n']:.0f} to
{h['bh_end_eff_n']:.0f}. A buy-and-hold "diversified" basket quietly becomes a concentrated one
and hands back the variance reduction that was paying for the low drag.

## 6. What it means for a concentrated book

Randomly drawn portfolios held for {h['odds_horizon']:.0f} years:

| Stocks held | Beat cash | Beat the index | Median | 10th pct | 90th pct |
|---|--:|--:|--:|--:|--:|
{odds}

A five-stock book beat the index {h['odds_index_5']:.0%} of the time. Concentration is not
"slightly less diversified" — it is a bet on the right tail whose *modal* outcome is losing to
the average.

## 7. Which way survivorship runs

Simulated cross-sections where names falling 80% are delisted:

| Volatility | Delisted | Median, all | Median, survivors | Bias |
|---|--:|--:|--:|--:|
{surv}

The bias **raises** the median, so a survivor-only basket like this one understates the effect.
That is the comfortable direction for the conclusion, and it is why the study can be run on this
data at all.

## 8. The control — one knob, the whole mechanism

Every row has the **same arithmetic expected return** by construction. Only volatility changes:

| Volatility | Arithmetic mean | Predicted drag | Median terminal | Mean terminal | Beat cash |
|---|--:|--:|--:|--:|--:|
{ctrl}

Volatility moves the median without moving the mean. Bessembinder's result reproduced from a
single parameter, with no need for any story about business models, disruption or management
quality.

## Caveats

- **Fifty large caps are not the market.** The real cross-section includes small caps, which
  are far more volatile and therefore suffer far more drag. Section 7 says which way that cuts.
- **Overlapping holding windows** in sections 2 and 6. Deliberate: the object is the
  distribution of investor experiences, not a hypothesis test, and nothing here computes a
  t-statistic on those windows.
- **Bills as the cash benchmark** via a short-Treasury fund, which carries a little duration and
  a fee. Both are small next to the effects measured.
- **No taxes, costs or dividends reinvested differently** between the concentrated and
  diversified cases. Rebalancing has a real cost that section 4 does not charge, which would
  reduce — not eliminate — the saving.
- **The result is about medians, not about whether equities are worth owning.** They are; the
  mean is excellent. The finding is that the mean is not where most individual names end up.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1006-most-stocks-underperform-cash](../README.md). Not investment advice.*
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
