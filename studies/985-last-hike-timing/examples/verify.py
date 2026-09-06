"""Real-tape verification — Study 985 (The Last Hike). Regenerates docs/results.md.

Reconstructs every Fed tightening cycle since 1994 from the published target-rate
record, measures what four asset classes did after each cycle's true final hike against an
unconditional base rate, then re-runs the identical event study on the dates a live
pause-recognition rule would have fired — counting its false alarms and the return given up
while waiting.

    python studies/985-last-hike-timing/examples/verify.py            # cache-only
    python studies/985-last-hike-timing/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lasthike import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


QUIET_MONTHS = 6


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "quiet_months": QUIET_MONTHS,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:8s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")

    path = st.policy_path()
    print(f"\n=== 1. the policy record ===")
    print(f"  {len(path)} target changes, {path.index[0].date()} -> {path.index[-1].date()}")
    print(f"  {int((path['direction'] > 0).sum())} hikes, "
          f"{int((path['direction'] < 0).sum())} cuts")
    print(f"  range of the target: {path['target'].min():.2f}% to {path['target'].max():.2f}%")
    h["n_moves"] = int(len(path))

    cycles = st.tightening_cycles(path)
    h["n_cycles"] = int(len(cycles))
    print(f"\n=== 2. tightening cycles ===")
    for _, c in cycles.iterrows():
        print(f"  {c['first_hike'].date()} -> {c['last_hike'].date()}  "
              f"{int(c['n_hikes']):2d} hikes over {c['months']:5.1f} months, "
              f"+{c['total_tightening']:.2f}pp")
    h["cycles"] = [{"first_hike": str(c["first_hike"].date()),
                    "last_hike": str(c["last_hike"].date()), "n_hikes": int(c["n_hikes"]),
                    "months": float(c["months"]),
                    "total_tightening": float(c["total_tightening"])}
                   for _, c in cycles.iterrows()]
    print(f"  -> {len(cycles)} cycles. That is the entire sample. Every statistic below rests "
          f"on it, and none of them can be strong.")

    print(f"\n=== 3. hindsight: after the true final hike ===")
    eq = px[data.EQUITY].dropna()
    tbl = st.event_table(eq, list(cycles["last_hike"]))
    print("  horizon   n   event mean   median    base    excess       t   hit")
    for m, r in tbl.iterrows():
        print(f"  {int(m):5d}m {int(r['n_events']):4d} {r['event_mean']:+11.1%} "
              f"{r['event_median']:+8.1%} {r['base_mean']:+8.1%} {r['excess']:+9.1%} "
              f"{r['t']:+7.2f} {r['hit_rate']:5.0%}")
    h["hindsight"] = tbl.reset_index().to_dict("records")
    h["hindsight_mean_12m"] = float(tbl.loc[12, "event_mean"])
    h["base_mean_12m"] = float(tbl.loc[12, "base_mean"])
    h["hindsight_excess_12m"] = float(tbl.loc[12, "excess"])
    h["hindsight_t_12m"] = float(tbl.loc[12, "t"])
    h["hindsight_hit_12m"] = float(tbl.loc[12, "hit_rate"])

    print(f"\n=== 4. the same test on other assets ===")
    others = {}
    for label, tk in (("long bonds", data.LONG_BOND), ("gold", data.GOLD),
                      ("small caps", data.SMALL)):
        s = px[tk].dropna()
        if len(s) < 500:
            continue
        t2 = st.event_table(s, list(cycles["last_hike"]))
        others[label] = t2.reset_index().to_dict("records")
        r = t2.loc[12]
        print(f"  {label:12s} 12m: event {r['event_mean']:+7.1%}, base {r['base_mean']:+7.1%}, "
              f"excess {r['excess']:+7.1%}, t {r['t']:+5.2f}  (n={int(r['n_events'])})")
    h["other_assets"] = others

    print(f"\n=== 5. real time: what could you actually have known? ===")
    fa = st.false_alarms(path, QUIET_MONTHS)
    h["n_live_signals"] = int(len(fa))
    h["false_alarm_rate"] = float(1 - fa["was_the_end"].mean())
    print(f"  a rule that declares the cycle over after {QUIET_MONTHS} quiet months fires "
          f"{len(fa)} times")
    print(f"  of those, {int((~fa['was_the_end']).sum())} were FALSE ALARMS "
          f"({h['false_alarm_rate']:.0%}) — another hike followed within two years")
    for _, r in fa.iterrows():
        nxt = r["next_hike"]
        print(f"    {r['signal_date'].date()}  "
              f"{'correct' if r['was_the_end'] else 'FALSE ALARM'}"
              + (f", next hike {pd.Timestamp(nxt).date()}" if nxt is not None else
                 ", no further hike"))
    h["live_signals"] = [{"signal_date": str(r["signal_date"].date()),
                          "was_the_end": bool(r["was_the_end"])} for _, r in fa.iterrows()]

    cmp = st.hindsight_vs_realtime(eq, cycles, path, QUIET_MONTHS)
    print(f"\n=== 6. hindsight vs delayed vs live ===")
    print("  horizon   hindsight        delayed           live")
    for m, r in cmp.iterrows():
        print(f"  {int(m):5d}m  {r['hindsight_excess']:+7.1%} (t{r['hindsight_t']:+5.2f})  "
              f"{r['delayed_excess']:+7.1%} (t{r['delayed_t']:+5.2f})  "
              f"{r['live_excess']:+7.1%} (t{r['live_t']:+5.2f})")
    h["comparison"] = cmp.reset_index().to_dict("records")
    h["live_excess_12m"] = float(cmp.loc[12, "live_excess"])
    h["live_t_12m"] = float(cmp.loc[12, "live_t"])
    h["delayed_excess_12m"] = float(cmp.loc[12, "delayed_excess"])

    print(f"\n=== 7. what the waiting cost ===")
    wd = st.what_the_delay_costs(eq, cycles, QUIET_MONTHS)
    for d, r in wd.iterrows():
        print(f"  last hike {d} -> acted {r['acted']}: gave up {r['missed_return']:+.1%}")
    h["missed"] = wd.reset_index().to_dict("records")
    h["median_missed"] = float(wd["missed_return"].median()) if len(wd) else np.nan
    h["missed_share"] = float(h["median_missed"] / h["hindsight_mean_12m"]) \
        if h["hindsight_mean_12m"] else np.nan
    print(f"  median forgone: {h['median_missed']:+.1%}, which is "
          f"{h['missed_share']:.0%} of the whole twelve-month move")

    print(f"\n=== 8. how sensitive is all this to the conventions? ===")
    sweep = []
    for min_h in (2, 3, 5):
        for gap in (6, 12, 24):
            c2 = st.tightening_cycles(path, min_hikes=min_h, gap_months=gap)
            if len(c2) < 2:
                continue
            t2 = st.event_table(eq, list(c2["last_hike"]), horizons_m=(12,))
            sweep.append({"min_hikes": min_h, "gap_months": gap, "n_cycles": len(c2),
                          "excess_12m": float(t2.loc[12, "excess"]),
                          "t_12m": float(t2.loc[12, "t"])})
            print(f"  min_hikes {min_h}, gap {gap:2d}m -> {len(c2)} cycles, "
                  f"12m excess {t2.loc[12, 'excess']:+.1%}, t {t2.loc[12, 't']:+.2f}")
    h["convention_sweep"] = sweep
    for q in (3, 6, 9, 12, 18):
        f2 = st.false_alarms(path, q)
        c3 = st.hindsight_vs_realtime(eq, cycles, path, q, horizons_m=(12,))
        print(f"  quiet {q:2d}m -> {len(f2)} signals, "
              f"{1 - f2['was_the_end'].mean():.0%} false, "
              f"live 12m excess {c3.loc[12, 'live_excess']:+.1%}")
    h["quiet_sweep"] = [{"quiet_months": q,
                         "n_signals": int(len(st.false_alarms(path, q))),
                         "false_rate": float(1 - st.false_alarms(path, q)["was_the_end"].mean()),
                         "live_excess": float(st.hindsight_vs_realtime(
                             eq, cycles, path, q, horizons_m=(12,)).loc[12, "live_excess"])}
                        for q in (3, 6, 9, 12, 18)]

    print(f"\n=== 9. synthetic control ===")
    for alpha, tag in ((0.30, "planted post-cycle rally"), (0.0, "null: no rally")):
        hind, live = [], []
        for s in range(6):
            w = st.synthetic_world(n_cycles=20, post_cycle_alpha=alpha, n_years=70, seed=985 + s)
            p2 = st.policy_path(w["moves"])
            c2 = st.tightening_cycles(p2)
            cm = st.hindsight_vs_realtime(w["prices"], c2, p2, QUIET_MONTHS, horizons_m=(12,))
            hind.append(cm.loc[12, "hindsight_excess"])
            live.append(cm.loc[12, "live_excess"])
        print(f"  {tag:28s} hindsight {np.nanmean(hind):+.1%}, live {np.nanmean(live):+.1%}")
        h[f"synthetic_{'planted' if alpha else 'null'}"] = {
            "hindsight": float(np.nanmean(hind)), "live": float(np.nanmean(live))}
    print("  -> even a genuine, generously-sized effect loses most of itself to the "
          "recognition delay. That is a property of the problem, not of this rule.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    cyc = "\n".join(
        f"| {c['first_hike']} → {c['last_hike']} | {c['n_hikes']} | {c['months']:.0f} | "
        f"+{c['total_tightening']:.2f} |" for c in h["cycles"])
    hind = "\n".join(
        f"| {int(r['horizon_m'])}m | {int(r['n_events'])} | {r['event_mean']:+.1%} | "
        f"{r['event_median']:+.1%} | {r['base_mean']:+.1%} | **{r['excess']:+.1%}** | "
        f"{r['t']:+.2f} | {r['hit_rate']:.0%} |" for r in h["hindsight"])
    oth = "\n".join(
        f"| {k} | {r['event_mean']:+.1%} | {r['base_mean']:+.1%} | {r['excess']:+.1%} | "
        f"{r['t']:+.2f} |" for k, rows in h["other_assets"].items()
        for r in rows if int(r["horizon_m"]) == 12)
    sig = "\n".join(f"| {s['signal_date']} | {'correct' if s['was_the_end'] else '**false alarm**'} |"
                    for s in h["live_signals"])
    cmp = "\n".join(
        f"| {int(r['horizon_m'])}m | {r['hindsight_excess']:+.1%} ({r['hindsight_t']:+.2f}) | "
        f"{r['delayed_excess']:+.1%} ({r['delayed_t']:+.2f}) | "
        f"{r['live_excess']:+.1%} ({r['live_t']:+.2f}) |" for r in h["comparison"])
    miss = "\n".join(f"| {m['last_hike']} | {m['acted']} | {m['missed_return']:+.1%} |"
                     for m in h["missed"])
    conv = "\n".join(
        f"| {c['min_hikes']} | {c['gap_months']}m | {c['n_cycles']} | {c['excess_12m']:+.1%} | "
        f"{c['t_12m']:+.2f} |" for c in h["convention_sweep"])
    qs = "\n".join(
        f"| {q['quiet_months']}m | {q['n_signals']} | {q['false_rate']:.0%} | "
        f"{q['live_excess']:+.1%} |" for q in h["quiet_sweep"])
    return f"""# Results — Study 985 (The Last Hike) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_moves']} federal-funds target
changes since February 1994 (hard-coded from the Fed's published record) against daily closes.
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The cycles

| Cycle | Hikes | Months | Total tightening (pp) |
|---|--:|--:|--:|
{cyc}

**{h['n_cycles']} cycles.** That is the entire sample, and no amount of daily data makes it
larger. Every number below should be read with that in front of it.

## 2. Hindsight: after the true final hike

| Horizon | n | Event mean | Median | Base rate | Excess | *t* | Hit rate |
|---|--:|--:|--:|--:|--:|--:|--:|
{hind}

The base rate is the same horizon measured from every month in the sample, not just from event
dates. Without it, "stocks rose {h['hindsight_mean_12m']:.0%} in the year after the last hike"
is not a finding — stocks rise {h['base_mean_12m']:.0%} in the year after most dates.

Twelve months, other assets:

| Asset | Event mean | Base | Excess | *t* |
|---|--:|--:|--:|--:|
{oth}

## 3. Real time: what you could actually have known

Nobody knew 2023-07-26 was the last hike until months of not-hiking had gone by. The most
generous live rule available uses no forecast at all: declare the cycle over once
**{h['quiet_months']} months** pass with no further hike. Here is every time it would have
fired:

| Signal date | |
|---|---|
{sig}

**{h['false_alarm_rate']:.0%} were false alarms** — another hike followed within two years. A
hindsight event study never sees these, because it only looks at dates that turned out to be
endings.

## 4. The three views side by side

| Horizon | Hindsight excess (*t*) | Delayed by {h['quiet_months']}m (*t*) | Live rule incl. false alarms (*t*) |
|---|--:|--:|--:|
{cmp}

## 5. What the waiting cost

| Last hike | Rule acted | Return given up |
|---|---|--:|
{miss}

Median forgone: **{h['median_missed']:+.1%}**, which is {h['missed_share']:.0%} of the entire
twelve-month move. The best part of the post-cycle rally happens while you are still waiting to
be sure the cycle is over.

## 6. Do the conventions drive it?

| Min hikes | Max gap | Cycles | 12m excess | *t* |
|---|---|--:|--:|--:|
{conv}

| Quiet period | Signals | False alarms | Live 12m excess |
|---|--:|--:|--:|
{qs}

## 7. Synthetic control

With a generous post-cycle rally **planted** in a world of 20 cycles: hindsight excess
{h['synthetic_planted']['hindsight']:+.1%}, live excess
**{h['synthetic_planted']['live']:+.1%}**. With no rally planted:
{h['synthetic_null']['hindsight']:+.1%} and {h['synthetic_null']['live']:+.1%}. Even a real,
large effect loses most of itself to the recognition delay — that is a property of the problem,
not a weakness of this particular rule.

## Caveats

- **{h['n_cycles']} cycles.** This is the binding constraint and it cannot be relieved. The
  *t*-statistics are reported because leaving them out would be worse, not because they carry
  weight.
- **Every cycle is its own regime.** 1994 ended in a soft landing, 2000 and 2006 in recessions,
  2018 in a policy reversal, 2023 in a disinflation nobody's model predicted. Averaging them
  assumes an exchangeability that is hard to defend.
- **The target rate is not the stance.** Balance-sheet policy from 2008 onward means the funds
  rate stopped being a sufficient statistic for how tight money was.
- **The cycle definition is a choice.** Section 6 sweeps it; the 1997 lone hike and the
  2015-2018 crawl are where the choices bite.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[985-last-hike-timing](../README.md). Not investment advice.*
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
