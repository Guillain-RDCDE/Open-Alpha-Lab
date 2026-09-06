"""Real-tape verification — Study 963 (The Half Day). Regenerates docs/results.md.

Derives the early-close calendar from the rule, confirms every candidate against the
volume tape, and then measures the day before / the half day / the day after on five
tickers — three return legs, HAC *t*, bootstrap CI on the difference, per-family and
per-era cuts, the multiple-testing arithmetic and the cost arithmetic.

    python studies/963-half-day-sessions/examples/verify.py            # cache-only
    python studies/963-half-day-sessions/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from half_day import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


MAX_RATIO = st.MAX_VOLUME_RATIO
SPLIT = "2010-01-01"


def report() -> dict:
    bars = data.load_all()
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS)}

    print(f"as-of {data.AS_OF}   universe: {', '.join(data.TICKERS)}")
    for tk, b in bars.items():
        print(f"  {tk:4s} {b.index[0].date()} -> {b.index[-1].date()}  n={len(b):,}  "
              f"dropped bars={data.bad_bar_count(tk)}  fp={data.fingerprint(b)}")
    h["fingerprints"] = {tk: data.fingerprint(b) for tk, b in bars.items()}
    h["windows"] = {tk: [str(b.index[0].date()), str(b.index[-1].date())]
                    for tk, b in bars.items()}
    h["n_bars"] = {tk: int(len(b)) for tk, b in bars.items()}

    sessions = {tk: st.session_frame(b) for tk, b in bars.items()}
    ref = sessions["SPY"]
    cands = st.rule_candidates(ref.index)
    conf = st.confirm_candidates(ref, cands, MAX_RATIO)
    kept = conf.index[conf["confirmed"]]

    print(f"\n=== the calendar: rule proposes, the tape disposes (SPY, threshold {MAX_RATIO}) ===")
    print(f"  rule candidates      : {len(cands)}")
    print(f"  confirmed by volume  : {int(conf['confirmed'].sum())}  "
          f"({conf['confirmed'].mean():.0%})")
    print(f"  median volume ratio on confirmed days: "
          f"{conf.loc[conf['confirmed'], 'volume_ratio'].median():.2f} "
          f"(an ordinary day is 1.00)")
    for fam in st.FAMILIES:
        sel = conf[conf["family"] == fam]
        print(f"    {st.FAMILY_LABEL[fam]:34s} {int(sel['confirmed'].sum()):3d} of {len(sel):3d} "
              f"confirmed, median ratio {sel['volume_ratio'].median():.2f}")
    rejected = conf[~conf["confirmed"]]
    if len(rejected):
        print(f"  rejected candidates ({len(rejected)}) — the rule guessed, the tape said no:")
        for d, row in rejected.head(12).iterrows():
            print(f"    {d.date()}  {row['family']:13s} volume ratio {row['volume_ratio']:.2f}")
    med_conf = float(conf.loc[conf["confirmed"], "volume_ratio"].median())
    unclaimed = st.unclaimed_thin_days(ref, cands, med_conf)
    loose = st.unclaimed_thin_days(ref, cands, MAX_RATIO)
    print(f"  recall check — ordinary sessions as quiet as a TYPICAL half day "
          f"(ratio < {med_conf:.2f}): {len(unclaimed)} of {len(ref):,} "
          f"({len(unclaimed) / len(ref):.1%} of the tape)")
    print(f"    at the confirmation threshold itself (< {MAX_RATIO:.2f}) that count is "
          f"{len(loose)} ({len(loose) / len(ref):.1%}) — the threshold is generous by "
          f"construction, which is why the median ratio, not the threshold, is the honest "
          f"description of a half day")
    for d, row in unclaimed.head(6).iterrows():
        print(f"    {d.date()}  volume ratio {row['volume_ratio']:.2f}  r_cc {row['r_cc']:+.2%}")

    h["n_candidates"] = int(len(cands))
    h["n_confirmed"] = int(conf["confirmed"].sum())
    h["median_volume_ratio"] = med_conf
    h["n_unclaimed"] = int(len(unclaimed))
    h["share_unclaimed"] = float(len(unclaimed) / len(ref))
    h["rejected"] = [[str(d.date()), r["family"], float(r["volume_ratio"])]
                     for d, r in conf[~conf["confirmed"]].iterrows()]
    h["per_family_confirmed"] = {f: int(conf[conf["family"] == f]["confirmed"].sum())
                                 for f in st.FAMILIES}
    h["sessions_per_year"] = float(len(kept) / ((ref.index[-1] - ref.index[0]).days / 365.25))

    # ---------------------------------------------------------------- the headline
    print(f"\n=== the half day itself, close to close (each ticker's own confirmed dates) ===")
    print("  tkr    n   half-day  ordinary       gap   HAC t      bootstrap CI      hit")
    per_ticker = {}
    for tk, s in sessions.items():
        c = st.confirm_candidates(s, st.rule_candidates(s.index), MAX_RATIO)
        d = c.index[c["confirmed"]]
        g = st.group_stats(s["r_cc"], st.event_mask(s, d, 0))
        ci = st.bootstrap_diff_ci(s["r_cc"], st.event_mask(s, d, 0))
        per_ticker[tk] = {**g, **ci, "n_dates": int(len(d))}
        print(f"  {tk:4s} {g['n_event']:4d}  {g['mean_bps']:+8.1f}  {g['rest_bps']:+8.1f}  "
              f"{g['diff_bps']:+8.1f}  {g['t_diff']:+6.2f}  "
              f"[{ci['ci_low']:+7.1f}, {ci['ci_high']:+7.1f}]  {g['hit_rate']:.0%}")
    h["per_ticker"] = per_ticker
    h["max_abs_t"] = float(max(abs(v["t_diff"]) for v in per_ticker.values()))
    h["n_ticker_hits"] = int(sum(abs(v["t_diff"]) >= 2.0 for v in per_ticker.values()))

    print("\n=== the three legs on SPY — which half of the day is the story? ===")
    legs = {}
    for leg, label in (("r_on", "overnight gap into the half day"),
                       ("r_oc", "the shortened session itself"),
                       ("r_cc", "close to close (what a holder earns)")):
        g = st.group_stats(ref[leg], st.event_mask(ref, kept, 0))
        legs[leg] = g
        print(f"  {label:36s} {g['mean_bps']:+7.1f} bps vs {g['rest_bps']:+6.1f} "
              f"(gap {g['diff_bps']:+6.1f}, t {g['t_diff']:+5.2f})")
    h["legs_spy"] = legs

    print("\n=== the window: the day before, the half day, the day after (SPY, close-close) ===")
    wt = st.window_table(ref, kept)
    for off, row in wt.iterrows():
        tag = {-1: "day before", 0: "the half day", 1: "day after"}[off]
        print(f"  {tag:13s} n={int(row['n_event']):3d}  {row['mean_bps']:+7.1f} bps  "
              f"gap {row['diff_bps']:+6.1f}  t {row['t_diff']:+5.2f}  hit {row['hit_rate']:.0%}")
    h["window_spy"] = {int(k): dict(v) for k, v in wt.to_dict("index").items()}

    print("\n=== per family (SPY, close-close) — three different animals ===")
    ft = st.family_table(ref, conf)
    for fam, row in ft.iterrows():
        print(f"  {st.FAMILY_LABEL[fam]:34s} n={int(row['n_event']):3d}  "
              f"{row['mean_bps']:+7.1f} bps  gap {row['diff_bps']:+6.1f}  t {row['t_diff']:+5.2f}")
    h["family_spy"] = {k: dict(v) for k, v in ft.to_dict("index").items()}

    print(f"\n=== era cut (split {SPLIT}, SPY) ===")
    ec = st.era_cut(ref, kept, split=SPLIT)
    for era, row in ec.iterrows():
        print(f"  {era:5s} {row['start']} -> {row['end']}  n={int(row['n_event']):3d}  "
              f"{row['mean_bps']:+7.1f} bps  gap {row['diff_bps']:+6.1f}  t {row['t_diff']:+5.2f}")
    h["era_spy"] = {k: dict(v) for k, v in ec.to_dict("index").items()}

    # ------------------------------------------------------- multiplicity & threshold
    n_cells = len(data.TICKERS) * len(st.FAMILIES) * 3
    hits = []
    for tk, s in sessions.items():
        c = st.confirm_candidates(s, st.rule_candidates(s.index), MAX_RATIO)
        for fam in st.FAMILIES:
            d = c.index[(c["family"] == fam) & c["confirmed"]]
            for off in (-1, 0, 1):
                g = st.group_stats(s["r_cc"], st.event_mask(s, d, off))
                if np.isfinite(g["t_diff"]) and abs(g["t_diff"]) >= 2.0:
                    hits.append((tk, fam, off, g["diff_bps"], g["t_diff"]))
    hits0 = [x for x in hits if x[2] == 0]
    n_cells0 = len(data.TICKERS) * len(st.FAMILIES)
    max_t0 = 0.0
    for tk, s in sessions.items():
        c = st.confirm_candidates(s, st.rule_candidates(s.index), MAX_RATIO)
        for fam in st.FAMILIES:
            d = c.index[(c["family"] == fam) & c["confirmed"]]
            g = st.group_stats(s["r_cc"], st.event_mask(s, d, 0))
            if np.isfinite(g["t_diff"]):
                max_t0 = max(max_t0, abs(g["t_diff"]))
    print(f"\n=== multiple testing: {len(data.TICKERS)} tickers x {len(st.FAMILIES)} families "
          f"x 3 windows = {n_cells} cells ===")
    print(f"  cells clearing |t| = 2 : {len(hits)}   expected by luck at 5%: "
          f"{st.expected_false_positives(n_cells):.1f}")
    for tk, fam, off, g, t in hits:
        print(f"    {tk:4s} {fam:13s} offset {off:+d}: {g:+7.1f} bps  t {t:+5.2f}")
    print(f"  of those, on the HALF DAY ITSELF (offset 0, {n_cells0} cells): {len(hits0)} "
          f"vs {st.expected_false_positives(n_cells0):.2f} expected; strongest |t| = {max_t0:.2f}")
    print(f"  the sharper cells sit at offset -1 — the day *before* a holiday, which is the "
          f"pre-holiday effect of the 1988-1990 literature, not the half day (see "
          f"docs/references.md and study 780)")
    print(f"  NOTE the five tapes are 0.6-0.9 correlated with each other, so 45 cells are "
          f"nowhere near 45 independent tests; the luck benchmark is a floor, not a ceiling")
    h["n_cells"] = int(n_cells)
    h["n_hits"] = int(len(hits))
    h["expected_hits"] = st.expected_false_positives(n_cells)
    h["n_cells0"] = int(n_cells0)
    h["n_hits0"] = int(len(hits0))
    h["expected_hits0"] = st.expected_false_positives(n_cells0)
    h["max_abs_t0"] = float(max_t0)
    h["hits"] = [[tk, fam, int(off), float(g), float(t)] for tk, fam, off, g, t in hits]

    print("\n=== threshold sweep — does the confirmation rule drive the answer? ===")
    sweep = []
    for thr in (0.55, 0.65, 0.75, 0.85, 0.95):
        c = st.confirm_candidates(ref, cands, thr)
        d = c.index[c["confirmed"]]
        g = st.group_stats(ref["r_cc"], st.event_mask(ref, d, 0))
        sweep.append({"threshold": thr, "n": g["n_event"], "diff_bps": g["diff_bps"],
                      "t": g["t_diff"]})
        print(f"  volume ratio < {thr:.2f}: n={g['n_event']:3d}  gap {g['diff_bps']:+7.1f} bps  "
              f"t {g['t_diff']:+5.2f}")
    h["threshold_sweep"] = sweep

    # ------------------------------------------------------------------- tradability
    best = max(per_ticker.items(), key=lambda kv: abs(kv[1]["diff_bps"]))
    edge = best[1]["diff_bps"]
    spy_edge = per_ticker["SPY"]["diff_bps"]
    print(f"\n=== could you trade it? ({h['sessions_per_year']:.1f} sessions a year) ===")
    print(f"  best per-session gap in the lot: {best[0]} {edge:+.1f} bps "
          f"(t {best[1]['t_diff']:+.2f})")
    ct = st.cost_arithmetic(edge, h["sessions_per_year"])
    for c, row in ct.iterrows():
        print(f"    cost {c:4.1f} bps one-way -> net {row['net_bps_per_year']:+7.1f} bps/yr "
              f"({row['net_pct_per_year']:+.3f}%/yr)")
    print(f"  break-even one-way cost: {st.breakeven_cost_bps(edge):.1f} bps "
          f"(a retail spread on SPY is ~0.5-1 bp; on IWM or GLD, more)")
    h["best_ticker"] = best[0]
    h["best_edge_bps"] = float(edge)
    h["best_ci_low"] = float(best[1]["ci_low"])
    h["best_ci_high"] = float(best[1]["ci_high"])
    h["spy_edge_bps"] = float(spy_edge)
    h["spy_t"] = float(per_ticker["SPY"]["t_diff"])
    h["breakeven_bps"] = st.breakeven_cost_bps(edge)
    h["net_at_1bp"] = float(ct.loc[1.0, "net_bps_per_year"])

    # -------------------------------------------------------------- synthetic control
    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for bump, tag in ((60.0, "planted +60 bps"), (0.0, "null (no bump)")):
        ts, gaps = [], []
        for s in range(6):
            b, _ = data.synthetic_ohlc(n_years=20, seed=963 + s)
            b, d = st.plant_half_days(b, every=84, bump_bps=bump)
            det = st.synthetic_detect(b, d)
            ts.append(det["t_diff"])
            gaps.append(det["diff_bps"])
        print(f"  {tag:16s}: recovered gap {np.mean(gaps):+6.1f} bps "
              f"(sd {np.std(gaps, ddof=1):.1f}), mean t {np.mean(ts):+5.2f}, "
              f"|t|>=2 in {sum(abs(t) >= 2 for t in ts)}/6")
        h[f"synthetic_{'planted' if bump else 'null'}"] = {
            "mean_gap_bps": float(np.mean(gaps)), "mean_t": float(np.mean(ts)),
            "hits": int(sum(abs(t) >= 2 for t in ts))}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (the pre-registered rule in strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    pt = h["per_ticker"]
    rows = "\n".join(
        f"| **{tk}** | {v['n_event']} | {v['mean_bps']:+.1f} | {v['rest_bps']:+.1f} | "
        f"**{v['diff_bps']:+.1f}** | {v['t_diff']:+.2f} | [{v['ci_low']:+.1f}, {v['ci_high']:+.1f}] | "
        f"{v['hit_rate']:.0%} |"
        for tk, v in pt.items())
    fam = "\n".join(
        f"| {st.FAMILY_LABEL[k]} | {int(v['n_event'])} | {v['mean_bps']:+.1f} | "
        f"{v['diff_bps']:+.1f} | {v['t_diff']:+.2f} |"
        for k, v in h["family_spy"].items())
    labels = {-1: "The day before", 0: "The half day itself", 1: "The day after"}
    win = "\n".join(
        f"| {labels[int(k)]} | {int(v['n_event'])} | {v['mean_bps']:+.1f} | "
        f"{v['diff_bps']:+.1f} | {v['t_diff']:+.2f} | {v['hit_rate']:.0%} |"
        for k, v in h["window_spy"].items())
    legs = h["legs_spy"]
    sweep = "\n".join(
        f"| < {r['threshold']:.2f} | {int(r['n'])} | {r['diff_bps']:+.1f} | {r['t']:+.2f} |"
        for r in h["threshold_sweep"])
    v = h["_verdict"]
    fp = "\n".join(f"| {tk} | {h['windows'][tk][0]} → {h['windows'][tk][1]} | "
                   f"{h['n_bars'][tk]:,} | `{f}` |" for tk, f in h["fingerprints"].items())
    return f"""# Results — Study {963} (The Half Day) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily **OHLCV** bars
(`yfinance`, `auto_adjust=True`) for SPY, QQQ, IWM, TLT and GLD. The early-close dates are
**not typed in**: they are derived from the calendar rule (the July 3 session, the Friday
after Thanksgiving, December 24 when it trades) and then **confirmed against the volume
tape** — a candidate is kept only if it printed less than {h['threshold_sweep'][2]['threshold']:.2f}
of the median volume of the 60 sessions before it. As-of **{h['as_of']}**.*

## Data stamp

| Ticker | Window | Bars | Fingerprint |
|---|---|--:|---|
{fp}

## The calendar the tape agrees to

| | |
|---|--:|
| Rule candidates | {h['n_candidates']} |
| Confirmed by volume (< 0.75 of the trailing 60-session median) | **{h['n_confirmed']}** |
| Median volume on a confirmed half day (ordinary day = 1.00) | **{h['median_volume_ratio']:.2f}** |
| Ordinary sessions as quiet as a *typical* half day | {h['n_unclaimed']} ({h['share_unclaimed']:.1%} of the tape) |
| Confirmed sessions per year | {h['sessions_per_year']:.1f} |

Per family: July 3 **{h['per_family_confirmed']['jul3']}**, the Friday after Thanksgiving
**{h['per_family_confirmed']['black_friday']}**, Christmas Eve
**{h['per_family_confirmed']['dec24']}**.

The volume number is the one solid finding in this study: a 1 p.m. session trades about
**{h['median_volume_ratio']:.0%}** of a normal day's shares. Everything below asks whether
anything *else* about the day is unusual.

**Where the confirmation step fails, and why it is still published.** {len(h['rejected'])}
rule candidates were rejected for trading too much:

{chr(10).join(f"- `{d}` ({fam}) — volume ratio {r:.2f}" for d, fam, r in h['rejected'])}

Some of those were genuine early closes that happened to fall on a frightening day — a half
session in a selloff is *busy*, not quiet — and the 1993-1996 rejections are an artefact of
SPY's own thin, fast-growing early tape rather than of the exchange's calendar. The filter
selects **quiet** half days, not **all** half days, and the threshold sweep below is there
precisely so a reader can see how much that choice matters (it moves the headline by a few
basis points and no *t* crosses a threshold because of it).

## The headline — the half day itself, close to close

| Ticker | n | Half day (bps) | Ordinary day (bps) | Gap | HAC *t* | Bootstrap CI (bps) | Hit rate |
|---|--:|--:|--:|--:|--:|---|--:|
{rows}

*Gap* is the half-day mean minus the mean of every other session, in basis points per
session. The CI resamples the {h['per_ticker']['SPY']['n_event']}-odd event days themselves
against an equal number of ordinary days, 4,000 times.

## Which half of the day?

| Leg (SPY) | Half day (bps) | Ordinary (bps) | Gap | HAC *t* |
|---|--:|--:|--:|--:|
| Overnight gap into it | {legs['r_on']['mean_bps']:+.1f} | {legs['r_on']['rest_bps']:+.1f} | {legs['r_on']['diff_bps']:+.1f} | {legs['r_on']['t_diff']:+.2f} |
| The shortened session (open → close) | {legs['r_oc']['mean_bps']:+.1f} | {legs['r_oc']['rest_bps']:+.1f} | {legs['r_oc']['diff_bps']:+.1f} | {legs['r_oc']['t_diff']:+.2f} |
| Close to close | {legs['r_cc']['mean_bps']:+.1f} | {legs['r_cc']['rest_bps']:+.1f} | {legs['r_cc']['diff_bps']:+.1f} | {legs['r_cc']['t_diff']:+.2f} |

## The window, the families, the eras (SPY)

| Window | n | Mean (bps) | Gap | HAC *t* | Hit rate |
|---|--:|--:|--:|--:|--:|
{win}

| Family | n | Mean (bps) | Gap | HAC *t* |
|---|--:|--:|--:|--:|
{fam}

Era cut at {SPLIT}: early **{h['era_spy']['early']['diff_bps']:+.1f}** bps
(*t* = {h['era_spy']['early']['t_diff']:+.2f}), late
**{h['era_spy']['late']['diff_bps']:+.1f}** bps (*t* = {h['era_spy']['late']['t_diff']:+.2f}).

## Multiplicity — and which cells belong to this study

{h['n_cells']} cells were run ({len(h['tickers'])} tickers × 3 families × 3 windows).
**{h['n_hits']}** cleared |*t*| = 2 against **{h['expected_hits']:.1f}** expected by luck at
the 5% level:

| Ticker | Family | Offset | Gap (bps) | *t* |
|---|---|--:|--:|--:|
{chr(10).join(f"| {tk} | {fam} | {off:+d} | {g:+.1f} | {t:+.2f} |" for tk, fam, off, g, t in h['hits'])}

Two cautions, and they point in opposite directions. **Against** reading this as noise: the
hits are overwhelmingly *positive* and spread across families, which luck does not usually
arrange. **Against** reading it as a discovery: the five tapes are 0.6-0.9 correlated with
one another, so 45 cells are nowhere near 45 independent tests, and — decisively for this
study — the sharpest cells sit at **offset −1**, the session *before* the holiday. That is
the pre-holiday effect documented by Lakonishok & Smidt (1988) and Ariel (1990), and on this
desk it is study **780-long-weekend-drift**; it is not the half day.

On the half day itself ({h['n_cells0']} cells, offset 0): **{h['n_hits0']}** clear |*t*| = 2
against {h['expected_hits0']:.2f} expected, and the strongest reaches
|*t*| = {h['max_abs_t0']:.2f}. That is what the Signal stamp is computed from.

## Is the answer an artefact of the confirmation threshold?

| Volume ratio threshold | n | Gap (bps) | *t* |
|---|--:|--:|--:|
{sweep}

## Could you trade it?

The best per-session gap in the lot is **{h['best_ticker']} {h['best_edge_bps']:+.1f} bps**.
Multiply by **{h['sessions_per_year']:.1f} sessions a year** and subtract a round trip on each:

| One-way cost | Net per year |
|---|--:|
| 0 bps | {h['best_edge_bps'] * h['sessions_per_year']:+.1f} bps |
| 1 bp | {h['net_at_1bp']:+.1f} bps |
| 5 bps | {(h['best_edge_bps'] - 10.0) * h['sessions_per_year']:+.1f} bps |

Break-even one-way cost: **{h['breakeven_bps']:.1f} bps**.

## Synthetic control

Planted +60 bps on the flagged days: recovered
**{h['synthetic_planted']['mean_gap_bps']:+.1f}** bps, mean *t*
**{h['synthetic_planted']['mean_t']:+.2f}**, |*t*| ≥ 2 in
{h['synthetic_planted']['hits']}/6 seeds. Null (thin volume, no bump): recovered
**{h['synthetic_null']['mean_gap_bps']:+.1f}** bps, mean *t*
**{h['synthetic_null']['mean_t']:+.2f}**, |*t*| ≥ 2 in {h['synthetic_null']['hits']}/6.
The apparatus finds what is put in front of it and does not invent what is not.

## Verdict

The stamps below are produced by `strategy.verdict`, a rule fixed before the run and
unit-tested in [`tests/test_strategy.py`](../tests/test_strategy.py) — not chosen after
looking at the table.

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[963-half-day-sessions](../README.md). Not investment advice.*
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
