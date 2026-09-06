"""Real-tape verification — Study 971 (Does the Tape Agree With Itself?). Regenerates docs/results.md.

Pulls four views of the same eight tickers from one provider — daily total return,
daily unadjusted plus corporate-action events, weekly and monthly — and runs six consistency
checks across them: resampling agreement, total-return reconstruction, calendar coverage, split
handling, dividend accounting and bar sanity.

    python studies/971-tape-self-consistency/examples/verify.py            # cache-only
    python studies/971-tape-self-consistency/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tape_audit import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def report() -> dict:
    tapes = {tk: data.load_all(tk) for tk in data.TICKERS}
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "n_tickers": len(data.TICKERS)}

    print(f"as-of {data.AS_OF}   flavours: {', '.join(data.FLAVOURS)}")
    stamps = {}
    for tk, fr in tapes.items():
        stamps[tk] = {fl: data.fingerprint(df) for fl, df in fr.items()}
        d = fr["daily_tr"]
        print(f"  {tk:6s} daily {len(d):5,} bars {d.index[0].date()} -> {d.index[-1].date()}  "
              f"weekly {len(fr['weekly']):5,}  monthly {len(fr['monthly']):4,}  "
              f"fp {stamps[tk]['daily_tr']}")
    h["fingerprints"] = stamps
    h["windows"] = {tk: [str(fr["daily_tr"].index[0].date()),
                         str(fr["daily_tr"].index[-1].date())] for tk, fr in tapes.items()}
    h["n_bars"] = {tk: int(len(fr["daily_tr"])) for tk, fr in tapes.items()}
    fetched = data.cache_dates()
    if fetched:
        days = sorted({v[:10] for v in fetched.values()})
        print(f"  all four flavours fetched on: {', '.join(days)} "
              f"({'one vintage' if len(days) == 1 else 'CAUTION: mixed vintages'})")
        h["fetch_days"] = days

    calendar = st.build_calendar({tk: fr["daily_tr"] for tk, fr in tapes.items()})
    print(f"\n=== reference calendar: {len(calendar):,} sessions "
          f"({calendar[0].date()} -> {calendar[-1].date()}), a date any half of the tapes traded")
    h["n_calendar"] = int(len(calendar))

    print("\n=== 1. do the weekly and monthly bars compound from the daily ones? ===")
    print("  tkr     freq      periods   median diff   worst diff   periods > 10bps")
    resample = {}
    for tk, fr in tapes.items():
        ra = st.resample_agreement(fr["daily_tr"], fr["weekly"], fr["monthly"])
        resample[tk] = {f: dict(v) for f, v in ra.to_dict("index").items()}
        for f, row in ra.iterrows():
            print(f"  {tk:6s} {f:8s} {int(row['n_periods']):9,} "
                  f"{row['median_abs_diff_bps']:12.2f} {row['max_abs_diff_bps']:12.1f} "
                  f"{int(row['n_beyond_10bps']):17d}")
    h["resample"] = resample
    h["max_resample_bps"] = float(max(v[f]["max_abs_diff_bps"]
                                      for v in resample.values() for f in v))

    print("\n=== 1b. what window does each weekly bar actually cover? ===")
    probes = {}
    for tk, fr in tapes.items():
        p = st.weekly_window_probe(fr["daily_tr"], fr["weekly"])
        probes[tk] = p
        print(f"  {tk:6s} {p['window']:42s} matched {p['matched']}/{p['n_probe']} bars, "
              f"{p.get('share_modal', 0):.0%} agree")
    h["weekly_probe"] = probes
    odd = [tk for tk, p in probes.items() if p.get("modal_offset") not in (3, 4)]
    h["odd_weekly_windows"] = odd
    if odd:
        print(f"  -> {', '.join(odd)} do NOT run Monday-to-Friday. The same provider, the same "
              f"call, a different weekly window per ticker: joining weekly bars across these "
              f"tickers compares different weeks.")

    print("\n=== 2. can the adjusted close be rebuilt from price + dividends + splits? ===")
    print("  tkr     sessions  divs  splits   terminal ratio   annualised gap   worst day")
    recon = {}
    for tk, fr in tapes.items():
        rec = st.reconstruct_total_return(fr["daily_raw"])
        recon[tk] = rec
        if rec.get("available"):
            print(f"  {tk:6s} {rec['n']:9,} {rec['n_dividends']:5d} {rec['n_splits']:7d}   "
                  f"{rec['terminal_ratio']:14.5f}   {rec['annualised_gap']:+14.4%}   "
                  f"{rec['max_daily_diff_bps']:8.1f} bps")
    h["reconstruction"] = recon
    gaps = {tk: abs(r.get("annualised_gap", 0.0)) for tk, r in recon.items()}
    h["worst_reconstruction_ticker"] = max(gaps, key=gaps.get)
    h["max_reconstruction_gap"] = float(gaps[h["worst_reconstruction_ticker"]])

    print("\n=== 3. calendar coverage ===")
    cover = {}
    total_missing = 0
    for tk, fr in tapes.items():
        cg = st.calendar_gaps(fr["daily_tr"], calendar)
        cover[tk] = cg
        total_missing += cg["n_missing"]
        print(f"  {tk:6s} {cg['n_sessions']:6,} of {cg['n_calendar']:6,} "
              f"(coverage {cg['coverage']:.4%})  missing {cg['n_missing']:3d}  "
              f"extra {cg['n_extra']:3d}" +
              (f"  e.g. {', '.join(cg['missing_dates'][:3])}" if cg["n_missing"] else ""))
    h["coverage"] = cover
    h["total_missing_sessions"] = int(total_missing)

    print("\n=== 4. splits ===")
    splits = {}
    for tk, fr in tapes.items():
        sc = st.split_check(fr["daily_raw"], fr["daily_tr"])
        if len(sc):
            splits[tk] = sc.to_dict("records")
            bad = int((~sc["raw_ok"] | ~sc["adjusted_ok"]).sum())
            print(f"  {tk:6s} {len(sc)} split event(s), {bad} inconsistent")
            for rec in sc.to_dict("records")[:4]:
                print(f"    {rec['date']}  ratio {rec['ratio']:.0f}:1  as-traded move "
                      f"{rec['raw_move']:.3f}  adjusted move {rec['adjusted_move']:.3f}  "
                      f"{'LOOKS UNADJUSTED' if rec['looks_unadjusted'] else 'ok'}")
    h["splits"] = splits

    print("\n=== 5. dividends: does the total-minus-price gap match what the feed reports? ===")
    divs = {}
    for tk, fr in tapes.items():
        dy = st.dividend_yield_check(fr["daily_raw"], fr["daily_tr"])
        if dy.get("available"):
            divs[tk] = dy
            print(f"  {tk:6s} price CAGR {dy['price_cagr']:+7.2%}  total CAGR "
                  f"{dy['total_cagr']:+7.2%}  implied yield {dy['implied_yield']:6.2%}  "
                  f"reported {dy['reported_yield']:6.2%}  gap {dy['gap']:+.2%}")
    h["dividends"] = divs

    print("\n=== 6. bar sanity ===")
    sanity = {}
    for tk, fr in tapes.items():
        bs = st.bar_sanity(fr["daily_tr"])
        sanity[tk] = bs
        print(f"  {tk:6s} dup {bs['duplicate_dates']}  non-positive {bs['non_positive']}  "
              f"NaN {bs['nan_close']}  |move|>40% {bs['moves_beyond_threshold']}  "
              f"worst {bs['worst_move']:.1%}")
    h["sanity"] = sanity

    print("\n=== the audit, all checks, all tickers ===")
    all_findings = pd.concat([st.audit(fr, calendar, tk) for tk, fr in tapes.items()],
                             ignore_index=True)
    counts = st.severity_counts(all_findings)
    print(f"  {len(all_findings)} checks -> {counts['error']} errors, {counts['warning']} "
          f"warnings, {counts['info']} clean")
    for _, row in all_findings[all_findings["severity"] != "info"].iterrows():
        print(f"    [{row['severity']:7s}] {row['ticker']:6s} {row['check']:26s} {row['detail']}")
    h["n_checks"] = int(len(all_findings))
    h["n_errors"] = int(counts["error"])
    h["n_warnings"] = int(counts["warning"])
    h["findings"] = all_findings[all_findings["severity"] != "info"].to_dict("records")

    print("\n=== would any of it change a published number? ===")
    print("  tkr      CAGR daily / weekly      vol daily / weekly     Sharpe daily / weekly")
    impact = {}
    for tk, fr in tapes.items():
        imp = st.backtest_impact(fr["daily_tr"], fr["weekly"])
        impact[tk] = imp
        print(f"  {tk:6s} {imp['cagr_daily']:+7.2%} / {imp['cagr_weekly']:+7.2%}   "
              f"{imp['vol_daily']:6.2%} / {imp['vol_weekly']:6.2%}   "
              f"{imp['sharpe_daily']:+6.3f} / {imp['sharpe_weekly']:+6.3f}  "
              f"(dSharpe {imp['sharpe_gap']:+.3f})")
    h["impact"] = impact
    h["worst_backtest_ticker"] = max(impact, key=lambda k: abs(impact[k]["sharpe_gap"]))
    h["max_sharpe_gap"] = float(abs(impact[h["worst_backtest_ticker"]]["sharpe_gap"]))
    h["max_cagr_gap"] = float(impact[h["worst_backtest_ticker"]]["cagr_gap"])

    print("\n=== control: the audit on a synthetic tape, clean and deliberately broken ===")
    frames, truth = data.synthetic_tape(n_years=12, seed=971)
    cal = st.build_calendar({"x": frames["daily_tr"]})
    clean_counts = st.severity_counts(st.audit(frames, cal, "SYNTH-CLEAN"))
    broken, planted = data.corrupt_tape(frames)
    broken_counts = st.severity_counts(st.audit(broken, cal, "SYNTH-BROKEN"))
    print(f"  clean : {clean_counts['error']} errors, {clean_counts['warning']} warnings")
    print(f"  broken: {broken_counts['error']} errors, {broken_counts['warning']} warnings "
          f"(planted: {planted})")
    h["control"] = {"clean_errors": int(clean_counts["error"]),
                    "broken_errors": int(broken_counts["error"]), "planted": planted}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    stamp = "\n".join(f"| {tk} | {w[0]} → {w[1]} | {h['n_bars'][tk]:,} | "
                      f"`{h['fingerprints'][tk]['daily_tr']}` |"
                      for tk, w in h["windows"].items())
    res = "\n".join(
        f"| {tk} | {f} | {r['n_periods']:,} | {r['median_abs_diff_bps']:.2f} | "
        f"{r['max_abs_diff_bps']:.1f} | {int(r['n_beyond_10bps'])} |"
        for tk, d in h["resample"].items() for f, r in d.items())
    rec = "\n".join(
        f"| {tk} | {r['n']:,} | {r['n_dividends']} | {r['n_splits']} | "
        f"{r['terminal_ratio']:.5f} | **{r['annualised_gap']:+.4%}** |"
        for tk, r in h["reconstruction"].items() if r.get("available"))
    cov = "\n".join(
        f"| {tk} | {c['n_sessions']:,} | {c['n_calendar']:,} | {c['n_missing']} | "
        f"{c['coverage']:.4%} |" for tk, c in h["coverage"].items())
    dv = "\n".join(
        f"| {tk} | {d['price_cagr']:+.2%} | {d['total_cagr']:+.2%} | {d['implied_yield']:.2%} | "
        f"{d['reported_yield']:.2%} | {d['gap']:+.2%} |"
        for tk, d in h["dividends"].items())
    imp = "\n".join(
        f"| {tk} | {i['cagr_daily']:+.2%} | {i['cagr_weekly']:+.2%} | {i['vol_daily']:.2%} | "
        f"{i['vol_weekly']:.2%} | {i['sharpe_daily']:+.3f} | {i['sharpe_weekly']:+.3f} |"
        for tk, i in h["impact"].items())
    find = ("\n".join(f"| {f['severity']} | {f['ticker']} | {f['check']} | {f['detail']} |"
                      for f in h["findings"]) or "| info | — | — | nothing above `info` |")
    return f"""# Results — Study 971 (Does the Tape Agree With Itself?)

*Generated by [`examples/verify.py`](../examples/verify.py). Four views of the same eight
tickers, pulled from one provider in a single pass: daily total-return bars, daily unadjusted
bars with the dividend and split events, weekly bars and monthly bars. Every number below is a
comparison between two things that should be identical. As-of **{h['as_of']}**.*

## Data stamp

| Ticker | Window | Daily bars | Fingerprint (daily TR) |
|---|---|--:|---|
{stamp}

Reference trading calendar: **{h['n_calendar']:,} sessions**, defined as any date at least half
the tapes traded on.

## 1. Do the weekly and monthly bars compound from the daily ones?

| Ticker | Frequency | Periods | Median \\|diff\\| (bps) | Worst (bps) | Periods > 10 bps |
|---|---|--:|--:|--:|--:|
{res}

### And what window does each weekly bar actually cover?

| Ticker | Convention | Bars matched | Agreement |
|---|---|--:|--:|
{chr(10).join(f"| {tk} | {p['window']} | {p['matched']}/{p['n_probe']} | {p.get('share_modal', 0):.0%} |" for tk, p in h["weekly_probe"].items())}

Each weekly close is matched back to the daily session that produced it, to a part in a
million. Seven of the eight tapes run **Monday to Friday**. **{', '.join(h['odd_weekly_windows']) or 'None'}**
{'does' if len(h['odd_weekly_windows']) == 1 else 'do'} not: the bars are stamped on a Friday
and close on the *following Thursday*. Same provider, same call, same minute — a different
week. Anyone who joins weekly bars across tickers is comparing one asset's Monday-to-Friday
return with another's Friday-to-Thursday return, and nothing in the response says so.

## 2. Can the adjusted close be rebuilt from price + dividends + splits?

| Ticker | Sessions | Dividends | Splits | Terminal ratio | Annualised gap |
|---|--:|--:|--:|--:|--:|
{rec}

This is the load-bearing check. Every total-return backtest on this desk — and everywhere else
— depends on the provider's adjustment being right, and the reconstruction is the only way to
test it without a second vendor. Worst case here:
**{h['worst_reconstruction_ticker']} at {h['max_reconstruction_gap']:+.4%}/yr**.

## 3. Calendar coverage

| Ticker | Sessions | Reference calendar | Missing | Coverage |
|---|--:|--:|--:|--:|
{cov}

## 4. Dividends

| Ticker | Price CAGR | Total CAGR | Implied yield | Reported yield | Gap |
|---|--:|--:|--:|--:|--:|
{dv}

## The findings

| Severity | Ticker | Check | Detail |
|---|---|---|---|
{find}

**{h['n_checks']}** checks: **{h['n_errors']} errors**, **{h['n_warnings']} warnings**.

## Would any of it change a published number?

| Ticker | CAGR (daily) | CAGR (weekly) | Vol (daily) | Vol (weekly) | Sharpe (daily) | Sharpe (weekly) |
|---|--:|--:|--:|--:|--:|--:|
{imp}

Most of the Sharpe difference is **not** a data fault: volatility measured weekly is genuinely
not volatility measured daily times √5 unless returns are independent, which is study
**970** on this desk. The point of the column is that a reader who switches frequency
inherits that difference whether or not they know it exists.

## The control

The whole audit is run on a synthetic tape where the answer is known: clean tape,
**{h['control']['clean_errors']} errors**; the same tape with a dropped session, an unapplied
split and a missing dividend planted in it, **{h['control']['broken_errors']} errors**. An
audit that cannot fail is not an audit.

## Caveats

- **One provider.** This is a *self*-consistency audit, not a vendor comparison: it can prove
  a feed contradicts itself, never that two consistent feeds are both right.
- **Total-return series are re-based on every new dividend**, so all of these fingerprints
  drift on a refetch. That is expected and is why the desk quotes fingerprints at all.
- **The reference calendar is inferred** from the tapes themselves, so a session that *every*
  ticker is missing is invisible to it.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[971-tape-self-consistency](../README.md). Not investment advice.*
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
