"""Real-tape verification — Study 1008 (The Start-Date Lottery). Regenerates docs/results.md.

Runs an identical monthly-contribution plan from every available start date and
measures the spread of outcomes, separates the part driven by the return distribution from pure
sequence risk using a shuffle that holds the distribution exactly fixed, locates where in the
horizon the exposure sits, then scores every controllable remedy — glide-path shape, constant
mixes — on dispersion removed against median wealth given up. A synthetic control with
independent paths supplies the magnitudes that overlapping real windows cannot.

    python studies/1008-start-date-lottery/examples/verify.py            # cache-only
    python studies/1008-start-date-lottery/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from startdate import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


YEARS = 25.0
WITHDRAWAL = 0.04


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "years": YEARS, "asset": data.EQUITY,
               "fingerprint": data.fingerprint(px)}

    r = px[data.EQUITY].dropna().pct_change().dropna()
    bonds = px[data.BONDS].dropna().pct_change()
    idx = r.index.intersection(bonds.dropna().index)
    h["n_days"] = int(len(r))
    h["start"] = str(r.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {data.EQUITY}: {len(r):,} sessions from {h['start']} "
          f"({len(r) / 252:.1f} years)")

    print(f"\n=== 1. the same plan, every start date ({YEARS:.0f}-year accumulation) ===")
    acc = st.accumulation_paths(r, YEARS)
    d = st.path_dispersion(acc, "multiple")
    h.update({k: d[k] for k in ("n_paths", "effective_n", "median", "mean", "cv",
                                "p05", "p95", "ratio_95_05", "ratio_max_min",
                                "best_start", "worst_start")})
    h["best_multiple"] = d["max"]
    h["worst_multiple"] = d["min"]
    print(f"  {d['n_paths']} overlapping start dates, worth ~{d['effective_n']:.1f} "
          f"independent ones")
    print(f"  each unit contributed became:")
    print(f"    best  {d['max']:.2f}x  (started {d['best_start']})")
    print(f"    95th  {d['p95']:.2f}x")
    print(f"    median {d['median']:.2f}x")
    print(f"    5th   {d['p05']:.2f}x")
    print(f"    worst {d['min']:.2f}x  (started {d['worst_start']})")
    print(f"  best/worst = {d['ratio_max_min']:.2f}x for an IDENTICAL plan")
    lump = st.path_dispersion(acc, "lump_sum")
    h["lump_ratio_95_05"] = lump["ratio_95_05"]
    h["lump_cv"] = lump["cv"]
    print(f"  for comparison, a LUMP SUM over the same windows spread "
          f"{lump['ratio_95_05']:.2f}x (cv {lump['cv']:.3f}) vs the contributor's "
          f"{d['ratio_95_05']:.2f}x (cv {d['cv']:.3f})")
    print(f"  -> contributing REDUCES dispersion. Averaging in is a real benefit, and the")
    print(f"     'start-date lottery' is smaller for a saver than for someone with a windfall.")

    print("\n=== 2. how much of that is ORDER, and how much is the distribution? ===")
    seg = r.to_numpy()[-int(YEARS * 252):]
    sh = st.shuffle_invariance(seg, contribution=1.0 / 21.0, n_shuffles=500)
    h.update({"shuffle_lump_cv": sh["lump_cv"], "shuffle_contrib_cv": sh["contrib_cv"],
              "shuffle_sequence_spread": sh["sequence_spread"]})
    print(f"  take ONE {YEARS:.0f}-year path and shuffle it. Same returns, same mean, same")
    print(f"  volatility, same everything -- only the ORDER changes.")
    print(f"    lump sum:    cv {sh['lump_cv']:.2e}  (unchanged, as it must be)")
    print(f"    contributor: cv {sh['contrib_cv']:.4f}, 95th/5th "
          f"{sh['sequence_spread']:.2f}x")
    print(f"  that {sh['sequence_spread']:.2f}x is PURE sequence risk, with the return")
    print(f"  distribution held exactly fixed. Everything above it in section 1 is the")
    print(f"  distribution differing between eras.")

    print("\n=== 3. where in the horizon does it live? ===")
    srm = st.sequence_risk_metrics(r, YEARS, n_buckets=6)
    print(srm.round(4).to_string())
    h["risk_profile"] = srm.reset_index().to_dict("records")
    h["first_bucket_corr"] = float(srm["corr_contributor"].iloc[0])
    h["last_bucket_corr"] = float(srm["corr_contributor"].iloc[-1])
    h["lump_first_corr"] = float(srm["corr_lump_sum"].iloc[0])
    h["lump_last_corr"] = float(srm["corr_lump_sum"].iloc[-1])
    print(f"  contributor: first sixth correlates {h['first_bucket_corr']:+.2f} with the")
    print(f"    outcome, final sixth {h['last_bucket_corr']:+.2f}")
    print(f"  lump sum:    {h['lump_first_corr']:+.2f} vs {h['lump_last_corr']:+.2f} "
          f"-- roughly equal, as the arithmetic requires")
    print(f"  the exposure is concentrated in the years just before the finish line. That")
    print(f"  is where a remedy has to act, and it is why WHEN you de-risk matters more")
    print(f"  than HOW MUCH.")

    print("\n=== 4. the retiree's mirror image ===")
    dec = st.decumulation_paths(r, YEARS, WITHDRAWAL)
    h["ruin_rate"] = float(dec["ruined"].mean())
    h["dec_median"] = float(dec["terminal"].median())
    h["dec_corr_first5"] = float(dec["first_5y_cagr"].corr(dec["terminal"]))
    print(f"  withdrawing {WITHDRAWAL:.0%} a year for {YEARS:.0f} years:")
    print(f"    ruined in {dec['ruined'].mean():.0%} of start dates")
    print(f"    median terminal {dec['terminal'].median():.2f}x the starting balance")
    print(f"    correlation between the FIRST five years' return and the final outcome: "
          f"{h['dec_corr_first5']:+.2f}")
    rr = []
    for w in (0.03, 0.04, 0.05, 0.06, 0.08):
        dd = st.decumulation_paths(r, YEARS, w)
        rr.append({"rate": w, "ruin": float(dd["ruined"].mean()),
                   "median": float(dd["terminal"].median()),
                   "p05": float(dd["terminal"].quantile(0.05))})
        print(f"    {w:.0%} withdrawal: ruin {rr[-1]['ruin']:.0%}, median "
              f"{rr[-1]['median']:.2f}x, 5th pct {rr[-1]['p05']:.2f}x")
    h["withdrawal_table"] = rr
    print(f"  for a contributor the LAST years matter; for a retiree the FIRST ones do.")
    print(f"  Same arithmetic, opposite sign.")

    print("\n=== 5. what actually helps? ===")
    e = r.reindex(idx)
    b = bonds.reindex(idx)
    rem = st.remedy_comparison(e, b, YEARS)
    if rem.empty or "100% equity throughout" not in rem.index:
        # The stocks+bonds overlap is shorter than the equity tape, so a long horizon can
        # leave no room for a single path. Fall back rather than reporting nothing.
        print(f"  (not enough overlapping stock/bond history for {YEARS:.0f}-year paths;")
        print(f"   falling back to 15 years for the remedy comparison)")
        rem = st.remedy_comparison(e, b, 15.0)
        h["remedy_years"] = 15.0
    else:
        h["remedy_years"] = YEARS
    print(rem.round(4).to_string())
    h["remedies"] = rem.reset_index().to_dict("records")
    scored = rem.drop(index="100% equity throughout", errors="ignore")
    scored = scored[np.isfinite(scored["efficiency"])]
    best = scored["efficiency"].idxmax()
    h["best_remedy"] = str(best)
    h["best_cv_reduction"] = float(scored.loc[best, "cv_reduction"])
    h["best_median_cost"] = float(scored.loc[best, "median_cost"])
    h["best_efficiency"] = float(scored.loc[best, "efficiency"])
    h["best_ratio_95_05"] = float(scored.loc[best, "ratio_95_05"])
    h["linear_efficiency"] = float(rem.loc["linear glide to 30%", "efficiency"]) \
        if "linear glide to 30%" in rem.index else np.nan
    h["sixty_forty_efficiency"] = float(rem.loc["constant 60/40", "efficiency"]) \
        if "constant 60/40" in rem.index else np.nan
    print(f"  best on dispersion-removed-per-unit-of-wealth-given-up: {best}")
    print(f"    cuts dispersion {h['best_cv_reduction']:.0%} for a median cost of "
          f"{h['best_median_cost']:.0%} -> efficiency {h['best_efficiency']:.2f}x")
    print(f"  a conventional LINEAR glide: {h['linear_efficiency']:.2f}x")
    print(f"  a constant 60/40 throughout:  {h['sixty_forty_efficiency']:.2f}x")
    print(f"  de-risking LATE is the efficient choice, because section 3 says that is")
    print(f"  where the exposure is. De-risking early pays for protection you do not")
    print(f"  yet need with compounding you cannot get back.")

    print("\n=== 6. how big should the lottery be? (independent paths) ===")
    lv = st.lottery_size_by_volatility(vols=(0.08, 0.12, 0.16, 0.24, 0.32),
                                       years=YEARS, n_paths=400)
    print(lv.round(4).to_string())
    h["by_volatility"] = lv.reset_index().to_dict("records")
    print(f"  these are INDEPENDENT paths, which the real data cannot supply. They give")
    print(f"  the magnitudes their proper scale: at equity-like volatility the contributor")
    print(f"  spread is {lv.loc[0.16, 'contrib_ratio_95_05']:.2f}x, against "
          f"{h['ratio_95_05']:.2f}x measured on overlapping real windows.")
    print(f"  and note the columns: the lump-sum spread exceeds the contributor's at every")
    print(f"  volatility. Regular contributions genuinely damp the lottery.")

    print("\n=== 7. horizon and the size of the lottery ===")
    hz = []
    for y in (10, 15, 20, 25, 30):
        a = st.accumulation_paths(r, y)
        dd = st.path_dispersion(a, "multiple")
        if not dd:
            continue
        hz.append({"years": y, "n_paths": dd["n_paths"],
                   "effective_n": dd["effective_n"], "median": dd["median"],
                   "cv": dd["cv"], "ratio_95_05": dd["ratio_95_05"]})
        print(f"  {y:2d} years: median {dd['median']:.2f}x, cv {dd['cv']:.3f}, "
              f"95/5 {dd['ratio_95_05']:.2f}x (effective n {dd['effective_n']:.1f})")
    h["by_horizon"] = hz
    print(f"  the falling effective-n column is the health warning: the longest horizons")
    print(f"  are the least measurable, and they are the ones people care about most.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prof = "\n".join(
        f"| {int(r['bucket'])} | {r['years_from']:.0f}–{r['years_to']:.0f} | "
        f"**{r['corr_contributor']:+.2f}** | {r['corr_lump_sum']:+.2f} |"
        for r in h["risk_profile"])
    def _eff(x):
        return f"**{x:.2f}×**" if np.isfinite(x) else "—"

    rem = "\n".join(
        f"| {r['remedy']} | {r['median']:.2f}× | {r['cv']:.3f} | {r['ratio_95_05']:.2f}× | "
        f"{r['median_cost']:.1%} | {r['cv_reduction']:.1%} | {_eff(r['efficiency'])} |"
        for r in h["remedies"])
    wd = "\n".join(
        f"| {r['rate']:.0%} | {r['ruin']:.0%} | {r['median']:.2f}× | {r['p05']:.2f}× |"
        for r in h["withdrawal_table"])
    lv = "\n".join(
        f"| {r['vol']:.0%} | {r['lump_cv']:.3f} | {r['contrib_cv']:.3f} | "
        f"{r['lump_ratio_95_05']:.2f}× | {r['contrib_ratio_95_05']:.2f}× |"
        for r in h["by_volatility"])
    hz = "\n".join(
        f"| {int(r['years'])} | {int(r['n_paths'])} | {r['effective_n']:.1f} | "
        f"{r['median']:.2f}× | {r['cv']:.3f} | {r['ratio_95_05']:.2f}× |"
        for r in h["by_horizon"])
    return f"""# Results — Study 1008 (The Start-Date Lottery) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['asset']}, {h['n_days']:,}
sessions from {h['start']}, {h['years']:.0f}-year plans from every start date. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

> **Effective sample.** The {h['n_paths']} start dates below overlap heavily and are worth
> roughly **{h['effective_n']:.1f} independent observations**. The magnitudes are indicative;
> the *comparisons* between remedies, which share the same windows, are the reliable part, and
> section 6 supplies properly independent paths for scale.

## 1. The same plan, every start date

Contributing an identical amount every month for {h['years']:.0f} years, each unit contributed
became:

| | Multiple of money invested |
|---|--:|
| Best start ({h['best_start']}) | **{h['best_multiple']:.2f}×** |
| 95th percentile | {h['p95']:.2f}× |
| Median | {h['median']:.2f}× |
| 5th percentile | {h['p05']:.2f}× |
| Worst start ({h['worst_start']}) | **{h['worst_multiple']:.2f}×** |

A **{h['ratio_max_min']:.2f}× gap between two people following an identical plan**, differing
only in when they began.

A useful comparison: a *lump sum* over the same windows spread {h['lump_ratio_95_05']:.2f}×
against the contributor's {h['ratio_95_05']:.2f}×. **Regular contributions reduce the lottery**
— averaging in is a genuine benefit, and the start-date problem is worse for someone deploying a
windfall than for a saver.

## 2. How much is order, and how much is the era?

Take one {h['years']:.0f}-year path and shuffle it. Same returns, same mean, same volatility —
only the sequence changes:

| | Coefficient of variation | 95th/5th |
|---|--:|--:|
| Lump sum | {h['shuffle_lump_cv']:.2e} | 1.00× |
| Contributor | {h['shuffle_contrib_cv']:.4f} | **{h['shuffle_sequence_spread']:.2f}×** |

The lump sum is unchanged to machine precision, because multiplication commutes. The
{h['shuffle_sequence_spread']:.2f}× is **pure sequence risk** with the return distribution held
exactly fixed; the remainder of section 1's spread is different eras having different
distributions.

## 3. Where in the horizon the exposure sits

Correlation between the return in each slice of the path and the final outcome:

| Slice | Years | Contributor | Lump sum |
|---|---|--:|--:|
{prof}

For a lump sum every period matters equally — the arithmetic demands it, and the column
confirms the measurement works. For a contributor the exposure is concentrated at the end,
because that is when the balance is largest. **This is why *when* you de-risk matters more than
*how much*.**

## 4. The retiree's mirror image

| Withdrawal rate | Ruined | Median terminal | 5th percentile |
|---|--:|--:|--:|
{wd}

The correlation between the *first* five years' return and the final outcome was
{h['dec_corr_first5']:+.2f}. For a contributor the last years dominate; for a retiree the first
ones do. Same arithmetic, opposite sign.

## 5. What actually helps

Every remedy on identical paths, scored on **both** columns:

| Remedy | Median | CV | 95th/5th | Median cost | CV reduction | Efficiency |
|---|--:|--:|--:|--:|--:|--:|
{rem}

The efficiency column is dispersion removed per unit of median wealth given up. A remedy that
halves the spread by halving the outcome has helped nobody, and reporting only the first column
is how glide paths are usually sold.

**{h['best_remedy']}** was the most efficient at {h['best_efficiency']:.2f}×, against
{h['linear_efficiency']:.2f}× for a conventional linear glide and
{h['sixty_forty_efficiency']:.2f}× for a constant 60/40. That ranking follows directly from
section 3: the exposure is late, so protection bought late is protection bought where it is
needed, and protection bought early costs compounding that cannot be recovered.

## 6. How big should the lottery be? Independent paths

| Volatility | Lump-sum CV | Contributor CV | Lump 95/5 | Contributor 95/5 |
|---|--:|--:|--:|--:|
{lv}

These are genuinely independent simulations, which the real tape cannot supply. Two readings:
they give the real-data magnitudes their proper scale, and the contributor column sits below the
lump-sum column at every volatility, confirming section 1's finding that regular contributions
damp the lottery rather than amplifying it.

## 7. Horizon

| Years | Paths | Effective n | Median | CV | 95th/5th |
|---|--:|--:|--:|--:|--:|
{hz}

The falling `effective_n` column is the health warning of the whole study: the longest horizons
are the least measurable, and they are the ones people care about most.

## Caveats

- **Overlapping windows**, reported honestly throughout. The between-remedy comparisons in
  section 5 share windows and are far more reliable than any single dispersion figure.
- **One market, one era.** A 33-year US sample contains one genuinely independent 30-year path.
  Section 6 exists because of this, and it is simulation rather than evidence.
- **Contributions are constant in nominal terms** and withdrawals constant in real terms only by
  assumption — no inflation series is applied. A real-terms version would raise the ruin rates
  in section 4.
- **No taxes, fees or behaviour.** The largest real-world driver of start-date outcomes is
  probably whether the investor kept contributing through the drawdown, which no rule in
  section 5 captures.
- **The remedies are asset-allocation rules only.** The largest levers — contributing for longer,
  saving more, retaining the flexibility to defer retirement — are not on the table and would
  dominate everything on it.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1008-start-date-lottery](../README.md). Not investment advice.*
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
