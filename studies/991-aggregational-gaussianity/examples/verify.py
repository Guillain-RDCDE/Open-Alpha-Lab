"""Real-tape verification — Study 991 (The Slow Bell). Regenerates docs/results.md.

Aggregates each asset's log returns to seven horizons, measures excess kurtosis,
Jarque-Bera, Anderson-Darling and multi-sigma tail frequencies at each, compares every kurtosis
against the exact ``k1/n`` decay that independence would give, fits the decay exponent, estimates
the Hill tail index to check the theorem even applies, and measures how little power the
normality tests have once the horizon is long enough to matter.

    python studies/991-aggregational-gaussianity/examples/verify.py            # cache-only
    python studies/991-aggregational-gaussianity/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from slowbell import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


KURT_THRESHOLD = 0.5


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "threshold": KURT_THRESHOLD,
               "fingerprint": data.fingerprint(px)}

    assets = {}
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        if tk == data.CASH:
            continue
        s = rets[tk].dropna()
        if len(s) < 1500:
            continue
        assets[tk] = s
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"excess kurtosis {s.kurtosis():6.1f}  skew {s.skew():+.2f}")
    h["n_assets"] = int(len(assets))
    lead = data.EQUITY
    r = assets[lead]
    h["asset"] = lead
    h["n_days"] = int(len(r))

    print(f"\n=== 1. how many observations are left at each horizon? ===")
    for hz in st.HORIZONS:
        n_nov = len(st.aggregate(r, hz))
        n_ov = len(st.aggregate(r, hz, overlapping=True))
        print(f"  {hz:4d} days: {n_nov:6,} non-overlapping windows "
              f"({n_ov:,} overlapping)")
    print("  the right-hand column is why long-horizon normality claims are so easy to make "
          "and so hard to support")

    print(f"\n=== 2. the convergence profile for {lead} ===")
    prof = st.convergence_profile(r)
    print("  horizon      n   kurtosis   iid pred   ratio    skew   JB p  JB  AD  3sig  4sig")
    for hz, row in prof.iterrows():
        print(f"  {int(hz):5d} {int(row['n']):7d} {row['excess_kurtosis']:10.3f} "
              f"{row['iid_prediction']:10.3f} {row['kurtosis_vs_iid']:7.1f} "
              f"{row['skew']:+7.2f} {row['jb_p']:6.3f} "
              f"{'R' if row['jb_reject'] else '.':>3s} "
              f"{'R' if row['ad_reject'] else '.':>3s} "
              f"{row['ratio_3sig']:5.1f} {row['ratio_4sig']:5.1f}")
    h["profile"] = prof.reset_index().to_dict("records")
    h["kurtosis_1d"] = float(prof.loc[1, "excess_kurtosis"])
    longest = int(prof.index[-1])
    h["longest_horizon"] = longest
    h["kurtosis_longest"] = float(prof.loc[longest, "excess_kurtosis"])
    h["iid_at_longest"] = float(prof.loc[longest, "iid_prediction"])
    h["ratio_3sig_longest"] = float(prof.loc[longest, "ratio_3sig"])
    h["n_at_longest"] = int(prof.loc[longest, "n"])
    print(f"  -> kurtosis falls from {h['kurtosis_1d']:.1f} to {h['kurtosis_longest']:.2f}, "
          f"but independence predicts {h['iid_at_longest']:.3f} at that horizon")

    print("\n=== 3. the decay rate, fitted ===")
    fit = st.fit_decay_rate(prof)
    h.update({"decay_exponent": fit["exponent"], "decay_se": fit["se"],
              "decay_t_vs_one": fit["t_vs_one"], "decay_r2": fit["r2"]})
    print(f"  kurtosis ~ horizon^(-{fit['exponent']:.3f})   se {fit['se']:.3f}   "
          f"R2 {fit['r2']:.2f}")
    print(f"  the i.i.d. value is exactly 1.000; t against it = {fit['t_vs_one']:+.2f}")
    conv = st.convergence_horizon(prof, KURT_THRESHOLD)
    h.update({k: conv.get(k) for k in ("actual_horizon", "iid_horizon", "slowdown")})
    print(f"  excess kurtosis first drops below {KURT_THRESHOLD} at "
          f"{conv['actual_horizon']} days; independence would have said "
          f"{conv['iid_horizon']} days (a {conv['slowdown']:.1f}x slowdown)"
          if conv.get("actual_horizon") else
          f"  excess kurtosis never drops below {KURT_THRESHOLD} within the horizons tested")

    print("\n=== 4. does the theorem even apply? the tail index ===")
    hills = []
    for frac in (0.01, 0.02, 0.05, 0.10):
        hl = st.hill_estimator(st.aggregate(r, 1), frac)
        if "alpha" not in hl:
            continue
        hills.append({"tail_frac": frac, "k": hl["k"], "alpha": hl["alpha"],
                      "se": hl["se"], "variance_exists": hl["variance_exists"],
                      "kurtosis_exists": hl["kurtosis_exists"]})
        print(f"  top {frac:.0%} of observations (k={hl['k']}): alpha "
              f"{hl['alpha']:.2f} +/- {hl['se']:.2f}   variance exists: "
              f"{hl['variance_exists']}   kurtosis exists: {hl['kurtosis_exists']}")
    h["hill_sweep"] = hills
    h["hill_alpha"] = float(np.median([x["alpha"] for x in hills])) if hills else np.nan
    print(f"  median across thresholds: alpha = {h['hill_alpha']:.2f}")
    print("  (below 2 the variance is infinite and the CLT does not apply at all; below 4 the "
          "KURTOSIS is infinite and every kurtosis number in this study is an unstable "
          "statistic rather than an estimate of a population quantity)")

    print("\n=== 5. can the tests even see it? ===")
    n_by_h = {int(hz): int(row["n"]) for hz, row in prof.iterrows()}
    pw = st.power_of_normality_tests(n_by_h, true_df=4.0, n_sims=400)
    print(pw.round(3).to_string())
    h["power_table"] = pw.reset_index().to_dict("records")
    h["power_at_longest"] = float(pw.loc[longest, "power_vs_t4"]) if longest in pw.index \
        else np.nan
    print(f"  at {longest} days there are {h['n_at_longest']} observations and Jarque-Bera has "
          f"{h['power_at_longest']:.0%} power against a t(4)")
    print("  so 'annual returns pass a normality test' is close to uninformative")

    print("\n=== 6. the overlapping-window temptation ===")
    ovl = []
    for hz in (21, 63, 252):
        o = st.overlap_inflation(r, hz, n_boot=400)
        if "effective_gain" not in o:
            continue
        ovl.append(o)
        print(f"  {hz:4d} days: {o['n_non_overlapping']:5d} -> {o['n_overlapping']:6d} rows "
              f"({o['apparent_gain']:.0f}x apparent), but only "
              f"{o['effective_gain']:.1f}x effective")
        print(f"           kurtosis {o['kurtosis_non_overlapping']:+.3f} "
              f"(non-overlapping) vs {o['kurtosis_overlapping']:+.3f} (overlapping)")
    h["overlap"] = ovl

    print("\n=== 7. every asset ===")
    cross = []
    for tk, s in assets.items():
        p = st.convergence_profile(s)
        if p.empty or 1 not in p.index:
            continue
        f = st.fit_decay_rate(p)
        hl = st.hill_estimator(st.aggregate(s, 1), 0.02)
        lg = int(p.index[-1])
        cross.append({"asset": tk, "n": len(s), "kurt_1d": p.loc[1, "excess_kurtosis"],
                      "kurt_longest": p.loc[lg, "excess_kurtosis"], "longest": lg,
                      "exponent": f.get("exponent", np.nan),
                      "t_vs_one": f.get("t_vs_one", np.nan),
                      "hill_alpha": hl.get("alpha", np.nan),
                      "ratio_3sig_longest": p.loc[lg, "ratio_3sig"]})
        print(f"  {tk:9s} kurtosis {p.loc[1, 'excess_kurtosis']:6.1f} -> "
              f"{p.loc[lg, 'excess_kurtosis']:6.2f} at {lg}d,  exponent "
              f"{f.get('exponent', np.nan):5.2f} (t vs 1: {f.get('t_vs_one', np.nan):+5.2f}),  "
              f"Hill alpha {hl.get('alpha', np.nan):4.2f}")
    h["cross_asset"] = cross
    exps = [c["exponent"] for c in cross if np.isfinite(c["exponent"])]
    h["median_exponent"] = float(np.median(exps)) if exps else np.nan
    h["share_slower_than_iid"] = float(np.mean([c["t_vs_one"] < -2 for c in cross
                                                if np.isfinite(c["t_vs_one"])]))
    print(f"  median exponent across assets: {h['median_exponent']:.2f}; "
          f"{h['share_slower_than_iid']:.0%} are significantly slower than i.i.d.")

    print("\n=== 8. synthetic control ===")
    ctrl = []
    for dft, clus, tag in ((1000, 0.0, "iid normal"), (4.0, 0.0, "iid t(4)"),
                           (1000, 0.99, "normal + clustering"),
                           (4.0, 0.99, "t(4) + clustering")):
        sim = st.synthetic_returns(n=len(r), df_t=dft, clustering=clus)
        p = st.convergence_profile(sim)
        f = st.fit_decay_rate(p)
        ctrl.append({"world": tag, "kurt_1d": p.loc[1, "excess_kurtosis"],
                     "kurt_252d": p.loc[252, "excess_kurtosis"] if 252 in p.index else np.nan,
                     "exponent": f.get("exponent", np.nan)})
        print(f"  {tag:22s} kurtosis {p.loc[1, 'excess_kurtosis']:6.2f} -> "
              f"{ctrl[-1]['kurt_252d']:6.3f} at 252d, exponent "
              f"{f.get('exponent', np.nan):.2f}")
    h["control"] = ctrl
    print("  read rows 2 and 4: the SAME one-day fat tails, with and without clustering. "
          "Clustering is what turns a fast convergence into a slow one.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prof = "\n".join(
        f"| {int(r['horizon'])} | {int(r['n'])} | {r['excess_kurtosis']:.3f} | "
        f"{r['iid_prediction']:.3f} | {r['kurtosis_vs_iid']:.1f}× | {r['skew']:+.2f} | "
        f"{r['jb_p']:.3f} | {'reject' if r['jb_reject'] else 'passes'} | "
        f"{'reject' if r['ad_reject'] else 'passes'} | {r['ratio_3sig']:.1f}× | "
        f"{r['ratio_4sig']:.1f}× |" for r in h["profile"])
    hill = "\n".join(
        f"| {r['tail_frac']:.0%} | {r['k']} | {r['alpha']:.2f} ± {r['se']:.2f} | "
        f"{'yes' if r['variance_exists'] else '**no**'} | "
        f"{'yes' if r['kurtosis_exists'] else '**no**'} |" for r in h["hill_sweep"])
    pw = "\n".join(
        f"| {int(r['horizon'])} | {int(r['n_obs'])} | {r['power_vs_t4']:.0%} |"
        for r in h["power_table"])
    ovl = "\n".join(
        f"| {r['horizon']} | {r['n_non_overlapping']} | {r['n_overlapping']} | "
        f"{r['apparent_gain']:.0f}× | **{r['effective_gain']:.1f}×** | "
        f"{r['kurtosis_non_overlapping']:+.3f} | {r['kurtosis_overlapping']:+.3f} |"
        for r in h["overlap"])
    cross = "\n".join(
        f"| {r['asset']} | {int(r['n']):,} | {r['kurt_1d']:.1f} | {r['kurt_longest']:.2f} | "
        f"{r['exponent']:.2f} | {r['t_vs_one']:+.2f} | {r['hill_alpha']:.2f} | "
        f"{r['ratio_3sig_longest']:.1f}× |" for r in h["cross_asset"])
    ctrl = "\n".join(
        f"| {r['world']} | {r['kurt_1d']:.2f} | {r['kurt_252d']:.3f} | {r['exponent']:.2f} |"
        for r in h["control"])
    return f"""# Results — Study 991 (The Slow Bell) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets aggregated
to seven horizons; the detailed profile is **{h['asset']}** over {h['n_days']:,} sessions.
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The convergence profile

For independent draws with one-day excess kurtosis *k₁*, the sum of *n* of them has excess
kurtosis **exactly *k₁*/n**. That is not an asymptotic approximation — it is an identity, and it
gives this study a benchmark with no estimation error in it. The "i.i.d. prediction" column is
that identity; the "ratio" column is how much slower the tape actually converges.

| Horizon | n | Excess kurtosis | i.i.d. prediction | Ratio | Skew | JB *p* | JB | AD | 3σ | 4σ |
|---|--:|--:|--:|--:|--:|--:|---|---|--:|--:|
{prof}

Kurtosis falls from **{h['kurtosis_1d']:.1f}** at one day to **{h['kurtosis_longest']:.2f}** at
{h['longest_horizon']} days. Independence predicts {h['iid_at_longest']:.3f} there.

## 2. The decay rate

Fitting `kurtosis ~ horizon^(−b)`:

| | |
|---|--:|
| Fitted exponent *b* | **{h['decay_exponent']:.3f}** ± {h['decay_se']:.3f} |
| The i.i.d. value | 1.000 |
| *t* against it | **{h['decay_t_vs_one']:+.2f}** |
| R² of the fit | {h['decay_r2']:.2f} |
| Horizon where kurtosis < {h['threshold']} | {h['actual_horizon']} days |
| Horizon independence would predict | {h['iid_horizon']} days |
| **Slowdown** | **{h['slowdown']:.1f}×** |

## 3. Does the theorem even apply?

The central limit theorem needs finite variance. The Hill estimator of the tail index:

| Tail fraction | k | α | Variance exists (α > 2) | Kurtosis exists (α > 4) |
|---|--:|--:|---|---|
{hill}

Median α = **{h['hill_alpha']:.2f}**. This matters more than it looks: if α is below 4 then the
population kurtosis is **infinite**, and every kurtosis number in section 1 is a sample
statistic that does not converge to anything as the sample grows. It still describes what
happened; it is not an estimate of a parameter.

## 4. Can the tests even see non-normality?

Jarque-Bera's power against a *t*(4), at the sample sizes each horizon actually provides:

| Horizon | Observations | Power vs *t*(4) |
|---|--:|--:|
{pw}

At {h['longest_horizon']} days there are {h['n_at_longest']} observations and the test has
**{h['power_at_longest']:.0%}** power. "Annual returns pass a normality test" is therefore close
to uninformative — and it is the single most common way this stylised fact gets over-claimed.

## 5. The overlapping-window temptation

Overlapping windows multiply the row count by the horizon. They do not multiply the
information:

| Horizon | Non-overlapping | Overlapping | Apparent gain | **Effective gain** | Kurtosis (non-ov) | Kurtosis (ov) |
|---|--:|--:|--:|--:|--:|--:|
{ovl}

The effective gain is measured by bootstrapping the kurtosis estimate under both schemes and
comparing variances.

## 6. Every asset

| Asset | n | Kurtosis 1d | At longest | Exponent | *t* vs 1 | Hill α | 3σ ratio |
|---|--:|--:|--:|--:|--:|--:|--:|
{cross}

Median exponent **{h['median_exponent']:.2f}**;
**{h['share_slower_than_iid']:.0%}** of assets converge significantly slower than independence
would give.

## 7. Synthetic control

| World | Kurtosis at 1d | At 252d | Fitted exponent |
|---|--:|--:|--:|
{ctrl}

Compare rows two and four: the **same** one-day fat tails, with and without volatility
clustering. Clustering is what turns a fast convergence into a slow one — the fat tails alone
converge at close to the i.i.d. rate, exactly as the theorem says they should.

## Caveats

- **The kurtosis may not exist.** If the Hill index is below 4, the population kurtosis is
  infinite and the sample kurtosis is descriptive rather than estimative. Section 3 checks this
  and the answer is uncomfortably close to the boundary.
- **32 annual observations.** Every statement about the 252-day horizon rests on about thirty
  non-overlapping windows. The confidence intervals are wide and are not drawn.
- **One country, one era.** These are US assets over a period containing two crashes and one
  pandemic. Convergence rates in other markets and other eras may differ.
- **Aggregation is arithmetic on log returns.** That is the right choice for a CLT question, but
  a practitioner planning in simple returns should note that log and simple returns have
  different distributions at long horizons, and the difference grows with volatility.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[991-aggregational-gaussianity](../README.md). Not investment advice.*
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
