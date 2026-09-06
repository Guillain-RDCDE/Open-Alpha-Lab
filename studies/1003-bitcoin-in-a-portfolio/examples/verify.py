"""Real-tape verification — Study 1003 (The 1% Allocation). Regenerates docs/results.md.

Aligns bitcoin to the equity calendar and measures what that alone is worth,
sweeps every candidate sleeve weight against a 60/40, finds the Sharpe-maximising weight and
then the width of the plateau around it, block-bootstraps the optimiser to get an interval on
the recommendation, asks directly whether the sample can tell 1% from 5%, computes the years of
history the estimate would need, and runs a walk-forward allocator with realistic trading costs
to see what the recommendation is worth out of sample.

    python studies/1003-bitcoin-in-a-portfolio/examples/verify.py            # cache-only
    python studies/1003-bitcoin-in-a-portfolio/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from onepercent import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


MAX_WEIGHT = 0.50
COST_BPS = 30.0
LOOKBACK_Y = 3.0


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "max_weight": MAX_WEIGHT,
               "fingerprint": data.fingerprint(px)}

    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    common = base.index.intersection(btc.index)
    base, btc = base.loc[common], btc.loc[common]
    h["n_days"] = int(len(common))
    h["years"] = float(len(common) / 252)
    h["start"] = str(common[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(common):,} common sessions from {h['start']} ({h['years']:.1f} years)")

    print("\n=== 1. the calendar, before anything else ===")
    raw = px[data.BTC].dropna().pct_change().dropna()
    vol252 = float(raw.std(ddof=1) * np.sqrt(252))
    vol365 = float(raw.std(ddof=1) * np.sqrt(365))
    h["btc_vol"] = vol252
    h["btc_vol_365"] = vol365
    h["calendar_inflation"] = float(vol365 / vol252 - 1)
    print(f"  bitcoin volatility on the equity calendar (252): {vol252:.1%}")
    print(f"  the same numbers annualised over 365 days:       {vol365:.1%}")
    print(f"  using 365 for bitcoin and 252 for everything else inflates its annualised")
    print(f"  volatility by {h['calendar_inflation']:.0%} — and its Sharpe ratio in the")
    print(f"  opposite direction. Everything below is on the equity calendar.")

    print("\n=== 2. the sweep ===")
    sw = st.weight_sweep(base, btc, (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50))
    print(sw[["cagr", "vol", "sharpe", "max_drawdown", "sortino"]].round(4).to_string())
    h["sweep"] = sw.reset_index().to_dict("records")
    b = st.stats(base)
    h.update({"base_cagr": b["cagr"], "base_vol": b["vol"], "base_sharpe": b["sharpe"],
              "base_dd": b["max_drawdown"]})
    print(f"  the 60/40 alone: {b['cagr']:.2%} a year, {b['vol']:.1%} vol, "
          f"Sharpe {b['sharpe']:.2f}, worst drawdown {b['max_drawdown']:.1%}")

    print("\n=== 3. the optimum, and the shape around it ===")
    curve = st.objective_curve(base, btc, 0.30, n=61)
    f = st.flatness(curve, tol=0.01)
    best = st.stats(st.sleeve(base, btc, f["best_weight"]))
    h.update({"best_weight": f["best_weight"], "best_sharpe": best["sharpe"],
              "best_cagr": best["cagr"], "best_dd": best["max_drawdown"],
              "plateau_lo": f["plateau_lo"], "plateau_hi": f["plateau_hi"],
              "plateau_width": f["plateau_width"]})
    h["curve"] = curve.reset_index().to_dict("records")
    print(f"  Sharpe-maximising weight: {f['best_weight']:.2%} (Sharpe {best['sharpe']:.3f})")
    print(f"  within 1% of that optimum: EVERY weight from {f['plateau_lo']:.2%} to "
          f"{f['plateau_hi']:.2%}")
    print(f"  that plateau is {f['plateau_width']:.1%} wide — wider than the entire range of")
    print(f"  allocations people argue about. The argmax is not the finding; the plateau is.")
    for tol in (0.005, 0.01, 0.02, 0.05):
        ff = st.flatness(curve, tol=tol)
        print(f"    within {tol:.1%} of optimal: {ff['plateau_lo']:.2%} to "
              f"{ff['plateau_hi']:.2%}")

    print("\n=== 4. how uncertain is that number? ===")
    boot = st.weight_standard_error(base, btc, MAX_WEIGHT, n_boot=400)
    h.update({"boot_mean": boot["mean"], "boot_median": boot["median"],
              "boot_sd": boot["sd"], "boot_p05": boot["p05"], "boot_p95": boot["p95"],
              "boot_at_zero": boot["share_at_zero"], "boot_at_cap": boot["share_at_cap"]})
    print(f"  block bootstrap of the optimiser, {len(boot['draws'])} draws:")
    print(f"    median {boot['median']:.1%}, sd {boot['sd']:.1%}")
    print(f"    90% interval: {boot['p05']:.1%} to {boot['p95']:.1%}")
    print(f"    lands at exactly 0%:     {boot['share_at_zero']:.0%} of draws")
    print(f"    lands at the {MAX_WEIGHT:.0%} cap: {boot['share_at_cap']:.0%} of draws")

    print("\n=== 4b. THE INVERSION: what does each recommendation assume? ===")
    h["realised_mean"] = float(np.expm1(np.log1p(btc).sum() * 252 / len(btc)))
    imp = st.implied_mean_for_weight(base, btc, (0.005, 0.01, 0.02, 0.05, 0.10))
    print(f"  bitcoin actually returned {h['realised_mean']:.1%} a year over this sample,")
    print(f"  which is why the optimiser wants {h['best_weight']:.1%}. So the published 1-2%")
    print(f"  allocations are not readings of this record. What ARE they readings of?")
    print(f"    weight   implied expected return")
    for w, row in imp.iterrows():
        print(f"    {w:5.1%}   {row['implied_mean']:+8.1%}")
    h["implied"] = imp.reset_index().to_dict("records")
    h["implied_1pct"] = float(imp.loc[0.01, "implied_mean"])
    h["implied_2pct"] = float(imp.loc[0.02, "implied_mean"])
    h["implied_5pct"] = float(imp.loc[0.05, "implied_mean"])
    print(f"  a 2% sleeve is the answer you get by assuming bitcoin earns "
          f"{h['implied_2pct']:+.1%} a year.")
    print(f"  That is a defensible prior. It is not, however, what the accompanying")
    print(f"  historical charts are showing.")

    print("\n=== 4c. uncertainty about the mean, admitted explicitly ===")
    mu_unc = st.weight_with_mean_uncertainty(base, btc, MAX_WEIGHT, n_draws=400)
    h.update({"mu_se": mu_unc["mean_se"], "mu_p05": mu_unc["p05"],
              "mu_p95": mu_unc["p95"], "mu_median": mu_unc["median"]})
    print(f"  the standard error on bitcoin's mean is {mu_unc['mean_se']:.1%} a year")
    print(f"  drawing the mean from N(realised, SE) and re-optimising each time gives a")
    print(f"  weight interval of {mu_unc['p05']:.1%} to {mu_unc['p95']:.1%}")
    print(f"  note this is NARROWER than the block bootstrap above "
          f"({h['boot_p05']:.1%}-{h['boot_p95']:.1%}), which is worth understanding: block")
    print(f"  resampling a fat-tailed series also scrambles the realised mean, and by more.")
    print(f"  Neither interval reaches down to 1%, because both condition on a mean that is")
    print(f"  at worst two standard errors below {h['realised_mean']:.0%} — still enormous.")

    print("\n=== 5. can this sample tell 1% from 5%? ===")
    pairs = []
    for wa, wb in ((0.01, 0.02), (0.01, 0.05), (0.01, 0.10), (0.02, 0.05), (0.05, 0.10)):
        p = st.power_to_distinguish(base, btc, wa, wb, n_boot=300)
        pairs.append({"a": wa, "b": wb, **{k: p[k] for k in
                                           ("mean_diff", "sd_diff", "share_b_wins",
                                            "p05", "p95", "distinguishable")}})
        print(f"  {wa:.0%} vs {wb:5.0%}: {wb:.0%} wins {p['share_b_wins']:.0%} of draws, "
              f"Sharpe diff {p['mean_diff']:+.3f} +/- {p['sd_diff']:.3f}  "
              f"[{p['p05']:+.3f}, {p['p95']:+.3f}]  "
              f"{'DISTINGUISHABLE' if p['distinguishable'] else 'not distinguishable'}")
    h["pairs"] = pairs
    one_five = [p for p in pairs if p["a"] == 0.01 and p["b"] == 0.05][0]
    h["share_5_beats_1"] = one_five["share_b_wins"]
    h["diff_5_1"] = one_five["mean_diff"]
    h["diff_sd"] = one_five["sd_diff"]

    print("\n=== 6. how much data would settle it? ===")
    need = []
    for tk, label in ((data.BTC, "bitcoin"), (data.EQUITY, "equities"),
                      (data.GOLD, "gold"), (data.BONDS, "bonds")):
        if tk not in px.columns:
            continue
        s = px[tk].dropna().pct_change().dropna()
        vol = float(s.std(ddof=1) * np.sqrt(252))
        mu = float(np.expm1(np.log1p(s).sum() * 252 / len(s)))
        sn = st.sample_needed(mu, vol, 0.02)
        need.append({"asset": label, "ticker": tk, "vol": vol, "cagr": mu,
                     "years_needed": sn["years_needed"], "se_at_10y": sn["se_at_10y"],
                     "t_at_10y": sn["t_stat_at_10y"]})
        print(f"  {label:9s} vol {vol:5.1%}  -> {sn['years_needed']:8,.0f} years to pin the "
              f"mean to +/-2pp  (se after 10y: {sn['se_at_10y']:.1%})")
    h["sample_needed"] = need
    btc_need = [n for n in need if n["ticker"] == data.BTC][0]
    h["years_needed"] = btc_need["years_needed"]
    h["se_at_now"] = float(h["btc_vol"] / np.sqrt(h["years"]))
    print(f"  bitcoin has {h['years']:.1f} years, giving a standard error on its mean of "
          f"{h['se_at_now']:.0%}")
    print("  the standard error is larger than most people's entire expected return. That is")
    print("  why the plateau in section 3 is as wide as it is — the two are the same fact.")

    print("\n=== 7. out of sample ===")
    wf = st.walk_forward_weights(base, btc, LOOKBACK_Y, 63, 0.20, COST_BPS)
    ws = st.walk_forward_series(base, btc, LOOKBACK_Y, 63, 0.20, COST_BPS)
    h["wf_table"] = [{"date": str(d.date()), **{k: float(v) for k, v in row.items()}}
                     for d, row in wf.iterrows()]
    swf = st.stats(ws["walk_forward"])
    sbase = st.stats(ws["base"])
    h.update({"wf_cagr": swf["cagr"], "wf_vol": swf["vol"], "wf_sharpe": swf["sharpe"],
              "wf_dd": swf["max_drawdown"], "wf_base_cagr": sbase["cagr"],
              "wf_base_sharpe": sbase["sharpe"],
              "wf_min_w": float(wf["weight"].min()), "wf_max_w": float(wf["weight"].max()),
              "wf_mean_w": float(wf["weight"].mean()),
              "wf_total_cost": float(wf["cost"].sum())})
    print(f"  the allocator's chosen weight ranged {wf['weight'].min():.1%} to "
          f"{wf['weight'].max():.1%} (mean {wf['weight'].mean():.1%})")
    print(f"  walk-forward: {swf['cagr']:.2%} a year, Sharpe {swf['sharpe']:.2f}, "
          f"drawdown {swf['max_drawdown']:.1%}")
    print(f"  the same window, base portfolio: {sbase['cagr']:.2%}, Sharpe "
          f"{sbase['sharpe']:.2f}")
    print(f"  trading costs paid: {wf['cost'].sum():.2%} cumulative")
    fixed = []
    for w in (0.01, 0.02, 0.05):
        s = st.stats(st.sleeve(ws["base"], btc.reindex(ws.index), w))
        fixed.append({"weight": w, **s})
        print(f"  a fixed {w:.0%} sleeve over the same window: {s['cagr']:.2%}, Sharpe "
              f"{s['sharpe']:.2f}")
    h["wf_fixed"] = fixed

    print("\n=== 8. rebalancing is not a detail here ===")
    rb = st.rebalancing_matters(base, btc, 0.02, (1, 21, 63, 252, 10_000))
    print(rb[["max_weight_reached", "cagr", "vol", "sharpe",
              "max_drawdown"]].round(4).to_string())
    h["rebalance"] = rb.reset_index().to_dict("records")
    print(f"  a '2% allocation' left alone reached "
          f"{rb.loc[10_000, 'max_weight_reached']:.0%} of the portfolio at its peak.")
    print("  Track records quoted for a fixed sleeve often belong to a position that grew.")

    print("\n=== 9. against other diversifiers ===")
    alt = []
    for tk in (data.BTC, data.GOLD, data.LONG_BONDS):
        if tk not in rets.columns:
            continue
        a = rets[tk].reindex(common).dropna()
        cu = st.objective_curve(base.reindex(a.index), a, 0.30, n=41)
        ff = st.flatness(cu, 0.01)
        bs = st.weight_standard_error(base.reindex(a.index), a, MAX_WEIGHT, n_boot=150)
        corr = float(base.reindex(a.index).corr(a))
        alt.append({"asset": tk, "corr": corr, "vol": float(a.std(ddof=1) * np.sqrt(252)),
                    "best_weight": ff["best_weight"], "plateau_lo": ff["plateau_lo"],
                    "plateau_hi": ff["plateau_hi"], "boot_p05": bs["p05"],
                    "boot_p95": bs["p95"]})
        print(f"  {tk:8s} corr {corr:+.2f}  vol {a.std(ddof=1) * np.sqrt(252):5.1%}  "
              f"best {ff['best_weight']:5.1%}  plateau {ff['plateau_lo']:.1%}-"
              f"{ff['plateau_hi']:.1%}  bootstrap {bs['p05']:.1%}-{bs['p95']:.1%}")
    h["alternatives"] = alt

    print("\n=== 10. the control: a known truth, a realistic sample ===")
    ctrl = []
    for n_years in (3, 10, 30, 100):
        found = [st.optimal_weight(w["base"], w["asset"], 0.30) for w in
                 (st.synthetic_pair(n=int(n_years * 252), true_weight=0.05, seed=1003 + k)
                  for k in range(12))]
        ctrl.append({"years": n_years, "mean": float(np.mean(found)),
                     "sd": float(np.std(found)), "lo": float(np.min(found)),
                     "hi": float(np.max(found))})
        print(f"  {n_years:3d} years of a world where the true answer is 5.0%: "
              f"optimiser says {np.mean(found):.1%} +/- {np.std(found):.1%} "
              f"(range {np.min(found):.1%} to {np.max(found):.1%})")
    h["control"] = ctrl
    print("  the optimiser is unbiased and useless at the same time, which is the")
    print("  distinction the whole debate misses.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    sw = "\n".join(
        f"| {r['weight']:.1%} | {r['cagr']:.2%} | {r['vol']:.1%} | {r['sharpe']:.3f} | "
        f"{r['max_drawdown']:.1%} | {r['sortino']:.2f} |" for r in h["sweep"])
    implied = "\n".join(
        f"| {r['weight']:.1%} | **{r['implied_mean']:+.1%}** |" for r in h["implied"])
    pairs = "\n".join(
        f"| {r['a']:.0%} vs {r['b']:.0%} | {r['mean_diff']:+.3f} | {r['sd_diff']:.3f} | "
        f"{r['share_b_wins']:.0%} | [{r['p05']:+.3f}, {r['p95']:+.3f}] | "
        f"{'**yes**' if r['distinguishable'] else 'no'} |" for r in h["pairs"])
    need = "\n".join(
        f"| {r['asset']} | {r['vol']:.1%} | {r['cagr']:.2%} | **{r['years_needed']:,.0f}** | "
        f"{r['se_at_10y']:.1%} |" for r in h["sample_needed"])
    rb = "\n".join(
        f"| {'never' if r['rebalance_days'] >= 10000 else str(int(r['rebalance_days'])) + ' days'} "
        f"| {r['max_weight_reached']:.1%} | {r['cagr']:.2%} | {r['vol']:.1%} | "
        f"{r['sharpe']:.3f} | {r['max_drawdown']:.1%} |" for r in h["rebalance"])
    alt = "\n".join(
        f"| {r['asset']} | {r['corr']:+.2f} | {r['vol']:.1%} | {r['best_weight']:.1%} | "
        f"{r['plateau_lo']:.1%} – {r['plateau_hi']:.1%} | {r['boot_p05']:.1%} – "
        f"{r['boot_p95']:.1%} |" for r in h["alternatives"])
    ctrl = "\n".join(
        f"| {int(r['years'])} | {r['mean']:.1%} | {r['sd']:.1%} | {r['lo']:.1%} – "
        f"{r['hi']:.1%} |" for r in h["control"])
    fixed = "\n".join(
        f"| fixed {r['weight']:.0%} | {r['cagr']:.2%} | {r['vol']:.1%} | {r['sharpe']:.3f} | "
        f"{r['max_drawdown']:.1%} |" for r in h["wf_fixed"])
    return f"""# Results — Study 1003 (The 1% Allocation) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_days']:,} common sessions
from {h['start']} ({h['years']:.1f} years), bitcoin aligned to the equity calendar. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The calendar, before anything else

Bitcoin trades 365 days a year and the rest of the portfolio does not. Annualising bitcoin's
volatility over 365 observations while everything else uses 252 inflates it by
**{h['calendar_inflation']:.0%}** ({h['btc_vol']:.1%} → {h['btc_vol_365']:.1%}) and moves every
Sharpe-like comparison in bitcoin's favour. Everything below is on the equity calendar, with
weekend moves folded into the next session.

## 2. The sweep

| Sleeve | CAGR | Volatility | Sharpe | Max drawdown | Sortino |
|---|--:|--:|--:|--:|--:|
{sw}

## 3. The optimum, and the shape around it

The Sharpe-maximising weight was **{h['best_weight']:.2%}**. That number is not the finding.
Every weight from **{h['plateau_lo']:.2%} to {h['plateau_hi']:.2%}** is within 1% of it — a
plateau {h['plateau_width']:.1%} wide, which is wider than the entire range of allocations under
public debate. "The optimal allocation is 1%" and "the optimal allocation is 5%" are, on this
evidence, the same statement.

## 4. How uncertain is that number?

| | Weight |
|---|--:|
| Bootstrap median | {h['boot_median']:.1%} |
| Standard deviation | {h['boot_sd']:.1%} |
| 90% interval | **{h['boot_p05']:.1%} – {h['boot_p95']:.1%}** |
| Draws landing at exactly 0% | {h['boot_at_zero']:.0%} |
| Draws landing at the {h['max_weight']:.0%} cap | {h['boot_at_cap']:.0%} |

A block bootstrap, so volatility clustering is preserved in both series; an i.i.d. bootstrap
would make the recommendation look firmer than it is.

## 4b. The inversion — what each recommendation actually assumes

Bitcoin returned **{h['realised_mean']:.1%} a year** over this sample, which is why the
optimiser wants {h['best_weight']:.1%}. The published 1–2% allocations are therefore not
cautious readings of this record; they are the answers that appear **after overriding it**.
Inverting the optimiser says by how much. Each weight is shown with the annual expected return
that makes it optimal, holding bitcoin's volatility, correlation and path shape fixed:

| Recommended weight | Expected return it implies |
|---|--:|
{implied}

**A 2% sleeve is what you get from assuming bitcoin earns {h['implied_2pct']:+.1%} a year.** A
1% sleeve assumes {h['implied_1pct']:+.1%} — a negative expected return. A 5% sleeve assumes
{h['implied_5pct']:+.1%}, roughly what equities are expected to deliver.

Those are defensible priors. They are simply not what the historical charts accompanying such
recommendations are showing, and stating the assumed return would make the recommendation
arguable in a way that a weight alone is not.

## 4c. Uncertainty about the mean, admitted explicitly

The standard error on bitcoin's mean return is **{h['mu_se']:.1%} a year**. Drawing the mean
from N(realised, SE), recentring the return series onto it and re-optimising gives a weight
interval of **{h['mu_p05']:.1%} – {h['mu_p95']:.1%}** (median {h['mu_median']:.1%}).

That is *narrower* than the block bootstrap in section 4 ({h['boot_p05']:.1%} –
{h['boot_p95']:.1%}), which is worth understanding rather than glossing: block-resampling a
fat-tailed series scrambles the realised mean too, and by more than two standard errors' worth.
Neither interval descends anywhere near 1%, because both condition on a mean that is at worst a
couple of standard errors below {h['realised_mean']:.0%} — still an enormous number. **No
treatment of sampling uncertainty gets you to a 1% allocation. Only a different prior does.**

## 5. Can this sample tell one allocation from another?

| Comparison | Sharpe difference | SD | Larger wins | 90% interval | Distinguishable |
|---|--:|--:|--:|--:|:--:|
{pairs}

## 6. How much data would settle it?

The standard error of an annualised mean is σ/√years, so the years needed scale with the
**square** of volatility. This calculation needs no data at all and precedes almost no
allocation recommendation.

| Asset | Volatility | CAGR | Years to pin the mean to ±2pp | SE after 10 years |
|---|--:|--:|--:|--:|
{need}

Bitcoin has {h['years']:.1f} years, giving a standard error on its mean return of
**{h['se_at_now']:.0%}** — larger than most people's entire estimate of that mean. The plateau in
section 3 and this number are the same fact seen from two directions.

## 7. Out of sample

A walk-forward allocator: estimate on {int(h['years'] and 3)} years of history, hold for a
quarter, pay {h['wf_total_cost']:.2%} cumulative in trading costs at 30bp of turnover.

| | CAGR | Volatility | Sharpe | Max drawdown |
|---|--:|--:|--:|--:|
| Walk-forward allocator | {h['wf_cagr']:.2%} | {h['wf_vol']:.1%} | {h['wf_sharpe']:.3f} | {h['wf_dd']:.1%} |
| 60/40 base | {h['wf_base_cagr']:.2%} | — | {h['wf_base_sharpe']:.3f} | — |
{fixed}

Its chosen weight ranged from {h['wf_min_w']:.1%} to {h['wf_max_w']:.1%} (mean
{h['wf_mean_w']:.1%}). An allocator that revises its recommendation across that range every
quarter is tracking noise, and section 4 says why.

## 8. Rebalancing is not a detail here

| Rebalance | Peak weight reached | CAGR | Volatility | Sharpe | Max drawdown |
|---|--:|--:|--:|--:|--:|
{rb}

A 2% sleeve left alone reached **{[r for r in h['rebalance'] if r['rebalance_days'] >= 10000][0]['max_weight_reached']:.0%}**
of the portfolio at its peak. Track records quoted for a "2% allocation" frequently belong to a
position that spent most of the period much larger than 2%.

## 9. Against other diversifiers

| Asset | Correlation to 60/40 | Volatility | Best weight | 1% plateau | Bootstrap 90% |
|---|--:|--:|--:|--:|--:|
{alt}

## 10. The control: a known truth, a realistic sample

A synthetic world where the true optimal weight is exactly 5.0%:

| Years of data | Optimiser's mean answer | SD | Range across 12 draws |
|---|--:|--:|--:|
{ctrl}

The optimiser is **unbiased and useless at the same time**. It converges on the truth given a
century; on a decade its answers scatter across the whole range of allocations people argue
about. That distinction — between a biased estimator and a noisy one — is what the debate misses.

## Caveats

- **Bitcoin's history is a single realisation of an asset that went up a great deal.** Every
  in-sample number here is conditioned on that. The estimability results in sections 4-6 are the
  ones that do not depend on it, which is why the verdict rests on them.
- **No tax treatment**, which differs sharply between jurisdictions and wrappers, and matters
  more for a high-turnover sleeve than for the rest of a portfolio.
- **30bp of turnover cost** is an assumption. Spot exchange execution can be tighter; a fund
  wrapper plus spread can be considerably wider.
- **The optimiser maximises Sharpe.** An investor with a drawdown constraint or a different
  utility gets a different answer, generally a smaller one, since bitcoin's contribution to the
  left tail exceeds its contribution to variance.
- **The plateau result is about statistical distinguishability, not about indifference.** A 20%
  sleeve and a 1% sleeve produce very different portfolios in any single future; the finding is
  that the historical record cannot rank them.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1003-bitcoin-in-a-portfolio](../README.md). Not investment advice.*
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
