"""Real-tape verification — Study 1022 (Behind the Curve). Regenerates docs/results.md.

Builds the Taylor (1993) rule from final-vintage US macro (statsmodels ``macrodata``) with two
output gaps an investor could have computed at the time — a one-sided HP gap on log real GDP
and an Okun-style unemployment gap — every macro print lagged one quarter for release. The
Taylor gap (T-bill minus the rule) is then asked to forecast next-quarter equity excess returns
(Fama-French total return) and AAA-yield changes (Moody's, via ``arch``), with Newey-West
errors, the Stambaugh (1999) correction, a simulation null that keeps the data's innovation
correlation, and 4- and 8-quarter versions whose inference comes from the same simulation.
Finally a sign-of-the-gap equity/bills rule is raced against buy-and-hold, excess of cash,
net of costs.

    python studies/1022-taylor-rule-gap/examples/verify.py

All tapes ship inside installed packages and are SHA-256 pinned; nothing touches the network.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from taylorgap import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

GAPS = {"tgap_hp": "one-sided HP gap", "tgap_u": "unemployment gap"}
LEGS = {"eq_xs": "equity excess return", "d_aaa": "AAA yield change"}
N_SIM = 4000
COST_BP = 10.0


def _q(ts) -> str:
    p = pd.Timestamp(ts).to_period("Q")
    return f"{p.year}Q{p.quarter}"


def report() -> dict:
    t0 = time.time()
    q = data.load_quarterly()
    p = st.real_panel()
    fp_cols = ["realgdp", "cpi", "tbilrate", "unemp", "mkt", "rf", "aaa"]
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(q[fp_cols]),
               "provenance": data.provenance()}
    h["start"] = _q(p.index[0])
    h["end"] = _q(p["tgap_hp"].dropna().index[-1])
    h["n_quarters"] = int(p["tgap_hp"].notna().sum())
    print(f"as-of {data.AS_OF}   fingerprint {h['fingerprint']}")
    print(f"  signals {h['start']} -> {h['end']} ({h['n_quarters']} quarters), "
          f"returns to {_q(p.index[-1])}")
    for k, v in h["provenance"].items():
        print(f"  {k}: {v}")

    print("\n=== 1. the rule and the gaps ===")
    desc = p[["i", "pi", "gap_hp", "gap_u", "gap_hind", "tgap_hp", "tgap_u",
              "tgap_hind"]].describe().T[["mean", "std", "min", "max"]]
    print(desc.round(2).to_string())
    h["describe"] = desc.reset_index().rename(columns={"index": "var"}).to_dict("records")
    ag = st.gap_agreement(p)
    h["agree"] = ag
    print({k: round(v, 3) for k, v in ag.items()})
    h["rho_hp"] = st.ar1(p["tgap_hp"].dropna().to_numpy())["rho"]
    h["rho_u"] = st.ar1(p["tgap_u"].dropna().to_numpy())["rho"]

    print("\n=== 2. one-quarter predictive regressions (Newey-West, 4 lags) ===")
    rows = []
    for sig, glab in list(GAPS.items()) + [("tgap_hind", "hindsight HP gap (look-ahead)")]:
        for tgt in ("eq_xs", "d_aaa", "bond_xs"):
            r = st.predictive_regression(p, sig, tgt, 1)
            rows.append({"gap": glab, "signal": sig, "target": tgt, "slope": r["slope"],
                         "t": r["t"], "r2": r["r2"], "n": r["n"]})
            print(f"  {sig:10s} {tgt:8s} slope {r['slope']:+.4f}  t {r['t']:+.2f}  "
                  f"R2 {r['r2']:.3f}  n {r['n']}")
    h["one_q"] = rows

    print("\n=== 3. persistence: Stambaugh correction ===")
    stb = []
    for sig in GAPS:
        for tgt in LEGS:
            s = st.stambaugh_correction(p, sig, tgt)
            stb.append({"signal": sig, "target": tgt, **s})
            print(f"  {sig:8s} {tgt:6s} rho {s['rho']:.3f} (corrected {s['rho_c']:.3f}) "
                  f"corr(u,v) {s['corr_uv']:+.2f}  slope {s['slope']:+.4f} -> "
                  f"{s['slope_c']:+.4f}  t {s['t']:+.2f} -> {s['t_c']:+.2f}")
    h["stambaugh"] = stb

    print(f"\n=== 4. simulation null ({N_SIM} paths) and long horizons ===")
    sims = {}
    specs = []
    for sig, glab in GAPS.items():
        for tgt, llab in LEGS.items():
            sm = st.simulation_null(p, sig, tgt, n_sim=N_SIM)
            sims[(sig, tgt)] = sm
            for hz in st.HORIZONS:
                d = sm[hz]
                specs.append({"label": f"{llab}, {glab}, {hz}q", "signal": sig, "target": tgt,
                              "h": hz, "slope": d["slope"], "t": d["t"], "r2": d["r2"],
                              "n": d["n"], "p_sim": d["p_two"], "p_claim": d["p_claim"],
                              "null_lo": d["null_t_q"][0], "null_hi": d["null_t_q"][2]})
                print(f"  {sig:8s} {tgt:6s} h={hz}: slope {d['slope']:+.4f} t {d['t']:+.2f} "
                      f"null 95% [{d['null_t_q'][0]:+.2f}, {d['null_t_q'][2]:+.2f}] "
                      f"p2 {d['p_two']:.3f} p_claim {d['p_claim']:.3f}")
    h["specs"] = specs

    print("\n=== 5. behind vs ahead, and the regime split ===")
    cm = []
    for sig in GAPS:
        for tgt in LEGS:
            c = st.conditional_means(p, sig, tgt)
            rg = st.regime_split(p, sig, tgt)
            cm.append({"signal": sig, "target": tgt, **c, **{f"rg_{k}": v for k, v in rg.items()}})
            print(f"  {sig:8s} {tgt:6s} behind {c['mean_behind']:+.4f} (n {c['n_behind']}) "
                  f"ahead {c['mean_ahead']:+.4f} (n {c['n_ahead']}) diff t {c['t_diff']:+.2f} | "
                  f"pre-1987Q3 slope {rg['slope_pre']:+.4f} (t {rg['t_pre']:+.2f}) post "
                  f"{rg['slope_post']:+.4f} (t {rg['t_post']:+.2f}) diff t {rg['t_diff']:+.2f}")
    h["conditional"] = cm

    print(f"\n=== 6. the timing rule (equity when behind, bills when ahead; {COST_BP:.0f} bp) ===")
    trades = []
    for sig, lab in (("tgap_hp", "one-sided HP gap"), ("tgap_u", "unemployment gap"),
                     ("tgap_hp_dm", "HP gap, demeaned in real time"),
                     ("tgap_u_dm", "unemployment gap, demeaned in real time")):
        bt = st.timing_backtest(p, sig, COST_BP)
        s = st.backtest_summary(bt)
        b = st.sharpe_diff_bootstrap(bt["strat_xs_net"].to_numpy(), bt["bh_xs"].to_numpy())
        trades.append({"signal": sig, "label": lab, **s, "ci_lo": b["ci_lo"],
                       "ci_hi": b["ci_hi"], "p_diff": b["p_one_sided"]})
        print(f"  {lab:40s} invested {s['share_invested']:.0%} switches/yr "
              f"{s['switches_per_year']:.2f} SR net {s['sr_net']:.2f} vs B&H {s['sr_bh']:.2f} "
              f"diff {s['sr_diff_net']:+.2f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}] "
              f"p {b['p_one_sided']:.2f} alpha {s['alpha_ann']:+.2%} (t {s['alpha_t']:+.2f})")
    h["trades"] = trades
    sweep = st.cost_sweep(p, "tgap_hp")
    print(sweep.round(3).to_string())
    h["sweep"] = sweep.reset_index().to_dict("records")

    print("\n=== 7. synthetic control (planted vs null — machinery proof, NOT market evidence) ===")
    ctrl = []
    for s_ in (1.0, 0.0):
        w, truth = data.synthetic_quarterly(n_quarters=h["n_quarters"], signal_strength=s_)
        sm = st.simulation_null(w, "tgap", "eq_xs", n_sim=1000)
        sb = st.simulation_null(w, "tgap", "d_aaa", n_sim=1000)
        tr = st.backtest_summary(st.timing_backtest(w, "tgap", COST_BP))
        ctrl.append({"signal_strength": s_, "beta_eq": truth["beta_eq"],
                     "eq_slope": sm[1]["slope"], "eq_t": sm[1]["t"], "eq_p": sm[1]["p_two"],
                     "bond_slope": sb[1]["slope"], "bond_t": sb[1]["t"], "bond_p": sb[1]["p_two"],
                     "sr_diff_net": tr["sr_diff_net"]})
        print(f"  strength {s_:.0f}: equity slope {sm[1]['slope']:+.4f} (planted "
              f"{truth['beta_eq']:+.4f}) t {sm[1]['t']:+.2f} p {sm[1]['p_two']:.3f} | bond "
              f"t {sb[1]['t']:+.2f} p {sb[1]['p_two']:.3f} | SR diff {tr['sr_diff_net']:+.2f}")
    h["control"] = ctrl

    # ---- the verdict inputs ----
    s_hp = next(x for x in stb if x["signal"] == "tgap_hp" and x["target"] == "eq_xs")
    eq_sim = sims[("tgap_hp", "eq_xs")][1]
    eq_u = sims[("tgap_u", "eq_xs")][1]
    bd = sims[("tgap_hp", "d_aaa")][1]
    tr0 = trades[0]
    h["eq_primary"] = {"slope": s_hp["slope"], "t": s_hp["t"], "slope_c": s_hp["slope_c"],
                       "t_c": s_hp["t_c"], "p_sim": eq_sim["p_two"], "rho": s_hp["rho"],
                       "corr_uv": s_hp["corr_uv"]}
    h["eq_u_slope"] = eq_u["slope"]
    h["eq_u_t"] = eq_u["t"]
    h["bond_primary"] = {"slope": bd["slope"], "t": bd["t"], "p_sim": bd["p_two"],
                         "p_claim": bd["p_claim"]}
    rg = next(x for x in cm if x["signal"] == "tgap_hp" and x["target"] == "d_aaa")
    h["bond_regime"] = {k: rg[f"rg_{k}"] for k in ("slope_pre", "t_pre", "slope_post",
                                                    "t_post", "t_diff")}
    h["trade"] = {k: tr0[k] for k in ("share_invested", "switches_per_year", "sr_net", "sr_bh",
                                      "sr_diff_net", "ci_lo", "ci_hi", "p_diff", "ann_xs_net",
                                      "ann_xs_bh", "alpha_ann", "alpha_t", "breakeven_bp")}
    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = time.time() - t0
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']:.1f}s")
    return h


def _f(x, nd=2, pct=False, sign=True):
    if pct:
        return f"{x:+.2%}" if sign else f"{x:.2%}"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    pv = h["provenance"]
    ag = h["agree"]
    tgt_lab = {"eq_xs": "equity excess return (decimal/q)", "d_aaa": "AAA yield change (pp/q)",
               "bond_xs": "approx. long-bond excess return (decimal/q)"}
    desc = "\n".join(
        f"| `{r['var']}` | {r['mean']:.2f} | {r['std']:.2f} | {r['min']:.2f} | {r['max']:.2f} |"
        for r in h["describe"])
    oneq = "\n".join(
        f"| {r['gap']} | {tgt_lab[r['target']]} | {r['slope']:+.4f} | {r['t']:+.2f} | "
        f"{r['r2']:.3f} | {r['n']} |" for r in h["one_q"])
    stb = "\n".join(
        f"| {GAPS[r['signal']]} | {LEGS[r['target']]} | {r['rho']:.3f} | {r['rho_c']:.3f} | "
        f"{r['corr_uv']:+.2f} | {r['slope']:+.4f} | {r['bias']:+.5f} | {r['slope_c']:+.4f} | "
        f"{r['t']:+.2f} | **{r['t_c']:+.2f}** |" for r in h["stambaugh"])
    sp = "\n".join(
        f"| {LEGS[r['target']]} | {GAPS[r['signal']]} | {r['h']} | {r['slope']:+.4f} | "
        f"{r['t']:+.2f} | [{r['null_lo']:+.2f}, {r['null_hi']:+.2f}] | **{r['p_sim']:.3f}** | "
        f"{r['p_claim']:.3f} |" for r in h["specs"])
    cm = "\n".join(
        f"| {GAPS[r['signal']]} | {LEGS[r['target']]} | {r['mean_behind']:+.4f} ({r['n_behind']}) | "
        f"{r['mean_ahead']:+.4f} ({r['n_ahead']}) | {r['t_diff']:+.2f} | "
        f"{r['rg_slope_pre']:+.4f} ({r['rg_t_pre']:+.2f}) | {r['rg_slope_post']:+.4f} "
        f"({r['rg_t_post']:+.2f}) | {r['rg_t_diff']:+.2f} |" for r in h["conditional"])
    trd = "\n".join(
        f"| {r['label']} | {r['share_invested']:.0%} | {r['switches_per_year']:.2f} | "
        f"{r['ann_xs_gross']:.2%} | {r['ann_xs_net']:.2%} | {r['ann_xs_bh']:.2%} | "
        f"{r['sr_gross']:.2f} | **{r['sr_net']:.2f}** | {r['sr_bh']:.2f} | "
        f"{r['sr_diff_net']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_diff']:.2f} | "
        f"{r['alpha_ann']:+.2%} ({r['alpha_t']:+.2f}) |" for r in h["trades"])
    sw = "\n".join(
        f"| {int(r['cost_bp'])} | {r['sr_net']:.3f} | {r['sr_bh']:.3f} | {r['ann_xs_net']:.2%} |"
        for r in h["sweep"])
    e0 = next(r for r in h["conditional"] if r["signal"] == "tgap_hp" and r["target"] == "eq_xs")
    if e0["mean_ahead"] > e0["mean_behind"]:
        eq_split = (f"For equities the raw split runs *against* the claim: quarters after the Fed "
                    f"was \"ahead of the curve\" earned {e0['mean_ahead']:+.2%} on average, after "
                    f"\"behind\" quarters {e0['mean_behind']:+.2%} (HAC *t* of the difference "
                    f"{e0['t_diff']:+.2f}).")
    else:
        eq_split = (f"For equities the raw split leans the claim's way: {e0['mean_behind']:+.2%} "
                    f"after \"behind\" quarters vs {e0['mean_ahead']:+.2%} after \"ahead\" "
                    f"(HAC *t* of the difference {e0['t_diff']:+.2f}).")
    n_pass = sum(1 for r in h["specs"] if r["slope"] < 0 and r["p_sim"] < 0.05)
    n_pass = "None" if n_pass == 0 else str(n_pass)
    best = min(h["specs"], key=lambda r: r["p_sim"])
    ctl = "\n".join(
        f"| {r['signal_strength']:.0f} | {r['beta_eq']:+.4f} | {r['eq_slope']:+.4f} | "
        f"{r['eq_t']:+.2f} | {r['eq_p']:.3f} | {r['bond_t']:+.2f} | {r['bond_p']:.3f} | "
        f"{r['sr_diff_net']:+.2f} |" for r in h["control"])
    return f"""# Results — Study 1022 (Behind the Curve) on the real quarterly tape

*Generated by [`examples/verify.py`](../examples/verify.py). Signals {h['start']} → {h['end']}
({h['n_quarters']} quarters), each forecasting the following quarter. As-of **{h['as_of']}** (the
last full quarter any number uses: the 2009Q4 return after the last macro print, 2009Q3);
fingerprint `{h['fingerprint']}`.*

## 0. Data and provenance

| Tape | Package | Pin (SHA-256) | Used as |
|---|---|---|---|
| `macrodata` (1959Q1–2009Q3) | statsmodels {pv['statsmodels']} | `{pv['macrodata_sha256'][:16]}…` | real GDP, CPI, **3-month T-bill (policy-rate proxy)**, unemployment — **final vintage** |
| `frenchdata` (monthly) | arch {pv['arch']} | `{pv['frenchdata_sha256'][:16]}…` | market **total return** and 1-month T-bill, compounded to quarters |
| `default` (monthly) | arch {pv['arch']} | `{pv['default_sha256'][:16]}…` | Moody's AAA yield at each quarter-end |

- **Taylor (1993):** `i* = 2 + π + 0.5(π − 2) + 0.5·gap`, π = four-quarter CPI inflation.
  **Taylor gap** = `i − i*` with `i` the quarterly-average 3-month T-bill. Negative = Fed
  **behind the curve** (too easy).
- **Real-time gaps:** (a) one-sided HP (λ = 1600) on 100·ln real GDP, re-estimated every quarter
  on data to date, ≥ 40 quarters of history; (b) Okun-style `−2 × (u − trailing 40-quarter mean
  of u)`. Every macro input is **lagged one quarter** for release; the T-bill is not (a market
  price, known at quarter-end).
- **Long-bond return (approximation):** `r ≈ y/4 − D_mod(y)·Δy + ½·C(y)·Δy²` for a
  {data.BOND_MATURITY_YEARS}-year par bond at the start-of-quarter AAA yield. It ignores
  roll-down, call features and credit migration. The bond leg's *headline* uses the yield change
  itself, which needs no approximation.
- **Orphanides (2001):** these are **final-vintage** GDP figures. The output gap a policymaker
  or investor actually saw in, say, 1973 was revised by several points later on. The one-sided
  filter removes *look-ahead in the filter*; it cannot remove *revisions in the data*. Every
  result here is an **upper bound** on what was knowable in real time.

## 1. The rule and the gaps

| Variable | Mean | Sd | Min | Max |
|---|--:|--:|--:|--:|
{desc}

By the T-bill measure the Fed was behind the curve **{ag['share_behind_hp']:.0%}** of quarters
(HP gap) and **{ag['share_behind_u']:.0%}** (unemployment gap). Part of that is the proxy: the
T-bill typically trades below the fed funds rate, which pushes the gap down by a fraction of a
point — a level shift that does not affect any regression slope, only the sign rule (§6 adds a
real-time-demeaned version for that reason). The Taylor gap is highly persistent (AR(1)
ρ = {h['rho_hp']:.3f} HP, {h['rho_u']:.3f} unemployment).

**Orphanides in miniature.** The real-time HP gap correlates only **{ag['corr_gap_hp_hind']:.2f}**
with the full-sample (hindsight) HP gap and differs from it by {ag['mean_abs_gap_rev']:.2f} points
on average; the two real-time gaps correlate {ag['corr_gap_hp_u']:.2f} with each other. The
*Taylor* gaps agree in sign {ag['sign_agree_tgap_hp_hind']:.0%} of the time (real-time vs
hindsight HP) because inflation and the policy rate dominate the level. And this is the *small*
part of Orphanides' point — data revisions, which final-vintage data cannot show, are the larger.

## 2. One-quarter predictive regressions

`target[t+1] = a + b · TaylorGap[t] + e`, Newey-West (4 lags). **The claim predicts b < 0 for
both legs** (behind the curve → higher stock returns, rising yields).

| Gap | Target | Slope b | NW t | R² | n |
|---|---|--:|--:|--:|--:|
{oneq}

The hindsight rows use a full-sample HP gap and are **not tradable**; they are there to show that
look-ahead in the filter does not rescue the equity leg either.

## 3. Persistence — the Stambaugh correction

| Gap | Leg | ρ | ρ corrected | corr(u, v) | b | bias | b corrected | t | t corrected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
{stb}

`bias = −γ(1 + 3ρ_c)/T`, γ = cov(u, v)/var(v). The innovation correlation is small for equities,
so the correction barely moves the slope; for the bond leg it shades the *t* toward zero.

## 4. Simulation null and long horizons

{N_SIM} paths per row under H0 (no predictability): an AR(1) gap with bias-corrected persistence,
return and gap innovations resampled **as pairs** so the null keeps the data's correlation, and
the identical 1-, 4- and 8-quarter regressions (NW lags 4, 8, 16) rerun on every path — so the
overlapping-horizon p-values are correct by construction. p (two-sided) = P(|t\\*| ≥ |t|);
p (claim) = P(t\\* ≤ t), one-sided in the claim's direction.

| Leg | Gap | h (q) | Slope | NW t | Null 95% t band | p (two-sided) | p (claim) |
|---|---|--:|--:|--:|:--:|--:|--:|
{sp}

## 5. Behind vs ahead, and the regime split

Mean next-quarter value when the gap is negative (behind) vs positive (ahead), with the HAC *t*
of the difference; then the slope before and after 1987Q3 (Greenspan; Taylor's own sample was
1987–92) with the HAC *t* of the interaction.

| Gap | Leg | Behind: mean (n) | Ahead: mean (n) | t diff | Slope pre (t) | Slope post (t) | t diff |
|---|---|--:|--:|--:|--:|--:|--:|
{cm}

{eq_split}

## 6. Could you trade it?

Equity when the gap at *t* is negative, T-bills otherwise, position held over *t+1* (one lag;
the macro inside the gap is already a quarter old). Costs {COST_BP:.0f} bp one-way × traded NAV.
All returns **in excess of the T-bill**; Sharpe annualised (×2); the Sharpe difference carries a
circular block bootstrap (8-quarter blocks, 2,000 resamples).

| Rule | Invested | Switches/yr | Excess/yr gross | net | B&H | SR gross | SR net | SR B&H | ΔSR net [95% CI] | one-sided p (H₁: ΔSR > 0) | Timing α/yr (t) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{trd}

Cost sweep, HP-gap rule:

| Cost (bp one-way) | SR net | SR B&H | Excess/yr net |
|--:|--:|--:|--:|
{sw}

The rule switches rarely, so costs are not what decides it; the gross edge is already negative.

## 7. Synthetic control — machinery proof, not market evidence

The same pipeline on the synthetic world (`data.synthetic_quarterly`, {h['n_quarters']} quarters,
same persistence class, innovation correlation −0.4). Strength 1 plants b = −0.012 per point;
strength 0 is the matched null.

| Strength | Planted b | Estimated b | t | p (sim) | Bond t | Bond p (sim) | ΔSR net |
|--:|--:|--:|--:|--:|--:|--:|--:|
{ctl}

The harness banks the planted slope on both legs and stays quiet on the null, so the real-tape
"nothing" is not a pipeline that cannot see.

## Caveats

- **Final-vintage macro (Orphanides 2001).** The biggest caveat and it cuts *in favour* of the
  claim: real-time GDP gaps were noisier and often wrong-signed. A null on revised data is a
  stronger null in real time.
- **T-bill, not fed funds.** The 3-month T-bill is a proxy for the policy rate. It moves with
  the funds rate but sits below it and anticipates it by weeks. A level offset changes the
  sign rule (hence the demeaned variant), not the regression slopes.
- **The rule's coefficients are Taylor's 1993 ones.** Taylor fitted them, loosely, to 1987–92.
  Applying them to the 1970s is anachronistic — but that is exactly how the claim is used.
- **CPI, not the GDP deflator,** and four-quarter inflation. Taylor used the deflator; CPI is
  the version an investor watches and is barely revised.
- **The bond return is approximated** from yield changes on a 20-year par bond. The bond leg's
  verdict rests on the yield change itself.
- **One sample, 1969–2009, ~160 quarters.** A sample this short with a highly persistent regressor
  is exactly where Stambaugh bias and over-rejection bite; the simulation null is the honest
  reference, and it is what the verdict uses.
- **Multiple specifications.** Twelve (2 gaps × 3 horizons × 2 legs) were run and all are
  reported. {n_pass} clear a two-sided simulation-null p < 0.05 with the claim's sign; the
  closest is {best['label']} (*t* = {best['t']:+.2f}, p = {best['p_sim']:.3f}, one-sided
  {best['p_claim']:.3f}). Four regime-split tests were also run; one nominally significant
  interaction among them is what chance alone delivers about one time in five.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [1022-taylor-rule-gap](../README.md).
Not investment advice.*
"""


def main() -> None:
    h = report()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    v = h["_verdict"]
    head = {"as_of": h["as_of"], "fingerprint": h["fingerprint"], "signal": v["signal"],
            "trad": v["trad"], "one_sentence": v["one_sentence"],
            "n_quarters": h["n_quarters"], "start": h["start"], "end": h["end"],
            "eq_primary": h["eq_primary"], "eq_u_slope": h["eq_u_slope"],
            "eq_u_t": h["eq_u_t"], "bond_primary": h["bond_primary"], "trade": h["trade"],
            "agree": h["agree"], "rho_hp": h["rho_hp"], "runtime_s": h["runtime_s"]}
    print("##HEADLINE## " + json.dumps(head, default=float))


if __name__ == "__main__":
    main()
