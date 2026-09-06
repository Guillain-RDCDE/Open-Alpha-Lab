"""Real-tape verification — Study 993 (Down Hurts More). Regenerates docs/results.md.

Measures the volatility response to up and down moves four ways — a forward sign
split, the same split matched on move size, an EGARCH gamma, and the return/volatility-change
correlation — block-bootstraps the difference, fits the news-impact curve's vertex, runs the
lead-lag test that distinguishes leverage from volatility feedback, and repeats everything on
gold and Bitcoin, which have no balance sheets for leverage to work through.

    python studies/993-leverage-effect-asymmetry/examples/verify.py            # cache-only
    python studies/993-leverage-effect-asymmetry/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from downhurts import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


HORIZON = 5
WINDOW = 21


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "horizon": HORIZON, "window": WINDOW,
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
        a = st.annualisation_factor(s)
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.std() * np.sqrt(a):.0%}  skew {s.skew():+.2f}")
    lead = data.EQUITY
    r = assets[lead]
    h["asset"] = lead
    h["n_days"] = int(len(r))
    h["n_assets"] = int(len(assets))

    print(f"\n=== 1. the simple version ({lead}) ===")
    ss = st.sign_split(r, WINDOW, HORIZON)
    h.update({"vol_after_up": ss["vol_after_up"], "vol_after_down": ss["vol_after_down"],
              "ratio": ss["ratio"], "naive_t": ss["naive_t"]})
    print(f"  after an up day   ({ss['n_up']:,} of them): next {HORIZON}d vol "
          f"{ss['vol_after_up']:.1%}")
    print(f"  after a down day  ({ss['n_down']:,} of them): next {HORIZON}d vol "
          f"{ss['vol_after_down']:.1%}")
    print(f"  ratio {ss['ratio']:.3f}, naive t {ss['naive_t']:+.2f}")

    print("\n=== 2. but down days are bigger. matched on size: ===")
    mm = st.magnitude_matched_split(r, HORIZON)
    print(mm.round(4).to_string())
    h["matched"] = mm.reset_index().to_dict("records")
    h["matched_ratio"] = float(mm["ratio"].mean())
    print(f"  average within-bucket ratio: {h['matched_ratio']:.3f} "
          f"(unmatched was {ss['ratio']:.3f})")
    print("  down days ARE bigger on average, and big moves are followed by volatility "
          "regardless of sign — this bucket comparison removes that channel")

    print("\n=== 3. honest inference ===")
    bs = st.bootstrap_asymmetry(r, HORIZON, n_boot=800)
    h.update({"boot_t": bs["t"], "boot_sd": bs["boot_sd"], "boot_lo": bs["lo"],
              "boot_hi": bs["hi"], "difference": bs["difference"]})
    print(f"  difference {bs['difference']:+.2%}, block-bootstrap sd {bs['boot_sd']:.2%}")
    print(f"  t {bs['t']:+.2f} (naive was {ss['naive_t']:+.2f}), 95% CI "
          f"[{bs['lo']:+.2%}, {bs['hi']:+.2%}], {bs['share_positive']:.0%} of resamples "
          f"positive")

    print("\n=== 4. the news-impact curve ===")
    nic = st.news_impact_curve(r, HORIZON)
    cm = st.curve_minimum(nic)
    h["nic"] = nic.reset_index().to_dict("records")
    h.update({"vertex_z": cm["vertex_z"], "curvature": cm["curvature"],
              "nic_r2": cm["r2"]})
    print(f"  fitted quadratic: vertex at z = {cm['vertex_z']:+.3f}, curvature "
          f"{cm['curvature']:.4f}, R2 {cm['r2']:.2f}")
    print("  a symmetric response puts the vertex at zero; the leverage story shifts it "
          "toward positive returns")

    print("\n=== 5. the parametric version ===")
    eg = st.egarch_asymmetry(r)
    h.update({"egarch_gamma": eg["gamma"], "egarch_alpha": eg["alpha"],
              "egarch_beta": eg["beta"], "egarch_ratio": eg["response_ratio"]})
    print(f"  EGARCH(1,1): omega {eg['omega']:.4f}, beta {eg['beta']:.4f}, "
          f"alpha {eg['alpha']:.4f}, gamma {eg['gamma']:+.4f}")
    print(f"  a -1 sigma shock raises volatility {eg['response_ratio']:.3f}x as much as a "
          f"+1 sigma shock")
    ca = st.correlation_asymmetry(r, WINDOW)
    h["corr_r_dvol"] = ca["corr"]
    print(f"  correlation of returns with volatility CHANGES: {ca['corr']:+.3f} "
          f"(Spearman {ca['corr_spearman']:+.3f})")

    print("\n=== 6. the control that decides the mechanism ===")
    p = st.panel(assets, HORIZON, WINDOW)
    print(p.round(4).to_string())
    h["panel"] = p.reset_index().to_dict("records")
    h["gold_ratio"] = float(p.loc[data.GOLD, "ratio"]) if data.GOLD in p.index else np.nan
    h["crypto_ratio"] = float(p.loc[data.CRYPTO, "ratio"]) if data.CRYPTO in p.index \
        else np.nan
    print(f"\n  Gold has no balance sheet. Its ratio: {h['gold_ratio']:.3f}")
    print(f"  Bitcoin has no balance sheet. Its ratio: {h['crypto_ratio']:.3f}")
    print(f"  Equities: {h['ratio']:.3f}")
    print("  -> if the mechanism were financial leverage, the first two would be 1.00")

    print("\n=== 7. which comes first? ===")
    ll = st.lead_lag_asymmetry(r, WINDOW)
    for lag, row in ll.iterrows():
        mark = " <-" if abs(lag) <= 3 else ""
        print(f"  {int(lag):+3d}  {row['correlation']:+.4f}   {row['description']}{mark}")
    w = st.which_story(ll)
    h["lead_lag"] = ll.reset_index().to_dict("records")
    h.update({"leverage_side": w["leverage_side"], "feedback_side": w["feedback_side"],
              "contemporaneous": w["contemporaneous"], "leans": w["leans"]})
    print(f"  return-leads-volatility (leverage): {w['leverage_side']:+.4f}")
    print(f"  volatility-leads-return (feedback): {w['feedback_side']:+.4f}")
    print(f"  contemporaneous: {w['contemporaneous']:+.4f}")
    print(f"  -> leans {w['leans']}")

    print("\n=== 8. synthetic control ===")
    ctrl = []
    for gamma in (0.0, -0.05, -0.10, -0.20):
        ratios, boots, gammas, naive = [], [], [], []
        for k in range(3):
            sim = st.synthetic_returns(n=min(len(r), 3000), gamma=gamma, seed=993 + k)
            s2 = st.sign_split(sim, WINDOW, HORIZON)
            ratios.append(s2["ratio"])
            naive.append(abs(s2["naive_t"]) >= 2)
            boots.append(abs(st.bootstrap_asymmetry(sim, HORIZON, n_boot=200)["t"]) >= 2)
            gammas.append(st.fit_egarch(sim)["gamma"])
        ctrl.append({"planted_gamma": gamma, "ratio": float(np.mean(ratios)),
                     "naive_rejects": float(np.mean(naive)),
                     "boot_rejects": float(np.mean(boots)),
                     "recovered_gamma": float(np.mean(gammas))})
        print(f"  gamma {gamma:+.2f}: ratio {np.mean(ratios):.3f}, naive rejects "
              f"{np.mean(naive):.0%}, bootstrap rejects {np.mean(boots):.0%}, "
              f"EGARCH recovers {np.mean(gammas):+.3f}")
    h["control"] = ctrl
    h["null_naive_reject"] = float(ctrl[0]["naive_rejects"])
    h["null_boot_reject"] = float(ctrl[0]["boot_rejects"])
    print("  read the first row: that is the false-positive rate on a world with NO asymmetry")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    matched = "\n".join(
        f"| {r['bucket']} | {r['mean_abs_move']:.2%} | {int(r['n_up'])} | {int(r['n_down'])} | "
        f"{r['vol_after_up']:.1%} | {r['vol_after_down']:.1%} | **{r['ratio']:.3f}×** |"
        for r in h["matched"])
    panel = "\n".join(
        f"| {r['asset']} | {int(r['n']):,} | {r['vol_after_up']:.1%} | "
        f"{r['vol_after_down']:.1%} | **{r['ratio']:.3f}×** | {r['naive_t']:+.1f} | "
        f"{r['corr_r_dvol']:+.3f} | {r['egarch_gamma']:+.4f} | {r['vertex_z']:+.2f} |"
        for r in h["panel"])
    ll = "\n".join(
        f"| {int(r['lag']):+d} | {r['correlation']:+.4f} | {r['description']} |"
        for r in h["lead_lag"] if abs(int(r["lag"])) <= 5)
    ctrl = "\n".join(
        f"| {r['planted_gamma']:+.2f} | {r['ratio']:.3f} | {r['naive_rejects']:.0%} | "
        f"{r['boot_rejects']:.0%} | {r['recovered_gamma']:+.3f} |" for r in h["control"])
    return f"""# Results — Study 993 (Down Hurts More) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets; the detailed
profile is **{h['asset']}** over {h['n_days']:,} sessions. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## 1. The effect

Volatility over the {h['horizon']} days following:

| | |
|---|--:|
| An up day | {h['vol_after_up']:.1%} |
| A down day | **{h['vol_after_down']:.1%}** |
| Ratio | **{h['ratio']:.3f}×** |
| Naive *t* | {h['naive_t']:+.1f} |

Forward volatility, not contemporaneous — a same-window comparison would find asymmetry even in
i.i.d. data, because a large down day is itself part of the window that contains it.

## 2. Matched on the size of the move

Down days are larger than up days on average (returns are negatively skewed), and big moves are
followed by volatility whatever their sign. Bucketing by |return| and comparing within buckets
removes that channel:

| Bucket | Mean \\|move\\| | n up | n down | Vol after up | Vol after down | Ratio |
|---|--:|--:|--:|--:|--:|--:|
{matched}

Average within-bucket ratio: **{h['matched_ratio']:.3f}×** against the unmatched
{h['ratio']:.3f}×.

## 3. Honest inference

| | |
|---|--:|
| Difference | {h['difference']:+.2%} |
| Block-bootstrap SD | {h['boot_sd']:.2%} |
| **Bootstrap *t*** | **{h['boot_t']:+.2f}** |
| Naive *t* | {h['naive_t']:+.2f} |
| 95% interval | [{h['boot_lo']:+.2%}, {h['boot_hi']:+.2%}] |

A note against the usual reflex. The block bootstrap here is **not** dramatically wider than the
naive two-sample formula — unlike in study **989**, where it cut the *t* by more than half. Two
effects offset. The up-day and down-day subsamples are interleaved in time and share the
prevailing volatility regime, so their difference cancels most of the common variation, which
the block bootstrap sees and a two-sample formula does not. But the naive formula also charges
each group its *total* variance, including that same common component, which makes it
conservative. The two roughly cancel. "Bootstrap it" is a good habit precisely because you
cannot tell in advance which way it will move.

## 4. The shape, not just the sign

Fitting a quadratic to the news-impact curve (Engle & Ng 1993): vertex at **z =
{h['vertex_z']:+.3f}**, curvature {h['curvature']:.4f}, R² {h['nic_r2']:.2f}. A symmetric
response puts the vertex at zero; the leverage story shifts it toward positive returns.

## 5. The parametric version

EGARCH(1,1): β = {h['egarch_beta']:.4f}, α = {h['egarch_alpha']:.4f}, **γ =
{h['egarch_gamma']:+.4f}**. A −1σ shock raises volatility **{h['egarch_ratio']:.3f}×** as much
as a +1σ shock. The correlation between returns and volatility *changes* — the quantity that
shows up as skew in an option surface — is **{h['corr_r_dvol']:+.3f}**.

## 6. The control that decides the mechanism

| Asset | n | Vol after up | Vol after down | Ratio | Naive *t* | corr(r, Δvol) | EGARCH γ | Curve vertex |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{panel}

**Gold has no balance sheet. Bitcoin has no balance sheet.** If financial leverage were the
mechanism, their ratios would be 1.00. Gold's is {h['gold_ratio']:.3f}×, Bitcoin's is
{h['crypto_ratio']:.3f}×, against equities' {h['ratio']:.3f}×. This single row of the table is
worth more than any amount of equity-only econometrics, and it is why the profession has spent
forty years failing to defend the name.

## 7. Which comes first?

The same contemporaneous correlation is consistent with both stories. Only the lags separate
them: **leverage** needs the return to move first, **volatility feedback** (Campbell & Hentschel
1992) needs the volatility change to.

| Lag | Correlation | |
|---|--:|---|
{ll}

| | |
|---|--:|
| Return leads volatility (leverage side) | {h['leverage_side']:+.4f} |
| Volatility leads return (feedback side) | {h['feedback_side']:+.4f} |
| Contemporaneous | {h['contemporaneous']:+.4f} |
| **Leans** | **{h['leans']}** |

## 8. Synthetic control

| Planted γ | Measured ratio | Naive rejects | Bootstrap rejects | EGARCH recovers γ |
|---|--:|--:|--:|--:|
{ctrl}

Read the first row: with **no asymmetry planted at all**, the naive test declares one in
{h['null_naive_reject']:.0%} of runs and the bootstrap in {h['null_boot_reject']:.0%}. Read the
rest: the apparatus recovers the planted gamma in the right order and the right sign, so it has
power as well as size.

## Caveats

- **The contemporaneous correlation is not identified.** Sections 5 and 7 measure association;
  no amount of lag structure in daily data settles causality between two variables that both
  respond to news. The lead-lag table narrows the field, it does not close it.
- **Realised volatility is a noisy proxy.** The lead-lag correlations are attenuated by that
  noise, and attenuated *unequally* across lags, which is a known weakness of this design
  (Bollerslev, Litvinova & Tauchen 2006 use high-frequency data precisely to avoid it).
- **Bitcoin's sample is short and its regimes are extreme.** Its row in section 6 should carry a
  wider error bar than it is given.
- **EGARCH is one parameterisation.** GJR-GARCH and the threshold models give qualitatively the
  same answer with different numbers; gamma is not comparable across specifications.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[993-leverage-effect-asymmetry](../README.md). Not investment advice.*
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
