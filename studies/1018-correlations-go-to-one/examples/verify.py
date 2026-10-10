"""Real-tape verification — Study 1018 (Correlations Go to One). Regenerates docs/results.md.

Measures the naive "crisis vs calm" average pairwise correlation of 20 large US stocks under six
regime definitions; prices the statistical artefact exactly by running the identical procedure on
paths from a constant-correlation GARCH fitted to the same tape; applies the Forbes–Rigobon
adjustment and a constant-beta benchmark; compares exceedance correlations with their Gaussian
and constant-correlation benchmarks; and asks what the genuine part does to an equal-weight
holder. Every interval is a circular block bootstrap (63-day blocks). A synthetic world with a
planted crisis-correlation jump — and its constant-correlation null — proves the machinery.

    python studies/1018-correlations-go-to-one/examples/verify.py

Offline: reads the pinned skfolio tapes from ``studies/_cache/bundled`` (quantlab.bundled).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from goestoone import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(HERE, "..", "docs"))

N_BOOT = 400
N_SIMS = 40
BLOCK = 63
SEED = 1018


def report() -> dict:
    t0 = time.time()
    R, m = data.load_returns()
    px = data.load_prices()
    ix = data.load_index()
    h: dict = {"as_of": data.AS_OF, "fp_stocks": data.fingerprint(px),
               "fp_index": data.fingerprint(ix), "n_days": int(len(R)),
               "first": str(R.index[0].date()), "last": str(R.index[-1].date()),
               "n_stocks": int(R.shape[1]), "n_boot": N_BOOT, "n_sims": N_SIMS,
               "block": BLOCK}
    h["n_years"] = (R.index[-1] - R.index[0]).days / 365.25
    print(f"as-of {data.AS_OF}  stocks fp {h['fp_stocks']}  index fp {h['fp_index']}")
    print(f"  {h['n_stocks']} stocks, {h['n_days']} days, {h['first']} -> {h['last']}")

    # ------------------------------------------------------------------ 1. naive
    print("\n=== 1. the naive measurement ===")
    labels = st.all_labels(m)
    naive = {}
    for k, lab in labels.items():
        rc = st.regime_corr(R, lab)
        naive[k] = rc
        print(f"  {k:10s} calm {rc['calm']:.3f} ({rc['n_calm']:5d} d)  crisis {rc['crisis']:.3f}"
              f" ({rc['n_crisis']:4d} d)  gap {rc['diff']:+.3f}  "
              f"{'past-only' if st.REGIMES[k]['past_only'] else 'CONTEMPORANEOUS'}")
    h["naive"] = naive
    hk = st.HEADLINE
    h["naive_calm"] = naive[hk]["calm"]
    h["naive_crisis"] = naive[hk]["crisis"]
    roll = st.rolling_avg_corr(R, 63, 5)
    h["max_rolling_corr"] = float(roll.max())
    h["max_rolling_date"] = str(roll.idxmax().date())
    h["min_rolling_corr"] = float(roll.min())
    h["median_rolling_corr"] = float(roll.median())
    full_C = st.corr_matrix(R.to_numpy())
    h["full_corr"] = st.avg_offdiag(full_C)
    print(f"  full-sample average pairwise correlation {h['full_corr']:.3f}")
    print(f"  63-day rolling: median {h['median_rolling_corr']:.3f}, max "
          f"{h['max_rolling_corr']:.3f} on {h['max_rolling_date']}")

    # ------------------------------------------------------------------ 2. artefact
    print("\n=== 2a. the artefact in closed form (Boyer-Gibson-Loretan) ===")
    bgl = []
    for k in (1.0, 2.0, 4.0, 10.0, 20.0):
        bgl.append({"var_ratio": k, "rho_0.20": st.bgl_conditional_corr(0.20, k),
                    "rho_0.40": st.bgl_conditional_corr(0.40, k)})
        print(f"  variance x{k:>4.0f}: true 0.20 reads {bgl[-1]['rho_0.20']:.3f}; "
              f"true 0.40 reads {bgl[-1]['rho_0.40']:.3f}")
    h["bgl"] = bgl

    print("\n=== 2b. exact simulation benchmark: constant-correlation GARCH ===")
    fit = st.fit_ccc_garch(R, m)
    h["ccc_avg_corr"] = fit["avg_corr_stocks"]
    h["ccc_persistence"] = float(np.median([p["alpha"] + p["beta"] for p in fit["params"]]))
    print(f"  fitted {len(fit['params'])} GARCH(1,1); median alpha+beta "
          f"{h['ccc_persistence']:.3f}; constant shock correlation (stocks) "
          f"{h['ccc_avg_corr']:.3f}")
    art = st.artefact_benchmark(fit, n_sims=N_SIMS, seed=SEED, exceed=True)
    print(f"  {N_SIMS} simulated paths ({time.time() - t0:.0f}s)")
    Z = st.devolatilise(R, m, fit)
    bt = st.bootstrap(R, m, labels, n_boot=N_BOOT, block=BLOCK, seed=SEED, exceed=True, Z=Z)
    print(f"  {N_BOOT} block-bootstrap replicates ({time.time() - t0:.0f}s)")

    dec = []
    for k in labels:
        a = art["regimes"][k]
        lo, hi = st.ci(bt[k]["diff"], centre_shift=a["diff_mean"])
        p = st.boot_p_positive(bt[k]["diff"], centre_shift=a["diff_mean"])
        g = naive[k]["diff"] - a["diff_mean"]
        dec.append({"regime": k, "past_only": st.REGIMES[k]["past_only"],
                    "naive_gap": naive[k]["diff"], "artefact": a["diff_mean"],
                    "artefact_sd": a["diff_sd"], "sim_calm": a["calm_mean"],
                    "sim_crisis": a["crisis_mean"], "genuine": g, "lo": lo, "hi": hi,
                    "p": p})
        print(f"  {k:10s} naive {naive[k]['diff']:+.3f}  artefact {a['diff_mean']:+.3f}"
              f" (sd {a['diff_sd']:.3f})  genuine {g:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  p {p:.3f}")
    h["decomp"] = dec
    D = {d["regime"]: d for d in dec}
    h["artefact"] = D[hk]["artefact"]
    h["genuine"] = D[hk]["genuine"]
    h["genuine_lo"], h["genuine_hi"], h["genuine_p"] = D[hk]["lo"], D[hk]["hi"], D[hk]["p"]
    h["genuine_t"] = float(h["genuine"] / np.nanstd(bt[hk]["diff"], ddof=1))
    h["n_regimes"] = len(dec)
    h["n_regimes_genuine_pos"] = int(sum(d["genuine"] > 0 for d in dec))
    h["n_regimes_genuine_sig"] = int(sum(d["lo"] > 0 for d in dec))
    h["share_regimes_genuine_pos"] = h["n_regimes_genuine_pos"] / h["n_regimes"]
    h["contemp_naive"] = D["month_vol"]["naive_gap"]
    h["contemp_artefact"] = D["month_vol"]["artefact"]
    h["bigdown_naive"] = D["big_down"]["naive_gap"]
    h["bigdown_artefact"] = D["big_down"]["artefact"]

    print("\n=== 2c. Forbes-Rigobon and the constant-beta benchmark ===")
    frs = []
    for k, lab in labels.items():
        fr = st.fr_panel(R, m, lab)
        cb = st.constant_beta_benchmark(R, m, lab)
        iv = st.idio_vol_ratio(R, m, lab)
        frs.append({"regime": k, "calm": fr["calm_raw"], "crisis": fr["crisis_raw"],
                    "crisis_fr": fr["crisis_fr"], "var_ratio": fr["var_ratio"],
                    "const_beta_crisis": cb["implied_crisis"],
                    "mkt_vol_ratio": iv["mkt_vol_ratio"], "idio_vol_ratio": iv["idio_vol_ratio"],
                    "beta_ratio": iv["beta_ratio"], "mkt_corr_calm": fr["mkt_corr_calm"],
                    "mkt_corr_crisis": fr["mkt_corr_crisis"],
                    "mkt_corr_crisis_fr": fr["mkt_corr_crisis_fr"]})
        print(f"  {k:10s} calm {fr['calm_raw']:.3f} crisis {fr['crisis_raw']:.3f} FR-adj "
              f"{fr['crisis_fr']:.3f} | const-beta implies {cb['implied_crisis']:.3f} | "
              f"mkt vol x{iv['mkt_vol_ratio']:.2f}, idio vol x{iv['idio_vol_ratio']:.2f}, "
              f"beta x{iv['beta_ratio']:.2f}")
    h["fr"] = frs
    F = {f["regime"]: f for f in frs}
    h["fr_crisis"] = F[hk]["crisis_fr"]
    h["var_ratio"] = F[hk]["var_ratio"]
    h["idio_ratio"] = F[hk]["idio_vol_ratio"]
    h["mkt_ratio"] = F[hk]["mkt_vol_ratio"]
    h["const_beta_crisis"] = F[hk]["const_beta_crisis"]

    # ------------------------------------------------------------------ 3. exceedance
    print("\n=== 3. exceedance correlations (stock vs S&P, averaged over 20 stocks) ===")
    cur = st.exceedance_panel(R, m)
    gau = st.gaussian_panel_curve(R, m)
    ccc = art["curve"]
    curz = st.exceedance_panel(*Z)
    exc = pd.DataFrame({"empirical": cur, "gaussian": gau, "const_corr_garch": ccc,
                        "devolatilised": curz})
    print(exc.round(3).to_string())
    h["exceed"] = exc.reset_index().rename(columns={"index": "theta"}).to_dict("records")
    h["exc_down_1"], h["exc_up_1"] = float(cur[-1.0]), float(cur[1.0])
    h["exc_gauss_1"] = float((gau[-1.0] + gau[1.0]) / 2)
    h["asym"] = st.asymmetry(cur)
    h["asym_lo"], h["asym_hi"] = st.ci(bt["_asym"])
    h["asym_devol"] = st.asymmetry(curz)
    h["asym_devol_lo"], h["asym_devol_hi"] = st.ci(bt["_asym_devol"])
    h["asym_null"] = art["asym_mean"]
    h["asym_null_sd"] = art["asym_sd"]
    h["h_down"] = st.ang_chen_h(cur, gau, "down")
    h["h_up"] = st.ang_chen_h(cur, gau, "up")
    print(f"  asymmetry (down - up, theta 0.5/1/1.5): raw {h['asym']:+.3f} "
          f"[{h['asym_lo']:+.3f}, {h['asym_hi']:+.3f}]; devolatilised {h['asym_devol']:+.3f} "
          f"[{h['asym_devol_lo']:+.3f}, {h['asym_devol_hi']:+.3f}]; const-corr GARCH "
          f"{h['asym_null']:+.3f} (sd {h['asym_null_sd']:.3f})")
    print(f"  Ang-Chen H vs Gaussian: downside {h['h_down']:.3f}, upside {h['h_up']:.3f}")

    # ------------------------------------------------------------------ 4. holder
    print("\n=== 4. the diversified holder (equal-weight, daily rebalanced) ===")
    port = []
    for k, lab in labels.items():
        ps = st.portfolio_stats(R, lab)
        a = art["regimes"][k]
        g = ps["shortfall"] - a["shortfall_mean"]
        lo, hi = st.ci(bt[k]["shortfall"], centre_shift=a["shortfall_mean"])
        dlo, dhi = st.ci(bt[k]["dr_gap"])
        port.append({"regime": k, **ps, "shortfall_artefact": a["shortfall_mean"],
                     "shortfall_genuine": g, "lo": lo, "hi": hi,
                     "dr_gap": ps["dr_calm"] - ps["dr_crisis"], "dr_gap_lo": dlo,
                     "dr_gap_hi": dhi})
        print(f"  {k:10s} vol calm {ps['vol_calm']:.1%} crisis {ps['vol_crisis']:.1%} pred "
              f"{ps['vol_pred']:.1%} | shortfall {ps['shortfall']:+.1%} (artefact "
              f"{a['shortfall_mean']:+.1%}) genuine {g:+.1%} [{lo:+.1%}, {hi:+.1%}] | DR "
              f"{ps['dr_calm']:.2f} -> {ps['dr_crisis']:.2f}")
    h["port"] = port
    P = {p["regime"]: p for p in port}
    for key in ("vol_calm", "vol_crisis", "vol_pred", "dr_calm", "dr_crisis"):
        h[key] = P[hk][key]
    h["shortfall"] = P[hk]["shortfall_genuine"]
    h["shortfall_raw"] = P[hk]["shortfall"]
    h["shortfall_artefact"] = P[hk]["shortfall_artefact"]
    h["shortfall_lo"], h["shortfall_hi"] = P[hk]["lo"], P[hk]["hi"]
    h["dr_gap_lo"], h["dr_gap_hi"] = P[hk]["dr_gap_lo"], P[hk]["dr_gap_hi"]
    h["dr_max"] = float(np.sqrt(R.shape[1]))

    ep = st.episode_table(R)
    print(ep.round(3).to_string())
    h["episodes"] = ep.reset_index().to_dict("records")
    h["n_episodes"] = int(len(ep))
    h["n_episodes_under"] = int((ep["ratio"] > 1.0).sum())
    h["episode_ratio_median"] = float(ep["ratio"].median())
    h["episode_corr_max"] = float(ep["corr_during"].max())
    h["episode_corr_max_name"] = str(ep["corr_during"].idxmax())

    # ------------------------------------------------------------------ 5. synthetic control
    print("\n=== 5. synthetic control (machinery proof, NOT market evidence) ===")
    syn = []
    for s in (0.0, 0.5, 1.0):
        Rs, ms, tr = data.synthetic_panel(n_assets=10, n_years=16, signal_strength=s,
                                          seed=SEED)
        d = st.crisis_correlation_test(Rs, ms, n_sims=12, n_boot=150, seed=SEED)
        mv = st.regime_corr(Rs, st.regime_labels(ms, "month_vol"))
        syn.append({"signal_strength": s, "planted_jump": tr["planted_jump"],
                    "naive": d["diff"], "artefact": d["artefact"], "genuine": d["genuine"],
                    "lo": d["ci_lo"], "hi": d["ci_hi"], "significant": d["significant"],
                    "contemp_naive": mv["diff"]})
        print(f"  strength {s:.1f} (planted +{tr['planted_jump']:.2f}): naive {d['diff']:+.3f}"
              f" artefact {d['artefact']:+.3f} genuine {d['genuine']:+.3f} "
              f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] -> "
              f"{'FIRES' if d['significant'] else 'quiet'} | month-vol naive gap "
              f"{mv['diff']:+.3f}")
    h["synthetic"] = syn

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = time.time() - t0
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  runtime {h['runtime_s']:.0f}s")
    return h


def _f(x, fmt="+.3f"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)


def results_md(h: dict) -> str:
    v = h["_verdict"]
    naive = "\n".join(
        f"| {st.REGIMES[k]['label']} | {'past-only' if st.REGIMES[k]['past_only'] else '**contemporaneous**'} "
        f"| {r['calm']:.3f} | {r['crisis']:.3f} | {r['diff']:+.3f} | {r['n_calm']} | {r['n_crisis']} |"
        for k, r in h["naive"].items())
    bgl = "\n".join(f"| ×{r['var_ratio']:.0f} | {r['rho_0.20']:.3f} | {r['rho_0.40']:.3f} |"
                    for r in h["bgl"])
    dec = "\n".join(
        f"| {d['regime']} | {'yes' if d['past_only'] else 'no'} | {d['naive_gap']:+.3f} | "
        f"{d['artefact']:+.3f} ± {d['artefact_sd']:.3f} | **{d['genuine']:+.3f}** | "
        f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p']:.3f} |" for d in h["decomp"])
    fr = "\n".join(
        f"| {f['regime']} | {f['calm']:.3f} | {f['crisis']:.3f} | {f['crisis_fr']:.3f} | "
        f"{f['const_beta_crisis']:.3f} | ×{f['var_ratio']:.1f} | ×{f['mkt_vol_ratio']:.2f} | "
        f"×{f['idio_vol_ratio']:.2f} | ×{f['beta_ratio']:.2f} | {f['mkt_corr_calm']:.3f} → "
        f"{f['mkt_corr_crisis']:.3f} (FR {f['mkt_corr_crisis_fr']:.3f}) |" for f in h["fr"])
    exc = "\n".join(
        f"| {r['theta']:+.2f} | {_f(r['empirical'], '.3f')} | {_f(r['gaussian'], '.3f')} | "
        f"{_f(r['const_corr_garch'], '.3f')} | {_f(r['devolatilised'], '.3f')} |"
        for r in h["exceed"])
    port = "\n".join(
        f"| {p['regime']} | {p['vol_calm']:.1%} | {p['vol_crisis']:.1%} | {p['vol_pred']:.1%} | "
        f"{p['shortfall']:+.1%} | {p['shortfall_artefact']:+.1%} | **{p['shortfall_genuine']:+.1%}** "
        f"| [{p['lo']:+.1%}, {p['hi']:+.1%}] | {p['dr_calm']:.2f} → {p['dr_crisis']:.2f} |"
        for p in h["port"])
    ep = "\n".join(
        f"| {e['episode']} | {e['days']} | {e['corr_before']:.2f} | {e['corr_during']:.2f} | "
        f"{e['vol_during']:.1%} | {e['vol_pred']:.1%} | **{e['ratio']:.2f}×** | "
        f"{e['dr_before']:.2f} → {e['dr_during']:.2f} |" for e in h["episodes"])
    syn = "\n".join(
        f"| {s['signal_strength']:.1f} | +{s['planted_jump']:.2f} | {s['naive']:+.3f} | "
        f"{s['artefact']:+.3f} | {s['genuine']:+.3f} | [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
        f"{'**fires**' if s['significant'] else 'quiet'} | {s['contemp_naive']:+.3f} |"
        for s in h["synthetic"])
    prov = data.PROVENANCE
    return f"""# Results — Study 1018 (Correlations Go to One) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). As-of
**{h['as_of']}** (last full month; the tape's partial December 2022 is dropped). {h['n_stocks']}
stocks, {h['n_days']} trading days, {h['first']} → {h['last']}. Circular block bootstrap,
{h['n_boot']} replicates of {h['block']}-day blocks; artefact benchmark from {h['n_sims']}
constant-correlation GARCH paths.*

**Data provenance.** {prov['package']}; wheel SHA-256 `{prov['wheel_sha256'][:16]}…`, loaded
through [`quantlab/bundled.py`](../../../quantlab/bundled.py).
`{prov['files']['sp500_dataset']['file']}` (SHA-256 `{prov['files']['sp500_dataset']['sha256'][:16]}…`)
— {prov['stocks']}; fingerprint `{h['fp_stocks']}`.
`{prov['files']['sp500_index']['file']}` (SHA-256 `{prov['files']['sp500_index']['sha256'][:16]}…`)
— {prov['index']}; fingerprint `{h['fp_index']}`. Daily simple returns.

## 1. The naive measurement

Average pairwise correlation of the 20 stocks, crisis days versus calm days. *Past-only* means
the label for day t uses the S&P 500 only through t-1. The two contemporaneous definitions are the
folklore's own way of measuring and are included on purpose.

| Regime | Label timing | Calm | Crisis | Gap | Calm days | Crisis days |
|---|---|--:|--:|--:|--:|--:|
{naive}

Full-sample average pairwise correlation: {h['full_corr']:.3f}. Rolling 63-day average: median
{h['median_rolling_corr']:.3f}, minimum {h['min_rolling_corr']:.3f}, **maximum
{h['max_rolling_corr']:.3f}** (window ending {h['max_rolling_date']}). Correlations rise a lot;
they do not go to one.

## 2. The artefact

### 2a. In closed form

Select days on which one variable's variance is k times its normal level, and a correlation whose
true value never changed reads higher (Boyer, Gibson & Loretan 1999):

| Variance ratio of the selected days | True ρ = 0.20 reads | True ρ = 0.40 reads |
|---|--:|--:|
{bgl}

### 2b. Exact simulation benchmark — constant-correlation GARCH

Each stock and the S&P get their own GARCH(1,1) fitted to this tape (median α+β
{h['ccc_persistence']:.3f}); their shocks share one **constant** correlation matrix (stocks'
average {h['ccc_avg_corr']:.3f}). The identical regime procedure is run on {h['n_sims']} simulated
paths: whatever gap it reports is pure artefact. Genuine = naive − artefact; the interval is the
block-bootstrap distribution of the naive gap shifted by the artefact; p is one-sided (genuine > 0).

| Regime | Past-only | Naive gap | Artefact (mean ± sd) | Genuine | 95% CI | p |
|---|:--:|--:|--:|--:|--:|--:|
{dec}

Past-only regimes are essentially immune to the artefact — the conditional correlation of a
constant-correlation GARCH does not depend on yesterday's volatility, and the procedure correctly
reports ~0. The contemporaneous definitions are not: conditioning on the month's own volatility
manufactures **{h['contemp_artefact']:+.3f}** of correlation out of nothing. Headline (past 21-day
vol): genuine **{h['genuine']:+.3f}**, CI [{h['genuine_lo']:+.3f}, {h['genuine_hi']:+.3f}],
bootstrap t ≈ {h['genuine_t']:.1f}; positive in {h['n_regimes_genuine_pos']} of {h['n_regimes']}
definitions, CI above zero in {h['n_regimes_genuine_sig']}.

### 2c. Forbes–Rigobon, and what the adjustment does and does not mean

FR-adjusted crisis correlation uses δ = Var_crisis(S&P)/Var_calm(S&P) − 1 on every pair. The
constant-beta benchmark holds calm betas and calm idiosyncratic variances fixed and moves only the
market's variance.

| Regime | Calm | Crisis | Crisis, FR-adjusted | Constant-beta implies | Mkt variance | Mkt vol | Idio vol | Beta | Stock–S&P corr calm → crisis |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
{fr}

Read this table carefully, because it is where the folklore and its debunking both go wrong.
The FR adjustment pulls the crisis correlation back to (or below) the calm level: on Forbes and
Rigobon's definition there is **no contagion** — the transmission from market to stock (beta) is
unchanged, and the rise is the arithmetic of a common factor whose variance went up ×{h['var_ratio']:.1f}
while stock-specific volatility rose only ×{h['idio_ratio']:.2f}. The constant-beta benchmark
over-predicts the crisis correlation ({h['const_beta_crisis']:.3f} vs {h['naive_crisis']:.3f})
precisely because idiosyncratic risk *also* rose. None of this makes the rise "fake". A louder
common factor is exactly what a diversified holder cannot diversify, so the FR-adjusted number
answers "did the mechanism change?" — not "did my diversification work?".

## 3. Exceedance correlations

Correlation of each stock with the S&P on days when *both* are beyond θ standard deviations
(below for θ < 0, above for θ > 0), averaged across the 20 stocks (Longin & Solnik 2001; Ang &
Chen 2002). The Gaussian column is a bivariate normal with each stock's own full-sample
correlation; the constant-correlation GARCH column is the mean over the simulated paths; the
devolatilised column divides every series by its own GARCH volatility first.

| θ | Empirical | Gaussian | Const-corr GARCH | Devolatilised |
|--:|--:|--:|--:|--:|
{exc}

Both tails are far more correlated than a normal distribution with the same correlation would
make them (Ang–Chen H: downside {h['h_down']:.3f}, upside {h['h_up']:.3f}) — the Gaussian curve
*falls* toward zero in the tails and the empirical one does not. The downside-minus-upside
asymmetry (mean over θ = 0.5, 1, 1.5) is **{h['asym']:+.3f}** on raw returns, CI
[{h['asym_lo']:+.3f}, {h['asym_hi']:+.3f}], and **{h['asym_devol']:+.3f}** on devolatilised
shocks, CI [{h['asym_devol_lo']:+.3f}, {h['asym_devol_hi']:+.3f}]; the constant-correlation GARCH
gives {h['asym_null']:+.3f} (sd {h['asym_null_sd']:.3f}). Raw daily tails are close to
symmetric because the biggest up-days (rebounds) arrive in the same high-volatility clusters as
the biggest down-days; measured relative to the prevailing volatility, the down-shocks are the
more correlated ones.

## 4. The diversified holder

An equal-weight, daily-rebalanced book of the 20 stocks. *Predicted* = calm correlation matrix
with each stock's crisis volatility plugged in ("volatility rose, correlations didn't").
*Shortfall* = realised / predicted − 1: the part of crisis risk owed to correlation. The genuine
shortfall subtracts what the same procedure shows under constant correlation; the interval is
the block bootstrap. DR is the diversification ratio Σwσ / σ_p (1 = no diversification; the
ceiling for 20 equal, uncorrelated assets is {h['dr_max']:.2f}).

| Regime | Book vol calm | Book vol crisis | Predicted | Shortfall | Artefact | Genuine | 95% CI | DR calm → crisis |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{port}

**Named episodes** (chosen with hindsight — descriptive, no inference). The reference is the
252 days ending a month before each episode: what a holder could actually have estimated.

| Episode | Days | Corr before | Corr during | Book vol | Predicted | Ratio | DR before → during |
|---|--:|--:|--:|--:|--:|--:|--:|
{ep}

In {h['n_episodes_under']} of {h['n_episodes']} episodes the book was riskier than
"pre-crisis correlations, crisis volatilities" predicted (median {h['episode_ratio_median']:.2f}×).
The highest episode correlation was {h['episode_corr_max']:.2f} ({h['episode_corr_max_name']}).
The diversification ratio shrank every time and never reached 1.

## 5. Synthetic control — the machinery, not the market

Ten assets sharing one stochastic volatility. At strength 0 the true correlation is a constant
0.30 on every day; at strength 1 it jumps by +0.30 in the top-decile latent volatility states.
Same detector as the headline (past 21-day regime, constant-correlation GARCH artefact, block
bootstrap).

| Strength | Planted jump | Naive gap | Artefact | Genuine | 95% CI | Detector | Month-vol naive gap |
|--:|--:|--:|--:|--:|--:|:--:|--:|
{syn}

The detector fires on the planted world and stays quiet on the null — while the contemporaneous
month-volatility measurement shows a positive gap even in the null, which is the artefact this
study is about. (The planted jump is diluted in the measured gap because a past-21-day window only
partly overlaps the latent crisis state.)

## Caveats

- **Survivor sample.** The 20 stocks were large and alive in 2022. For correlation this is a
  small bias with a reasoned direction: firms that failed in a crisis typically decoupled into
  idiosyncratic collapse, so omitting them most plausibly *raises* measured crisis correlation a
  little; firms that survived to be mega-caps may also be more market-like than the average stock.
  The study makes no claim about returns. It is also a 20-stock large-cap panel, not "all
  correlations": cross-asset correlations (stocks vs bonds, gold) behave differently and are
  another study.
- **Price index as the state variable.** The S&P 500 series has no dividends; it is used only for
  volatility, drawdown and daily-move labels, where that is immaterial.
- **Early prices are rounded to three decimals** (e.g. AAPL ≈ $0.26 in 1990), which adds a little
  quantisation noise to early daily returns and slightly *depresses* early correlations.
- **Full-sample thresholds.** Regime cut-offs (deciles of past volatility) and the GARCH parameters
  are estimated on the whole sample; the *label* of each day is past-only, the threshold is not.
  This is a measurement study, not a timing rule, so that is the right trade-off — but it is a
  choice.
- **The artefact benchmark is Gaussian constant-correlation GARCH.** Fatter-tailed shocks or a
  richer volatility model would change the artefact for the contemporaneous definitions; for the
  past-only ones it is ≈ 0 by construction of any model whose conditional correlation is
  constant.
- **The bootstrap carries labels with days** and resamples 63-day blocks; crisis days come from a
  handful of episodes (1998, 2000-02, 2008-09, 2011, 2020, 2022), so the effective sample is far
  smaller than the day count. The interval reflects that; the episode table shows the effect in
  each episode separately.
- **Daily, close-to-close, US only.** Non-synchronous closes are not an issue within one market;
  they would be across countries.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1018-correlations-go-to-one](../README.md). Not investment advice.*
"""


HEADLINE_KEYS = ("as_of", "fp_stocks", "fp_index", "naive_calm", "naive_crisis", "artefact",
                 "genuine", "genuine_lo", "genuine_hi", "genuine_p", "genuine_t",
                 "n_regimes_genuine_pos", "n_regimes", "contemp_naive", "contemp_artefact",
                 "fr_crisis", "var_ratio", "idio_ratio", "const_beta_crisis",
                 "max_rolling_corr", "asym", "asym_lo", "asym_hi", "asym_devol",
                 "asym_devol_lo", "asym_devol_hi", "vol_crisis", "vol_pred", "shortfall",
                 "shortfall_lo", "shortfall_hi", "dr_calm", "dr_crisis",
                 "episode_ratio_median", "runtime_s")


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    head = {k: h[k] for k in HEADLINE_KEYS}
    head["signal"] = h["_verdict"]["signal"]
    head["trad"] = h["_verdict"]["trad"]
    print("##HEADLINE## " + json.dumps(head, default=float))


if __name__ == "__main__":
    main()
