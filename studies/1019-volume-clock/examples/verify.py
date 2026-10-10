"""Real-tape verification — Study 1019 (The Market Runs on Volume Time). Regenerates docs/results.md.

On the S&P 500 and Nasdaq Composite daily tapes frozen inside ``arch`` (1999-2018, price
indices, composite volume), this script

1. standardises daily returns by four clocks — Clark's volume clock (past-only detrended),
   the day's own range (Parkinson, an upper benchmark), a past-only GARCH(1,1) (the usual
   benchmark) and GARCH times volume surprise — and measures how much of the raw excess
   kurtosis each removes, with paired circular-block-bootstrap intervals;
2. measures the volume–volatility correlation on the same day and at lags 1 and 5;
3. races a range-based log-HAR against the same model plus *lagged* volume and against GARCH,
   out of sample, scored by QLIKE and MSE with HAC Diebold–Mariano tests;
4. builds a 15%-target vol overlay on each forecast with one execution lag and compares
   Sharpe ratios net of costs by block bootstrap;
5. reruns the measurement on the synthetic subordinated process with the clock planted
   (``signal_strength=1``) and switched off (``0``) — a machinery check, never evidence.

    python studies/1019-volume-clock/examples/verify.py
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

from volclock import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
BASE_COST_BPS = 2.0
N_BOOT = 1000


def run_tape(name: str) -> dict:
    tape = data.load_tape(name)
    out: dict = {"name": name, "label": data.TAPE_LABELS[name],
                 "fingerprint": data.fingerprint(tape), "sha": data.sha_pin(name),
                 "rows": int(len(tape)), "first": str(tape.index[0].date()),
                 "last": str(tape.index[-1].date()),
                 "zero_volume_days": int(tape["volume"].isna().sum())}
    print(f"\n######## {data.TAPE_LABELS[name]} — {out['first']} → {out['last']}, "
          f"fingerprint {out['fingerprint']}")

    print("=== 1. the clocks ===")
    P = st.clock_panel(tape)
    K = st.kurtosis_removed(P, n_boot=N_BOOT)
    print(K.round(3).to_string())
    out["panel_start"] = str(P.index[0].date())
    out["panel_n"] = int(len(P))
    out["kurt"] = K.reset_index().to_dict("records")
    T = st.tail_table(P)
    print(T.round(2).to_string())
    out["tails"] = T.reset_index().to_dict("records")
    inc = st.incremental_removal(P, n_boot=N_BOOT)
    out["incremental"] = inc
    print(f"  GARCH x volume surprise vs GARCH: kurtosis {inc['kurt_reduction']:+.2f} lower "
          f"(CI {inc['ci_lo']:+.2f} to {inc['ci_hi']:+.2f})")

    print("=== 1b. robustness ===")
    W = st.window_sensitivity(tape)
    print(W.round(3).to_string())
    out["windows"] = W.reset_index().to_dict("records")
    E = st.exponent_curve(tape)
    out["exponents"] = {str(k): float(v) for k, v in E.items()}
    out["best_exponent"] = float(E.idxmin())
    out["best_exponent_k"] = float(E.min())
    print(f"  in-sample best exponent b = {out['best_exponent']:.1f} "
          f"(k = {out['best_exponent_k']:.2f}; b=0 {E.iloc[0]:.2f}, b=1 {E.loc[1.0]:.2f})")
    B = st.by_period(P)
    print(B.round(2).to_string())
    out["periods"] = B.reset_index().to_dict("records")

    print("=== 2. volume-volatility correlation ===")
    C = st.volume_vol_correlation(tape)
    print(C.round(3).to_string())
    out["corr"] = C.to_dict("records")

    print("=== 3. forecast race ===")
    fc = st.forecast_race(tape)
    S = st.score_race(fc)
    print(S.round(4).to_string())
    out["race"] = S.to_dict("records")
    out["race_start"] = str(fc.dropna().index[0].date())
    out["race_n"] = int(len(fc.dropna()))
    ins = st.har_in_sample(tape)
    out["in_sample"] = ins
    print(f"  in-sample log-HAR-X: volume coefficient {ins['beta_lv']:+.3f} "
          f"(NW t {ins['t_lv']:.2f}); R² {ins['r2_har']:.4f} -> {ins['r2_harx']:.4f}")

    print("=== 4. overlay ===")
    rf = data.load_daily_rf(tape.index)
    O = st.overlay_race(tape, fc, rf, n_boot=N_BOOT)
    print(O["table"].round(3).to_string())
    print(O["tests"].round(3).to_string())
    out["overlay"] = O["table"].to_dict("records")
    out["overlay_tests"] = O["tests"].to_dict("records")
    out["overlay_span"] = (O["start"], O["end"], O["n"])
    la = st.lookahead_clock_sharpe(tape, fc, rf)
    out["lookahead"] = la
    print(f"  look-ahead (impossible): zero-lag HAR {la['sharpe_zero_lag']:.2f} -> "
          f"x same-day volume {la['sharpe_gross']:.2f} (gross Sharpe)")

    # Fields read by strategy.verdict
    k = K["excess_kurtosis"]
    c0 = C[(C["lag"] == 0) & (C["target"] == "|r|")].iloc[0]
    c1 = C[(C["lag"] == 1) & (C["target"] == "|r|")].iloc[0]

    def race(tgt, loss, vs):
        return S[(S["target"] == tgt) & (S["loss"] == loss) & (S["vs"] == vs)].iloc[0]

    tst = O["tests"]
    base = tst[(tst["cost_bps"] == BASE_COST_BPS) & (tst["vs"] == "har")].iloc[0]
    tab = O["table"]
    har_sr = tab[(tab["cost_bps"] == 0.0) & (tab["model"] == "har")]["sharpe"].iloc[0]
    out["v"] = {
        "k_raw": float(k["raw"]), "k_volume": float(k["volume"]),
        "share_removed": float(K.loc["volume", "share_removed"]),
        "share_ci_lo": float(K.loc["volume", "share_ci_lo"]),
        "share_ci_hi": float(K.loc["volume", "share_ci_hi"]),
        "share_removed_range": float(K.loc["range", "share_removed"]),
        "share_removed_garch": float(K.loc["garch", "share_removed"]),
        "share_removed_garch_volume": float(K.loc["garch_volume", "share_removed"]),
        "jb_p_volume": float(K.loc["volume", "jb_p"]),
        "corr0": float(c0["corr"]), "corr0_t": float(c0["hac_t"]),
        "corr1": float(c1["corr"]), "corr1_t": float(c1["hac_t"]),
        "dm_t": float(race("r2", "qlike", "har")["dm_t"]),
        "dm_p": float(race("r2", "qlike", "har")["p"]),
        "dm_t_pk": float(race("pk", "qlike", "har")["dm_t"]),
        "dm_t_vs_garch": float(race("r2", "qlike", "garch")["dm_t"]),
        "sharpe_diff_net": float(base["diff"]), "sharpe_diff_p": float(base["p"]),
        "lookahead_sharpe": float(la["sharpe_gross"]),
        "lookahead_base_sharpe": float(la["sharpe_zero_lag"]), "har_sharpe": float(har_sr),
        "inc_ci_lo": float(inc["ci_lo"]),
    }
    return out


def synthetic_control() -> dict:
    rows = []
    for s in (1.0, 0.5, 0.0):
        tape, truth = data.synthetic_tape(n_days=5000, signal_strength=s, seed=1019)
        P = st.clock_panel(tape)
        K = st.kurtosis_removed(P, n_boot=300)
        C = st.volume_vol_correlation(tape, lags=(0, 1))
        S = st.score_race(st.forecast_race(tape))
        c0 = C[(C["lag"] == 0) & (C["target"] == "|r|")].iloc[0]
        dm = S[(S["target"] == "pk") & (S["loss"] == "qlike") & (S["vs"] == "har")].iloc[0]
        rows.append({"signal_strength": s, "k_raw": float(K.loc["raw", "excess_kurtosis"]),
                     "k_volume": float(K.loc["volume", "excess_kurtosis"]),
                     "share_removed": float(K.loc["volume", "share_removed"]),
                     "share_ci_lo": float(K.loc["volume", "share_ci_lo"]),
                     "corr0": float(c0["corr"]), "corr0_t": float(c0["hac_t"]),
                     "dm_t_pk": float(dm["dm_t"])})
        print(f"  synthetic s={s:.1f}: k {rows[-1]['k_raw']:.2f} -> {rows[-1]['k_volume']:.2f}"
              f" ({rows[-1]['share_removed']:+.0%}), corr {rows[-1]['corr0']:+.2f}, "
              f"DM(pk) {rows[-1]['dm_t_pk']:+.2f}")
    return {"rows": rows}


def report() -> dict:
    t0 = time.time()
    h: dict = {"as_of": data.AS_OF, "source": data.SOURCE, "arch_version": data.arch_version(),
               "base_cost_bps": BASE_COST_BPS,
               "labels": {"sp500": "S&P 500", "nasdaq": "Nasdaq"}}
    print(f"as-of {data.AS_OF}   source {data.SOURCE}   arch {h['arch_version']}")
    h["per_tape"] = {name: run_tape(name) for name in data.TAPES}
    h["tapes"] = {name: h["per_tape"][name]["v"] for name in data.TAPES}
    print("\n=== 5. synthetic control (machinery check, NOT market evidence) ===")
    h["synthetic"] = synthetic_control()
    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    return h


# --------------------------------------------------------------------------- #
# results.md
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{x:+.0%}" if np.isfinite(x) else "—"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    names = list(data.TAPES)
    P = h["per_tape"]

    prov = "\n".join(
        f"| {P[n]['label']} | `arch` {h['arch_version']} → `quantlab.bundled.load_arch(\"{n}\")` "
        f"| {P[n]['first']} → {P[n]['last']} | {P[n]['rows']:,} | {P[n]['zero_volume_days']} "
        f"| `{P[n]['sha'][:16]}…` | `{P[n]['fingerprint']}` |" for n in names)

    def kurt_rows(n):
        return "\n".join(
            f"| {st.CLOCK_LABELS[r['clock']]} | {r['excess_kurtosis']:.2f} "
            f"[{r['kurt_ci_lo']:.2f}, {r['kurt_ci_hi']:.2f}] | {r['skew']:+.2f} | "
            + ("—" if r["clock"] == "raw" else
               f"**{r['share_removed']:.0%}** [{r['share_ci_lo']:.0%}, {r['share_ci_hi']:.0%}]")
            + f" | {r['jb']:,.0f} | {r['jb_p']:.1e} |" for r in P[n]["kurt"])

    def tail_rows(n):
        return "\n".join(
            f"| {st.CLOCK_LABELS[r['clock']]} | {r['q0.1%']:+.2f} | {r['q1%']:+.2f} | "
            f"{r['q99%']:+.2f} | {r['q99.9%']:+.2f} | {r['beyond3_vs_normal']:.1f}× | "
            f"{r['beyond4_vs_normal']:.0f}× |" for r in P[n]["tails"])

    sec1 = "\n\n".join(
        f"### {P[n]['label']} — {P[n]['panel_n']:,} sessions from {P[n]['panel_start']}\n\n"
        "| Clock | Excess kurtosis [95% CI] | Skew | Share of raw excess kurtosis removed "
        "[95% CI] | Jarque–Bera | p |\n|---|--:|--:|--:|--:|--:|\n" + kurt_rows(n)
        for n in names)
    sec2 = "\n\n".join(
        f"### {P[n]['label']}\n\n| Clock | 0.1% | 1% | 99% | 99.9% | P(\\|z\\|>3) vs normal "
        f"| P(\\|z\\|>4) vs normal |\n|---|--:|--:|--:|--:|--:|--:|\n"
        "| *N(0,1)* | −3.09 | −2.33 | +2.33 | +3.09 | 1.0× | 1× |\n" + tail_rows(n)
        for n in names)

    win = "\n".join(
        f"| {P[n]['label']} | " + " | ".join(
            f"{w['share_removed']:.0%}" for w in P[n]["windows"]) + " |" for n in names)
    win_hdr = " | ".join(f"{int(w['window'])}d" for w in P[names[0]]["windows"])
    expo = "\n".join(
        f"| {P[n]['label']} | {P[n]['exponents']['0.0']:.2f} | {P[n]['exponents']['1.0']:.2f} | "
        f"{P[n]['best_exponent']:.1f} | {P[n]['best_exponent_k']:.2f} | "
        f"{P[n]['exponents']['2.0']:.2f} |" for n in names)
    per = "\n".join(
        f"| {P[n]['label']} | {r['period']} | {r['k_raw']:.2f} | {_pct(r['share_volume'])} | "
        f"{_pct(r['share_range'])} | {_pct(r['share_garch'])} | {_pct(r['share_garch_volume'])} |"
        for n in names for r in P[n]["periods"])
    inc = "\n".join(
        f"| {P[n]['label']} | {P[n]['incremental']['kurt_reduction']:+.2f} | "
        f"[{P[n]['incremental']['ci_lo']:+.2f}, {P[n]['incremental']['ci_hi']:+.2f}] |"
        for n in names)
    corr = "\n".join(
        f"| {P[n]['label']} | {r['lag']} | {r['target'].replace('|', chr(92) + '|')} | {r['corr']:+.3f} | {r['hac_t']:.2f} | "
        f"{r['spearman']:+.3f} |" for n in names for r in P[n]["corr"])

    def race_rows(n):
        return "\n".join(
            f"| {P[n]['label']} | {'r²' if r['target'] == 'r2' else 'range var'} | "
            f"{r['loss'].upper()} | HAR-X vs {r['vs'].upper()} | {r['mean_diff']:+.2e} | "
            f"**{r['dm_t']:+.2f}** | {r['p']:.3f} |" for r in P[n]["race"])
    race = "\n".join(race_rows(n) for n in names)
    ins = "\n".join(
        f"| {P[n]['label']} | {P[n]['in_sample']['beta_lv']:+.3f} | "
        f"{P[n]['in_sample']['t_lv']:.2f} | {P[n]['in_sample']['r2_har']:.4f} | "
        f"{P[n]['in_sample']['r2_harx']:.4f} |" for n in names)

    def ov_rows(n):
        return "\n".join(
            f"| {P[n]['label']} | {r['model'].upper() if r['model'] != 'buy & hold' else 'buy & hold'}"
            f" | {r['cost_bps']:.0f} | {r['sharpe']:.3f} | {r['ann_ret']:.2%} | {r['ann_vol']:.2%} | "
            f"{r['turnover_ann']:.1f}× | {r['mean_pos']:.2f} |" for r in P[n]["overlay"])
    ov = "\n".join(ov_rows(n) for n in names)
    ovt = "\n".join(
        f"| {P[n]['label']} | {r['cost_bps']:.0f} | HAR-X − {r['vs'].upper()} | "
        f"**{r['diff']:+.3f}** | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.2f} |"
        for n in names for r in P[n]["overlay_tests"])
    la = "\n".join(
        f"| {P[n]['label']} | {h['tapes'][n]['har_sharpe']:.2f} | "
        f"{P[n]['lookahead']['sharpe_zero_lag']:.2f} | "
        f"{P[n]['lookahead']['sharpe_gross']:.2f} |" for n in names)
    syn = "\n".join(
        f"| {r['signal_strength']:.1f} | {r['k_raw']:.2f} | {r['k_volume']:.2f} | "
        f"**{r['share_removed']:+.0%}** | {r['corr0']:+.2f} ({r['corr0_t']:.1f}) | "
        f"{r['dm_t_pk']:+.2f} |" for r in h["synthetic"]["rows"])
    span = P[names[0]]["overlay_span"]

    return f"""# Results — Study 1019 (The Market Runs on Volume Time) on the real daily tapes

*Generated by [`examples/verify.py`](../examples/verify.py). As-of
**{h['as_of']}** (the last session of a complete year; the bundled tapes end there).*

## Data provenance

| Tape | Source | Span | Rows | Zero-volume sessions (set missing) | SHA-256 pin | Fingerprint |
|---|---|---|--:|--:|---|---|
{prov}

**Labels that travel with every number below.** Both tapes are **price indices** — dividends
are excluded, so overlay returns are price-only. `Volume` is the **composite volume Yahoo!
Finance reports for the index** (a sum of constituent share counts), never used in levels: every
volume figure is relative to its own **past-only** 63-session rolling median. The cash leg is the
Fama–French monthly RF spread over each month's sessions (December 2018 carries November's rate).
The early S&P 500 *open* prints are unreliable (≈40% equal the prior close), so the open is
never used; the range estimator reads high and low only.

## 1. How much excess kurtosis does each clock remove?

Every clock is applied to the **same sessions** (the first two years go to warming the
past-only GARCH), and each standardised series is rescaled to unit variance. Intervals are a
paired circular block bootstrap (21-session blocks, {N_BOOT:,} resamples): raw and clocked series
are resampled together, so the CI on the *share removed* is a CI on the ratio, not on two
independent numbers.

{sec1}

Reading the table: **volume** is Clark's clock, read at the close of the day it explains.
**Range** divides by that same day's high-low volatility — an *upper* benchmark, since it is
the day's volatility measured; a share above 100% means it over-corrects into thinner-than-normal
tails. **GARCH** is past-only and is the usual benchmark. **GARCH × volume surprise** divides the
GARCH residual by same-day volume relative to its trailing 21-session mean.

## 2. Tail quantiles in volume time

Each clock rescaled to unit variance and read against N(0,1).

{sec2}

## 3. Robustness

**Detrending window.** Share of excess kurtosis removed by the volume clock as the past-only
median window changes (common sample after a one-year burn):

| Tape | {win_hdr} |
|---|{'--:|' * len(P[names[0]]['windows'])}
{win}

**Clock exponent.** Excess kurtosis of `r / relvol^(b/2)`; `b = 1` is Clark's clock, `b = 0` is
raw returns. The minimiser is chosen **in sample** and is therefore optimistic:

| Tape | k at b=0 | k at b=1 | in-sample best b | k at best b | k at b=2 |
|---|--:|--:|--:|--:|--:|
{expo}

**Sub-periods.** Share of each sub-period's raw excess kurtosis removed:

| Tape | Period | Raw excess kurtosis | Volume | Range | GARCH | GARCH × volume |
|---|---|--:|--:|--:|--:|--:|
{per}

**Volume on top of GARCH** (the Lamoureux–Lastrapes question). Excess kurtosis of the GARCH
residual minus that of GARCH × volume surprise, paired block bootstrap:

| Tape | Kurtosis reduction | 95% CI |
|---|--:|--:|
{inc}

## 4. The volume–volatility correlation: same day vs next day

Log relative volume at `t` against volatility at `t + lag`. Series standardised, so the slope is
the correlation; Newey–West t-statistics.

| Tape | Lag | Volatility proxy | Correlation | NW t | Spearman |
|---|--:|---|--:|--:|--:|
{corr}

## 5. Does yesterday's volume improve tomorrow's variance forecast?

Out of sample from {P[names[0]]['race_start']} ({P[names[0]]['race_n']:,} forecasts per tape).
**HAR** is a log-HAR on Parkinson range-variance (day / week / month); **HAR-X** adds log relative
volume of the forecast day's close; both refitted every 21 sessions on an expanding window and
mapped to a variance level with the same past-only calibration. **GARCH** is a past-only
GARCH(1,1) refitted yearly. Targets: next-day squared return (`r²`, what the overlay bears) and
next-day range variance. QLIKE in Patton's `a/f + log f` form. Diebold–Mariano with HAC: a
**negative** t favours HAR-X.

| Tape | Target | Loss | Comparison | Mean loss diff | DM t | p |
|---|---|---|---|--:|--:|--:|
{race}

In-sample check (full-sample log-HAR-X on next-day log range-variance, Newey–West):

| Tape | Volume coefficient | NW t | R² HAR | R² HAR-X |
|---|--:|--:|--:|--:|
{ins}

## 6. The vol-targeting overlay

Weight `min(1.5, 15% / forecast vol)`, formed at the close of `t`, **traded at the close of
`t+1`** (one execution lag: the day's composite volume is published after the close), earning
`t+2`. Cash leg at RF; costs one-way × traded NAV. Excess-of-cash returns, price-only. Common
span {span[0]} → {span[1]} ({span[2]:,} sessions).

| Tape | Forecast | Cost (bp) | Sharpe (excess) | Ann. excess return | Ann. vol | Turnover/yr | Mean weight |
|---|---|--:|--:|--:|--:|--:|--:|
{ov}

Sharpe differences, paired circular block bootstrap ({N_BOOT:,} resamples, 21-session blocks):

| Tape | Cost (bp) | Comparison | ΔSharpe | 95% CI | p |
|---|--:|---|--:|--:|--:|
{ovt}

**The impossible version.** Sizing today's position by *today's* volume clock (yesterday's
HAR forecast times today's relative volume, no lag at all) — not tradable, shown only to locate
the clock's value in time. The middle column is the same HAR forecast with zero lag and no
volume, so the step from it to the last column is the clock alone:

| Tape | Tradable HAR overlay (one lag), gross Sharpe | HAR, zero lag, gross Sharpe | HAR × same-day volume (look-ahead), gross Sharpe |
|---|--:|--:|--:|
{la}

## 7. Synthetic control — machinery check, not market evidence

The subordinated process of `volclock.data.synthetic_tape` (5,000 sessions): returns
N(0, σ²·m_t), with `signal_strength` the share of `log m_t` carried by the volume clock. At 1.0
Clark is true by construction; at 0.0 the returns are exactly as fat-tailed but run on an
independent clock.

| signal_strength | Raw excess kurtosis | Volume-clock kurtosis | Share removed | Same-day corr (NW t) | DM t, HAR-X vs HAR (range var, QLIKE) |
|--:|--:|--:|--:|--:|--:|
{syn}

The harness finds the clock when it is planted and reports a *negative* share when it is not
(dividing by an unrelated clock adds noise to the denominator). These are synthetic numbers and
support nothing about markets.

## Caveats

- **Index composite volume is a crude clock.** Clark and Ané–Geman used the volume or trade count
  *of the instrument itself*; an index's composite volume sums share counts across hundreds of
  stocks of very different prices, and it carries level shifts (decimalisation, venue
  fragmentation, the 2008–09 surge) that a 63-session median only partly absorbs. The longer
  detrending windows in section 3 do better; none reaches the bar on both tapes. A trade-count
  or dollar-volume clock on single stocks or futures could fare better and is the natural fork.
- **Daily data cannot test the intraday version.** Ané & Geman's near-normality is measured on
  intraday sampling by trade count; aggregating to one day blends the clock with everything else
  that moves in a session.
- **Kurtosis is an unstable statistic** on returns whose fourth moment may not exist (study
  991 puts the Hill tail index near 3); hence the block-bootstrap intervals, which are wide.
- **Parkinson misses the overnight gap.** The range clock is an upper benchmark for the *intraday*
  part only; HAR forecasts are recalibrated to the close-to-close level past-only.
- **Price-only, optimistic financing.** The overlay borrows at RF above a weight of one; the same
  for every overlay compared, so the differences are unaffected.

## Verdict

Produced by `strategy.verdict`, with thresholds fixed before the real run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [1019-volume-clock](../README.md).
Not investment advice.*
"""


def main() -> None:
    h = report()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print(f"\nwrote docs/results.md  ({h['runtime_s']:.0f} s)")
    head = {"as_of": h["as_of"], "signal": h["_verdict"]["signal"],
            "trad": h["_verdict"]["trad"], "one_sentence": h["_verdict"]["one_sentence"],
            "tapes": h["tapes"],
            "fingerprints": {n: h["per_tape"][n]["fingerprint"] for n in data.TAPES}}
    print("##HEADLINE## " + json.dumps(head, default=float))


if __name__ == "__main__":
    main()
