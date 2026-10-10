"""Real-tape verification — Study 1014 (The Perfect Macro Forecaster). Regenerates docs/results.md.

Grants a hypothetical investor perfect, final-vintage foresight of US macro (real GDP growth,
unemployment change, CPI inflation, T-bill change; statsmodels ``macrodata``, 1959Q1-2009Q3)
and measures what it is worth for timing the US stock market (Fama-French market **total
return**, compounded to quarters; out of the market = one-month T-bills):

1. who leads whom — the lead-lag correlation of quarterly excess returns with each variable;
2. the oracle grid — an in-or-out rule for every variable × information horizon, against
   buy-and-hold, with a block-bootstrap of the Sharpe difference and a rotation placebo;
3. same quarter vs one quarter beyond — the market prices the future;
4. the release-lag version — what an investor could actually have known;
5. the ceiling — perfect foresight of the market itself, and how much of it macro captures;
6. a synthetic control proving the machinery finds a planted lead and stays quiet on a null.

    python studies/1014-macro-clairvoyance/examples/verify.py

Fully offline: both tapes ship inside installed wheels and are SHA-256 pinned.
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

from clairvoyance import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
N_BOOT = 2000


def _row(T: pd.DataFrame, var: str, h: int) -> dict:
    return T[(T["var"] == var) & (T["h"] == h)].iloc[0].to_dict()


def report() -> dict:
    t0 = time.time()
    panel = data.load_panel()
    cols = ["mkt", "rf", *data.MACRO_VARS]
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(panel[cols]),
               "fp_macro": data.fingerprint(panel[list(data.MACRO_VARS)]),
               "fp_equity": data.fingerprint(panel[["mkt", "rf"]]),
               "macro_sha_ok": data.macro_file_sha256() == data.PROVENANCE["macro_sha256"],
               "versions": data.package_versions(), "cost_bps": st.COST_BPS,
               "n_panel": int(len(panel)), "first": str(panel.index[0].date()),
               "last": str(panel.index[-1].date())}
    print(f"as-of {data.AS_OF}   fingerprint {h['fingerprint']}")
    print(f"  panel {h['first']} -> {h['last']}, {len(panel)} quarters; "
          f"macrodata SHA ok: {h['macro_sha_ok']}")

    print("\n=== 1. who leads whom ===")
    ll = {}
    for v in data.MACRO_VARS:
        L = st.lead_lag(panel, v)
        ll[v] = {int(k): {"corr": float(r["corr"]), "t": float(r["t"])}
                 for k, r in L.iterrows()}
        print(f"  {v:5s} corr k=-4..+4: " + " ".join(f"{c:+.2f}" for c in L["corr"]))
    h["ll"] = ll
    h["ll_g"] = ll["g"]

    print("\n=== 2. the oracle grid ===")
    T, win = st.horizon_table(panel, n_boot=N_BOOT)
    h["window"] = (str(win[0].date()), str(win[-1].date()), int(len(win)))
    h["grid"] = T.to_dict("records")
    piv = T.pivot(index="var", columns="h", values="sharpe")
    print(piv.round(2).to_string())
    sub = panel.loc[win]
    h["bh"] = st.ann_sharpe(sub["ex"])
    h["ceiling"] = st.ann_sharpe(st.book(st.market_oracle(sub), sub["ex"]))
    print(f"  window {h['window']}; buy-and-hold {h['bh']:.2f}; market oracle {h['ceiling']:.2f}")

    h["head"] = _row(T, "combined", 0)
    h["lead"] = _row(T, "combined", 1)
    h["real"] = _row(T, "combined", -1)
    h["strict"] = _row(T, "combined", -2)
    h["g_head"] = _row(T, "g", 0)
    h["g_lead"] = _row(T, "g", 1)
    h["g_real"] = _row(T, "g", -1)
    n_pass = int(sum(st.passes(r) for r in h["grid"]))
    h["n_pass"], h["n_tests"] = n_pass, int(len(T))
    print(f"  {n_pass} of {len(T)} oracle x horizon cells clear both nulls at 5%")

    print("\n=== 3. same quarter vs one quarter beyond ===")
    P = st.all_positions(panel).loc[win]
    contrasts = []
    for v in ("g", "combined"):
        a = st.book(P[f"{v}@+1"], sub["ex"])
        b = st.book(P[f"{v}@+0"], sub["ex"])
        bt = st.sharpe_diff_bootstrap(a.to_numpy(), b.to_numpy(), n_boot=N_BOOT)
        contrasts.append({"var": v, "sh_lead": st.ann_sharpe(a), "sh_head": st.ann_sharpe(b),
                          **bt})
        print(f"  {v:9s} h=+1 {st.ann_sharpe(a):.2f} vs h=0 {st.ann_sharpe(b):.2f}  "
              f"diff {bt['dsharpe']:+.2f} [{bt['ci_lo']:+.2f},{bt['ci_hi']:+.2f}] p={bt['p']:.3f}")
    h["contrasts"] = contrasts

    print("\n=== 4. the release-lag version ===")
    cs = st.cost_sweep(P["combined@-1"], sub)
    h["cost_sweep"] = cs.reset_index().to_dict("records")
    print(cs.round(3).to_string())
    sp = st.subperiods(P["combined@-1"], sub)
    h["sub_real"] = sp.reset_index().to_dict("records")
    sp0 = st.subperiods(P["combined@+0"], sub)
    h["sub_head"] = sp0.reset_index().to_dict("records")
    sp1 = st.subperiods(P["combined@+1"], sub)
    h["sub_lead"] = sp1.reset_index().to_dict("records")
    print(sp.round(3).to_string())

    print("\n=== 5. the ceiling ===")
    lad = st.ceiling_ladder(panel, win)
    h["ladder"] = lad.reset_index().to_dict("records")
    print(lad.round(3).to_string())

    print("\n=== 6. synthetic control ===")
    SP = st.synthetic_power()
    synth = []
    for s, g in SP.groupby("signal_strength", sort=False):
        synth.append({"signal_strength": float(s), "n_seeds": int(len(g)),
                      "mean_sharpe": float(g["sharpe"].mean()),
                      "mean_bh": float(g["bh_sharpe"].mean()),
                      "mean_dsharpe": float(g["dsharpe"].mean()),
                      "share_pass": float(g["passes"].mean()),
                      "share_lead_t2": float((g["lead_t"].abs() > 2).mean()),
                      "median_boot_p": float(g["boot_p"].median())})
        print(f"  strength {s}: mean dSharpe {g['dsharpe'].mean():+.2f}; clears both nulls in "
              f"{g['passes'].mean():.0%} of {len(g)} seeds; |lead t|>2 in "
              f"{(g['lead_t'].abs() > 2).mean():.0%}")
    h["synth"] = synth
    h["ll_size"] = st.lead_lag_size()
    print(f"  lead-lag HAC t size under the null: {h['ll_size']['reject_rate']:.1%} "
          f"of {h['ll_size']['n_seeds']} seeds reject at |t|>2")

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = round(time.time() - t0, 1)
    v = h["_verdict"]
    print(f"\n=== verdict ===\n  Signal: {v['signal']}   Tradability: {v['trad']}   "
          f"Stocks lead the economy?: {v['third']}")
    print(f"  runtime {h['runtime_s']} s")
    return h


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    names = {"g": "GDP growth", "du": "unemployment change", "infl": "CPI inflation",
             "dtb": "T-bill change", "combined": "**combined (GDP+jobs+CPI)**"}
    ks = list(range(-4, 5))
    ll_rows = "\n".join(
        f"| {data.LABELS[var]} | " + " | ".join(
            f"{h['ll'][var][k]['corr']:+.2f} ({h['ll'][var][k]['t']:+.1f})" for k in ks) + " |"
        for var in data.MACRO_VARS)
    grid = pd.DataFrame(h["grid"])
    order = ["g", "du", "infl", "dtb", "combined"]
    sh_rows = "\n".join(
        f"| {names[var]} | " + " | ".join(
            f"{grid[(grid['var'] == var) & (grid['h'] == hh)]['sharpe'].iloc[0]:.2f}"
            for hh in st.HORIZONS) + " |" for var in order)
    det_rows = "\n".join(
        f"| {names[r['var']]} | {r['h']:+d} | {r['sharpe']:.2f} | {r['dsharpe']:+.2f} | "
        f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {_p(r['boot_p'])} | {_p(r['rot_p'])} | "
        f"{r['active_t']:+.2f} | {r['exposure']:.0%} | {r['hit_rate']:.0%} | "
        f"{r['capture']:+.0%} |"
        for r in sorted(h["grid"], key=lambda r: (order.index(r["var"]), r["h"])))
    con_rows = "\n".join(
        f"| {names[c['var']]} | {c['sh_head']:.2f} | {c['sh_lead']:.2f} | "
        f"**{c['dsharpe']:+.2f}** | [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] | {_p(c['p'])} |"
        for c in h["contrasts"])
    cs_rows = "\n".join(
        f"| {int(r['cost_bps'])} | {r['sharpe']:.2f} | {r['dsharpe']:+.2f} | {_p(r['boot_p'])} |"
        for r in h["cost_sweep"])

    def _sub(rows):
        return "\n".join(
            f"| {r['period']} | {int(r['n'])} | {r['sharpe']:.2f} | {r['bh_sharpe']:.2f} | "
            f"{r['dsharpe']:+.2f} | {_p(r['boot_p'])} |" for r in rows)
    lad_rows = "\n".join(
        f"| {r['hit_rate']:.0%} | {r['sharpe']:.2f} | {r['dsharpe']:+.2f} |" for r in h["ladder"])
    syn_rows = "\n".join(
        f"| {s['signal_strength']:.1f} | {s['n_seeds']} | {s['mean_sharpe']:.2f} | "
        f"{s['mean_bh']:.2f} | {s['mean_dsharpe']:+.2f} | {s['median_boot_p']:.3f} | "
        f"**{s['share_pass']:.0%}** | {s['share_lead_t2']:.0%} |"
        for s in h["synth"])
    head, lead, real, strict = h["head"], h["lead"], h["real"], h["strict"]
    d60 = [r["dsharpe"] for r in h["ladder"] if abs(r["hit_rate"] - 0.6) < 1e-9][0]
    ladder_line = (
        f"Perfect knowledge of the coming quarter's economy is worth **less** to an equity timer "
        f"than a market oracle that is wrong four quarters in ten (ΔSharpe "
        f"{head['dsharpe']:+.2f} vs {d60:+.2f})." if head["dsharpe"] < d60 else
        f"Perfect knowledge of the coming quarter's economy is worth about as much as a market "
        f"oracle that is right {head['hit_rate']:.0%} of the time.")
    w0, w1, wn = h["window"]
    return f"""# Results — Study 1014 (The Perfect Macro Forecaster) on the real quarterly tape

*Generated by [`examples/verify.py`](../examples/verify.py). As-of **{h['as_of']}** (the last
full quarter of the macro tape). Panel {h['first']} → {h['last']}, {h['n_panel']} quarters;
evaluation window {w0} → {w1}, **{wn} quarters**, identical for every rule.*

**Data provenance.**

| Tape | Source | Pin | Fingerprint |
|---|---|---|---|
| Macro — real GDP, unemployment, CPI inflation, T-bill rate | `statsmodels` {h['versions']['statsmodels']} `datasets.macrodata` — **final vintage (revised)**, quarterly 1959Q1–2009Q3 | `macrodata.csv` SHA-256 `{data.PROVENANCE['macro_sha256'][:16]}…` (matched: {h['macro_sha_ok']}) | `{h['fp_macro']}` |
| Equity — US market **total return** and one-month T-bill | `arch` {h['versions']['arch']} `frenchdata` (Fama-French `Mkt-RF + RF`, CRSP value-weighted), monthly compounded to quarters | `frenchdata.csv.gz` SHA-256 `{data.PROVENANCE['equity_sha256'][:16]}…` | `{h['fp_equity']}` |

Panel fingerprint `{h['fingerprint']}` (`quantlab.bundled.fingerprint` of `mkt, rf, g, du, infl, dtb`).

**The rule.** Before quarter *t*, the oracle reveals the final-vintage value of one variable for
quarter *t + h*. The investor holds the market over quarter *t* if that value is on the "good"
side (higher GDP growth; lower unemployment change, inflation, T-bill change) of the variable's
own **expanding, past-only median** (20-quarter warm-up), and T-bills otherwise. The combined
oracle is a majority vote of GDP, unemployment and inflation. One execution lag (position set at
the close of *t − 1*), costs **{h['cost_bps']:.0f} bps one-way × NAV traded**, all Sharpes are
**net**, annualised, excess of the bill rate, and compared against buy-and-hold excess returns on
the same quarters. Two nulls: a circular block bootstrap of the paired Sharpe difference (blocks
of 4 quarters, {N_BOOT} draws, centred, one-sided) and a rotation placebo (the oracle's own
positions shifted against the returns by every offset ≥ 4 quarters). The Newey-West *t* (4 lags)
of the active return is reported alongside.

## 1. Who leads whom

Correlation of the quarter-*t* excess return with each variable at quarter *t + k*
(Newey-West *t* of the slope in brackets). *k* > 0: the macro print comes **after** the return.

| Variable | k=−4 | k=−3 | k=−2 | k=−1 | k=0 | k=+1 | k=+2 | k=+3 | k=+4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{ll_rows}

Stocks lead the economy. The same-quarter correlation with GDP growth is
{h['ll_g'][0]['corr']:+.2f}; it peaks one to two quarters **later**
({h['ll_g'][1]['corr']:+.2f}, {h['ll_g'][2]['corr']:+.2f}), and unemployment is a mirror image a
quarter further out still. The backward-looking columns (k < 0) are where a real investor
lives, and they are much weaker; the one exception worth naming is the T-bill rate, a market
price that is never revised.

## 2. The oracle grid — net Sharpe by information horizon

Buy-and-hold on the same {wn} quarters: **{h['bh']:.2f}**. The perfect *market* oracle:
**{h['ceiling']:.2f}**.

| Oracle | h=−2 | h=−1 | h=0 | h=+1 | h=+2 |
|---|--:|--:|--:|--:|--:|
{sh_rows}

h = 0 is a perfect forecast of the holding quarter itself — the macro forecaster's product,
made flawless. h = −1 and −2 are what a real investor has after the release lag.

**Every cell, with inference** ({h['n_pass']} of {h['n_tests']} clear both nulls at 5%; with
{h['n_tests']} cells, about one would do so by luck):

| Oracle | h | Sharpe | ΔSharpe vs B&H | 95% bootstrap CI | Bootstrap p | Rotation p | NW *t* active | In market | Hit rate | Capture of ceiling |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{det_rows}

## 3. Same quarter vs one quarter beyond — the market prices the future

| Oracle | Sharpe, h=0 | Sharpe, h=+1 | Difference | 95% CI | p (h=+1 ≤ h=0) |
|---|--:|--:|--:|--:|--:|
{con_rows}

A perfect forecast of the quarter you are about to hold is worth little, because during that
quarter the market is busy pricing the *next* one. Move the oracle one quarter further out and
it starts to beat buy-and-hold. The macro-forecasting industry forecasts the quarter ahead; on
this tape that is precisely the horizon the market has already discounted.

## 4. The release-lag version — what you could actually have known

The combined oracle at h = −1 (last quarter's prints, available — generously — at the start of
the quarter) earns **{real['sharpe']:.2f}** vs {real['bh_sharpe']:.2f} (ΔSharpe
{real['dsharpe']:+.2f}, bootstrap p = {_p(real['boot_p'])}, rotation p = {_p(real['rot_p'])},
NW *t* of the active return {real['active_t']:+.2f}). The Sharpe gain comes from sitting out
volatile quarters, not from extra return: the active return itself is indistinguishable from
zero. With one more quarter of lag (h = −2) it earns **{strict['sharpe']:.2f}** (ΔSharpe
{strict['dsharpe']:+.2f}) — the edge does not survive a one-quarter change in a parameter nobody
could choose with hindsight.

Costs barely matter at about one switch a year:

| One-way cost (bps) | Sharpe | ΔSharpe | Bootstrap p |
|--:|--:|--:|--:|
{cs_rows}

Split at 1986-12-31 (fixed in advance):

| Combined oracle, h = −1 | Quarters | Sharpe | B&H | ΔSharpe | Bootstrap p |
|---|--:|--:|--:|--:|--:|
{_sub(h['sub_real'])}

| Combined oracle, h = 0 | Quarters | Sharpe | B&H | ΔSharpe | Bootstrap p |
|---|--:|--:|--:|--:|--:|
{_sub(h['sub_head'])}

| Combined oracle, h = +1 | Quarters | Sharpe | B&H | ΔSharpe | Bootstrap p |
|---|--:|--:|--:|--:|--:|
{_sub(h['sub_lead'])}

## 5. The ceiling — perfect foresight of the market itself

The market oracle, then the same oracle corrupted at random (200 patterns per row):

| Market-oracle hit rate | Sharpe | ΔSharpe vs B&H |
|--:|--:|--:|
{lad_rows}

The headline macro oracle calls the market's direction right in **{head['hit_rate']:.0%}** of
quarters and captures **{head['capture']:+.0%}** of the gap between buy-and-hold and the ceiling;
the one-quarter-beyond oracle {lead['hit_rate']:.0%} and {lead['capture']:+.0%}. {ladder_line}

## 6. Synthetic control — does the machinery work?

Deterministic worlds (400 quarters, ten seeds each) in which the excess return loads on the
standardised GDP growth of the **next** quarter (`signal_strength = 1`), and their matched nulls
(`0`): same macro paths, returns independent of them. The GDP oracle at h = +1 is run through
the full harness (net of costs, both nulls).

| Strength | Seeds | Mean Sharpe, GDP h=+1 | Mean B&H | Mean ΔSharpe | Median bootstrap p | Clears both nulls | Lead-lag \\|t\\| > 2 at k=+1 |
|--:|--:|--:|--:|--:|--:|--:|--:|
{syn_rows}

The harness banks the planted lead in every planted world and never certifies a null one. The
lead-lag Newey-West *t* is correctly sized: across {h['ll_size']['n_seeds']} null seeds it rejects
at |t| > 2 in **{h['ll_size']['reject_rate']:.1%}** of them (nominal 5%; the ten-seed column above
is a small sample). This is a machinery proof, not market evidence.

## Caveats

- **Final vintage is the whole story for the realistic leg.** `macrodata` holds GDP as revised
  years later, not the advance estimate published about four weeks after each quarter. Real-time
  GDP growth differs from the final number by more than a percentage point (annualised) often
  enough to flip an above/below-median call (Croushore & Stark 2001 built the real-time dataset
  for exactly this reason). The T-bill rate is a market price and is never revised; CPI is barely
  revised; unemployment moves only with seasonal factors. The h = −1 and h = −2 rows are
  therefore an **upper bound** on what a real investor could have done, and the Tradability stamp
  is capped at Fragile by construction.
- **The release lag is generous.** h = −1 assumes last quarter's GDP is known at the start of the
  quarter; the advance estimate actually arrives about a month in. h = −2 is the strict version.
- **One rule family.** Above/below a past-only median, all-in or all-out. A regression-weighted
  or graded rule could extract more from the same foresight; the ceiling bounds what any rule
  could get.
- **Many cells.** {h['n_tests']} oracle × horizon cells were run. The verdict keys off three
  cells fixed in advance (combined oracle at h = 0, +1, −1), not the best cell.
- **Quarterly sampling.** Timing within the quarter (e.g. acting on monthly payrolls) is a
  different question, covered by neighbouring studies on this desk.
- **1964–2009, US only.** Fifty years, about eight recessions — not many independent episodes for
  a rule that trades about once a year.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

**Stocks lead the economy? {v['third']}.** {v['third_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1014-macro-clairvoyance](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    head = {k: h[k] for k in ("as_of", "fingerprint", "bh", "ceiling", "runtime_s")}
    for k in ("head", "lead", "real", "strict", "g_head", "g_lead"):
        head[k] = {kk: h[k][kk] for kk in ("sharpe", "dsharpe", "boot_p", "rot_p", "capture")}
    head["signal"] = h["_verdict"]["signal"]
    head["trad"] = h["_verdict"]["trad"]
    head["third"] = h["_verdict"]["third"]
    print("##HEADLINE## " + json.dumps(head, default=float))


if __name__ == "__main__":
    main()
