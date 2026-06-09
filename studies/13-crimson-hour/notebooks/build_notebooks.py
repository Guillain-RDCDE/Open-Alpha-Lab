"""Generate the two narrative notebooks for Study 13 (Crimson-Hour) from source.

Like the other studies, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic tape** — toy sessions with a *baked-in*
afternoon momentum and a deliberately uninformative IB-rejection flag — because the cached real
bars are git-ignored and the desk's reproducible core must run with no network. The synthetic is
where the decomposition *works* (it recovers a real OC-red lift, a null IB increment, and a
mostly-mechanical headline), which is exactly the point: it proves the code, so the **real
verdict** (quoted from [`docs/results.md`](../docs/results.md), produced by `examples/verify.py`)
is a fact about the market, not a bug. Both notebooks follow the SAME seven desk beats
(see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (crimson_hour/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from crimson_hour import data, signals, decompose

# Offline synthetic tape: sessions with a KNOWN afternoon momentum (0.12) and an IB-rejection
# flag drawn INDEPENDENTLY of the close -- so the decomposition has a ground truth to recover.
# The real verdict (SPY/QQQ + ES/NQ) is in ../docs/results.md via examples/verify.py.
feat, truth = data.synthetic_sessions(seed=0)
print(f"{len(feat)} synthetic sessions | baked-in momentum {truth.momentum} | "
      f"baseline red {truth.baseline_red:.1%} | IB flag null = {truth.ib_is_null}")
"""

# Real headline numbers (from docs/results.md, as-of 2026-06-01) quoted in the prose.
R = dict(
    spy_head="68.9%", spy_head_lift="+23.1 pp", spy_cont="50.9%", spy_cont_lift="+6.7 pp",
    spy_mech="71%", spy_base="45.8%",
    qqq_head="72.1%", qqq_head_lift="+27.0 pp", qqq_cont="49.0%", qqq_cont_lift="+5.6 pp",
    qqq_mech="79%", qqq_base="45.1%",
    es_conf="5/11 = 45.5%", es_ci="[21.3%, 72.0%]", nq_conf="7/16 = 43.8%", nq_ci="[23.1%, 66.8%]",
    es_fisher="0.62", nq_fisher="1.00",
    fork_true="70.5%", fork_best="84.6%", fork_p95="92.0%", fork_p88="36.2%",
)


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a red first hour really call the close? 🌅🩸\n"
            "### \"88% of the time, the day closes red\" — tested honestly, in plain English\n\n"
            "Here's the hook, straight from a trading newsletter: on the S&P 500 futures, when the "
            "**first hour (9:30–10:30 ET) closes red** *and* the hour's **high prints before its "
            "low** (so the low looks set to break), the whole day closes red **88% of the time** — "
            "**90%** on the Nasdaq. The dashboard that found it was built *\"in 5 minutes, one "
            "prompt, no code\"* against an API. By 10:30 you supposedly already know how the day "
            "ends.\n\n"
            "It sounds like a crystal ball. It is mostly a **head-start dressed up as a "
            "forecast** — plus a tiny, real grain of truth, plus the oldest trick in statistics. "
            "Let's pull those apart.\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core runs on a **synthetic** tape "
            "where we *bake in* the answer (a small, real morning→afternoon momentum, and an "
            "IB-rejection flag that is pure coin-flip noise). That's on purpose: it's where our "
            "tools *should* work, which proves the code — so the real-market numbers (quoted from "
            "[`../docs/results.md`](../docs/results.md)) are a measurement, not a hope.\n\n"
            "*Follows the desk's seven beats ([METHODOLOGY.md](../../../METHODOLOGY.md)). The "
            "rigorous version is the companion,* "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Does a red first hour tilt the close red? | ✅ **Yes** — real and sizeable: "
            f"on real SPY a red opening hour closes red **{R['spy_head']}** of the time vs "
            f"**{R['spy_base']}** baseline. |\n"
            "| Is that a *forecast* of the rest of the day? | ⚠️ **Mostly not** — strip the "
            f"head-start and the *rest* of the day (10:30→close) is red just **{R['spy_cont']}** of "
            f"the time. **{R['spy_mech']}** of the headline is mechanical. |\n"
            "| Does the \"IB-high rejection\" add anything? | ❌ **No** — once the hour is red, "
            f"the rejection flag changes nothing (Fisher p = {R['es_fisher']}). |\n"
            "| So where does **88%** come from? | 🎰 **Small samples + cherry-picking** — a true "
            f"~70% edge, mined across a dozen \"confluences\" of 25 days, *expects* a best of "
            f"**{R['fork_best']}**. |\n\n"
            "> Desk shorthand: **Signal `WEAK` · Tradability `MIRAGE` · \"88% predictive?\" "
            "`INFLATED`** — let's earn the stamps."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Two things you can see by 10:30 ET: the **opening candle** (the 9:30–10:30 hour) "
            "closes **red**, and within that hour the **high comes before the low** — the "
            "newsletter calls this *\"IB-high rejection\"*, the idea being the low is next to "
            "break. Stack them and, on the quoted sample, the day closed red **22 of 25 times "
            "(88%)** on ES and **28 of 31 (90%)** on NQ — against ~46% and ~45% baselines. The "
            "post is careful to call this *\"bias, not a trade\"* and ends with a free-dashboard "
            "sign-up. Our job is the space between *\"88% on these 25 days\"* and *\"the morning "
            "predicts the day.\"*"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If 10:30 really told you the close 9 times in 10, that would be one of the cleanest "
            "intraday edges anywhere — you'd lean every afternoon the morning's way and print. The "
            "subtler, likelier story is the one that actually helps you: a red first hour **does** "
            "tilt the day red, but mostly for a boring mechanical reason (the day is already "
            "*down* at 10:30 and just has to not fully climb back), with only a sliver of genuine "
            "*continuation* on top. Confuse the two and you'll bet real money on a \"forecast\" "
            "that's 70% bookkeeping."
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "One clean split decides it. *\"The day closes red\"* is measured from the **9:30 "
            "open** — so if the first hour is red, the day starts the afternoon already in the "
            "hole. The honest question isn't *\"does the day close red?\"* but *\"does the **rest "
            "of the day** (10:30→16:00) keep going down?\"* — the part you don't yet know at 10:30. "
            "If the rest-of-day red rate barely beats a coin, the 88% is a head-start, not a "
            "crystal ball. We prove the split works on the synthetic, where we baked the answer in."
        ),
        code(
            "m = signals.condition_masks(feat)\n"
            "tab = decompose.conditional_table(m, signals.session_red(feat))\n"
            "print('P(session closes red | morning condition) — synthetic:')\n"
            "display(tab.loc[['baseline','oc_red','confluence (oc_red & ib_rej)','oc_red & ib_NOT_rej']]\n"
            "        [['k','n','rate','wilson_low','wilson_high','lift_pp']].round(3))"
        ),
        md("Notice already: the **confluence** and **OC-red-but-not-rejected** rows sit on top of "
           "each other — adding the rejection flag does nothing. The lift is the *red hour*, full "
           "stop."),

        md(
            "## 4 · The teardown 🔬\n\n"
            "Now the same machine on the **real tape** (from [`../docs/results.md`](../docs/results.md), "
            "via `examples/verify.py`). The chart below is the whole study in one picture: for each "
            "market, the **headline** lift (day closes red | red hour) next to the **honest** lift "
            "(rest-of-day red | red hour)."
        ),
        code(
            "# Synthetic illustration of the split (the real numbers are quoted below).\n"
            "s = decompose.mechanical_vs_predictive(feat)\n"
            "labels = ['Headline\\n(day red | red hour)', 'Honest forecast\\n(rest-of-day red | red hour)']\n"
            "vals = [s['headline_lift_pp'], s['continuation_lift_pp']]\n"
            "plt.bar(labels, vals, color=['#b22222','#999999'])\n"
            "plt.axhline(0, color='k', lw=0.6); plt.ylabel('lift over baseline (pp)')\n"
            "plt.title('Synthetic: most of the headline lift is a mechanical head-start')\n"
            "for i,v in enumerate(vals): plt.text(i, v+0.3, f'{v:+.1f} pp', ha='center')\n"
            "plt.show()\n"
            "print(f\"synthetic: {s['mechanical_share']:.0%} of the headline lift is mechanical, \"\n"
            "      f\"not forecast\")"
        ),
        md(
            "**What the real tape says** (SPY = the S&P, QQQ = the Nasdaq, ~700 sessions each):\n\n"
            f"- A red first hour **does** tilt the close: SPY closes red **{R['spy_head']}** of the "
            f"time after a red hour (baseline {R['spy_base']}) — a real **{R['spy_head_lift']}**. "
            f"QQQ: **{R['qqq_head']}** ({R['qqq_head_lift']}). And the baselines (~46%, ~45%) match "
            "the newsletter's own 46.1% / 44.5% almost exactly — so this *is* the same effect.\n"
            f"- But the **rest of the day** barely continues: SPY rest-of-day red just "
            f"**{R['spy_cont']}** ({R['spy_cont_lift']} over baseline), QQQ **{R['qqq_cont']}** "
            f"({R['qqq_cont_lift']}). So **{R['spy_mech']}–{R['qqq_mech']}%** of the famous lift is "
            "the head-start; the genuine forecast is ~6 points of tilt.\n"
            f"- The **IB-rejection adds nothing**: on the real ES tape the full confluence "
            f"({R['es_conf']}) is statistically the same as a red hour that *wasn't* rejected "
            f"(Fisher p = {R['es_fisher']}; NQ p = {R['nq_fisher']}).\n"
            f"- And **88% is what mining produces**: a true ~{R['fork_true']} edge, taken as the "
            f"best of a dozen 25-day \"confluences,\" *expects* a top score of **{R['fork_best']}** "
            f"(95th pct {R['fork_p95']}) — reaching 88% by luck alone happens **{R['fork_p88']}** "
            "of the time."
        ),
        code(
            "# Why 'best of several tiny samples' manufactures a headline (synthetic Monte-Carlo).\n"
            "rng = np.random.default_rng(0)\n"
            "best = (rng.binomial(25, 0.705, size=(20000, 12)) / 25).max(axis=1)\n"
            "plt.hist(best, bins=24, color='#cccccc', edgecolor='white')\n"
            "plt.axvline(0.705, color='#1f77b4', lw=2, label='true edge 70.5%')\n"
            "plt.axvline(0.88, color='#b22222', lw=2, label='quoted 88%')\n"
            "plt.xlabel('best observed rate across 12 confluences (n=25 each)')\n"
            "plt.ylabel('frequency'); plt.legend(); plt.title('A modest edge, mined, looks like a crystal ball')\n"
            "plt.show()"
        ),

        md(
            "## 5 · The verdict ⚖️\n\n"
            "**Signal `WEAK`** — there's a real morning→afternoon continuation, but it's ~6 points "
            "of tilt, and ~3/4 of the headline is a mechanical head-start; the IB-rejection leg is "
            "nothing. **Tradability `MIRAGE`** — a *bias on the direction of the close* isn't an "
            "entry, and 6 points of directional tilt won't survive the spread and a sensible "
            "target. **\"88% predictive?\" `INFLATED`** — it's a small-sample, cherry-picked draw "
            "from a ~70% true rate. The newsletter's own hedge — *\"bias, not a trade\"* — is the "
            "honest read."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "Not as a money-maker. The thing that's *real* — a red first hour leaves the rest of "
            "the day slightly more likely to drift down — is about **6 percentage points** of "
            "direction. To trade it you'd short at 10:30 and cover at the close, paying the spread "
            "and commission every day, and eating the times the afternoon rips back up (which is "
            "nearly half of them). Six points of edge on a coin-flip-sized move is exactly the kind "
            "of thing that looks great as a *\"bias\"* and bleeds out as a *strategy*. As the "
            "newsletter says, it's a lean, not a trade — and a thin one."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **Make the head-start explicit:** for each red-hour day, plot how much of the "
            "close-red outcome was already locked in by 10:30 vs decided after.\n"
            "- **Does the tilt survive costs?** Back-test the 10:30→close short on real bars with a "
            "realistic spread and a profit target — find the break-even cost.\n"
            "- **Other sessions / regimes:** the baseline red rate swung from ~46% (the quoted "
            "window) to ~33% in our recent 60-day futures sample — the \"88%\" is partly *when* you "
            "look. Map the conditional rate across rolling windows.\n\n"
            "The deep version — Wilson intervals, the beta-binomial posterior on 22/25, the "
            "two-proportion tests, and the mining Monte-Carlo — is in "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Crimson-Hour — the teardown 🩸🔬\n"
            "### A red opening candle + IB-high rejection vs the close: head-start, increment, and the garden of forking paths\n\n"
            "The rigorous companion to [`01_for_the_curious.ipynb`](01_for_the_curious.ipynb). "
            "Same seven beats, full method. Thesis: the edgeful confluence is a **real but modest "
            "intraday-continuation effect (~+6 pp on the *forecastable* leg)** wrapped in a "
            "**mechanical head-start (~75% of the headline lift)** and inflated by **small-sample "
            "selection** to an 88–90% headline; the IB-rejection term adds nothing over the "
            "opening candle's sign.\n\n"
            "> ⚠️ **Executed on the synthetic tape** (baked-in momentum 0.12, null IB flag), where "
            "the decomposition recovers what we put in — this validates the machinery end-to-end. "
            "The real verdict is quoted from [`../docs/results.md`](../docs/results.md) "
            "(`examples/verify.py`). Fixed seeds; no network."
        ),
        code(BOOT),

        md(
            "## 1 · The claim, decomposed\n\n"
            "Let `r₁` be the opening-candle return (9:30→10:30) and `r₂` the rest-of-day return "
            "(10:30→16:00). The session closes red iff `r₁ + r₂ < 0` (in log space). The pitch "
            "quotes **P(r₁+r₂ < 0 | r₁ < 0 ∧ IB-rejected)**. Decompose it:\n\n"
            "- **Mechanical:** conditioning on `r₁ < 0` already puts the day below its open; "
            "P(session red | r₁<0) is high largely because `r₂` must *overcome* `|r₁|`, not because "
            "`r₂` is forecast.\n"
            "- **Forecast:** the only genuinely predictive quantity at 10:30 is "
            "**P(r₂ < 0 | r₁ < 0)** vs its baseline P(r₂ < 0) — intraday return continuation.\n"
            "- **Increment:** does the IB-rejection event `1[argmax_t high < argmin_t low]` shift "
            "the close-rate *given* `r₁ < 0`? H₀: it is conditionally independent of `r₂`.\n\n"
            "The synthetic bakes exactly this: `r₂ = β·r₁ + ε` with β = 0.12 (real continuation) "
            "and an IB flag drawn independently of `(r₁ magnitude, r₂)` — so the increment's true "
            "value is 0."
        ),
        code(
            "m = signals.condition_masks(feat)\n"
            "tab = decompose.conditional_table(m, signals.session_red(feat))\n"
            "display(tab.round(4))"
        ),

        md(
            "## 3–4 · The mechanical/forecast split — the load-bearing number\n\n"
            "`mechanical_vs_predictive` reports the headline conditional, the continuation "
            "conditional, both baselines, and the share of the headline lift that is the "
            "head-start. On the synthetic the split is recovered; on the real tape it is the "
            "verdict."
        ),
        code(
            "s = decompose.mechanical_vs_predictive(feat)\n"
            "for k, v in s.items():\n"
            "    print(f'  {k:>22}: {v:.4f}' if isinstance(v, float) else f'  {k:>22}: {v}')"
        ),
        md(
            f"> **Real tape** (../docs/results.md): SPY headline P(session red | OC-red) "
            f"**{R['spy_head']}** (lift {R['spy_head_lift']}) collapses to continuation "
            f"**{R['spy_cont']}** (lift {R['spy_cont_lift']}) — **{R['spy_mech']}** mechanical. "
            f"QQQ: **{R['qqq_head']}** → **{R['qqq_cont']}**, **{R['qqq_mech']}** mechanical. The "
            f"forecastable continuation edge is ~+6 pp on ~700 sessions: small, real, and an order "
            "of magnitude below the headline."
        ),

        md(
            "## 4 · Does IB-rejection add anything over OC-red?\n\n"
            "Two-proportion z and Fisher exact on (OC-red ∧ IB-rejected) vs (OC-red ∧ ¬rejected). "
            "Under the synthetic null the increment is ~0 and non-significant."
        ),
        code(
            "inc = decompose.ib_increment(feat)\n"
            "print({k: (round(v,4) if isinstance(v,float) else v) for k,v in inc.items()})"
        ),
        md(
            f"> **Real tape:** the OC-red-not-rejected control is *tiny* (ES n=6, NQ n=1 in the "
            f"60-day 5-minute window — itself a small-sample warning), and the increment is "
            f"statistically nothing (ES Fisher p = **{R['es_fisher']}**, NQ p = **{R['nq_fisher']}**). "
            "The second signal is redundant given the first — as it must be if both are just "
            "reading 'the morning was weak.'"
        ),

        md(
            "## 4b · Small-sample honesty — Wilson and the beta-binomial on 22/25\n\n"
            "The quoted 88% is `k/n = 22/25`. The textbook normal interval is invalid near 1.0 and "
            "at n=25; the Wilson score interval and a Beta(1,1) posterior are not. Both say the "
            "same thing: the point is near the top of a wide band."
        ),
        code(
            "for (k,n,label) in [(22,25,'ES 88%'), (28,31,'NQ 90%')]:\n"
            "    lo,hi = decompose.wilson_ci(k,n)\n"
            "    post = decompose.beta_binomial(k,n, thresholds=(0.7,0.6))\n"
            "    print(f'{label}: {k}/{n}={k/n:.1%}  Wilson95 [{lo:.1%},{hi:.1%}]  '\n"
            "          f\"post.mean {post['posterior_mean']:.1%} cred95 [{post['cred_low']:.1%},{post['cred_high']:.1%}]  \"\n"
            "          f\"P(rate>70%)={post['P(rate>0.7)']:.0%}\")"
        ),

        md(
            "## 4c · The garden of forking paths — \"one prompt, combine the reports\"\n\n"
            "The dashboard was built by combining reports until a confluence stood out. That is "
            "multiple comparisons. Model it: `n_candidates` confluences, each selecting ~25 "
            "sessions whose *true* close-red rate is the realistic ~70% from the high-power leg; "
            "quote the **max** observed rate. `mining_inflation` returns its distribution."
        ),
        code(
            "mining = decompose.mining_inflation(p_true=0.705, n_cond=25, n_candidates=12,\n"
            "                                    observed=0.88, seed=0)\n"
            "print({k:(round(v,4) if isinstance(v,float) else v) for k,v in mining.items()})\n"
            "# distribution\n"
            "rng = np.random.default_rng(0)\n"
            "best = (rng.binomial(25, 0.705, size=(20000,12))/25).max(axis=1)\n"
            "plt.hist(best, bins=24, color='#cccccc', edgecolor='white')\n"
            "plt.axvline(0.705, color='#1f77b4', lw=2, label='true edge 70.5%')\n"
            "plt.axvline(0.88, color='#b22222', lw=2, label='quoted 88%')\n"
            "plt.axvline(best.mean(), color='k', ls='--', lw=1.5, label=f'E[best] {best.mean():.1%}')\n"
            "plt.xlabel('best of 12 confluences (n=25)'); plt.ylabel('freq'); plt.legend()\n"
            "plt.title('Selection inflates a 70.5% edge to an ~85% expected headline'); plt.show()"
        ),
        md(
            f"> A true **{R['fork_true']}** edge mined this way *expects* a best of **{R['fork_best']}** "
            f"(95th pct **{R['fork_p95']}**), and clears the published 88% **{R['fork_p88']}** of the "
            "time. The headline is the expected output of the search, not evidence against the null. "
            "(frank's own pre-NFP dashboard returning 64% ≈ baseline is the same machine's "
            "un-published draw.)"
        ),

        md(
            "## 5 · The verdict, with the numbers\n\n"
            f"**Signal `WEAK`** — genuine continuation lift only **{R['spy_cont_lift']}** / "
            f"**{R['qqq_cont_lift']}** (SPY/QQQ, ~700 sessions); **{R['spy_mech']}–{R['qqq_mech']}%** "
            f"of the headline is a mechanical head-start; IB-rejection increment ≈ 0 (Fisher p "
            f"{R['es_fisher']}/{R['nq_fisher']}). **Tradability `MIRAGE`** — a close-*direction* "
            "bias is not an entry, and ~6 pp of directional tilt does not survive a round-trip "
            "spread plus a realistic target. **\"88% predictive?\" `INFLATED`** — a small-sample, "
            f"selection-maximised draw from a ~{R['fork_true']} true rate (E[best of 12]≈"
            f"{R['fork_best']}). Real numbers as-of 2026-06-01 in [`../docs/results.md`](../docs/results.md)."
        ),

        md(
            "## 6 · Could you trade it — the cost the bias can't pay\n\n"
            "The only positive-expectancy leg is the continuation: short at 10:30 on a red hour, "
            "cover at 16:00. Its edge is the rest-of-day directional tilt, ~**+6 pp** of "
            "P(down) — call it a few bps of expected move on a near-symmetric distribution. Charge "
            "a round-trip equity-index spread + commission *every session* (per the house rule, "
            "costs against the alpha, not the gross) and add the variance of a full-afternoon hold, "
            "and the thin directional tilt is gone. There is no capacity question — SPY/ES are "
            "bottomless — because the edge dies on costs long before size matters."
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Cost sweep on the continuation leg** — break-even spread for the 10:30→close "
            "short; pre-registered expectation: below realistic costs.\n"
            "- **Regime map** — the conditional rate across rolling windows; the baseline red rate "
            "alone moved 46%→33% between the quoted window and our recent 60-day futures sample.\n"
            "- **Magnitude conditioning** — does a *bigger* red hour forecast more continuation, or "
            "just a bigger head-start? Separate the two with a continuation regression on `r₁`.\n"
            "- **The mining audit** — re-run the dashboard's report-combination over a held-out "
            "year and count how many \"confluences\" clear 80%; compare to the null.\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md). Real run: `examples/verify.py` → "
            "[`../docs/results.md`](../docs/results.md). Literature: "
            "[`../docs/references.md`](../docs/references.md)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    targets = {
        "01_for_the_curious.ipynb": build_curious(),
        "02_for_the_quants.ipynb": build_quants(),
    }
    for fname, nb in targets.items():
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {fname}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
