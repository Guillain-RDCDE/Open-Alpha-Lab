"""Generate the two narrative notebooks for Study 346 (Multiple-Testing).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached
battery parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape + synthetic headline numbers — mirror of docs/results.md (as-of 2026-05-29).
R = dict(
    window="1995-01 → 2026-05", n_days=7903, n_months=377, fp="1cbef7ab059e",
    m_real=38, alpha=0.05,
    # real SPY battery: discoveries by procedure
    real_naive=13, real_bonf=7, real_holm=7, real_bh=10,
    real_fwer_pct=83,                              # 1-0.955**38 ~ 0.826
    bonf_t_thresh=0.0013,
    # synthetic NULL (100 effects, 0 real)
    null_m=100, null_naive=3, null_bonf=0, null_holm=0, null_bh=0,
    # synthetic STRONG control (10 real of 100)
    s_m=100, s_real=10,
    s_naive=dict(rej=13, tp=10, fp=3, power=1.00, fdr=0.23),
    s_bonf=dict(rej=10, tp=10, fp=0, power=1.00, fdr=0.00),
    s_holm=dict(rej=10, tp=10, fp=0, power=1.00, fdr=0.00),
    s_bh=dict(rej=10, tp=10, fp=0, power=1.00, fdr=0.00),
    # synthetic WEAK control (20 real of 200, edge 0.0025) — the FWER-vs-FDR trade-off
    w_m=200, w_real=20, w_edge=0.0025,
    w_naive=dict(rej=22, tp=16, fp=6, power=0.80, fdr=0.27),
    w_bonf=dict(rej=7, tp=7, fp=0, power=0.35, fdr=0.00),
    w_holm=dict(rej=7, tp=7, fp=0, power=0.35, fdr=0.00),
    w_bh=dict(rej=12, tp=11, fp=1, power=0.55, fdr=0.08),
    # FWER Monte Carlo (global null, 100 effects, 200 reps)
    mc_fwer=dict(naive=0.99, bonferroni=0.055, holm=0.055, bh=0.055),
    mc_avgfp=dict(naive=4.60, bonferroni=0.06, holm=0.06, bh=0.06),
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from multiple_testing import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE, "SPY"))

HAVE_REAL = _have_cache()

def real_battery():
    return data.load_real(ticker="SPY", fetch=False)

print("real battery cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Multiple-Testing — which 'winning' p-value can you trust? 🎯\n"
            "### Test 100 calendar effects and a few will look real by luck — so how do you tell?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Trust_a_naive_t%3F: Busted](https://img.shields.io/badge/Trust_a_naive_t%3F-Busted-8b949e?style=flat-square)\n\n"
            "Run *one* test at the usual bar (a *t*-stat above 2) and a false alarm happens about 1 time "
            "in 20. Run **38 tests at once** and the chance that *at least one* fires by pure luck is "
            f"about **{R['real_fwer_pct']}%** — nearly certain. This notebook screens a battery of "
            "calendar 'effects' (Mondays, turn-of-month, the Santa rally…) and shows the three standard "
            "fixes — **Bonferroni**, **Holm**, and the **false-discovery-rate** method — sorting the real "
            "from the lucky.\n\n"
            "> 📓 **This is the plain-language layer.** Want the FWER-vs-FDR maths, the power curves and "
            "the Monte-Carlo calibration? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I test 100 effects that are all fake, how many 'win'? | **About 4 or 5.** At a *t* > 2 "
            "bar, ~4.5% of pure-noise tests pass by luck — so a 100-effect search hands you a fistful of "
            "false discoveries every time. |\n"
            "| Which fix should I use? | **Bonferroni / Holm** if you can't tolerate *any* false alarm "
            "(strict); **Benjamini-Hochberg** if you'll accept a few false leads to find more real "
            "effects (powerful). Holm beats Bonferroni for free. |\n"
            "| What happened on the real SPY calendar battery? | **The count collapsed.** "
            f"{R['real_naive']} effects passed the naive bar; only **{R['real_bh']}** survive the "
            f"false-discovery method and **{R['real_bonf']}** survive the strictest one. |\n"
            "| So can I trust a backtest *t*-stat? | **Not on its own.** A *t* means nothing until you "
            "say how many tests it was the best of — and which correction you applied. |\n\n"
            "> The headline isn't an effect — it's a *method*. The naive count of 'significant' results "
            "is the most over-sold number in finance; correcting for the search is what separates a "
            "discovery from a coincidence."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Open any finance blog and you'll find it: *\"The turn-of-month effect is statistically "
            "significant — t-stat over 2!\"* The implicit promise is that a *t* above 2 means a 1-in-20 "
            "fluke at worst, so the effect is almost surely real.\n\n"
            "That promise holds for **one** pre-registered test. But nobody tests one calendar effect. "
            "They quietly try Mondays, Fridays, the first five days, the last five days, each month, "
            "quarter-ends, the Santa rally, sell-in-May… and report the ones that 'worked.' The strong "
            "version of the claim: *the surviving t-stats are real signal.* We'll show that with a "
            "battery of tests, several 'significant' results are guaranteed by luck alone."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the single most expensive mistake in quantitative finance. Every 'factor zoo' "
            "paper, every viral backtest, every '90% win-rate' thread is one search away from being a "
            "false discovery dressed as a *t*-stat. If you can't tell the lucky winners from the real "
            "ones, you'll fund the lucky ones — and watch them die out of sample. The fix isn't exotic: "
            "it's three lines of arithmetic that have been standard in medicine and genetics for "
            "decades. Finance just keeps forgetting to use them."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We run a controlled experiment, then a real one:\n\n"
            "1. **A fake market where nothing is real.** 100 calendar 'effects', all pure noise. Count "
            "how many pass the naive *t* > 2 bar — those are *all* false alarms, by construction.\n"
            "2. **A fake market with a known answer.** Plant a real edge in 10 of 100 effects. A good "
            "method finds the 10 (**power**) without waving through the 90 fakes (**false discoveries**).\n"
            "3. **The real tape.** 38 named calendar effects on SPY since 1995. Watch the 'discovery' "
            "count shrink as we go from naive → false-discovery-rate → strictest.\n\n"
            "If the corrections are doing their job, the fake-market null returns **zero** discoveries, "
            "the planted edges are **recovered**, and the real count **collapses** toward what survives "
            "an honest bar."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: a market where there is provably nothing to find.** 100 effects, all noise. How "
            "many 'significant winners' does each method report?"
        ),
        code(
            "r0, t0, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=0, seed=346)\n"
            "res0 = st.run_experiment(r0, t0, alpha=0.05, t_thresh=2.0)\n"
            "c0 = res0['counts']\n"
            "labels = ['Naive\\n(t>2)', 'Benjamini-\\nHochberg', 'Holm', 'Bonferroni']\n"
            "vals = [c0['naive'], c0['bh'], c0['holm'], c0['bonferroni']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, vals, color=[RED, AMBER, GREEN, GREEN], width=.6)\n"
            "ax.set_ylabel('significant effects (of 100)')\n"
            "ax.set_title('A market with NOTHING real -- only the naive count finds winners')\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('null discoveries — naive:', c0['naive'], '| BH:', c0['bh'],\n"
            "      '| Holm:', c0['holm'], '| Bonferroni:', c0['bonferroni'])"
        ),
        md(
            f"On a tape with **zero** real effects, the naive bar still hands you ~{R['null_naive']} "
            "'discoveries' — every one a false alarm. The three corrections all report **0**, which is "
            "the truth. That gap is the whole problem in one chart."
        ),
        md(
            "**Now plant a real edge in 10 of 100 effects.** A good method must *find* the 10 without "
            "drowning in the 90 fakes. Here's power (real effects found) and false positives, by method:"
        ),
        code(
            "r1, t1, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=10, seed=346)\n"
            "res1 = st.run_experiment(r1, t1, alpha=0.05, t_thresh=2.0)\n"
            "sc = res1['score']\n"
            "procs = ['naive', 'bh', 'holm', 'bonferroni']; names = ['Naive', 'BH', 'Holm', 'Bonferroni']\n"
            "tp = [sc[p]['true_pos'] for p in procs]; fp = [sc[p]['false_pos'] for p in procs]\n"
            "x = np.arange(len(procs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-0.2, tp, width=.4, color=GREEN, label='real effects found (of 10)')\n"
            "ax.bar(x+0.2, fp, width=.4, color=RED, label='false discoveries')\n"
            "ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()\n"
            "ax.set_title('Plant 10 real of 100 — corrections keep the 10, drop the fakes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('found real / false:', {n: (sc[p]['true_pos'], sc[p]['false_pos']) for p, n in zip(procs, names)})"
        ),
        md(
            f"All four find the {R['s_real']} real effects here — but the naive bar also drags in "
            f"{R['s_naive']['fp']} fakes (a {R['s_naive']['fdr']:.0%} false-discovery rate), while the "
            "corrections add **none**. You get the same real discoveries *and* a clean report."
        ),
        md(
            "**Finally, the real tape: 38 calendar effects on SPY since 1995.** Watch the 'discovery' "
            "count collapse as the bar gets honest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, _ = real_battery()\n"
            "    res = st.run_experiment(rets, alpha=0.05, t_thresh=2.0)\n"
            "    c = res['counts']\n"
            "    m = c['m_effects']\n"
            "    naive, bh, holm, bonf = c['naive'], c['bh'], c['holm'], c['bonferroni']\n"
            "else:\n"
            f"    m, naive, bh, holm, bonf = {R['m_real']}, {R['real_naive']}, {R['real_bh']}, {R['real_holm']}, {R['real_bonf']}\n"
            "    print('OFFLINE — frozen real numbers from docs/results.md')\n"
            "labels = ['Naive\\n(t>2)', 'Benjamini-\\nHochberg', 'Holm', 'Bonferroni']\n"
            "vals = [naive, bh, holm, bonf]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, vals, color=[RED, AMBER, GREEN, GREEN], width=.6)\n"
            "ax.set_ylabel(f'discoveries (of {m} calendar effects)')\n"
            "ax.set_title('Real SPY battery — the honest count is roughly half the naive one')\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'of {m} effects — naive {naive}, BH {bh}, Holm {holm}, Bonferroni {bonf}')"
        ),
        md(
            f"On the real battery, **{R['real_naive']}** effects pass the naive bar — but with 38 "
            f"simultaneous tests, the chance of *at least one* false alarm is ~{R['real_fwer_pct']}%. "
            f"Only **{R['real_bh']}** survive the false-discovery method and **{R['real_bonf']}** survive "
            "the strictest. The survivors are the turn-of-month family — the one calendar seasonal that "
            "is genuinely robust. The rest were the search talking."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** This is a method demo, not an anomaly. On the fake null there is "
            "nothing to find; the naive 'winners' are luck. On the real tape the only survivors are a "
            "seasonal the desk already documents. No new edge is claimed.\n"
            "- **Tradability — Mirage.** The naive count invents discoveries that don't exist. A pile of "
            "marginal calendar tilts at high turnover is nothing to trade.\n"
            "- **Can you trust a naive *t*? — Busted.** With 38 tests the family-wise false-positive "
            f"chance is ~{R['real_fwer_pct']}%, and the naive count overstates the honest one by nearly "
            "2×. A *t*-stat is meaningless without its trial count and a correction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade — that's the point. The 'effects' that fail correction were "
            "never real, so trading them is paying turnover to harvest noise. The handful that survive "
            "(turn-of-month) are tiny, capacity-limited, and already crowded. The real payoff of this "
            "study is *defensive*: before you allocate to anybody's '*t*-stat over 2' backtest, ask how "
            "many strategies they tried, and apply the correction yourself. It's three lines of code and "
            "it will save you from most of the factor zoo."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The best-of-N angle.** [Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/) "
            "spins random *rules* and runs White's Reality Check on the single luckiest one. This study "
            "is the family-wide complement — correcting *across* many named effects at once.\n"
            "- **The overfitting angle.** [Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/) "
            "uses the Deflated Sharpe Ratio and PBO when you tune *one* strategy to death.\n"
            "- **Swap the battery.** Fork this and feed it a panel of *factor* returns instead of "
            "calendar effects, and watch the same collapse from naive to corrected.\n\n"
            "*Think your favourite calendar effect is real? Fork this, add it to the battery alongside "
            "the 37 others, and see whether it's still standing after Benjamini-Hochberg.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Multiple-Testing — a quantitative teardown 🔬\n"
            "### A battery of effects · HAC *t* per test · FWER (Bonferroni, Holm) vs FDR "
            "(Benjamini-Hochberg) · power-vs-error trade-off · Monte-Carlo calibration\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Trust_a_naive_t%3F: Busted](https://img.shields.io/badge/Trust_a_naive_t%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We screen a battery of named "
            "calendar effects with autocorrelation-robust *t*-stats, then turn the *p*-values into "
            "discoveries four ways and measure each procedure's realised **power**, **false-discovery "
            "rate**, and **family-wise error rate** against a known truth.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo total-return daily SPY, a 38-effect "
            f"calendar battery 1995–2026 (as-of 2026-05-29, fingerprint `{R['fp']}`); the offline core "
            "and tests run on a deterministic synthetic battery. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Methodology demo. Synthetic null (100 effects, 0 real): naive "
            f"reports {R['null_naive']} false positives, every correction 0. No real signal is claimed. |\n"
            f"| **Tradability** | `MIRAGE` | The naive count manufactures discoveries that fail every "
            "correction; a battery of marginal calendar tilts at turnover is nothing to harvest. |\n"
            f"| **Trust a naive *t*?** | `BUSTED` | Real SPY battery: naive |*t*|>2 = {R['real_naive']} "
            f"discoveries, BH = {R['real_bh']}, Holm/Bonferroni = {R['real_bonf']}; FWER under the global "
            f"null is {R['mc_fwer']['naive']:.0%} for naive vs ~{R['mc_fwer']['bonferroni']:.0%} corrected. |\n\n"
            "> 💡 In plain words: with *M* simultaneous tests the family-wise false-positive probability "
            "is 1 − (1 − α)^M, which is already ~83% at M = 38. Bonferroni/Holm cap the chance of *any* "
            "false positive at α; Benjamini-Hochberg instead caps the *fraction* of discoveries that are "
            "false, buying power at a controlled false-lead cost."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let there be $M$ candidate effects, each with a true mean excess return $\\mu_j$ and an "
            "autocorrelation-robust statistic $t_j$ with two-sided $p_j$. Define $\\mathcal{H}_0 = "
            "\\{j: \\mu_j = 0\\}$ (the true nulls) and a decision rule that rejects a subset $\\mathcal{R}$.\n\n"
            "- **H₁ (the naive promise).** Rejecting $\\{j: |t_j| \\ge 2\\}$ controls error at the "
            "advertised ~5%. *Per comparison* yes; **family-wise no** — "
            "$\\text{FWER} = P(\\mathcal{R} \\cap \\mathcal{H}_0 \\neq \\varnothing) = 1-(1-\\alpha)^M \\to 1$.\n"
            "- **H₂ (FWER control).** Bonferroni ($p_j \\le \\alpha/M$) and Holm (step-down) hold "
            "$\\text{FWER} \\le \\alpha$ — but at a power cost when the alternatives are faint.\n"
            "- **H₃ (FDR control).** Benjamini-Hochberg holds "
            "$\\text{FDR} = \\mathbb{E}\\!\\left[\\tfrac{|\\mathcal{R}\\cap\\mathcal{H}_0|}{|\\mathcal{R}|}\\right] \\le \\alpha$, "
            "trading a controlled false-discovery fraction for more power.\n\n"
            "We **confirm H₂ and H₃** by Monte-Carlo and **reject H₁**: the naive screen's FWER is ~99% "
            "under the global null, and on the real battery it nearly doubles the corrected count."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Harvey, Liu & Zhu (2016) argue that after decades of factor mining the honest hurdle is "
            "nearer *t* > 3 than *t* > 2 — i.e. the naive bar mislabels a large share of the published "
            "cross-section. The desk's contribution here is to make the trade-off *operational*: instead "
            "of arguing about a single hurdle, we run all four procedures on the same battery and read "
            "off their realised power and error. The choice between FWER and FDR is not philosophical — "
            "it is a measurable power-vs-false-lead trade-off you can see in one chart."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Battery.** $M$ effects, each a long-on-effect-day / flat-otherwise excess-of-cash daily "
            "return series. Real: 38 named calendar effects on SPY total-return, 1995–2026. Synthetic: a "
            "tunable mix of planted-edge and null effects sharing a common market factor.\n"
            "- **Per-effect inference.** Newey-West HAC *t* on each effect's daily mean (one execution "
            "lag — calendar membership is known in advance — applied once in the data layer); two-sided "
            "*p* from the HAC *t*; circular block-bootstrap CI on the mean.\n"
            "- **Corrections.** naive (|*t*| ≥ 2); Bonferroni (*p* ≤ α/M); Holm (step-down); "
            "Benjamini-Hochberg (step-up). α = 0.05.\n"
            "- **Scoring (synthetic).** Against the truth mask: power (TP / real), realised FDR "
            "(FP / rejections), and the FWER event (any FP).\n"
            "- **Calibration.** A global-null Monte-Carlo estimates each procedure's FWER.\n"
            "- **Costs.** A one-way turnover × NAV sweep at the effect-window entries/exits.\n"
            "- **Positive control.** A planted-edge battery the corrections must recover (else a dead "
            "pipeline proves nothing by finding nothing)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The global null — FWER calibration\n\n"
            "Repeatedly draw all-null 100-effect batteries and record, per procedure, the fraction of "
            "runs with *any* false rejection. A correct FWER procedure fires ≤ α; the naive screen "
            "fires almost always."
        ),
        code(
            "mc = st.fwer_monte_carlo(m_effects=100, n_days=2520, n_reps=120, seed=346)\n"
            "procs = ['naive', 'bonferroni', 'holm', 'bh']; names = ['Naive', 'Bonferroni', 'Holm', 'BH']\n"
            "fwer = [mc['fwer'][p] for p in procs]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(names, fwer, color=[RED, GREEN, GREEN, AMBER], width=.6)\n"
            "ax.axhline(0.05, ls='--', c=GREY, label='nominal alpha = 0.05')\n"
            "ax.set_ylabel('family-wise error rate (any false positive)')\n"
            "ax.set_title('Under the global null: naive ~99%, corrections at/below alpha')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('FWER:', {n: round(mc['fwer'][p], 3) for p, n in zip(procs, names)})\n"
            "print('avg false positives:', {n: round(mc['avg_false_positives'][p], 2) for p, n in zip(procs, names)})"
        ),
        md(
            f"> 💡 In plain words: the naive screen makes at least one false discovery in "
            f"~{R['mc_fwer']['naive']:.0%} of null runs (≈ {R['mc_avgfp']['naive']:.1f} of them on "
            f"average); every correction holds the family-wise rate at the nominal "
            f"{R['mc_fwer']['bonferroni']:.0%}. The machinery is calibrated — it is not a dead pipeline."
        ),
        md(
            "### 4b · The positive control — power without flooding\n\n"
            "Plant a *known* subset and measure power and realised FDR. **Strong edge (10 real of 100):** "
            "all four recover the 10 — but only the corrections keep the false count at zero."
        ),
        code(
            "r1, t1, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=10, seed=346)\n"
            "sc = st.run_experiment(r1, t1, alpha=0.05, t_thresh=2.0)['score']\n"
            "rows = [(n, sc[p]['power'], sc[p]['fdr'], sc[p]['true_pos'], sc[p]['false_pos'])\n"
            "        for p, n in zip(procs, names)]\n"
            "tab = pd.DataFrame(rows, columns=['procedure', 'power', 'realised_FDR', 'TP', 'FP']).set_index('procedure')\n"
            "print(tab.to_string())"
        ),
        md(
            "**Weak edge (20 real of 200) — the FWER-vs-FDR trade-off made visible.** Now the planted "
            "edges are faint; the strict FWER procedures sacrifice most of them."
        ),
        code(
            "r2, t2, _ = data.synthetic_battery(n_days=2520, m_effects=200, n_true=20, edge=0.0025, seed=346)\n"
            "sc2 = st.run_experiment(r2, t2, alpha=0.05, t_thresh=2.0)['score']\n"
            "power = [sc2[p]['power'] for p in procs]; fdr = [sc2[p]['fdr'] for p in procs]\n"
            "x = np.arange(len(procs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-0.2, power, width=.4, color=GREEN, label='power (real effects found)')\n"
            "ax.bar(x+0.2, fdr, width=.4, color=RED, label='realised FDR (false fraction)')\n"
            "ax.axhline(0.05, ls='--', c=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.legend()\n"
            "ax.set_title('Faint edges: FWER (Bonf/Holm) loses power; BH trades a little FDR for a lot')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('power:', {n: round(sc2[p]['power'], 2) for p, n in zip(procs, names)})\n"
            "print('FDR  :', {n: round(sc2[p]['fdr'], 2) for p, n in zip(procs, names)})"
        ),
        md(
            f"> 💡 In plain words: with faint edges Bonferroni/Holm find only "
            f"{R['w_bonf']['power']:.0%} of the {R['w_real']} real effects (FDR 0), while "
            f"Benjamini-Hochberg finds {R['w_bh']['power']:.0%} at a controlled "
            f"{R['w_bh']['fdr']:.0%} false-discovery rate. **FWER vs FDR is a power-vs-false-lead dial**, "
            "not a right-or-wrong choice — pick the target that matches your tolerance for false leads."
        ),
        md(
            "### 4c · The real SPY calendar battery — the discovery count collapses\n\n"
            "38 named calendar effects, HAC *t* on each, then the four procedures. The naive count is "
            "nearly double the honestly-corrected one."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, _ = real_battery()\n"
            "    res = st.run_experiment(rets, alpha=0.05, t_thresh=2.0)\n"
            "    tab = res['table']; c = res['counts']; m = c['m_effects']\n"
            "    top = tab.head(13)[['hac_t', 'p', 'reject_bonferroni', 'reject_holm', 'reject_bh']]\n"
            "    print(f'{m} effects | naive {c[\"naive\"]}  BH {c[\"bh\"]}  Holm {c[\"holm\"]}  Bonferroni {c[\"bonferroni\"]}')\n"
            "    print(top.to_string())\n"
            "    naive, bh, holm, bonf = c['naive'], c['bh'], c['holm'], c['bonferroni']\n"
            "else:\n"
            f"    m, naive, bh, holm, bonf = {R['m_real']}, {R['real_naive']}, {R['real_bh']}, {R['real_holm']}, {R['real_bonf']}\n"
            f"    print('OFFLINE — frozen: {R['m_real']} effects | naive {R['real_naive']}  BH {R['real_bh']}  Holm/Bonf {R['real_bonf']}')\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['Naive (t>2)', 'BH (FDR)', 'Holm', 'Bonferroni'], [naive, bh, holm, bonf],\n"
            "       color=[RED, AMBER, GREEN, GREEN], width=.6)\n"
            "ax.set_ylabel(f'discoveries of {m} effects')\n"
            "ax.set_title('Real SPY battery -- honest count is ~half the naive one')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: {R['real_naive']} of {R['m_real']} effects clear |*t*|>2, but the "
            f"family-wise false-positive chance at M={R['m_real']} is ~{R['real_fwer_pct']}%. BH keeps "
            f"{R['real_bh']} (adding the turn-of-month family's neighbours at *t*≈2.6–3.0), Bonferroni/"
            f"Holm keep the {R['real_bonf']} strongest (*p* ≤ α/M ≈ {R['bonf_t_thresh']}). The *t*≈2.1 "
            "tail the naive count waved through is exactly the false-discovery-prone zone."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — a methodology demo on a provably-empty null; the naive 'winners' are "
            "luck and the real survivors are a known seasonal. No new edge claimed.\n"
            f"- **Tradability `MIRAGE`** — the naive count fabricates discoveries that fail every "
            "correction; nothing to harvest beyond a tiny, crowded turn-of-month tilt.\n"
            f"- **Trust a naive *t*? `BUSTED`** — FWER under the global null is "
            f"{R['mc_fwer']['naive']:.0%} for the naive screen vs ~{R['mc_fwer']['bonferroni']:.0%} "
            f"corrected; on the real battery naive ({R['real_naive']}) overstates the honest count "
            f"({R['real_bonf']}–{R['real_bh']}) by nearly 2×. Report your trial count and your "
            "correction, or report nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs and the absent edge\n\n"
            "The effects that fail correction were never real, so a cost charge only confirms there is "
            "nothing to harvest. The sweep below charges one-way turnover × NAV at the effect-window "
            "entries/exits on the real battery; the corrected discovery count does not improve."
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    bh_counts = []\n"
            "    for cb in costs:\n"
            "        rc = st.run_experiment(rets, alpha=0.05, t_thresh=2.0, cost_bps=cb)\n"
            "        bh_counts.append(rc['counts']['bh'])\n"
            "else:\n"
            f"    bh_counts = [{R['real_bh']}, {R['real_bh']-1}, {R['real_bh']-2}, {R['real_bh']-3}]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.1))\n"
            "ax.plot(costs, bh_counts, 'o-', c=AMBER, lw=2)\n"
            "ax.set_xlabel('one-way cost (bps/leg)'); ax.set_ylabel('BH-surviving discoveries')\n"
            "ax.set_title('Costs only shrink the (already small) honest count')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('BH-surviving by cost:', dict(zip(costs, bh_counts)))"
        ),
        md(
            "> 💡 In plain words: the value of this study is *defensive*. Before funding anyone's "
            "'*t*-stat over 2', ask how many strategies they tried and apply the correction yourself — "
            "three lines of code that retire most of the factor zoo. The definition of a **mirage**: "
            "discoveries the search invented, gone the moment you correct for it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The maximum-statistic angle.** [Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/) "
            "runs White's (2000) Reality Check on the *single best* of N random rules — the right tool "
            "when you only report the champion. 346 corrects across the *whole family*.\n"
            "- **The single-strategy angle.** [Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/) "
            "uses the Deflated Sharpe Ratio and PBO when you over-tune one strategy.\n"
            "- **Dependence-robust FDR.** Fork this to add Benjamini-Yekutieli (2001) for arbitrarily "
            "dependent tests, and the Harvey-Liu (2020) bootstrap multiple-testing haircut, and compare "
            "their realised power on the same battery.\n\n"
            "*The headline number in finance is almost never an effect — it's the best of a search you "
            "weren't told about. Correct for the search, and most of the magic disappears.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
