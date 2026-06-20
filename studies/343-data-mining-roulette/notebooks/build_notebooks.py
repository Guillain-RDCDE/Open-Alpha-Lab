"""Generate the two narrative notebooks for Study 343 (Data-Mining-Roulette).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
roulette runs anywhere, offline and deterministic; the real-tape cells read the cached SPY
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs end-to-end for any reader. Every code
cell carries a banner saying which tape it is showing — synthetic cells never wear a
real-tape banner.
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


# Frozen real-tape + synthetic headline numbers — mirror of docs/results.md (as-of 2026-05-28).
R = dict(
    # real SPY
    real_window="1995-03-14 → 2026-05-28", real_days=7854, real_fp="3ad587d3997f",
    real_best_feat="ma_gap_10 < q0.78", real_best_sharpe=0.75, real_best_t=4.86,
    real_best_mean=13.0, real_npass=765, real_pass_bonf=156, real_rc_p=0.001,
    real_bh_sharpe=0.65,
    real_cost_bps=[0, 5, 10, 20], real_cost_sharpe=[0.75, 0.65, 0.55, 0.35],
    real_cost_t=[4.86, 4.25, 3.64, 2.38],
    real_is_sharpe=0.78, real_is_t=3.65, real_oos_sharpe=0.64, real_oos_t=2.99,
    # synthetic null (seed 343, vol_cluster=0)
    null_best_sharpe=0.66, null_best_t=2.20, null_best_p=0.028,
    null_npass=25, null_bonf_t=4.06, null_pass_bonf=0, null_rc_p=0.30, null_exp_fp=45.5,
    # synthetic null across 10 seeds
    null_med_sharpe=0.89, null_med_t=2.68, null_med_npass=44, null_med_rc_p=0.17,
    null_min_t=1.35, null_max_t=5.07,
    # synthetic positive control (planted_edge=0.10)
    ctrl_best_feat="f0_signal", ctrl_best_sharpe=1.42, ctrl_best_t=4.67,
    ctrl_pass_bonf=31, ctrl_rc_p=0.001,
    n_rules=1000,
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

from data_mining_roulette import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE, "SPY"))

HAVE_REAL = _have_cache()

def real_tape():
    return data.load_real(fetch=False)

print("real SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Data-Mining-Roulette — how many random rules beat the market by luck? 🎰\n"
            "### Spin a thousand random trading rules on pure noise — and watch the luckiest look like genius\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_luck_mimic_skill%3F: Confirmed](https://img.shields.io/badge/Does_luck_mimic_skill%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Here is the most dangerous number in all of finance: the backtest score of the rule you "
            "decided to *keep*. This notebook builds a little p-hacking machine — it invents **1,000 "
            "random trading rules**, tests every one, and shows you the best. The twist: we run it on a "
            "tape we *built to contain no signal whatsoever*. If the best of 1,000 coin-flips still "
            "looks brilliant, then a brilliant backtest tells you nothing — until you ask the one "
            "question almost nobody asks.\n\n"
            "> 📓 **This is the plain-language layer.** Want the HAC *t*-stats, the Bonferroni bar and "
            "White's Reality Check? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it, on a deterministic synthetic tape (the real-SPY numbers are frozen in "
            "[docs/results.md](../docs/results.md)). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I test 1,000 random rules on *pure noise*, how good is the best one? | **It looks "
            f"great.** Best Sharpe ~**{R['null_best_sharpe']}**, *t*-stat ~**{R['null_best_t']}** — a "
            "naive 'significant'. The *median* best-of-1,000 across seeds has a *t* of "
            f"**{R['null_med_t']}**. |\n"
            "| How many of the 1,000 'beat the market' by chance? | **Dozens.** About "
            f"**{R['null_npass']}** clear the famous *t* > 2 bar on a tape with *nothing in it* (it "
            "swings ~11–500 by luck). |\n"
            "| So is a *t* > 2 backtest meaningless? | **On its own, yes.** The fix is to correct for "
            f"how many rules you tried. Do that and **{R['null_pass_bonf']} of {R['n_rules']}** survive — "
            "the mirage vanishes. |\n"
            "| Does the machine actually work, or is it just blind? | **It works.** Plant a *real* edge "
            "and it finds it instantly and ranks it #1. The 'nothing' result is real, not a broken "
            "tool. |\n\n"
            "> The luckiest of a thousand random rules will always look like skill. The only defence is "
            "to remember how many times you spun the wheel."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Open any trading forum and you'll find a backtest with a Sharpe of 2 and a smooth equity "
            "curve, presented as proof. The strong version of the warning — from serious researchers — "
            "is darker than 'be careful':\n\n"
            "> *With enough trials, a backtest Sharpe of 2 is **easy** to achieve on data that is pure "
            "noise.* — Bailey, Borwein, López de Prado & Zhu (2014)\n\n"
            "In 1999, Sullivan, Timmermann & White ran **~7,800** technical trading rules on the Dow. "
            "The best looked wonderful. Under a test that accounts for the *search*, the magic "
            "evaporated. The claim we're testing is the uncomfortable one: **a great backtest is the "
            "expected outcome of trying many rules, even when none of them work.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the best of many random rules looks real, then almost every 'discovered' strategy you "
            "see — yours included — is suspect. You stare at a chart, try a moving-average rule, it's "
            "mediocre; you try RSI, better; you tweak the threshold, *great*. You didn't find an edge — "
            "you ran a search and kept the luckiest result. The money you'd lose trading it is the "
            "*tuition* for not asking how many rules you tried. This is the single most common way "
            "retail and professional quants alike set fire to capital."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We build the p-hacking machine and aim it at a tape where we **know the truth**:\n\n"
            "1. **The null tape.** A fake market that is pure random walk; every indicator we compute "
            "is noise, *uncorrelated with tomorrow*. There is, by construction, **no rule that beats "
            "the market except by luck**.\n"
            "2. **1,000 random rules.** Each picks an indicator, a direction, and a threshold, then "
            "goes long tomorrow when the condition holds (else sits in cash). The roulette wheel.\n"
            "3. **The honest question.** Not 'is the best rule good?' (it always is) but 'how good "
            "*should* the best of 1,000 look, when nothing is real?' — and does correcting for the "
            "search erase it?\n"
            "4. **The positive control.** Re-run on a tape with a *planted* real edge. If the machine "
            "can't find a signal that's genuinely there, finding nothing on the null would prove "
            "nothing.\n\n"
            "If the corrected test says 'this is just luck' on the null **and** 'this is real' on the "
            "planted-edge tape, the machine is trustworthy — and the lesson lands."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: spin the wheel on pure noise.** A tape with no signal at all. Here is the "
            "Sharpe ratio of all 1,000 random rules — and where the single best one lands."
        ),
        code(
            "# SYNTHETIC NULL TAPE — built to contain NO signal (planted_edge=0).\n"
            "feats, fwd, truth = data.synthetic_tape(planted_edge=0.0, vol_cluster=0.0, seed=343)\n"
            "res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=400, seed=343)\n"
            "tbl = res['table']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(tbl['sharpe'].dropna(), bins=40, color=GREY, alpha=.85)\n"
            "best = tbl['sharpe'].iloc[0]\n"
            "ax.axvline(best, c=RED, lw=2.5, label=f'luckiest rule (Sharpe {best:.2f})')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('annualised Sharpe of a random rule'); ax.set_ylabel('number of rules')\n"
            "ax.set_title('1,000 random rules on PURE NOISE — and the luckiest one')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic null (nothing to find)')\n"
            "print(f\"best rule: {tbl['feature'].iloc[0]} {tbl['op'].iloc[0]} q{tbl['q'].iloc[0]}  \"\n"
            "      f\"Sharpe={best:.2f}  HAC t={tbl['hac_t'].iloc[0]:.2f}\")"
        ),
        md(
            f"On a tape with **nothing in it**, the luckiest of 1,000 rules posts a Sharpe of "
            f"~**{R['null_best_sharpe']}** and a *t*-stat of ~**{R['null_best_t']}** — the textbook "
            "'statistically significant' threshold. It is a slot machine that always pays out a winner; "
            "the winner is just whoever sat down at the luckiest seat."
        ),
        md(
            "**How many rules 'beat the market' on this empty tape?** Count the ones clearing the famous "
            "*t* > 2 bar — and compare to the bar you *should* use after trying 1,000."
        ),
        code(
            "# Still the SYNTHETIC NULL tape.\n"
            "pc = res['pass_counts']\n"
            "naive = pc['n_pass_naive']; bonf = pc['n_pass_bonferroni']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['clear t>2\\n(naive)', 'survive the\\ncorrection'], [naive, bonf],\n"
            "       color=[RED, GREEN], width=.55)\n"
            "ax.set_ylabel('number of rules (of 1,000)')\n"
            "ax.set_title('Dozens of \\'winners\\' on noise — none survive correcting for the search')\n"
            "for i, v in enumerate([naive, bonf]):\n"
            "    ax.annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic null')\n"
            "print(f'rules clearing t>2 by chance: {naive} of 1000  |  surviving Bonferroni: {bonf}')"
        ),
        md(
            f"About **{R['null_npass']}** random rules 'beat the market' at *t* > 2 — on a tape where "
            f"that is impossible by design. Once you demand the rule clear the bar appropriate for "
            f"having tried 1,000 of them (Bonferroni), **{R['null_pass_bonf']}** survive. The dozens of "
            "winners were the wheel paying out, nothing more."
        ),
        md(
            "**Is the machine just blind to everything?** No — here's the proof. Plant a *real* edge in "
            "the tape and spin again. A trustworthy machine should find it and put it on top."
        ),
        code(
            "# SYNTHETIC POSITIVE CONTROL — a real edge planted on the 'f0_signal' feature.\n"
            "fc, rc_, tc = data.synthetic_tape(planted_edge=0.10, vol_cluster=0.0, seed=343)\n"
            "res_c = st.run_experiment(fc, rc_, n_rules=1000, rc_boot=400, seed=343)\n"
            "top = res_c['table'].head(8)\n"
            "colors = [GREEN if f == tc['signal_feature'] else GREY for f in top['feature']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(range(len(top)), top['hac_t'], color=colors, width=.7)\n"
            "ax.axhline(2, ls=':', c=GREY); ax.set_xticks(range(len(top)))\n"
            "ax.set_xticklabels([f for f in top['feature']], rotation=35, ha='right', fontsize=8)\n"
            "ax.set_ylabel('HAC t-stat'); ax.set_title('Plant a real edge -> the machine ranks it #1 (green)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic positive control (planted edge on f0_signal)')\n"
            "print(f\"top rule feature: {res_c['table']['feature'].iloc[0]}  \"\n"
            "      f\"(planted signal = {tc['signal_feature']})\")"
        ),
        md(
            f"The machine instantly puts the **planted-edge feature** ({R['ctrl_best_feat']}) at #1, "
            f"HAC *t* ~{R['ctrl_best_t']}, and {R['ctrl_pass_bonf']} of its rules survive the "
            "correction. So when it says 'nothing' on the null, that means *nothing* — not a broken "
            "tool. The slot machine can tell a jackpot from a coincidence; the naive *t*-stat cannot."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No rule mined from a 1,000-rule search is trustworthy on its own. "
            "The 'winners' are the maximum of a thousand noisy numbers, not a thousand discoveries.\n"
            "- **Tradability — Mirage.** The search creates no edge to trade — only the illusion of one, "
            "which dies the moment you account for the search (or pay costs, or step out of sample).\n"
            "- **Does luck mimic skill? — Confirmed.** On a provably-empty tape the best rule looks "
            f"significant ({R['null_best_t']} *t*); correcting for the search ({R['null_pass_bonf']} "
            "survivors, Reality-Check *p* = "
            f"{R['null_rc_p']}) reveals pure luck. The machine proves the point — and proves itself by "
            "nailing a planted edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There is nothing to trade — that's the whole point. But it's worse than 'nothing': if you "
            "*did* deploy the luckiest backtest, you'd be trading a rule chosen precisely *because* it "
            "got lucky, so its future is, if anything, below average (the winner's curse). And the more "
            "rules you search, the more confident — and more wrong — you become. On the real tape we "
            "even found a rule that survives the correction (a faint, real mean-reversion effect), but "
            "it's a known, low-capacity micro-effect that fades with costs and out of sample — *not* "
            "something the roulette invented. The honest takeaway: a backtest is a hypothesis, not a "
            "result, until it has survived data it was never mined on."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Turn up the search.** Re-run with `n_rules=5000` or add more features — watch the best "
            "naive *t* climb while the corrected verdict stays 'nothing'. The mirage grows with effort.\n"
            "- **The single-rule cousin.** [Study 301 — Triple-RSI](../../301-triple-rsi/) dismantles "
            "one viral '90% win-rate' rule; this study is the *factory* that mints such rules from "
            "noise.\n"
            "- **The other randomness.** [Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/) "
            "randomises *which stocks* you hold; here we randomise the *rule*. Together: random holdings "
            "vs random logic.\n\n"
            "*Got a backtest you love? Fork this, add your rule's feature to the roulette, and see how "
            "many random rules beat it by luck. If lots do, you didn't find an edge — you found a seat "
            "at the wheel.*"
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
            "# Data-Mining-Roulette — a quantitative teardown 🔬\n"
            "### Best-of-N null distribution · HAC *t* vs Bonferroni vs White's Reality Check · "
            "planted-edge positive control · cost & out-of-sample decay\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_luck_mimic_skill%3F: Confirmed](https://img.shields.io/badge/Does_luck_mimic_skill%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We characterise the distribution "
            "of the **maximum** of N noisy backtest statistics, and show that the only honest inference "
            "tests that maximum (Bonferroni + White 2000), not each rule's own *t*.\n\n"
            "> ⚠️ **Not investment advice.** The offline core and tests run on a deterministic synthetic "
            "tape; the real-SPY headline (1995–2026, as-of 2026-05-28, fingerprint "
            f"`{R['real_fp']}`) is frozen in [`docs/results.md`](../docs/results.md). Methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Null tape: best-of-1,000 HAC *t* = {R['null_best_t']} (naive "
            f"*p* = {R['null_best_p']}), {R['null_npass']} rules at \\|*t*\\|≥2 — all artefacts of the "
            "search. No certifiable rule. |\n"
            f"| **Tradability** | `MIRAGE` | Real-tape champion Sharpe {R['real_cost_sharpe'][0]} → "
            f"{R['real_cost_sharpe'][-1]} across 0→20 bps; IS *t* {R['real_is_t']} → OOS "
            f"{R['real_oos_t']}. The search manufactures no harvestable edge. |\n"
            f"| **Does luck mimic skill?** | `CONFIRMED` | Bonferroni survivors on the null = "
            f"{R['null_pass_bonf']}; White Reality-Check *p* = {R['null_rc_p']}; positive control "
            f"recovers the planted edge (*t* {R['ctrl_best_t']}, RC *p* {R['ctrl_rc_p']}). |\n\n"
            "> 💡 In plain words: the distribution of the *best* of N random rules is not the "
            "distribution of *a* rule — its mean and its tail both grow with N. Quoting the best rule's "
            "own *t*-stat ignores that, which is exactly how noise gets published as signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\{r^{(i)}_t\\}_{i=1}^N$ be the excess-return series of $N$ candidate rules, each with "
            "true mean $\\mu_i$. The data-snooping null is $H_0: \\max_i \\mu_i \\le 0$.\n\n"
            "- **H₁ (the naive trap).** A practitioner reports $t_{(1)} = \\max_i t_i$ and compares it "
            "to 1.96. Under $H_0$ with $N$ large and rules even mildly independent, "
            "$\\mathbb{E}[\\max_i t_i] \\gg 2$ — so H₁ 'confirms' a signal that isn't there.\n"
            "- **H₂ (the correct test).** Bonferroni requires $\\max_i t_i \\ge \\Phi^{-1}(1-\\alpha/2N)$; "
            "White's (2000) Reality Check bootstraps the distribution of $\\max_i \\bar r^{(i)}$ under "
            "the null and reads off $p = \\Pr(\\max_i \\bar r^{*(i)} \\ge \\max_i \\bar r^{(i)})$.\n"
            "- **H₃ (the positive control).** With a genuine $\\mu_j > 0$ planted on feature $j$, the "
            "corrected test must *reject* $H_0$ and rank rules on $j$ at the top.\n\n"
            "We find **H₁ fires on a provably-null tape** (best *t* "
            f"{R['null_best_t']}), **H₂ correctly clears it** ({R['null_pass_bonf']} Bonferroni "
            f"survivors, RC *p* {R['null_rc_p']}), and **H₃ holds** (planted edge recovered)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Harvey, Liu & Zhu (2016) argue that, given the hundreds of published factors, the honest "
            "significance hurdle for a 'new' anomaly is a *t* near **3**, not 2 — precisely because the "
            "literature is the maximum of an enormous, hidden search. This study makes that abstract "
            "warning tactile: we *are* the hidden search, we *know* the null, and we can read the gap "
            "between the naive bar and the corrected bar directly. The stake is every backtest-driven "
            "allocation decision ever made — the difference between a discovery and a coincidence is the "
            "multiple-testing correction, and it is routinely omitted."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tapes.** A deterministic synthetic generator with one knob, `planted_edge`: at 0 the "
            "price is a random walk and all 12 features are noise (clean null); at >0 one designated "
            "feature genuinely predicts $r_{t+1}$ (positive control). Plus real SPY total-return daily, "
            "1995–2026, the same feature builder.\n"
            "- **Rules.** $N=1{,}000$ random `(feature, op, quantile-threshold)` triples; long the next "
            "day when on, else cash. One execution lag — encoded once in `fwd_ret` (signal at close of "
            "$t$ earns $r_{t+1}$) and never shifted again. Excess-of-cash, so cash-heavy rules race the "
            "fully-invested benchmark like-for-like.\n"
            "- **Per-rule inference.** Newey-West HAC *t*; circular block-bootstrap CI on the mean.\n"
            "- **Family inference.** Bonferroni threshold $\\Phi^{-1}(1-0.05/2N)$; White's Reality Check "
            "*p* on the maximum, via the **stationary bootstrap** (Politis-Romano, geometric blocks).\n"
            "- **Costs.** One-way turnover × NAV at `cost_bps` (each flip in/out is a trade).\n"
            "- **What would make us say 'mirage' (well, *signal=none*):** if correcting for the search "
            "leaves the null's champion significant, the method would be vindicated and the study would "
            "fail. It doesn't — the correction collapses the null champion every time."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The distribution of the *maximum* — naive *t* of best-of-N on the null\n\n"
            "Re-spin the null roulette across many seeds and collect the best rule's HAC *t* each time. "
            "The point is the whole distribution sits *above* 2."
        ),
        code(
            "# SYNTHETIC NULL across seeds — distribution of best-of-1000 HAC t (nothing to find).\n"
            "best_t = []\n"
            "for s in range(343, 363):\n"
            "    f, r, _ = data.synthetic_tape(planted_edge=0.0, vol_cluster=0.0, seed=s)\n"
            "    tbl = st.spin_roulette(f, r, st.random_rules(list(f.columns), n_rules=1000, seed=s))\n"
            "    best_t.append(tbl['hac_t'].iloc[0])\n"
            "best_t = np.array(best_t)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(best_t, bins=12, color=GREY, alpha=.85)\n"
            "ax.axvline(2, ls=':', c=RED, lw=2, label='naive |t|=2 bar')\n"
            "ax.axvline(np.median(best_t), c=GREEN, lw=2, label=f'median best-of-1000 (t={np.median(best_t):.2f})')\n"
            "ax.set_xlabel('HAC t of the BEST of 1,000 random rules'); ax.set_ylabel('count (20 seeds)')\n"
            "ax.set_title('On pure noise, the luckiest rule routinely clears t=2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic null, 20 seeds')\n"
            "print(f'median best-of-1000 HAC t = {np.median(best_t):.2f}  (min {best_t.min():.2f}, max {best_t.max():.2f})')"
        ),
        md(
            f"> 💡 In plain words: the **median** best-of-1,000 rule on a no-signal tape has HAC *t* = "
            f"~{R['null_med_t']} (range ~{R['null_min_t']}–{R['null_max_t']}). The naive bar isn't just "
            "occasionally crossed by luck — it is crossed *in the median*. The maximum statistic has a "
            "completely different distribution from a single statistic, and that gap is the whole "
            "problem."
        ),
        md(
            "### 4b · The three bars — naive *t*, Bonferroni, and the Reality Check\n\n"
            "On one null spin: how many rules clear *t* > 2, how high the family-wise bar actually sits, "
            "and what White's Reality Check says about the champion."
        ),
        code(
            "# SYNTHETIC NULL, single canonical spin (seed 343).\n"
            "feats, fwd, _ = data.synthetic_tape(planted_edge=0.0, vol_cluster=0.0, seed=343)\n"
            "res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=1000, seed=343)\n"
            "pc, rc = res['pass_counts'], res['rc']\n"
            "tbl = res['table']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "axes[0].hist(tbl['hac_t'].dropna(), bins=40, color=GREY, alpha=.8)\n"
            "axes[0].axvline(2, ls=':', c=RED, label='naive |t|=2')\n"
            "axes[0].axvline(pc['bonferroni_t'], ls='--', c=GREEN, label=f\"Bonferroni |t|={pc['bonferroni_t']:.2f}\")\n"
            "axes[0].axvline(-2, ls=':', c=RED); axes[0].axvline(-pc['bonferroni_t'], ls='--', c=GREEN)\n"
            "axes[0].set_xlabel('HAC t'); axes[0].set_ylabel('rules'); axes[0].legend(fontsize=8)\n"
            "axes[0].set_title('All 1,000 rule t-stats vs the two bars')\n"
            "axes[1].bar(['clear t>2','survive\\nBonferroni'], [pc['n_pass_naive'], pc['n_pass_bonferroni']],\n"
            "            color=[RED, GREEN], width=.5)\n"
            "axes[1].set_ylabel('rules (of 1,000)'); axes[1].set_title('Winners before and after the correction')\n"
            "for i, v in enumerate([pc['n_pass_naive'], pc['n_pass_bonferroni']]):\n"
            "    axes[1].annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic null')\n"
            "print(f\"naive winners {pc['n_pass_naive']}  |  Bonferroni |t| bar {pc['bonferroni_t']:.2f}  |  \"\n"
            "      f\"Bonferroni survivors {pc['n_pass_bonferroni']}  |  Reality-Check p {rc['reality_check_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the naive bar nets **{R['null_npass']}** 'winners'; the family-wise "
            f"bar (|*t*| ≥ {R['null_bonf_t']} for N=1,000) nets **{R['null_pass_bonf']}**. White's "
            f"Reality Check — the proper test of the maximum — returns *p* = **{R['null_rc_p']}**: the "
            "champion is exactly what a search of this size produces under the null. Note the Reality "
            "Check has its own 5% false-positive rate, so across seeds an occasional spin reads *p* < "
            "0.05 — that *is* the test, calibrated correctly, not a real edge."
        ),
        md(
            "### 4c · The positive control — the engine is a faithful detector\n\n"
            "Sweep the planted edge from 0 upward. The best rule's corrected significance should switch "
            "on, and the top-ranked feature should become the signal feature — proof the null result "
            "isn't just a dead pipeline."
        ),
        code(
            "# SYNTHETIC POSITIVE CONTROL sweep — edge planted on f0_signal.\n"
            "edges = [0.0, 0.03, 0.06, 0.10]\n"
            "rc_ps, top_is_signal = [], []\n"
            "for e in edges:\n"
            "    f, r, tr = data.synthetic_tape(planted_edge=e, vol_cluster=0.0, seed=343)\n"
            "    rr = st.run_experiment(f, r, n_rules=1000, rc_boot=400, seed=343)\n"
            "    rc_ps.append(rr['rc']['reality_check_p'])\n"
            "    top_is_signal.append(rr['table']['feature'].iloc[0] == tr['signal_feature'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if s else GREY for s in top_is_signal]\n"
            "ax.bar([f'{e:.2f}' for e in edges], rc_ps, color=cols, width=.6)\n"
            "ax.axhline(0.05, ls=':', c=RED, label='p=0.05')\n"
            "ax.set_xlabel('planted edge strength'); ax.set_ylabel('Reality-Check p (best rule)')\n"
            "ax.set_title('Plant a real edge -> Reality Check fires; green = signal feature ranked #1')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic positive control sweep')\n"
            "print('Reality-Check p by planted edge:', [f'{p:.3f}' for p in rc_ps])\n"
            "print('top-ranked feature is the signal feature:', top_is_signal)"
        ),
        md(
            f"> 💡 In plain words: at zero edge the Reality Check sits high (no signal); as a real edge "
            f"is planted, its *p* drops below 0.05 and the **signal feature** climbs to #1 (green bars). "
            "The machine detects what is genuinely there and dismisses what isn't — so 'signal=none' on "
            "the null is a finding, not a failure."
        ),
        md(
            "### 4d · The real tape — a faint *real* effect, drowned in snooped noise\n\n"
            "Real SPY is *not* a clean null (short-horizon mean reversion is real). The Reality Check "
            "correctly fires — but watch how the naive winner-count overstates the number of real "
            "signals, and how the champion decays with costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rfeats, rfwd = real_tape()\n"
            "    rr = st.run_experiment(rfeats, rfwd, n_rules=1000, rc_boot=500, seed=343)\n"
            "    naive = rr['pass_counts']['n_pass_naive']; bonf = rr['pass_counts']['n_pass_bonferroni']\n"
            "    rcp = rr['rc']['reality_check_p']\n"
            "    champ = rr['table'].iloc[0].to_dict()\n"
            "    rule = {'feature':champ['feature'],'op':champ['op'],'q':champ['q'],'long_short':False}\n"
            "    costs = [0.0, 5.0, 10.0, 20.0]\n"
            "    sh = [st.sharpe(st.backtest_rule(rfeats, rfwd, rule, cost_bps=c)) for c in costs]\n"
            "    banner = 'TAPE: REAL SPY total-return 1995-2026'\n"
            "else:\n"
            f"    naive, bonf, rcp = {R['real_npass']}, {R['real_pass_bonf']}, {R['real_rc_p']}\n"
            f"    costs, sh = {R['real_cost_bps']}, {R['real_cost_sharpe']}\n"
            "    banner = 'TAPE: OFFLINE — frozen real-SPY numbers from docs/results.md'\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "axes[0].bar(['naive\\nt>2','survive\\nBonferroni'], [naive, bonf], color=[RED, AMBER], width=.5)\n"
            "axes[0].set_ylabel('rules (of 1,000)'); axes[0].set_title(f'Real SPY: {naive} naive \\'winners\\'')\n"
            "for i, v in enumerate([naive, bonf]):\n"
            "    axes[0].annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "axes[1].plot(costs, sh, 'o-', c=RED, lw=2)\n"
            "axes[1].set_xlabel('round-trip cost (bps/leg)'); axes[1].set_ylabel('best-rule Sharpe')\n"
            "axes[1].set_title('The snooped champion erodes with costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(banner)\n"
            "print(f'naive winners {naive} / Bonferroni survivors {bonf} / Reality-Check p {rcp:.3f}')\n"
            "print('best-rule Sharpe by cost:', [f'{x:.2f}' for x in sh])"
        ),
        md(
            f"> 💡 In plain words: on real SPY **{R['real_npass']} of {R['n_rules']}** rules show "
            f"*t* > 2 — but they are mostly correlated reflections of one faint mean-reversion effect "
            "plus luck, not hundreds of independent edges. The champion (a mean-reversion rule) even "
            f"survives the Reality Check (*p* {R['real_rc_p']}), yet it is a known, low-capacity "
            f"micro-effect that fades with costs (Sharpe {R['real_cost_sharpe'][0]} → "
            f"{R['real_cost_sharpe'][-1]} at 20 bps) and out of sample (IS *t* {R['real_is_t']} → OOS "
            f"{R['real_oos_t']}). The roulette didn't *create* it — and it can't certify it either."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — best-of-1,000 on the null: Sharpe {R['null_best_sharpe']}, HAC *t* "
            f"{R['null_best_t']}, naive *p* {R['null_best_p']}; median across seeds *t* {R['null_med_t']}. "
            "All of it the distribution of a maximum, none of it a certifiable rule.\n"
            f"- **Tradability `MIRAGE`** — the real-tape champion is a snooped pick of a known micro-"
            f"effect; Sharpe {R['real_cost_sharpe'][0]} → {R['real_cost_sharpe'][-1]} across costs, IS "
            f"*t* {R['real_is_t']} → OOS {R['real_oos_t']}. No edge the search itself created.\n"
            f"- **Does luck mimic skill? `CONFIRMED`** — Bonferroni survivors on the null = "
            f"{R['null_pass_bonf']}, White Reality-Check *p* = {R['null_rc_p']}, and the positive control "
            f"recovers the planted edge (*t* {R['ctrl_best_t']}, RC *p* {R['ctrl_rc_p']}). The naive "
            "*t*-stat is uninformative about a searched-for rule; the corrected test is the only honest "
            "arbiter."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the winner's curse and the deflated Sharpe\n\n"
            "Selecting the best of N backtests doesn't just *fail* to find alpha — it actively "
            "anti-selects: the rule you keep is the one whose noise was kindest in-sample, so its "
            "expected future performance is *below* the family average (the winner's curse). Bailey & "
            "López de Prado's **deflated Sharpe ratio** formalises this: divide the observed Sharpe by "
            "the Sharpe you'd *expect* from the best of N under the null. Here is that haircut as N grows."
        ),
        code(
            "# SYNTHETIC NULL — expected best-of-N Sharpe (the bar the deflated Sharpe divides by).\n"
            "Ns = [10, 50, 200, 1000, 3000]\n"
            "exp_best = []\n"
            "for N in Ns:\n"
            "    f, r, _ = data.synthetic_tape(planted_edge=0.0, vol_cluster=0.0, seed=343)\n"
            "    tbl = st.spin_roulette(f, r, st.random_rules(list(f.columns), n_rules=N, seed=343))\n"
            "    exp_best.append(tbl['sharpe'].iloc[0])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "ax.plot(Ns, exp_best, 'o-', c=RED, lw=2)\n"
            "ax.set_xscale('log'); ax.set_xlabel('number of rules searched (N)')\n"
            "ax.set_ylabel('best-of-N Sharpe on PURE NOISE')\n"
            "ax.set_title('The more you search, the better noise looks — the bar a real edge must clear')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TAPE: synthetic null')\n"
            "print('expected best-of-N Sharpe (noise) by N:', dict(zip(Ns, [round(x,2) for x in exp_best])))"
        ),
        md(
            "> 💡 In plain words: the more rules you try, the higher the Sharpe pure noise will hand "
            "you — so a Sharpe of 1 means something very different after 10 rules than after 3,000. The "
            "deflated Sharpe is just 'your Sharpe minus the Sharpe noise would have given the best of N'. "
            "If you can't say how many rules you tried, you can't compute it — and you can't trust the "
            "number."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Hansen's SPA.** Replace White's Reality Check with the Superior Predictive Ability "
            "test (Hansen 2005), which down-weights poor alternatives for more power. Fork "
            "`strategy.reality_check`.\n"
            "- **Correlated families.** Our rules share features, so they are correlated — which is "
            "*more* realistic than independent tests and makes Bonferroni conservative. Swap in Holm or "
            "a bootstrap-FWER step-down to recover power without losing honesty.\n"
            "- **Real-data factor zoo.** Point the roulette at a cross-section instead of one ETF and "
            "watch the Harvey-Liu-Zhu *t*≈3 hurdle emerge naturally from the maximum distribution.\n\n"
            "*The headline number of any backtest is meaningless without its denominator: how many rules "
            "were tried to find it. Fork this, feed in your own strategy, and let the wheel tell you "
            "whether you found a signal or a seat.*"
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
