"""Generate the two narrative notebooks for Study 839 (The t > 3 Threshold).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
synthetic-only method demo: every figure runs offline and deterministic on the seed-839
factor zoo — there is NO real-tape cell (real free factor data can never certify "zero
edge"), so the study is capped at NONE. The dict ``R`` below mirrors the headline numbers
in docs/results.md; the notebooks recompute them live from the engine so they always agree.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; null
# zoo fp 2a151cf27292; mixture fp 7ade07aad75d; 1,000 factors x 240 periods; seed 839).
# Recomputed live in the notebooks.
R = dict(
    fp_null="2a151cf27292", fp_mix="7ade07aad75d",
    sim_fp="839:N1000:T240:true50:et4.0:a0.05",
    n_factors=1000, n_periods=240, n_true=50, exp_t=4.0, hlz_n=316, alpha=0.05,
    # pure-null single zoo
    null_n_gt2=44, null_frac_gt2=4.40, null_n_gt3=3, null_frac_gt3=0.30, null_max_t=3.23,
    theory_frac_gt2=4.55, theory_frac_gt3=0.27, ratio=16.9,
    # Bonferroni cutoff by N
    bonf=[(1, 1.96), (10, 2.81), (100, 3.48), (316, 3.78), (1000, 4.06)],
    bonf_hlz=3.78,
    # mixture detection
    mix2=dict(n=93, tp=50, fp=43, fdr=46.2, power=100.0),
    mix3=dict(n=44, tp=41, fp=3, fdr=6.8, power=82.0),
    # correction table on mixture: (method, cutoff, n)
    corr_mix=[("naive t>2", 2.00, 93), ("naive t>3", 3.00, 44), ("Bonferroni", 4.06, 19),
              ("Holm", 4.28, 19), ("BH", 3.18, 43), ("BHY", 3.73, 31)],
    # publication haircut: (reported_t, p_adj, eff_t, haircut%, survives)
    haircut=[(2.0, 1.000, 0.00, 100, False), (2.5, 1.000, 0.00, 100, False),
             (3.0, 0.853, 0.19, 94, False), (3.5, 0.147, 1.45, 59, False),
             (4.0, 0.020, 2.33, 42, True)],
    # seed-robust null (20 seeds)
    sn_frac_gt2=4.62, sn_frac_gt3=0.31, sn_n_gt2=46.2, sn_n_gt3=3.1, sn_ratio=14.9,
    sn_max_t=3.42,
    # seed-robust mixture (20 seeds)
    sm_fdr2=47.6, sm_fdr3=6.4, sm_collapse=7.4, sm_pw2=97.2, sm_pw3=85.0,
    sm_bhy_n=30.8, sm_bhy_cut=3.76, sm_bhy_fdr=0.3,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Does t>2 inflate false discoveries?: Confirmed](https://img.shields.io/badge/Does_t%3E2_inflate_false_discoveries%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from tstat_threshold import data, strategy as st

# The whole demo runs on the deterministic, offline seed-839 factor zoo (1,000 candidate
# factors x 240 monthly periods). The NULL is pure noise; the MIXTURE plants 50 true
# factors (expected |t|=4) as a positive control. Synthetic-only by design — no real tape.
R0, _, TR0 = data.synthetic_zoo(n_factors=1000, n_periods=240, n_true=0, seed=839)
RM, IS_TRUE, TRM = data.synthetic_zoo(n_factors=1000, n_periods=240, n_true=50,
                                      expected_t=4.0, seed=839)
T0 = st.factor_tstats(R0)          # per-factor t-stats, pure-noise zoo
TM = st.factor_tstats(RM)          # per-factor t-stats, planted mixture
print("null zoo:", R0.shape, "fp", data.fingerprint(R0), "| has_edge", TR0.has_edge)
print("mixture :", RM.shape, "fp", data.fingerprint(RM), "|", int(IS_TRUE.sum()), "true factors")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The t > 3 Threshold — why a 't-stat above 2' proves almost nothing 🚩\n"
            "### How a factory of pure-noise 'factors' manufactures a paper's worth of discoveries — in plain English\n\n"
            + BADGES +
            "You will read it in almost every quant paper: *\"our strategy has a t-stat above 2, "
            "so the effect is real (only a 5% chance it's a fluke).\"* That sentence is true — "
            "**if you tested exactly one idea**. But the finance profession has tested *hundreds* "
            "of candidate 'factors', reported the winners, and quietly buried the rest. Once you "
            "know that, a t of 2 stops being impressive.\n\n"
            "So we run the experiment on a **factory of fake factors we built to contain nothing**: "
            "1,000 candidate strategies, each one pure coin-flips, zero real edge. Then we ask the "
            "simple question — *how many clear the famous t > 2 bar anyway?* — and watch what happens "
            "when we raise the bar to 3.\n\n"
            "> 📓 **This is the plain-language layer.** Want the Bonferroni / Benjamini-Yekutieli "
            "cutoffs and the publication haircut maths? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it, on a deterministic synthetic factor zoo. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Test 1,000 *pure-noise* factors — how many clear t > 2? | **{R['null_n_gt2']}** "
            f"of 1,000 (**{R['null_frac_gt2']}%**). A whole paper's worth of 'discoveries', all "
            "fake. |\n"
            f"| How many clear t > 3? | Just **{R['null_n_gt3']}** (**{R['null_frac_gt3']}%**) — "
            f"about **{R['ratio']:.0f}× rarer**. |\n"
            f"| So how high should the bar be? | With a few hundred factors tested, the honest bar "
            f"climbs to **~{R['bonf_hlz']}** (Bonferroni). Harvey-Liu-Zhu round it to a memorable "
            "**t ≈ 3.0**. |\n"
            f"| Does raising the bar throw out the real stuff? | Barely. On a test with 50 *genuine* "
            f"factors hidden inside, going t>2 → t>3 cuts the false-discovery rate from "
            f"**{R['mix2']['fdr']:.0f}% → {R['mix3']['fdr']:.0f}%** while keeping "
            f"**{R['mix3']['power']:.0f}%** of the real ones. |\n\n"
            "> A t-stat is only as meaningful as the number of things you tried. Two is the bar for "
            "*one* idea; for a zoo of hundreds, three is the honest floor."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"With hundreds of factors data-mined, the usual t > 2 hurdle produces far too many "
            "false discoveries; the honest, multiple-testing-adjusted bar is about t ≈ 3.0.\"* "
            "— Harvey, Liu & Zhu (2016)\n\n"
            "A **t-stat** measures how many standard errors a strategy's average return sits above "
            "zero. The convention: t above 2 ≈ 'less than 5% chance this is luck'. The catch: that "
            "5% is *per test*. Run 100 tests on pure noise and you *expect* ~5 to clear the bar — "
            "not because they're real, but because 5% of 100 is 5."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because the entire published factor zoo — value, momentum, quality, and hundreds of "
            "lesser-known 'anomalies' — was selected *because* it cleared t > 2. If a big chunk of "
            "that bar is just the 5%-of-noise that had to clear it, then a large fraction of "
            "'established' factors are mirages that will quietly vanish out-of-sample. The desk has "
            "shown you can fake an edge by **searching for a lucky backtest** "
            "([344 Backtest-Overfitting](../../344-backtest-overfitting/)); here the search is "
            "spread across the whole *profession* testing a whole *cross-section* of factors."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We build a world where the answer is certain: **1,000 candidate factors that are pure "
            "noise** (zero-mean coin-flip returns, 240 months each). There is *nothing* real in "
            "here by construction — so every factor that clears a significance bar is, definitionally, "
            "a false discovery. Then we simply count."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — a factory of nothing\n\n"
            "Here is the distribution of t-stats across all 1,000 pure-noise factors. It is a bell "
            "curve centred on zero (as it must be — nothing is real). The shaded tails are the "
            "factors that clear t > 2; the darker slivers clear t > 3."
        ),
        code(
            "t = np.abs(T0)\n"
            "n2 = int((t > 2).sum()); n3 = int((t > 3).sum())\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "bins = np.linspace(-4, 4, 61)\n"
            "ax.hist(T0, bins=bins, color=GREY, alpha=.55, label='1,000 pure-noise factors')\n"
            "for c in (-2, 2):\n"
            "    ax.axvline(c, color=RED, ls='--', lw=1.5)\n"
            "for c in (-3, 3):\n"
            "    ax.axvline(c, color='k', ls=':', lw=1.5)\n"
            "ax.text(2.05, ax.get_ylim()[1]*0.9, 't = 2', color=RED)\n"
            "ax.text(3.05, ax.get_ylim()[1]*0.7, 't = 3', color='k')\n"
            "ax.set_xlabel('single-test t-stat'); ax.set_ylabel('# of factors')\n"
            "ax.set_title(f'A zoo of PURE NOISE: {n2} factors clear |t|>2, only {n3} clear |t|>3')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'cleared |t|>2: {n2}/1000 ({n2/10:.2f}%)   |   cleared |t|>3: {n3}/1000 ({n3/10:.2f}%)')\n"
            "print(f'theory: |t|>2 -> {st.prob_exceed(2)*100:.2f}%   |t|>3 -> {st.prob_exceed(3)*100:.3f}%')"
        ),
        md(
            f"There it is. **{R['null_n_gt2']} of 1,000** noise factors clear the famous t > 2 bar "
            f"— that is an entire research paper of 'significant' findings conjured from coin-flips. "
            f"Only **{R['null_n_gt3']}** clear t > 3. The fractions ({R['null_frac_gt2']}% and "
            f"{R['null_frac_gt3']}%) match the textbook bell-curve tails almost exactly, because "
            "that is all they ever were."
        ),
        md(
            "**So how high should the bar be?** The more factors you test, the higher the bar has to "
            "rise to keep the *expected number of false discoveries* under control. This is the "
            "Bonferroni rule — and it climbs steeply with the size of the search:"
        ),
        code(
            "Ns = [1, 5, 10, 25, 50, 100, 200, 316, 500, 1000]\n"
            "cuts = [st.bonferroni_t(n, 0.05) for n in Ns]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "ax.plot(Ns, cuts, 'o-', color=GREEN, lw=2)\n"
            "ax.axhline(2.0, color=RED, ls='--', lw=1.5, label='the naive t>2 bar')\n"
            "ax.axhline(3.0, color='k', ls=':', lw=1.5, label='the ~3.0 recommendation')\n"
            "ax.axvline(316, color=GREY, ls=':', lw=1)\n"
            "ax.set_xscale('log'); ax.set_xlabel('number of factors tested (log scale)')\n"
            "ax.set_ylabel('honest |t| bar (Bonferroni, 5%)')\n"
            "ax.set_title('The more you test, the higher the honest bar climbs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Bonferroni |t| bar:  ' + '   '.join(f'N={n}:{c:.2f}' for n, c in zip([1,10,100,316,1000], [st.bonferroni_t(n) for n in (1,10,100,316,1000)])))"
        ),
        md(
            f"For a single idea, the bar is the familiar **1.96**. For Harvey-Liu-Zhu's tally of "
            f"**{R['hlz_n']}** tested factors it is **{R['bonf_hlz']}**. The round-number "
            "recommendation of **t ≈ 3.0** sits sensibly between the too-lax 2.0 and the strict 3.8."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The zoo is pure noise by construction; the only 'discoveries' are "
            "the 4.4% of factors that had to clear t>2 by chance. Nothing real to detect.\n"
            "- **Tradability — Mirage.** A factor that exists only because the bar was too low has no "
            "out-of-sample paycheck — that is the whole point of the haircut.\n"
            "- **Does the t>2 bar inflate false discoveries? — Confirmed.** Yes — a whole paper's "
            "worth, from nothing. The honest bar is ~3."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "This is the trap that makes the factor zoo so expensive: you *can* trade a t>2 'factor' "
            "— you just won't get paid. A factor that cleared the bar only because 4.4% of noise had "
            "to will, on average, earn zero going forward (it was never real). Worse, you'll pay real "
            "trading costs to hold it. The honest fix costs nothing: demand a t nearer 3 before you "
            "believe — and *always* ask how many other factors were tried first."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The generic cousin.** [346 Multiple-Testing](../../346-multiple-testing/) is the "
            "same lesson for *any* battery of tests; this study is the factor-zoo specialisation, "
            "framed around the **3.0 hurdle** and the **publication haircut**.\n"
            "- **What happens next.** [536 Anomaly-Decay-Post-Publication](../../536-anomaly-decay-post-publication/) "
            "follows factors *after* they clear the bar — and watches them fade.\n"
            "- **The searching cousin.** [343 Data-Mining-Roulette](../../343-data-mining-roulette/) "
            "mines a single series for a lucky rule.\n\n"
            "*Think your favourite anomaly is real? Fork this, plug in its t-stat and an honest count "
            "of how many factors the literature tested, and read off its publication haircut in the "
            "quants notebook.*"
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
            "# The t > 3 Threshold — a quantitative teardown 🔬\n"
            "### per-factor t-stats · Gaussian-tail clearing rates · Bonferroni / Holm / BH / BHY hurdles · the FDR collapse on a planted mixture · the publication haircut · seed-robust controls\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its arithmetic.* We take a synthetic factor zoo "
            "of 1,000 candidates × 240 monthly periods with **zero real edge**, count what clears "
            "each bar, and derive the honest hurdle four different ways.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the zoo is built to have "
            f"no edge (null fp `{R['fp_null']}`, sim fp `{R['sim_fp']}`), so real free data can "
            "never certify it and the study is capped at `NONE`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pure-noise zoo: {R['null_n_gt2']}/1,000 clear t>2 "
            f"({R['null_frac_gt2']}%), {R['null_n_gt3']} clear t>3 ({R['null_frac_gt3']}%) — the "
            "Gaussian tails 4.55% / 0.27%. Synthetic-only demo (no real tape). |\n"
            f"| **Tradability** | `MIRAGE` | A factor selected only by a too-low bar earns zero "
            "out-of-sample by construction — the haircut is the point. |\n"
            f"| **Does t>2 inflate false discoveries?** | `CONFIRMED` | Bonferroni bar "
            f"{R['bonf'][0][1]} → {R['bonf_hlz']} for N = 1 → {R['hlz_n']}; on a planted mixture the "
            f"realized FDR collapses {R['mix2']['fdr']:.0f}% → {R['mix3']['fdr']:.0f}% going t>2 → "
            "t>3. |\n\n"
            "> 💡 In plain words: the naive bar keeps a paper's worth of noise; the corrected bar "
            "(~3) purges it while keeping genuinely strong factors."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let each candidate factor $i$ have per-period returns with sample mean $\\bar r_i$ and "
            "standard error $\\hat\\sigma_i/\\sqrt{T}$; its single-test statistic is "
            "$t_i = \\bar r_i / (\\hat\\sigma_i/\\sqrt{T})$. Under the null (no edge) $t_i "
            "\\stackrel{\\cdot}{\\sim} N(0,1)$.\n\n"
            "- **H₁ (the lax bar).** $P(|t_i| > 2) = 2\\Phi(-2) = 4.55\\%$ and $P(|t_i| > 3) = "
            "2\\Phi(-3) = 0.27\\%$. Test $N$ noise factors and you *expect* $0.0455N$ to clear t>2.\n"
            "- **H₂ (FWER control).** To keep $P(\\text{any false discovery}) \\le \\alpha$, "
            "Bonferroni tests each at $\\alpha/N$: the cutoff is $\\Phi^{-1}(1-\\alpha/2N)$, rising "
            "with $N$.\n"
            "- **H₃ (FDR control).** Benjamini-Hochberg (and Yekutieli's dependency-robust variant "
            "with $c(N)=\\sum 1/i$) instead bounds the *expected proportion* of false discoveries — "
            "less conservative, the right goal for a factor zoo.\n"
            "- **H₄ (the recommendation).** For a few-hundred-factor zoo all of these land in the "
            "$3.4$–$3.8$ range; HLZ round to **$t \\approx 3.0$**.\n\n"
            "We **confirm H₁** (clearing rates match the tails), **confirm H₂** (the bar rises to "
            "~3.8 at N=316), **confirm H₃** (BHY ~3.7 and it keeps real factors), and **land H₄** "
            "(the FDR collapses ~7× from t>2 to t>3)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "The single-test bar controls the *per-test* false-positive rate. The quantity that "
            "actually matters for a search is the **family-wise error rate** (probability of *any* "
            "false discovery) or the **false-discovery rate** (expected *fraction* of discoveries "
            "that are false). Both grow with the number of tests, so a fixed t>2 bar lets the "
            "expected count of false positives grow linearly in $N$: test 316 factors and you expect "
            f"$0.0455 \\times 316 \\approx 14$ false positives at t>2. This is the same haircut "
            "family as the Deflated Sharpe Ratio in "
            "[344](../../344-backtest-overfitting/) (there: haircut for *trials*; here: for the "
            "*cross-section of factors*), and the generic version in "
            "[346](../../346-multiple-testing/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Null zoo.** 1,000 candidate factors × 240 monthly periods, each iid zero-mean — "
            "*nothing real*, so every clearing is a false discovery.\n"
            "- **Positive-control mixture.** 50 of the 1,000 factors carry a genuine mean sized to an "
            "expected single-test $|t|=4$; the corrections must keep these while purging the 950 "
            "noise factors.\n"
            "- **Statistics.** Vectorised `factor_tstats`; two-sided p-values; the naive t>2 / t>3 "
            "counts; Bonferroni / Holm (FWER) and BH / BHY (FDR) cutoffs expressed as an implied "
            "$|t|$; realized FDR / power against the known truth.\n"
            "- **The haircut.** `publication_haircut(t, N)` — the effective $|t|$ a claimed factor "
            "retains once an $N$-test search is disclosed.\n"
            "- **Seed robustness.** The null fractions and the mixture FDR/power are averaged over "
            "**20 seeds** (house rule)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The clearing rates — H₁\n\n"
            "How many pure-noise factors clear each bar, observed vs the Gaussian-tail expectation."
        ),
        code(
            "ts = st.threshold_summary(T0, thresholds=(2.0, 2.5, 3.0, 3.5), n_factors=1000)\n"
            "print(ts.round(4).to_string())\n"
            "thr = ts.index.values; obs = ts['frac_cleared'].values*100; exp = ts['exp_false_frac'].values*100\n"
            "x = np.arange(len(thr)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.bar(x - w/2, obs, w, color=RED, label='observed (noise zoo)')\n"
            "ax.bar(x + w/2, exp, w, color=GREY, label='Gaussian-tail expectation')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'|t|>{t:g}' for t in thr])\n"
            "ax.set_ylabel('% of factors clearing'); ax.set_yscale('log')\n"
            "ax.set_title('Pure-noise clearing rates match the bell-curve tails exactly')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed.** {R['null_frac_gt2']}% clear t>2 vs a theoretical "
            f"4.55%, and {R['null_frac_gt3']}% clear t>3 vs 0.27% — the zoo is *exactly* as noisy as "
            "theory predicts, because it is nothing but noise."
        ),
        md(
            "### 4b · The corrected hurdles — H₂, H₃, H₄\n\n"
            "Every rule expressed as an implied $|t|$ cutoff and a discovery count, on the pure-null "
            "zoo (which should yield ~0 corrected discoveries) and on the planted mixture (where the "
            "corrections should keep the real subset)."
        ),
        code(
            "print('--- pure-null zoo (nothing real) ---')\n"
            "print(st.multiple_testing_table(T0, alpha=0.05).round(3).to_string())\n"
            "print('\\n--- planted mixture (50 true of 1,000) ---')\n"
            "mt = st.multiple_testing_table(TM, alpha=0.05)\n"
            "print(mt.round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "order = ['naive t>2','naive t>3','BH','BHY','Bonferroni','Holm']\n"
            "cuts = [mt.loc[m,'t_cutoff'] for m in order]\n"
            "cols = [RED, AMBER, GREEN, GREEN, GREY, GREY]\n"
            "ax.bar(order, cuts, color=cols)\n"
            "ax.axhline(3.0, color='k', ls=':', lw=1.5, label='the ~3.0 recommendation')\n"
            "ax.set_ylabel('implied |t| hurdle'); ax.set_title('Every correction lifts the bar to ~3-4')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **H₂/H₃/H₄ confirmed.** On the pure null every correction rejects "
            f"**0** (they refuse to manufacture discoveries). On the mixture the implied hurdles land "
            f"at Bonferroni {R['corr_mix'][2][1]}, Holm {R['corr_mix'][3][1]}, BH "
            f"{R['corr_mix'][4][1]}, BHY {R['corr_mix'][5][1]} — all comfortably above the naive 2.0, "
            "bracketing the ~3.0 recommendation."
        ),
        md(
            "### 4c · The FDR collapse — the quantitative case for ~3.0\n\n"
            "On the planted mixture, count the true and false discoveries at t>2 vs t>3 and read off "
            "the realized false-discovery rate."
        ),
        code(
            "d2 = st.detection(TM, IS_TRUE, 2.0); d3 = st.detection(TM, IS_TRUE, 3.0)\n"
            "for lab, d in [('t>2', d2), ('t>3', d3)]:\n"
            "    print(f\"{lab}: {d['n_disc']} discoveries = {d['tp']} true + {d['fp']} false \"\n"
            "          f\"-> FDR {d['fdr']*100:.1f}%  power {d['power']*100:.1f}%\")\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['t>2','t>3'], [d2['tp'], d3['tp']], color=GREEN, label='true discoveries')\n"
            "a1.bar(['t>2','t>3'], [d2['fp'], d3['fp']], bottom=[d2['tp'], d3['tp']], color=RED, label='FALSE discoveries')\n"
            "a1.set_ylabel('# discoveries'); a1.set_title('t>3 purges the false discoveries'); a1.legend()\n"
            "a2.bar(['t>2','t>3'], [d2['fdr']*100, d3['fdr']*100], color=[RED, AMBER])\n"
            "a2.set_ylabel('realized FDR (%)'); a2.set_title('False-discovery rate collapses')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at t>2, **{R['mix2']['fp']} of {R['mix2']['n']}** discoveries are "
            f"false — an FDR of **{R['mix2']['fdr']:.0f}%**, nearly a coin flip. At t>3 the FDR "
            f"collapses to **{R['mix3']['fdr']:.0f}%** for only a modest power loss "
            f"({R['mix2']['power']:.0f}% → {R['mix3']['power']:.0f}%). That trade — halve-your-"
            "discoveries'-credibility vs keep-almost-all-the-real-ones — is the case for the ~3.0 bar."
        ),
        md(
            "### 4d · The publication haircut\n\n"
            "The haircut peculiar to the factor zoo: a factor is only 'discovered' against the "
            "backdrop of every other factor tried. Disclose the search size and its reported $|t|$ "
            "shrinks to an effective $|t|$."
        ),
        code(
            "rows = [st.publication_haircut(t, 316) for t in (2.0, 2.5, 3.0, 3.5, 4.0)]\n"
            "for h in rows:\n"
            "    print(f\"reported |t|={h['t_reported']:.1f}: p_adj={h['p_adjusted']:.3f} \"\n"
            "          f\"eff |t|={h['t_adjusted']:.2f}  haircut {h['haircut']*100:.0f}%  \"\n"
            "          f\"survives={h['survives_005']}\")\n"
            "tt = np.array([h['t_reported'] for h in rows]); hc = np.array([h['haircut']*100 for h in rows])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.plot(tt, hc, 'o-', color=RED, lw=2)\n"
            "ax.set_xlabel('reported |t| (single-test)'); ax.set_ylabel('publication haircut (%)')\n"
            "ax.set_title('A claimed t=2 factor is haircut to nothing once a 316-test search is disclosed')\n"
            "ax.invert_xaxis(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: reported **|t|=2.0** → adjusted p of **1.00**, a **100% haircut**, "
            f"dead. Even **|t|=3.0** is haircut **94%** and fails α=0.05; only **|t|=4.0** survives "
            f"(haircut still **42%**). The reported significance was mostly an artefact of the search."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pure-noise zoo, {R['null_n_gt2']}/1,000 clear t>2 by chance; "
            "nothing real, and a synthetic-only demo can never earn `REAL`.\n"
            "- **Tradability `MIRAGE`** — a factor selected by a too-low bar has no out-of-sample "
            "return by construction.\n"
            f"- **Does t>2 inflate false discoveries? `CONFIRMED`** — the corrected bar rises to "
            f"~{R['bonf_hlz']} and the FDR collapses {R['mix2']['fdr']:.0f}% → {R['mix3']['fdr']:.0f}% "
            "from t>2 to t>3."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the seed-robust null control\n\n"
            "Is the clearing-rate result a fluke of one lucky seed? Average the pure-null fractions "
            "over 20 independent zoos — they must hug the Gaussian tails (an unbiased machinery "
            "proof), with *nothing* tradable in any of them."
        ),
        code(
            "sn = st.seed_robust_null(data, n_factors=1000, n_periods=240, n_seeds=20)\n"
            "print(f\"mean frac t>2 = {sn['mean_frac_gt2']*100:.2f}% (theory {sn['theory_frac_gt2']*100:.2f}%)\")\n"
            "print(f\"mean frac t>3 = {sn['mean_frac_gt3']*100:.3f}% (theory {sn['theory_frac_gt3']*100:.3f}%)\")\n"
            "print(f\"mean n t>2 = {sn['mean_n_gt2']:.1f} | mean n t>3 = {sn['mean_n_gt3']:.1f} | ratio {sn['ratio_gt2_over_gt3']:.1f}x\")\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['t>2\\nobserved','t>2\\ntheory','t>3\\nobserved','t>3\\ntheory'],\n"
            "       [sn['mean_frac_gt2']*100, sn['theory_frac_gt2']*100, sn['mean_frac_gt3']*100, sn['theory_frac_gt3']*100],\n"
            "       color=[RED, GREY, AMBER, GREY])\n"
            "ax.set_ylabel('% clearing (20-seed mean)'); ax.set_title('The machinery is unbiased: observed ~ theory')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: over 20 seeds, **{R['sn_frac_gt2']}%** clear t>2 (theory 4.55%) and "
            f"**{R['sn_frac_gt3']}%** clear t>3 (theory 0.27%). The estimator is unbiased — and there "
            "is nothing to harvest in a world of noise. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust positive control\n\n"
            "Is the corrected bar just numb (rejecting everything)? Plant 50 genuine factors and check "
            "that BHY *keeps* them while its realized false-discovery rate stays near zero — averaged "
            "over 20 seeds so no lucky seed can fake it."
        ),
        code(
            "sm = st.seed_robust_mixture(data, n_factors=1000, n_true=50, expected_t=4.0, n_seeds=20)\n"
            "print(f\"FDR: t>2 {sm['mean_fdr_t2']*100:.1f}% -> t>3 {sm['mean_fdr_t3']*100:.1f}% (collapse {sm['fdr_collapse']:.1f}x)\")\n"
            "print(f\"power: t>2 {sm['mean_power_t2']*100:.1f}% -> t>3 {sm['mean_power_t3']*100:.1f}%\")\n"
            "print(f\"BHY: {sm['mean_bhy_n']:.1f} discoveries, implied |t| cutoff {sm['mean_bhy_cutoff']:.2f}, realized FDR {sm['mean_bhy_fdr']*100:.1f}%\")\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.bar(['t>2','t>3','BHY'], [sm['mean_fdr_t2']*100, sm['mean_fdr_t3']*100, sm['mean_bhy_fdr']*100],\n"
            "       color=[RED, AMBER, GREEN])\n"
            "ax.set_ylabel('realized FDR (%, 20-seed mean)')\n"
            "ax.set_title('Raising the bar (and BHY) purges false discoveries while keeping real factors')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The realized FDR collapses **{R['sm_fdr2']:.0f}% → {R['sm_fdr3']:.0f}%** from t>2 to "
            f"t>3 (a {R['sm_collapse']:.1f}× cut), and BHY keeps **{R['sm_bhy_n']:.0f}** genuinely "
            f"strong factors (implied $|t|$ ≈ {R['sm_bhy_cut']:.1f}) with a realized FDR of just "
            f"**{R['sm_bhy_fdr']:.1f}%** — neither 'reject nothing' nor 'reject everything'. The "
            "correction sees through the noise *and* banks the real edge. For the generic version see "
            "[346 Multiple-Testing](../../346-multiple-testing/); for what happens to factors *after* "
            "publication, [536 Anomaly-Decay-Post-Publication](../../536-anomaly-decay-post-publication/)."
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
