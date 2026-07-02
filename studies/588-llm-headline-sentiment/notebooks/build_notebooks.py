"""Generate the two narrative notebooks for Study 588 (LLM-Headline-Sentiment).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The whole study is
**synthetic and offline** — there is no real tape (dated LLM headline scores are not freely
reachable), so every cell runs deterministically anywhere. The frozen headline numbers in ``R``
mirror docs/results.md and are recomputed live in the quant notebook (the synthetic core is fast).
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic, seed 588;
# 1,500 business days). Edge fp 06461296a1f8, null 802a27d1a3e5, bank 47ff60cb8a0e.
R = dict(
    fp_series="06461296a1f8", fp_null="802a27d1a3e5", fp_bank="47ff60cb8a0e",
    n_days=1500,
    slope=0.0031, hac_t=6.33, ols_t=6.39, r2=2.7, placebo_p=0.0005,
    lagged_t=6.33, contemp_t=16.1,
    sharpe_gross=2.70, sharpe_net=2.58, turnover=0.77, mean_net=0.45,
    null_hac_t=1.79, null_placebo_p=0.071, null_sharpe_net=0.92,
    k_features=40, best_t=2.90, best_feature="sent_19", naive_p=0.007, maxt_p=0.173,
    naive_fpr=0.84, maxt_fpr=0.0,
    # positive control: (beta, mean HAC-t over 25 seeds)
    ctrl=[(0.0, 0.09), (0.0008, 1.79), (0.0015, 3.27), (0.0022, 4.76),
          (0.0030, 6.46), (0.0045, 9.65)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from llm_headline_sentiment import data, strategy as st

# The whole study is synthetic & offline — there is no real tape to load.
# EDGE world: a small day-ahead link is planted. NULL world: sentiment is noise vs the future.
EDGE, _ = data.synthetic_series(seed=588)          # beta>0 planted (the positive control)
NULL, _ = data.synthetic_series(beta=0.0, seed=588)
print("synthetic tape:", len(EDGE), "business days | edge fp",
      data.fingerprint(EDGE), "| null fp", data.fingerprint(NULL))
print("NOTE: synthetic-only study -> SIGNAL axis is capped at WEAK by construction.")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# LLM Headline Sentiment — can a robot reading the news predict tomorrow? 🤖📰\n"
            "### Letting a language model score the day's mood, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Overfitting_trap_survived%3F: Busted](https://img.shields.io/badge/Overfitting_trap_survived%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a very 2020s idea. Every morning, thousands of news headlines land. What if you "
            "pointed a **language model** (the ChatGPT kind) at them, asked *\"is the mood good or "
            "bad?\"*, and used that score to bet on tomorrow's market? It sounds like the natural "
            "upgrade of the old trick of *counting* positive and negative words — now the machine "
            "actually **reads** the sentence.\n\n"
            "It's a real, testable method. But it's also a magnet for two classic mistakes that make "
            "*nothing* look like *something*. This notebook shows the method working on a clean, "
            "made-up world — and then shows you both traps springing shut.\n\n"
            "> ⚠️ **This study is entirely synthetic.** There is no free, honest history of "
            "LLM-scored headlines to test on (the headlines are paywalled, and scoring old news with "
            "a modern model without cheating is genuinely hard). So this is a **method demo**, not a "
            "market claim — and by the desk's rules it can never score better than **Weak**.\n"
            ">\n"
            "> 📓 Want the *t*-stats, the permutation tests and the cost maths? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Not investment advice."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can the method work *in principle*? | **Yes.** On a clean world where the mood really "
            f"does nudge tomorrow, the honest pipeline catches it (a *t*-stat of **+{R['hac_t']}**, "
            "odds of a fluke about 1-in-2000). |\n"
            "| Can we prove it works on *real* markets? | **No — not here.** There's no honest free "
            "tape of dated LLM headline scores, so this stays a demo. Capped at **Weak**. |\n"
            "| What's the catch, trap #1? | **Try enough recipes and noise looks like gold.** With "
            f"**40 pure-noise** sentiment signals, the best one looks 'significant' (*p* = "
            f"{R['naive_p']}) — but it's a mirage: do it on noise 25 times and you 'win' **"
            f"{int(R['naive_fpr']*100)}%** of the time. |\n"
            "| Trap #2? | **Cheating with hindsight.** Score the headlines *after* you know how the "
            f"day went and the signal balloons (a *t* of {R['lagged_t']} → **{R['contemp_t']}**) — "
            "but no trader could have used it. |\n\n"
            "> The method is real; the hype usually isn't. Most \"AI predicts the market\" headlines "
            "are one of these two traps wearing a nice suit."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"An LLM reads today's news headlines, scores the mood, and that score forecasts "
            "tomorrow's market.\"* — the modern LLM-sentiment pitch (Lopez-Lira & Tang 2023), the "
            "context-aware descendant of counting good/bad words (Tetlock 2007; Loughran-McDonald "
            "2011).\n\n"
            "The intuition: a bad-news day should tilt the odds for tomorrow. Old-school sentiment "
            "just tallied scary words; an LLM understands *\"not as bad as feared\"* is actually "
            "**good** news. Richer reading → better signal. That's the hope."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked and you could trade it, it would be a money machine — which is exactly why "
            "you should be suspicious. The desk has already taken apart curated news-tone "
            "([259](../../259-news-tone/)), survey sentiment ([257](../../257-aaii-sentiment/)) and "
            "buzz ([335](../../335-buzz-sentiment-etf/)). Here we go after the **LLM method itself** "
            "— not to debunk the idea, but to show the two traps that make it *look* far better than "
            "it is."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build a clean world.** We *simulate* daily mood scores and returns, with a dial. "
            "Turn the dial up and mood genuinely nudges tomorrow (a real edge); turn it to zero and "
            "mood is pure noise.\n"
            "2. **Check the machine is honest.** On the edge world it should find the signal; on the "
            "noise world it should find *nothing*.\n"
            "3. **Spring the traps.** Then we (a) try dozens of sentiment recipes on pure noise and "
            "report the winner, and (b) 'cheat' by scoring the news with hindsight — and watch fake "
            "signals appear.\n\n"
            "*Timing:* today's mood is used to trade **tomorrow** — one day's lag, no peeking. That's "
            "the only honest way; the trap version peeks on purpose to show you the damage."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does the machine catch a *real* edge, and stay quiet on noise?**"
        ),
        code(
            "reg_edge = st.predictive_regression(EDGE['sentiment'], EDGE['forward_ret'])\n"
            "reg_null = st.predictive_regression(NULL['sentiment'], NULL['forward_ret'])\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['EDGE world\\n(real link)','NULL world\\n(noise)'],\n"
            "       [reg_edge['t'], reg_null['t']], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('predictive t-stat (HAC)')\n"
            "ax.set_title('The machine catches a real edge, stays flat on noise'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"EDGE: t {reg_edge['t']:+.2f}, explains {reg_edge['r2']*100:.1f}% of tomorrow\")\n"
            "print(f\"NULL: t {reg_null['t']:+.2f} (below the bar)\")"
        ),
        md(
            f"So the pipeline is honest: on the edge world it reads *t* **+{R['hac_t']}** (explains "
            f"~{R['r2']}% of tomorrow's move — small but real); on the noise world it reads *t* "
            f"**+{R['null_hac_t']}**, below the bar. Good. **Now the traps.**"
        ),
        md(
            "**Trap 1 — 'report the best prompt.'** A researcher rarely tries one recipe; they try "
            "many (different prompts, models, ways of averaging) and headline the winner. Watch what "
            "happens when we try **40 recipes that are all pure noise**:"
        ),
        code(
            "feats, fr, _ = data.synthetic_feature_bank(n_features=40, n_true=0, seed=588)\n"
            "ts = st.all_feature_tstats(feats, fr)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(range(len(ts)), np.sort(ts), color=GREY, width=.8)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = 2 ('+'\"significant\"'+')')\n"
            "ax.set_xlabel('40 sentiment recipes (all pure noise), sorted'); ax.set_ylabel('|t-stat|')\n"
            "ax.set_title(f'Best of 40 noise recipes: |t| = {ts.max():.2f} — looks significant!')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Best-of-40 |t| = {ts.max():.2f} on PURE NOISE. Try enough and one always wins.')"
        ),
        md(
            f"There it is: the best of 40 **noise** recipes hits |*t*| = **{R['best_t']}**, which a "
            f"naive reader calls significant (*p* = {R['naive_p']}). It isn't — it's the *maximum of "
            "40 coin-flips*, which is naturally big. The quant notebook prices this properly and "
            f"finds the honest *p* is **{R['maxt_p']}** (not significant). Do the whole exercise 25 "
            f"times and the naive test 'finds' a signal **{int(R['naive_fpr']*100)}%** of the time; "
            f"the honest correction: **{int(R['maxt_fpr']*100)}%**."
        ),
        md(
            "**Trap 2 — cheating with hindsight.** If you score the headlines *after* the market has "
            "already moved (a mood label written with the benefit of knowing the day), the 'signal' "
            "explodes — but it's unusable:"
        ),
        code(
            "la = st.look_ahead_contrast(EDGE)\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['HONEST\\n(mood -> tomorrow)','CHEATING\\n(hindsight mood)'],\n"
            "       [la['lagged_t'], la['contemporaneous_t']], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('t-stat')\n"
            "ax.set_title('Hindsight balloons the signal — but no trader could use it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"honest t {la['lagged_t']:+.2f}  vs  hindsight t {la['contemporaneous_t']:+.2f}\")"
        ),
        md(
            f"The hindsight version reads *t* **{R['contemp_t']}** versus the honest **{R['lagged_t']}** "
            "— roughly 2.5× bigger, for free, by reading the answer. A lot of breathless 'the AI "
            "predicted it!' results are exactly this."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The method works on a clean world, but with no real tape it can't "
            "be certified. Capped at Weak by the desk's rules.\n"
            "- **Tradability — Mirage.** The only Sharpe that survives costs is a *planted* edge in a "
            "frictionless simulation; a real day-ahead timer trades constantly, and the published "
            "'edges' are mostly the traps above.\n"
            "- **Overfitting trap survived? — Busted.** Report-the-best-recipe and hindsight-scoring "
            "both manufacture signals from nothing."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not on this evidence. Even if a small real edge exists, a one-day sign timer flips "
            "almost daily, you'd need a live, licensed, *point-in-time* LLM headline feed nobody "
            "gives away, and you'd be competing with everyone else running the same prompt. The "
            "method is worth studying; the get-rich pitch built on it usually collapses into Trap 1 "
            "or Trap 2."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sentiment cousins.** [259 News-Tone](../../259-news-tone/) (curated daily tone, "
            "the within-month placebo), [257 AAII](../../257-aaii-sentiment/) & "
            "[335 Buzz](../../335-buzz-sentiment-etf/) (survey / buzz), "
            "[566 Earnings-Call-Tone](../../566-earnings-call-tone/) (call transcripts).\n"
            "- **The correction that saves you.** The quant notebook implements the "
            "*White Reality Check / Westfall-Young max-t* test — the one that turns a fake *p* = "
            "0.007 into an honest *p* = 0.17.\n\n"
            "*Think you've got a real LLM-headline edge? Fork this, bring a genuinely point-in-time "
            "score, and show it clears t = 2 **after** a best-of-K correction and **without** "
            "same-day leakage.*"
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
            "# LLM Headline Sentiment — a quantitative teardown 🔬\n"
            "### HAC predictive regression · label-shuffle placebo · best-of-K max-*t* (Reality Check) · look-ahead contrast · costed sign timer · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Overfitting_trap_survived%3F: Busted](https://img.shields.io/badge/Overfitting_trap_survived%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the honest "
            "LLM-headline-sentiment pipeline, prove it recovers a planted day-ahead edge, then price "
            "the two overfitting traps that generate most \"LLM beats the market\" results.\n\n"
            "> ⚠️ **Synthetic-only, offline, deterministic (seed 588).** No real tape of dated LLM "
            f"headline scores is freely reachable, so the SIGNAL axis is **capped at WEAK** (a `REAL` "
            f"stamp needs a robust *t* ≥ 2 on a real tape). Edge fp `{R['fp_series']}`, null "
            f"`{R['fp_null']}`, bank `{R['fp_bank']}`; as-of 2026-06-30. Methods in "
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
            f"| **Signal** | `WEAK` | Synthetic-only (capped). Planted edge: HAC *t* **+{R['hac_t']}**, "
            f"R² **{R['r2']}%**, placebo *p* **{R['placebo_p']}**; null flat (HAC *t* +{R['null_hac_t']}). "
            "A machinery proof, not market evidence. |\n"
            f"| **Tradability** | `MIRAGE` | Costed sign timer nets Sharpe **{R['sharpe_net']}** — but "
            "only on a *planted* edge in a frictionless world (turnover **"
            f"{R['turnover']}**/day). No real tape. |\n"
            f"| **Overfitting trap survived?** | `BUSTED` | Best-of-40-noise |*t*| **{R['best_t']}** "
            f"(naive *p* {R['naive_p']}) → max-*t* *p* **{R['maxt_p']}**; naive FPR **"
            f"{int(R['naive_fpr']*100)}%** vs corrected **{int(R['maxt_fpr']*100)}%**. Look-ahead: "
            f"*t* {R['lagged_t']} → {R['contemp_t']}. |\n\n"
            "> 💡 In plain words: the engine is faithful (it banks a planted edge and stays flat at "
            "the null); the story is how that same engine is *misused* to print false positives."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t$ be the LLM mood score from day-$t$ headlines and $r_{t+1}$ the next-day return.\n\n"
            "- **H₁ (predictability).** slope of $r_{t+1}$ on $s_t$ is $>0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (not data-mined).** the winner among $K$ candidate recipes survives a "
            "family-wise (max-*t*) correction.\n"
            "- **H₃ (no look-ahead).** the edge is the *lagged* one ($s_t \\to r_{t+1}$), not a "
            "contemporaneous leak.\n"
            "- **H₄ (net).** it survives turnover costs and borrow.\n\n"
            "On the synthetic tape we can **verify the machinery** (H₁ holds where an edge is "
            "planted; H₂/H₃ are where naive practice fails) — but H₁ can never be *certified for "
            "markets* without a real tape, which is why the Signal axis is capped at `WEAK`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. LLM sentiment is the plausible successor to lexicon "
            "sentiment (Loughran-McDonald 2011): it reads context a word-count can't. But the two "
            "things that make it *look* like alpha are structural, not incidental — (a) the space of "
            "prompts × models × aggregations is huge, so **best-of-K** significance is nearly "
            "guaranteed on noise (White 2000; Harvey-Liu-Zhu 2016), and (b) scoring historical text "
            "with a modern model is a **look-ahead** minefield. This study isolates both, on a tape "
            "where we *know* the truth."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Regression.** $r_{t+1} = \\alpha + \\beta s_t + \\varepsilon$, **Newey-West HAC** *t* "
            "on $\\beta$ (persistent sentiment ⇒ autocorrelated regressor; naive OLS *t* is "
            "over-confident).\n"
            "- **Placebo.** Label-shuffle the returns 2000× ; *p* = tail prob of the observed |*t*|.\n"
            "- **Best-of-K.** A bank of $K=40$ candidate recipes; the naive winner vs the "
            "**permutation-max** (Westfall-Young / White Reality Check) null.\n"
            "- **Look-ahead.** Honest lagged ($s_t \\to r_{t+1}$) vs a hindsight-tainted "
            "contemporaneous fit.\n"
            "- **Frictions.** A one-day sign timer, one-way cost × NAV per flip + borrow on shorts.\n"
            "- **Positive control.** Planted $\\beta$ of increasing strength, mean HAC *t* over 25 "
            "seeds (house rule).\n\n"
            "One execution lag, baked into the schema: $s_t$ trades $r_{t+1}$ (the `forward_ret` "
            "column *is* the $t\\to t{+}1$ return). No second silent shift; the contemporaneous "
            "variant is labelled a **leak**, never the headline."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁ (machinery proof)\n\n"
            "Edge world vs null world, HAC *t* and the label-shuffle placebo."
        ),
        code(
            "reg = st.predictive_regression(EDGE['sentiment'], EDGE['forward_ret'])\n"
            "p = st.placebo_pvalue(EDGE['sentiment'], EDGE['forward_ret'], n_perm=2000, seed=588)\n"
            "reg0 = st.predictive_regression(NULL['sentiment'], NULL['forward_ret'])\n"
            "p0 = st.placebo_pvalue(NULL['sentiment'], NULL['forward_ret'], n_perm=2000, seed=588)\n"
            "sc = EDGE.sample(400, random_state=1)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.scatter(sc['sentiment'], sc['forward_ret']*100, s=10, color=GREY, alpha=.5)\n"
            "xs = np.linspace(-1, 1, 50)\n"
            "a1.plot(xs, (reg['slope']*xs)*100 + EDGE['forward_ret'].mean()*100, c=GREEN, lw=2)\n"
            "a1.set_xlabel('today mood'); a1.set_ylabel('tomorrow return %')\n"
            "a1.set_title(f\"EDGE: slope {reg['slope']:.4f}, HAC t {reg['t']:+.2f}, R2 {reg['r2']*100:.1f}%\")\n"
            "a2.bar(['EDGE','NULL'], [reg['t'], reg0['t']], color=[GREEN, GREY], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=1); a2.set_ylabel('HAC t')\n"
            "a2.set_title('faithful: catches edge, flat on null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"EDGE HAC t {reg['t']:+.2f} (OLS {reg['t_ols']:+.2f}), placebo p {p:.4f}\")\n"
            "print(f\"NULL HAC t {reg0['t']:+.2f}, placebo p {p0:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **holds where an edge is planted** — HAC *t* **+{R['hac_t']}** "
            f"(R² {R['r2']}%, placebo *p* {R['placebo_p']}) — and the null is flat (*t* "
            f"+{R['null_hac_t']}, *p* {R['null_placebo_p']}). The pipeline is a faithful detector; "
            "everything below is about *misuse*, not a broken engine. (Synthetic — not a market "
            "claim.)"
        ),
        md(
            "### 4b · Trap 1 — best-of-K and the max-*t* (Reality Check) correction — H₂\n\n"
            "A bank of **40 pure-noise** recipes. The naive winner vs the family-wise null."
        ),
        code(
            "feats, fr, is_true = data.synthetic_feature_bank(n_features=40, n_true=0, seed=588)\n"
            "mt = st.maxt_pvalue(feats, fr, n_perm=1000, seed=588)\n"
            "ts = st.all_feature_tstats(feats, fr)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [RED if t==ts.max() else GREY for t in np.sort(ts)]\n"
            "ax.bar(range(len(ts)), np.sort(ts), color=cols, width=.8)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t=2')\n"
            "ax.set_xlabel('40 recipes (all noise), sorted'); ax.set_ylabel('|HAC t|')\n"
            "ax.set_title(f\"best-of-40 |t|={mt['best_t']:.2f}: naive p {mt['naive_p']:.3f} -> max-t p {mt['maxt_p']:.3f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"winner {mt['best_t']:.2f} | naive single-test p {mt['naive_p']:.3f} | max-t p {mt['maxt_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected the naive way.** The best of 40 *noise* recipes hits "
            f"|*t*| **{R['best_t']}**, single-test *p* **{R['naive_p']}** ('significant'). But it is "
            f"the max of 40 statistics — the **max-*t*** null puts it at *p* **{R['maxt_p']}**. The "
            "correction is the whole ballgame."
        ),
        md(
            "**And the false-positive rates**, over 25 independent pure-noise banks — how often each "
            "test 'discovers' a signal that isn't there:"
        ),
        code(
            "fp = st.synthetic_maxt_falsepos(data, n_features=40, n_seeds=25, n_perm=400, alpha=0.05)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['naive\\n(report best)','max-t\\n(corrected)'],\n"
            "       [fp['naive_fpr']*100, fp['maxt_fpr']*100], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(5, ls='--', c=GREY, lw=1, label='nominal 5%')\n"
            "ax.set_ylabel('false-positive rate %'); ax.legend()\n"
            "ax.set_title('Report-the-best fires 84% on noise; the correction sits near 0%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"naive FPR {fp['naive_fpr']*100:.0f}%  vs  max-t FPR {fp['maxt_fpr']*100:.0f}%  (target 5%)\")"
        ),
        md(
            f"> 💡 In plain words: the naive 'report the winner' rule fires **{int(R['naive_fpr']*100)}%** "
            f"of the time on pure noise; the max-*t* correction drops it to **{int(R['maxt_fpr']*100)}%**. "
            "*This* is why an uncorrected best-of-K LLM-sentiment result means almost nothing."
        ),
        md(
            "### 4c · Trap 2 — the look-ahead contrast — H₃\n\n"
            "Honest lagged design vs a hindsight-tainted contemporaneous fit."
        ),
        code(
            "la = st.look_ahead_contrast(EDGE)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['lagged\\n(honest)','contemporaneous\\n(hindsight leak)'],\n"
            "       [la['lagged_t'], la['contemporaneous_t']], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Hindsight scoring inflates the t-stat ~2.5x')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"lagged (tradable) t {la['lagged_t']:+.2f}  vs  contemporaneous (leak) t {la['contemporaneous_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: H₃ matters — the contemporaneous fit reads *t* **{R['contemp_t']}** "
            f"vs the honest lagged **{R['lagged_t']}**. A mood label written with hindsight partly "
            "*reads the answer*; the gap is the look-ahead tax."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — synthetic-only (capped). Machinery verified (HAC *t* +{R['hac_t']} "
            "on a planted edge, flat at the null), but no real tape to certify.\n"
            f"- **Tradability `MIRAGE`** — the surviving Sharpe (**{R['sharpe_net']}** net) is a "
            "planted, frictionless edge; a real day-ahead timer trades daily and no free PIT feed "
            "exists.\n"
            "- **Overfitting trap survived? `BUSTED`** — best-of-K (naive FPR "
            f"{int(R['naive_fpr']*100)}%) and hindsight (*t* {R['lagged_t']} → {R['contemp_t']}) both "
            "manufacture signal from nothing."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the costed sign timer\n\n"
            "Even on the *planted* edge, a one-day sign timer flips constantly."
        ),
        code(
            "tm = st.sign_timer(EDGE, cost_bps=1.0, borrow_ann_bps=50.0, long_short=True)\n"
            "tm0 = st.sign_timer(NULL, cost_bps=1.0, borrow_ann_bps=50.0, long_short=True)\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['EDGE gross','EDGE net','NULL net'],\n"
            "       [tm['sharpe_gross'], tm['sharpe_net'], tm0['sharpe_net']],\n"
            "       color=[GREEN, AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title(f\"planted-edge Sharpe survives costs ({tm['sharpe_net']:.2f}); null is noise ({tm0['sharpe_net']:.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"EDGE gross {tm['sharpe_gross']:.2f} -> net {tm['sharpe_net']:.2f} (turnover {tm['turnover']:.2f}/day)\")\n"
            "print(f\"NULL net {tm0['sharpe_net']:.2f}  (1 bp/flip one-way + 50 bps/yr borrow)\")"
        ),
        md(
            f"> 💡 In plain words: the net Sharpe **{R['sharpe_net']}** looks great — but it exists "
            "*only because the edge was planted at full strength in a frictionless synthetic world*. "
            f"The null nets **{R['null_sharpe_net']}** (noise). Neither is a market claim; `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the HAC *t* a faithful detector, or does it drift? Plant a $\\beta$ of increasing "
            "strength and average the HAC *t* over **25 seeds** (house rule) so no lucky seed can "
            "fake it."
        ),
        code(
            "betas = [b for b,_ in " + repr(R["ctrl"]) + "]\n"
            "ts = [st.synthetic_mean_t(data, beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted beta (0 = null)'); ax.set_ylabel('mean HAC t (25 seeds)')\n"
            "ax.set_title('flat at the null, monotone up past t=2 as the edge grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'beta {b:.4f} -> mean HAC t {t:+.2f}')"
        ),
        md(
            "The mean HAC *t* is ≈ 0 at the null and climbs monotonically past 2 as the planted edge "
            "grows — a faithful detector. So the traps in 4b/4c are about *practice*, not the engine. "
            "For the sentiment cousins on real tapes see [259 News-Tone](../../259-news-tone/), "
            "[257 AAII](../../257-aaii-sentiment/), [335 Buzz](../../335-buzz-sentiment-etf/) and "
            "[566 Earnings-Call-Tone](../../566-earnings-call-tone/) — and remember: on a "
            "synthetic-only footing this study is `WEAK` by construction, never `REAL`."
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
