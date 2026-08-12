"""Generate the two narrative notebooks for Study 845 (Stadium Naming-Rights Curse).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY /
sponsor tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + 28 listed
# sponsor tapes 1997 -> 2026-06-30; 34 hardcoded naming-rights deals 1997-12 -> 2023-06).
R = dict(
    n_total=34, n_tradable=29, n_untradable=5, n_on_tape=28,
    # 1-year window
    w1_mean=-11.26, w1_med=-18.02, w1_t=-2.54, w1_nw=-2.41,
    w1_hit=21, w1_hit_n=28, w1_hit_pct=75.0, w1_wilson=(56.6, 87.3),
    w1_pl_obs=-11.26, w1_pl_mean=5.75, w1_pl_sd=10.91, w1_pl_p=0.003, w1_pl_draws=3000,
    w1_pre_n=12, w1_pre_mean=-5.96, w1_pre_t=-0.84,
    w1_post_n=16, w1_post_mean=-15.24, w1_post_t=-2.70,
    w1_ov_gross=11.26, w1_ov_net=10.06, w1_ov_t=2.27, w1_ov_win=71, w1_ov_drag=120,
    # 2-year window
    w2_mean=-10.91, w2_med=-13.34, w2_t=-1.82, w2_nw=-1.69,
    w2_hit=17, w2_hit_pct=60.7, w2_wilson=(42.4, 76.4),
    w2_pl_obs=-10.91, w2_pl_mean=11.66, w2_pl_sd=19.12, w2_pl_p=0.022,
    w2_pre_n=12, w2_pre_mean=-8.88, w2_pre_t=-1.15,
    w2_post_n=16, w2_post_mean=-12.44, w2_post_t=-1.39,
    w2_ov_gross=10.91, w2_ov_net=8.71, w2_ov_t=1.45, w2_ov_win=61, w2_ov_drag=220,
    # robustness: drop k most-negative names -> (mean_pct, t)
    dropworst={0: (-11.26, -2.54), 1: (-9.83, -2.26), 2: (-8.69, -1.99),
               3: (-7.63, -1.73), 4: (-6.52, -1.46)},
    loo=(-3.12, -2.26),
    worst=[("CZR", -49.99, "Caesars Superdome"), ("ALGT", -39.37, "Allegiant Stadium"),
           ("TFC", -35.29, "Truist Park"), ("C", -34.23, "Citi Field"),
           ("BALL", -31.42, "Ball Arena"), ("BCS", -31.16, "Barclays Center")],
    best=[("JPM", 31.87, "Chase Center"), ("FDX", 35.58, "FedExField"),
          ("TM", 37.96, "Toyota Center")],
    # synthetic control
    syn_null_mean=0.26, syn_null_sd=1.08, syn_null_fire=1,
    syn_planted_t=-3.66, syn_planted_pct=-17.7,
    fp_spy="d8056fdb1c84",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Curse: Mixed](https://img.shields.io/badge/Curse-Mixed-dab617?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from stadium_curse import data, strategy as st

TABLE = data.deal_table()
TRADABLE = data.tradable_deals()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, PRICES = data.load_prices()
else:
    SPY = PRICES = None
print("real cache present:", HAVE_REAL,
      "| deals in table:", len(TABLE),
      "| tradable:", len(TRADABLE))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does buying a stadium's name curse the company? 🏟️📉\n"
            "### The \"naming-rights curse\" — Enron Field, the FTX Arena, and whether "
            "there's anything to it beyond two famous disasters\n\n"
            + BADGES +
            "It's one of the tidiest stories in markets. A company pays a fortune to bolt "
            "its name onto a stadium — and then it blows up. **Enron Field** opened in "
            "1999; Enron was bankrupt by 2001. The **FTX Arena** got its name in 2021; FTX "
            "imploded in 2022. The **MCI Center**? WorldCom, one of the biggest accounting "
            "frauds in history. The moral practically writes itself: a company splurging on "
            "a vanity trophy is a company at a cocky peak, about to fall.\n\n"
            "That's the claim we test — not on the two or three disasters everyone "
            "remembers, but on **every** big naming-rights deal we could pin a date and a "
            "ticker to. Do sponsors really underperform afterward? Or is the \"curse\" just "
            "a couple of vivid stories we've stitched into a law?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the sub-era "
            "split and the tail jackknife? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do sponsors underperform in the year after the deal? | **Yes, and it's "
            f"real** — the average listed sponsor lags the S&P 500 by "
            f"**{R['w1_mean']:.1f}%** over the next year (*t* = {R['w1_t']:.2f}, and a "
            f"random entry date on the same names beats it only "
            f"**{R['w1_pl_p']*100:.1f}%** of the time). |\n"
            f"| Is it just Enron and FTX? | **No** — those two aren't even in the test "
            "(both untradable). Even the *surviving* sponsors underperform. |\n"
            "| So the curse is proven? | **No — and this is the catch.** The effect "
            f"**vanishes by year two**, it only exists **after 2010**, and it falls apart "
            "if you remove the two names (Caesars, Allegiant) that COVID happened to gut. |\n"
            "| Could you trade it? | **Not really.** Shorting the sponsors looks great on "
            "paper (+10%/yr) but rides the same fragile tail and needs you to short the "
            "hardest-to-borrow names in a crisis. |\n\n"
            "> A folklore claim that turns out to be **more than a myth but less than a "
            "law** — significant in year one, gone by year two."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A 20-year, nine-figure deal to slap your logo on a stadium is what a "
            "company does at the top — flush with cash, drunk on its own success, spending "
            "shareholders' money on a trophy instead of the business. Watch what happens "
            "next.\"*\n\n"
            "This is the **managerial-hubris / peak-earnings** story, and it has a serious "
            "academic cousin (Jensen's free-cash-flow agency costs: firms that over-invest "
            "in low-return vanity projects tend to underperform). The folklore just picks "
            "the most photogenic example of vanity spending — a building with your name on "
            "it — and the most photogenic failures: Enron, WorldCom, FTX."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were a real, robust law, it would be a clean, almost poetic market "
            "inefficiency — a *press release* (\"we bought the naming rights!\") that "
            "predicts a stock's next year. And the honest way to test it is brutal: line "
            "up **all** the deals, not just the famous graves, and see whether the average "
            "sponsor really lags — or whether we've been fooled by a handful of "
            "unforgettable blow-ups (the ones that, tellingly, we can't even include here "
            "because they went to zero)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The deals.** **{R['n_total']}** major naming-rights deals with dates and "
            f"sponsor tickers. **{R['n_untradable']}** are the untradable cautionary tales "
            "(Enron, WorldCom, FTX, Crypto.com, SoFi — private or bankrupt, no usable "
            f"stock tape); **{R['n_on_tape']}** listed sponsors make it into the test.\n"
            "- **The measure.** Each sponsor's total return **minus the S&P 500's** over "
            "the 1 (and 2) years after the deal — did the sponsor beat or lag the market?\n"
            "- **The luck check.** Pick a *random* date on each of the same 28 stocks, "
            "3,000 times — how often does random timing produce a drop this big?\n"
            "- **The robustness checks.** Does it hold at 2 years? Before *and* after 2010? "
            "If you remove the worst couple of names?\n\n"
            "> ⚠️ **Survivorship, stated up front.** Enron and WorldCom went to *zero* — "
            "there's no clean stock tape to include them, so they're **left out**. That "
            "means our test, if anything, *understates* the curse: we're only measuring "
            "the sponsors that survived."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average sponsor return minus the S&P over the next "
            "year, vs. what random timing on the same names looks like."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.car_stats(SPY, PRICES, TRADABLE, window=252)\n"
            "    obs = cs['mean'] * 100\n"
            "else:\n"
            "    obs = R['w1_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['bought a stadium name\\n(n=28, next year)',\n"
            "        'random timing\\n(same names)'],\n"
            "       [obs, R['w1_pl_mean']], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([obs, R['w1_pl_mean']]):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('return vs S&P 500 over the next year (%)')\n"
            "ax.set_title('Sponsors lag the market after the deal — random timing does not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  vs  random-timing {R[\"w1_pl_mean\"]:+.2f}%  '\n"
            "      f'(placebo p = {R[\"w1_pl_p\"]:.3f})')"
        ),
        md(
            f"So it's **not** an Enron/FTX artifact: the 28 sponsors that *survived* still "
            f"lag the S&P by **{R['w1_mean']:.1f}%** in year one, while random timing on "
            f"those very same stocks would have them *up* {R['w1_pl_mean']:+.1f}%. Random "
            f"timing beats the real outcome only {R['w1_pl_p']*100:.1f}% of the time. On "
            "its own, that looks like a real effect.\n\n"
            "**But look at the individual sponsors** — is it a broad drag, or a few "
            "catastrophes dragging the average down?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, kept = st.stack_bhar(SPY, PRICES, TRADABLE, window=252)\n"
            "    kept = kept.sort_values('bhar')\n"
            "    labels = [f\"{r.ticker}\" for r in kept.itertuples()]\n"
            "    vals = list(kept['bhar'] * 100)\n"
            "else:\n"
            "    w = R['worst']; b = R['best']\n"
            "    labels = [x[0] for x in w] + ['...'] + [x[0] for x in b]\n"
            "    vals = [x[1] for x in w] + [0] + [x[1] for x in b]\n"
            "fig, ax = plt.subplots(figsize=(11, 4.6))\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "ax.bar(range(len(vals)), vals, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(range(len(labels)))\n"
            "ax.set_xticklabels(labels, rotation=90, fontsize=7)\n"
            "ax.set_ylabel('1-yr return vs S&P (%)')\n"
            "ax.set_title('A fat LEFT tail: a handful of sponsors cratered 30-50%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('worst:', [(x[0], x[1]) for x in R['worst'][:4]])"
        ),
        md(
            "It's a **left-skewed pile-up**, not a uniform curse. A cluster of sponsors — "
            "Caesars (−50%), Allegiant (−39%), Truist, Citi, Ball, Barclays — collapsed "
            "30–50% in the year after their deal, while the winners (Toyota, FedEx, "
            "JPMorgan) only gained ~30–38%. And notice *who* those losers are: Citi and "
            "Barclays signed just before the 2008 crash; Caesars, Allegiant, Truist and "
            "Ball signed just before COVID and the 2022 rate shock. Hubris — or cyclical "
            "companies that happened to sign right before a crisis?\n\n"
            "**Now the checks that decide whether it's real.** Does it hold in different "
            "eras, and does it survive dropping the worst couple of names?"
        ),
        code(
            "eras = ['pre-2010\\n(n=12)', 'post-2010\\n(n=16)']\n"
            "if HAVE_REAL:\n"
            "    es = st.era_split(SPY, PRICES, TRADABLE, window=252)\n"
            "    em = [es['pre']['mean'] * 100, es['post']['mean'] * 100]\n"
            "    et = [es['pre']['t'], es['post']['t']]\n"
            "else:\n"
            "    em = [R['w1_pre_mean'], R['w1_post_mean']]; et = [R['w1_pre_t'], R['w1_post_t']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.bar(eras, em, color=[GREY, RED], width=.55)\n"
            "for i, (v, t) in enumerate(zip(em, et)):\n"
            "    a1.annotate(f'{v:+.1f}%\\n(t={t:+.2f})', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean 1-yr return vs S&P (%)')\n"
            "a1.set_title('The curse is a POST-2010 thing only')\n"
            "ks = sorted(R['dropworst']); ts = [R['dropworst'][k][1] for k in ks]\n"
            "a2.plot(ks, ts, 'o-', color=RED)\n"
            "a2.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "a2.annotate('desk bar (t=-2)', (0.1, -2), va='bottom', fontsize=8, color=GREY)\n"
            "a2.set_xlabel('# of worst names removed'); a2.set_ylabel('one-sample t')\n"
            "a2.set_title('Drop the 2 COVID-hit names -> below the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pre-2010 t =', round(et[0], 2), '| post-2010 t =', round(et[1], 2))\n"
            "print('t after dropping worst 0/1/2/3/4:', [R['dropworst'][k][1] for k in ks])"
        ),
        md(
            f"There it is. **Before 2010 there's basically nothing** (pre-2010 *t* = "
            f"{R['w1_pre_t']:.2f}); the entire effect lives after 2010. And removing just "
            f"the **two** worst names — Caesars and Allegiant, both stadium sponsors then "
            f"gutted by COVID — drops the *t* from {R['dropworst'][0][1]:.2f} to "
            f"**{R['dropworst'][2][1]:.2f}**, below the desk's significance bar. A real "
            "law shouldn't depend on two names and one decade.\n\n"
            "That's why this is a **Weak** signal, not a real one: significant in year "
            "one, but too fragile — and too tangled up with crisis-timing luck — to bank."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Sponsors really do lag the S&P by **{R['w1_mean']:.1f}%** "
            f"in year one (*t* = {R['w1_t']:.2f}, placebo *p* = {R['w1_pl_p']:.3f}), and "
            "it's more than the Enron/FTX legend — but it's gone by year two, exists only "
            "post-2010, and evaporates if you drop the two COVID-hit tail names.\n"
            "- **Tradability — Fragile.** Shorting the sponsors nets +10%/yr on paper but "
            "rides the same fragile tail and needs you to short the least-borrowable names "
            "in a crisis.\n"
            "- **\"Is the naming-rights curse real?\" — Mixed.** More than a myth, less "
            "than a law."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Hubris vs. bad luck.** The honest limit: with 28 deals and a 2019–2021 "
            "cluster that ran straight into COVID, \"vanity spending predicts doom\" and "
            "\"cyclical firms signed right before a crisis\" make the *same* prediction "
            "here. Separating them needs many more deals across many more cycles.\n"
            "- **The survivorship twist cuts the other way for once.** Because the actual "
            "bankruptcies (Enron, WorldCom) are excluded, the real curse — if it exists — "
            "is *worse* than the −11% we can measure. That's a rare case where "
            "survivorship bias makes a finding *conservative*, not inflated.\n"
            "- **Sibling studies:** [160-skyscraper-curse](../../160-skyscraper-curse/) "
            "(the tallest *building* marks the top), "
            "[746-hq-relocation](../../746-hq-relocation/) (the shiny new *HQ*), and "
            "[722-logo-rebrand](../../722-logo-rebrand/) (the expensive *rebrand*) — the "
            "same corporate-vanity family, different trophies.\n\n"
            "*Think you can turn a Weak year-one blip into a robust, tradable law? Show it "
            "holding at 2 years, in both eras, without the two crisis names — then we'll "
            "talk.*"
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
            "# The Stadium Naming-Rights Curse — a quantitative teardown 🔬\n"
            "### A cross-sectional BHAR event study · a random-entry placebo · the sub-era "
            "split and tail-jackknife that demote it to Weak · a costed short overlay · a "
            "20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **managerial hubris / peak-earnings "
            "signaling**: a firm buying expensive stadium naming rights subsequently "
            "underperforms. The folklore's evidence is a short list of blow-ups (Enron, "
            "WorldCom, FTX). The job here is to measure it honestly across every dated, "
            "listed deal, then ask the only questions that matter: *is it more than the "
            "cherry-pick, is it robust, and is it tradable?*\n\n"
            "> ⚠️ **Data note.** SPY + 28 listed sponsor total-return tapes (1997→2026), "
            "yfinance, cached; **34 hardcoded naming-rights deals** 1997→2023, of which 5 "
            "are untradable cautionary tales (private/bankrupt sponsors) excluded from the "
            "return test. **Survivorship is named on the Signal axis** — the actual "
            "bankruptcies (Enron, WorldCom) have no tape, so the measured curse "
            "*understates* the folklore's worst cases. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (SPY fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 1-yr sponsor BHAR **{R['w1_mean']:.1f}%**, one-sample "
            f"**t = {R['w1_t']:.2f}** (NW {R['w1_nw']:.2f}), hit {R['w1_hit_pct']:.0f}%, "
            f"placebo **p = {R['w1_pl_p']:.3f}** — but fails 2-yr (t={R['w2_t']:.2f}), "
            f"fails sub-era (pre-2010 t={R['w1_pre_t']:.2f}), fails drop-2 "
            f"(t={R['dropworst'][2][1]:.2f}) |\n"
            f"| **Tradability** | `FRAGILE` | short-sponsor/long-SPY nets "
            f"**{R['w1_ov_net']:+.1f}%**/yr (t={R['w1_ov_t']:.2f}) but dies at 2yr "
            f"(t={R['w2_ov_t']:.2f}), tail-driven, unshortable names |\n"
            f"| **Curse: cherry-pick?** | `MIXED` | even survivors lag "
            f"({R['w1_mean']:.1f}%, placebo p={R['w1_pl_p']:.3f}) — more than the anecdote, "
            "less than a law |\n\n"
            "> 💡 In plain words: a folklore claim that is genuinely significant in year "
            "one yet fails every robustness cut a Real stamp requires — the textbook shape "
            "of a **Weak** result."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For each deal $i$ with announcement date $\\tau_i$ and sponsor ticker, define "
            "the **buy-and-hold abnormal return** over a $W$-session forward window "
            "(Barber & Lyon 1997):\n\n"
            "$$\\mathrm{BHAR}_i(W) = \\left(\\frac{P^{spon}_{\\tau_i+W}}{P^{spon}_{\\tau_i}} "
            "- 1\\right) - \\left(\\frac{P^{spy}_{\\tau_i+W}}{P^{spy}_{\\tau_i}} - 1\\right)$$\n\n"
            "entering at the close of the first session on/after $\\tau_i$. The claims:\n\n"
            "- **H₁ (the curse).** $E[\\mathrm{BHAR}_i] < 0$, large and systematic across "
            "deals — not just Enron/FTX.\n"
            "- **H₂ (robustness).** It holds at both $W=252$ (1yr) *and* $W=504$ (2yr), "
            "and in **both** sub-eras.\n"
            "- **H₃ (not a tail artifact).** It survives removing the worst couple of names.\n"
            "- **H₄ (tradable).** A short-sponsor/long-SPY overlay earns it net of costs "
            "and borrow.\n\n"
            "We find **H₁ supported** (t = −2.54, placebo p = 0.003), but **H₂ rejected** "
            "(dead at 2yr, pre-2010 t = −0.84), **H₃ rejected** (drop-2 → t = −1.99), and "
            "**H₄ Fragile** (positive but non-robust). Net: a **Weak** signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Deals are **distinct names on mostly non-overlapping dates**, so the planned "
            "primary is a **one-sample t** across the per-deal BHARs. Because several "
            "deals still cluster in one era (and thus share market weather), a "
            "**Newey-West** HAC t on the time-ordered BHARs is reported as a conservative "
            "cross-check; the hit rate carries a **Wilson** interval. The falsification is "
            "a **random-entry placebo**: keep the same 28 tickers but read each BHAR from "
            "a random pseudo-announcement date on that ticker's own tape, preserving each "
            "name's return distribution and the sample size while cutting the deal→outcome "
            "link. Crucially, the desk's **Real** bar demands the effect survive a "
            "**sub-era split** and not hang on a couple of names — the two cuts that "
            "decide Weak-vs-Real here."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Deals.** {R['n_total']} hardcoded, {R['n_untradable']} untradable "
            f"cautionary tales excluded, {R['n_on_tape']} listed sponsors on tape "
            "(Comerica/CMA a named no-coverage drop).\n"
            "- **Tape.** SPY + sponsor total-return closes, 1997 → 2026-06-30 (as-of, last "
            "complete month).\n"
            "- **Headline.** Cross-event mean BHAR at W=252, one-sample + NW t, Wilson hit "
            "rate, 3,000-draw random-entry placebo.\n"
            "- **Robustness (the Real gate).** W=504 repeat; pre/post-2010 sub-era split; "
            "drop-the-worst-k and leave-one-out jackknife.\n"
            "- **Execution (overlay).** Short the sponsor / long SPY at the announcement's "
            "first close (zero look-ahead — the deal is public by that close); 2 legs × "
            "one-way cost × NAV on entry+exit + borrow on the short.\n"
            "- **Control.** Synthetic sponsor+SPY tapes, planted post-deal drift, "
            "martingale-corrected null; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline and its placebo\n\n"
            "One-sample t on the cross-event mean 1-year BHAR, a Wilson hit rate, and the "
            "random-entry null. In the notebook we run a lighter placebo (800 draws) and "
            "quote the canonical 3,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.car_stats(SPY, PRICES, TRADABLE, window=252)\n"
            "    print(f\"1-yr mean BHAR {cs['mean']*100:+.2f}%  one-sample t = {cs['t']:+.3f}\"\n"
            "          f\"  (NW t = {cs['t_nw']:+.3f}, n={cs['n']})\")\n"
            "    wlo, whi = cs['wilson']\n"
            "    print(f\"hit rate (sponsor < SPY): {cs['hit']}/{cs['n']} = \"\n"
            "          f\"{cs['hit_rate']*100:.1f}%  (Wilson [{wlo*100:.1f}%, {whi*100:.1f}%])\")\n"
            "    pl = st.placebo_pvalue(SPY, PRICES, TRADABLE, window=252, n_draws=800, seed=845)\n"
            "    obs, draws = pl['obs'], pl['draws']\n"
            "else:\n"
            "    obs = R['w1_mean'] / 100\n"
            "    rng = np.random.default_rng(845)\n"
            "    draws = rng.normal(R['w1_pl_mean']/100, R['w1_pl_sd']/100, 800)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws * 100, bins=40, color=GREY, alpha=.85,\n"
            "        label='null: random entry dates on the same 28 names (light run)')\n"
            "ax.axvline(obs * 100, c=RED, lw=2.5, label=f'observed {obs*100:+.2f}%')\n"
            "ax.axvline(R['w1_pl_mean'], c='k', ls=':', lw=1, label=f\"placebo mean {R['w1_pl_mean']:+.1f}%\")\n"
            "ax.set_xlabel('mean 1-yr BHAR of a random-entry calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Observed sits in the left tail: canonical p = {R['w1_pl_p']:.3f}\")\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['w1_pl_mean']:+.2f}%, \"\n"
            "      f\"sd {R['w1_pl_sd']:.2f}%, p = {R['w1_pl_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['w1_mean']:.1f}%** sits in the left "
            f"tail of a null whose *mean is positive* (**{R['w1_pl_mean']:+.1f}%** — these "
            f"large-cap survivors normally beat SPY at a random entry), so a random "
            f"calendar produces a drop this large only **{R['w1_pl_p']*100:.1f}%** of the "
            f"time. With one-sample t = **{R['w1_t']:.2f}**, NW t = **{R['w1_nw']:.2f}** "
            f"and a {R['w1_hit_pct']:.0f}% hit rate, H₁ clears the |t| ≥ 2 bar on its own. "
            "Now the robustness that decides Weak vs Real."
        ),
        md(
            "### 4b · Horizon robustness — H₂, part 1\n\n"
            "Repeat at the 2-year window. A real hubris effect should, if anything, "
            "*compound*; a crisis-timing artifact should wash out as the crisis passes."
        ),
        code(
            "windows = [('1-year', 252), ('2-year', 504)]\n"
            "means, ts, ps = [], [], []\n"
            "for label, W in windows:\n"
            "    if HAVE_REAL:\n"
            "        cs = st.car_stats(SPY, PRICES, TRADABLE, window=W)\n"
            "        means.append(cs['mean']*100); ts.append(cs['t'])\n"
            "    else:\n"
            "        means.append(R['w1_mean'] if W==252 else R['w2_mean'])\n"
            "        ts.append(R['w1_t'] if W==252 else R['w2_t'])\n"
            "    ps.append(R['w1_pl_p'] if W==252 else R['w2_pl_p'])\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "bars = ax.bar([w[0] for w in windows], means,\n"
            "              color=[RED if abs(t)>=2 else AMBER for t in ts], width=.5)\n"
            "for i, (m, t, p) in enumerate(zip(means, ts, ps)):\n"
            "    ax.annotate(f'{m:+.1f}%\\nt={t:+.2f}\\np={p:.3f}', (i, m), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean sponsor BHAR vs S&P (%)')\n"
            "ax.set_title('Significant at 1yr (red), NOT at 2yr (amber) — it washes out')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1yr t =', round(ts[0],2), '| 2yr t =', round(ts[1],2))"
        ),
        md(
            f"> 💡 In plain words: at 2 years the effect is **{R['w2_mean']:.1f}%** but "
            f"t = **{R['w2_t']:.2f}** — below the bar. It does not compound; it fades. "
            "That is the first crack in H₂ (a genuine hubris-driven derating of an "
            "over-investing firm would not politely reverse within the second year)."
        ),
        md(
            "### 4c · Sub-era split & tail jackknife — H₂ part 2 and H₃, the demotion\n\n"
            "The two cuts that separate Weak from Real: does it hold **before and after "
            "2010**, and does it survive removing the worst couple of names?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    es = st.era_split(SPY, PRICES, TRADABLE, window=252)\n"
            "    em = [es['pre']['mean']*100, es['post']['mean']*100]\n"
            "    et = [es['pre']['t'], es['post']['t']]\n"
            "    _, kept = st.stack_bhar(SPY, PRICES, TRADABLE, window=252)\n"
            "    bs = np.sort(kept['bhar'].to_numpy())\n"
            "    ks = list(range(5)); dt = []\n"
            "    for k in ks:\n"
            "        _, t = st.one_sample_t(bs[k:]); dt.append(t)\n"
            "else:\n"
            "    em = [R['w1_pre_mean'], R['w1_post_mean']]; et = [R['w1_pre_t'], R['w1_post_t']]\n"
            "    ks = sorted(R['dropworst']); dt = [R['dropworst'][k][1] for k in ks]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.bar(['pre-2010\\n(n=12)', 'post-2010\\n(n=16)'], em, color=[GREY, RED], width=.55)\n"
            "for i, (v, t) in enumerate(zip(em, et)):\n"
            "    a1.annotate(f'{v:+.1f}%\\nt={t:+.2f}', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean 1-yr BHAR (%)')\n"
            "a1.set_title('Entirely a post-2010 effect (H2 fails)')\n"
            "a2.plot(ks, dt, 'o-', color=RED, lw=2)\n"
            "a2.axhline(-2, ls='--', c=GREY); a2.axhline(0, c='k', lw=.8)\n"
            "a2.annotate('desk bar', (3, -2), va='bottom', color=GREY, fontsize=8)\n"
            "a2.set_xlabel('# worst names removed'); a2.set_ylabel('one-sample t')\n"
            "a2.set_title('Drop the 2 COVID names -> t under the bar (H3 fails)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pre/post-2010 t:', round(et[0],2), round(et[1],2))\n"
            "print('t dropping 0..4 worst:', [round(x,2) for x in dt])\n"
            "print('leave-one-out t range:', R['loo'])"
        ),
        md(
            f"> 💡 In plain words: **pre-2010 t = {R['w1_pre_t']:.2f}** (nothing) vs "
            f"**post-2010 t = {R['w1_post_t']:.2f}** — the whole effect is one era. And "
            f"the tail jackknife: removing the two worst names (Caesars, Allegiant — both "
            f"COVID casualties) takes t from {R['dropworst'][0][1]:.2f} to "
            f"**{R['dropworst'][2][1]:.2f}**, under the bar. Leave-one-out t stays in "
            f"[{R['loo'][0]:.2f}, {R['loo'][1]:.2f}] (no *single* name flips it), but a "
            "law that dies on removing two names in one decade is **Weak**, not Real. "
            "Six of the sixteen post-2010 deals cluster in 2019–2021, right before COVID — "
            "hubris and crisis-timing luck are not separable at n=28."
        ),
        md(
            "### 4d · The tradable overlay — H₄, a costed short\n\n"
            "Short the sponsor, long SPY for the window; 2 legs × one-way cost × NAV on "
            "entry+exit + borrow on the short. If the curse is real, this earns +BHAR."
        ),
        code(
            "rows = []\n"
            "for label, W in [('1-year', 252), ('2-year', 504)]:\n"
            "    if HAVE_REAL:\n"
            "        ov = st.curse_overlay(SPY, PRICES, TRADABLE, window=W,\n"
            "                              cost_bps=5.0, borrow_bps_yr=100.0)\n"
            "        rows.append((label, ov['gross_mean']*100, ov['net_mean']*100,\n"
            "                     ov['t_net'], ov['win_rate']*100))\n"
            "    else:\n"
            "        if W == 252:\n"
            "            rows.append((label, R['w1_ov_gross'], R['w1_ov_net'], R['w1_ov_t'], R['w1_ov_win']))\n"
            "        else:\n"
            "            rows.append((label, R['w2_ov_gross'], R['w2_ov_net'], R['w2_ov_t'], R['w2_ov_win']))\n"
            "import pandas as pd\n"
            "df = pd.DataFrame(rows, columns=['window', 'gross %', 'net %', 't(net)', 'win %'])\n"
            "print(df.to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "x = np.arange(len(rows)); w = .35\n"
            "ax.bar(x - w/2, [r[1] for r in rows], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, [r[2] for r in rows], w, color=AMBER, label='net (5bps+borrow)')\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f't={r[3]:+.2f}', (i + w/2, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])\n"
            "ax.set_ylabel('short-sponsor/long-SPY return per deal (%)')\n"
            "ax.set_title('Positive at 1yr (t=2.3) but Fragile — dies at 2yr, tail-driven')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the 1-year short nets **{R['w1_ov_net']:+.1f}%** per "
            f"deal (t = {R['w1_ov_t']:.2f}) and survives costs — but it *is* the same "
            f"fragile 1-year signal, it dies at 2 years (t = {R['w2_ov_t']:.2f}), and its "
            "payoff comes from shorting exactly the names that are hardest and priciest to "
            "borrow (Caesars, crypto-adjacent, high-short-interest) in exactly the crises "
            "that pay it. Real borrow on those dwarfs the 100 bps/yr charged here. With ~1 "
            "independent bet per year, H₄ is **Fragile**, not investable."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic sponsor+SPY tapes with a TUNABLE planted post-deal drift, "
            "martingale-corrected so the null BHAR is mean-zero. The null (edge=0) is "
            "checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    spy_s, pr_s, ev_s = data.synthetic_world(edge=0.0, seed=845 + s_)\n"
            "    null_ts.append(st.synthetic_detect(spy_s, pr_s, ev_s, window=252)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "spy_s, pr_s, ev_s = data.synthetic_world(edge=-0.25, seed=845)\n"
            "planted_t = st.synthetic_detect(spy_s, pr_s, ev_s, window=252)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted curse edge=-25%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (cross-event BHAR)')\n"
            "ax.set_title('Control: null sits at zero; a planted curse lights up negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the "
            f"bar {R['syn_null_fire']}/20 (~the 5% you'd expect); a planted −25% curse "
            f"reads t = {R['syn_planted_t']:.2f}, correctly negative. The machinery is "
            "unbiased — the real-tape t = −2.54 is a genuine reading, and its **failure** "
            "of the robustness cuts is equally genuine. *(A faithful-engine / power check "
            "only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 1-yr sponsor BHAR **{R['w1_mean']:.1f}%**, one-sample "
            f"t = **{R['w1_t']:.2f}** (NW {R['w1_nw']:.2f}), hit {R['w1_hit_pct']:.0f}% "
            f"(Wilson [{R['w1_wilson'][0]:.1f}%, {R['w1_wilson'][1]:.1f}%]), placebo "
            f"p = **{R['w1_pl_p']:.3f}**. Real and correctly signed — more than the "
            f"Enron/FTX cherry-pick — but it **fails every Real gate**: 2-yr "
            f"t = {R['w2_t']:.2f}, pre-2010 t = {R['w1_pre_t']:.2f}, drop-2-names "
            f"t = {R['dropworst'][2][1]:.2f}. n=28, survivorship-trimmed, a 2019–2021 "
            "crisis cluster.\n"
            f"- **Tradability `FRAGILE`** — short-sponsor/long-SPY nets "
            f"**{R['w1_ov_net']:+.1f}%**/yr (t = {R['w1_ov_t']:.2f}) but dies at 2yr "
            f"(t = {R['w2_ov_t']:.2f}), rides the same tail, and needs to short the "
            "least-borrowable names in a crisis. ~1 bet/year.\n"
            "- **\"Is the naming-rights curse real?\" `MIXED`** — genuinely more than the "
            "anecdote (even survivors lag, placebo-significant) but not the robust law the "
            "legend implies: one era, a fat left tail of crisis-timed cyclicals, on a "
            "sample that already excludes the very bankruptcies (Enron, WorldCom) that "
            "started it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Identification is the honest limit.** With n=28 and a 2019–2021 cluster "
            "that ran into COVID, \"hubris predicts underperformance\" and \"cyclical "
            "sponsors signed pre-crisis\" are observationally equivalent here. A matched "
            "control (same-sector, same-size non-sponsors) and many more cycles would be "
            "the real test.\n"
            "- **Survivorship makes this finding *conservative*.** Because the actual "
            "bankruptcies are excluded, a true curse would be worse than −11% — a rare "
            "case where the bias understates rather than inflates.\n"
            "- **Dedup map:** [160-skyscraper-curse](../../160-skyscraper-curse/) (a "
            "*building*, macro-timing), [746-hq-relocation](../../746-hq-relocation/) (a "
            "*capex/HQ* move) and [722-logo-rebrand](../../722-logo-rebrand/) (a *rebrand* "
            "of the firm's own identity) — same corporate-vanity family, different "
            "trophies and event sets.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
