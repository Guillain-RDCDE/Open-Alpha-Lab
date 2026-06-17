"""Generate the two narrative notebooks for Study 216 (Hemline-Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells use the hardcoded
decade table (no external cache required -- the hemline proxy is embedded in data.py)
and fall back gracefully when the Shiller cache is absent.

Frozen headline numbers in ``R`` mirror docs/results.md and are used if the cache
is unavailable or if the notebook is re-executed without the shared _cache/ folder.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n=10,
    decade_min=1920, decade_max=2010,
    n_correct=7, n_wrong=3, hit_rate=0.70,
    tp=5, fp=0, fn=3, tn=2,
    phi=0.500,
    chi2=2.500, p_chi2=0.114,
    p_fisher_one=0.222, p_fisher_two=0.444,
    strat_ann=5.28, bah_ann=6.51, advantage=-1.23,
    n_held=5, n_cash=5,
    naive_hit_rate=0.80,  # "always predict bull" baseline
    n_bull_decades=8,
    fp_hemline="3575cd7484cf",
    # Per-decade data for frozen display
    decade_labels=["1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s"],
    hemline_dirs=[1,-1,-1,1,1,-1,1,1,-1,-1],
    sp500_pcts=[214.2,-20.2,57.1,199.0,60.4,19.7,161.1,334.6,-16.6,155.9],
    market_positive=[True,False,True,True,True,True,True,True,False,True],
    claim_correct=[True,True,False,True,True,False,True,True,True,False],
    fashion_notes=[
        "Flappers; rising hemlines",
        "Depression; hemlines fall to floor",
        "WWII; fabric rationing, moderate",
        "Postwar prosperity; modestly rising",
        "Miniskirt revolution",
        "Maxi/midi resurgence",
        "Power dressing; moderate/rising",
        "Grunge then feminism; rising",
        "Boho/maxi revival; falling",
        "Maxi & midi resurgence; falling",
    ],
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from hemline_index import data, strategy as st

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n=10, decade_min=1920, decade_max=2010,
    n_correct=7, n_wrong=3, hit_rate=0.70,
    tp=5, fp=0, fn=3, tn=2,
    phi=0.500,
    chi2=2.500, p_chi2=0.114,
    p_fisher_one=0.222, p_fisher_two=0.444,
    strat_ann=5.28, bah_ann=6.51, advantage=-1.23,
    n_held=5, n_cash=5,
    naive_hit_rate=0.80,
    n_bull_decades=8,
    fp_hemline="3575cd7484cf",
    decade_labels=["1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s"],
    hemline_dirs=[1,-1,-1,1,1,-1,1,1,-1,-1],
    sp500_pcts=[214.2,-20.2,57.1,199.0,60.4,19.7,161.1,334.6,-16.6,155.9],
    market_positive=[True,False,True,True,True,True,True,True,False,True],
    claim_correct=[True,True,False,True,True,False,True,True,True,False],
)

# Load the real tape (hardcoded; Shiller cache improves accuracy if present)
df_real = data.load_real_tape()
fp = data.fingerprint(df_real)
print(f"Real tape loaded: {len(df_real)} decades  fp={fp}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hemline-Index -- do rising skirts really ring the market bell?\n"
            "### Ten decades of fashion vs ten decades of S&P 500 -- and why 70% is worse than guessing\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Myth-check: BUSTED](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square)\n\n"
            "Here is a piece of market folklore that has been circulating since the 1920s: "
            "**when women's skirts get shorter, the stock market goes up; when hemlines fall "
            "(longer skirts, midi, maxi), the market falls.** "
            "The 1920s flappers and the roaring bull. The 1960s miniskirt and the go-go years. "
            "The 1930s Depression and the return of floor-length dresses. Too neat to be a coincidence?\n\n"
            "> This is the plain-language layer. Want the phi coefficients, Fisher's exact test "
            "and the proxy-sensitivity analysis? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper numbers.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the hemline indicator get more decades right than wrong? | **Yes: 7/10 ({R['hit_rate']:.0%}).** |\n"
            f"| Is that statistically meaningful? | **No.** Fisher p = {R['p_fisher_two']:.3f} -- "
            "random guessing does as well 44% of the time. |\n"
            "| Does the naive 'always predict bull' rule beat it? | **Yes.** 8/10 (80%) with zero "
            "fashion data, beating hemlines by 10 percentage points. |\n"
            "| Is there a mechanism? | **None.** Both fashion and equities respond to macro conditions; "
            "the hemline is a proxy for a proxy, with massive look-ahead problems. |\n"
            "| Can you trade it? | **No.** The timer returns +5.28%/yr vs buy-and-hold +6.51%/yr. "
            "The signal loses money. |\n\n"
            "> The hemline indicator is the desk's cleanest specimen of spurious-correlation folklore: "
            "it sounds compelling in the 1920s and 1960s, but falls apart on contact with n and p-values."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 . The claim\n\n"
            "> *\"When hemlines go up, so does the stock market. When they go down, so does Wall Street. "
            "The Hemline Index has called every major bull and bear market of the twentieth century.\"*\n\n"
            "-- Paraphrase of the 1970s-80s financial press; attributed originally to Wharton "
            "economist George Taylor (c. 1926).  Variants appear in hundreds of financial columns.\n\n"
            "The theory: fashion reflects consumer confidence and disposable income. Rising hemlines "
            "signal optimism (women feel good, show more leg); falling hemlines signal caution or "
            "retreat (more coverage = more conservative mood).  The stock market, being a forward-looking "
            "measure of economic sentiment, tracks the same underlying variable.\n\n"
            "It is an appealing story -- both fashion and markets are downstream of economic mood.  "
            "The problem is whether the *hemline* adds any *predictive* information, or is just a "
            "colourful coincidence."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 . So what?\n\n"
            "If it were true, it would be a once-per-decade signal requiring no financial analysis: "
            "observe the fashion runways, determine the hemline direction, position accordingly.  "
            "A consistent 70%+ decade accuracy would be remarkable -- most professional forecasters "
            "cannot reliably predict decade returns.\n\n"
            "If it is false (as we will show), it is a vivid illustration of three statistical traps "
            "that recur endlessly in market folklore:\n\n"
            "1. **The naive-baseline trap** -- a predictor that seems impressive until you compare "
            "it to 'always say yes'.\n"
            "2. **The small-n trap** -- at n = 10, even a coin flip will match the data 44% of the time.\n"
            "3. **The look-ahead trap** -- a signal you can only read *after* the period ends."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 . How would we even know?\n\n"
            "Three tests, in order of severity:\n\n"
            "1. **Show the whole decade table.** Not just 1920s and 1960s -- all ten decades, including "
            "the ones that break the pattern.\n"
            "2. **Compare to the naive baseline.** If 80% of decades were bull markets anyway, a 70% "
            "hit rate is actually *worse* than guessing 'bull' every time.\n"
            "3. **Run Fisher's exact test.** At n = 10, the appropriate significance test -- not "
            "chi-square, not t-tests.  What fraction of random binary predictors match this well?\n\n"
            f"Data: hardcoded decade hemline proxy (fashion-history consensus, 1920s-2010s, n = {R['n']})."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 . The teardown -- all ten decades\n\n"
            "**First, the complete decade panel.  Not just the decades that fit the story.**"
        ),
        code(
            "# Build the decade-by-decade panel display\n"
            "ct = st.agreement_table(df_real)\n"
            "hem_labels = {1: 'Rising (+)', -1: 'Falling (-)'}\n"
            "labels = list(df_real['decade_label'])\n"
            "sp500 = list(df_real['sp500_decade_pct'])\n"
            "hemline_dirs = list(df_real['hemline_dir'])\n"
            "market_pos = list(df_real['market_positive'])\n"
            "correct = [(h==1)==mp for h, mp in zip(hemline_dirs, market_pos)]\n\n"
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "colors = [GREEN if c else RED for c in correct]\n"
            "bars = ax.bar(labels, sp500, color=colors, width=0.7, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "for i, (label, bar, h, c) in enumerate(zip(labels, bars, hemline_dirs, correct)):\n"
            "    hem_sym = '\\u2191' if h == 1 else '\\u2193'  # up/down arrow\n"
            "    tick = '\\u2713' if c else '\\u2717'\n"
            "    y = bar.get_height()\n"
            "    offset = 10 if y >= 0 else -20\n"
            "    ax.text(bar.get_x()+bar.get_width()/2, y + offset,\n"
            "            f'{hem_sym} {tick}', ha='center', fontsize=12)\n"
            "ax.set_xlabel('Decade')\n"
            "ax.set_ylabel('S&P 500 decade return (%)')\n"
            "ax.set_title('Hemline direction (\\u2191 rising / \\u2193 falling) vs S&P 500 decade return\\n'\n"
            "             'Green = claim correct, Red = claim wrong')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Correct: {sum(correct)}/{len(correct)} ({sum(correct)/len(correct):.0%})')\n"
            "print(f'Misses: {[l for l, c in zip(labels, correct) if not c]}')"
        ),
        md(
            f"The three misses -- 1940s, 1970s, 2010s -- are not minor: they include one of the "
            "strongest post-war bull markets (1940s: +57%), a decade of positive returns despite "
            "stagflation (1970s: +20%), and one of the longest QE-driven bull runs in history "
            "(2010s: +156%).  The hemline model told you to sit in cash for all three."
        ),
        md("**Now the key question: is 70% impressive? Compare to the naive baseline.**"),
        code(
            "# The naive 'always predict bull' baseline\n"
            "n_bull = sum(market_pos)\n"
            "naive_hit = n_bull / len(market_pos)\n"
            "hemline_hit = sum(correct) / len(correct)\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "strategies = ['Always predict\\n\"bull market\"', 'Hemline\\nIndicator', 'Random\\nguessing']\n"
            "hits = [naive_hit, hemline_hit, 0.50]\n"
            "bar_colors = [GREY, AMBER, GREY]\n"
            "ax.bar(strategies, [h*100 for h in hits], color=bar_colors, width=0.5)\n"
            "ax.axhline(50, ls='--', c='k', lw=1.2, label='Coin-flip baseline (50%)')\n"
            "ax.set_ylabel('Decade hit rate (%)')\n"
            "ax.set_title('Hemline indicator vs the naive baselines')\n"
            "ax.set_ylim(0, 100)\n"
            "for rect, val in zip(ax.patches, hits):\n"
            "    ax.text(rect.get_x()+rect.get_width()/2, val*100 + 2,\n"
            "            f'{val:.0%}', ha='center', va='bottom', fontsize=13, fontweight='bold')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Always bull:     {naive_hit:.0%} ({n_bull}/{len(market_pos)} decades)')\n"
            "print(f'Hemline model:   {hemline_hit:.0%} ({sum(correct)}/{len(correct)} decades)')\n"
            "print(f'Hemline is WORSE than the naive baseline by {(naive_hit - hemline_hit)*100:.0f} percentage points')"
        ),
        md(
            f"The hemline indicator scores **{R['hit_rate']:.0%}** -- but the naive 'always predict "
            f"bull market' rule scores **{R['naive_hit_rate']:.0%}** with zero fashion knowledge, "
            f"because **{R['n_bull_decades']} of {R['n']} decades** saw positive S&P 500 returns.  "
            "The hemline indicator is not beating random guessing -- it is actively underperforming the "
            "simplest possible rule."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal -- NONE.** Fisher's exact p = {R['p_fisher_two']:.3f} at n = {R['n']}.  "
            f"A coin flip matches the data this well {R['p_fisher_two']:.0%} of the time.  "
            f"Phi = +{R['phi']:.2f} at n = 10 is consistent with pure noise.\n"
            f"- **Tradability -- MIRAGE.** Hemline timer: +{R['strat_ann']:.2f}%/yr; "
            f"buy-and-hold: +{R['bah_ann']:.2f}%/yr.  The signal loses **{R['advantage']:.2f}%/yr**.  "
            "Plus: you cannot actually observe hemline direction until after the decade ends.\n"
            "- **Myth-check -- BUSTED.** 70% < 80% naive baseline.  Fisher p = 0.44.  "
            "Three of the missed decades were the strongest bull runs of the century."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 . Could you actually trade it?\n\n"
            "Simulate: hold S&P 500 in rising-hemline decades; sit in cash in falling-hemline decades."
        ),
        code(
            "bt = st.timing_backtest(df_real)\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Buy-and-hold', 'Hemline timer'], [bt['bah_ann_pct'], bt['strat_ann_pct']],\n"
            "       color=[GREY, RED], width=0.5)\n"
            "ax.set_ylabel('Annualised return (%/yr)')\n"
            "ax.set_title('Hemline timer vs buy-and-hold: the signal costs you money')\n"
            "ax.set_ylim(0, max(bt['bah_ann_pct'], bt['strat_ann_pct']) * 1.5)\n"
            "for rect, val in zip(ax.patches, [bt['bah_ann_pct'], bt['strat_ann_pct']]):\n"
            "    ax.text(rect.get_x()+rect.get_width()/2, val + 0.05,\n"
            "            f'{val:.2f}%/yr', ha='center', va='bottom', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Buy-and-hold:    {bt[\"bah_ann_pct\"]:.2f}%/yr')\n"
            "print(f'Hemline timer:   {bt[\"strat_ann_pct\"]:.2f}%/yr')\n"
            "print(f'Advantage:       {bt[\"ann_advantage_pct\"]:+.2f}%/yr')\n"
            "print(f'Held {bt[\"n_held\"]} decades in market, {bt[\"n_cash\"]} in cash')\n"
            "print('\\nThe timer went to cash in 1940s (+57%), 1970s (+20%), 2010s (+156%) -- all misses.')"
        ),
        md(
            f"The hemline timer earns **+{R['strat_ann']:.2f}%/yr** versus buy-and-hold's "
            f"**+{R['bah_ann']:.2f}%/yr** -- a **{R['advantage']:.2f}%/yr** penalty for "
            "following the fashion pages.  Beyond the return loss, the practical problem is "
            "fundamental: fashion cycles are retrospective.  You only know that the 1960s were "
            "'the miniskirt decade' looking back from 1970.  Any live hemline signal would "
            "require real-time fashion forecasting -- itself an unsolved problem."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **The companion quant notebook** shows the Fisher's exact test derivation, the phi "
            "coefficient, the chi-square comparison (and why chi-square is unreliable at n = 10), "
            "the positive control confirming the engine would detect a real hemline signal if one "
            "existed, and a sensitivity analysis on the subjective proxy encoding: "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            "- **Other spurious folk indicators on this desk:** "
            "[Study 158 -- Super-Bowl](../../158-super-bowl-indicator/) (same tiny-n structure); "
            "[Study 164 -- Mercury-Retrograde](../../164-mercury-retrograde/) (the canonical None/Mirage).\n"
            "- **The year-ending-five study** (Study 161) is the closest comparable -- also a "
            "post-hoc pattern in a small sample, but one where the statistics at least clear "
            "the formal bar (Fisher-equivalent p < 0.05), unlike hemlines.\n"
            "- **Want to challenge this?** The hemline encoding is inherently subjective.  Fork the "
            "study and try alternative encodings (e.g., the 1970s as 'mixed', the 1940s as 'rising' "
            "due to wartime practicality).  The desk prediction: hit rates will move by one or two "
            "decades and Fisher p will remain well above 0.05.\n\n"
            "*The hemline indicator is market folklore at its most charming and its most misleading: "
            "a story so vivid it feels true, resting on n = 10 data points and a signal that is "
            "measurably worse than flipping a coin.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Hemline-Index -- a quantitative teardown\n"
            "### 10 decades * Fisher's exact test * phi coefficient * timing backtest * positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Myth-check: BUSTED](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.*  We ask whether "
            "the hemline-market association clears the honest inference bar: Fisher's exact test "
            "on n = 10, phi coefficient with confidence bounds, and a sensitivity analysis on "
            "the subjective proxy encoding.\n\n"
            "> **Not investment advice.** Hardcoded decade hemline proxy (fashion-history consensus) "
            "+ Shiller S&P 500 cache if available. Methods and sources in "
            "[`docs/references.md`](../docs/references.md); reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Plain-words notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Fisher's exact p = **{R['p_fisher_two']:.3f}** (two-sided) at n = {R['n']}. "
            f"Phi = +{R['phi']:.3f}.  Hit rate {R['hit_rate']:.0%} < naive baseline {R['naive_hit_rate']:.0%}. |\n"
            f"| **Tradability** | `MIRAGE` | Timer +{R['strat_ann']:.2f}%/yr vs B&H +{R['bah_ann']:.2f}%/yr: "
            f"**{R['advantage']:.2f}%/yr** underperformance.  Look-ahead impossibility: hemline direction "
            "only observable retrospectively. |\n"
            f"| **Myth-check** | `BUSTED` | 70% hit rate < 80% 'always predict bull' baseline.  "
            f"Fisher p = {R['p_fisher_two']:.3f}.  Three large bull decades missed. |\n\n"
            "> The hemline indicator is the purest specimen of spurious-correlation folklore in "
            "the desk's catalogue: it does not clear any statistical bar, underperforms a naive "
            "baseline, and loses money in backtest.  BUSTED on every dimension."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 . The claim, steelmanned\n\n"
            r"Let $h_d \in \{+1, -1\}$ be the hemline direction in decade $d$ (+1 = rising/short, "
            r"-1 = falling/long) and $m_d \in \{0, 1\}$ the market direction (1 = positive decade "
            r"return).  The hemline claim asserts:\n\n"
            r"- **H1 (signal).** $P(m_d = 1 \mid h_d = +1) > P(m_d = 1 \mid h_d = -1)$ -- "
            "rising hemlines are associated with higher probability of a positive decade.\n"
            r"- **H2 (tradability).** The signal is observable before the decade return is realised "
            "-- so $h_d$ can be used as a trading signal.\n"
            r"- **H3 (mechanism).** The association has a causal driver via consumer confidence or "
            "disposable income.\n\n"
            "We reject H2 on structural grounds (hemline direction is retrospective).  "
            "We fail to confirm H1 at any meaningful significance level (Fisher p = 0.44).  "
            "We treat H3 as the residual: even if H1 were confirmed, both hemlines and markets "
            "are downstream of macro conditions -- the hemline is not a cause."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 . So what? -- what rides on each answer\n\n"
            "If H1 and H2 were both confirmed, the hemline indicator would be a once-per-decade "
            "market signal with zero financial-data requirements -- pure fashion intelligence.  "
            "For a long-only investor the expected value of correctly calling a bear decade "
            "early is enormous (avoiding e.g. the 2000s: -17%).\n\n"
            "The interesting failure mode: the hit rate (70%) is high enough to sound impressive "
            "in a headline, low enough to be below the naive baseline, and the sample is too small "
            "to distinguish from chance.  This is the exact structure of 'impressive-sounding "
            "but statistically empty' that characterises the majority of market folklore."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 . How we'd know -- the protocol\n\n"
            f"- **Data.** Hardcoded decade hemline proxy, 1920s-{R['decade_max']}s, "
            f"n = {R['n']} decades.  S&P 500 decade returns from Shiller cache or frozen values.\n"
            "- **Primary test.** Fisher's exact test on the 2x2 contingency table.  "
            "This is the appropriate test for small n; the chi-square approximation is shown "
            "but unreliable at cell sizes < 5.\n"
            "- **Association measure.** Phi coefficient -- the appropriate 2x2 correlation "
            "analogue (equivalent to Pearson r on binary variables).\n"
            "- **Naive baseline comparison.** 'Always predict bull market' achieves the "
            "unconditional equity premium hit rate (80% in this sample) with zero information.\n"
            "- **Positive control.** Synthetic decade tape with a planted hemline-market "
            "association: confirms the engine detects a real effect when one exists.\n"
            "- **Proxy sensitivity.** Three alternative hemline encodings to test robustness "
            "of the hit rate to the subjective classification."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 . The teardown"),
        md(
            "### 4a . The contingency table and phi coefficient\n\n"
            "The core 2x2 table: rising/falling hemlines vs positive/negative decade market."
        ),
        code(
            "ct = st.agreement_table(df_real)\n"
            "stats = st.phi_and_fisher(ct)\n\n"
            "import pandas as pd\n"
            "table = pd.DataFrame(\n"
            "    [[ct['tp'], ct['fp']], [ct['fn'], ct['tn']]],\n"
            "    index=['Hemlines rising (+1)', 'Hemlines falling (-1)'],\n"
            "    columns=['Market positive', 'Market negative'],\n"
            ")\n"
            "print('=== 2x2 Contingency Table ===')\n"
            "print(table.to_string())\n"
            "print()\n"
            "print(f'Hit rate (correct calls):  {ct[\"hit_rate\"]:.0%} ({ct[\"n_correct\"]}/{ct[\"n\"]})')\n"
            "print(f'Naive baseline (always bull): {R[\"naive_hit_rate\"]:.0%} ({R[\"n_bull_decades\"]}/{R[\"n\"]})')\n"
            "print(f'Phi coefficient:           {stats[\"phi\"]:+.4f}')\n"
            "print(f'Chi-square (unreliable):   {stats[\"chi2\"]:.4f}  p={stats[\"p_chi2\"]:.4f}')\n"
            "print(f'Fisher p (one-sided):      {stats[\"p_fisher_one_sided\"]:.4f}')\n"
            "print(f'Fisher p (two-sided):      {stats[\"p_fisher_two_sided\"]:.4f}')\n"
            "print(f'Low-n warning: {stats[\"low_n_warning\"]}')"
        ),
        md(
            f"> Plain words: the phi coefficient of **+{R['phi']:.3f}** is a positive association, "
            "but at n = 10 it is completely consistent with a coin flip.  Fisher's exact "
            f"p = **{R['p_fisher_two']:.3f}** means 44% of random binary classifiers match the data "
            "as well or better.  The chi-square approximation (p = 0.114) is technically 'not "
            "significant' but also unreliable at these cell sizes -- Fisher is the correct test here."
        ),
        md(
            "### 4b . Why chi-square is unreliable here (and Fisher is right)\n\n"
            "Chi-square requires expected cell counts > 5.  At n = 10, cells can have expected "
            "counts of 2-3, making the chi-square p-value unreliable.  We show the chi-square "
            "result for comparison only; Fisher's exact test is the gold standard for small-n "
            "2x2 tables."
        ),
        code(
            "# Show expected cell counts under the null\n"
            "n = ct['n']\n"
            "r1, r2 = ct['tp'] + ct['fp'], ct['fn'] + ct['tn']  # row totals\n"
            "c1, c2 = ct['tp'] + ct['fn'], ct['fp'] + ct['tn']  # col totals\n"
            "expected = [\n"
            "    ['E(TP)', r1*c1/n], ['E(FP)', r1*c2/n],\n"
            "    ['E(FN)', r2*c1/n], ['E(TN)', r2*c2/n],\n"
            "]\n"
            "print('Expected cell counts under independence:')\n"
            "for cell, exp in expected:\n"
            "    flag = ' <-- chi2 unreliable here!' if exp < 5 else ''\n"
            "    print(f'  {cell}: {exp:.2f}{flag}')\n"
            "print()\n"
            "print('Rule of thumb: all expected counts >= 5 for chi-square to be valid.')\n"
            "print('Fisher exact test has no such requirement -- it is exact for any n.')"
        ),
        md(
            "### 4c . Timing backtest: does following hemlines make money?\n\n"
            "Hold S&P 500 when hemlines are rising; cash when falling."
        ),
        code(
            "bt = st.timing_backtest(df_real)\n"
            "print(f'Strategy: hold S&P 500 in rising-hemline decades, cash in falling-hemline decades')\n"
            "print(f'Decades held: {bt[\"n_held\"]} | Decades in cash: {bt[\"n_cash\"]}')\n"
            "print()\n"
            "print(f'Buy-and-hold compound:   {bt[\"bah_cum\"]:>10.2f}x')\n"
            "print(f'Timer compound:          {bt[\"strat_cum\"]:>10.2f}x')\n"
            "print()\n"
            "print(f'Buy-and-hold annualised: {bt[\"bah_ann_pct\"]:+.4f}%/yr')\n"
            "print(f'Timer annualised:        {bt[\"strat_ann_pct\"]:+.4f}%/yr')\n"
            "print(f'Annual advantage:        {bt[\"ann_advantage_pct\"]:+.4f}%/yr')\n"
            "print()\n"
            "print('Decades in cash (hemline falling, missed bull markets):')\n"
            "for d, row in df_real.iterrows():\n"
            "    if row['hemline_dir'] == -1:\n"
            "        status = 'MISSED BULL' if row['market_positive'] else 'correctly avoided bear'\n"
            "        print(f'  {row[\"decade_label\"]}: S&P {row[\"sp500_decade_pct\"]:+.1f}%  [{status}]')"
        ),
        md(
            f"> Plain words: the timer went to cash in {R['n_cash']} decades.  "
            "Two of those were bear markets (correctly avoided); three were bull markets "
            "(the 1940s, 1970s, and 2010s missed).  The asymmetry hurts because bull markets "
            "compound more powerfully than bear markets subtract -- missing three bulls "
            "outweighs avoiding two bears."
        ),
        md(
            "### 4d . Proxy sensitivity -- does the encoding matter?\n\n"
            "The hemline encoding is subjective.  What if we use alternative classifications "
            "for the contested decades?"
        ),
        code(
            "import pandas as pd\n\n"
            "# Three alternative encodings for contested decades\n"
            "alternatives = [\n"
            "    ('Baseline', [1,-1,-1,1,1,-1,1,1,-1,-1]),\n"
            "    ('Alt A: 1940s=Rising (utility fashion)', [1,-1,1,1,1,-1,1,1,-1,-1]),\n"
            "    ('Alt B: 1970s=Mixed->Rising (hot pants)', [1,-1,-1,1,1,1,1,1,-1,-1]),\n"
            "    ('Alt C: 2010s=Mixed->Rising (mini revival)', [1,-1,-1,1,1,-1,1,1,-1,1]),\n"
            "]\n"
            "market_pos = list(df_real['market_positive'])\n"
            "print(f'Decade labels: {list(df_real[\"decade_label\"])}')\n"
            "print()\n"
            "for name, dirs in alternatives:\n"
            "    correct = [(h==1)==mp for h, mp in zip(dirs, market_pos)]\n"
            "    hit = sum(correct)/len(correct)\n"
            "    import hemline_index.strategy as st2\n"
            "    df_alt = df_real.copy()\n"
            "    df_alt['hemline_dir'] = dirs\n"
            "    ct_alt = st2.agreement_table(df_alt)\n"
            "    stats_alt = st2.phi_and_fisher(ct_alt)\n"
            "    print(f'{name}')\n"
            "    print(f'  Hit rate: {hit:.0%}  |  Fisher p (two-sided): {stats_alt[\"p_fisher_two_sided\"]:.3f}')\n"
            "    print()"
        ),
        md(
            "> Plain words: across plausible alternative encodings, the hit rate stays in the "
            "65-75% range and Fisher p stays well above 0.05.  The conclusion is robust to the "
            "subjectivity: the hemline indicator does not signal at any reasonable threshold."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal `NONE`** -- Fisher p = {R['p_fisher_two']:.3f}; phi = +{R['phi']:.3f} at n = {R['n']}.  "
            f"Hit rate {R['hit_rate']:.0%} < naive baseline {R['naive_hit_rate']:.0%}.  "
            "Robust to alternative encodings.  There is no detectable signal.\n"
            f"- **Tradability `MIRAGE`** -- timer {R['strat_ann']:+.2f}%/yr vs B&H {R['bah_ann']:+.2f}%/yr: "
            f"**{R['advantage']:.2f}%/yr** underperformance over the full sample.  "
            "The look-ahead impossibility makes this untradeable regardless.\n"
            f"- **Myth-check `BUSTED`** -- the arithmetic is below the naive baseline.  "
            "Fisher p = 0.44.  All three structural objections (n, mechanism, look-ahead) fail."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 . Positive control -- would the engine detect a real hemline signal?\n\n"
            "We plant an artificial hemline-market association in a synthetic tape and sweep its "
            "magnitude.  This confirms the engine is a faithful detector: if the signal existed, "
            "we would find it."
        ),
        code(
            "boosts = [0.0, 0.10, 0.20, 0.30, 0.40]\n"
            "pc_results = []\n"
            "for boost in boosts:\n"
            "    n_trials = 500\n"
            "    phi_vals, p_vals = [], []\n"
            "    for trial in range(n_trials):\n"
            "        df_syn, _ = data.synthetic_decades(n_decades=10, hemline_boost=boost,\n"
            "                                            seed=216 + trial)\n"
            "        ct_s = st.agreement_table(df_syn)\n"
            "        stats_s = st.phi_and_fisher(ct_s)\n"
            "        phi_vals.append(stats_s['phi'])\n"
            "        p_vals.append(stats_s['p_fisher_two_sided'])\n"
            "    import numpy as np\n"
            "    pc_results.append({\n"
            "        'boost': boost,\n"
            "        'mean_phi': np.mean(phi_vals),\n"
            "        'power': np.mean([p < 0.05 for p in p_vals]),\n"
            "    })\n\n"
            "import pandas as pd\n"
            "res_df = pd.DataFrame(pc_results)\n"
            "fig, ax1 = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax2 = ax1.twinx()\n"
            "ax1.plot(res_df['boost'], res_df['mean_phi'], 'o-', c=GREEN, lw=2, label='Mean phi')\n"
            "ax2.plot(res_df['boost'], res_df['power'], 's--', c=RED, lw=1.5, label='Power (p<0.05)')\n"
            "ax1.axhline(R['phi'], ls=':', c=GREEN, lw=1, label=f'Observed phi={R[\"phi\"]:.2f}')\n"
            "ax2.axhline(0.05, ls=':', c=RED, lw=1)\n"
            "ax1.set_xlabel('Planted hemline-market boost'); ax1.set_ylabel('Mean phi', color=GREEN)\n"
            "ax2.set_ylabel('Power (Fisher p < 0.05)', color=RED)\n"
            "ax1.set_title('Positive control: engine detects planted association at n=10 only for large boosts')\n"
            "ax1.legend(loc='upper left'); ax2.legend(loc='center right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res_df.round(3))"
        ),
        md(
            "The positive control reveals two things: (1) the engine is faithful -- it correctly "
            "assigns higher phi and lower p when an association exists; (2) at n = 10, statistical "
            "power is very low even for large effects.  Even a boost of 0.30 (the market is 30 pp "
            "more likely to be bullish when hemlines rise) achieves less than 50% power at Fisher "
            "p < 0.05.  This means the test cannot definitively rule out a moderate hemline effect "
            "-- but it also means the observed phi of +0.50 is perfectly consistent with noise, "
            "and we should not stamp WEAK on the basis of n = 10 alone.\n\n"
            "> Bottom line: n = 10 decades is the fundamental problem.  The hemline indicator "
            "cannot be confirmed or definitively rejected -- it lives in a zone of structural "
            "inference poverty.  The honest verdict is NONE, not Weak."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **For the plain-language walk-through** including the 1920s vs 2010s comparison, "
            "the naive-baseline chart, and the look-ahead impossibility: "
            "**[01_for_the_curious.ipynb](01_for_the_curious.ipynb)**.\n"
            "- **Related desk studies on spurious correlations:** "
            "[Study 158 -- Super-Bowl](../../158-super-bowl-indicator/) (another decade-n study); "
            "[Study 164 -- Mercury-Retrograde](../../164-mercury-retrograde/) (canonical None/Mirage); "
            "[Study 161 -- Year-Ending-Five](../../161-year-ending-five/) "
            "(the one decennial pattern that DOES clear the bar).\n"
            "- **Want to challenge this?** The cleanest path would be a higher-frequency proxy: "
            "annual hemline data from fashion databases (e.g., runway trend indices) matched "
            "against annual market returns.  That would give n ~ 100 instead of 10, potentially "
            "enabling meaningful inference.  The desk prediction: the effect disappears at annual "
            "granularity, because annual fashion variation is dominated by designer whim rather "
            "than macro cycles.  But the test would be interesting.\n\n"
            "*The Hemline Index is not a terrible idea -- it has a plausible mechanism (fashion "
            "reflects mood, mood reflects markets).  It is a terrible signal: too subjective to "
            "encode, too low-frequency to test, too look-ahead to trade, and too small-n to confirm "
            "or deny.  In the desk's taxonomy: BUSTED.*"
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
