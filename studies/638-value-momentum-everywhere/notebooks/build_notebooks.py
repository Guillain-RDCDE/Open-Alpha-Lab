"""Generate the two narrative notebooks for Study 638 (Value-Momentum-Everywhere).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached sleeve
panels under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance country ETFs +
# G10 spot FX + shared continuous-futures cache; monthly, as-of 2026-05-31).
R = dict(
    as_of="2026-05-31", fingerprint="2cd41e1339cb",
    # per sleeve: (name, n_assets, val_sr, val_t, mom_sr, mom_t, combo_sr, combo_t, rho, n, start)
    sleeves=[
        ("EQ", 13, -0.25, -1.35, -0.01, -0.06, -0.16, -0.73, -0.16, 349, "1997-05"),
        ("FX", 9, 0.10, 0.46, -0.01, -0.07, 0.04, 0.20, -0.03, 270, "2002-09"),
        ("BOND", 3, -0.22, -1.19, -0.19, -0.97, -0.36, -1.63, -0.11, 203, "2001-11"),
        ("CMD", 8, 0.14, 0.67, -0.32, -1.72, -0.15, -0.86, 0.17, 296, "2001-10"),
    ],
    mean_rho=-0.03,
    # global: (leg, SR, t, mean bps/mo, n)
    glob=dict(val=(0.01, 0.03, 0.4, 248), mom=(-0.38, -1.95, -28.0, 296),
              combo=(-0.29, -1.49, -15.7, 296)),
    sub_pre=(-0.50, -1.64, 123), sub_post=(-0.13, -0.54, 174),
    # diversification arithmetic
    rho_global=0.145, pred_sr=-0.267, real_sr=-0.267,
    sr_val=0.01, sr_mom=-0.41,
    placebo=dict(mean_sharpe=0.01, sd=0.19, mean_t=0.07, frac=4, n_seeds=50),
    turnover=0.65,
    # costs: (one-way bps, gross %/yr, net %/yr, net SR, net t)
    costs=[(5.0, -1.88, -2.42, -0.37, -1.91), (10.0, -1.88, -2.81, -0.43, -2.21),
           (20.0, -1.88, -3.59, -0.55, -2.83)],
    # robustness: (variant, SR, t)
    robust=[("halves not thirds", -0.48, -2.40), ("mom 12-0 (no skip)", -0.34, -1.76),
            ("value incl. last year", -0.19, -0.88)],
    # synthetic: (label, val_sr, mom_sr, combo_sr, combo_t, rho, pred, realized)
    syn=[("null (no edges)", -0.04, -0.07, -0.08, -0.52, 0.03, -0.13, -0.13),
         ("planted VAL+MOM", 2.27, 1.49, 2.32, 14.12, 0.18, 2.51, 2.51)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Just_diversification_arithmetic%3F: Confirmed](https://img.shields.io/badge/Just_diversification_arithmetic%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from value_momentum_everywhere import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANELS = data.load_sleeves()
    SLEEVES = {k: st.sleeve_series(v) for k, v in PANELS.items()}
else:
    PANELS = SLEEVES = None
print("real sleeve cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    s_by = {s[0]: s for s in R["sleeves"]}
    cells = [
        md(
            "# Value + momentum everywhere — the famous \"free lunch\" combo, retested 🌍\n"
            "### Two legendary strategies that supposedly work in *every* market and hedge each "
            "other — do they still, on data anyone can download?\n\n"
            + BADGES +
            "In 2013 three of the most respected people in quantitative finance — Cliff Asness, Tobias "
            "Moskowitz and Lasse Pedersen — published *\"Value and Momentum Everywhere\"*: buy what's "
            "**cheap** (value) and what's **rising** (momentum), in stocks, countries, currencies, bonds "
            "*and* commodities, and because the two strategies zig when the other zags, the **50/50 "
            "blend** is the closest thing finance has to a free lunch. Their numbers were stunning — "
            "blended Sharpe ratios near 1, everywhere, for forty years.\n\n"
            "We rebuild the whole thing on **free data** — 13 country stock-market ETFs, 9 major "
            "currencies, 3 US Treasury futures and 8 commodity futures — and ask the only question that "
            "matters: **does the free lunch show up on the part of the tape you could actually have "
            "traded?**\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo and the diversification "
            "arithmetic? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data notes up front.** Our ETF panel is a *survivor* list (all still trade in 2026); "
            "FX is spot only (no carry); futures are continuous front-month series with known roll "
            "noise. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does value pay everywhere? | **Not here.** Across the four sleeves its Sharpe sits "
            f"between {min(s[2] for s in R['sleeves']):.2f} and {max(s[2] for s in R['sleeves']):.2f} "
            "— statistical zero in every one. |\n"
            f"| Does momentum pay everywhere? | **Not here.** Same story — the *best* sleeve is roughly "
            "flat, the worst (commodities) is outright negative. |\n"
            f"| Do they hedge each other? | **Barely.** The famous strong negative correlation (~−0.5 in "
            f"the paper) shows up as a feeble **{R['mean_rho']:+.2f}** average on our tape. |\n"
            f"| So does the 50/50 combo rescue it? | **No.** The everywhere-combo comes out at Sharpe "
            f"**{R['glob']['combo'][0]:+.2f}** (*t* = {R['glob']['combo'][1]:+.2f}) — the *wrong sign*. "
            "Blending can only average what the ingredients bring, and here they brought nothing. |\n\n"
            "> The paper's arithmetic is impeccable — diversification genuinely works. What's missing on "
            "our post-1997 free tape is the **raw material**: neither premium shows up to be blended."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Cheap assets beat expensive ones. Rising assets keep rising. This is true in stocks, "
            "country indices, currencies, bonds and commodities — and because value and momentum are "
            "negatively correlated, holding both is dramatically better than holding either.\"*\n\n"
            "That's Asness–Moskowitz–Pedersen (2013), *Journal of Finance* — 1972–2011 data, blended "
            "Sharpes near 1. It launched a thousand multi-factor funds. We test it the way a retail "
            "reader could: **value** = what fell over the past 5 years (skipping the last year), "
            "**momentum** = what rose over the past 12 months (skipping the last month), long the top "
            "third, short the bottom third, in each of four asset classes, rebalanced monthly."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is arguably **the** foundational claim of factor investing — if value and momentum "
            "pay everywhere and hedge each other, every portfolio should hold both, always. Hundreds of "
            "billions of dollars are allocated on that premise. But almost all of the evidence comes "
            "from *one* sample ending in 2011, and the desk has already caught the pieces decaying one "
            "by one: FX momentum ([147](../../147-fx-momentum/)) — gone; long-term reversal on stocks "
            "([196](../../196-long-term-reversal/)) — beta in disguise. The combo is the last line of "
            "defence: even weak ingredients, blended, might still clear the bar. That's what we test."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Four sleeves, one identical recipe:\n\n"
            "1. **Country stocks** — 13 country ETFs (US, Japan, Germany, UK, France, Italy, Spain, "
            "Australia, Canada, Hong Kong, Singapore, Sweden, Switzerland), total-return, since 1996.\n"
            "2. **Currencies** — 9 majors vs the dollar (spot).\n"
            "3. **Bonds** — 5-, 10- and 30-year US Treasury futures.\n"
            "4. **Commodities** — oil, gas, gold, silver, copper, corn, soybeans, wheat futures.\n\n"
            "Each month, in each sleeve: rank by the signal, go long the top third, short the bottom "
            "third, hold next month (that's the one-month execution lag — no peeking). Average the four "
            "sleeves into a **global value**, **global momentum** and **global 50/50 combo** portfolio. "
            "Then let a robust *t*-test judge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the ingredients.** Value and momentum Sharpe ratios, sleeve by sleeve. The paper "
            "says every bar should be comfortably positive."
        ),
        code(
            "names = [s[0] for s in R['sleeves']]\n"
            "if HAVE_REAL:\n"
            "    stats = {k: st.sleeve_stats(SLEEVES[k]) for k in names}\n"
            "    vals = [stats[k]['val']['sharpe'] for k in names]\n"
            "    moms = [stats[k]['mom']['sharpe'] for k in names]\n"
            "else:\n"
            "    vals = [s[2] for s in R['sleeves']]; moms = [s[4] for s in R['sleeves']]\n"
            "x = np.arange(len(names))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(x-0.2, vals, 0.38, color=AMBER, label='VALUE (5y reversal)')\n"
            "ax.bar(x+0.2, moms, 0.38, color=GREY, label='MOMENTUM (12-1)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Country\\nstocks','Currencies','Treasury\\nfutures','Commodity\\nfutures'])\n"
            "ax.set_ylabel('annualised Sharpe (gross, long/short)')\n"
            "ax.set_title('The ingredients: value and momentum, sleeve by sleeve (1997/2001 - 2026)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('VAL Sharpes:', [round(v,2) for v in vals]); print('MOM Sharpes:', [round(m,2) for m in moms])"
        ),
        md(
            f"Not one bar looks like the paper. Value tops out at Sharpe **{s_by['CMD'][2]:+.2f}** "
            f"(commodities) and bottoms at **{s_by['EQ'][2]:+.2f}** (country stocks); momentum's best is "
            f"flat and its worst — commodities — is **{s_by['CMD'][4]:+.2f}**. None of the eight legs is "
            "statistically distinguishable from zero.\n\n"
            "**Now the famous combo.** Average the four sleeves into global portfolios and cumulate a "
            "dollar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gv = st.global_series(SLEEVES, 'val'); gm = st.global_series(SLEEVES, 'mom')\n"
            "    gc = st.global_series(SLEEVES, 'combo')\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "    for s, cname, lab in ((gv, AMBER, 'global VALUE'), (gm, GREY, 'global MOMENTUM'),\n"
            "                          (gc, RED, 'global 50/50 COMBO')):\n"
            "        ax.plot((1+s).cumprod(), color=cname, lw=1.8, label=f'{lab}  (SR {st.sharpe(s):+.2f})')\n"
            "    ax.axhline(1, c='k', lw=.8)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log scale, gross)')\n"
            "    ax.set_title('The everywhere portfolios: the combo has nothing to combine')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache miss - frozen numbers:', R['glob'])\n"
            "print('global combo: SR %.2f  HAC t %.2f' % (R['glob']['combo'][0], R['glob']['combo'][1]))"
        ),
        md(
            f"A dollar in the global 50/50 combo *shrinks* — Sharpe **{R['glob']['combo'][0]:+.2f}**, "
            f"robust *t* = **{R['glob']['combo'][1]:+.2f}**, over {R['glob']['combo'][3]} months. And "
            "that's **gross**: no trading costs, no borrow on the short legs. Costs only push it further "
            "down.\n\n"
            "**But wait — wasn't the magic the *hedge*?** The paper's deepest claim is that value and "
            "momentum are strongly negatively correlated (~−0.5), so the blend is much better than the "
            "parts. Here's that correlation, sleeve by sleeve."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rhos = [st.sleeve_stats(SLEEVES[k])['rho'] for k in names]\n"
            "else:\n"
            "    rhos = [s[8] for s in R['sleeves']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(names, rhos, color=[GREEN if r < -0.3 else AMBER for r in rhos], width=.55)\n"
            "ax.axhline(-0.5, ls='--', c=GREY, label='the paper: ~ -0.5')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, r in enumerate(rhos): ax.annotate(f'{r:+.2f}', (i, r), ha='center',\n"
            "                                         va='top' if r < 0 else 'bottom')\n"
            "ax.set_ylabel('corr(value, momentum) monthly'); ax.set_ylim(-0.6, 0.3)\n"
            "ax.set_title('The famous hedge is barely there on this tape')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('per-sleeve rho:', [round(r,2) for r in rhos])"
        ),
        md(
            f"Average correlation **{R['mean_rho']:+.2f}** — a whisper of the paper's −0.5. With ~zero "
            "correlation, blending two strategies still helps (risk averages out) — but the *return* of "
            "the blend is just the average of the two returns. Average of zero and zero is zero.\n\n"
            "> 🔬 **For the quants:** the combo's Sharpe matches the two-asset diversification formula "
            "to the third decimal — see the quants notebook. There is no residual \"everywhere magic\" "
            "beyond arithmetic."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Forty years of literature (and a genuinely great paper) say the "
            f"premia are real; **our** free-data tape (1997/2001→2026) can't certify any of it — all "
            f"eight sleeve-legs are statistical zero and the global combo lands at Sharpe "
            f"**{R['glob']['combo'][0]:+.2f}** (*t* = {R['glob']['combo'][1]:+.2f}), the wrong sign.\n"
            "- **Tradability — Mirage.** The combo is negative **before** costs; after 10 bps one-way "
            "and short-leg borrow it's worse. There is nothing to harvest here.\n"
            "- **\"Just diversification arithmetic?\" — Confirmed.** The combo's Sharpe equals the "
            "textbook two-asset formula exactly — diversification works, but it can only blend what the "
            "ingredients deliver. (In [study 401](../../401-signal-stacking/) the same arithmetic "
            "multiplied *noise*; here it multiplies two once-real premia that this tape no longer pays.)"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Our tape is not their tape.** AMP had 1972–2011, dozens of markets per class, futures "
            "with proper roll handling, and value measures like book-to-market. We have 13 survivor "
            "ETFs, 9 spot pairs, 11 noisy continuous futures, 1997/2001–2026. Absence of evidence on a "
            "thin free tape is not proof the premia never existed — it *is* proof you couldn't have "
            "harvested them here.\n"
            "- **The dates rhyme with McLean–Pontiff.** Most of our sample is *post-publication* — "
            "exactly where documented premia historically fade. Our pre/post-2012 split shows nothing "
            "clears the bar in either half.\n"
            "- **The honest lesson survives.** Negative-to-zero correlation between strategies is real "
            "and valuable — *when the strategies themselves pay*. Hunt the ingredients, not the blender.\n\n"
            "*Think the free lunch is still being served somewhere? Bring a sleeve where value or "
            "momentum clears a HAC t of 2 on post-2012 free data — then we'll blend.*"
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
            "# Value-Momentum-Everywhere — a quantitative teardown 🔬\n"
            "### Four sleeves x two signals x one recipe · HAC t's on the global combo · the "
            "diversification-arithmetic decomposition · a 50-seed random-rank placebo · costs x "
            "turnover · a planted-effect synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Asness–Moskowitz–Pedersen (2013) claim (i) value and momentum premia in every asset "
            "class, (ii) negative VAL–MOM correlation everywhere, (iii) a 50/50 combo that dominates "
            "either leg. We rebuild all three on free data with one uniform construction and judge "
            "with autocorrelation-robust statistics.\n\n"
            "> ⚠️ **Data notes.** EQ = 13 country ETFs, total-return, *survivors* (named on the Signal "
            "axis); FX = G10 **spot** (price-only, no carry); BOND/CMD = Yahoo continuous front-month "
            "futures daily returns (roll noise, ±25% clip — the shared cache built by "
            "[study 31](../../31-trade-winds/)). As-of **" + R["as_of"] + "** (June 2026 partial, "
            "dropped). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Global 50/50 combo Sharpe **{R['glob']['combo'][0]:+.2f}**, HAC "
            f"**t = {R['glob']['combo'][1]:+.2f}** ({R['glob']['combo'][3]} months); no sleeve-leg "
            "clears \\|t\\| ≥ 2; the literature says real, this tape can't certify it. |\n"
            f"| **Tradability** | `MIRAGE` | Gross is already negative "
            f"({R['costs'][0][1]:+.1f}%/yr); at 10 bps one-way + EQ borrow the net is "
            f"**{R['costs'][1][2]:+.1f}%/yr**. Nothing to harvest. |\n"
            f"| **Just diversification arithmetic?** | `CONFIRMED` | Realized combo SR "
            f"**{R['real_sr']:+.3f}** vs two-asset-formula prediction **{R['pred_sr']:+.3f}** — "
            "identical; the blend adds nothing beyond the formula, and the ingredients are zero. |\n\n"
            "> 💡 In plain words: the blender works exactly as advertised — but on this tape somebody "
            "forgot to put fruit in it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For sleeve $s$ with assets $i$, month-end $t$:\n\n"
            "$$\\mathrm{VAL}_{i,t} = -\\!\\!\\prod_{k=t-59}^{t-12}\\!(1+r_{i,k}),\\qquad "
            "\\mathrm{MOM}_{i,t} = \\prod_{k=t-11}^{t-1}(1+r_{i,k})$$\n\n"
            "(the 5-year reversal skipping the momentum year — AMP's own non-stock value proxy — and "
            "the classic 12-1). Long the top third, short the bottom third, equal weight, hold month "
            "$t{+}1$ (the ONE execution lag). The combo averages the *weights* 50/50; the global "
            "portfolio averages live sleeves equally.\n\n"
            "- **H₁ (everywhere).** Each sleeve's VAL and MOM Sharpe > 0.\n"
            "- **H₂ (the hedge).** corr(VAL, MOM) strongly negative (~−0.5) within sleeves.\n"
            "- **H₃ (the free lunch).** The global 50/50 combo clears HAC *t* ≥ 2.\n\n"
            "We find **H₁ rejected on this tape** (all eight legs statistical zero), **H₂ mostly "
            "absent** (mean ρ ≈ 0), **H₃ rejected** (combo *t* negative). The verdict grades what the "
            "tape says, not what the literature deserves."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The combo claim is special because it is *arithmetically guaranteed* to look good **if** "
            "the ingredients are good: for a 50/50 mix of series with Sharpes $S_v, S_m$ and "
            "correlation $\\rho$ (equal vols),\n\n"
            "$$S_{combo} \\approx \\frac{S_v + S_m}{\\sqrt{2(1+\\rho)}}.$$\n\n"
            "With $\\rho = -0.5$ that's a **1.41×** multiplier on the average ingredient — AMP's free "
            "lunch. The third axis tests whether the realized combo does anything *beyond* this "
            "formula (it must not — and does not). The null twin is "
            "[401-signal-stacking](../../401-signal-stacking/): stacked **noise** obeys the same "
            "arithmetic and delivers nothing. The new question here: stack the two most *documented* "
            "premia in finance — does the free tape supply the ingredients? Survivorship (EQ ETF "
            "panel), spot-only FX and roll-noisy futures are named where they bite."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Sleeves.** EQ: 13 country ETFs (total-return, 1996→, survivors). FX: 9 G10 pairs vs "
            "USD (spot, price-only). BOND: ZF/ZN/ZB futures (excess). CMD: 8 commodity futures "
            "(excess). Monthly, as-of " + R["as_of"] + ".\n"
            "- **Signals.** VAL = 5y reversal (months t-59..t-12); MOM = 12-1 (t-11..t-1); complete "
            "windows required; computed from data through month-end *t* only.\n"
            "- **Books.** Top/bottom thirds, equal-weight, 100/100 (gross 2× NAV), rebalanced monthly, "
            "held month t+1 — **one** execution lag, documented.\n"
            "- **Inference.** HAC/Newey–West t's (Bartlett, automatic lag) on every monthly series; "
            "Sharpe races excess-vs-excess (L/S is self-financing; futures returns are excess).\n"
            "- **Placebo.** Random-rank signals through the identical pipeline, averaged over **50 "
            "seeds** (house rule ≥ 20).\n"
            "- **Costs.** One-way bps × traded notional; the EQ short leg pays 50 bps/yr borrow; "
            "5/10/20 bps scenarios.\n"
            "- **Splits & variants.** Pre/post-2012 (AMP's sample ends 2011 — a justified, "
            "non-snooped split); halves-not-thirds; 12-0 momentum; value including the last year.\n"
            "- **Control.** A 4-sleeve synthetic world with plantable VAL/MOM edges: the zero-edge "
            "null must stay flat, planted edges must light up and obey the diversification formula."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Eight legs, four sleeves — the ingredient matrix\n\n"
            "Per-sleeve Sharpe and HAC *t* for VAL, MOM, COMBO, plus the VAL–MOM correlation."
        ),
        code(
            "names = [s[0] for s in R['sleeves']]\n"
            "if HAVE_REAL:\n"
            "    stats = {k: st.sleeve_stats(SLEEVES[k]) for k in names}\n"
            "    rows = [(k, stats[k]['val']['sharpe'], stats[k]['val']['t'],\n"
            "             stats[k]['mom']['sharpe'], stats[k]['mom']['t'],\n"
            "             stats[k]['combo']['sharpe'], stats[k]['combo']['t'],\n"
            "             stats[k]['rho'], stats[k]['n']) for k in names]\n"
            "else:\n"
            "    rows = [(s[0], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]) for s in R['sleeves']]\n"
            "print(f\"{'sleeve':6s} {'VAL SR':>7s} {'t':>6s} {'MOM SR':>7s} {'t':>6s} \"\n"
            "      f\"{'COMBO':>7s} {'t':>6s} {'rho':>6s} {'n':>4s}\")\n"
            "for r in rows:\n"
            "    print(f'{r[0]:6s} {r[1]:+7.2f} {r[2]:+6.2f} {r[3]:+7.2f} {r[4]:+6.2f} '\n"
            "          f'{r[5]:+7.2f} {r[6]:+6.2f} {r[7]:+6.2f} {r[8]:4d}')\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "x = np.arange(len(rows))\n"
            "ax.bar(x-0.27, [r[2] for r in rows], 0.25, color=AMBER, label='VALUE t')\n"
            "ax.bar(x,      [r[4] for r in rows], 0.25, color=GREY, label='MOMENTUM t')\n"
            "ax.bar(x+0.27, [r[6] for r in rows], 0.25, color=RED, label='COMBO t')\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t = +2 bar'); ax.axhline(-2, ls='--', c=GREEN)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('HAC t')\n"
            "ax.set_title('No leg, in no sleeve, gets anywhere near the bar')\n"
            "ax.legend(ncol=2); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the paper needs every bar clearly above the green line. The best we "
            f"find anywhere is \\|t\\| < 2, and half the legs point the *wrong way* — commodity momentum "
            f"is the worst (t = {R['sleeves'][3][5]:+.2f}). The everywhere-premia are simply not on "
            "this tape.\n\n"
            f"The hedge (H₂) fares no better: mean per-sleeve ρ(VAL, MOM) = **{R['mean_rho']:+.2f}** "
            "vs the paper's ~−0.5."
        ),
        md(
            "### 4b · The claim under test — the global 50/50 combo\n\n"
            "Equal-weight the live sleeves into global VAL / MOM / COMBO and put a HAC t on each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    G = {leg: st.global_series(SLEEVES, leg) for leg in ('val','mom','combo')}\n"
            "    tbl = {leg: (st.sharpe(G[leg]), st.hac_mean(G[leg])['t'],\n"
            "                 st.hac_mean(G[leg])['mean_bps'], len(G[leg])) for leg in G}\n"
            "else:\n"
            "    tbl = R['glob']\n"
            "for leg, v in tbl.items():\n"
            "    print(f'GLOBAL {leg.upper():6s}: SR {v[0]:+.2f}   HAC t {v[1]:+.2f}   '\n"
            "          f'{v[2]:+.1f} bps/mo   n={v[3]}')\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "    for leg, cname in (('val', AMBER), ('mom', GREY), ('combo', RED)):\n"
            "        s = G[leg]\n"
            "        ax.plot((1+s).cumprod(), color=cname, lw=1.8,\n"
            "                label=f'global {leg.upper()}  (SR {st.sharpe(s):+.2f}, t {st.hac_mean(s)[\"t\"]:+.2f})')\n"
            "    ax.axvline(pd.Timestamp('2011-12-31'), ls=':', c='k', lw=1.2)\n"
            "    ax.annotate(' AMP sample ends', (pd.Timestamp('2011-12-31'), ax.get_ylim()[1]*0.95))\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log, gross)')\n"
            "    ax.set_title('The everywhere portfolios, gross of costs')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the headline claim — a strong, significant global combo — lands at "
            f"**SR {R['glob']['combo'][0]:+.2f}, t = {R['glob']['combo'][1]:+.2f}**. Sub-periods: "
            f"≤2011 (overlapping AMP's sample) SR **{R['sub_pre'][0]:+.2f}** "
            f"(t = {R['sub_pre'][1]:+.2f}, n = {R['sub_pre'][2]}); ≥2012 SR "
            f"**{R['sub_post'][0]:+.2f}** (t = {R['sub_post'][1]:+.2f}, n = {R['sub_post'][2]}). "
            "Neither half clears anything — and the AMP-overlap half is actually the *worse* one, so "
            "this is not even a clean \"decay\" story: the combo was never visible on this free tape."
        ),
        md(
            "### 4c · The third axis — diversification arithmetic, nothing more\n\n"
            "Two-asset formula: predicted combo Sharpe from (μ, σ, ρ) of the global VAL and MOM legs "
            "vs the realized Sharpe of the 50/50 mix of the same two series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dc = st.diversification_check(st.global_series(SLEEVES,'val'),\n"
            "                                  st.global_series(SLEEVES,'mom'))\n"
            "else:\n"
            "    dc = dict(rho=R['rho_global'], predicted=R['pred_sr'], realized=R['real_sr'],\n"
            "              sr_val=R['sr_val'], sr_mom=R['sr_mom'])\n"
            "print(f\"global rho(VAL,MOM) = {dc['rho']:+.3f}\")\n"
            "print(f\"ingredients: SR_val {dc['sr_val']:+.3f}   SR_mom {dc['sr_mom']:+.3f}\")\n"
            "print(f\"predicted combo SR (formula) = {dc['predicted']:+.3f}\")\n"
            "print(f\"realized  combo SR           = {dc['realized']:+.3f}\")\n"
            "print(f\"gap = {abs(dc['realized']-dc['predicted']):.4f}  -> pure arithmetic\")\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.3))\n"
            "ax.bar(['SR value','SR momentum','combo\\n(formula)','combo\\n(realized)'],\n"
            "       [dc['sr_val'], dc['sr_mom'], dc['predicted'], dc['realized']],\n"
            "       color=[AMBER, GREY, GREY, RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualised Sharpe'); ax.set_title('The combo IS the formula - no residual magic')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: realized **{R['real_sr']:+.3f}** vs predicted "
            f"**{R['pred_sr']:+.3f}** — the combo does *exactly* what two-asset arithmetic says, no "
            "more. That is the honest content of \"everywhere\": diversification is real, but it "
            "**multiplies the ingredients** — in [401](../../401-signal-stacking/) the ingredients "
            "were noise; here they are two famous premia that this tape pays at ≈ 0. Zero times 1.41 "
            "is still zero."
        ),
        md(
            "### 4d · Placebo — random ranks through the identical pipeline (50 seeds)\n\n"
            "If the construction (thirds, lag, sleeve-averaging) baked in any drift, random signals "
            "would show it."
        ),
        code(
            "# canonical numbers from the frozen run (50 seeds is minutes of compute);\n"
            "# recompute with st.random_placebo(PANELS) if you want to reproduce live\n"
            "pl = R['placebo']\n"
            "print(f\"random-rank global combo, {pl['n_seeds']} seeds: mean SR {pl['mean_sharpe']:+.2f} \"\n"
            "      f\"(sd {pl['sd']:.2f}), mean HAC t {pl['mean_t']:+.2f}, frac |t|>=2 = {pl['frac']:.0f}%\")\n"
            "print(f\"observed real-signal combo SR {R['glob']['combo'][0]:+.2f} sits inside the noise band\")"
        ),
        md(
            f"> 💡 In plain words: noise portfolios built the same way average SR "
            f"**{R['placebo']['mean_sharpe']:+.2f}** — the machine is unbiased, and the real-signal "
            f"combo (**{R['glob']['combo'][0]:+.2f}**) sits *inside* the noise band, on the wrong side."
        ),
        md(
            "### 4e · Costs and robustness — it only gets worse\n\n"
            "One-way costs × traded notional (avg combo turnover ≈ "
            f"{R['turnover']:.2f}× NAV/month), EQ short leg pays 50 bps/yr borrow; then the "
            "construction variants."
        ),
        code(
            "print('costs (global combo):')\n"
            "for cb, g, n, sr, t in R['costs']:\n"
            "    print(f'  {cb:>4.0f} bps one-way: gross {g:+.1f}%/yr -> net {n:+.1f}%/yr   '\n"
            "          f'net SR {sr:+.2f}   net t {t:+.2f}')\n"
            "print()\n"
            "print('construction variants (global combo, gross):')\n"
            "for v, sr, t in R['robust']:\n"
            "    print(f'  {v:24s}: SR {sr:+.2f}   HAC t {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: gross **{R['costs'][0][1]:+.1f}%/yr** becomes "
            f"**{R['costs'][2][2]:+.1f}%/yr** at 20 bps. Halves instead of thirds, momentum without "
            "the skip-month, value including the last year — every variant stays negative-to-flat. "
            "There is no construction under which this tape pays the combo."
        ),
        md(
            "### 4f · Synthetic control — the machinery can see the effect when it exists\n\n"
            "A 4-sleeve world generated month-by-month with plantable cross-sectional value and "
            "momentum edges. Zero edges → the pipeline must report nothing. Planted edges → both legs "
            "light up AND the combo obeys the same diversification formula."
        ),
        code(
            "rows = []\n"
            "for ve, me, label in ((0.0, 0.0, 'null (no edges)'), (0.004, 0.004, 'planted edges')):\n"
            "    world = data.synthetic_world(val_edge=ve, mom_edge=me, seed=638)\n"
            "    ssl = {k: st.sleeve_series(v) for k, v in world.items()}\n"
            "    gv, gm = st.global_series(ssl,'val'), st.global_series(ssl,'mom')\n"
            "    gc = st.global_series(ssl,'combo')\n"
            "    dc = st.diversification_check(gv, gm)\n"
            "    rows.append((label, st.sharpe(gv), st.sharpe(gm), st.sharpe(gc),\n"
            "                 st.hac_mean(gc)['t'], dc['predicted'], dc['realized']))\n"
            "for r in rows:\n"
            "    print(f'{r[0]:18s}: VAL SR {r[1]:+.2f}  MOM SR {r[2]:+.2f}  COMBO SR {r[3]:+.2f} '\n"
            "          f'(t {r[4]:+.2f})   formula {r[5]:+.2f} vs realized {r[6]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with nothing planted the pipeline reports SR ≈ "
            f"**{R['syn'][0][3]:+.2f}** (it cannot hallucinate a free lunch); with modest planted "
            f"edges it reports **{R['syn'][1][3]:+.2f}** (t = {R['syn'][1][4]:+.1f}) and the combo "
            "again equals the formula. The machine works — the real tape simply has nothing for it to "
            "find. *(A machinery proof only — never cited in support of a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the literature (AMP 2013, 1972–2011) says the premia are real; "
            f"this free tape cannot certify any of it: all eight sleeve-legs at \\|t\\| < 2, global "
            f"combo SR **{R['glob']['combo'][0]:+.2f}** with HAC **t = {R['glob']['combo'][1]:+.2f}** "
            f"({R['glob']['combo'][3]} months), hedge correlation ≈ 0, and neither the ≤2011 nor the "
            "≥2012 half shows anything. Survivor ETFs, spot FX and roll-noisy futures are named — "
            "they blur, but they cannot manufacture a premium that isn't there.\n"
            f"- **Tradability `MIRAGE`** — negative gross ({R['costs'][0][1]:+.1f}%/yr), "
            f"~{R['turnover']:.2f}× NAV monthly turnover, and costs push it to "
            f"**{R['costs'][1][2]:+.1f}%/yr** at 10 bps. Nothing to deploy.\n"
            f"- **Just diversification arithmetic? `CONFIRMED`** — realized combo SR "
            f"**{R['real_sr']:+.3f}** = formula **{R['pred_sr']:+.3f}**. The \"free lunch\" is the "
            "two-asset formula, which multiplies its ingredients — real ones in AMP's sample, absent "
            "ones here, noise in [401](../../401-signal-stacking/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **What would change the verdict.** AMP's own tape: dozens of markets per class, "
            "point-in-time universes, proper futures rolls, fundamental value measures (B/M, real "
            "yield differentials), 1972→. If someone rebuilds that from free sources and a leg clears "
            "HAC t ≥ 2, the Signal axis reopens.\n"
            "- **The individual pieces already died separately on this desk** — FX momentum "
            "([147](../../147-fx-momentum/)), stock long-term reversal "
            "([196](../../196-long-term-reversal/)), FX carry ([364](../../364-fx-carry-trade/)), "
            "time-series momentum ([31](../../31-trade-winds/), Weak). A combo of Weak-to-None "
            "ingredients grading Weak is exactly what the arithmetic predicts.\n"
            "- **The blender is still worth owning.** ρ ≈ 0 between strategy legs *does* cut risk — "
            "diversification survives every audit on this desk. What doesn't survive is the promise "
            "that the ingredients are premia rather than noise.\n\n"
            "*The reproducible core is offline and deterministic; every number above is printed by "
            "[`examples/verify.py`](../examples/verify.py). Sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
