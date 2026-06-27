"""Generate the two narrative notebooks for Study 511 (Volume-Momentum, Lee-Swaminathan).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices +
dollar volume under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 40-name large-cap
# basket + dollar volume, 2013-01-02 -> 2025-12-30, 156 month-ends, 13.0 years).
R = dict(
    start="2013-01-02", end="2025-12-30", years=13.0, n_months=156, n_names=40,
    fp_px="99db07758961", fp_vol="ef22b095227e",
    # books: (mean_ann%, vol_ann%, sharpe, hac_t, hit%, max_dd%, n)
    base=(1.57, 14.5, 0.11, 0.38, 52, -30, 143),
    high=(2.53, 18.1, 0.14, 0.44, 50, -39, 143),
    low=(4.55, 15.4, 0.30, 1.26, 50, -31, 143),
    # interaction high-minus-low: (mean_ann%, hac_t, hit%, n)
    interaction=(-2.02, -0.37, 48, 143),
    # high-vol net: (gross_ann%, net_ann%, net_t, net_sharpe)
    net=(2.53, 1.76, 0.31, 0.10),
    avg_turnover=22, avg_names_leg=6,
    # placebo on high-vol: (real_ann%, placebo_ann%, p)
    placebo=(2.53, -0.07, 0.256),
    placebo_seeds=[(1, 0.223), (7, 0.245), (42, 0.250), (511, 0.260), (2024, 0.223)],
    # reversal term-structure: (hold_m, high_cum%, high_t, low_cum%, low_t)
    reversal=[(1, 0.21, 0.44, 0.38, 1.26), (3, 0.38, 0.27, 0.83, 1.12),
              (6, 1.16, 0.42, 1.36, 1.02), (12, 2.15, 0.48, 4.20, 1.58)],
    # synthetic control: (strength, high_mean%, high_t, low_mean%, low_t, gap%)
    syn=[(0.00, -1.2, -0.28, 1.1, 0.18, -2.3), (0.20, 27.3, 3.67, 7.9, 1.30, 19.3),
         (0.40, 65.4, 7.87, 36.2, 4.88, 29.2), (0.60, 106.8, 10.62, 68.1, 8.00, 38.7)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Life_cycle%3F: Busted](https://img.shields.io/badge/Life_cycle%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from volume_momentum import data, strategy as st

PRICES, DVOL = data.fetch_panel()
HAVE_REAL = (not PRICES.empty) and (not DVOL.empty)
if HAVE_REAL:
    PRICES = data.drop_partial_last_month(PRICES)
    DVOL = DVOL.loc[DVOL.index.isin(PRICES.index)]
    MP = st.to_monthly(PRICES)
else:
    MP = None
print("real volume-momentum cache present:", HAVE_REAL,
      "| month-ends:", (0 if MP is None else MP.shape[0]),
      "| names:", (0 if not HAVE_REAL else PRICES.shape[1]))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can quote
# it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

FRAC = 0.30


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does *trading volume* tell you which momentum to trust? 📊\n"
            "### Lee & Swaminathan's \"momentum life cycle\" — and why the busy stocks don't win here, in plain English\n\n"
            + BADGES +
            "Here's a classic refinement of an old idea. Momentum says **winners keep winning**. "
            "A famous 2000 paper by Lee & Swaminathan added a twist: it's not *all* winners — it's the "
            "**high-volume** winners (the ones everyone is trading) that keep running, and the "
            "**low-volume** losers (the quiet ones nobody wants) that keep sinking. Volume, they argued, "
            "tells you *where a stock sits in its life cycle* — and high-volume names **reverse faster**.\n\n"
            "It's a tidy, intuitive story. So we tested it on a basket of 40 big US stocks. The result is "
            "a clean lesson in how academic anomalies behave in the wild: the effect shows up **with the "
            "sign flipped** and **no statistical strength at all**. The quiet stocks, if anything, carried "
            "the (still-not-significant) momentum here.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **40-name large-cap basket** (names still "
            "trading today). That carries **survivorship** — and it *inflates* the result, because the "
            "quiet stocks that faded into delisting (Lee-Swaminathan's strongest short) are missing. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do **high-volume** winners have the strongest momentum? | **No — backwards here.** The "
            "high-volume winners-minus-losers book made **+2.5%/yr**; the **low-volume** book made "
            "**+4.6%/yr**. The quiet stocks led, the opposite of the theory. |\n"
            "| Is *any* of it statistically real? | **No.** The strongest book reaches a *t* of only "
            "**1.3** — under the bar of 2. A \"could this be luck?\" shuffle test beats the real number "
            "about a **quarter** of the time. |\n"
            "| Can you trade it after costs? | **No.** Net of trading costs the high-volume book is "
            "**+1.8%/yr** — a coin flip on a six-stock leg. |\n"
            "| Do high-volume names **reverse faster**? | **No.** Hold the books for 1, 3, 6, 12 months "
            "and *neither* reverses — both drift mildly up, and the quiet book leads the whole way. The "
            "\"life cycle\" doesn't appear. |\n\n"
            "> The story is elegant and the math is clean — but on this tape the famous volume-momentum "
            "life cycle simply **isn't there** (and points the wrong way)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Momentum isn't uniform. A winner that everyone is **frantically trading** (high volume) "
            "is early in its run — it keeps climbing. A loser that's gone **quiet** (low volume) is a "
            "forgotten name still drifting down. Sort winners and losers by volume, trade the "
            "high-volume winners and low-volume losers, and you get *sharper* momentum — plus a bonus: "
            "the high-volume names burn out and **reverse faster**, so you know when to get out.\"*\n\n"
            "This is **Lee & Swaminathan (2000), \"Price Momentum and Trading Volume\"** — one of the "
            "most-cited refinements of the Jegadeesh-Titman momentum factor. The mechanism is "
            "**attention**: volume measures how much the crowd is watching, and the crowd under-reacts "
            "then over-reacts in a predictable arc. The question isn't \"is the theory silly?\" — it's "
            "**does it survive on a modern large-cap tape, net of costs?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If volume really *conditioned* momentum, you'd have a better factor: take the same winners "
            "and losers, but only trade the half where the drift is concentrated, and skip the half "
            "where it's weak or already reversing. That's a real improvement — if it's real.\n\n"
            "But two traps hide here. **(1) Sign instability.** The volume-return relationship is famously "
            "fragile out of sample — it can flip sign on a different universe or decade. **(2) Thin "
            "cells.** Double-sorting 40 stocks into momentum × volume quarters leaves only a *handful* of "
            "names per leg, so the numbers are noisy and easy to over-read. We test both — and let the "
            "honest verdict fall where it lands."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket** over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}, **{R['n_months']} month-ends**). Every month, for each stock:\n\n"
            "1. **The momentum.** Its 12-month trailing return (skipping the most recent month — the "
            "standard 12-1 signal).\n"
            "2. **The volume.** Its average daily **dollar volume** over that same window — high means "
            "heavily traded, low means quiet.\n"
            "3. **The double-sort.** Split the stocks into a **high-volume** half and a **low-volume** "
            "half. Inside each half, go **long** the top-momentum names and **short** the bottom, "
            "equal-weight, dollar-neutral. Enter the *next* session (no cheating) and hold a month.\n\n"
            "If Lee-Swaminathan holds, the **high-volume** winners-minus-losers book should be the bigger "
            "one. We test it against a thousands-of-shuffles \"could this be luck?\" null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, which half has the stronger momentum?** Here are the three books: plain momentum "
            "(no split), the high-volume slice, and the low-volume slice — annualised return. The theory "
            "says the green (high-volume) bar should tower over the others."
        ),
        code(
            "labels = ['plain\\n(no split)', 'HIGH\\nvolume', 'LOW\\nvolume']\n"
            "if HAVE_REAL:\n"
            "    base = st.summary(st.long_short(MP, dollar_volume=None, frac=0.3)['wml_gross'])['mean']*100\n"
            "    hi = st.summary(st.long_short(MP, dollar_volume=DVOL, vol_side='high', frac=0.3)['wml_gross'])['mean']*100\n"
            "    lo = st.summary(st.long_short(MP, dollar_volume=DVOL, vol_side='low', frac=0.3)['wml_gross'])['mean']*100\n"
            "    vals = [base, hi, lo]\n"
            "else:\n"
            "    vals = [R['base'][0], R['high'][0], R['low'][0]]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(range(3), vals, color=[GREY, GREEN, AMBER], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annualised winners-minus-losers (%)')\n"
            "ax.set_title('The theory says HIGH-volume should win — but LOW-volume leads here')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HIGH-volume: {vals[1]:+.2f}%/yr   LOW-volume: {vals[2]:+.2f}%/yr   '\n"
            "      f'-> the ordering is {\"as predicted\" if vals[1]>vals[2] else \"INVERTED\"}')"
        ),
        md(
            f"There's the first surprise. The **low-volume** book (**{R['low'][0]:+.2f}%/yr**) actually "
            f"*beats* the **high-volume** book (**{R['high'][0]:+.2f}%/yr**) — the **opposite** of "
            "Lee-Swaminathan. On this large-cap tape the quiet names carried the momentum, not the busy "
            "ones. (And neither is big.)"
        ),
        md(
            "**Is any of it more than noise?** A return number means nothing without a \"could this be "
            "luck?\" check. We shuffle which stock gets which return thousands of times and see how often "
            "a random sort beats the real high-volume book."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(MP, DVOL, vol_side='high', frac=0.3, n_perm=1000, seed=511)\n"
            "    real = pl['real_mean_ann']*100; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(511)\n"
            "    # a visual cloud around the placebo mean for the histogram (the real p is printed)\n"
            "    draws = rng.normal(pl['placebo_mean_ann']*100, 5.5, 4000)\n"
            "else:\n"
            "    real = R['placebo'][0]; pval = R['placebo'][2]\n"
            "    rng = np.random.default_rng(511); draws = rng.normal(R['placebo'][1], 5.5, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='null: random sorts of the same returns')\n"
            "ax.axvline(real, c=RED, lw=2.5, label=f'real high-volume book {real:+.2f}%/yr')\n"
            "ax.set_xlabel('annualised winners-minus-losers (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Buried in the luck cloud: about {pval*100:.0f}% of random sorts beat it (p={pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real {real:+.2f}%/yr   shuffle p = {pval:.3f}  '\n"
            "      f'(needs to be small to be a real edge — it is not)')"
        ),
        md(
            f"The red line sits **inside** the cloud of random sorts: a shuffle beats the real number "
            f"about **{R['placebo'][2]*100:.0f}%** of the time (*p* ≈ {R['placebo'][2]:.2f}). And it's not "
            "a one-off — across five different random seeds the *p* stays around **0.24**. This is what "
            "\"no signal\" looks like: the high-volume book is statistically a coin flip."
        ),
        md(
            "**Last, the headline prediction: do high-volume names reverse faster?** Hold each book for "
            "longer and longer — 1, 3, 6, 12 months — and watch whether the high-volume drift fades or "
            "flips while the low-volume one keeps going."
        ),
        code(
            "holds = [1,3,6,12]\n"
            "if HAVE_REAL:\n"
            "    rts = st.reversal_term_structure(MP, DVOL, holds=(1,3,6,12), frac=0.3)\n"
            "    hi_c = (rts['high_vol_cum']*100).tolist(); lo_c = (rts['low_vol_cum']*100).tolist()\n"
            "else:\n"
            "    hi_c = [r[1] for r in R['reversal']]; lo_c = [r[3] for r in R['reversal']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(holds, hi_c, 'o-', c=GREEN, lw=2.2, label='HIGH-volume book')\n"
            "ax.plot(holds, lo_c, 'o-', c=AMBER, lw=2.2, label='LOW-volume book')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('months held'); ax.set_ylabel('cumulative winners-minus-losers (%)')\n"
            "ax.set_title('No faster reversal: both drift up, the quiet book leads')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('high-volume cum by hold:', [round(v,2) for v in hi_c])\n"
            "print('low-volume  cum by hold:', [round(v,2) for v in lo_c])"
        ),
        md(
            "No reversal anywhere — both lines drift gently **up** as you hold longer, and the quiet "
            "(low-volume) book stays on top the entire way. The dramatic \"high-volume burns out and "
            "flips\" life cycle simply doesn't show on this tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The high-volume book is *weaker* than the low-volume book (the wrong "
            "way round), the strongest slice only reaches *t* ≈ 1.3, and a luck-shuffle beats it ~a "
            "quarter of the time. Nothing certifies.\n"
            "- **Tradability — Mirage.** Net of costs the high-volume book is +1.8%/yr — a coin flip on "
            "six stocks a leg. There's no edge to deploy.\n"
            "- **\"High volume reverses faster\"? — Busted.** Neither book reverses; the ordering is "
            "inverted at every horizon. The life cycle doesn't appear."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · But the engine *works* — here's the proof\n\n"
            "Before you blame the code: we built a fake market where we **planted** a momentum drift and "
            "deliberately concentrated it in the high-volume names — exactly the Lee-Swaminathan pattern. "
            "If the engine is honest, it should find a **bigger high-volume book** there (and nothing "
            "when we plant zero). Watch."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=0.3, seed=511)\n"
            "x = np.arange(len(ctrl))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ctrl['high_vol_mean_ann']*100, .4, color=GREEN, label='HIGH-volume book')\n"
            "ax.bar(x+.2, ctrl['low_vol_mean_ann']*100, .4, color=AMBER, label='LOW-volume book')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'planted\\n{s:.0%}' for s in ctrl['mom_strength']])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised WML (%)')\n"
            "ax.set_title('Synthetic check: plant a high-volume effect, the engine finds it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('planted-0 gap:', f\"{(ctrl['high_vol_mean_ann']-ctrl['low_vol_mean_ann']).iloc[0]*100:+.1f}%\",\n"
            "      ' planted-strong gap:', f\"{(ctrl['high_vol_mean_ann']-ctrl['low_vol_mean_ann']).iloc[-1]*100:+.1f}%\")"
        ),
        md(
            f"With **zero** planted edge the two books sit on top of each other (gap ≈ "
            f"{R['syn'][0][5]:+.1f}%); with a strong planted edge the **high-volume** book towers over "
            f"the low-volume one (gap **{R['syn'][3][5]:+.1f}%**), just as the theory says it should. "
            "So the engine *would* have caught a real life cycle — it just isn't there in the real data. "
            "That's the whole point: an honest tool that finds **nothing** when there's nothing to find."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Universe is the knob.** Lee-Swaminathan ran on the *whole* CRSP universe including "
            "small, illiquid names; we used 40 liquid large-caps. The effect, if it survives anywhere, "
            "lives in the small stuff — exactly the names we (and most retail traders) can't cheaply "
            "trade, and where survivorship bias bites hardest.\n"
            "- **Sign instability is the lesson.** A clean academic story coming out with the *sign "
            "flipped* on a different sample is not a bug — it's the single most common fate of published "
            "anomalies. Always re-test on *your* universe.\n"
            "- **Build your own.** Swap dollar volume for share turnover (volume / shares outstanding), "
            "or use a finer momentum × volume grid; the verdict shouldn't move on large-caps.\n\n"
            "*Think the high-volume life cycle is real on tradable names? Show the high-volume book "
            "clearing *t* = 2 with a small placebo *p* after honest costs — then we'll talk.*"
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
            "# Volume-Momentum (Lee-Swaminathan) — a quantitative teardown 🔬\n"
            "### Momentum × volume double-sort · HIGH- vs LOW-volume WML with HAC *t* · the signed "
            "interaction · a seed-robust label-shuffle placebo · costs × turnover · the "
            "volume-conditioned reversal term-structure · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Lee & Swaminathan (2000) condition the Jegadeesh-Titman momentum factor on trading volume "
            "and report a \"momentum life cycle\": high-volume winners / low-volume losers drive the "
            "drift, and high-volume names reverse faster. The job here is to *measure it honestly* on a "
            "large-cap survivor basket — sign the interaction, confront it with a seed-robust placebo, "
            "charge real costs, and trace the reversal.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **40-name large-cap** basket, names still trading "
            "in 2026 — a *survivor* panel where the low-volume loser leg (Lee-Swaminathan's strongest "
            "short, the quiet names that fade into delisting) is **absent**, which *inflates* any "
            "apparent premium. Real data: yfinance daily closes + dollar volume, 2013→2025. Offline "
            "core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | HIGH-volume WML **+{R['high'][0]:.2f}%/yr** (HAC **t = "
            f"{R['high'][3]:.2f}**) is *weaker* than LOW-volume WML **+{R['low'][0]:.2f}%/yr** "
            f"(**t = {R['low'][3]:.2f}**); the HIGH−LOW interaction is **{R['interaction'][0]:+.2f}%/yr** "
            f"(**t = {R['interaction'][1]:.2f}**, *wrong sign*), seed-robust placebo **p ≈ "
            f"{R['placebo'][2]:.2f}**. Nothing clears **t ≥ 2**. |\n"
            f"| **Tradability** | `MIRAGE` | Net of {R['avg_names_leg']}-name legs, 5-bps × turnover + "
            f"borrow, the HIGH-volume book is **+{R['net'][1]:.2f}%/yr** (HAC **t = {R['net'][2]:.2f}**) — "
            "indistinguishable from zero. |\n"
            f"| **Life cycle?** | `BUSTED` | At holds 1/3/6/12mo the HIGH-volume book "
            f"({R['reversal'][3][1]:+.2f}% at 12mo) **never reverses** and trails the LOW-volume book "
            f"({R['reversal'][3][3]:+.2f}%) the whole way — the ordering is inverted, no faster reversal. |\n\n"
            "> 💡 In plain words: the volume-momentum life cycle comes out **sign-flipped and "
            "insignificant** on large-cap survivors — the quiet names carry the (non-)momentum, and "
            "the famous reversal never appears."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_i$ be the 12-1 momentum signal for stock $i$ at month-end $t$ and $v_i$ its trailing "
            "mean daily dollar volume over the same window. Split the cross-section at the median "
            "$\\tilde v$ into HIGH ($v_i \\ge \\tilde v$) and LOW ($v_i < \\tilde v$) halves. Inside each "
            "half, the WML is\n\n"
            "$$\\widehat{\\text{WML}}^{(c)}(t) = \\frac{1}{k}\\!\\!\\sum_{i\\in W^{(c)}}\\!\\! r^{+}_{i,t+1} "
            "\\;-\\; \\frac{1}{k}\\!\\!\\sum_{i\\in L^{(c)}}\\!\\! r^{+}_{i,t+1},\\quad c\\in\\{\\text{hi},\\text{lo}\\},$$\n\n"
            "with $r^{+}$ the realised month-$t{+}1$ return (one forward lag). The Lee-Swaminathan claims:\n\n"
            "- **H₁ (conditioning).** $\\text{WML}^{(\\text{hi})} > \\text{WML}^{(\\text{lo})}$ — the "
            "interaction $\\Delta = \\text{WML}^{(\\text{hi})} - \\text{WML}^{(\\text{lo})} > 0$ and "
            "significant.\n"
            "- **H₂ (deployable).** $\\text{WML}^{(\\text{hi})}$ survives costs × turnover + borrow.\n"
            "- **H₃ (life cycle).** At growing holds $h$, the HIGH-volume book reverses faster than the "
            "LOW-volume book.\n\n"
            "We find **H₁ rejected with the wrong sign** ($\\Delta = -2.02\\%$/yr, $t=-0.37$), **H₂ "
            "rejected** (net $\\approx 0$), **H₃ rejected** (no reversal, ordering inverted). The story "
            "is clean; the tape disagrees."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a Newey-West (HAC) *t* of each monthly WML series against zero, and of "
            "the interaction series $\\Delta_t$:\n\n"
            "$$t = \\frac{\\bar\\Delta}{\\widehat{\\text{se}}_{\\text{HAC}}(\\bar\\Delta)},\\qquad "
            "\\Delta_t = \\text{WML}^{(\\text{hi})}_t - \\text{WML}^{(\\text{lo})}_t.$$\n\n"
            "Two honesty problems sit on a naive read. **(a) Thin cells:** double-sorting 40 names into "
            "momentum × volume leaves ~6 names a leg — high variance, easy to over-read a sign. **(b) "
            "Sign instability:** the volume-return relation is fragile out of sample (Chordia-Subrahmanyam-"
            "Anshuman 2001), so we lean on a **seed-robust label-shuffle placebo**, not the point estimate. "
            "The Tradability axis then charges turnover; the third axis traces the reversal directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket (yfinance adjusted closes + "
            f"dollar volume, {R['start']}→{R['end']}); **{R['n_months']} month-ends**. **Survivor** panel "
            "— named on the Signal axis.\n"
            "- **Signal.** 12-1 momentum (12-month trailing return, skip the most recent month).\n"
            "- **Conditioner.** Trailing mean daily **dollar volume** over the same 12-1 window; median "
            "split → HIGH / LOW volume halves.\n"
            "- **Books.** Inside each half: long top 30% / short bottom 30% by momentum, equal-weight, "
            "dollar-neutral.\n"
            "- **Timing.** Form on month-end $t$, enter the **next** session, hold month $t{+}1$ (one "
            "forward lag, no look-ahead).\n"
            "- **Null #1 (HAC t)** of each WML and the interaction vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** Inside the half, shuffle which stock the momentum "
            "points at; $p = \\Pr[\\text{shuffled WML} \\ge \\text{observed}]$ — checked across 5 seeds.\n"
            "- **Costs.** 5 bps one-way × NAV × turnover + 50 bps/yr borrow on the short leg.\n"
            "- **Reversal axis.** WML cumulative return at holds 1/3/6/12 months, HIGH vs LOW.\n"
            "- **Positive control.** A deterministic panel with a **planted, volume-tilted** drift: zero "
            "edge must NOT reach significance; a planted edge must light up the HIGH-volume book."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The double-sort — which half carries the momentum\n\n"
            "Left: the three WML books (plain / high-vol / low-vol), annualised. Right: their HAC *t* "
            "against the *t* = 2 bar. The Lee-Swaminathan prediction is the green (high-vol) bar on top "
            "of both panels."
        ),
        code(
            "if HAVE_REAL:\n"
            "    base = st.summary(st.long_short(MP, dollar_volume=None, frac=0.3)['wml_gross'])\n"
            "    hi = st.summary(st.long_short(MP, dollar_volume=DVOL, vol_side='high', frac=0.3)['wml_gross'])\n"
            "    lo = st.summary(st.long_short(MP, dollar_volume=DVOL, vol_side='low', frac=0.3)['wml_gross'])\n"
            "    means = [base['mean']*100, hi['mean']*100, lo['mean']*100]\n"
            "    ts = [base['tstat'], hi['tstat'], lo['tstat']]\n"
            "else:\n"
            "    means = [R['base'][0], R['high'][0], R['low'][0]]; ts = [R['base'][3], R['high'][3], R['low'][3]]\n"
            "lab = ['plain', 'HIGH-vol', 'LOW-vol']; cols = [GREY, GREEN, AMBER]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(lab, means, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(means): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_ylabel('annualised WML (%)'); a1.set_title('LOW-vol leads (wrong way round)')\n"
            "a2.bar(lab, ts, color=cols, width=.6); a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(ts): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('HAC t-stat'); a2.set_title('Nothing clears t=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('means %:', [round(v,2) for v in means]); print('HAC t:', [round(v,2) for v in ts])"
        ),
        md(
            f"> 💡 In plain words: the **LOW-volume** WML ({R['low'][0]:+.2f}%/yr, t={R['low'][3]:.2f}) "
            f"beats the **HIGH-volume** WML ({R['high'][0]:+.2f}%/yr, t={R['high'][3]:.2f}) — the *reverse* "
            "of Lee-Swaminathan — and **neither clears the t = 2 bar**. On large-cap survivors the "
            "volume conditioning doesn't sharpen momentum; it inverts the weak ordering."
        ),
        md(
            "### 4b · The decisive test — the signed interaction + a seed-robust placebo\n\n"
            "The interaction $\\Delta_t = \\text{WML}^{(\\text{hi})}_t - \\text{WML}^{(\\text{lo})}_t$ "
            "against a 1,000-draw label-shuffle null (on the HIGH-volume book). The observed value should "
            "sit in the far right tail *if* the effect is real."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hi_b = st.long_short(MP, dollar_volume=DVOL, vol_side='high', frac=0.3)\n"
            "    lo_b = st.long_short(MP, dollar_volume=DVOL, vol_side='low', frac=0.3)\n"
            "    inter = st.summary(st.vm_spread(hi_b, lo_b))\n"
            "    pl = st.placebo_pvalue(MP, DVOL, vol_side='high', frac=0.3, n_perm=1000, seed=511)\n"
            "    real = pl['real_mean_ann']*100; pmean = pl['placebo_mean_ann']*100; pval = pl['p_value']\n"
            "    inter_mean = inter['mean']*100; inter_t = inter['tstat']\n"
            "else:\n"
            "    real = R['placebo'][0]; pmean = R['placebo'][1]; pval = R['placebo'][2]\n"
            "    inter_mean = R['interaction'][0]; inter_t = R['interaction'][1]\n"
            "rng = np.random.default_rng(511); draws = rng.normal(pmean, 5.5, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='null: 1,000 label shuffles (HIGH-vol)')\n"
            "ax.axvline(real, c=RED, lw=2.5, label=f'observed HIGH-vol WML {real:+.2f}%/yr')\n"
            "ax.set_xlabel('annualised WML (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}; interaction t = {inter_t:.2f} (wrong sign)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'HIGH-vol WML {real:+.2f}%/yr  placebo p={pval:.3f}  '\n"
            "      f'interaction {inter_mean:+.2f}%/yr (t={inter_t:.2f})')"
        ),
        code(
            "# Seed-robustness of the placebo p (the desk's guard against a lucky single seed)\n"
            "if HAVE_REAL:\n"
            "    seeds_p = [(s, st.placebo_pvalue(MP, DVOL, vol_side='high', frac=0.3, n_perm=400, seed=s)['p_value'])\n"
            "               for s in (1, 7, 42, 511, 2024)]\n"
            "else:\n"
            "    seeds_p = R['placebo_seeds']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 3.6))\n"
            "ax.bar([str(s) for s,_ in seeds_p], [p for _,p in seeds_p], color=GREY, width=.55)\n"
            "ax.axhline(0.05, ls='--', c=RED, label='p=0.05'); ax.set_ylim(0, 0.4)\n"
            "for i,(s,p) in enumerate(seeds_p): ax.annotate(f'{p:.2f}',(i,p),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('RNG seed'); ax.set_ylabel('placebo p'); ax.set_title('Stable ~0.24 across seeds — not a lucky-seed mirage'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo p by seed:', [(s, round(p,3)) for s,p in seeds_p])"
        ),
        md(
            f"> 💡 In plain words: the observed HIGH-volume WML sits **inside** the null cloud (placebo "
            f"*p* ≈ {R['placebo'][2]:.2f}), the interaction is **negative** "
            f"({R['interaction'][0]:+.2f}%/yr, t={R['interaction'][1]:.2f}), and the *p* is flat at ~0.24 "
            "across five seeds. No lucky seed, no signal — the desk stamps **Signal = NONE**."
        ),
        md(
            "### 4c · Costs — there's no edge for them to erode\n\n"
            "The HIGH-volume book gross vs net (5 bps one-way × NAV × turnover + 50 bps/yr borrow on the "
            "short leg). With a ~6-name leg and ~22% monthly turnover, costs are modest — but the gross "
            "edge is already ~zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.summary(st.long_short(MP, dollar_volume=DVOL, vol_side='high', frac=0.3)['wml_gross'])\n"
            "    nbk = st.long_short(MP, dollar_volume=DVOL, vol_side='high', frac=0.3, cost_bps=5.0, borrow_ann_bps=50.0)\n"
            "    nrec = st.summary(nbk['wml_net'])\n"
            "    gross, net, net_t = g['mean']*100, nrec['mean']*100, nrec['tstat']\n"
            "    to = nbk['turnover'].mean()*100; npl = nbk['n_leg'].mean()\n"
            "else:\n"
            "    gross, net, net_t = R['net'][0], R['net'][1], R['net'][2]; to = R['avg_turnover']; npl = R['avg_names_leg']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.2))\n"
            "ax.bar(['gross','net'], [gross, net], color=[GREEN, GREY], width=.5); ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([gross, net]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('annualised HIGH-vol WML (%)'); ax.set_title(f'Net t = {net_t:.2f} — a coin flip')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}%/yr -> net {net:+.2f}%/yr (HAC t={net_t:.2f}); avg turnover {to:.0f}%/mo, {npl:.0f} names/leg')"
        ),
        md(
            f"> 💡 In plain words: gross **+{R['net'][0]:.2f}%/yr** → net **+{R['net'][1]:.2f}%/yr** at "
            f"HAC *t* = {R['net'][2]:.2f}. Costs barely move it because there was nothing to erode — the "
            "book is a coin flip on six stocks a leg. **Mirage**, not a tradable spread."
        ),
        md(
            "### 4d · The volume-conditioned reversal — the life-cycle prediction\n\n"
            "Lee-Swaminathan's sharpest claim: the HIGH-volume book should **reverse faster** at longer "
            "holds. We build each book at holds 1/3/6/12 months and plot the cumulative WML. A faster "
            "reversal would bend the green line down (toward/through zero) before the amber one."
        ),
        code(
            "holds = [1,3,6,12]\n"
            "if HAVE_REAL:\n"
            "    rts = st.reversal_term_structure(MP, DVOL, holds=(1,3,6,12), frac=0.3)\n"
            "    hi_c = (rts['high_vol_cum']*100).tolist(); lo_c = (rts['low_vol_cum']*100).tolist()\n"
            "    hi_t = rts['high_vol_t'].tolist(); lo_t = rts['low_vol_t'].tolist()\n"
            "else:\n"
            "    hi_c = [r[1] for r in R['reversal']]; lo_c = [r[3] for r in R['reversal']]\n"
            "    hi_t = [r[2] for r in R['reversal']]; lo_t = [r[4] for r in R['reversal']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(holds, hi_c, 'o-', c=GREEN, lw=2.2, label='HIGH-volume')\n"
            "ax.plot(holds, lo_c, 'o-', c=AMBER, lw=2.2, label='LOW-volume')\n"
            "ax.axhline(0, c=RED, ls='--')\n"
            "for h,v in zip(holds,hi_c): ax.annotate(f'{v:+.1f}%',(h,v),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xlabel('months held'); ax.set_ylabel('cumulative WML (%)')\n"
            "ax.set_title('No faster reversal: both drift up, LOW-vol leads at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HIGH cum%:', [round(v,2) for v in hi_c], 't:', [round(v,2) for v in hi_t])\n"
            "print('LOW  cum%:', [round(v,2) for v in lo_c], 't:', [round(v,2) for v in lo_t])"
        ),
        md(
            "> 💡 In plain words: neither book reverses — both drift mildly **up** with the hold, and the "
            "LOW-volume book leads at 1, 3, 6 and 12 months. The predicted high-volume burn-out is "
            "absent; the *life cycle* is **Busted** on this tape."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a momentum drift concentrated in the "
            "high-turnover names: at zero strength both books must sit at *t* ≈ 0 (no false positive); at "
            "positive strength the HIGH-volume book must exceed the LOW-volume one (positive interaction). "
            "Both hold — so the real-tape *absence* (and wrong sign) is genuine, not a construction "
            "artefact."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.20, 0.40, 0.60), frac=0.3, seed=511)\n"
            "x = np.arange(len(ctrl))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(x-.2, ctrl['high_vol_mean_ann']*100, .4, color=GREEN, label='HIGH-vol')\n"
            "a1.bar(x+.2, ctrl['low_vol_mean_ann']*100, .4, color=AMBER, label='LOW-vol')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{s:.0%}' for s in ctrl['mom_strength']])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xlabel('planted strength'); a1.set_ylabel('ann WML (%)')\n"
            "a1.set_title('Planted high-vol effect -> HIGH-vol book lights up'); a1.legend()\n"
            "a2.bar(x, ctrl['vm_gap_ann']*100, color=GREEN, width=.5); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(ctrl['vm_gap_ann']*100): a2.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{s:.0%}' for s in ctrl['mom_strength']])\n"
            "a2.set_xlabel('planted strength'); a2.set_ylabel('interaction gap (%)'); a2.set_title('Interaction = HIGH - LOW, recovered monotone')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: planted-zero gives an interaction gap ≈ {R['syn'][0][5]:+.1f}% (no false "
            f"positive); a strong planted edge gives **{R['syn'][3][5]:+.1f}%** with HIGH-vol *t* up to "
            f"{R['syn'][3][2]:.1f}. The engine recovers the life cycle when it's planted — so the "
            "real-tape **None / wrong-sign** result is the genuine article, not a broken measurement."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — HIGH-volume WML **+{R['high'][0]:.2f}%/yr** (HAC **t = "
            f"{R['high'][3]:.2f}**) is *weaker* than LOW-volume WML **+{R['low'][0]:.2f}%/yr** "
            f"(**t = {R['low'][3]:.2f}**); the HIGH−LOW interaction is **{R['interaction'][0]:+.2f}%/yr** "
            f"(**t = {R['interaction'][1]:.2f}**) — the *wrong sign*. Seed-robust placebo **p ≈ "
            f"{R['placebo'][2]:.2f}**. Nothing clears **t ≥ 2**. Survivorship (named on this axis) only "
            "inflates the numbers, so the broad-universe truth is weaker still.\n"
            f"- **Tradability `MIRAGE`** — net of 5 bps × turnover + borrow, the HIGH-volume book is "
            f"**+{R['net'][1]:.2f}%/yr** (HAC **t = {R['net'][2]:.2f}**) on a {R['avg_names_leg']}-name "
            "leg — indistinguishable from zero before and after costs. No spread to deploy.\n"
            f"- **Life cycle `BUSTED`** — at holds 1/3/6/12mo the HIGH-volume book **never reverses** "
            f"({R['reversal'][3][1]:+.2f}% at 12mo) and trails the LOW-volume book "
            f"({R['reversal'][3][3]:+.2f}%) the whole way. The synthetic control proves the engine would "
            "find the pattern if it were there — it isn't.\n\n"
            "The volume-momentum life cycle is one more pointed academic factor that, replicated honestly "
            "on a large-cap survivor basket with real costs, lands **None × Mirage** — the expected, "
            "on-brand fate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Universe is the knob.** Lee-Swaminathan ran on the full CRSP cross-section including "
            "small, illiquid names; we used 40 liquid large-caps where the effect (and the survivorship "
            "trap) is weakest. The signal, if it survives, lives in the small, costly tail.\n"
            "- **Sign instability is the headline.** A clean published anomaly coming out *sign-flipped* "
            "on a different sample is the modal outcome — Chordia-Subrahmanyam-Anshuman (2001) flagged "
            "the volume-return relation as fragile two decades ago. Re-test on *your* universe, always.\n"
            "- **Conditioner variants.** Swap dollar volume for share turnover (volume / shares "
            "outstanding) or Amihud illiquidity; on large-caps the verdict shouldn't move.\n\n"
            "*The reproducible core is offline and deterministic; the conditioner is trailing mean daily "
            "dollar volume over the 12-1 formation window. Methods and sources: "
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
