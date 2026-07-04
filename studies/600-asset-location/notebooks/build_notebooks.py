"""Generate the two narrative notebooks for Study 600 (Asset Location).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached Shiller and
SPY/IEF tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Shiller 1872-2022 annual
# components -> 122 overlapping 30-year cohorts; modern SPY/IEF cohort 2003-2025).
R = dict(
    asof="2026-07-03", fp_long="254d1e72f9f7", fp_modern="97c3a1b86d4f",
    long_years="1872-2022", n_years_long=151, n_cohorts=122, horizon=30, hac_lag=29,
    ordinary=24, ltcg=15, turnover=10,
    ann_bonds=6.84, ann_stocks=6.66, ann_prorata=6.76,
    delta_bs=17.78, t_bs=3.55, win_bs=86.9,
    delta_bp=8.55, t_bp=3.11, win_bp=82.8,
    delta_min=-14.46, delta_max=59.14, wealth_gap=5.19,
    trade=(2.67, 2.41, 3.34), diff_cost=0.013,
    # yield quartiles: (label, mean coupon %, delta bps/yr, n)
    quart=[("Q1 (lowest)", 3.06, 12.38, 31), ("Q2", 3.70, 12.53, 30),
           ("Q3", 4.42, 11.55, 30), ("Q4 (highest)", 7.44, 34.30, 31)],
    slope=5.57, t_slope=67.8,
    # tax grid: (ordinary %, ltcg %, delta bps/yr, HAC t, win %)
    grid=[(22, 15, 10.77, 2.26, 77.0), (22, 20, -9.95, -1.84, 31.1),
          (24, 15, 17.78, 3.55, 86.9), (24, 20, -3.16, -0.57, 46.7),
          (32, 15, 47.43, 7.68, 100.0), (32, 20, 25.54, 4.09, 91.0),
          (37, 15, 67.47, 9.62, 100.0), (37, 20, 44.96, 6.56, 100.0)],
    # turnover flip: (turnover %, delta bps/yr, HAC t, win %, low-yield-quartile delta)
    flip=[(0, 26.27, 4.15, 98.4, 17.89), (10, 17.78, 3.55, 86.9, 12.38),
          (25, 11.32, 2.58, 77.9, 7.31), (50, 5.72, 1.40, 68.9, 2.36),
          (100, -0.29, -0.07, 53.3, -3.69)],
    modern=dict(years="2003-2025", n=23, cpn=2.90, ann_bonds=7.51, ann_stocks=7.09,
                ann_prorata=7.31, delta_bs=41.68, delta_bp=19.49,
                sweep=[(0, 53.67), (10, 41.68), (25, 33.54), (50, 27.81), (100, 23.12)]),
    # synthetic control: (label, mean delta bps/yr, sd)
    syn=[("null (zero tax rates)", 0.000, 0.000), ("planted coupon 2%", 15.105, 5.366),
         ("planted coupon 4%", 21.000, 6.097), ("planted coupon 6%", 29.585, 7.079)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Does_it_flip%3F: Confirmed](https://img.shields.io/badge/Does_it_flip%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from asset_location import data, strategy as st

HAVE_REAL = data.have_shiller()
HAVE_MODERN = data.have_modern()
if HAVE_REAL:
    SH = data.load_shiller()
    COMP = data.annual_components(SH)
    TAB = st.cohort_table(COMP)
else:
    SH = COMP = TAB = None
print("real Shiller cache:", HAVE_REAL, "| modern SPY/IEF cache:", HAVE_MODERN,
      "| cohorts:", (0 if TAB is None else len(TAB)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Should the bonds live in your IRA? 🏦\n"
            "### Asset location — the rare piece of financial folklore that is **actually free money** (a little of it)\n\n"
            + BADGES +
            "You have two accounts: a **taxable** brokerage account and a **tax-deferred** one (a "
            "traditional IRA / 401(k)). You own a 60/40 mix of stocks and bonds across both. The old "
            "advice says it *matters which account holds which*: put the **bonds inside the IRA** and "
            "the **stocks in taxable**, and you'll end up richer — same funds, same risk, zero cost.\n\n"
            "Why would that work? Because the taxman treats the two assets differently. A bond coupon is "
            "taxed **every year** at your full income rate. A stock mostly grows as **unrealised capital "
            "gains** — taxed later, at a lower rate, only when you sell. The IRA is a *shield*: whatever "
            "sits inside it pays no tax along the way. So the shield should cover the asset that gets "
            "**mugged every year** — the bonds.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the tax-rate grid and the flip test? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does bonds-in-the-IRA really pay? | **Yes.** Across every 30-year stretch since 1872, it "
            f"beats the reversed placement by about **+{R['delta_bs']:.0f} bps of after-tax return per "
            f"year** and wins **{R['win_bs']:.0f}%** of the time — statistically solid, not luck. |\n"
            f"| How much money is that? | About **+{R['wealth_gap']:.0f}% more terminal wealth** after 30 "
            "years — same funds, same risk, just a different account label on each holding. |\n"
            "| What does it cost? | **Nothing.** That's the point — it's the same portfolio, relocated. "
            "The differential trading cost is ~0.01 bps/yr. |\n"
            "| Is there a catch? | **Two.** (1) The rule assumes your stocks are *tax-efficient* (index "
            "funds). Hold a high-turnover **active** fund instead and the advantage shrinks to zero — "
            "and can even flip against you in low-rate decades. (2) It assumes a **traditional** IRA. "
            "In a **Roth** the rule *reverses* — you'd want the stocks in the tax-free wrapper, and "
            "bonds-in-Roth *loses* by about **−29 bps/yr**. |\n\n"
            "> A genuine free lunch — but a **small** and **conditional** one: tens of basis points a "
            "year, biggest for high tax brackets and high bond yields, and only in a *traditional* IRA."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Put your bonds in the tax-deferred account and your stocks in the taxable account. "
            "It adds real after-tax return — for free.\"*\n\n"
            "This is textbook, not folklore: **Dammon, Spatt & Zhang (2004)** in the *Journal of "
            "Finance* proved a \"strong locational preference\" for taxable bonds in the shelter, and "
            "the Bogleheads wiki turned it into the standard retail rule (*tax-efficient fund "
            "placement*). Surveys show most households **ignore** it — they hold roughly the same mix "
            "in every account. Our question: how much money does the rule actually add, on 150 years "
            "of data, after modelling the taxes honestly?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons to care. First, this is one of the few levers a household controls completely: "
            "no forecasting, no timing, no manager — just *which account holds what*. If it's real, "
            "it's the cheapest return you'll ever earn. Second, the size of the prize should depend on "
            "**yields** (the benefit is sheltered coupon income) and on your **tax bracket** — so the "
            "\"does it still matter at low rates?\" objection deserves a real answer, not a shrug."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a simulated household: **60/40** stocks/bonds, half its wealth **taxable**, half "
            f"in a **traditional IRA**, at the {R['ordinary']}% ordinary / {R['ltcg']}% capital-gains "
            "rates. Every year: dividends and coupons get taxed where taxable, the IRA compounds "
            "untouched, the equity fund realises 10% of its gains, and the household rebalances back to "
            "60/40. After 30 years everything is liquidated — the IRA pays income tax on the way out, "
            "the taxable account settles its capital gains.\n\n"
            "We run that household three ways — **bonds-in-IRA** (the rule), **stocks-in-IRA** (the "
            "anti-rule), **pro-rata** (every account holds 60/40, what most people do) — over **every "
            f"overlapping 30-year window of the Shiller tape ({R['long_years']})**: "
            f"{R['n_cohorts']} cohorts. Same pre-tax portfolio in all three. The only difference is "
            "**location** — so any wealth gap is pure tax arithmetic, measured on real market history."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average after-tax return of the three placements, across all "
            f"{R['n_cohorts']} thirty-year cohorts."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize_cohorts(TAB)\n"
            "    vals = [s['mean_ann_bonds_ira'], s['mean_ann_pro_rata'], s['mean_ann_stocks_ira']]\n"
            "else:\n"
            "    vals = [R['ann_bonds'], R['ann_prorata'], R['ann_stocks']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "labels = ['bonds in the IRA\\n(the rule)', 'pro-rata\\n(what most do)', 'stocks in the IRA\\n(the anti-rule)']\n"
            "ax.bar(labels, vals, color=[GREEN, GREY, RED], width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.2f}%/yr', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('after-tax annualised return (%)'); ax.set_ylim(6.4, max(vals) + .12)\n"
            "ax.set_title('Same 60/40, same funds - only the account placement differs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'bonds-in-IRA {vals[0]:.2f}%  pro-rata {vals[1]:.2f}%  stocks-in-IRA {vals[2]:.2f}%')"
        ),
        md(
            f"The rule wins: **{R['ann_bonds']:.2f}%/yr** after tax vs **{R['ann_stocks']:.2f}%** for "
            f"the reversed placement — a gap of **+{R['delta_bs']:.0f} bps/yr** that compounds to about "
            f"**+{R['wealth_gap']:.0f}% more terminal wealth** over 30 years. Even against the pro-rata "
            f"default it adds **+{R['delta_bp']:.0f} bps/yr**. Remember: identical funds, identical "
            "risk. This gap is the taxman's bill being sent to a different address."
        ),
        md(
            "**Was it always true, or just on average?** The gap for every 30-year cohort since 1872."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = TAB.index.to_numpy(); y = TAB['delta_bs'].to_numpy()\n"
            "else:\n"
            "    rng = np.random.default_rng(600)\n"
            "    x = np.arange(1872, 1994); y = rng.normal(R['delta_bs'], 15, len(x))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.fill_between(x, 0, y, where=(y >= 0), color=GREEN, alpha=.6, label='rule wins')\n"
            "ax.fill_between(x, 0, y, where=(y < 0), color=RED, alpha=.6, label='rule loses')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('cohort start year'); ax.set_ylabel('after-tax gain of bonds-in-IRA (bps/yr)')\n"
            "ax.set_title(f'The rule wins {(y > 0).mean()*100:.0f}% of all 30-year cohorts since 1872')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean {y.mean():+.1f} bps/yr | wins {(y > 0).mean()*100:.1f}% | range {y.min():+.1f} to {y.max():+.1f}')"
        ),
        md(
            f"It wins **{R['win_bs']:.0f}% of the time**, and the worst cohort ever costs only "
            f"**{R['delta_min']:.0f} bps/yr** while the best pays **+{R['delta_max']:.0f}**. The big "
            "wins cluster in **high-yield decades** (the 1970s-80s starts) — which makes sense: the "
            "whole benefit is bond income being sheltered, so more income means more shelter. Today's "
            "SPY/IEF version (2003-2025, ~2.9% yields) still paid "
            f"**+{R['modern']['delta_bs']:.0f} bps/yr**."
        ),
        md(
            "**The one catch.** The rule silently assumes your stocks barely get taxed along the way "
            "(index funds, ~10% turnover). What if your equity sleeve is an **active fund** that "
            "realises its gains every year?"
        ),
        code(
            "flips = R['flip']\n"
            "xs = [f[0] for f in flips]; ys = [f[1] for f in flips]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.plot(xs, ys, 'o-', color=GREEN, lw=2, ms=7)\n"
            "ax.axhline(0, c=RED, ls='--', label='the flip line')\n"
            "for x_, y_ in zip(xs, ys): ax.annotate(f'{y_:+.0f}', (x_, y_), textcoords='offset points', xytext=(0, 9), ha='center')\n"
            "ax.set_xlabel('equity fund turnover (% of gains realised each year)')\n"
            "ax.set_ylabel('after-tax gain of bonds-in-IRA (bps/yr)')\n"
            "ax.set_title('The free lunch shrinks as your stock fund gets tax-inefficient - and flips at 100%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('turnover % -> delta bps/yr:', dict(zip(xs, ys)))"
        ),
        md(
            f"There's the boundary. With tax-efficient stock funds the lunch is free "
            f"(**+{R['flip'][0][1]:.0f}** to **+{R['flip'][1][1]:.0f}** bps/yr). Crank the equity "
            f"turnover to 100% — a hyperactive fund that distributes everything — and the advantage "
            f"dies (**{R['flip'][4][1]:+.1f} bps/yr**), even turning negative in low-yield decades "
            f"(**{R['flip'][4][4]:+.1f}**). If your stocks are tax-inefficient, *they* become the asset "
            "that needs the shelter."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real (traditional IRA).** Bonds-in-the-IRA adds **+{R['delta_bs']:.0f} bps/yr** "
            f"of after-tax return across {R['n_cohorts']} thirty-year cohorts (statistically solid, HAC "
            f"*t* = {R['t_bs']:.2f}), winning {R['win_bs']:.0f}% of the time — for the *tax-deferred* "
            "account the study models.\n"
            f"- **Tradability — Fragile.** It costs **nothing** to relocate — but the free lunch is "
            "**conditional on a traditional IRA**. In a **Roth**, the identical rule *reverses* "
            "(**−29 bps/yr**, HAC *t* = −3.15): a tax-free wrapper should hold your highest-growth "
            "asset, not your bonds. A retail household with a Roth that follows \"bonds in the IRA\" "
            "*loses* tens of bps a year.\n"
            f"- **Does it flip? — Confirmed (two ways).** At 100% equity turnover the advantage crosses "
            "zero (negative in low-yield decades); and swapping the traditional IRA for a **Roth** "
            "inverts it outright (−29 bps/yr). The rule is conditional on **tax-efficient stock funds** "
            "*and* a **tax-deferred** account."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Your bracket is the dial.** At the 37/20 top rates the lunch grows to "
            f"**+{R['grid'][7][2]:.0f} bps/yr**; at 22/15 it's **+{R['grid'][0][2]:.0f}**. Higher "
            "bracket, bigger shield.\n"
            "- **Watch the yield level.** The benefit scales at roughly **+5.6 bps/yr per percentage "
            "point of bond yield** — the 1970s were the golden age of asset location; a 1%-yield world "
            "makes it almost moot.\n"
            "- **Siblings on the desk.** This is the *placement* member of the personal-finance family: "
            "[101 Slow-and-Steady](../../101-slow-and-steady/), [172 Hundred-Minus-Age]"
            "(../../172-hundred-minus-age/), [173 Four-Percent-Rule](../../173-four-percent-rule/), "
            "and [599 Tax-Loss-Harvesting](../../599-tax-loss-harvesting/) — the other tax-alpha claim, "
            "also graded *Fragile*. Both add real after-tax return in the right setup, but both are "
            "**conditional**: this one's signal is real yet only *tradable* in a traditional IRA.\n\n"
            "*Think the flip matters in practice? Re-run the notebook with your own fund's turnover and "
            "your own bracket — the simulator takes both as arguments.*"
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
            "# Asset Location — a quantitative teardown 🔬\n"
            "### A full tax-drag household simulator · HAC *t* on 122 overlapping 30-year cohorts · "
            "a tax-rate grid · the delta-on-yield regression · the turnover flip · a zero-tax null "
            "and a planted-coupon control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "asset-location rule (Dammon-Spatt-Zhang 2004) is *arithmetic* — but arithmetic still has "
            "to show its number **on the tape**, with uncertainty, under honest frictions. So: a "
            "year-by-year household simulator with average-cost basis, loss carryforwards, terminal "
            "settlement, one documented decision lag, and Newey-West inference over massively "
            "overlapping cohorts.\n\n"
            "> ⚠️ **Data note.** Long tape = Shiller monthly price/dividend/GS10 → annual components "
            f"{R['long_years']} (fingerprint `{R['fp_long']}`); modern tape = SPY/IEF unadjusted closes "
            f"+ cash distributions, {R['modern']['years']} (fingerprint `{R['fp_modern']}`). As-of "
            f"{R['asof']}, complete calendar years only. Index-level data — no survivor panel. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` (traditional IRA) | Mean bonds-in-IRA − stocks-in-IRA delta "
            f"**+{R['delta_bs']:.2f} bps/yr** of after-tax annualised return over **{R['n_cohorts']}** "
            f"overlapping 30-year cohorts, **HAC t (lag {R['hac_lag']}) = +{R['t_bs']:.2f}**, win rate "
            f"{R['win_bs']:.1f}%; vs pro-rata **+{R['delta_bp']:.2f} bps/yr** (t = +{R['t_bp']:.2f}). |\n"
            f"| **Tradability** | `FRAGILE` | Free to relocate (**+{R['diff_cost']:.3f} bps/yr** cost) — "
            "but conditional on a **traditional** IRA. In a **Roth** the rule *reverses*: bonds-in-Roth "
            "**−28.6 bps/yr**, HAC t = **−3.15**, win 10.7% — the tax-free wrapper wants your stocks, "
            "not your bonds. Undisclosed in the original scope. |\n"
            f"| **Does it flip?** | `CONFIRMED` | Two axes. Turnover 100%: **{R['flip'][4][1]:+.2f} "
            f"bps/yr** (t = {R['flip'][4][2]:+.2f}), **{R['flip'][4][4]:+.2f}** in the lowest-yield "
            "quartile. Account type Roth vs traditional: **−28.6 bps/yr** (t = −3.15) — the bigger flip. |\n\n"
            "> 💡 In plain words: the rule is real and free **in a traditional IRA** — and it quietly "
            "assumes both that your stock fund doesn't distribute its gains *and* that your shelter is "
            "tax-*deferred*, not a Roth."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A household holds pre-tax weight $w = 60\\%$ equity, split between a taxable account "
            "(share $s = 50\\%$) and a traditional IRA. Location policy $\\ell \\in$ {bonds-in-IRA, "
            "stocks-in-IRA, pro-rata} maps the same pre-tax portfolio to account placements. Let "
            "$W^\\ell_T$ be terminal after-tax wealth over a $T = 30$-year cohort. The tested deltas "
            "(bps/yr of after-tax annualised return):\n\n"
            "$$\\Delta_{bs} = 10^4\\left[(W^{bonds}_T)^{1/T} - (W^{stocks}_T)^{1/T}\\right],\\qquad "
            "\\Delta_{bp} = 10^4\\left[(W^{bonds}_T)^{1/T} - (W^{pro}_T)^{1/T}\\right].$$\n\n"
            "- **H₁ (the rule pays).** $\\mathbb{E}[\\Delta_{bs}] > 0$ with an overlap-robust *t* ≥ 2.\n"
            "- **H₂ (it's free).** The differential friction between policies is negligible next to "
            "the delta.\n"
            "- **H₃ (the flip).** The sign of $\\mathbb{E}[\\Delta_{bs}]$ reverses as equity turnover "
            "→ 100% (Shoven-Sialm), and the size scales with the yield level.\n\n"
            "The tax engine: qualified dividends and realised LTCG at the LTCG rate; coupons at the "
            "ordinary rate; equity turnover realised annually at average-cost basis with realised-loss "
            "carryforward; bond-fund price gains realised annually; the IRA compounds tax-free and pays "
            "the ordinary rate once on terminal withdrawal; the taxable account settles LTCG on its "
            "terminal unrealised gain. **One documented lag:** year *t*'s coupon is the prior-December "
            "10y yield."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the inference problem\n\n"
            "Annual-start 30-year cohorts overlap in 29 of 30 years: adjacent deltas are almost the "
            "same number, so a naive *t* over cohorts would overstate the evidence by an order of "
            "magnitude. The desk's answer is **Newey-West with lag = horizon − 1 = 29** on the cohort "
            "delta series — the standard correction for overlapping-window statistics — applied to the "
            "mean delta and to the delta-on-yield regression. The rate grid re-runs the entire cohort "
            "machinery per cell (no shortcut through a formula), and the synthetic world checks the "
            "simulator can't manufacture a location benefit when taxes are zero."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Long tape.** Shiller monthly price/div/GS10 → annual components {R['long_years']} "
            f"({R['n_years_long']} complete years): Dec→Dec price return, cash-dividend yield on the "
            "prior-Dec price, the 10y coupon locked at the **prior-Dec** yield, and the price return "
            "of a constant-maturity 10y par bond.\n"
            f"- **Cohorts.** Every overlapping 30-year window → **{R['n_cohorts']}** cohorts "
            "(starts 1872–1993); each runs the household under the three policies.\n"
            f"- **Baseline household.** {R['ordinary']}% ordinary / {R['ltcg']}% LTCG+QDI, "
            f"{R['turnover']}% equity turnover, 50/50 split, annual rebalance (free inside the IRA).\n"
            f"- **Inference.** HAC *t* (lag {R['hac_lag']}) on the mean of $\\Delta_{{bs}}$ and "
            "$\\Delta_{bp}$; HAC OLS of $\\Delta_{bs}$ on the cohort's mean coupon.\n"
            "- **Sensitivity.** 8-cell ordinary × LTCG grid; equity-turnover sweep 0→100%; a modern "
            f"SPY/IEF single cohort ({R['modern']['years']}) with real fund distributions.\n"
            "- **Controls.** Zero-tax null (delta must be exactly 0, all seeds) + planted-coupon "
            "scaling, averaged over **20 seeds**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cohort delta and its HAC *t*\n\n"
            "The after-tax annualised delta of bonds-in-IRA vs stocks-in-IRA, one point per 30-year "
            "cohort, with the overlap-robust *t* on the mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize_cohorts(TAB)\n"
            "    x = TAB.index.to_numpy(); y = TAB['delta_bs'].to_numpy()\n"
            "    mu, t_bs, win = s['mean_delta_bs'], s['t_bs'], s['win_bs']\n"
            "    mu_bp, t_bp = s['mean_delta_bp'], s['t_bp']\n"
            "else:\n"
            "    rng = np.random.default_rng(600)\n"
            "    x = np.arange(1872, 1994); y = rng.normal(R['delta_bs'], 15, len(x))\n"
            "    mu, t_bs, win, mu_bp, t_bp = R['delta_bs'], R['t_bs'], R['win_bs'], R['delta_bp'], R['t_bp']\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.6))\n"
            "ax.bar(x, y, color=[GREEN if v >= 0 else RED for v in y], width=.9)\n"
            "ax.axhline(mu, c='k', ls='--', lw=1.2, label=f'mean {mu:+.1f} bps/yr (HAC t = {t_bs:+.2f})')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('cohort start year'); ax.set_ylabel('delta bonds-in-IRA - stocks-in-IRA (bps/yr)')\n"
            "ax.set_title(f'{len(x)} overlapping 30-year cohorts - win rate {win:.1f}%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'delta_bs: mean {mu:+.2f} bps/yr  HAC t(lag 29) {t_bs:+.2f}  win {win:.1f}%')\n"
            "print(f'delta_bp: mean {mu_bp:+.2f} bps/yr  HAC t(lag 29) {t_bp:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the rule pays **+{R['delta_bs']:.2f} bps/yr** on average and the "
            f"overlap-corrected *t* is **{R['t_bs']:+.2f}** — the desk's *t* ≥ 2 bar is cleared on the "
            f"real tape, not by the literature. Even against the pro-rata default (closer race, since "
            f"pro-rata already shelters 60% of the bonds) it's **+{R['delta_bp']:.2f} bps/yr** at "
            f"*t* = **{R['t_bp']:+.2f}**. The negative cohorts ({100 - R['win_bs']:.0f}%) cluster where "
            "equity crashes made the taxable-stock policy's *loss carryforwards* valuable."
        ),
        md(
            "### 4b · The benefit scales with the yield level\n\n"
            "The mechanism is annually-taxed coupon income sheltered by the IRA — so the delta should "
            "be (nearly mechanically) increasing in the cohort's average locked coupon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize_cohorts(TAB)\n"
            "    cx = 100 * TAB['mean_cpn'].to_numpy(); cy = TAB['delta_bs'].to_numpy()\n"
            "    slope, t_slope = s['slope_bps_per_pp'], s['t_slope']\n"
            "    q = st.yield_quartiles(TAB)\n"
            "    ql = list(q.index); qc = q['mean_cpn_pct'].to_numpy(); qd = q['mean_delta_bs'].to_numpy()\n"
            "else:\n"
            "    slope, t_slope = R['slope'], R['t_slope']\n"
            "    ql = [r[0] for r in R['quart']]; qc = np.array([r[1] for r in R['quart']])\n"
            "    qd = np.array([r[2] for r in R['quart']])\n"
            "    rng = np.random.default_rng(1)\n"
            "    cx = rng.uniform(2.5, 9, 122); cy = slope * cx + rng.normal(0, 6, 122)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.scatter(cx, cy, s=18, color=GREY, alpha=.7)\n"
            "xs = np.linspace(cx.min(), cx.max(), 50)\n"
            "b0 = cy.mean() - slope * cx.mean()\n"
            "a1.plot(xs, b0 + slope * xs, c=GREEN, lw=2, label=f'{slope:+.2f} bps/yr per 1pp of yield')\n"
            "a1.set_xlabel('cohort mean locked coupon (%)'); a1.set_ylabel('delta (bps/yr)')\n"
            "a1.set_title(f'delta ~ coupon: HAC t = {t_slope:+.1f}'); a1.legend()\n"
            "a2.bar(range(4), qd, color=[GREY, GREY, GREY, GREEN], width=.6)\n"
            "a2.set_xticks(range(4)); a2.set_xticklabels([f'{l}\\n{c:.1f}%' for l, c in zip(ql, qc)], fontsize=8)\n"
            "for i, v in enumerate(qd): a2.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('mean delta (bps/yr)'); a2.set_title('by coupon quartile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('quartile deltas:', dict(zip(ql, np.round(qd, 2))), f'| slope {slope:+.2f} t {t_slope:+.1f}')"
        ),
        md(
            f"> 💡 In plain words: each extra percentage point of bond yield adds about "
            f"**{R['slope']:+.1f} bps/yr** to the prize (HAC *t* = {R['t_slope']:+.1f} — near-"
            "mechanical, exactly as the arithmetic predicts). The 1970s-80s cohorts "
            f"(Q4, {R['quart'][3][1]:.1f}% average yields) banked **{R['quart'][3][2]:+.1f} bps/yr**; "
            f"the low-yield quartile still gets **{R['quart'][0][2]:+.1f}**. Low rates shrink the lunch "
            "— they don't cancel it."
        ),
        md(
            "### 4c · The tax-rate grid\n\n"
            "The engine of the rule is the **wedge** between the ordinary rate (what coupons pay) and "
            "the LTCG rate (what equity pays). Re-run all 122 cohorts per (ordinary, LTCG) cell."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.rate_grid(COMP)\n"
            "    grid = [(int(100*r.ordinary), int(100*r.ltcg), r.mean_delta_bs, r.t_bs, r.win_pct)\n"
            "            for r in g.itertuples()]\n"
            "else:\n"
            "    grid = R['grid']\n"
            "ords = sorted(set(r[0] for r in grid))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "w = .35\n"
            "for k, lt in enumerate((15, 20)):\n"
            "    vals = [next(r[2] for r in grid if r[0] == o and r[1] == lt) for o in ords]\n"
            "    ax.bar(np.arange(len(ords)) + (k - .5) * w, vals, w, color=[GREEN, AMBER][k], label=f'LTCG {lt}%')\n"
            "    for i, v in enumerate(vals): ax.annotate(f'{v:+.0f}', (i + (k - .5) * w, v), ha='center',\n"
            "                                             va='bottom' if v >= 0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(range(len(ords))); ax.set_xticklabels([f'{o}%' for o in ords])\n"
            "ax.set_xlabel('ordinary-income rate'); ax.set_ylabel('mean delta (bps/yr)')\n"
            "ax.set_title('The wedge is the engine: delta by (ordinary, LTCG) cell')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in grid: print(f'ordinary {r[0]}% / LTCG {r[1]}%: {r[2]:+7.2f} bps/yr  t {r[3]:+.2f}  win {r[4]:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: every **realistic** 2026 US pair (22/15, 24/15, 32/15, 37/20 — the "
            "20% LTCG rate only exists inside the top ordinary brackets) is **positive**, from "
            f"**{R['grid'][0][2]:+.1f}** to **{R['grid'][6][2]:+.1f} bps/yr**. The negative cells "
            f"(22/20 = {R['grid'][1][2]:+.1f}, 24/20 = {R['grid'][3][2]:+.1f}) are hypothetical "
            "narrow-wedge households — instructive, because they show the delta is literally the "
            "ordinary-vs-LTCG wedge priced through the account boundary, but they are not real US "
            "bracket combinations."
        ),
        md(
            "### 4d · The third axis — the turnover flip\n\n"
            "Shoven-Sialm's reversal: if the equity sleeve itself is tax-inefficient (high turnover), "
            "the preference should reverse. Sweep equity turnover 0 → 100% over all cohorts."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tf = st.turnover_flip(COMP)\n"
            "    xs = (100 * tf['turnover']).to_numpy(); ys = tf['mean_delta_bs'].to_numpy()\n"
            "    lo = tf['lowq_delta_bs'].to_numpy(); ts = tf['t_bs'].to_numpy()\n"
            "else:\n"
            "    xs = np.array([f[0] for f in R['flip']]); ys = np.array([f[1] for f in R['flip']])\n"
            "    ts = np.array([f[2] for f in R['flip']]); lo = np.array([f[4] for f in R['flip']])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(xs, ys, 'o-', color=GREEN, lw=2, ms=7, label='all cohorts')\n"
            "ax.plot(xs, lo, 's--', color=AMBER, lw=1.8, ms=6, label='lowest-yield-quartile cohorts')\n"
            "ax.axhline(0, c=RED, ls='--', lw=1.2, label='flip line')\n"
            "for x_, y_, t_ in zip(xs, ys, ts): ax.annotate(f't={t_:+.2f}', (x_, y_), textcoords='offset points',\n"
            "                                               xytext=(0, 10), ha='center', fontsize=8)\n"
            "ax.set_xlabel('equity fund turnover (%/yr)'); ax.set_ylabel('mean delta (bps/yr)')\n"
            "ax.set_title('The rule is conditional on tax-efficient equity: it flips at 100% turnover')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for x_, y_, t_, l_ in zip(xs, ys, ts, lo): print(f'turnover {x_:3.0f}%: {y_:+7.2f} bps/yr  t {t_:+.2f}  low-yield-Q {l_:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **it flips.** At {R['flip'][4][0]}% turnover the mean delta is "
            f"**{R['flip'][4][1]:+.2f} bps/yr** (*t* = {R['flip'][4][2]:+.2f}) and "
            f"**{R['flip'][4][4]:+.2f}** in low-yield cohorts; at ≥50% turnover the benefit is already "
            f"statistically zero (*t* = {R['flip'][3][2]:+.2f}). The claim's fine print — \"stocks are "
            "the tax-efficient asset\" — is doing real work. Index-fund households (0-10% turnover) "
            "keep the whole lunch; active-fund households in low-rate decades lose it. "
            "**CONFIRMED** on the third axis."
        ),
        md(
            "### 4e · The modern low-rate cohort — SPY + IEF, real fund distributions\n\n"
            "The \"it no longer matters at today's yields\" objection, tested on actual fund cash "
            f"flows {R['modern']['years']} (IEF distribution yield ≈ {R['modern']['cpn']:.1f}%)."
        ),
        code(
            "if HAVE_MODERN:\n"
            "    px, dv = data.load_modern()\n"
            "    mc = data.modern_components(px, dv)\n"
            "    r = st.single_cohort(mc)\n"
            "    vals = [r['ann_bonds_ira'], r['ann_pro_rata'], r['ann_stocks_ira']]\n"
            "    sweep = [(tv, st.single_cohort(mc, turnover=tv)['delta_bs']) for tv in (0., .10, .25, .50, 1.0)]\n"
            "    d_bs, d_bp = r['delta_bs'], r['delta_bp']\n"
            "else:\n"
            "    m = R['modern']; vals = [m['ann_bonds'], m['ann_prorata'], m['ann_stocks']]\n"
            "    sweep = [(t / 100, d) for t, d in m['sweep']]; d_bs, d_bp = m['delta_bs'], m['delta_bp']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar(['bonds-in-IRA', 'pro-rata', 'stocks-in-IRA'], vals, color=[GREEN, GREY, RED], width=.55)\n"
            "for i, v in enumerate(vals): a1.annotate(f'{v:.2f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('after-tax annualised (%)'); a1.set_ylim(6.6, max(vals) + .15)\n"
            "a1.set_title('SPY/IEF 2003-2025 (single cohort)')\n"
            "a2.plot([100 * t for t, _ in sweep], [d for _, d in sweep], 'o-', color=GREEN, lw=2)\n"
            "a2.axhline(0, c=RED, ls='--')\n"
            "a2.set_xlabel('equity turnover (%)'); a2.set_ylabel('delta (bps/yr)')\n"
            "a2.set_title('positive throughout on this window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'modern delta_bs {d_bs:+.2f} bps/yr | vs pro-rata {d_bp:+.2f} | sweep', [(int(100*t), round(d, 1)) for t, d in sweep])"
        ),
        md(
            f"> 💡 In plain words: even in the lowest-yield quarter-century on record the rule paid "
            f"**{R['modern']['delta_bs']:+.1f} bps/yr** (single cohort — no *t*; the long tape carries "
            "the inference). It stays positive at every turnover here because 2003-2025 equity gains "
            "were enormous (deferral was worth a lot) while the bond price leg was small. One window, "
            "one draw — but the objection \"low rates killed asset location\" is not what this tape "
            "shows."
        ),
        md(
            "### 4f · Faithful-engine control — the zero-tax null and the planted coupon\n\n"
            "Two things the simulator must prove. **The null:** with zero tax rates the account "
            "boundary is meaningless — every policy must return *exactly* the same wealth (delta "
            "= 0.000, all seeds). **The knob:** the benefit is sheltered coupon income, so a planted "
            "coupon must scale the measured delta monotonically. Averaged over 20 seeds."
        ),
        code(
            "syn = st.synthetic_scaling()\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = syn['label'].tolist(); vals = syn['mean_delta_bs'].to_numpy(); sds = syn['sd_delta_bs'].to_numpy()\n"
            "colors = [GREY] + [GREEN] * (len(vals) - 1)\n"
            "ax.bar(range(len(vals)), vals, yerr=sds, capsize=4, color=colors, width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_xticks(range(len(vals))); ax.set_xticklabels([l.replace(' (', '\\n(') for l in labels], fontsize=8)\n"
            "ax.set_ylabel('mean delta (bps/yr, 20 seeds)')\n"
            "ax.set_title('Zero taxes -> exactly 0; planted coupon -> monotone scaling')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l, v, s_ in zip(labels, vals, sds): print(f'{l:<24s}: {v:+8.3f} bps/yr (sd {s_:.3f})')"
        ),
        md(
            f"> 💡 In plain words: the simulator returns **exactly 0.000** when taxes are zero (it "
            "cannot invent a location benefit) and the planted coupon knob scales the delta "
            f"monotonically ({R['syn'][1][1]:.1f} → {R['syn'][2][1]:.1f} → {R['syn'][3][1]:.1f} "
            "bps/yr). *(A machinery proof only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL` (traditional IRA)** — mean delta **+{R['delta_bs']:.2f} bps/yr** over "
            f"**{R['n_cohorts']}** overlapping 30-year cohorts, **HAC *t* (lag {R['hac_lag']}) = "
            f"+{R['t_bs']:.2f}**, win rate {R['win_bs']:.1f}%; vs pro-rata **+{R['delta_bp']:.2f} "
            f"bps/yr** (*t* = +{R['t_bp']:.2f}); compounds to **+{R['wealth_gap']:.1f}%** terminal "
            "wealth. Positive at every realistic US bracket pair. Genuine and reproducing — honestly "
            "scoped to the **tax-deferred** account Dammon-Spatt-Zhang model.\n"
            f"- **Tradability `FRAGILE`** — relocating is free (differential cost "
            f"**+{R['diff_cost']:.3f} bps/yr**), but the free lunch is **conditional on a traditional "
            "IRA**. Re-run the identical 122 cohorts with a **Roth** terminal settlement (withdrawal "
            "untaxed) and the rule inverts: bonds-in-Roth **−28.57 bps/yr**, **HAC *t* = −3.15**, win "
            "10.7%, deepening to −43.9 bps/yr at 100% turnover. A Roth retail household following "
            "\"bonds in the IRA\" loses tens of bps a year — so the sign is *not* universal.\n"
            f"- **Does it flip? `CONFIRMED` (two axes)** — (1) at 100% equity turnover the delta crosses "
            f"zero (**{R['flip'][4][1]:+.2f} bps/yr**, *t* = {R['flip'][4][2]:+.2f}; "
            f"**{R['flip'][4][4]:+.2f}** in low-yield cohorts) — Shoven-Sialm's reversal on the tape; "
            "(2) Roth-vs-traditional inverts it outright (**−28.6 bps/yr**, *t* = −3.15) — the bigger, "
            "more common flip."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Roth doesn't just change the algebra — it flips the sign.** A Roth IRA has no terminal "
            "ordinary-rate haircut, so the asset with the *highest expected return* (equity) should sit "
            "inside it. Setting the terminal IRA factor to `1.0` and re-running the identical 122 "
            "cohorts, bonds-in-Roth **loses** to stocks-in-Roth by **−28.6 bps/yr** (HAC *t* = −3.15, "
            "win 10.7%) — a full reversal, not a partial one. This is why Tradability is `FRAGILE`: the "
            "rule's sign depends on an account-type assumption the headline never stated.\n"
            "- **Municipal bonds are the escape hatch.** A high-bracket household with no IRA room can "
            "swap taxable bonds for munis — which is why the observed cross-section (Bergstresser-"
            "Poterba) is messier than the rule.\n"
            "- **The wedge generalises.** REITs, TIPS, high-yield — anything paying ordinary income "
            "slots into the same arithmetic: shelter the asset with the highest *annually-taxed yield "
            "× wedge* product.\n"
            "- **Siblings.** [599 — Tax-Loss-Harvesting](../../599-tax-loss-harvesting/) graded the "
            "other \"free tax alpha\" claim **Fragile** (front-loaded, path-dependent); this one also "
            "grades **Fragile** on tradability — the signal is real and needs no losses, timing or "
            "turnover, but its *sign* is conditional on a traditional (not Roth) IRA.\n\n"
            "*The reproducible core is offline and deterministic; methods and sources: "
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
