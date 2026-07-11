"""Generate the two narrative notebooks for Study 655 (Ivy Portfolio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached VTI/VEU/VNQ/AGG/
DBC/BIL tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance VTI/VEU/VNQ/AGG/DBC/
# BIL, 2007-06-30 -> 2026-06-30, 229 months; SMA timer live 2008-03-31, 220 months).
R = dict(
    asof="2026-06-30", fp="6e25c7829c9c",
    start="2007-06-30", end="2026-06-30", n_months=229, n_years=19.1,
    sma_live="2008-03-31", n_timed=220,
    ivy_cagr=5.70, ivy_vol=12.89, ivy_sharpe=0.392, ivy_dd=-44.8,
    b64_cagr=7.84, b64_vol=10.25, b64_sharpe=0.657, b64_dd=-32.3,
    vti_cagr=10.61, vti_vol=16.07, vti_sharpe=0.627, vti_dd=-50.8,
    act_ivy_64_ann=-1.69, act_ivy_64_t=-1.12,
    act_ivy_vti_ann=-5.03, act_ivy_vti_t=-3.10,
    boot_ivy_64=(-0.265, -0.470, -0.018, 1.7),
    boot_ivy_vti=(-0.234, -0.434, 0.003, 2.8),
    assets=dict(
        VTI=(10.61, 0.627, -50.8), VEU=(5.00, 0.289, -58.4), VNQ=(5.40, 0.291, -64.6),
        AGG=(3.06, 0.385, -17.1), DBC=(1.33, 0.096, -74.6),
    ),
    ivy_w_cagr=5.87, ivy_w_sharpe=0.411, ivy_w_dd=-44.8,
    b64_w_cagr=8.43, b64_w_sharpe=0.716, b64_w_dd=-29.9,
    timed_cagr=4.67, timed_vol=6.87, timed_sharpe=0.520, timed_dd=-13.3,
    dd_ratio_ts=0.30, dd_ratio_t64=0.44,
    act_ts_ann=-1.77, act_ts_t=-0.79,
    act_t64_ann=-3.84, act_t64_t=-2.26,
    boot_ts=(0.109, -0.240, 0.414, 68.4),
    boot_t64=(-0.196, -0.589, 0.166, 13.5),
    exgfc_ivy=(8.06, 0.644, -19.3), exgfc_64=(10.05, 0.924, -20.6),
    exgfc_act_ivy64=(-1.68, -1.33),
    exgfc_timed=(5.78, 0.673, -13.1), exgfc_static_w=(8.23, 0.660, -19.3),
    exgfc_act_ts=(-2.68, -1.92), exgfc_act_t64=(-4.32, -2.78),
    turn_static=28, turn_timed=361, flips_per_yr=8.89,
    timed_cost5=(4.67, 0.520, -13.3), timed_cost10=(4.49, 0.494, -13.4),
    rb_base_cagr=4.38, rb_base_sharpe=0.372, rb_base_dd=-32.3,
    rb_dd_beat=100, rb_sharpe_beat=95, rb_n_seeds=40,
    syn_null_sharpe_share=49.5, syn_null_dd_share=69.5,
    syn_planted_sharpe_share=75.0, syn_planted_dd_share=82.5,
    syn_persistence=0.85, syn_outer=10, syn_inner=20,
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Risk_reduction%3F: Confirmed](https://img.shields.io/badge/Risk_reduction%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from ivy_portfolio import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RET, CLOSE = data.monthly_panel()
    SIG = st.sma_signal(CLOSE)
    RF = RET[data.CASH]
else:
    RET = CLOSE = SIG = RF = None
print("real cache present:", HAVE_REAL, "| months:", (0 if RET is None else len(RET)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Could you have just bought what Yale buys? 🌿\n"
            "### Faber's Ivy Portfolio — five ETFs, 20% each, endowment-style diversification "
            "for the rest of us. Does it work?\n\n"
            + BADGES +
            "Big university endowments (Yale, Harvard) have famously beaten simple stock/bond "
            "portfolios for decades by spreading bets across US stocks, foreign stocks, real "
            "estate, bonds and commodities — asset classes that (in theory) don't all crash "
            "together. In 2009, Mebane Faber's *The Ivy Portfolio* said: you can approximate "
            "this with five liquid ETFs, 20% each, and — optionally — a simple trend rule "
            "(hold each asset only while it's above its own 10-month average) to dodge the "
            "worst of each one's bear markets.\n\n"
            "We rebuilt it on 19 years of real prices. The diversification story and the "
            "timing story turn out to be **two very different claims** — one busted, one real.\n\n"
            "> 📓 **Plain-language layer.** Want the bootstrap CIs, the random-timing control "
            "and the ex-GFC robustness check? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** VTI/VEU/VNQ/AGG/DBC/BIL, real daily closes, 2007→2026 "
            "(bound by BIL's inception). Sharpe is always excess-of-cash. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the 5-asset mix beat a plain 60/40? | **No.** Its risk-adjusted return "
            f"(Sharpe **{R['ivy_sharpe']:.2f}**) is robustly *worse* than 60/40's "
            f"(**{R['b64_sharpe']:.2f}**) — and its drawdown (**{R['ivy_dd']:.0f}%**) is worse "
            f"too (60/40: **{R['b64_dd']:.0f}%**). |\n"
            "| Why? | Two of the five legs were themselves disasters: commodities (DBC) "
            "returned almost nothing for 19 years with a −75% drawdown of its own; REITs "
            "(VNQ) crashed −65% and moved *with* stocks in 2008, not against them. |\n"
            f"| Does the 10-month timing rule help? | **It cuts risk, hard** — max drawdown "
            f"falls from **{R['ivy_w_dd']:.0f}%** to **{R['timed_dd']:.0f}%** — and it's *real* "
            "(a random \"switch at random times\" control gets nowhere close). |\n"
            "| Does the timing rule make you money? | **No.** It costs return — timed Ivy "
            f"significantly *underperforms* 60/40 (by **{abs(R['act_t64_ann']):.1f}%/yr**). "
            "It's a crash shield, not a return engine. |\n\n"
            "> The diversification story is a myth on this tape. The timing story is real — "
            "and it's exactly the trade-off it's supposed to be: less pain, less gain."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Endowments don't beat the market by stock-picking — they beat it by owning "
            "asset classes that don't all fall at once: US stocks, foreign stocks, real "
            "estate, bonds, commodities. Put 20% in each, rebalance, and you capture most of "
            "the diversification benefit institutions pay millions in fees for. Add a "
            "10-month trend filter on each piece and you dodge the worst bear markets too.\"*"
            "\n\n— the pitch of *The Ivy Portfolio* (2009), built on Faber's earlier single-"
            "asset timing research."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this is one of the most actionable ideas in retail investing: five "
            "cheap ETFs, one spreadsheet's worth of rebalancing, and you're running a scaled-"
            "down Yale endowment. If it's not true — if the extra legs are just extra risk "
            "without extra reward — then \"more asset classes\" is a diversification *story*, "
            "not a diversification *result*, and a plain 60/40 was the better trade all along."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The static test.** 20% each in VTI/VEU/VNQ/AGG/DBC, monthly rebalance, "
            f"{R['start']} → {R['end']} ({R['n_months']} months). Compare Sharpe (excess of "
            "cash) and max drawdown to a 60/40 (VTI/AGG) with a **bootstrap confidence "
            "interval on the Sharpe gap** — not just a point estimate.\n"
            "- **The timing test.** Each of the five sleeves held only when its price is above "
            "its own trailing 10-month average, else that 20% sits in T-bills. Compared "
            "against static Ivy *and* against a **random-timing control** that switches at "
            "random times but spends the same total time invested — the only fair way to ask "
            "\"is this specific rule doing something, or just holding less risk on average?\"\n"
            "- **The luck check.** Does the story survive dropping the 2008 financial crisis "
            "entirely?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the diversification claim.** Sharpe and max drawdown, three ways."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.static_weights(RET.index)\n"
            "    static_net, _ = st.weighted_portfolio(RET, sw, cost_bps=5.0)\n"
            "    b64w = st.sixty_forty_weights(RET.index)\n"
            "    b64_net, _ = st.weighted_portfolio(RET, b64w, cost_bps=5.0)\n"
            "    ps, pb, pv = st.perf(static_net, RF), st.perf(b64_net, RF), st.perf(RET['VTI'], RF)\n"
            "    sharpes = [ps['sharpe'], pb['sharpe'], pv['sharpe']]\n"
            "    dds = [ps['max_dd']*100, pb['max_dd']*100, pv['max_dd']*100]\n"
            "else:\n"
            "    sharpes = [R['ivy_sharpe'], R['b64_sharpe'], R['vti_sharpe']]\n"
            "    dds = [R['ivy_dd'], R['b64_dd'], R['vti_dd']]\n"
            "labels = ['Ivy static\\n(20% x 5)', '60/40\\n(VTI/AGG)', '100% VTI']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(labels, sharpes, color=[RED, GREEN, GREY], width=.6)\n"
            "for i, v in enumerate(sharpes): a1.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('Sharpe (excess of cash)'); a1.set_title('Risk-adjusted return')\n"
            "a2.bar(labels, dds, color=[RED, GREEN, GREY], width=.6)\n"
            "for i, v in enumerate(dds): a2.annotate(f'{v:.0f}%', (i, v), ha='center', va='top')\n"
            "a2.set_ylabel('max drawdown (%)'); a2.set_title('Worst peak-to-trough loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', {l: round(s,3) for l,s in zip(labels, sharpes)})\n"
            "print('maxDD :', {l: round(d,1) for l,d in zip(labels, dds)})"
        ),
        md(
            f"Ivy static's Sharpe (**{R['ivy_sharpe']:.2f}**) is *worse* than 60/40's "
            f"(**{R['b64_sharpe']:.2f}**), and its drawdown (**{R['ivy_dd']:.0f}%**) is worse "
            f"too. A bootstrap check (2,000 resamples) puts the entire 95% confidence interval "
            f"of that Sharpe gap **below zero** — this isn't noise, it's a real, if modest, "
            "underperformance. **Why?** Look at the five legs on their own:"
        ),
        code(
            "names = list(R['assets'].keys())\n"
            "cagr = [R['assets'][n][0] for n in names]\n"
            "sh = [R['assets'][n][1] for n in names]\n"
            "dd = [R['assets'][n][2] for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [AMBER if n in ('VNQ','DBC') else GREY for n in names]\n"
            "ax.bar(names, sh, color=cols, width=.6)\n"
            "for i, (s, d) in enumerate(zip(sh, dd)): ax.annotate(f'Sharpe {s:.2f}\\nmaxDD {d:.0f}%',\n"
            "    (i, s), ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title('Two of Faber\\'s five legs were themselves disasters')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: (c, s, d) for n, c, s, d in zip(names, cagr, sh, dd)})"
        ),
        md(
            "**Commodities (DBC)** returned almost nothing for 19 years (Sharpe **0.10**, its "
            "own **−75%** drawdown) — a well-documented casualty of the negative \"roll yield\" "
            "commodity index funds paid through the 2011-2020 bear market. **REITs (VNQ)** "
            "crashed **−65%** in 2008 — moving *with* the stock market in the crisis it was "
            "supposed to diversify *against*. Diversification only pays off when the "
            "diversifiers are actually good investments on their own; two of Faber's five "
            "weren't, on this stretch of history.\n\n"
            "**Now, the timing rule.** Hold each asset only above its 10-month average:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tw = st.timed_weights(SIG)\n"
            "    timed_net, _ = st.weighted_portfolio(RET, tw, cost_bps=5.0)\n"
            "    pt = st.perf(timed_net, RF)\n"
            "    ps_w = st.perf(static_net.loc[timed_net.index], RF)\n"
            "    dd3 = [ps_w['max_dd']*100, pt['max_dd']*100]\n"
            "else:\n"
            "    dd3 = [R['ivy_w_dd'], R['timed_dd']]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Ivy static', 'Ivy TIMED\\n(10-mo SMA)'], dd3, color=[RED, GREEN], width=.55)\n"
            "for i, v in enumerate(dd3): ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='top')\n"
            "ax.set_ylabel('max drawdown (%)')\n"
            "ax.set_title('The timing rule is a genuine crash shield')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('maxDD static vs timed:', dd3)"
        ),
        md(
            f"Drawdown falls from **{R['ivy_w_dd']:.0f}%** to **{R['timed_dd']:.0f}%** — less "
            "than a third of the pain. Is that real, or is it just \"any strategy that's out "
            "of the market some of the time has a smaller drawdown\"? We tested that directly: "
            f"a control that switches at *random* times but holds each asset for the *same "
            f"total time* comes nowhere close — the real rule beats **{R['rb_dd_beat']:.0f}%** "
            f"of {R['rb_n_seeds']} such random controls on drawdown. This is genuine trend "
            "information, not an accounting trick.\n\n"
            "**But it isn't free.** Timing costs return:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = st.active_stats(timed_net, b64_net.loc[timed_net.index])\n"
            "    active = a['ann_pct']\n"
            "else:\n"
            "    active = R['act_t64_ann']\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.4))\n"
            "ax.bar(['timed Ivy vs 60/40'], [active], color=RED, width=.4)\n"
            "ax.annotate(f'{active:+.2f}%/yr', (0, active), ha='center',\n"
            "    va='top' if active < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('active return vs 60/40 (%/yr)')\n"
            "ax.set_title('The crash shield has a real cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'active timed vs 60/40: {active:+.2f}%/yr')"
        ),
        md(
            f"Timed Ivy trails a plain 60/40 by **{abs(R['act_t64_ann']):.1f}%/yr** — "
            "statistically real, not noise. So: real drawdown protection, real return cost. "
            "Neither Ivy version — static or timed — ends up paying you more than the boring "
            "benchmark for the trouble of running it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** *Real* on the timing rule's drawdown cut (validated against "
            "a random-timing control, holds up with 2008 removed) · *None* on the static "
            "diversification claim (Sharpe robustly worse than 60/40).\n"
            "- **Tradability — Mirage.** Cheap to run, either version — but neither beats a "
            "free 60/40 on a risk-adjusted basis, so there's nothing to actually get paid for.\n"
            "- **\"Risk reduction or alpha?\" — Confirmed: risk reduction, not alpha.** The "
            "10-month rule is a genuine crash shield with a genuine cost, exactly the trade-off "
            "Faber himself described — now shown to hold across five assets, not just one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The diversification lesson generalizes.** Adding legs only helps if the legs "
            "are themselves decent investments — \"more asset classes\" is not automatically "
            "\"more diversification\". Commodities and REITs both had genuinely bad 15-year "
            "stretches; a portfolio built on 2003-2007 backtest data couldn't have known that "
            "in advance, and neither can we about the *next* 15 years.\n"
            "- **The timing lesson generalizes too** — see [110-faber-timing](../../110-faber-timing/), "
            "the same 10-month rule on a single asset (SPY), with the identical risk-reduction-"
            "not-alpha verdict.\n"
            "- **Sibling studies:** [68-all-weather](../../68-all-weather/) (risk parity, "
            "different weighting logic), [144-permanent-portfolio](../../144-permanent-portfolio/) "
            "and [203-golden-butterfly](../../203-golden-butterfly/) (equal-weight, no timing, "
            "different legs), [592-dual-momentum-gem](../../592-dual-momentum-gem/) (a momentum "
            "decision tree, not a diversified blend).\n\n"
            "*Think a different five assets, or a different trend window, would fix the "
            "diversification leg? The engine's right there — show a bootstrap CI that clears "
            "zero, then we'll talk.*"
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
            "# The Ivy Portfolio — a quantitative teardown 🔬\n"
            "### Bootstrap Sharpe-difference CIs on the static allocation · a matched-exposure "
            "random-timing control on the SMA overlay · an ex-GFC robustness check · costs & "
            "turnover · a 10-seed x 20-seed synthetic null/planted control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Faber's Ivy Portfolio bundles two separable claims — a 5-asset equal-weight "
            "diversifier, and an independent per-sleeve 10-month SMA timer — that this study "
            "measures and stamps **separately**, on purpose.\n\n"
            "> ⚠️ **Data note.** VTI/VEU/VNQ/AGG/DBC/BIL daily auto-adjusted closes (yfinance), "
            f"cached; monthly window **{R['start']} → {R['end']}** ({R['n_months']} months, "
            f"bound by BIL's 2007-05-30 inception), SMA timer live from **{R['sma_live']}** "
            f"({R['n_timed']} months). No survivorship — five broad, still-listed ETFs. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            f"[`docs/results.md`](../docs/results.md) (fingerprint `{R['fp']}`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | static Ivy Sharpe **{R['ivy_sharpe']:.2f}** vs 60/40 "
            f"**{R['b64_sharpe']:.2f}**: bootstrap 95% CI of the gap "
            f"**[{R['boot_ivy_64'][1]:+.3f}, {R['boot_ivy_64'][2]:+.3f}]** (entirely negative) "
            f"— *None*. Timer cuts max DD {R['ivy_w_dd']:.0f}% -> {R['timed_dd']:.0f}%, beats "
            f"{R['rb_dd_beat']:.0f}%/{R['rb_n_seeds']} matched-exposure random-timing shuffles "
            "on drawdown — *Real* |\n"
            f"| **Tradability** | `MIRAGE` | turnover {R['turn_static']}%/yr (static) / "
            f"{R['turn_timed']}%/yr (timed), costs ≤36 bps/yr — cheap, but neither arm beats "
            "60/40 risk-adjusted |\n"
            f"| **Risk reduction or alpha?** | `CONFIRMED` (risk reduction) | timed vs 60/40 "
            f"active **{R['act_t64_ann']:+.2f}%/yr** (HAC t = **{R['act_t64_t']:+.2f}**) — "
            "genuine cost for genuine protection |\n\n"
            "> 💡 In plain words: the diversification story is a myth on this tape; the timing "
            "story is real, and it's exactly the trade-off it's supposed to be."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Two separable hypotheses:\n\n"
            "- **H₁ (diversification).** An equal-weight blend of five imperfectly-correlated "
            "asset classes (VTI/VEU/VNQ/AGG/DBC) has a *better* risk-adjusted profile "
            "(Sharpe, drawdown) than a plain 60/40 (VTI/AGG).\n"
            "- **H₂ (timing).** Applying a 10-month SMA independently to each sleeve — hold "
            "above the average, else cash — further improves the risk-adjusted profile, or at "
            "minimum trades return for a genuine reduction in drawdown (not merely an "
            "artefact of reduced average exposure).\n\n"
            "We find **H₁ REJECTED** (Sharpe gap vs 60/40 bootstrap-certified negative) and "
            "**H₂ PARTIALLY SUPPORTED** — the risk-reduction half is real and validated; the "
            "\"further improves\" half is not (timed Ivy still trails 60/40)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "H₁ is a **risk-adjusted** claim, not a mean-return claim, so the primary test is a "
            "**circular block bootstrap** on the Sharpe *difference* (2,000 resamples, "
            "6-month blocks — stable across 3/6/12-month blocks in a sensitivity check), the "
            "same convention used on "
            "[144-permanent-portfolio](../../144-permanent-portfolio/) and "
            "[203-golden-butterfly](../../203-golden-butterfly/). H₂ needs a control that "
            "isolates *timing* from *exposure*: a **matched-exposure random-timing baseline** "
            "(each sleeve's in/out calendar shuffled, holding its time-in-market fixed), "
            "averaged over **40 seeds** — mirrors the random-switching control in "
            "[592-dual-momentum-gem](../../592-dual-momentum-gem/) and the random-timing "
            "control in [110-faber-timing](../../110-faber-timing/). Both hypotheses are "
            "re-checked on an ex-GFC sub-period so neither verdict rests on one crisis quarter."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** VTI/VEU/VNQ/AGG/DBC/BIL, {R['start']} → {R['end']} "
            f"({R['n_months']} months). As-of 2026-06-30 (last complete month).\n"
            "- **Rebalance.** Both arms reset to target weights every month; turnover is "
            "tracked as the sum of absolute weight changes, costed one-way x NAV per rebalance.\n"
            f"- **Timing.** 10-month SMA on each sleeve's price, evaluated at month *t-1*'s "
            f"close, applied to month *t*'s return — one lag, at the source. Live from "
            f"{R['sma_live']}.\n"
            "- **H1 test.** Bootstrap Sharpe-diff CI, Ivy static vs 60/40 and vs VTI; HAC t on "
            "active return as a secondary check.\n"
            "- **H2 test.** Timed vs static and vs 60/40 (HAC t, bootstrap Sharpe-diff CI); "
            "40-seed matched-exposure random-timing control; ex-GFC sub-period.\n"
            "- **Control.** Synthetic 5-asset + cash world, planted trend-persistence knob; "
            "the exposure-matched detector must sit at chance under the null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — does the 5-asset mix out-Sharpe a 60/40?\n\n"
            "Bootstrap Sharpe-difference CI (2,000 resamples, 6-month blocks) plus the HAC t "
            "on active return, both arms."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.static_weights(RET.index)\n"
            "    static_net, static_turn = st.weighted_portfolio(RET, sw, cost_bps=5.0)\n"
            "    b64w = st.sixty_forty_weights(RET.index)\n"
            "    b64_net, _ = st.weighted_portfolio(RET, b64w, cost_bps=5.0)\n"
            "    boot = st.bootstrap_sharpe_diff(static_net, b64_net, rf=RF, seed=655)\n"
            "    a = st.active_stats(static_net, b64_net)\n"
            "    point, lo, hi = boot['point'], boot['ci95'][0], boot['ci95'][1]\n"
            "    ann_t = a['ann_pct'], a['hac_t']\n"
            "else:\n"
            "    point, lo, hi = R['boot_ivy_64'][0], R['boot_ivy_64'][1], R['boot_ivy_64'][2]\n"
            "    ann_t = R['act_ivy_64_ann'], R['act_ivy_64_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 3.6))\n"
            "ax.errorbar([point], [0], xerr=[[point-lo],[hi-point]], fmt='o', color=RED,\n"
            "    ecolor=RED, capsize=6, markersize=9)\n"
            "ax.axvline(0, c='k', lw=1, ls='--')\n"
            "ax.set_yticks([]); ax.set_xlabel('Sharpe(Ivy) - Sharpe(60/40), 95% bootstrap CI')\n"
            "ax.set_title('The whole CI sits below zero: NOT a better risk-adjusted mix')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe diff {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')\n"
            "print(f'active return {ann_t[0]:+.2f}%/yr  HAC t = {ann_t[1]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the point estimate ({R['boot_ivy_64'][0]:+.3f}) is negative, "
            f"and the ENTIRE 95% CI (**[{R['boot_ivy_64'][1]:+.3f}, {R['boot_ivy_64'][2]:+.3f}]**"
            f") stays below zero, stable across 3/6/12-month block sizes. Ivy's Sharpe is "
            "**robustly worse** than 60/40's — H₁ is rejected, not merely unproven."
        ),
        md(
            "### 4b · Why — two of the five legs were themselves poor investments\n\n"
            "Standalone CAGR / Sharpe / max drawdown for each of the five sleeves."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {t: st.perf(RET[t], RF) for t in data.ASSETS}\n"
            "    names = list(rows.keys())\n"
            "    sh = [rows[n]['sharpe'] for n in names]\n"
            "    dd = [rows[n]['max_dd']*100 for n in names]\n"
            "else:\n"
            "    names = list(R['assets'].keys())\n"
            "    sh = [R['assets'][n][1] for n in names]\n"
            "    dd = [R['assets'][n][2] for n in names]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "cols = [AMBER if n in ('VNQ', 'DBC') else GREY for n in names]\n"
            "a1.bar(names, sh, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('Sharpe (excess of cash)'); a1.set_title('Standalone risk-adjusted return')\n"
            "a2.bar(names, dd, color=cols, width=.6)\n"
            "a2.set_ylabel('max drawdown (%)'); a2.set_title('Standalone worst drawdown')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: (round(s,3), round(d,1)) for n,s,d in zip(names, sh, dd)})"
        ),
        md(
            "> 💡 In plain words: DBC's Sharpe (**0.10**) is near zero with its own **−75%** "
            "drawdown — commodity futures paid a negative roll yield through the 2011-2020 "
            "bear market (Bhardwaj, Gorton & Rouwenhorst 2015). VNQ crashed **−65%** in the "
            "GFC, correlated *with* equities rather than against them. A blend is only as good "
            "as its worst-behaved legs, and two of Faber's five underperformed badly here."
        ),
        md(
            "### 4c · H₂ — the 10-month SMA overlay: risk reduction, alpha, or both?\n\n"
            "Timed vs static Ivy (same window), and timed vs 60/40."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tw = st.timed_weights(SIG)\n"
            "    timed_net, timed_turn = st.weighted_portfolio(RET, tw, cost_bps=5.0)\n"
            "    common = timed_net.index\n"
            "    ps_c = st.perf(static_net.loc[common], RF)\n"
            "    pb_c = st.perf(b64_net.loc[common], RF)\n"
            "    pt = st.perf(timed_net, RF)\n"
            "    dds = [ps_c['max_dd']*100, pb_c['max_dd']*100, pt['max_dd']*100]\n"
            "    shs = [ps_c['sharpe'], pb_c['sharpe'], pt['sharpe']]\n"
            "else:\n"
            "    dds = [R['ivy_w_dd'], R['b64_w_dd'], R['timed_dd']]\n"
            "    shs = [R['ivy_w_sharpe'], R['b64_w_sharpe'], R['timed_sharpe']]\n"
            "labels = ['Ivy static', '60/40', 'Ivy TIMED']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.2))\n"
            "a1.bar(labels, dds, color=[RED, GREY, GREEN], width=.6)\n"
            "for i, v in enumerate(dds): a1.annotate(f'{v:.1f}%', (i, v), ha='center', va='top')\n"
            "a1.set_ylabel('max drawdown (%)'); a1.set_title('Drawdown: TIMED wins clearly')\n"
            "a2.bar(labels, shs, color=[RED, GREEN, AMBER], width=.6)\n"
            "for i, v in enumerate(shs): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Sharpe (excess of cash)'); a2.set_title('Sharpe: TIMED improves on static, still trails 60/40')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('maxDD:', dict(zip(labels, dds))); print('Sharpe:', dict(zip(labels, shs)))"
        ),
        md(
            f"> 💡 In plain words: drawdown falls from **{R['ivy_w_dd']:.0f}%** (static) to "
            f"**{R['timed_dd']:.0f}%** (timed) — better than even 60/40's **{R['b64_w_dd']:.0f}%**"
            f". Sharpe improves too (**{R['ivy_w_sharpe']:.2f}** → **{R['timed_sharpe']:.2f}**) "
            f"but the bootstrap CI on timed-vs-static Sharpe still spans zero "
            f"(**[{R['boot_ts'][1]:+.3f}, {R['boot_ts'][2]:+.3f}]**, not certified), and timed "
            f"Ivy's Sharpe never overtakes 60/40's ({R['b64_w_sharpe']:.2f}). Active return "
            f"vs 60/40 is **{R['act_t64_ann']:+.2f}%/yr at HAC t = {R['act_t64_t']:+.2f}** — "
            "certified, and negative. Real risk reduction; no alpha."
        ),
        md(
            "### 4d · Is the drawdown cut real timing skill, or just less exposure?\n\n"
            "Matched-exposure random-timing control: shuffle each sleeve's in/out calendar at "
            "random, holding its time-in-market **fixed**. Averaged over 40 seeds "
            "(single-seed baselines are banned on this desk)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.random_timing_baseline(RET, SIG, cost_bps=5.0, n_seeds=40, base_seed=655)\n"
            "    actual_sh, actual_dd = rb['actual']['sharpe'], rb['actual']['max_dd']*100\n"
            "    base_sh, base_dd = rb['base_sharpe_mean'], rb['base_dd_mean']*100\n"
            "    dd_beat, sh_beat = rb['dd_beat_share']*100, rb['sharpe_beat_share']*100\n"
            "else:\n"
            "    actual_sh, actual_dd = R['timed_sharpe'], R['timed_dd']\n"
            "    base_sh, base_dd = R['rb_base_sharpe'], R['rb_base_dd']\n"
            "    dd_beat, sh_beat = R['rb_dd_beat'], R['rb_sharpe_beat']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(['actual SMA\\ntiming', 'random-timing\\nbaseline (mean)'], [actual_dd, base_dd],\n"
            "       color=[GREEN, GREY], width=.55)\n"
            "for i, v in enumerate([actual_dd, base_dd]): a1.annotate(f'{v:.1f}%', (i, v), ha='center', va='top')\n"
            "a1.set_ylabel('max drawdown (%)'); a1.set_title(f'Beats {dd_beat:.0f}% of 40 shuffles on DD')\n"
            "a2.bar(['actual SMA\\ntiming', 'random-timing\\nbaseline (mean)'], [actual_sh, base_sh],\n"
            "       color=[GREEN, GREY], width=.55)\n"
            "for i, v in enumerate([actual_sh, base_sh]): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Sharpe'); a2.set_title(f'Beats {sh_beat:.0f}% of 40 shuffles on Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'actual dd {actual_dd:.1f}% vs baseline {base_dd:.1f}%  (beats {dd_beat:.0f}%)')\n"
            "print(f'actual Sh {actual_sh:.2f} vs baseline {base_sh:.2f}  (beats {sh_beat:.0f}%)')"
        ),
        md(
            f"> 💡 In plain words: a random timer with the SAME average exposure per sleeve "
            f"gets nowhere close — the actual 10-month rule beats **{R['rb_dd_beat']:.0f}%** of "
            f"40 random-timing shuffles on drawdown and **{R['rb_sharpe_beat']:.0f}%** on "
            "Sharpe. This is genuine, if partial, trend information — not an artefact of "
            "spending less time invested."
        ),
        md(
            "### 4e · Sub-period — is this all 2008?\n\n"
            "Both findings re-run with 2007-07 → 2009-06 (the GFC) dropped entirely."
        ),
        code(
            "if HAVE_REAL:\n"
            "    def ex_gfc(s):\n"
            "        return s[(s.index < '2007-07-01') | (s.index > '2009-06-30')]\n"
            "    sg, bg = ex_gfc(static_net), ex_gfc(b64_net)\n"
            "    a_g = st.active_stats(sg, bg)\n"
            "    tg = ex_gfc(timed_net)\n"
            "    sg2 = static_net.loc[tg.index]\n"
            "    a_ts_g = st.active_stats(tg, sg2)\n"
            "    a_t64_g = st.active_stats(tg, ex_gfc(b64_net.loc[common]))\n"
            "    exgfc_vals = [a_g['ann_pct'], a_ts_g['ann_pct'], a_t64_g['ann_pct']]\n"
            "    exgfc_ts = [a_g['hac_t'], a_ts_g['hac_t'], a_t64_g['hac_t']]\n"
            "else:\n"
            "    exgfc_vals = [R['exgfc_act_ivy64'][0], R['exgfc_act_ts'][0], R['exgfc_act_t64'][0]]\n"
            "    exgfc_ts = [R['exgfc_act_ivy64'][1], R['exgfc_act_ts'][1], R['exgfc_act_t64'][1]]\n"
            "labels = ['Ivy-60/40', 'timed-static', 'timed-60/40']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(labels, exgfc_vals, color=[RED if v<0 else GREEN for v in exgfc_vals], width=.55)\n"
            "for i, (v, t) in enumerate(zip(exgfc_vals, exgfc_ts)):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\n(t={t:+.2f})', (i, v), ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('active return, ex-GFC (%/yr)')\n"
            "ax.set_title('Both findings survive dropping 2008 entirely')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(labels, zip(exgfc_vals, exgfc_ts))))"
        ),
        md(
            f"> 💡 In plain words: ex-GFC, Ivy static still trails 60/40 by "
            f"**{R['exgfc_act_ivy64'][0]:+.2f}%/yr** (*t* = {R['exgfc_act_ivy64'][1]:+.2f}, "
            "below the certification bar but consistent) and the timer still trails 60/40 by "
            f"**{R['exgfc_act_t64'][0]:+.2f}%/yr** (*t* = {R['exgfc_act_t64'][1]:+.2f}, "
            "certified) — and its drawdown cut still holds "
            f"({R['exgfc_static_w'][2]:.1f}% → {R['exgfc_timed'][2]:.1f}%). Neither result is "
            "\"one great trade in 2008.\""
        ),
        md(
            "### 4f · Costs and turnover\n\n"
            "One-way bps x NAV on the total absolute weight change at each monthly rebalance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ann_static = float(static_turn.mean()*12)*100\n"
            "    ann_timed = float(timed_turn.mean()*12)*100\n"
            "    rows = []\n"
            "    for cb in (5.0, 10.0):\n"
            "        tn, _ = st.weighted_portfolio(RET, tw, cost_bps=cb)\n"
            "        p = st.perf(tn, RF)\n"
            "        rows.append((cb, p['cagr']*100, p['sharpe']))\n"
            "else:\n"
            "    ann_static, ann_timed = R['turn_static'], R['turn_timed']\n"
            "    rows = [(5.0, R['timed_cost5'][0], R['timed_cost5'][1]),\n"
            "            (10.0, R['timed_cost10'][0], R['timed_cost10'][1])]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(['Ivy static', 'Ivy timed'], [ann_static, ann_timed], color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([ann_static, ann_timed]): a1.annotate(f'{v:.0f}%/yr', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('one-way turnover (%/yr)'); a1.set_title('Timing trades ~13x more, but still cheap in bps')\n"
            "cbs = [r[0] for r in rows]; cagrs = [r[1] for r in rows]\n"
            "a2.plot(cbs, cagrs, marker='o', color=AMBER)\n"
            "a2.set_xlabel('one-way cost (bps)'); a2.set_ylabel('timed Ivy CAGR (%)')\n"
            "a2.set_title('Cost sweep: a rounding error either way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('turnover %/yr:', ann_static, ann_timed)\n"
            "print('cost sweep:', rows)"
        ),
        md(
            f"> 💡 In plain words: static Ivy trades almost nothing ({R['turn_static']}%/yr); "
            f"even the timer's ~{R['flips_per_yr']:.0f} sleeve-flips/year only cost "
            "**~18-36 bps/yr** across the 5/10 bps sweep. Costs are not why this underperforms "
            "60/40 — the legs and the timing trade-off are."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "A synthetic 5-asset + cash world where a single regime drives all five legs, with "
            "a TUNABLE ``persistence`` knob. Unconditional means (persistence=0, a fair 50/50 "
            "coin flip) are pinned at realistic positive asset premia **above cash** in every "
            "world, so hiding in cash can never mechanically win — only genuine trend "
            "information can. Same matched-exposure random-timing test, run inside the "
            "synthetic world, averaged over **10 world-seeds x 20 inner shuffle-seeds**."
        ),
        code(
            "def synthetic_beat_shares(persistence, n_outer=10, n_inner=20, n_months=300):\n"
            "    sh, dd = [], []\n"
            "    for s in range(n_outer):\n"
            "        panel = data.synthetic_world(persistence=persistence, seed=655+s, n_months=n_months)\n"
            "        closes = panel[data.ASSETS]\n"
            "        rets = pd.DataFrame(index=panel.index)\n"
            "        rets[data.ASSETS] = closes.pct_change()\n"
            "        rets[data.CASH] = panel[data.CASH+'_lvl'].pct_change()\n"
            "        rets = rets.dropna(how='all')\n"
            "        sig_syn = st.sma_signal(closes)\n"
            "        r = st.random_timing_baseline(rets, sig_syn, cost_bps=5.0, n_seeds=n_inner, base_seed=1000+s)\n"
            "        sh.append(r['sharpe_beat_share']); dd.append(r['dd_beat_share'])\n"
            "    return float(np.mean(sh))*100, float(np.mean(dd))*100\n"
            "\n"
            "null_sh, null_dd = synthetic_beat_shares(0.0)\n"
            "planted_sh, planted_dd = synthetic_beat_shares(0.85)\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['null\\n(persistence=0)', 'planted\\n(persistence=0.85)'], [null_sh, planted_sh],\n"
            "       color=[GREY, GREEN], width=.5)\n"
            "for i, v in enumerate([null_sh, planted_sh]): ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='chance = 50%')\n"
            "ax.set_ylabel('Sharpe-beat-share vs random-timing (%)')\n"
            "ax.set_title('The detector sits at chance under the null, rises with genuine trend')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: Sharpe-beat {null_sh:.1f}%  DD-beat {null_dd:.1f}%')\n"
            "print(f'planted: Sharpe-beat {planted_sh:.1f}%  DD-beat {planted_dd:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: under the null the Sharpe-beat-share sits at "
            f"**{R['syn_null_sharpe_share']:.1f}%** — chance is 50% — and rises to "
            f"**{R['syn_planted_sharpe_share']:.1f}%** once a genuine trend is planted. The "
            f"null's DD-beat-share (**{R['syn_null_dd_share']:.1f}%**, above chance) is a "
            "named, honest artefact: a trailing-average rule mechanically tends to have "
            "already de-risked after a run of bad returns, capping some downside even with "
            "zero informational content — a structural property of trend-following, not "
            "skill. The real tape's beat-shares (100% / 95%) exceed even the planted-0.85 "
            "world, consistent with one strong sustained trend episode (the GFC) in the "
            "sample. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real* on the SMA timer's drawdown cut "
            f"({R['ivy_w_dd']:.0f}% → {R['timed_dd']:.0f}%, beats {R['rb_dd_beat']:.0f}%/"
            f"{R['rb_n_seeds']} matched-exposure shuffles on drawdown, {R['rb_sharpe_beat']:.0f}%"
            f"/{R['rb_n_seeds']} on Sharpe; synthetic detector at chance under the null, rises "
            "with planted persistence) · *None* on the static diversification's Sharpe claim "
            f"(bootstrap 95% CI **[{R['boot_ivy_64'][1]:+.3f}, {R['boot_ivy_64'][2]:+.3f}]**, "
            "entirely negative, stable across block sizes).\n"
            f"- **Tradability `MIRAGE`** — turnover {R['turn_static']}%/yr (static) / "
            f"{R['turn_timed']}%/yr (timed), costs ≤36 bps/yr either way — cheap, but neither "
            "arm beats a free 60/40 on a risk-adjusted basis (static: robustly worse Sharpe; "
            f"timed: active {R['act_t64_ann']:+.2f}%/yr at HAC t = {R['act_t64_t']:+.2f} vs "
            "60/40, Sharpe-diff CI still spans zero).\n"
            f"- **\"Risk reduction or alpha?\" `CONFIRMED` (risk reduction)** — genuine, "
            "repeated (ex-GFC) drawdown protection with a genuine, certified return cost. "
            "Exactly Faber's own single-asset framing (Study 110), now validated across a "
            "5-asset composite."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson on H₁:** \"more asset classes\" is not automatically \"more "
            "diversification\" — the marginal legs need to be decent investments on their own. "
            "Commodity futures' roll-yield drag (Erb & Harvey 2006; Bhardwaj, Gorton & "
            "Rouwenhorst 2015) and REITs' equity-like crash behaviour in systemic crises "
            "(2008) are both well-documented, and both bit here.\n"
            "- **The general lesson on H₂** is Study 110's, now confirmed to generalize: a "
            "trailing-average timer is a real, validated crash shield whose cost is a real, "
            "certified return drag — not a free lunch, but not nothing either.\n"
            "- **Natural sequel:** does risk-parity weighting (Study 68's approach) rescue the "
            "static allocation by down-weighting DBC and VNQ's outsized volatility instead of "
            "equal-weighting blindly? The engine here is ready for that comparison.\n"
            "- **Dedup map:** [68-all-weather](../../68-all-weather/) (risk parity, different "
            "quartet), [110-faber-timing](../../110-faber-timing/) (the same SMA rule, single "
            "asset), [144-permanent-portfolio](../../144-permanent-portfolio/) and "
            "[203-golden-butterfly](../../203-golden-butterfly/) (equal-weight, no timing, "
            "different legs), [592-dual-momentum-gem](../../592-dual-momentum-gem/) (a "
            "momentum decision tree, not a diversified blend).\n\n"
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
