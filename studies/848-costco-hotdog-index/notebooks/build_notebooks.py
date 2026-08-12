"""Generate the two narrative notebooks for Study 848 ("Costco Hot-Dog Index").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pull under ../_cache/ (COST, SPY, XLP) and the real FRED/BLS CPIAUCSL series
(BLS ``CUSR0000SA0``, cached from the BLS API; embedded real values as the offline
fallback) from the package; on a cache miss they fall back to the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere (fast).
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


# Frozen headline numbers — mirror of docs/results.md. COST/SPY/XLP are month-end
# total-return via yfinance; CPI is the REAL FRED/BLS CPIAUCSL series (BLS CUSR0000SA0).
# as-of 2026-06-30, fingerprint [cost,spy,cpi] = cbd3b6799d57.
R = dict(
    win="2000-01 → 2026-06", n=318, asof="2026-06-30", fp="cbd3b6799d57", years=26.5,
    combo=1.50, base_year=1985, cpi_1985=107.6, cpi_now=332.57,
    cpi_mult=3.09, real_now=0.49, eroded_pct=67.6, infl_adj=4.64, giveaway=3.14,
    cost_mult=28.7, cost_cagr=13.5, cost_real=14.6, cost_10k=286935,
    spy_mult=8.5, spy_cagr=8.4, spy_real=4.3, spy_10k=85393,
    cpi_mult_win=2.0, cpi_cagr=2.6, cpi_10k=19644, xlp_mult=6.75,
    yoy_peak=9.0,
    cost_beta=0.115, cost_beta_t=0.13, cost_beta_r2=0.01,
    spy_beta=0.879, spy_beta_t=1.27, xlp_beta=0.302, xlp_beta_t=0.72,
    cost_pred_t=0.89,
    gap_xlp=-0.186, gap_xlp_t=-0.27, gap_spy_t=-1.45,
    gap_spy_pred_t=1.75, gap_xlp_pred_t=1.43,
    costspy_alpha_t=1.37,
    placebo_p=0.88, era_early_t=-0.05, era_late_t=0.36,
    timer_sh=0.34, cost_bh_sh=0.46, spy_sh=0.40, timer_switches=4.1, timer_invested=0.49,
    syn_edge=8, syn_slope=7.72, syn_t=7.02, syn_gap_t=6.05, syn_null_t=-0.24,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Real-price_erosion: Confirmed](https://img.shields.io/badge/Real--price_erosion-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # quantlab (optional)
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from hotdog_index import data, strategy as st

CPI = data.load_cpi()                                # REAL FRED/BLS CPIAUCSL (BLS CUSR0000SA0)
PANEL = data.build_panel()                           # empty if the COST/SPY/XLP cache is absent
HAVE = not PANEL.empty
print("market cache present:", HAVE, "| CPI months:", CPI.index[0].date(), "->", CPI.index[-1].date())
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The $1.50 hot dog that beat inflation? 🌭\n"
            "### \"Costco froze the combo since 1985 — and the stock crushes inflation, so buy COST\", in plain English\n\n"
            + BADGES +
            "It's one of the great retail legends: the Costco **hot-dog-and-soda combo has cost "
            "$1.50 since 1985** — through Volcker's hangover, three recessions, and the worst "
            "inflation in 40 years. The internet loves it as an *anti-inflation* mascot. And the "
            "leap the story quietly makes is: *Costco has such pricing power that its stock outruns "
            "inflation and the market too — so just buy COST as your inflation hedge.*\n\n"
            "This notebook separates the part that's **true** from the part that's a **story**. Yes, "
            "the $1.50 combo really has melted in real terms. Yes, COST has been a phenomenal stock. "
            "But does COST actually *hedge inflation* — does it move up when inflation surprises to "
            "the upside — any better than a boring can-of-soup staples fund? That last claim is the "
            "only tradable one, and it's the one we test.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo null and the "
            "inflation-beta regressions? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** CPI here is the **real** FRED/BLS "
            "`CPIAUCSL` series (BLS `CUSR0000SA0`), fetched from the BLS public API and cached (one "
            "point, 2025-10, interpolated — the SA suspension). COST, SPY and XLP are real month-end "
            "total-return prices via yfinance. Every chart is drawn by the code beside it."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Has the $1.50 combo really lost value? | **Yes — dramatically.** In 1985 dollars it now "
            f"costs about **\\${R['real_now']:.2f}** — a **{R['eroded_pct']:.0f}%** real haircut. To hold "
            f"its 1985 purchasing power it'd sell for **\\${R['infl_adj']:.2f}**. |\n"
            "| Has COST beaten inflation and the market? | **Yes — massively.** \\$10,000 in COST (2000) "
            f"became **\\${R['cost_10k']:,.0f}**, vs **\\${R['spy_10k']:,.0f}** in SPY and just "
            f"**\\${R['cpi_10k']:,.0f}** of CPI. |\n"
            "| So is COST an *inflation hedge*? | **No.** When inflation *surprises* higher, COST barely "
            f"budges (*t* = **{R['cost_beta_t']:.2f}**) — and it's **no better than a boring staples fund** "
            f"(gap *t* = **{R['gap_xlp_t']:.2f}**). |\n"
            "| Then why does the story feel true? | Because COST **went up a lot**, and we remember the "
            "winner. \"It rose\" and \"it hedges inflation\" are different claims. |\n\n"
            "> The erosion is real. COST's returns are real. The **inflation-hedge trade** is a mirage — "
            "COST is a great *growth* stock, not an inflation shield."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Costco has frozen the hot-dog combo at $1.50 since 1985 — that's the pricing power of a "
            "membership juggernaut. So COST will protect you from inflation: its stock outruns CPI and "
            "the S&P regardless of the macro. Buy it as an inflation hedge.\"*\n\n"
            "It's steelman-able. Costco leans on **recurring membership fees** (a near-annuity that "
            "resets with inflation) and legendary discipline, and it has genuinely trounced the market "
            "for decades. If any consumer name deserves an \"inflation-proof\" halo, it's this one."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "\"Buy this one stock and you're hedged against inflation\" is a big, specific promise — the "
            "kind that sells newsletters. But it bundles **three different claims**: (1) the frozen combo "
            "loses real value (arithmetic), (2) COST has beaten inflation historically (a fact about the "
            "past), and (3) COST *reacts* to inflation shocks in a protective way you can trade (a "
            "mechanism). Only (3) is an *inflation hedge*. (1) is a calculator. (2) is hindsight about a "
            "known winner. Let's look at all three — honestly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks:\n\n"
            "1. **The calculator.** Deflate the frozen $1.50 by CPI — how much real value has it shed?\n"
            "2. **The race.** Grow $10,000 in COST, in SPY, and by CPI since 2000 — who wins, and by how "
            "much *after* inflation?\n"
            "3. **The hedge test.** When inflation *surprises* higher, does COST rise — and does it rise "
            "**more than a plain staples basket (XLP)**? That's what \"pricing-power inflation hedge\" "
            "actually predicts.\n\n"
            "**What would make us say \"tradable hedge\"?** COST's inflation-surprise beta is positive, "
            "significant, beats staples, and holds across eras. Anything less and the halo is just a "
            "great stock we're admiring after the fact."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the combo really has melted.** Deflate the frozen $1.50 into 1985 dollars."
        ),
        code(
            "rp = st.real_price(CPI, nominal=R['combo'], base_cpi=R['cpi_1985'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(rp.index, rp.values, c=RED, lw=2.2, label='real price of the $1.50 combo (1985 $)')\n"
            "ax.axhline(R['combo'], ls='--', c=GREY, alpha=.8, label='nominal $1.50 (frozen)')\n"
            "ax.fill_between(rp.index, rp.values, R['combo'], color=RED, alpha=.10)\n"
            "ax.set_ylabel('real price, 1985 dollars'); ax.set_xlabel('year')\n"
            "ax.set_title('The $1.50 combo has quietly lost two-thirds of its real value'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"real price today ~${R['real_now']:.2f}  (down {R['eroded_pct']:.0f}% since 1985)\")\n"
            "print(f\"to hold 1985 value it would cost ${R['infl_adj']:.2f}  (Costco 'gives away' ~${R['giveaway']:.2f}/combo)\")"
        ),
        md(
            f"Nobody disputes this part. In 1985 purchasing power the combo now costs about "
            f"**\\${R['real_now']:.2f}** — a **{R['eroded_pct']:.0f}%** real cut. But notice what this is: "
            "**pure arithmetic** from a frozen nominal price. It's a wonderful fact about Costco's "
            "*marketing*. It says nothing yet about its *stock*."
        ),
        md(
            "**Second: COST has crushed everything.** Grow $10,000 since 2000 in COST, in SPY, and by CPI."
        ),
        code(
            "if HAVE:\n"
            "    base = PANEL.index[0]\n"
            "    cost = PANEL['cost']/PANEL['cost'].iloc[0]*10000\n"
            "    spy  = PANEL['spy']/PANEL['spy'].iloc[0]*10000\n"
            "    cpi  = PANEL['cpi']/PANEL['cpi'].iloc[0]*10000\n"
            "    idx = PANEL.index\n"
            "else:\n"
            "    idx = pd.date_range('2000-01-31','2026-06-30',freq='ME')\n"
            "    g=lambda c: 10000*(1+c/100)**(np.arange(len(idx))/12)\n"
            "    cost, spy, cpi = g(R['cost_cagr']), g(R['spy_cagr']), g(R['cpi_cagr'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(idx, cost, c=GREEN, lw=2.2, label=f\"COST -> ${R['cost_10k']:,.0f}\")\n"
            "ax.plot(idx, spy,  c=GREY,  lw=2.0, label=f\"SPY  -> ${R['spy_10k']:,.0f}\")\n"
            "ax.plot(idx, cpi,  c=RED,   lw=2.0, ls='--', label=f\"CPI  -> ${R['cpi_10k']:,.0f}\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('value of $10,000 (log)'); ax.set_xlabel('year')\n"
            "ax.set_title('COST outran the market and inflation — by a lot'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"COST {R['cost_mult']:.0f}x  |  SPY {R['spy_mult']:.0f}x  |  CPI {R['cpi_mult_win']:.0f}x  over {R['years']:.0f} years\")"
        ),
        md(
            f"COST turned \\$10k into **\\${R['cost_10k']:,.0f}** — a **{R['cost_real']:.0f}×** gain even "
            "*after* inflation. Case closed? Not so fast. This is one **hindsight-picked** stock we're "
            "admiring *because* it's the famous hot-dog company that won. \"It went up\" is not the same "
            "as \"it protects you from inflation.\" For that, we need the hedge test."
        ),
        md(
            "**Third — the actual hedge test.** When inflation *surprises* to the upside (its yearly rate "
            "jumps), does COST rise? And does it beat a boring **consumer-staples** fund (XLP) at it?"
        ),
        code(
            "if HAVE:\n"
            "    cb = st.inflation_beta(PANEL,'cost'); sb = st.inflation_beta(PANEL,'spy'); xb = st.inflation_beta(PANEL,'xlp')\n"
            "    ts = [cb['t_nw'], sb['t_nw'], xb['t_nw']]\n"
            "else:\n"
            "    ts = [R['cost_beta_t'], R['spy_beta_t'], R['xlp_beta_t']]\n"
            "labs = ['COST\\n(the hero)', 'SPY\\n(market)', 'XLP\\n(boring staples)']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labs, ts, .55, color=[GREEN, GREY, AMBER])\n"
            "ax.axhline(2, ls='--', c=RED, alpha=.7, label='|t| = 2 bar for a real signal')\n"
            "ax.axhline(-2, ls='--', c=RED, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('inflation-surprise beta t-stat (HAC)'); ax.set_ylim(-2.5, 2.5)\n"
            "ax.set_title('Nobody hedges inflation here — and COST least of all'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"COST beta t={ts[0]:+.2f}  (basically zero)  |  staples XLP t={ts[2]:+.2f}\")\n"
            "print(f\"COST-minus-staples gap t={R['gap_xlp_t']:+.2f}  ->  COST is NOT a better inflation hedge than a soup fund\")"
        ),
        md(
            f"There it is. COST's response to inflation surprises is **statistically indistinguishable "
            f"from zero** (*t* = {R['cost_beta_t']:.2f}), and it is **no better than a plain staples "
            f"basket** — the COST-minus-XLP gap is a negative, insignificant **{R['gap_xlp_t']:.2f}**. "
            "The one thing the \"pricing-power inflation hedge\" story actually predicts — a distinctive "
            "positive inflation beta — simply isn't in the tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Real-price erosion — Confirmed (but mechanical).** The $1.50 combo has shed "
            f"**{R['eroded_pct']:.0f}%** of its real value; it'd cost **\\${R['infl_adj']:.2f}** to keep "
            "1985 purchasing power. True — and pure arithmetic.\n"
            "- **Signal — None.** COST's inflation-surprise beta is essentially zero "
            f"(*t* = {R['cost_beta_t']:.2f}) and no better than a boring staples fund "
            f"(gap *t* = {R['gap_xlp_t']:.2f}). No inflation-hedge mechanism.\n"
            "- **Tradability — Mirage.** COST's huge win is a **hindsight-selected** growth story, not a "
            "repeatable inflation trade. Admiring the winner is not a strategy."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "\"Fine,\" you say, \"I'll hold COST only when inflation is heating up.\" Let's build exactly "
            "that — own COST when the inflation surprise is positive, cash otherwise — and race it against "
            "just buying and holding COST, and against SPY."
        ),
        code(
            "if HAVE:\n"
            "    tm = st.timer_stats(PANEL, cost_bps=10.0)\n"
            "    sh = {'inflation-timing\\nCOST (net)': tm['timer_sharpe'], 'buy & hold\\nCOST': tm['cost_bh_sharpe'], 'SPY': tm['spy_sharpe']}\n"
            "else:\n"
            "    sh = {'inflation-timing\\nCOST (net)': R['timer_sh'], 'buy & hold\\nCOST': R['cost_bh_sh'], 'SPY': R['spy_sh']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(list(sh), list(sh.values()), .55, color=[AMBER, GREEN, GREY])\n"
            "for i,v in enumerate(sh.values()): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.set_title('The inflation-timing overlay LOSES to just holding COST — the signal hurts')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k,v in sh.items(): print(f'{k.replace(chr(10),\" \"):26s} Sharpe {v:.2f}')"
        ),
        md(
            f"The twist: the inflation-timing overlay (Sharpe **{R['timer_sh']:.2f}**) actually "
            f"**underperforms** simply buy-and-holding COST (**{R['cost_bh_sh']:.2f}**) — and even trails "
            f"plain SPY (**{R['spy_sh']:.2f}**). Timing on the inflation surprise *destroys* return: the "
            "signal forecasts nothing (we just saw that), so all the overlay does is **sit in cash through "
            "half of COST's compounding**. The only reason buy-hold COST beats SPY is that **COST is a "
            "spectacular hindsight-selected stock** — nothing to do with inflation. The \"hedge\" does "
            "negative work. That's a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Use core CPI or PCE instead.** We already use the real `CPIAUCSL` (BLS `CUSR0000SA0`); "
            "swap in core CPI or PCE and re-run — a beta that's ~zero and inside its placebo won't wake "
            "up.\n"
            "- **Try other 'pricing-power' names.** WMT, MCD, PG: does *any* consumer champion carry a "
            "real, distinctive inflation-surprise beta, or is \"inflation-proof brand\" always a story?\n"
            "- **The survivorship trap.** We picked COST *because* it won. Re-run the race on a basket "
            "chosen in 2000 (not with hindsight) and watch the halo dim — see "
            "[docs/references.md](../docs/references.md).\n\n"
            "*Think a consumer name genuinely hedges inflation shocks? Build it, measure the "
            "inflation-surprise beta, beat a staples control, and show it holds across eras — then charge "
            "it costs.*"
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
            "# The Costco Hot-Dog Index — a quantitative teardown 🔬\n"
            "### A real-price identity · a total-return race (COST vs CPI vs SPY) · inflation-surprise "
            "betas with a staples control · a block-rotation placebo · a two-era cut · a costed "
            "inflation-timing overlay · a synthetic planted-beta positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The folk "
            "claim bundles three separable hypotheses: (H₀) the frozen $1.50 combo's *real* price decays "
            "as `1/CPI` (a mechanical identity); (H₁) COST total return beats CPI and SPY (descriptive); "
            "(H₂) COST carries a **beta to inflation surprises** that is robustly non-zero **and** beats "
            "a consumer-staples basket (XLP) — the tradable \"distinctive inflation hedge\" claim. We "
            "find **H₀ true but mechanical**, **H₁ true but hindsight-selected**, and **H₂ rejected**: "
            f"COST's inflation-surprise beta is *t* = **{R['cost_beta_t']:.2f}**, the COST−staples gap is "
            f"**{R['gap_xlp_t']:.2f}**, the placebo *p* = **{R['placebo_p']:.2f}**, and it is a flat "
            "nothing in both sub-eras.\n\n"
            "> ⚠️ **Not investment advice — data provenance.** CPI is the **real** FRED/BLS `CPIAUCSL` "
            "series (BLS `CUSR0000SA0`), fetched from the BLS public API and cached (one point, 2025-10, "
            "interpolated — the SA suspension); the inflation surprise is measured on ΔYoY. `COST`, `SPY`, "
            "`XLP` are month-end total-return via yfinance (as-of 2026-06-30, fp `cbd3b6799d57`). Offline "
            "core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | COST inflation-surprise beta *t* = **{R['cost_beta_t']:.2f}** "
            f"(≈0); COST−XLP gap *t* = **{R['gap_xlp_t']:.2f}**; placebo *p* = **{R['placebo_p']:.2f}**; "
            f"sub-eras *t* = {R['era_early_t']:.2f} / {R['era_late_t']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Inflation-timing overlay Sharpe **{R['timer_sh']:.2f}** "
            f"**loses** to buy-hold COST **{R['cost_bh_sh']:.2f}** (the signal hurts); COST beats SPY on "
            "one hindsight-selected mega-cap (excess-return *t* = 1.37). |\n"
            f"| **Real-price erosion** | `CONFIRMED` | The $1.50 combo has lost **{R['eroded_pct']:.0f}%** "
            f"of real value (would cost **\\${R['infl_adj']:.2f}** to hold 1985 value) — a *mechanical* "
            "identity, not a stock signal. |\n\n"
            "> 💡 In plain words: the folk icon's *fact* (real erosion) is true and huge; its *investment "
            "punchline* (COST as inflation hedge) has no beta behind it. COST is a great growth stock we "
            "are admiring in hindsight."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\pi_t$ be trailing-12-month CPI inflation and $s_t = \\Delta\\pi_t$ the **inflation "
            "surprise** (is inflation accelerating?). Let $r^{\\text{COST}}_t$, $r^{\\text{SPY}}_t$, "
            "$r^{\\text{XLP}}_t$ be monthly total returns. The trade is a joint hypothesis:\n\n"
            "- **H₀ (the identity).** The frozen combo's real price is "
            "$P^{\\text{real}}_t = 1.50\\cdot \\text{CPI}_{1985}/\\text{CPI}_t$ — decays mechanically.\n"
            "- **H₁ (it won).** COST's total-return multiple $\\gg$ SPY's $\\gg$ CPI's.\n"
            "- **H₂ (it hedges).** In $r^{\\text{COST}}_t = \\alpha + \\beta s_t + \\varepsilon_t$ the "
            "inflation beta $\\beta > 0$ with Newey-West *t* > 2, **and** $\\beta_{\\text{COST}} > "
            "\\beta_{\\text{XLP}}$ (pricing power beats a boring staples basket), holding across eras.\n\n"
            "The steelman for H₂ is Costco's membership-fee annuity and brand discipline. But H₀ is a "
            "calculator, H₁ is a statement about a **hindsight-selected** winner, and only H₂ is a "
            "tradable *mechanism*. Note the finance prior (Fama-Schwert 1977): equities are *poor* "
            "short-run inflation hedges — nominal returns load *negatively* on inflation surprises."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H₀–H₂ are routinely blurred into \"the $1.50 hot dog proves COST beats inflation, so buy "
            "it.\" Separated, each is falsifiable. H₀ needs only CPI and arithmetic. H₁ is a descriptive "
            "race (with a **survivorship** asterisk we keep visible). H₂ is the real test: a *contemporaneous* "
            "inflation-surprise beta, a *predictive* version (does $s_t$ forecast $r^{\\text{COST}}_{t+1}$?), "
            "and — decisively — a **relative** test against XLP, because \"pricing power\" is a claim about "
            "beating an ordinary staples basket, not about beating cash."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **CPI (real).** The real monthly `CPIAUCSL` (SA, 1982-84=100 = BLS `CUSR0000SA0`), fetched "
            "from the BLS public API and cached, 2000-01→2026-06, plus the 1985 anchor (107.6). One point "
            "(2025-10) interpolated; the surprise is ΔYoY.\n"
            "- **COST, SPY, XLP.** Month-end total-return (yfinance, cached), as-of 2026-06-30, fp "
            "`cbd3b6799d57`. **Survivorship named**: COST is a hindsight-selected single name — its win "
            "is history, not a promise.\n"
            "- **No look-ahead, one lag.** The predictive regressor is $s_t$ against $r_{t+1}$; the timer "
            "shifts its signal one month — one `shift`, once.\n"
            "- **Inference.** Newey-West (6-lag) HAC *t* on every slope; a 3k block-rotation placebo for "
            "the inflation beta; a two-era cut at 2013; the COST−XLP and COST−SPY beta gaps.\n"
            "- **Positive control.** A deterministic world where $r^{\\text{COST}}_t = \\text{edge}\\cdot "
            "s_t + $ noise (SPY/XLP flat) — the engine must recover *t* ≥ 2 and light up the gap; the "
            "null (edge=0) must stay flat, averaged over ≥20 seeds.\n"
            "- **What would make us say \"hedge\":** H₂ contemporaneous *t* > 2 **and** COST−XLP gap "
            "*t* > 2 **and** era-robust. We find none."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The identity + the race — true, but not a signal\n\n"
            "The mechanical erosion and the descriptive total-return race. Both real; neither is H₂."
        ),
        code(
            "er = st.erosion_stats(CPI, nominal=R['combo'], base_cpi=R['cpi_1985'])\n"
            "print(f\"real price of $1.50 combo today : ${er['real_now']:.2f}  (eroded {er['eroded_pct']:.1f}%)\")\n"
            "print(f\"price to hold 1985 real value    : ${er['inflation_adjusted']:.2f}  (giveaway ${er['giveaway']:.2f}/combo)\")\n"
            "if HAVE:\n"
            "    race = st.total_return_race(PANEL)\n"
            "    for k in ('cost','spy','cpi'):\n"
            "        print(f\"  {k.upper():4s} mult {race[k]['mult']:5.1f}x  CAGR {race[k]['cagr']*100:+5.1f}%  real {race[k]['real_mult']:5.1f}x\")\n"
            "else:\n"
            "    print(f\"  COST mult {R['cost_mult']}x  CAGR +{R['cost_cagr']}%  real {R['cost_real']}x\")\n"
            "    print(f\"  SPY  mult {R['spy_mult']}x  CAGR +{R['spy_cagr']}%  real {R['spy_real']}x\")\n"
            "    print(f\"  CPI  mult {R['cpi_mult_win']}x  CAGR +{R['cpi_cagr']}%\")"
        ),
        md(
            f"> 💡 In plain words: the combo lost **{R['eroded_pct']:.0f}%** of its real value (H₀ ✓, but "
            f"it's `1.50·CPI₁₉₈₅/CPIₜ` — a calculator), and COST returned a **{R['cost_real']:.0f}×** "
            "*real* multiple, trouncing SPY and CPI (H₁ ✓, but this is the *one name we picked because it "
            "won*). Neither line is an inflation-hedge *mechanism*. That is H₂ — next."
        ),
        md(
            "### 4b · The inflation-surprise beta — the actual hedge test\n\n"
            "Regress each asset's monthly return on the inflation surprise $s_t=\\Delta\\pi_t$ (HAC, "
            "6-lag). H₂ predicts a positive, significant COST beta that **beats XLP**."
        ),
        code(
            "if HAVE:\n"
            "    cb = st.inflation_beta(PANEL,'cost'); sb = st.inflation_beta(PANEL,'spy')\n"
            "    xb = st.inflation_beta(PANEL,'xlp'); pb = st.inflation_beta(PANEL,'cost',lag=1)\n"
            "    gap = st.beta_gap(PANEL)\n"
            "    rows = [('COST ~ s_t (same month)', cb['slope'], cb['t_nw']),\n"
            "            ('SPY  ~ s_t', sb['slope'], sb['t_nw']),\n"
            "            ('XLP  ~ s_t (staples)', xb['slope'], xb['t_nw']),\n"
            "            ('COST ~ s_t (next month)', pb['slope'], pb['t_nw']),\n"
            "            ('COST - XLP gap', gap['cost_minus_xlp']['slope'], gap['cost_minus_xlp']['t_nw']),\n"
            "            ('COST - SPY gap', gap['cost_minus_spy']['slope'], gap['cost_minus_spy']['t_nw'])]\n"
            "else:\n"
            "    rows = [('COST ~ s_t (same month)', R['cost_beta'], R['cost_beta_t']),\n"
            "            ('SPY  ~ s_t', R['spy_beta'], R['spy_beta_t']),\n"
            "            ('XLP  ~ s_t (staples)', R['xlp_beta'], R['xlp_beta_t']),\n"
            "            ('COST ~ s_t (next month)', None, R['cost_pred_t']),\n"
            "            ('COST - XLP gap', R['gap_xlp'], R['gap_xlp_t']),\n"
            "            ('COST - SPY gap', None, R['gap_spy_t'])]\n"
            "print(f\"{'regression':32s} {'slope':>8s} {'t_nw':>7s}\")\n"
            "for nm,s,t in rows:\n"
            "    print(f\"{nm:32s} {('' if s is None else f'{s:+.3f}'):>8s} {t:+7.2f}\")\n"
            "labs=['COST','SPY','XLP\\n(staples)','COST-XLP\\ngap']\n"
            "tv=[rows[0][2], rows[1][2], rows[2][2], rows[4][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(labs, tv, .55, color=[GREEN, GREY, AMBER, RED])\n"
            "ax.axhline(2, ls='--', c=RED, alpha=.7, label='|t|=2'); ax.axhline(-2, ls='--', c=RED, alpha=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('inflation-surprise beta t (HAC)'); ax.set_ylim(-2.6,2.6)\n"
            "for i,t in enumerate(tv): ax.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_title('COST has no distinctive inflation beta — and does not beat staples'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: COST's contemporaneous inflation-surprise beta is *t* = "
            f"**{R['cost_beta_t']:.2f}** — dead zero — *smaller* than the market's ({R['spy_beta_t']:.2f}) "
            f"and a plain staples basket's ({R['xlp_beta_t']:.2f}), and the decisive **COST−XLP gap** is a "
            f"negative, insignificant **{R['gap_xlp_t']:.2f}**. The predictive (next-month) beta is "
            f"nothing too (*t* = {R['cost_pred_t']:.2f}). **No cell clears |t| = 2 in any direction** — the "
            f"strongest are the *predictive* COST−staples / COST−SPY gaps at only *t* = "
            f"{R['gap_xlp_pred_t']:.2f} / {R['gap_spy_pred_t']:.2f}, and the placebo buries even those. "
            "**H₂ rejected.**"
        ),
        md(
            "### 4c · Placebo + sub-era cut — is even the whisper real?\n\n"
            "Block-rotate $s_t$ against COST's return (6-month blocks, 3k draws); split the sample at 2013."
        ),
        code(
            "if HAVE:\n"
            "    pl = st.placebo_pvalue(PANEL,'cost',n_perm=3000)\n"
            "    ec = st.era_cut(PANEL, split='2013-01-01')\n"
            "    obs, p = pl['obs_slope'], pl['p_value']; lo, hi = np.quantile, None\n"
            "    sd = pl['placebo_sd']; e_t, l_t = ec.loc['early','t_nw'], ec.loc['late','t_nw']\n"
            "else:\n"
            "    obs, p, sd = R['cost_beta'], R['placebo_p'], 0.71; e_t, l_t = R['era_early_t'], R['era_late_t']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))\n"
            "band = 1.96*sd\n"
            "axes[0].axvspan(-band, band, color=GREY, alpha=.2, label='placebo 95% band')\n"
            "axes[0].axvline(obs, c=RED, lw=2.4, label=f'observed slope {obs:+.2f}'); axes[0].axvline(0,c='k',lw=1)\n"
            "axes[0].set_yticks([]); axes[0].set_xlabel('inflation-beta slope under the null')\n"
            "axes[0].set_title(f'Inside its own noise (p={p:.2f})'); axes[0].legend()\n"
            "axes[1].bar(['early\\n2000-12','late\\n2013-26'], [e_t, l_t], .5, color=[GREY, GREY])\n"
            "axes[1].axhline(2, ls='--', c=RED, alpha=.7); axes[1].axhline(0,c='k',lw=1); axes[1].set_ylim(-2.6,2.6)\n"
            "for i,t in enumerate([e_t,l_t]): axes[1].annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "axes[1].set_ylabel('COST inflation-beta t (HAC)'); axes[1].set_title('Flat in both eras')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"placebo p={p:.3f}  |  early t={e_t:+.2f}  late t={l_t:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: COST's inflation beta sits **squarely inside** its placebo band "
            f"(*p* = **{R['placebo_p']:.2f}**) and is a flat nothing in **both** halves "
            f"(*t* = {R['era_early_t']:.2f} then {R['era_late_t']:.2f}). There is no era, and no honest "
            "null, in which the hot-dog pricing-power hedge shows up. `Signal → NONE`."
        ),
        md(
            "### 4d · The money test — an inflation-timing overlay, net of costs\n\n"
            "Own COST when the inflation surprise known at $t-1$ is positive, cash at 2%/yr otherwise; "
            "10 bp one-way per switch. Raced on **excess-of-cash** Sharpe vs buy-hold COST and SPY."
        ),
        code(
            "if HAVE:\n"
            "    tm = st.timer_stats(PANEL, cost_bps=10.0)\n"
            "    sh=[tm['timer_sharpe'], tm['cost_bh_sharpe'], tm['spy_sharpe']]; inv=tm['invested_frac']; sw=tm['switches_per_yr']\n"
            "else:\n"
            "    sh=[R['timer_sh'], R['cost_bh_sh'], R['spy_sh']]; inv=R['timer_invested']; sw=R['timer_switches']\n"
            "names=['inflation-timing\\nCOST (net 10bp)','buy & hold\\nCOST','SPY']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(names, sh, .55, color=[AMBER, GREEN, GREY])\n"
            "ax.axhline(sh[1], ls='--', c=GREEN, alpha=.5)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.set_title('The overlay LOSES to holding COST (and to SPY) — the signal does negative work')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"overlay {sh[0]:.2f}  |  buy-hold COST {sh[1]:.2f}  |  SPY {sh[2]:.2f}   ({sw:.1f} switches/yr, invested {inv*100:.0f}%)\")"
        ),
        md(
            f"> 💡 In plain words: the overlay (Sharpe **{R['timer_sh']:.2f}**) **underperforms** buy-and-hold "
            f"COST (**{R['cost_bh_sh']:.2f}**) and even trails SPY (**{R['spy_sh']:.2f}**): timing on the "
            "surprise is **negative-value cash-parking** (§4b–c showed the surprise predicts nothing). "
            "Buy-hold COST beats SPY only via **COST's own hindsight-selected quality**, not an inflation "
            "edge — and even that excess return is *t* = 1.37, not robust. Concentrated single-name risk "
            "for a signal that does negative work. `MIRAGE`."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted inflation beta\n\n"
            "A deterministic world where $r^{\\text{COST}}_t = 8\\cdot s_t + \\text{noise}$ (SPY/XLP flat). "
            "The harness must recover *t* ≥ 2 and light up the COST−XLP gap; the edge=0 null must stay "
            "flat — averaged over 20 seeds. Proving the real-tape `NONE` is a true null."
        ),
        code(
            "sm8 = st.synthetic_mean_t(data, edge=8.0, n_seeds=20)\n"
            "sm0 = st.synthetic_mean_t(data, edge=0.0, n_seeds=20)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar(['planted\\n(edge=8)','null\\n(edge=0)'], [sm8['mean_t'], sm0['mean_t']], .5, color=[GREEN, GREY])\n"
            "ax.axhline(2, ls='--', c=RED, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for i,t in enumerate([sm8['mean_t'], sm0['mean_t']]): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean COST inflation-beta t (HAC)'); ax.set_title('Machinery proof: recovers a planted beta, stays flat on the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted: mean slope {sm8['mean_slope']:+.2f}  mean t {sm8['mean_t']:+.2f}  gap t {sm8['mean_gap_t']:+.2f}  fires {sm8['fire_frac']*100:.0f}%\")\n"
            "print(f\"null   : mean slope {sm0['mean_slope']:+.2f}  mean t {sm0['mean_t']:+.2f}  fires {sm0['fire_frac']*100:.0f}%\")"
        ),
        md(
            f"> 💡 In plain words: when a real inflation beta is present, the engine nails it — mean "
            f"*t* ≈ **{R['syn_t']:.1f}**, gap *t* ≈ **{R['syn_gap_t']:.1f}**, fires 100% — and on the null "
            f"it reads *t* ≈ **{R['syn_null_t']:.2f}** (flat, fires ~5%). So the real-tape `NONE` is a "
            "*true* null: COST genuinely has no distinctive inflation beta, not a harness that couldn't "
            "see one. A synthetic control is a machinery proof, never market evidence."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — COST's contemporaneous inflation-surprise beta *t* = "
            f"{R['cost_beta_t']:.2f} (≈0), **not** above a staples basket (COST−XLP gap "
            f"*t* = {R['gap_xlp_t']:.2f}), placebo *p* = {R['placebo_p']:.2f}, flat in both sub-eras "
            f"({R['era_early_t']:.2f} / {R['era_late_t']:.2f}). No robust inflation-hedge beta.\n"
            f"- **Tradability `MIRAGE`** — inflation-timing overlay Sharpe {R['timer_sh']:.2f} **loses** to "
            f"buy-hold COST {R['cost_bh_sh']:.2f} (and to SPY {R['spy_sh']:.2f}); the signal hurts. COST "
            f"beating SPY ({R['cost_bh_sh']:.2f} vs {R['spy_sh']:.2f}) is one survivor (excess-return "
            f"*t* = {R['costspy_alpha_t']:.2f}), not a repeatable edge.\n"
            f"- **Real-price erosion `CONFIRMED`** — the $1.50 combo shed {R['eroded_pct']:.0f}% of real "
            f"value (would cost \\${R['infl_adj']:.2f} to hold 1985 value). True, huge — and a *mechanical "
            "identity* of a frozen nominal price, carrying zero information about the stock."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — capacity & the survivorship trap\n\n"
            "Even granting COST's spectacular past, \"buy COST as an inflation hedge\" fails on both "
            "mechanism and construction: the inflation beta is zero (§4b–c), and the whole case rests on "
            "**one stock we selected because it already won**. The honest question is what an *ex-ante* "
            "pick would have done — but the folklore only ever quotes the survivor."
        ),
        code(
            "start=10_000.0\n"
            "ends={'COST (the survivor)':R['cost_10k'], 'SPY (market)':R['spy_10k'], 'CPI (inflation)':R['cpi_10k']}\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(list(ends), list(ends.values()), .55, color=[GREEN, GREY, RED])\n"
            "for i,v in enumerate(ends.values()): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'value of $10,000 over {R[\"years\"]:.0f} years'); ax.set_yscale('log')\n"
            "ax.set_title('A hindsight-selected winner beats everything — that is what survivorship looks like')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k,v in ends.items(): print(f'{k:22s} ${v:>10,.0f}')"
        ),
        md(
            "> 💡 In plain words: COST's terminal wealth dwarfs SPY and CPI — but that gap is the "
            "**signature of survivorship**, not of an inflation hedge. We conditioned on the winner. The "
            "inflation-surprise beta (the part that would generalise to a *rule* you could run ex-ante) is "
            "zero. There is no size, venue, or overlay that turns a frozen loss-leader's fame into a "
            "tradable inflation edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Try core CPI / PCE.** We use the real `CPIAUCSL` (BLS `CUSR0000SA0`); swap in core CPI or "
            "PCE and re-run — a beta that's ~zero and inside its placebo won't wake up.\n"
            "- **A cross-section of 'pricing-power' names.** WMT, MCD, PG, KO: estimate each inflation-"
            "surprise beta vs XLP — is *any* consumer champion a real hedge, or is the halo always a "
            "story?\n"
            "- **Kill the survivorship.** Form a 2000-vintage basket without hindsight and re-race — the "
            "COST-vs-SPY gap is a selection artifact ([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; CPI is the **real** FRED/BLS `CPIAUCSL` "
            "series (fetched from BLS and cached) and COST/SPY/XLP are real month-end total-return tape. "
            "Methods: [`docs/references.md`](../docs/references.md); frozen numbers (fp `cbd3b6799d57`): "
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
