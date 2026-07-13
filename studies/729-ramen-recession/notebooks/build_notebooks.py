"""Generate the two narrative notebooks for Study 729 ("the ramen index" — a downturn tell?).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ (2897.T, 2875.T, ^N225) plus the hardcoded, cited WINA
demand series and NBER recession windows from the package; on a cache miss they fall back to
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic lead-lag and
defensive controls run anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (2897.T/2875.T/^N225 month-end Adj
# Close via yfinance, as-of 2026-06-30; WINA demand + NBER windows hardcoded/cited).
R = dict(
    win="2001 → 2026",
    n_stock=305, n_n225=317,
    # THE HEADLINE — lead-lag of WINA demand growth vs Nikkei annual return (n=19)
    ll_leads=[-2, -1, 0, 1, 2],
    ll_r=[-0.34, 0.31, 0.12, -0.25, -0.03],
    ll_t=[-1.51, 1.32, 0.51, -1.05, -0.12],
    ll_best_lead=1, ll_best_r=-0.25, ll_best_t=-1.05, ll_best_p=0.31, ll_n=19,
    # demand in-vs-out recession years
    dem_in=2.02, dem_out=1.93, dem_diff=0.09, dem_t=0.02, dem_p=0.98, dem_nin=4, dem_nout=15,
    # ramen index shape (WINA, bn servings)
    ri_2005=85.7, ri_2024=122.0, ri_cagr=1.88,
    dem_2008=-4.2, dem_2009=-3.8, dem_2020=9.6, dem_2014=-2.7, dem_2015=-4.9,
    # full-sample risk/return
    nis_cagr=6.14, nis_vol=20.2, nis_sharpe=0.40, nis_mdd=-42.7, nis_wealth=4.54,
    toy_cagr=11.53, toy_vol=25.8, toy_sharpe=0.55, toy_mdd=-36.6, toy_wealth=16.02,
    n225_cagr=4.95, n225_vol=19.2, n225_sharpe=0.35, n225_mdd=-62.8, n225_wealth=3.59,
    hold_yrs=25.4,
    # CAPM alpha vs N225 (Newey-West, 6-lag) — the honest complication
    nis_alpha=6.54, nis_beta=0.20, nis_t=2.03, nis_p=0.044,
    toy_alpha=12.20, toy_beta=0.31, toy_t=2.79, toy_p=0.006,
    # bull/bear beta (split at N225=0)
    nis_down_beta=0.47, nis_up_beta=-0.05, nis_asym=0.52, nis_def=0,
    toy_down_beta=0.14, toy_up_beta=0.27, toy_asym=-0.13, toy_def=1,
    n_down=130, n_up=175,
    # recession-window excess vs N225 (paired t)
    rec_n=31,
    nis_rec_mean=-0.60, nis_rec_excess=1.55, nis_rec_t=0.94, nis_rec_p=0.352, nis_rec_cum=-25.0,
    toy_rec_mean=1.37, toy_rec_excess=3.53, toy_rec_t=1.34, toy_rec_p=0.189, toy_rec_cum=17.0,
    n225_rec_mean=-2.15, n225_rec_cum=-54.1,
    # per-recession compounded % (stock vs N225)
    rec_labels=["2001 dot-com", "2008 GFC", "2020 COVID"],
    nis_by_rec=[-7.5, -24.6, 7.6],
    toy_by_rec=[6.9, -3.1, 13.0],
    n225_by_rec=[-17.0, -36.5, -13.0],
    # look-ahead
    wina_lag=6, nber_lag=12,
    # synthetic controls
    syn_lead=1, syn_lead_r=-0.93, syn_lead_t=-15.1,
    syn_down=0.71, syn_up=0.98, syn_plant_down=0.50, syn_plant_up=1.00,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![A leading tell%3F: Busted](https://img.shields.io/badge/A_leading_tell%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"
NOOD = "#d97b29"   # a ramen-broth orange for the noodle series

from ramen_recession import data, strategy as st

HAVE_PRICES = data.have_prices()
PX = data.load_prices() if HAVE_PRICES else None
RI = data.load_ramen_index()        # WINA demand (hardcoded, cited) — always available
print("price cache present:", HAVE_PRICES,
      "| ramen index", RI.index[0].year, "->", RI.index[-1].year,
      "| NBER recessions:", [r[0] for r in data.NBER_RECESSIONS])
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The ramen index 🍜\n"
            "### \"When times get hard, people trade down to instant noodles\" — does the ramen "
            "index *warn* you a downturn is coming?\n\n"
            + BADGES +
            "It's a beloved bit of recession folklore: when money is tight, people swap the "
            "restaurant meal for a 30-cent brick of instant noodles — so a **spike in noodle "
            "sales is an early warning** that the economy is rolling over, and the noodle makers "
            "themselves are a safe place to hide while stocks fall.\n\n"
            "This notebook puts that on the clock. We take the closest thing to a real \"ramen "
            "index\" — the **world instant-noodle demand** series (billions of servings a year, "
            "from the industry's own association) — and the two noodle stocks you can actually "
            "buy: **Nissin (`2897.T`, the Cup Noodle inventor)** and **Toyo Suisan (`2875.T`, the "
            "Maruchan maker)**. Then we ask two blunt questions: *does noodle demand rise **before** "
            "a downturn?* and *do the noodle stocks beat the market **when** the downturn hits?*\n\n"
            "> 📓 **Plain-language layer.** Want the lead-lag cross-correlations, the Newey-West "
            "alpha and the recession-window *t*-stats? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Stock prices are real (month-end, "
            "yfinance). The \"ramen index\" is the **WINA** world-demand series — a small, "
            "**cited, approximate** hardcoded series (a *labelled proxy*, not a live feed); the "
            "recession dates are the **NBER**'s official ones. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does noodle demand **rise before** a downturn? | **No.** Line up demand growth "
            "against the stock market a year later and the correlation is a coin-flip "
            f"(*t* = {R['ll_best_t']:.2f}). Demand grows about the **same** in recession years "
            f"(+{R['dem_in']:.1f}%) as in good years (+{R['dem_out']:.1f}%). |\n"
            "| Did demand even *spike* in the last recessions? | **Only in 2020.** Global noodle "
            f"demand actually **fell** in the 2008 crash ({R['dem_2008']:.1f}% in 2008, "
            f"{R['dem_2009']:.1f}% in 2009) and fell again in 2014–2016 with *no* recession. The "
            f"only jump was 2020 (+{R['dem_2020']:.1f}%) — a COVID pantry-stocking story. |\n"
            "| Are the noodle *stocks* a safe hide-out? | **Mixed and mild.** They fell in the "
            "2001 and 2008 crashes too — just less than a collapsing Nikkei — and only clearly "
            "\"won\" in 2020. |\n"
            "| So is 'the ramen index' a real recession signal? | **No — it's a story.** Instant "
            "noodles are a booming *Asian growth* market, not a business-cycle instrument. |\n\n"
            "> The folklore mixes up two things: a genuine long-run *growth* trend in noodles "
            "(real, and nothing to do with recessions) and a *recession warning* (not there)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Instant noodles are the ultimate inferior good: cheap, filling, shelf-stable. When "
            "a recession is coming, belts tighten and people trade **down** — so noodle sales climb "
            "*ahead of* the downturn. Watch the ramen index and you get an early warning the "
            "official statistics won't give you for months. And the noodle makers? A defensive "
            "staple that holds up while everything else is crashing.\"*\n\n"
            "It's a genuinely *steelman-able* idea. Instant noodles really are counter-cyclical in "
            "microeconomic theory — a classic *inferior good*, demand rising as incomes fall. There "
            "are real anecdotes (US instant-ramen sales rose during 2008). The finance version is "
            "specific and testable: (1) noodle **demand growth should lead** a market downturn, and "
            "(2) the noodle **stocks should out-return the market in recessions**. If both hold, "
            "you have a free macro alarm *and* a place to hide."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A working \"ramen index\" would be a small marvel: a cheap, public, real-time gauge "
            "that front-runs the National Bureau of Economic Research — you'd rotate to safety "
            "*before* the crash, not after. And a genuinely counter-cyclical staple stock is a "
            "rare thing worth real money — a diversifier you can hold instead of bonds. But watch "
            "the sleight of hand. \"People eat more noodles in hard times\" is a claim about "
            "**volumes**. \"Noodle *sales lead* the cycle\" is a claim about **timing**. And "
            "\"noodle *stocks* beat the market in a recession\" is a claim about **prices** — which "
            "already have the whole outlook baked in. Three different claims, and we can check each "
            "one directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest tests:\n\n"
            "1. **Does demand *lead*?** Line up each year's noodle-demand growth against the stock "
            "market's return *the following year*. A real warning signal means high demand now → a "
            "*bad* market later (a clear negative correlation at a positive lead).\n"
            "2. **Do the stocks beat the market in recessions?** Take the actual NBER recession "
            "months (2001, 2008, 2020) and compare each noodle stock's return to the Nikkei's, "
            "head-to-head.\n"
            "3. **Could you even trade it?** The demand figure is published ~6 months *after* the "
            "year ends, and the NBER dates a recession ~a year late — so \"act on the ramen "
            "index\" is a rule you can't actually follow in time.\n\n"
            "**What would make us say \"real tell\"?** A demand series that clearly turns up "
            "*before* downturns *and* noodle stocks that beat the market when recessions hit — not "
            "one lucky COVID data point. Anything less is a barstool story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the ramen index itself.** Here is world instant-noodle demand since 2005, "
            "with the recession years shaded. If the folklore were right, the line would surge "
            "*into* the grey bands."
        ),
        code(
            "g = data.ramen_growth()\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)\n"
            "a1.plot(RI.index.year, RI.values, '-o', color=NOOD, lw=2, ms=4)\n"
            "a1.set_ylabel('bn servings / yr'); a1.set_title('The \"ramen index\": world instant-noodle demand (WINA, approx.)')\n"
            "a2.bar(g.index.year, g.values, color=[RED if v<0 else NOOD for v in g.values])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('YoY growth (%)'); a2.set_xlabel('year')\n"
            "for yr in sorted({d.year for d in data.recession_months()}):\n"
            "    for ax in (a1, a2): ax.axvspan(yr-.5, yr+.5, color=GREY, alpha=.25, zorder=0)\n"
            "a2.set_title('Growth is secular — it FELL in the 2008 crash, only spiked in COVID-2020')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"demand 2008 {g[g.index.year==2008].iloc[0]:+.1f}%  2009 {g[g.index.year==2009].iloc[0]:+.1f}%  \"\n"
            "      f\"2020 {g[g.index.year==2020].iloc[0]:+.1f}%  (grey = recession years)\")"
        ),
        md(
            f"Look at the grey bands. In the **2008 GFC** — the deepest, most \"trade-down\" "
            f"recession of the sample — global noodle demand *fell* ({R['dem_2008']:.1f}% then "
            f"{R['dem_2009']:.1f}%), as a spike in wheat and palm-oil prices pushed volumes down. It "
            f"also fell in **2014–2016** with no recession at all. The one true jump is **2020** "
            f"(+{R['dem_2020']:.1f}%) — people stockpiling instant noodles in lockdown, a supply-"
            "shock story, not an economic-warning one. This does not look like a cycle indicator."
        ),
        md(
            "**Now the direct test of *leading*.** Does a big noodle-demand year foretell a bad "
            "market year? We slide the two series past each other and measure the correlation at "
            "each \"lead.\" A genuine early-warning signal shows a **negative** bar at a "
            "**positive** lead (demand up now → market down later)."
        ),
        code(
            "mkt = PX['^N225'].resample('YE').last().pct_change().dropna() if HAVE_PRICES else None\n"
            "if HAVE_PRICES:\n"
            "    ll = st.lead_lag_corr(data.ramen_growth()/100.0, mkt, leads=range(-2,3))\n"
            "    leads = list(ll['per_lead']); rs = [ll['per_lead'][k]['r'] for k in leads]; ts=[ll['per_lead'][k]['t'] for k in leads]\n"
            "else:\n"
            "    leads, rs, ts = R['ll_leads'], R['ll_r'], R['ll_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [GREY]*len(leads)\n"
            "for i,k in enumerate(leads):\n"
            "    if k==R['ll_best_lead']: cols[i]=RED\n"
            "ax.bar([str(k) for k in leads], rs, .6, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('lead (years the ramen index PRECEDES the market)  — positive = the \"tell\"')\n"
            "ax.set_ylabel('correlation with market return')\n"
            "for i,k in enumerate(leads): ax.annotate(f't={ts[i]:+.2f}',(i,rs[i]),ha='center',va='bottom' if rs[i]>=0 else 'top',fontsize=8)\n"
            "ax.set_title('A real tell needs a strong NEGATIVE bar at a POSITIVE lead — there is none')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"best 'lead' (+{R['ll_best_lead']}y): r={R['ll_best_r']:+.2f}  t={R['ll_best_t']:+.2f}  -> a coin flip\")"
        ),
        md(
            f"There is no signal. The \"tell\" lead (+1 year) is a weak, *insignificant* "
            f"correlation (*t* = {R['ll_best_t']:.2f}), and none of the other leads clear the bar "
            "either. The single strongest bar is actually at lead **−2** — i.e. if anything the "
            "*market* moves before the noodles, the exact opposite of a leading indicator. And the "
            "blunt version says the same thing: demand grew "
            f"**+{R['dem_in']:.1f}%/yr** in recession years vs **+{R['dem_out']:.1f}%/yr** "
            f"otherwise — a difference of {R['dem_diff']:+.1f}% (*t* = {R['dem_t']:.2f}). "
            "Statistically indistinguishable."
        ),
        md(
            "**So the *tell* fails. What about the *stocks* as a recession hide-out?** Here is how "
            "each noodle maker did, compounded, inside each of the three NBER recessions — "
            "head-to-head with the Nikkei."
        ),
        code(
            "labels = R['rec_labels']\n"
            "if HAVE_PRICES:\n"
            "    br = PX['^N225'].pct_change().dropna()\n"
            "    nb = st.recession_breakdown(PX['2897.T'].pct_change().dropna(), br, data.NBER_RECESSIONS)\n"
            "    tb = st.recession_breakdown(PX['2875.T'].pct_change().dropna(), br, data.NBER_RECESSIONS)\n"
            "    nisv=[nb[l]['stock']*100 for l in labels]; toyv=[tb[l]['stock']*100 for l in labels]; benv=[nb[l]['bench']*100 for l in labels]\n"
            "else:\n"
            "    nisv, toyv, benv = R['nis_by_rec'], R['toy_by_rec'], R['n225_by_rec']\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.bar(x-.25, nisv, .25, color=NOOD, label='Nissin (2897.T)')\n"
            "ax.bar(x,      toyv, .25, color='#8e44ad', label='Toyo Suisan (2875.T)')\n"
            "ax.bar(x+.25, benv, .25, color=GREEN, label='Nikkei 225')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('compounded return in the recession (%)')\n"
            "ax.set_title('Noodle stocks fell in 2001 & 2008 too — just less than a collapsing Nikkei'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for i,l in enumerate(labels): print(f'{l:14s}  Nissin {nisv[i]:+6.1f}%  Toyo {toyv[i]:+6.1f}%  Nikkei {benv[i]:+6.1f}%')"
        ),
        md(
            "It's a mild cushion, not counter-cyclicality. Nissin *fell* in 2001 (−7.5%) and fell "
            "hard in the 2008 GFC (−24.6%) — less than the Nikkei's −36.5%, but a loss is a loss. "
            "Both names only clearly *won* in **2020** — again the COVID window. \"Falls less than a "
            "terrible index\" is *low beta*, a boring and freely-available property, not a "
            "recession alarm."
        ),
        md(
            "**And the twist the story never mentions:** over 25 years these noodle stocks actually "
            "*beat* the Nikkei outright. Here is \\$1 left in each since 2001."
        ),
        code(
            "if HAVE_PRICES:\n"
            "    ends = [st.terminal_wealth(PX[t]) for t in ['^N225','2897.T','2875.T']]\n"
            "else:\n"
            "    ends = [R['n225_wealth'], R['nis_wealth'], R['toy_wealth']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['Nikkei 225','Nissin','Toyo Suisan'], ends, .55, color=[GREEN, NOOD, '#8e44ad'])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:.1f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('$1 invested in 2001 -> ...'); ax.set_title('The noodle makers beat the Nikkei — but that is survivorship, not a tell')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"$1 -> Nikkei ${ends[0]:.2f}  |  Nissin ${ends[1]:.2f}  |  Toyo Suisan ${ends[2]:.2f}\")"
        ),
        md(
            f"Yes — \\$1 became **\\${R['toy_wealth']:.0f}** in Toyo Suisan vs just "
            f"**\\${R['n225_wealth']:.1f}** in the Nikkei. But notice the trick: we hand-picked the "
            "two noodle makers that *survived and won*. A defensive staple beating Japan's "
            "lost-decades index is the well-known *low-volatility* effect plus **survivorship** — "
            "it is not the ramen index telling anyone a recession is coming. (The quant notebook "
            "shows this beat clears a significance bar — and why it still isn't the claim.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The ramen index does not lead: the best \"tell\" correlation is "
            f"*t* = {R['ll_best_t']:.2f}, and demand grows the same in and out of recessions "
            f"(*t* = {R['dem_t']:.2f}). The noodle stocks' recession edge is insignificant too "
            f"(*t* = {R['nis_rec_t']:.2f} / {R['toy_rec_t']:.2f}).\n"
            "- **Tradability — Mirage.** You'd learn the year's noodle demand ~6 months late and "
            "the recession ~a year late — you can't act on the tell in real time. The only thing "
            "that beat the market is two hand-picked survivors, not a signal.\n"
            "- **A leading tell? — Busted.** Global demand *fell* in the 2008 crash and in "
            "2014–2016; the one spike was COVID-2020. Instant noodles are an Asian growth story, "
            "not a business-cycle instrument."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Suppose you *believed* it and wanted to use the ramen index as your recession alarm. "
            "When would you actually get the reading?"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.4, 3.2))\n"
            "ax.barh(['You want to act:\\n\"a recession is coming\"','WINA publishes the\\nyear\\'s noodle demand','NBER confirms the\\nrecession'],\n"
            "        [0, R['wina_lag'], R['nber_lag']], color=[GREEN, AMBER, RED])\n"
            "for i,v in enumerate([0, R['wina_lag'], R['nber_lag']]): ax.annotate(f'+{v} mo' if v else 'month 0',(v,i),ha='left',va='center')\n"
            "ax.set_xlabel('months after the year you cared about'); ax.set_xlim(-1, 16)\n"
            "ax.set_title('The ramen alarm rings 6-12 months AFTER you needed it (a double look-ahead)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"WINA demand lag ~{R['wina_lag']} mo + NBER lag ~{R['nber_lag']} mo -> the tell is un-actable in real time\")"
        ),
        md(
            "There it is. Even if the ramen index *had* led the market (it doesn't), the number "
            "reaches you half a year after the year it describes, and the recession it was supposed "
            "to warn about is confirmed only a year in. \"Buy defense when the ramen index spikes\" "
            "is a rule you can only follow with a time machine. And the one thing that genuinely "
            "beat the market here — owning the noodle makers — you can only claim by picking the "
            "winners in hindsight. The ramen index is a great bar-story and a poor trading signal."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The US-only tape.** The original anecdote is *American* instant-ramen sales rising "
            "in 2008. Pull a US-specific series (Nielsen/IRI) instead of the global WINA figure — "
            "does the tell survive on home turf, or is it also just one recession's noise?\n"
            "- **A real-time nowcast.** Replace the ex-post NBER flag with a Sahm-rule / "
            "yield-curve recession probability and re-test whether *any* consumer-staple demand "
            "leads it with an honest, no-look-ahead entry.\n"
            "- **The staples cousins.** The same test on the \"lipstick index,\" discount retail "
            "(`DG`, `WMT`), or spam (`HRL`) — is *anything* a real downturn tell, or are they all "
            "secular growth stories with one lucky recession? (see "
            "[docs/references.md](../docs/references.md)).\n\n"
            "*Think a specific noodle series led a recession you can name? Pull its tape, mark the "
            "NBER window, and show the lead — then check it wasn't one COVID-shaped data point "
            "doing all the work.*"
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
            "# The ramen index — a quantitative teardown 🔬\n"
            "### Lead-lag cross-correlation (does demand *precede* the market?) · bull/bear beta · "
            "Newey-West CAPM alpha vs the Nikkei · the recession-window paired *t* · the double "
            "look-ahead that kills the trade · planted-lead & planted-defensive positive controls\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "test the strongest form of \"the ramen index is a downturn tell\" on the WINA "
            "world-demand series and on `2897.T` (Nissin) / `2875.T` (Toyo Suisan) vs `^N225`: "
            "(H₁) **demand leads** — a negative lead-lag correlation at a positive lead with "
            "*t* > 2; (H₂) **recession outperformance** — positive recession-window excess with "
            "*t* > 2; (H₃) it is **investable** net of the WINA + NBER publication lags. We find "
            "**H₁ rejected** (best lead *t* = −1.05; demand grows the same in/out of recessions), "
            "**H₂ rejected** (no leg clears *t* = 2), **H₃ rejected** (a double look-ahead). A "
            "significant CAPM α vs the Nikkei **does** appear — and we show why it is a "
            "survivorship + weak-benchmark artefact, not the tell.\n\n"
            "> ⚠️ **Not investment advice — data provenance.** `2897.T`, `2875.T`, `^N225` are "
            "month-end Adj Close via yfinance (as-of 2026-06-30; split+dividend adjusted). The "
            "**ramen index** is the WINA world-demand series and the NBER windows are "
            "**hardcoded, cited** (labelled facts, not a feed). Offline core + synthetic controls "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md); numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | The tell fails: best lead-lag *t* = **{R['ll_best_t']:+.2f}** "
            f"(lead +{R['ll_best_lead']}y), demand in-vs-out recession *t* = **{R['dem_t']:+.2f}**, "
            f"recession-window excess *t* = **{R['nis_rec_t']:+.2f}** / **{R['toy_rec_t']:+.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Double look-ahead: WINA ~{R['wina_lag']} mo + NBER "
            f"~{R['nber_lag']} mo late. The only market-beating item (α vs Nikkei) is survivorship. |\n"
            f"| **A leading tell?** | `BUSTED` | Demand **fell** in the 2008 GFC "
            f"({R['dem_2008']:+.1f}%, {R['dem_2009']:+.1f}%) and in 2014–2016; the only spike was "
            f"COVID-2020 (+{R['dem_2020']:.1f}%). Secular Asian growth, not a cycle instrument. |\n\n"
            "> 💡 In plain words: the ramen index carries no cycle information — it doesn't lead "
            "downturns and doesn't even reliably rise in them. The noodle *stocks* did beat a "
            "moribund Nikkei, but that is the low-beta staple premium on two hand-picked survivors, "
            "not the recession tell the folklore sells."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d_t$ be the ramen index's YoY demand growth in year $t$ and $r^M_t$ the market's "
            "annual return. The folklore is a joint hypothesis:\n\n"
            "- **H₁ (demand leads).** For some lead $k \\ge 1$, $\\;\\rho(d_t, r^M_{t+k}) < 0$ with "
            "$|t| > 2$: high noodle-demand growth today precedes a *weak* market $k$ years out. "
            "(Equivalently, demand growth is higher in recession years.)\n"
            "- **H₂ (recession outperformance).** Over NBER recession months $\\mathcal{R}$, each "
            "noodle stock's mean excess $\\;\\bar{x}=\\text{mean}_{t\\in\\mathcal{R}}(r^P_t-r^M_t)$ "
            "is positive with $t>2$.\n"
            "- **H₃ (investable).** After the WINA release lag (~6 mo) and the NBER announcement "
            "lag (~12 mo), a ramen-conditional rule beats simply holding the index.\n\n"
            "The steelman is textbook: instant noodles are a canonical **inferior good**, so demand "
            "*should* be income-elastic in the wrong direction — up when incomes fall. The test is "
            "whether that shows up in the **aggregate demand series**, **ahead of** the cycle, in a "
            "way you could **act on** — or only in a 2008 US anecdote and one COVID spike."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, the ramen index would be a genuine leading indicator (H₁) *and* the "
            "noodle makers a recession hedge (H₂) you could actually implement (H₃). Each leg is "
            "separately falsifiable. H₁ is a **lead-lag** statement on an annual macro series — "
            "low-frequency, tiny-sample, so we announce the bar (a *negative* correlation at a "
            "*positive* lead, |t|>2) before looking. H₂ is the **event-study** on three recessions "
            "(31 months) — fragile to one idiosyncratic window. H₃ is the **realisability** check: "
            "the demand figure is published with a lag *and* the recession is dated with a lag, so "
            "any ramen-conditional strategy is a double look-ahead. The claim needs all three; "
            "failing H₁ alone already reduces \"the ramen index\" to a coincidence museum."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** `2897.T`, `2875.T`, `^N225` month-end Adj Close (yfinance, cached; splits "
            "+ dividends folded in — *total-return-ish*). Sample 2001-01 → 2026-06 "
            f"(noodle names n={R['n_stock']} monthly returns; `^N225` from 2000, n={R['n_n225']}). "
            "The **ramen index** = WINA world demand, 2005–2024 (19 growth observations), "
            "**hardcoded, cited, approximate** — a labelled proxy.\n"
            "- **Survivorship — named on the Signal axis.** These are the two *surviving* noodle "
            "champions, selected ex-post; the bias points **for** the outperformance/defensiveness "
            "claim, so any positive result (notably the §4d alpha) is treated as suspect and "
            "explicitly quarantined from the Signal verdict.\n"
            "- **H₁ test.** Lead-lag Pearson correlation $\\rho(d_t, r^M_{t+k})$ for "
            "$k\\in\\{-2..+2\\}$ with a small-sample $t=r\\sqrt{(n-2)/(1-r^2)}$; plus a Welch *t* of "
            "demand growth in vs out of recession years. `REAL` needs a negative, significant "
            "lead at $k\\ge1$.\n"
            "- **H₂ test.** Paired *t* of the recession-window monthly excess $r^P-r^M$ over the "
            "union of NBER recession months; a per-recession breakdown checks it isn't one window.\n"
            "- **Bull/bear beta & CAPM α.** Conditional betas (split at $r^M=0$) and a Newey-West "
            "(6-lag) HAC *t* of the full-sample alpha vs `^N225`.\n"
            "- **H₃ / cost (beat 6).** The WINA + NBER publication lags (the double look-ahead) and "
            "the full-sample terminal-wealth comparison.\n"
            "- **Positive controls.** (a) A deterministic index engineered to *lead* the market by "
            "1 year — the cross-correlation engine must recover it; (b) a planted asymmetric-beta "
            "stock — the bull/bear engine must recover it. Machinery proofs, never market evidence.\n"
            "- **What would make us say \"real tell\":** a negative lead-lag *t* < −2 at $k\\ge1$ "
            "**and** a recession-window *t* > 2. We find neither."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — does demand *lead*? (the headline)\n\n"
            "Cross-correlation of WINA demand growth $d_t$ against the Nikkei's annual return "
            "$r^M_{t+k}$. A real tell lives in the **negative, positive-lead** quadrant."
        ),
        code(
            "if HAVE_PRICES:\n"
            "    mkt = PX['^N225'].resample('YE').last().pct_change().dropna()\n"
            "    ll = st.lead_lag_corr(data.ramen_growth()/100.0, mkt, leads=range(-2,3))\n"
            "    leads=list(ll['per_lead']); rs=[ll['per_lead'][k]['r'] for k in leads]; ts=[ll['per_lead'][k]['t'] for k in leads]\n"
            "    dv = st.demand_in_vs_out_recession(data.ramen_growth(), data.recession_years())\n"
            "else:\n"
            "    leads, rs, ts = R['ll_leads'], R['ll_r'], R['ll_t']\n"
            "    dv = dict(in_mean=R['dem_in'], out_mean=R['dem_out'], diff=R['dem_diff'], t=R['dem_t'], p=R['dem_p'])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols=[RED if k==R['ll_best_lead'] else GREY for k in leads]\n"
            "ax.bar([str(k) for k in leads], rs, .6, color=cols)\n"
            "ax.axhline(0,c='k',lw=1); ax.axvspan(2.5,4.5,color=GREEN,alpha=.06)\n"
            "ax.set_xlabel('lead k (years demand PRECEDES market); k>=1 = the tell'); ax.set_ylabel(r'$\\rho(d_t, r^M_{t+k})$')\n"
            "for i,k in enumerate(leads): ax.annotate(f't={ts[i]:+.2f}',(i,rs[i]),ha='center',va='bottom' if rs[i]>=0 else 'top',fontsize=8)\n"
            "ax.set_title('H1: no negative, significant correlation at any positive lead'); plt.tight_layout(); plt.show()\n"
            "print('lead-lag:'); [print(f'  k={k:+d}: r={rs[i]:+.2f}  t={ts[i]:+.2f}') for i,k in enumerate(leads)]\n"
            "print(f\"demand growth: recession yrs {dv['in_mean']:+.2f}%  vs other {dv['out_mean']:+.2f}%  diff {dv['diff']:+.2f}  t={dv['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the \"tell\" lead (+{R['ll_best_lead']}y) is "
            f"*r* = {R['ll_best_r']:+.2f}, *t* = {R['ll_best_t']:+.2f} — a coin flip. The largest "
            f"|correlation| sits at lead **−2** (*t* = {R['ll_t'][0]:+.2f}), i.e. the market moving "
            "*before* the noodles if anything — the opposite of leading. And demand grows "
            f"**+{R['dem_in']:.1f}%/yr** in recession years vs **+{R['dem_out']:.1f}%** otherwise "
            f"(*t* = {R['dem_t']:+.2f}). H₁ rejected: the ramen index carries no cycle information."
        ),
        md(
            "### 4b · H₂ — recession-window returns (the stocks as a hedge)\n\n"
            "Paired excess $r^P-r^M$ over the union of NBER recession months, with the "
            "per-recession breakdown that shows whether any aggregate edge is broad or one window."
        ),
        code(
            "labels=R['rec_labels']\n"
            "if HAVE_PRICES:\n"
            "    br=PX['^N225'].pct_change().dropna()\n"
            "    nre=st.recession_excess_t(PX['2897.T'].pct_change().dropna(), br, data.recession_mask)\n"
            "    tre=st.recession_excess_t(PX['2875.T'].pct_change().dropna(), br, data.recession_mask)\n"
            "    nb=st.recession_breakdown(PX['2897.T'].pct_change().dropna(), br, data.NBER_RECESSIONS)\n"
            "    tb=st.recession_breakdown(PX['2875.T'].pct_change().dropna(), br, data.NBER_RECESSIONS)\n"
            "    nisv=[nb[l]['stock']*100 for l in labels]; toyv=[tb[l]['stock']*100 for l in labels]; benv=[nb[l]['bench']*100 for l in labels]\n"
            "    nt,tt=nre['t'],tre['t']; nex,tex=nre['mean_excess']*100,tre['mean_excess']*100\n"
            "else:\n"
            "    nisv,toyv,benv=R['nis_by_rec'],R['toy_by_rec'],R['n225_by_rec']\n"
            "    nt,tt,nex,tex=R['nis_rec_t'],R['toy_rec_t'],R['nis_rec_excess'],R['toy_rec_excess']\n"
            "x=np.arange(3); fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(x-.25, nisv, .25, color=NOOD, label=f'Nissin (agg excess {nex:+.1f}%/mo, t={nt:+.2f})')\n"
            "ax.bar(x,      toyv, .25, color='#8e44ad', label=f'Toyo Suisan (agg excess {tex:+.1f}%/mo, t={tt:+.2f})')\n"
            "ax.bar(x+.25, benv, .25, color=GREEN, label='Nikkei 225')\n"
            "ax.axhline(0,c='k',lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('compounded return in recession (%)'); ax.set_title('H2: no leg clears |t|>=2; the clean wins are all COVID-2020'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "for i,l in enumerate(labels): print(f'{l:14s}  Nissin {nisv[i]:+6.1f}%  Toyo {toyv[i]:+6.1f}%  Nikkei {benv[i]:+6.1f}%')\n"
            "print(f'aggregate excess: Nissin t={nt:+.2f}  Toyo t={tt:+.2f}  (bar is |t|>=2)')"
        ),
        md(
            f"> 💡 In plain words: Nissin's recession excess is *t* = {R['nis_rec_t']:+.2f}, Toyo "
            f"Suisan's *t* = {R['toy_rec_t']:+.2f} — neither clears the bar. Both *fell* in the 2008 "
            "GFC (−24.6% / −3.1%), just less than the Nikkei's −36.5%; the unambiguous wins "
            "(2020 +7.6% / +13.0%) are the COVID pantry-loading window. \"Falls less than a "
            "collapsing index\" is low beta, not counter-cyclicality. H₂ rejected."
        ),
        md(
            "### 4c · Bull vs bear beta — is it even defensive?\n\n"
            "Conditional OLS slope on `^N225`, split by the sign of the market return. Defensive "
            "requires $\\beta^- < \\beta^+ < 1$."
        ),
        code(
            "if HAVE_PRICES: br = PX['^N225'].pct_change().dropna()\n"
            "rows={}\n"
            "for t,pre in [('2897.T','nis'),('2875.T','toy')]:\n"
            "    if HAVE_PRICES:\n"
            "        bb=st.bull_bear_beta(PX[t].pct_change().dropna(), br, split=0.0)\n"
            "        rows[t]=dict(d=bb['down_beta'],u=bb['up_beta'],f=bb['full_beta'],a=bb['asymmetry'],defn=bb['defensive'])\n"
            "    else:\n"
            "        rows[t]=dict(d=R[f'{pre}_down_beta'],u=R[f'{pre}_up_beta'],f=R[f'{pre}_beta'],a=R[f'{pre}_asym'],defn=R[f'{pre}_def'])\n"
            "x=np.arange(2); fig, ax = plt.subplots(figsize=(9.2, 4.4)); ks=list(rows)\n"
            "ax.bar(x-.2, [rows[t]['d'] for t in ks], .4, color=RED, label=r'$\\beta^-$ (down markets)')\n"
            "ax.bar(x+.2, [rows[t]['u'] for t in ks], .4, color=GREEN, alpha=.7, label=r'$\\beta^+$ (up markets)')\n"
            "ax.axhline(1.0, ls='--', c=GREY); ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(['Nissin','Toyo Suisan'])\n"
            "ax.set_ylabel('conditional beta vs Nikkei'); ax.set_title(r'Defensive $\\Leftrightarrow \\beta^-<\\beta^+<1$ — the two names disagree'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in ks: r=rows[t]; print(f\"{t}: down-beta {r['d']:+.2f}  up-beta {r['u']:+.2f}  full {r['f']:.2f}  asym {r['a']:+.2f}  defensive={r['defn']}\")"
        ),
        md(
            f"> 💡 In plain words: same split as the sibling beer study. Nissin loads *more* on the "
            f"downside ($\\beta^-$={R['nis_down_beta']:.2f} > $\\beta^+$={R['nis_up_beta']:.2f}, "
            f"asymmetry {R['nis_asym']:+.2f}) — the opposite of defensive; Toyo Suisan is weakly "
            f"defensive ($\\beta^-$={R['toy_down_beta']:.2f} < $\\beta^+$={R['toy_up_beta']:.2f}). "
            "Both full betas are ≈0.2–0.3: these are simply very low-beta stocks, which is not "
            "counter-cyclicality."
        ),
        md(
            "### 4d · CAPM α vs the Nikkei — the honest complication\n\n"
            "Newey-West (6-lag) HAC regression of each name on `^N225`. This is the one place a "
            "significant number appears — so we handle it carefully."
        ),
        code(
            "rows={}\n"
            "for t,pre in [('2897.T','nis'),('2875.T','toy')]:\n"
            "    if HAVE_PRICES:\n"
            "        nw=st.newey_west_alpha_t(PX[t].pct_change().dropna(), br, 6); s=st.summarize(PX[t])\n"
            "        rows[t]=dict(alpha=nw['alpha_ann']*100, beta=nw['beta'], tt=nw['t_alpha'], cagr=s['cagr']*100, vol=s['vol']*100, sharpe=s['sharpe'], mdd=s['mdd']*100)\n"
            "    else:\n"
            "        rows[t]=dict(alpha=R[f'{pre}_alpha'], beta=R[f'{pre}_beta'], tt=R[f'{pre}_t'], cagr=R[f'{pre}_cagr'], vol=R[f'{pre}_vol'], sharpe=R[f'{pre}_sharpe'], mdd=R[f'{pre}_mdd'])\n"
            "ks=list(rows); alphas=[rows[t]['alpha'] for t in ks]; ts=[rows[t]['tt'] for t in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3)); ax.bar(['Nissin','Toyo Suisan'], alphas, .5, color=[NOOD,'#8e44ad'])\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('annualised CAPM alpha vs Nikkei (%)')\n"
            "for i,t in enumerate(ks): ax.annotate(f'NW t={ts[i]:+.2f}',(i,alphas[i]),ha='center',va='bottom')\n"
            "ax.set_title('Significant alpha vs the Nikkei — but this is survivorship + a weak benchmark, NOT the tell')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in ks: r=rows[t]; print(f\"{t}: CAGR {r['cagr']:5.2f}%  vol {r['vol']:4.1f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  |  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['tt']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: both alphas *do* clear the bar (NW *t* = {R['nis_t']:+.2f} / "
            f"{R['toy_t']:+.2f}) — and it would be dishonest to hide that under a `NONE` stamp. But "
            "it is **not the claim**. Three reasons it is not evidence for a ramen *tell*: (1) it is "
            "about *beating the Nikkei*, not *leading recessions* — §4a already killed the timing; "
            "(2) **survivorship** — we hand-picked two surviving noodle champions, which manufactures "
            f"outperformance; (3) β≈{R['nis_beta']:.2f}–{R['toy_beta']:.2f} against an index that "
            "returned ~5%/yr through Japan's lost decades is the **low-vol / quality premium**, "
            "freely available. Real number, wrong claim — the Signal axis tracks *the tell*, which "
            "is absent."
        ),
        md(
            "### 4e · Positive controls — the engines are faithful\n\n"
            "Two machinery proofs (never market evidence): an index engineered to *lead* the market "
            "by 1 year, and a stock with a planted asymmetric (defensive) beta."
        ),
        code(
            "ig, mk = data.synthetic_leading_index(lead=1, seed=729)\n"
            "cl = st.control_recovers_lead(ig, mk, planted_lead=1)\n"
            "mkt2, stock2 = data.synthetic_defensive(beta_down=0.5, beta_up=1.0, seed=7290)\n"
            "cd = st.control_recovers_defensive(stock2, mkt2, 0.5, 1.0)\n"
            "ll = st.lead_lag_corr(ig, mk, leads=range(-2,4))\n"
            "leads=list(ll['per_lead']); rs=[ll['per_lead'][k]['r'] for k in leads]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.4,4.0))\n"
            "a1.bar([str(k) for k in leads], rs, .6, color=[GREEN if k==1 else GREY for k in leads]); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_title(f'planted lead=1 -> recovered at k={cl[\"best_lead\"]} (r={cl[\"best_r\"]:.2f}, t={cl[\"best_t\"]:.1f})'); a1.set_xlabel('lead k'); a1.set_ylabel(r'$\\rho$')\n"
            "down=mkt2[mkt2<0]; up=mkt2[mkt2>=0]\n"
            "a2.scatter(mkt2, stock2, s=8, c=np.where(mkt2<0,RED,GREEN), alpha=.5)\n"
            "xs=np.linspace(mkt2.min(),0,20); a2.plot(xs, cd['down_beta']*xs, c=RED, lw=2, label=f\"beta- {cd['down_beta']:.2f}\")\n"
            "xs2=np.linspace(0,mkt2.max(),20); a2.plot(xs2, cd['up_beta']*xs2, c=GREEN, lw=2, label=f\"beta+ {cd['up_beta']:.2f}\")\n"
            "a2.set_title('planted defensive stock: beta- < beta+ recovered'); a2.set_xlabel('market ret'); a2.set_ylabel('stock ret'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"lead control: planted 1 -> best_lead {cl['best_lead']} r={cl['best_r']:.2f} t={cl['best_t']:.1f} ok={cl['recovered_lead_ok']}\")\n"
            "print(f\"defensive control: planted 0.50/1.00 -> recovered {cd['down_beta']:.2f}/{cd['up_beta']:.2f} defensive={cd['recovered_defensive']}\")"
        ),
        md(
            f"> 💡 In plain words: when an index genuinely leads, the engine nails it "
            f"(recovered lead **{R['syn_lead']}**, *r* ≈ {R['syn_lead_r']:.2f}, *t* ≈ "
            f"{R['syn_lead_t']:.1f}); when a stock is genuinely defensive, the bull/bear split sees "
            f"it ($\\beta^-$≈{R['syn_down']:.2f} < $\\beta^+$≈{R['syn_up']:.2f}). So the flat, "
            "name-dependent nulls on the real ramen tape are *genuine* nulls, not broken detectors. "
            "A synthetic control is a machinery proof, never market evidence."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — on the claim under test (the tell): no lead clears the bar "
            f"(best *t* = {R['ll_best_t']:+.2f} at lead +{R['ll_best_lead']}y), demand grows the "
            f"same in/out of recessions (*t* = {R['dem_t']:+.2f}), and the recession-window excess "
            f"is insignificant (*t* = {R['nis_rec_t']:+.2f} / {R['toy_rec_t']:+.2f}). The one "
            f"significant item — CAPM α vs the Nikkei (NW *t* = {R['nis_t']:+.2f} / {R['toy_t']:+.2f}) "
            "— is a survivorship + weak-benchmark artefact, not the tell.\n"
            f"- **Tradability `MIRAGE`** — a double look-ahead (WINA ~{R['wina_lag']} mo + NBER "
            f"~{R['nber_lag']} mo ex-post) makes the tell un-implementable; the market-beating leg "
            "rests on picking the survivors in hindsight.\n"
            f"- **A leading tell? `BUSTED`** — global demand *fell* in the 2008 GFC "
            f"({R['dem_2008']:+.1f}%, {R['dem_2009']:+.1f}%) and in 2014–2016; the only spike was "
            f"COVID-2020 (+{R['dem_2020']:.1f}%). Instant noodles are a secular Asian growth story."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the look-ahead reality\n\n"
            "Two walls. **The double look-ahead:** the ramen-conditional rule needs both this "
            "year's demand (WINA publishes it ~6 months into the *next* year) and confirmation a "
            "recession is underway (NBER dates it ~12 months ex-post). **The survivorship trap:** "
            "the only market-beating expression — owning the noodle makers — is available only if "
            "you knew *ex-ante* which brands would survive and win."
        ),
        code(
            "start=10_000.0\n"
            "if HAVE_PRICES:\n"
            "    mult=[st.terminal_wealth(PX[t]) for t in ['^N225','2897.T','2875.T']]\n"
            "else:\n"
            "    mult=[R['n225_wealth'],R['nis_wealth'],R['toy_wealth']]\n"
            "ends=[start*m for m in mult]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(['Nikkei 225','Nissin (2897.T)','Toyo Suisan (2875.T)'], ends, .55, color=[GREEN, NOOD, '#8e44ad'])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(start, ls=':', c=GREY, label='$10,000 start'); ax.set_ylabel('value of $10,000, 2001->2026'); ax.legend()\n"
            "ax.set_title('The survivors beat the index — but that is hindsight selection, not the tell')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,v in zip(['Nikkei','Nissin','Toyo Suisan'], ends): print(f'{l}: ${v:,.0f}')\n"
            "print(f\"WINA lag ~{R['wina_lag']} mo + NBER lag ~{R['nber_lag']} mo -> the ramen tell is a double look-ahead\")"
        ),
        md(
            "> 💡 In plain words: there is no implementable version of the *tell*. The leading-"
            "indicator claim is false (§4a) and, even if it weren't, the reading arrives half a year "
            "late. The buy-and-hold-the-noodle-makers version *did* win — but only because we "
            "selected the winners after the fact, and what we bought was low-beta staples beating a "
            "uniquely weak index, not a recession forecast. Real edge, wrong claim, and "
            "un-front-runnable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **US-only demand.** The folklore's anecdote is *American* ramen sales in 2008; swap "
            "the global WINA figure for a US Nielsen/IRI series and re-run H₁ — does the tell exist "
            "on home turf, or is it also one recession's noise plus COVID?\n"
            "- **Real-time recession conditioning.** Replace the ex-post NBER flag with a Sahm-rule "
            "/ term-spread nowcast and test whether *any* staple-demand series leads it with an "
            "honest, no-look-ahead entry — the only way the timing question becomes tradable.\n"
            "- **The staples zoo.** Extend to the lipstick index, discount retail (`DG`, `WMT`), "
            "Spam (`HRL`), pooling across many candidate \"indices\" *with* a White (2000) "
            "Reality-Check correction — the honest way to ask whether *any* consumer oddity is a "
            "true tell ([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; prices are real (yfinance, "
            "total-return-ish), the WINA series and NBER windows are **cited**. Methods: "
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
