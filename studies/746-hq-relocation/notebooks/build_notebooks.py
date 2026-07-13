"""Generate the two narrative notebooks for Study 746 (HQ-Relocation).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily closes
under ../_cache/ (one parquet per ticker + SPY), sliced to the frozen as-of, and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic
positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 20-name HQ-move
# table + SPY, 2010-01-04 -> 2026-06-30 as-of; 20 events priced; fingerprint 3fdda93c2d9a).
R = dict(
    n_table=20, n_priced=20, n_tax=14, n_other=6,
    as_of="2026-06-30", fingerprint="3fdda93c2d9a",
    # CAR[0,+2]: bucket -> (mean%, win%, t)
    car_tax=(0.45, 50, 0.35),
    car_other=(-0.40, 67, -0.35),
    car_all=(0.20, 55, 0.21),
    diff_pct=0.85, diff_t=0.50, all_placebo_p=0.735,
    # announcement day [0,0]: (mean%, t, win%, placebo_p)
    day00=(0.18, 0.23, 55, 0.588),
    # drift: window -> (all_mean%, all_t, tax_mean%, tax_t, net%, placebo_p, win%)
    drift21=(-0.06, -0.03, 0.85, 0.26, -0.16, None, None),
    drift63=(3.39, 1.34, 4.66, 1.46, 3.29, 0.301, 70),
    # robustness: (window_label, all_mean%, all_t, diff_pp, diff_t)
    robust=[("[0,0]", 0.18, 0.23, -0.49, -0.41),
            ("[0,+2]", 0.20, 0.21, 0.85, 0.50),
            ("[-1,+1]", 1.10, 0.88, 0.23, 0.12),
            ("[0,+4]", 0.34, 0.32, 1.86, 0.97)],
    # synthetic: planted_bps, tax_mean%, tax_t, other_mean%, diff_t
    syn=[(0.0, -0.43, -0.49, -1.10, 0.45),
         (500.0, 4.57, 5.12, -1.10, 3.79)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Signal_or_distraction%3F: Non-event](https://img.shields.io/badge/Signal_or_distraction%3F-Non--event-8b949e?style=flat-square)\n\n"
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

from hq_relocation import data, strategy as st

AS_OF = "2026-06-30"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    PRICES = PRICES[PRICES.index <= AS_OF]
    PANEL = st.car_panel(PRICES, EVENTS)                 # canonical CAR[0,+2]
else:
    PRICES = EVENTS = PANEL = None
print("real HQ-move cache present:", HAVE_REAL,
      "| events priced:", (0 if PANEL is None else len(PANEL)),
      "| table fingerprint:", data.fingerprint(data.HQ_MOVES))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The headquarters move — does a new address pay? 🏢\n"
            "### When a company decamps to Ireland or Texas for the taxes, is that a *buy*, a *fade*, or nothing at all?\n\n"
            + BADGES +
            "Here's a story you've heard: a big company **announces moving its headquarters** — "
            "Medtronic re-flags to Ireland, Tesla and Oracle and Chevron pull up stakes for "
            "**Texas** — and it's supposed to *mean* something. One camp says **buy it**: the tax "
            "bill just dropped, the cost base just got lighter, the market will price that in. The "
            "other camp says **fade it**: a splashy new address is management theatre, a shiny "
            "distraction bolted onto a business that isn't working.\n\n"
            "So which is it — **signal or distraction**? This notebook builds a transparent table "
            "of **~20 real HQ moves** from 2010–2025 and just *looks* at what the stocks did around "
            "each announcement.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** There's no free, tidy database of HQ moves, so we "
            "hand-list the famous, dated ones — which means we're looking at the moves *big enough "
            "to be remembered*, by companies that *survived*. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock pop when the move is announced? | **No.** Across ~20 moves the "
            f"three-day reaction averages **+{R['car_all'][0]:.2f}%** — a rounding error, positive "
            f"about half the time. |\n"
            "| Does the *reason* (tax vs. talent) change the reaction? | **No.** Tax-driven moves "
            f"average **+{R['car_tax'][0]:.2f}%**, other moves **{R['car_other'][0]:.2f}%** — a "
            f"**{R['diff_pct']:+.1f}pp** gap that's pure noise. |\n"
            "| Is there a slow 'the tax saving gets priced in' drift? | **A whiff, but you can't "
            f"trust it.** Over the next quarter the stocks drift **+{R['drift63'][0]:.1f}%** — but "
            "that's inside the noise (a random quarter looks this good ~30% of the time). |\n"
            "| So is it a signal or a distraction? | **Neither.** No reliable pop (not a signal), "
            "no reliable slump (not a distraction). On the tape, a new HQ is a **non-event**. |\n\n"
            "> The market greets a headquarters move with a shrug. The tax saving may be real for "
            "the company; it just isn't a tradable event for the stock."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch the corporate address. When a company re-domiciles to a low-tax country or "
            "state, its after-tax cash flow structurally jumps — and the market reprices it. The "
            "tax-driven moves are the buys; the vague 'we want to be closer to talent' moves are "
            "the tells that management is chasing optics. Trade the reason.\"*\n\n"
            "It's not a crazy claim. A tax inversion genuinely lowers the effective tax rate; a "
            "move from California to Texas genuinely cuts costs. The academic anchor is real too: "
            "**Desai & Hines (2002)** found inversion announcements earned *positive* average "
            "abnormal returns — small, heterogeneous, but positive. The believers extend that into "
            "a **buy-the-tax-move / fade-the-fluff** rule. We'll test both halves on a real, "
            "representative table."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real and repeatable, it'd be a tidy little calendar: every corporate "
            "relocation a scheduled catalyst you could position for. It would also say something "
            "clean about markets — that a **domicile change** is fresh information the price hasn't "
            "absorbed. But two things have to hold. The reaction has to be **real on average** (not "
            "just in the one or two moves we remember), and the *reason* has to **actually sort** "
            "the winners from the theatre. Miss either and the 'trade the reason' rule is just a "
            "story we tell after the fact."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent table of ~{R['n_table']} real HQ moves** (2010–2025) — the "
            "Ireland inversions (Eaton, Medtronic, Johnson Controls), the Texas migration (Schwab, "
            "CBRE, HPE, Oracle, Tesla, Caterpillar, Chevron), and the 'talent/proximity' moves (GE "
            "→ Boston, Boeing → Arlington). For each one:\n\n"
            "1. **Line up the stock against the market (SPY)** around the announcement, and measure "
            "the **abnormal return** — the bit of the move that *isn't* just the market that day.\n"
            "2. **Add it up** over a few days after the headline (the reaction), and over the next "
            "quarter (the slow drift the 'signal' camp needs).\n"
            "3. **Stress the luck.** Draw the same number of *random* windows on the same stocks "
            "thousands of times and ask how often chance produces a reaction this big. With only "
            "~20 events, that's the honest test.\n\n"
            "And we say the catch out loud: the tax-vs-talent label is **fuzzy** (a Texas move cuts "
            "*both* taxes and rent), and we only see the moves famous enough to remember. If even "
            "*this* stacked deck shows nothing, that's telling."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, every move's three-day reaction.** For each HQ announcement: the abnormal "
            "(market-adjusted) return over the announcement day plus the next two, coloured by "
            "whether the move was tax-driven or not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = PANEL.sort_values('announce_date')\n"
            "    cars = df['car'].values*100; labs = df['ticker'].values; istax = df['tax'].values\n"
            "else:\n"
            "    rng=np.random.default_rng(746)\n"
            "    cars=rng.normal(R['car_all'][0],4,R['n_priced']); labs=[f'M{i}' for i in range(R['n_priced'])]\n"
            "    istax=np.array([True]*R['n_tax']+[False]*R['n_other'])\n"
            "colors=[AMBER if t else GREY for t in istax]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "x=np.arange(len(cars))\n"
            "ax.bar(x, cars, color=colors, edgecolor='k', linewidth=.4)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(cars), c=GREEN, ls='--', label=f'mean {np.mean(cars):+.2f}%')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs, rotation=60, ha='right', fontsize=8)\n"
            "ax.set_ylabel('3-day abnormal return (%)')\n"
            "ax.set_title('The reaction to an HQ move (amber = tax-driven, grey = other) — scattered around zero')\n"
            "from matplotlib.patches import Patch\n"
            "ax.legend(handles=[Patch(color=AMBER,label='tax/incentive'),Patch(color=GREY,label='other'),\n"
            "                   plt.Line2D([],[],c=GREEN,ls='--',label=f'mean {np.mean(cars):+.2f}%')])\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean 3-day reaction {np.mean(cars):+.2f}%   (a real catalyst would be well away from 0)')"
        ),
        md(
            f"There's the first tell. The bars scatter on both sides of zero — some moves popped "
            f"(JCI +11.5% around the Tyco inversion), some sank (CAT, CVX, BA all *negative*) — and "
            f"the average is a nothing **+{R['car_all'][0]:.2f}%**. The colours don't separate: "
            "tax-driven amber and talent-driven grey are mixed all through."
        ),
        md(
            "**Does the *reason* sort them?** The believers' core claim: tax-driven moves should "
            "beat the vague 'talent' ones. Here are the two buckets' averages, side by side."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tax=PANEL.loc[PANEL['tax'],'car'].values*100; oth=PANEL.loc[~PANEL['tax'],'car'].values*100\n"
            "    tm, om = tax.mean(), oth.mean()\n"
            "    tw, ow = (tax>0).mean()*100, (oth>0).mean()*100\n"
            "else:\n"
            "    tm, om = R['car_tax'][0], R['car_other'][0]; tw, ow = R['car_tax'][1], R['car_other'][1]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.2,4.2))\n"
            "a1.bar(['tax /\\nincentive','other\\nrationale'], [tm, om], color=[AMBER, GREY], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean 3-day abnormal return (%)')\n"
            "a1.set_title(f'The reason barely moves the needle ({tm-om:+.1f}pp gap)')\n"
            "for i,v in enumerate([tm,om]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['tax','other'], [tw, ow], color=GREY, width=.55)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of moves with a positive reaction'); a2.set_title('Both buckets ~ a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'tax {tm:+.2f}% (win {tw:.0f}%)   other {om:+.2f}% (win {ow:.0f}%)  -> gap {tm-om:+.2f}pp')"
        ),
        md(
            f"The tax bucket is **+{R['car_tax'][0]:.2f}%**, the other bucket "
            f"**{R['car_other'][0]:.2f}%** — a **{R['diff_pct']:+.1f}pp** gap, and note the 'other' "
            f"moves actually won *more often* (67% vs 50%). The reason for the move doesn't sort the "
            "reaction. The 'trade the tax moves' half of the rule is already gone."
        ),
        md(
            "**Could ~20 random windows look this 'special'?** The honest small-sample test: draw "
            f"**{R['n_priced']}** *random* three-day windows on the same stocks, over and over, and "
            "see where the real HQ moves land against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(PANEL), n_draws=4000)*100\n"
            "    obs = PANEL['car'].mean()*100\n"
            "    pval = st.placebo_pvalue(PANEL['car'].mean(), null/100)\n"
            "else:\n"
            "    rng=np.random.default_rng(746); null=rng.normal(0,1.0,4000); obs=R['car_all'][0]; pval=R['all_placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=45, color=GREY, alpha=.85, label=f'{R[\"n_priced\"]} RANDOM windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual HQ moves ({obs:+.2f}%)')\n"
            "ax.set_xlabel('average 3-day abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The reaction sits dead-centre in the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_priced\"]}-window draw matches the HQ-move reaction {pval*100:.0f}% of the time — not rare at all')"
        ),
        md(
            f"The green line — the real HQ moves — sits **smack in the middle** of the grey luck "
            f"cloud (placebo *p* ≈ **{R['all_placebo_p']:.2f}**). In plain terms: **twenty random "
            "dates on these same stocks would look about this 'eventful' three-quarters of the "
            "time.** There's no announcement effect to find."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The reaction is **+{R['car_all'][0]:.2f}%** (placebo *p* = "
            f"**{R['all_placebo_p']:.2f}**), the reason doesn't sort it (**{R['diff_pct']:+.1f}pp** "
            "gap), and even the slow drift is inside the noise. Nothing clears the bar.\n"
            "- **Tradability — Mirage.** No day-one move to grab, no reliable drift to hold. The "
            f"one positive whiff (+{R['drift63'][0]:.1f}% over a quarter) is exactly what a couple "
            "of big Texas movers in a bull market produce by accident.\n"
            "- **\"Signal or distraction?\" — Non-event.** Both camps lose: no reliable pop (not a "
            "signal), no reliable slump (not a distraction). The market shrugs at a new address."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the drift that isn't there\n\n"
            "Forget the announcement instant and give the 'signal' camp its best shot: **buy the "
            "day after and hold for a quarter**, so the tax saving has time to get priced in. "
            "Here's the abnormal return of that trade — over a month, and over a quarter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d21 = st.car_panel(PRICES, EVENTS, window=(1,21))['car'].values*100\n"
            "    d63 = st.car_panel(PRICES, EVENTS, window=(1,63))['car'].values*100\n"
            "    m21, m63 = d21.mean(), d63.mean(); t63 = st.welch_t(d63/100)\n"
            "else:\n"
            "    m21, m63, t63 = R['drift21'][0], R['drift63'][0], R['drift63'][1]\n"
            "net63 = m63 - 0.10   # one-way 10bps round trip\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar(['hold ~1 month\\n[+1,+21]','hold ~1 quarter\\n[+1,+63]','quarter,\\nnet @10bps'],\n"
            "       [m21, m63, net63], color=[GREY, AMBER, RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('abnormal return of the trade (%)')\n"
            "ax.set_title(f'A quarter-long drift of {m63:+.1f}% — but t = {t63:.2f}, inside the noise')\n"
            "for i,v in enumerate([m21,m63,net63]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'month {m21:+.2f}%   quarter {m63:+.2f}% (t={t63:.2f}, placebo p={R[\"drift63\"][5]})   net {net63:+.2f}%')"
        ),
        md(
            f"The month is flat (**{R['drift21'][0]:+.2f}%**). The quarter is **+{R['drift63'][0]:.1f}%** "
            f"— the single number that leans the believers' way — but it lands at **t = {R['drift63'][1]:.2f}** "
            f"(placebo *p* = **{R['drift63'][5]}**): a random quarter on these names looks this good "
            "about a third of the time. It survives the tiny cost (net "
            f"**+{R['drift63'][4]:.1f}%**), but you can't bank a *t* of 1.3 — especially when it's "
            "carried by a handful of Texas movers in the 2020–21 melt-up. There's no machine here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other 'cosmetic change' study.** [Study 389 — Name-Change-Effect]"
            "(../389-name-change-effect/) asks whether a *theme-chasing rebrand* pops — same family "
            "(a label/address, not a fundamental), same shrug.\n"
            "- **The direct sibling.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "market-model event study, on firing the boss instead of moving the office.\n"
            "- **Add the corpses & the conditioning.** Our tape is the moves famous enough to "
            "remember; a survivorship-free relocation feed, or a split by *how much* the effective "
            "tax rate actually fell, might isolate a real inversion sub-effect — Desai & Hines say "
            "it's small and conditional, which is exactly why ~20 blended events can't see it.\n\n"
            "*Think the tax-move signal is real and harvestable? Capture the events, draw the same "
            "number of random windows, and show the reaction landing **outside** the cloud **and** a "
            "drift that clears **t = 2** — then we'll talk.*"
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
            "# HQ-Relocation — a quantitative teardown 🔬\n"
            "### Market-model CAR on a relocation table · tax vs other buckets · a Welch *t* + "
            "placebo randomization null · the announcement-day vs holdable-window split · a lagged "
            "drift leg + costs · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "treat the folklore as a **market-model event-study hypothesis** — a positive "
            "announcement CAR (the 'signal' camp) or a negative post-announcement drift (the "
            "'distraction' camp), sorted by the move's tax vs other motive — and confront it with "
            "the **sample size** and with **salience selection**. The decisive objects are a "
            f"cross-section of {R['n_priced']} abnormal-return events and a placebo null sized to "
            "that count.\n\n"
            "> ⚠️ **Data + selection note.** The relocation table is hardcoded and transparent "
            f"(~{R['n_table']} real moves, 2010–2025); the tape is **salience-selected** (the moves "
            "famous enough to date) and **survivor-only** (named on the Signal axis), and the "
            "tax/other label is the believers' own, subjective framing. Real data: yfinance daily "
            f"adjusted closes, 2010→{R['as_of']} (as-of), fingerprint **{R['fingerprint']}**. "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | CAR[0,+2] **+{R['car_all'][0]:.2f}%** (Welch **t = "
            f"{R['car_all'][2]:.2f}**, placebo **p = {R['all_placebo_p']:.2f}**); day-one "
            f"**+{R['day00'][0]:.2f}%** (**t = {R['day00'][1]:.2f}**); tax−other "
            f"**+{R['diff_pct']:.2f}pp** (**t = {R['diff_t']:.2f}**); quarter drift "
            f"**+{R['drift63'][0]:.1f}%** (**t = {R['drift63'][1]:.2f}**). Nothing clears t = 1. |\n"
            f"| **Tradability** | `MIRAGE` | no day-one repricing and no significant drift; the one "
            f"whiff (+{R['drift63'][0]:.1f}%/qtr, net +{R['drift63'][4]:.1f}%) is un-certifiable and "
            "beta-in-disguise on a 20-name survivor tape. |\n"
            f"| **Signal or distraction?** | `NON-EVENT` | both camps fail — no significant +pop "
            "(no signal), no significant −drift (no distraction). |\n\n"
            "> 💡 In plain words: an HQ move is a strategic, long-horizon decision that the market "
            "has largely anticipated and that carries no clean short-window catalyst — so neither "
            "the 'buy the tax saving' nor the 'fade the theatre' story survives contact with the "
            "tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For event $i$ with announcement day $\\tau_i$, fit the **market model** "
            "$r_{i,t} = \\alpha_i + \\beta_i r_{m,t} + \\varepsilon_{i,t}$ on a clean estimation "
            "window $[\\tau_i - g - L,\\ \\tau_i - g]$ ($L=120$ days, gap $g=10$), and cumulate the "
            "**abnormal return** $\\mathrm{CAR}_i[a,b] = \\sum_{t=\\tau_i+a}^{\\tau_i+b}"
            "(r_{i,t} - \\hat\\alpha_i - \\hat\\beta_i r_{m,t})$.\n\n"
            "- **H₁ (signal / pop).** $\\mathbb{E}[\\mathrm{CAR}_i[0,2]] > 0$ — the market prices "
            "the move in (Desai-Hines found a small positive inversion effect).\n"
            "- **H₂ (the reason sorts it).** $\\mathbb{E}[\\mathrm{CAR}\\mid \\text{tax}] > "
            "\\mathbb{E}[\\mathrm{CAR}\\mid \\text{other}]$ — tax-driven moves beat talk of "
            "'talent'.\n"
            "- **H₃ (drift / deployable).** $\\mathbb{E}[\\mathrm{CAR}_i[+1,+63]] \\ne 0$ and "
            "harvestable net of costs — the slow repricing the 'signal' camp needs.\n\n"
            "We find **H₁ not supported** (CAR $\\approx 0$, $t\\approx 0.2$, placebo $p=0.74$), "
            "**H₂ rejected** (the gap is $+0.85$pp at $t=0.50$, and 'other' wins *more* often), "
            "**H₃ not supported** ($+3.4\\%$/qtr at $t=1.34$, inside its placebo). The distraction "
            "camp fares no better — the drift is *positive*, not negative. It is a **non-event**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is a set of one- and two-sample tests on small cross-sections, judged by "
            "their **standard error** and by a **randomization null**:\n\n"
            "$$t_{\\text{bucket}} = \\frac{\\overline{\\mathrm{CAR}}}{s/\\sqrt{k}},\\qquad "
            "k\\in\\{6,14,20\\}.$$\n\n"
            "With $k\\le 20$ and daily-return volatility over multi-day windows, $s/\\sqrt{k}$ is "
            "large — a sub-percent mean drowns in its own SE. Worse, the sample is **conditioned on "
            "salience**: we see the moves memorable enough to date, by firms that survived. The "
            "honest instrument is a **placebo test**: resample $k$ random non-event windows on the "
            "same tickers (same market-model machinery) and ask how often chance matches the "
            "observed CAR. That randomization $p$, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Relocation table.** ~{R['n_table']} documented HQ moves 2010–2025 "
            f"(**{R['n_tax']}** tax/incentive, **{R['n_other']}** other), hardcoded & transparent; "
            "famous un-priceable moves (Burger King, Mylan, Walgreens' *abandoned* inversion) are "
            "listed for the selection caveat.\n"
            "- **Abnormal returns.** Market model $r = \\alpha+\\beta\\,r_{SPY}$, 120-day "
            "estimation window, 10-day gap; CAR over the event window.\n"
            "- **Null #1 (Welch t).** Each bucket's mean CAR vs zero; the tax−other two-sample t.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random non-event windows on the same "
            "tickers; $p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample "
            "workhorse.\n"
            "- **Drift + costs.** A [+1,+63] leg entered the day after the headline, net of a "
            "one-way 10-bps large-cap round trip.\n"
            "- **Positive control.** Deterministic panels with a **planted** tax-bucket CAR edge: "
            "the inference must recover a large edge **and** must NOT manufacture significance when "
            "the edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The buckets — nothing, and the reason doesn't sort it\n\n"
            "Mean CAR[0,+2] per bucket with $\\pm$ standard error, against zero (dashed). Tax and "
            "other straddle zero; the gap is inside the error bars."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tax=PANEL.loc[PANEL['tax'],'car'].values; oth=PANEL.loc[~PANEL['tax'],'car'].values\n"
            "    tm, om = tax.mean()*100, oth.mean()*100\n"
            "    tse, ose = tax.std(ddof=1)/np.sqrt(len(tax))*100, oth.std(ddof=1)/np.sqrt(len(oth))*100\n"
            "    tt, ot = st.welch_t(tax), st.welch_t(oth); dt = st.welch_t(tax, oth)\n"
            "else:\n"
            "    tm, om, tt, ot, dt = R['car_tax'][0], R['car_other'][0], R['car_tax'][2], R['car_other'][2], R['diff_t']\n"
            "    tse, ose = abs(tm/max(tt,1e-9)), abs(om/max(abs(ot),1e-9))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['tax / incentive','other rationale'], [tm, om], yerr=[tse, ose], capsize=6,\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean CAR[0,+2] (%)')\n"
            "ax.set_title(f'tax t={tt:.2f}, other t={ot:.2f}, tax-other t={dt:.2f} — all n.s.')\n"
            "for i,v in enumerate([tm,om]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tax {tm:+.2f}% (t={tt:.2f})  other {om:+.2f}% (t={ot:.2f})  diff {tm-om:+.2f}pp (t={dt:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the tax bucket is **+{R['car_tax'][0]:.2f}%** (t = "
            f"{R['car_tax'][2]:.2f}), the other bucket **{R['car_other'][0]:.2f}%** (t = "
            f"{R['car_other'][2]:.2f}), and the gap H₂ needs is **+{R['diff_pct']:.2f}pp at t = "
            f"{R['diff_t']:.2f}** — indistinguishable from zero. The move's *reason* carries no "
            "information about its reaction."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_priced']} random non-event windows 8,000 times (same market model); the "
            "histogram is the null for the all-events mean CAR. The real reaction is the green line; "
            "the *p*-value is the two-sided tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(PANEL), n_draws=8000)*100\n"
            "    obs = PANEL['car'].mean()*100; pval = st.placebo_pvalue(PANEL['car'].mean(), null/100)\n"
            "else:\n"
            "    rng=np.random.default_rng(746); null=rng.normal(0,1.0,8000); obs=R['car_all'][0]; pval=R['all_placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_priced\"]} random windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed CAR {obs:+.2f}%')\n"
            "ax.axvline(null.mean(), c='k', ls=':', lw=1, label=f'null mean {null.mean():+.2f}%')\n"
            "ax.set_xlabel('mean 3-day abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the reaction is dead-centre in the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_priced\"]}-window mean| >= |CAR|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['all_placebo_p']*100:.0f}%** of random {R['n_priced']}-window "
            "draws match or beat the observed reaction in magnitude. A real effect would push the "
            "green line into the tail; instead it sits on top of the null mean. H₁ is **not "
            "supported** — this is what a couple-dozen random dates look like."
        ),
        md(
            "### 4c · Announcement-day vs holdable window + robustness\n\n"
            "Unlike a forced-CEO ouster, there isn't even an announcement-*day* jolt to lose to the "
            "close: the [0,0] window is a shrug too. And the non-result is flat across every window "
            "choice — the *t* never clears 1."
        ),
        code(
            "wins = [(0,0),(0,2),(-1,1),(0,4)]; labs=['[0,0]','[0,+2]','[-1,+1]','[0,+4]']\n"
            "if HAVE_REAL:\n"
            "    allm=[]; allt=[]; difft=[]\n"
            "    for w in wins:\n"
            "        pw=st.car_panel(PRICES, EVENTS, window=w); sw=st.summarize(pw)\n"
            "        allm.append(sw['all']['mean_pct']); allt.append(sw['all']['t']); difft.append(sw['diff_t'])\n"
            "else:\n"
            "    allm=[r[1] for r in R['robust']]; allt=[r[2] for r in R['robust']]; difft=[r[4] for r in R['robust']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2)); xx=np.arange(len(labs))\n"
            "a1.bar(xx, allm, color=AMBER, width=.55); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_xticks(xx); a1.set_xticklabels(labs); a1.set_ylabel('all-events mean CAR (%)')\n"
            "a1.set_title('Mean CAR by window — flat near zero')\n"
            "for i,v in enumerate(allm): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom',fontsize=8)\n"
            "a2.bar(xx-.2, allt, .4, color=AMBER, label='all-events t'); a2.bar(xx+.2, difft, .4, color=GREY, label='tax-other t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(xx); a2.set_xticklabels(labs); a2.set_ylabel('Welch t'); a2.set_title('No window clears t=1, let alone t=2')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('windows:', list(zip(labs, [round(m,2) for m in allm], [round(t,2) for t in allt])))"
        ),
        md(
            f"> 💡 In plain words: the announcement day itself is **+{R['day00'][0]:.2f}%** at **t = "
            f"{R['day00'][1]:.2f}** (placebo *p* = {R['day00'][3]:.2f}) — there is no instantaneous "
            "repricing to find, tradable or not. Widen or shift the window and it stays flat. The "
            "non-result is not a window-choice artefact."
        ),
        md(
            "### 4d · The drift leg — the believers' last stand\n\n"
            "Give H₃ its best shot: enter the day after and hold. Mean abnormal return over a month "
            "and a quarter, all-events and the tax bucket, with the quarter's placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d63 = st.car_panel(PRICES, EVENTS, window=(1,63))\n"
            "    alld = d63['car'].values; taxd = d63.loc[d63['tax'],'car'].values\n"
            "    am, tmn = alld.mean()*100, taxd.mean()*100; at = st.welch_t(alld)\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(alld), window=(1,63), n_draws=4000)*100\n"
            "    pval = st.placebo_pvalue(alld.mean(), null/100)\n"
            "else:\n"
            "    am, tmn, at = R['drift63'][0], R['drift63'][2], R['drift63'][1]\n"
            "    rng=np.random.default_rng(7); null=rng.normal(0,2.5,4000); pval=R['drift63'][5]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=45, color=GREY, alpha=.85, label='null: random quarters')\n"
            "ax.axvline(am, c=GREEN, lw=2.5, label=f'observed drift {am:+.1f}% (t={at:.2f})')\n"
            "ax.set_xlabel('quarter [+1,+63] abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The quarter drift is a whiff — placebo p = {pval:.2f}, inside the null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'quarter drift: all {am:+.2f}% (t={at:.2f}, placebo p={pval:.2f})   tax bucket {tmn:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: the quarter drift is **+{R['drift63'][0]:.1f}%** (tax bucket "
            f"**+{R['drift63'][2]:.1f}%**) — the one number that leans the 'signal' way — but at **t "
            f"= {R['drift63'][1]:.2f}**, placebo **p = {R['drift63'][5]}**, it is squarely inside "
            "the null. On a 20-name survivor tape it is what a handful of Texas movers in the "
            "2020–21 melt-up produce; it is beta-in-disguise, not a certifiable drift, and it fails "
            "the t ≥ 2 bar."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic per-event panels with a **planted** tax-bucket CAR edge: with "
            "**0 bps** the inference must stay flat (a couple-dozen noisy events can't fake "
            "significance); with a **+500 bps** planted edge it must light up. Both hold — proving "
            "the engine is unbiased *and* that this sample size only detects implausibly large "
            "effects."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 500.0):\n"
            "    syn = data.synthetic_events(car_bps=edge, seed=746)\n"
            "    tt = st.summarize_bucket(syn['tax_car']); dt = st.welch_t(syn['tax_car'], syn['other_car'])\n"
            "    res.append((edge, tt['mean_pct'], tt['t'], st.summarize_bucket(syn['other_car'])['mean_pct'], dt))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=[f'planted\\n{int(e)}bps' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "tts=[r[2] for r in res]; dts=[r[4] for r in res]\n"
            "ax.bar(xx-.2, tts, .4, color=AMBER, label='tax-bucket t'); ax.bar(xx+.2, dts, .4, color=GREY, label='tax-other t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t')\n"
            "ax.set_title('Control: only a HUGE planted edge lights up the buckets'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,tm,tt,om,dt in res: print(f'planted {int(e):>4}bps: tax={tm:+.2f}%(t={tt:+.2f}) other={om:+.2f}% diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's tax *t* is "
            f"**{R['syn'][0][2]:.2f}** and the tax−other *t* is **{R['syn'][0][4]:.2f}** (no false "
            f"positive); only the **+500 bps** plant reaches tax *t* **{R['syn'][1][2]:.2f}** / "
            f"diff *t* **{R['syn'][1][4]:.2f}**. So the machinery is honest, and the real-tape CAR "
            f"*t* of **{R['car_all'][2]:.2f}** is exactly what an *absent* effect looks like through "
            f"a {R['n_priced']}-event keyhole."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — CAR[0,+2] **+{R['car_all'][0]:.2f}%** at Welch **t = "
            f"{R['car_all'][2]:.2f}** / placebo **p = {R['all_placebo_p']:.2f}**; the announcement "
            f"day is **+{R['day00'][0]:.2f}%** (**t = {R['day00'][1]:.2f}**); tax−other "
            f"**+{R['diff_pct']:.2f}pp** (**t = {R['diff_t']:.2f}**); the quarter drift "
            f"**+{R['drift63'][0]:.1f}%** (**t = {R['drift63'][1]:.2f}**, placebo p = "
            f"{R['drift63'][5]}). Nothing clears t = 1. Salience-selection & the subjective "
            "tax/other label named on this axis.\n"
            f"- **Tradability `MIRAGE`** — no day-one move to capture and no significant drift; the "
            f"one whiff (+{R['drift63'][0]:.1f}%/qtr, net +{R['drift63'][4]:.1f}%) is un-certifiable "
            "and beta-in-disguise on a 20-name survivor tape. No sign-stable edge at any horizon.\n"
            f"- **Signal or distraction? `NON-EVENT`** — both camps fail: no significant positive "
            "pop (no signal) and no significant negative drift (no distraction penalty). On the "
            "tape, an HQ-move announcement is, on average, a non-event."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* announcement CAR have "
            "to be for a $k$-event study to detect it at $t=2$? At $k=20$ you'd need a CAR several "
            "times the one observed; the real reaction lives far below the detection floor — and "
            "there is no day-one move and no certifiable drift on top."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = PANEL['car'].std(ddof=1); obs = PANEL['car'].mean()\n"
            "else:\n"
            "    sd = 0.045; obs = R['car_all'][0]/100\n"
            "ks = np.arange(5, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='CAR needed for t=2')\n"
            "ax.axhline(abs(obs)*100, c=GREEN, ls='--', label=f'|observed CAR| ~{abs(obs)*100:.2f}%')\n"
            "ax.axvline(R['n_priced'], c=GREY, ls=':', label=f\"our k={R['n_priced']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('3-day CAR (%)')\n"
            "ax.set_title('Detection floor vs the real reaction: badly under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_priced'])*100\n"
            "print(f'at k={R[\"n_priced\"]} you need ~{need:.1f}% CAR for t=2; observed ~{abs(obs)*100:.2f}% -> under-powered')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable CAR**; the green line "
            "is what we see. They don't meet until $k$ is many times larger than the relocation "
            "calendar will ever deliver. Even the literature's honest read (Desai-Hines: a small, "
            "conditional inversion effect) is *below* this detection floor for a blended 20-event "
            "table — which is precisely why the tape says non-event. There is no sizing, threshold, "
            "or cost assumption that manufactures an edge from a reaction indistinguishable from "
            "zero."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The cosmetic-change cousin.** [Study 389 — Name-Change-Effect]"
            "(../389-name-change-effect/): does a theme-chasing rebrand pop? Same family (a "
            "label/address, not a fundamental), same survivorship-and-selection pathology.\n"
            "- **The direct sibling.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "market-model CAR event study over a hardcoded, labelled table of corporate "
            "announcements.\n"
            "- **Condition on the tax delta.** Split by *how much* the effective tax rate actually "
            "fell (an inversion to Ireland vs a same-country state move), or reconstruct a "
            "survivorship-free relocation feed; Desai-Hines say a real inversion sub-effect is "
            "small and conditional, which is exactly why ~20 blended events can't isolate it.\n\n"
            "*The reproducible core is offline and deterministic; the relocation table is hardcoded "
            "and the tape is salience-selected & survivor-only (named). Methods and sources: "
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
