"""Generate the two narrative notebooks for Study 648 (Grain Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CORN/WEAT/SOYB /
ZC=F/ZW=F/ZS=F tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance CORN/WEAT/SOYB adj
# close + ZC=F/ZW=F/ZS=F raw close, 2011-10-03 -> 2026-06-30, 176 monthly obs/series).
R = dict(
    start="2011-10-03", end="2026-06-30", n_months=176,
    bonf_crit=4.05, bonf_survive=0, bonf_n=36,
    # per-grain best/worst month: (month, mean%, welch t)
    best={"corn": (10, 1.37, 1.60), "wheat": (10, 0.96, 1.80), "soy": (10, 1.26, 1.05)},
    worst={"corn": (7, -2.45, -0.98), "wheat": (8, -3.21, -1.68), "soy": (7, -1.53, -1.13)},
    # nominal (uncorrected) top HAC t-stats, all negative / off-thesis
    top_hac={"wheat_nov": -2.78, "corn_nov": -2.45, "wheat_aug": -2.27},
    # per-grain spring vs harvest: spring_mean%, n_spring, harvest_mean%, n_harvest, spread%, t
    sh={
        "corn": (-2.29, 29, -0.38, 43, -1.92, -1.20),
        "wheat": (0.20, 45, -2.10, 29, 2.30, 1.13),
        "soy": (-1.03, 28, -0.02, 43, -1.01, -0.84),
    },
    # pooled spring vs harvest
    pooled_spring=-0.85, pooled_harvest=-0.68, pooled_spread=-0.17, pooled_t=-0.19,
    pooled_n_spring=102, pooled_n_harvest=115,
    ci_lo=-1.97, ci_hi=1.75, ci_boot=5000,
    # ETF vs futures-splice roll drag: etf bps/mo, fut bps/mo, drag bps/mo, t
    drag={
        "corn": (-53.8, -25.5, -28.3, -0.89),
        "wheat": (-94.8, -4.5, -90.3, -5.09),
        "soy": (5.3, -4.4, 9.8, 0.45),
    },
    # seasonal timer: buy&hold sharpe/cagr%, timer-gross sharpe/cagr%, timer-net sharpe/cagr%
    timer={
        "corn": (-0.35, -7.8, -0.25, -4.3, -0.28, -4.6),
        "wheat": (-0.49, -13.2, 0.27, 3.3, 0.25, 2.9),
        "soy": (0.04, -0.8, -0.18, -2.4, -0.22, -2.8),
        "pooled": (-0.34, -6.9, -0.03, -0.5, -0.08, -0.8),
    },
    # coin-flip hit rate on the timer's active legs, pooled
    hit_k=100, hit_n=216, hit_rate=46.3, hit_lo=39.8, hit_hi=53.0, hit_t=-0.09,
    # synthetic control
    syn_null_mean=0.36, syn_null_sd=1.02, syn_null_fire=1, syn_planted_t=5.64,
    fp_corn="81eafd725d20", fp_wheat="38378019b4f4", fp_soy="a1a9d64ef5f8",
    fp_zc="1d148da3b3a1", fp_zw="6b8a61301bbf", fp_zs="c488e2ab74e4",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
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

from grain_seasonality import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    ETF, FUT = data.load_real()
    MONTHLY = {g: st.monthly_log_returns(ETF[g]) for g in data.GRAINS}
    FUT_MONTHLY = {g: st.monthly_log_returns(FUT[g]) for g in data.GRAINS}
else:
    ETF = FUT = MONTHLY = FUT_MONTHLY = None
print("real cache present:", HAVE_REAL, "| grains:", data.GRAINS,
      "| months/grain:", (0 if MONTHLY is None else {g: len(MONTHLY[g]) for g in data.GRAINS}))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the grain market really have a spring high and a harvest low? 🌾\n"
            "### The old-crop/new-crop calendar — a century-old trading floor legend, tested on "
            "the three ETFs anyone can actually buy\n\n"
            + BADGES +
            "Grain traders have talked about this one forever: *plant the crop in spring, and "
            "nobody knows yet whether the weather will cooperate* — so prices carry a little "
            "insurance premium through planting, pollination and (for winter wheat) the spring "
            "green-up. Then the crop comes in. All at once, every elevator in the Midwest fills "
            "up, and the **harvest-time low** arrives on schedule, every single year.\n\n"
            "It's a great story — it even has a textbook mechanism (the \"theory of storage\": "
            "old-crop stocks get scarce right before harvest, then flush right after). So we "
            "tested it, on CORN, WEAT and SOYB — the three ETFs a retail account can actually "
            "hold — from 2011 to 2026.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni table and the "
            "roll-drag math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Each grain's \"spring\" and \"harvest\" windows come from USDA's "
            "own published crop-progress calendar — hardcoded facts, not a fit. Every chart is "
            "drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is spring really more expensive than harvest? | **Not on this tape.** Pooled "
            f"across all three grains, spring months average **{R['pooled_spread']:+.2f}%** "
            "*less* than harvest months (not the other way around), and that tiny gap could "
            "easily be zero (its uncertainty band runs from "
            f"{R['ci_lo']:+.2f}% to {R['ci_hi']:+.2f}%). |\n"
            "| What's the single best calendar month, then? | **October — for all three grains "
            "at once.** Which is *inside* the calendar's own harvest window, not the spring "
            "scare window the story predicts. |\n"
            "| Can you at least trade the *idea* — long spring, short harvest? | On paper, "
            "sort of — mostly by sitting in cash half the year on ETFs that have lost money on "
            "average. Test the *active* trades on their own and they hit **"
            f"{R['hit_rate']:.0f}%** of the time — a coin flip, and not even a lucky one. |\n"
            "| Is there a real cost hiding underneath all this? | **Yes, a big one.** WEAT alone "
            f"gives back **{-R['drag']['wheat'][2]:.0f} basis points every month** to its own "
            "roll mechanics — more than any seasonal in this study is worth. |\n\n"
            "> The legend is older than any of these ETFs. The ETFs don't know the legend."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Grain prices rally into planting and pollination — a spring 'weather market' "
            "where every drought scare or late frost gets priced in — and give it all back at "
            "harvest, when new supply floods the market. Buy the scare, sell the combine.\"*\n\n"
            "It's not just floor talk. Economists have documented a real seasonal cycle in "
            "storage costs and futures curves tied to the harvest calendar (the \"theory of "
            "storage\" literature). The mechanism is real. The question is whether it survives "
            "in the actual instrument a retail trader would use."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this pattern is real and tradable, it's about as clean a calendar trade as "
            "exists: three uncorrelated-ish crops, a mechanism with a hundred years of academic "
            "support, and three liquid ETFs anyone can buy with no futures account. If it's "
            "*not* real — or real but eaten alive by ETF roll costs — that's an equally useful "
            "answer: it tells you why \"everyone knows\" a seasonal that nobody can actually bank."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The calendar.** USDA's own crop-progress windows for corn, wheat and soybeans — "
            "planting, the weather-risk stretch, and harvest — hardcoded, not fitted.\n"
            "- **The comparison.** Every calendar month's average ETF return, for all three "
            "grains — 36 numbers — checked against a strict bar that accounts for testing 36 "
            "things at once (Bonferroni), so a single lucky month can't sneak through.\n"
            "- **The trade check.** Buy the spring window, short the harvest window, sit in cash "
            "otherwise — costs included — and separately ask: on the months the trade is "
            "actually *on*, does it win more than half the time?\n"
            "- **The roll check.** Compare each ETF's own return to a simple futures-splice "
            "spot proxy, to see how much the ETF's own mechanics already eat before any "
            "seasonal math even starts."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average return in the spring \"weather-scare\" window vs "
            "the harvest window, for each grain."
        ),
        code(
            "grains = list(data.GRAINS)\n"
            "if HAVE_REAL:\n"
            "    sh = {g: st.spring_harvest_tstat(MONTHLY[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g]) for g in grains}\n"
            "    spring_v = [sh[g]['spring_mean']*100 for g in grains]\n"
            "    harvest_v = [sh[g]['harvest_mean']*100 for g in grains]\n"
            "else:\n"
            "    spring_v = [R['sh'][g][0] for g in grains]\n"
            "    harvest_v = [R['sh'][g][2] for g in grains]\n"
            "x = np.arange(len(grains)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "ax.bar(x - w/2, spring_v, w, label='spring (weather-scare)', color=AMBER)\n"
            "ax.bar(x + w/2, harvest_v, w, label='harvest', color=GREY)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([g.upper() for g in grains])\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title('Spring is NOT reliably higher than harvest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({g: (round(spring_v[i],2), round(harvest_v[i],2)) for i,g in enumerate(grains)})"
        ),
        md(
            f"Two of three bars are the *wrong* way round: corn and soy actually average **less** "
            "in their spring window than in their harvest window — the opposite of the legend. "
            "Only wheat points the claimed direction, and it's a small, noisy gap. Pool all three "
            f"grains and the spread is **{R['pooled_spread']:+.2f}%/month** — practically nothing, "
            f"with an uncertainty band from **{R['ci_lo']:+.1f}%** to **{R['ci_hi']:+.1f}%** that "
            "comfortably contains zero (and the *opposite* sign).\n\n"
            "**Next, the full calendar.** Which single month is best, for each grain?"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "months = [R['best'][g][0] for g in grains]\n"
            "means = [R['best'][g][1] for g in grains]\n"
            "cols = [RED if m in data.HARVEST_MONTHS[g] else (AMBER if m in data.SPRING_MONTHS[g] else GREY)\n"
            "        for g, m in zip(grains, months)]\n"
            "ax.bar([g.upper() for g in grains], means, color=cols, width=.55)\n"
            "for i, (m, v) in enumerate(zip(months, means)):\n"
            "    ax.annotate(f'month {m}\\n{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('best single month, mean return (%)')\n"
            "ax.set_title('The best month for every grain is October — inside the HARVEST window')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "October — the tail end of the calendar's own harvest window — is the single best "
            "month for **all three grains**. That's the opposite of \"harvest is the low\": if "
            "anything, the calendar's low shows up mid-harvest (September) and *recovers* by its "
            "own final month. None of these individual months, best or worst, clears a strict "
            "bar that accounts for testing 36 different month/grain combinations at once — a "
            "single standout cell out of 36 draws is exactly what you'd expect from chance "
            "alone, and none of ours even reaches that lower bar.\n\n"
            "**Now, the trade.** Long spring, short harvest, cash otherwise — does it beat just "
            "holding the ETF?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled_bh = sum(MONTHLY[g].reindex(MONTHLY['corn'].index) for g in grains) / 3\n"
            "    pooled_timer = sum(st.seasonal_timer(MONTHLY[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g]) for g in grains) / 3\n"
            "    pooled_net = st.apply_costs(pooled_timer, n_trades_per_year=4, cost_bps_one_way=10.0)\n"
            "    bh, tg, tn = st.summary(pooled_bh)['sharpe'], st.summary(pooled_timer)['sharpe'], st.summary(pooled_net)['sharpe']\n"
            "else:\n"
            "    bh, tg, tn = R['timer']['pooled'][0], R['timer']['pooled'][2], R['timer']['pooled'][4]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['buy & hold', 'timer (gross)', 'timer (net)'], [bh, tg, tn], color=[GREY, AMBER, RED], width=.55)\n"
            "for i, v in enumerate([bh, tg, tn]): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Sharpe (equal-weight 3-grain basket)')\n"
            "ax.set_title('The timer looks better only because it is IN CASH half the year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {bh:+.2f}   timer gross {tg:+.2f}   timer net {tn:+.2f}')"
        ),
        md(
            "The timer's Sharpe does creep above buy-and-hold — but that's a magic trick, not a "
            "discovery: the timer sits in cash 5 months a year, and this basket of grain ETFs has "
            "*lost money on average since 2011*. Doing less of a losing trade looks better almost "
            "by definition. The honest test is to check **only the months the trade is actually "
            f"on**: those active months win **{R['hit_rate']:.0f}%** of the time, pooled across "
            "all three grains — a coin flip, and a slightly unlucky one. Even after costs, the "
            "net Sharpe is still negative.\n\n"
            "**Finally, the hidden cost.** Even if the calendar edge were real, can you actually "
            "collect it in the ETF, or does the ETF's own machinery eat it first?"
        ),
        code(
            "gs = grains\n"
            "if HAVE_REAL:\n"
            "    drags = [st.roll_drag(MONTHLY[g], FUT_MONTHLY[g])['drag_bps'] for g in gs]\n"
            "else:\n"
            "    drags = [R['drag'][g][2] for g in gs]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar([g.upper() for g in gs], drags, color=[RED if d < -30 else AMBER for d in drags], width=.55)\n"
            "for i, v in enumerate(drags): ax.annotate(f'{v:+.0f} bps/mo', (i, v), ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel(\"ETF return minus its own futures-splice proxy (bps/month)\")\n"
            "ax.set_title('WEAT quietly pays away ~1%/yr to its own roll, every year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({g: round(d,1) for g, d in zip(gs, drags)})"
        ),
        md(
            f"WEAT gives back **{-R['drag']['wheat'][2]:.0f} basis points a month** — about "
            "**1.1% a year** — to its own rolling mechanics, relative to a simple futures-price "
            "proxy, regardless of any calendar view you might have. That drag alone is bigger "
            "than every spring-vs-harvest spread this study measured. Whatever seasonal edge "
            "exists in the raw futures curve, a WEAT holder pays most of it away before the "
            "calendar even gets a vote."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No calendar month, in any of the three grains, clears the bar "
            "once you account for testing 36 things at once. Two of three grains point *against* "
            "the legend, and the best month for every grain sits inside the harvest window, not "
            "the spring window.\n"
            "- **Tradability — Mirage.** The seasonal timer's apparent improvement is a cash-drag "
            "artifact, not skill (its active months hit at coin-flip odds), and WEAT's own roll "
            "costs more than the entire theoretical edge is worth.\n"
            "- **\"Beats a coin?\" — Busted.** The trade's active legs win 46% of the time — worse "
            "than a fair coin, not better."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The mechanism might still be real in the futures curve** — this study tests the "
            "*ETF*, the thing you can actually buy, not the raw contract-to-contract basis. The "
            "next step would be measuring the calendar spread directly in the futures term "
            "structure, the way [639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) "
            "does for gasoline's *statutory* deadline.\n"
            "- **Weather years might matter more than calendar months.** A single bad drought "
            "(2012) can dominate 15 years of \"average June,\" and averaging it away is exactly "
            "what a clean seasonality test is supposed to do — which is one reason this one comes "
            "up empty.\n"
            "- **Sibling studies:** [307-coffee-seasonality](../../307-coffee-seasonality/) (a "
            "frost story, not a planting story) and [651-sugar-seasonality](../../651-sugar-seasonality/) "
            "(a different crush cycle) ask the same shape of question about different crops — "
            "worth comparing notes.\n\n"
            "*Think the calendar edge is real somewhere this study didn't look — the raw futures "
            "curve, a specific weather year, a different window? Show a net, certifiable edge on "
            "a tradable instrument, then we'll talk.*"
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
            "# Grain Seasonality — a quantitative teardown 🔬\n"
            "### A 36-cell Bonferroni month table · per-grain and pooled Welch/HAC splits · a "
            "block-bootstrap CI · the ETF-vs-futures roll-drag test · a costed calendar timer · "
            "a coin-flip hit-rate test · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **grain prices carry a spring weather-risk premium that decays into harvest "
            "— has a real mechanism (Working's theory of storage) and a century of trading-floor "
            "conviction behind it. The job here is to test it on the instruments a retail account "
            "can actually hold, with the multiple-testing correction the 3-grain × 12-month grid "
            "demands.\n\n"
            "> ⚠️ **Data note.** CORN/WEAT/SOYB daily adjusted closes + ZC=F/ZW=F/ZS=F daily raw "
            "closes, yfinance, 2011-10-03 → 2026-06-30 (176 monthly observations per series); "
            "**hardcoded USDA crop-progress windows** per grain (facts, no network). No "
            "survivorship (single continuously-listed tickers). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_corn"] + "` / `" +
            R["fp_wheat"] + "` / `" + R["fp_soy"] + "` / `" + R["fp_zc"] + "` / `" + R["fp_zw"] +
            "` / `" + R["fp_zs"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | pooled spring-vs-harvest spread **{R['pooled_spread']:+.2f}%/mo**, "
            f"Welch **t = {R['pooled_t']:+.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]**; "
            f"**{R['bonf_survive']}/{R['bonf_n']}** month×grain cells clear the Bonferroni bar "
            f"(\\|t\\| ≥ {R['bonf_crit']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | WEAT roll drag **{R['drag']['wheat'][2]:.1f} bps/mo** "
            f"(t = {R['drag']['wheat'][3]:.2f}); pooled timer net Sharpe **{R['timer']['pooled'][4]:+.2f}** |\n"
            f"| **Beats a coin?** | `BUSTED` | active-leg hit rate **{R['hit_k']}/{R['hit_n']} = "
            f"{R['hit_rate']:.1f}%** (Wilson [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), t = {R['hit_t']:.2f} |\n\n"
            "> 💡 In plain words: the storage-cycle mechanism is real economics; the ETF tape "
            "shows no certified trace of it, and what little the timer earns is a cash-drag "
            "illusion, not skill."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{g,t}$ be grain $g$'s monthly log return and $S_g, H_g \\subset \\{1..12\\}$ "
            "its own USDA-calendar spring (weather-scare) and harvest month sets. The claims:\n\n"
            "- **H₁ (level).** $E[r_{g,t} \\mid m_t \\in S_g] \\gg E[r_{g,t} \\mid m_t \\in H_g]$ "
            "for each grain, and pooled across grains.\n"
            "- **H₂ (concentration).** The seasonal is a genuine calendar effect, not one lucky "
            "month out of the 3×12 = 36 cells on the board.\n"
            "- **H₃ (tradability).** A calendar-known long-spring/short-harvest timer on the "
            "*ETF* earns net of costs and beats a coin on its active legs.\n"
            "- **H₄ (roll).** Whatever the raw futures curve shows, the ETF the retail account "
            "actually holds captures most of it (the roll doesn't eat the edge).\n\n"
            "We find **H₁ rejected** (pooled t = −0.19, wrong sign in 2/3 grains), **H₂ rejected** "
            "(0/36 Bonferroni survivors), **H₃ rejected** (46.3% active-leg hit rate, net Sharpe "
            "negative pooled) and **H₄ rejected** for wheat specifically (a certified, large roll "
            "drag)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Calendar months within a year are **not** independent draws (weather years are "
            "autocorrelated across adjacent months), so every per-month test reports both a "
            "naive one-sample *t* and a **Newey-West (HAC)** *t*. Testing 3 grains × 12 months at "
            "once is 36 simultaneous comparisons, so the honest bar is **Bonferroni**-corrected: "
            "$\\alpha/36 \\approx 0.0014$, which at $df \\approx 13$ needs $|t| \\ge 4.05$ — far "
            "stricter than the naive $|t| \\ge 2$. The spring-vs-harvest headline uses a **Welch** "
            "two-sample *t* (unequal variances, unequal group sizes by construction) plus a "
            "**circular block-bootstrap** (12-month blocks, respecting each grain's own annual "
            "seasonal structure) for a distribution-free CI on the pooled spread."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** CORN/WEAT/SOYB adjusted closes + ZC=F/ZW=F/ZS=F raw closes, "
            f"{R['start']} → {R['end']}, resampled to **{R['n_months']} monthly observations "
            "per series**. As-of 2026-06-30 (last complete month).\n"
            "- **Calendar.** Hardcoded USDA NASS crop-progress windows per grain "
            "(`GRAIN_CALENDAR` in `data.py`) — planting / weather-scare / harvest, facts not fit.\n"
            "- **Headline.** Per-month naive + HAC *t* (36 cells, Bonferroni-corrected); best/"
            "worst month Welch *t* vs rest, per grain; spring-vs-harvest Welch *t*, per grain and "
            "pooled, with a 5,000-draw block-bootstrap CI.\n"
            "- **Execution (calendar-known).** The seasonal timer's position is set from the "
            "fixed USDA windows alone — no signal-to-trade lag is needed, because the calendar "
            "repeats every year and is known at the start of it.\n"
            "- **Costs.** 4 one-way legs/yr × 10 bps × NAV, spread evenly across the 12 months.\n"
            "- **Roll cross-check.** ETF monthly return minus the roll-naive futures-splice "
            "monthly return, one-sample *t* of the gap — sizes what the ETF's own mechanics "
            "already take before any calendar math starts.\n"
            "- **Control.** i.i.d. synthetic monthly-return world, planted spring/harvest spread "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The 36-cell month table and the Bonferroni bar\n\n"
            "Per-grain, per-month naive + HAC *t*-stats. The Bonferroni-corrected bar for 36 "
            "simultaneous tests dwarfs the naive |t| ≥ 2 line."
        ),
        code(
            "grains = list(data.GRAINS)\n"
            "if HAVE_REAL:\n"
            "    crit = st.bonferroni_crit_t(36, df=13)\n"
            "    tabs = {g: st.month_stats(MONTHLY[g]) for g in grains}\n"
            "    hac = {g: tabs[g]['tstat_hac'].to_numpy() for g in grains}\n"
            "else:\n"
            "    crit = R['bonf_crit']\n"
            "    hac = {'corn': [0.97,-0.78,-0.12,1.02,-1.26,-1.47,-1.24,-0.82,-0.38,1.63,-2.45,0.68],\n"
            "           'wheat': [-0.68,-0.55,0.57,0.52,-0.36,-0.84,-1.09,-2.27,-0.25,1.08,-2.78,-0.64],\n"
            "           'soy': [0.33,1.22,0.29,1.08,-1.52,0.40,-1.10,-0.37,-0.94,1.71,0.07,0.26]}\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)\n"
            "for ax, g in zip(axes, grains):\n"
            "    t = hac[g]\n"
            "    cols = [RED if abs(v) >= crit else GREY for v in t]\n"
            "    ax.bar(range(1, 13), t, color=cols, width=.62)\n"
            "    ax.axhline(crit, ls='--', c=RED, lw=1); ax.axhline(-crit, ls='--', c=RED, lw=1)\n"
            "    ax.axhline(2, ls=':', c=AMBER, lw=1); ax.axhline(-2, ls=':', c=AMBER, lw=1)\n"
            "    ax.axhline(0, c='k', lw=.6)\n"
            "    ax.set_title(g.upper()); ax.set_xlabel('month')\n"
            "axes[0].set_ylabel('HAC t-stat (one-sample, vs 0)')\n"
            "fig.suptitle(f'36 cells, Bonferroni bar |t| >= {crit:.2f} (dashed red) vs naive |t| >= 2 (dotted amber)')\n"
            "plt.tight_layout(); plt.show()\n"
            "n_survive = sum(int(abs(v) >= crit) for g in grains for v in hac[g])\n"
            "print(f'cells clearing Bonferroni: {n_survive}/36')"
        ),
        md(
            f"**{R['bonf_survive']}/{R['bonf_n']}** cells clear the Bonferroni bar "
            f"(|t| ≥ {R['bonf_crit']:.2f}). The largest *nominal* HAC t-stats — WEAT-November "
            f"({R['top_hac']['wheat_nov']:.2f}), CORN-November ({R['top_hac']['corn_nov']:.2f}), "
            f"WEAT-August ({R['top_hac']['wheat_aug']:.2f}) — clear only the naive |t| ≥ 2 line, "
            "not the corrected one, and all three are *negative*: bearish months, not the "
            "claimed spring highs. This is exactly the shape a null grid of 36 tests should "
            "produce — a couple of nominal-looking cells by chance, none surviving correction."
        ),
        md(
            "### 4b · Best/worst month and spring-vs-harvest, per grain\n\n"
            "Welch *t* of the single best/worst month vs every other month pooled, and of the "
            "grain's own spring vs harvest window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bw = {g: st.best_worst_vs_rest(MONTHLY[g]) for g in grains}\n"
            "    sh = {g: st.spring_harvest_tstat(MONTHLY[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g]) for g in grains}\n"
            "    spreads = [sh[g]['spread']*100 for g in grains]\n"
            "    spread_ts = [sh[g]['t'] for g in grains]\n"
            "else:\n"
            "    spreads = [R['sh'][g][4] for g in grains]\n"
            "    spread_ts = [R['sh'][g][5] for g in grains]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in spread_ts]\n"
            "a1.bar([g.upper() for g in grains], spreads, color=cols, width=.55)\n"
            "for i, (v, t) in enumerate(zip(spreads, spread_ts)):\n"
            "    a1.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('spring - harvest spread (%)')\n"
            "a1.set_title('Two of three grains point the WRONG way')\n"
            "a2.bar([g.upper() for g in grains], spread_ts, color=cols, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.6)\n"
            "a2.set_ylabel('Welch t'); a2.set_title('None clear even the naive |t| >= 2 bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({g: (round(spreads[i],2), round(spread_ts[i],2)) for i, g in enumerate(grains)})"
        ),
        md(
            "> 💡 In plain words: corn's spread is −1.92% (t = −1.20) and soy's is −1.01% "
            "(t = −0.84) — both *against* the claim's sign — and wheat's, the only grain pointing "
            "the right way, is +2.30% at t = +1.13. None is close to significant even before any "
            "multiple-testing penalty."
        ),
        md(
            "### 4c · The pooled headline and its bootstrap CI\n\n"
            "Pool spring-tagged and harvest-tagged monthly returns across all three grains; a "
            "circular block-bootstrap (12-month blocks per grain, so each resample respects a "
            "grain's own annual seasonal structure) gives a distribution-free CI on the spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_spring_harvest(MONTHLY, data.SPRING_MONTHS, data.HARVEST_MONTHS)\n"
            "    ci = st.pooled_spread_bootstrap_ci(MONTHLY, data.SPRING_MONTHS, data.HARVEST_MONTHS)\n"
            "    spread, t, lo, hi = pooled['spread']*100, pooled['t'], ci['lo']*100, ci['hi']*100\n"
            "else:\n"
            "    spread, t, lo, hi = R['pooled_spread'], R['pooled_t'], R['ci_lo'], R['ci_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.barh(['pooled spread'], [spread], color=GREY, height=.4)\n"
            "ax.errorbar([spread], ['pooled spread'], xerr=[[spread-lo],[hi-spread]], fmt='none', ecolor=RED, capsize=8, lw=2)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('spring - harvest spread (%), 95% block-bootstrap CI')\n"
            "ax.set_title(f'Spread {spread:+.2f}% (t={t:+.2f}), CI [{lo:+.2f}%, {hi:+.2f}%] — straddles zero by a mile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pooled spread {spread:+.2f}%  t={t:+.2f}  CI [{lo:+.2f}%, {hi:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the pooled point estimate ({R['pooled_spread']:+.2f}%) is "
            "smaller than the width of its own confidence interval by an order of magnitude. "
            "H₁ is rejected across the board — not \"weak,\" genuinely absent on this tape."
        ),
        md(
            "### 4d · The ETF's own roll — the 'contango' caveat, sized\n\n"
            "Each ETF's monthly return minus the roll-naive futures-splice monthly return, "
            "one-sample *t* of the gap. A real cost, independent of any calendar view."
        ),
        code(
            "if HAVE_REAL:\n"
            "    drag = {g: st.roll_drag(MONTHLY[g], FUT_MONTHLY[g]) for g in grains}\n"
            "    d = [drag[g]['drag_bps'] for g in grains]; dt = [drag[g]['t'] for g in grains]\n"
            "else:\n"
            "    d = [R['drag'][g][2] for g in grains]; dt = [R['drag'][g][3] for g in grains]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in dt]\n"
            "ax.bar([g.upper() for g in grains], d, color=cols, width=.55)\n"
            "for i, (v, t) in enumerate(zip(d, dt)): ax.annotate(f'{v:+.1f} bps/mo\\n(t={t:+.2f})', (i, v),\n"
            "    ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('ETF minus futures-splice, mean monthly gap (bps)')\n"
            "ax.set_title('WEAT: a certified, large, unconditional roll drag')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({g: (round(x,1), round(y,2)) for g, x, y in zip(grains, d, dt)})"
        ),
        md(
            f"> 💡 In plain words: WEAT's drag ({R['drag']['wheat'][2]:.1f} bps/mo, "
            f"t = {R['drag']['wheat'][3]:.2f}) clears |t| ≥ 2 easily and is, in absolute size, "
            "bigger than every spring-vs-harvest spread this study measured — the roll is a "
            "*bigger and more certain* effect than the seasonal it would have to beat. Corn's "
            "drag is smaller and not certified; soy's is nominally positive — roll drag is a "
            "property of each grain's own curve shape, not a universal tax."
        ),
        md(
            "### 4e · The costed timer and the coin-flip test\n\n"
            "Long spring, short harvest, cash otherwise (no execution lag — the calendar repeats "
            "every year); 4 one-way legs/yr × 10 bps × NAV. The honest read isolates the "
            "*active* legs from the cash months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {}\n"
            "    for g in grains:\n"
            "        r = MONTHLY[g]\n"
            "        timer = st.seasonal_timer(r, data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])\n"
            "        net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10.0)\n"
            "        rows[g] = (st.summary(r)['sharpe'], st.summary(timer)['sharpe'], st.summary(net)['sharpe'])\n"
            "    pooled_bh = sum(MONTHLY[g].reindex(MONTHLY['corn'].index) for g in grains) / 3\n"
            "    pooled_timer = sum(st.seasonal_timer(MONTHLY[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g]) for g in grains) / 3\n"
            "    pooled_net = st.apply_costs(pooled_timer, n_trades_per_year=4, cost_bps_one_way=10.0)\n"
            "    rows['pooled'] = (st.summary(pooled_bh)['sharpe'], st.summary(pooled_timer)['sharpe'], st.summary(pooled_net)['sharpe'])\n"
            "else:\n"
            "    rows = {g: (R['timer'][g][0], R['timer'][g][2], R['timer'][g][4]) for g in list(grains)+['pooled']}\n"
            "labels = [g.upper() for g in grains] + ['POOLED']\n"
            "bh = [rows[g][0] for g in list(grains)+['pooled']]\n"
            "tn = [rows[g][2] for g in list(grains)+['pooled']]\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x - w/2, bh, w, label='buy & hold', color=GREY)\n"
            "ax.bar(x + w/2, tn, w, label='timer (net)', color=AMBER)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Sharpe'); ax.set_title('Timer vs buy-and-hold, net of costs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(rows)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    active = []\n"
            "    for g in grains:\n"
            "        timer = st.seasonal_timer(MONTHLY[g], data.SPRING_MONTHS[g], data.HARVEST_MONTHS[g])\n"
            "        active.append(timer[timer != 0.0].dropna())\n"
            "    allx = pd.concat(active).to_numpy()\n"
            "    k, n = int((allx > 0).sum()), len(allx)\n"
            "    rate = k / n\n"
            "    t = st._one_sample_t(allx)\n"
            "else:\n"
            "    k, n, rate, t = R['hit_k'], R['hit_n'], R['hit_rate']/100, R['hit_t']\n"
            "z = 1.96; p = rate; z2 = z*z\n"
            "mid = (p + z2/(2*n)) / (1 + z2/n)\n"
            "half = z * np.sqrt(p*(1-p)/n + z2/(4*n*n)) / (1 + z2/n)\n"
            "lo, hi = mid - half, mid + half\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['active-leg hit rate'], [rate*100], color=RED if rate < 0.5 else AMBER, width=.4)\n"
            "ax.errorbar([0], [rate*100], yerr=[[  (rate-lo)*100],[(hi-rate)*100]], fmt='none', ecolor='k', capsize=8)\n"
            "ax.axhline(50, ls='--', c='k', lw=1.2, label='coin flip')\n"
            "ax.set_ylabel('% of active months with a positive return')\n"
            "ax.set_title(f'{k}/{n} = {rate*100:.1f}% (Wilson [{lo*100:.1f}%, {hi*100:.1f}%]) — a coin, at best')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'hit rate {k}/{n} = {rate*100:.1f}%  Wilson [{lo*100:.1f}%, {hi*100:.1f}%]  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the Sharpe bar chart flatters the timer (it's mostly just "
            "cash), but the coin-flip test strips that out — on the months the trade is actually "
            f"live, it wins **{R['hit_rate']:.1f}%** of the time (Wilson "
            f"[{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), a point estimate *below* 50% with an "
            f"interval straddling it. Mean per active month is statistically zero "
            f"(t = {R['hit_t']:.2f}). **H₃ rejected.**"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic i.i.d. monthly-return world with a TUNABLE planted spring-premium/"
            "harvest-discount pair. The null is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    df = data.synthetic_world(seasonal=0.0, seed=648 + s_)\n"
            "    null_ts.append(st.synthetic_detect(df, (6, 7), (9, 10, 11))['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "df = data.synthetic_world(seasonal=0.06, seed=648)\n"
            "planted_t = st.synthetic_detect(df, (6, 7), (9, 10, 11))['t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (seasonal=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted seasonal = +6%/yr')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (spring vs harvest)')\n"
            "ax.set_title('Control: the null rarely fires; a planted spread lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires at roughly the "
            f"nominal false-positive rate ({R['syn_null_fire']}/20 ≈ 5%, mean t = "
            f"{R['syn_null_mean']:+.2f}, sd {R['syn_null_sd']:.2f}) and a planted 6%/yr spread "
            f"reads t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape "
            "t ≈ −0.2 is the genuine article, not a broken detector. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled spring-vs-harvest spread "
            f"**{R['pooled_spread']:+.2f}%/month**, Welch t = **{R['pooled_t']:+.2f}**, "
            f"bootstrap CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]**; "
            f"**{R['bonf_survive']}/{R['bonf_n']}** grid cells clear Bonferroni; corn and soy "
            "point the wrong direction; the best month for every grain is October, inside the "
            "harvest window.\n"
            f"- **Tradability `MIRAGE`** — WEAT's roll drag is **{R['drag']['wheat'][2]:.1f} bps/"
            f"month at t = {R['drag']['wheat'][3]:.2f}**, larger and more certain than any "
            f"spring/harvest spread measured; the timer's pooled net Sharpe is "
            f"**{R['timer']['pooled'][4]:+.2f}**, and its apparent gross improvement is a cash-"
            "drag artifact of an already-losing basket.\n"
            f"- **\"Beats a coin?\" `BUSTED`** — active-leg hit rate "
            f"**{R['hit_k']}/{R['hit_n']} = {R['hit_rate']:.1f}%** (Wilson "
            f"[{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), mean per active month t = "
            f"**{R['hit_t']:.2f}**. Not *wrong* about the mechanism existing — wrong that it's "
            "still there, tradable, in these three ETFs, today."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The mechanism could still live in the raw futures basis.** This study tests the "
            "*ETF*, not the contract-to-contract calendar spread directly. A term-structure study "
            "(front vs deferred, calendar-spread level) is the natural sequel, the way "
            "[639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) separates the "
            "curve from the holder.\n"
            "- **Weather-year clustering could be masking a conditional effect** — a seasonal "
            "that only shows up in *drought* years (2012) is a different, testable claim; this "
            "study deliberately averages across all weather regimes, which is why an "
            "unconditional calendar test comes up empty even if a conditional one wouldn't.\n"
            "- **Dedup map:** [307-coffee-seasonality](../../307-coffee-seasonality/) (frost, not "
            "planting), [308-cocoa-squeeze](../../308-cocoa-squeeze/) (a squeeze, not a "
            "calendar), [309-oj-frost](../../309-oj-frost/) (tail risk), "
            "[226-crude-seasonality](../../226-crude-seasonality/) / "
            "[227-natgas-winter](../../227-natgas-winter/) (energy, no storage-cycle mechanism), "
            "[639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) (a *statutory* "
            "deadline — `REAL`, unlike this study), [651-sugar-seasonality](../../651-sugar-seasonality/) "
            "(a different crush cycle).\n\n"
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
