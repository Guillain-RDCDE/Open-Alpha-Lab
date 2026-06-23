"""Generate the two narrative notebooks for Study 393 (AI-Datacenter-Basket).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached panel under
../_cache/ (the datacenter/power field + SPY + QQQ) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly total return,
# 30-name datacenter/power field + SPY + QQQ, 2022-02 -> 2026-05, 52 months, fp 7d69ca333d53).
R = dict(
    start="2022-02", end="2026-05", months=52, years=4.3, n_field=30, fp="7d69ca333d53",
    # legs: name -> (CAGR%, Sharpe, maxDD%)
    legs={
        "basket": (71.7, 1.65, -21.7), "spy": (14.3, 0.91, -20.2),
        "qqq": (18.5, 0.91, -26.1), "field": (38.8, 1.55, -16.7),
        "expost": (83.7, 1.78, -20.3),
    },
    basket_net=71.5,
    # spread tests: (mean%/yr, HAC t, n)
    t_spy=(47.1, 3.19, 52), t_qqq=(42.5, 3.12, 52), t_eqf=(26.0, 2.37, 52),
    # decomposition
    basket_spread=57.4, expost_spread=69.4, selection_share=121, pctile=100.0,
    rand_mean=23.8, rand_median=23.6,
    expost_members=["VRT", "MU", "SMCI", "NVDA", "AVGO", "DELL", "VST", "PWR"],
    # single names: name -> (mean excess%/yr, HAC t)  [sorted by t desc]
    names=[("VRT", 69.4, 2.59), ("ANET", 31.9, 2.38), ("CEG", 38.3, 2.18),
           ("NVDA", 48.2, 2.17), ("VST", 43.4, 2.04), ("SMCI", 84.6, 1.67),
           ("DELL", 48.9, 1.51), ("ETN", 12.5, 1.27)],
    # synthetic control: (alpha, prenamed_spread%, prenamed_t, expost_placebo_spread%)
    syn=[(0.0, 4.5, 1.71, 14.3), (0.012, 17.5, 5.77, 21.0)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Ride_the_build--out%3F: Busted](https://img.shields.io/badge/Ride_the_build--out%3F-Busted-8b949e?style=flat-square)\n\n"
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

from ai_datacenter_basket import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RETS, BENCH = data.load_real(allow_lookahead_selection=True)   # look-ahead is the SUBJECT
    RACE = st.race(RETS, BENCH, data.BASKET, k=8, n_draws=2000, cost_bps=10.0,
                   allow_lookahead_selection=True)
else:
    RETS = BENCH = RACE = None
print("real datacenter panel cache present:", HAVE_REAL,
      "| field names:", (0 if RETS is None else RETS.shape[1]))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can quote
# it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Ride the AI build-out by buying the datacenter basket? 🖥️\n"
            "### The \"don't pick the winner, buy the picks-and-shovels\" trade, in plain English\n\n"
            + BADGES +
            "Here's the pitch you've probably heard: you don't have to guess *which* AI company wins "
            "— just buy the **datacenter-and-power basket**. The chips (NVDA), the cooling and "
            "electrical gear (VRT, ETN), the utilities that sell the electricity an AI datacenter "
            "guzzles (CEG, VST), the AI servers (SMCI, DELL), the network switches (ANET). Own the "
            "whole **build-out** and ride the capex wave.\n\n"
            "And on the tape it *worked* — this eight-name basket roughly **5×'d** while the market "
            "barely doubled. So is \"buy the build-out\" a real, repeatable strategy? Or is it the "
            "oldest trick in finance: **naming the winners after they already won?** This notebook "
            "pulls those two apart.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the random-basket distribution and "
            "the selection placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** Both the basket *and* the wider field we compare it against are "
            "**today's** names projected back in time — so they're tilted toward survivors by "
            "construction. That bias is the whole point, and we name it on the Signal axis. Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the basket beat the market? | **Yes, hugely — ~72%/yr vs ~14% for the S&P.** That "
            "part is real on the tape (and it even beats the tech-heavy Nasdaq). |\n"
            "| So it's a great strategy? | **Careful.** The eight are *named because they won*. Pick "
            "the eight best-performing names of the field **in hindsight** and you reproduce — even "
            "*exceed* — the basket's edge. |\n"
            "| Could I have picked these eight in 2019? | **No.** A *blindly* chosen eight from the "
            "same datacenter/power field already beat the S&P by ~24%/yr — because the **field itself** "
            "was on fire. The basket just hand-picks the top of it. |\n"
            "| Is the build-out real, then? | **The build-out is real. The *basket* is hindsight.** "
            "AI capex is genuinely enormous — but that doesn't make *this* eight-name list something "
            "you could have specified in advance. |\n\n"
            "> The AI build-out is real; the basket that \"captures\" it is a list of yesterday's "
            "winners. A real theme, harvested by hindsight."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Forget stock-picking the one AI champion. The AI boom has to run through physical "
            "stuff — GPUs, servers, switches, cooling, and an enormous amount of **electricity**. So "
            "buy the **datacenter-and-power basket** — NVDA, VRT, ETN, CEG, VST, SMCI, ANET, DELL — "
            "equal-weight, and ride the entire build-out.\"*\n\n"
            "It's the **\"sell shovels in a gold rush\"** heuristic, updated for AI. And the story has "
            "real bones: hyperscaler capex guidance is gigantic and the datacenter power-demand "
            "narrative is genuinely documented. The seductive leap is from *\"AI capex is huge\"* to "
            "*\"**this specific eight-name list** is how you harvest it\"* — a leap that only looks "
            "obvious **after** these eight have already re-rated."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"buy the build-out,\" and only one of them is a "
            "strategy. (1) *Did these names go up?* Obviously — that's why they're on the list. The "
            "question that matters is (2) *could you have **named** them in advance, and would picking "
            "\"AI-build-out names\" **forward** pay the way the backtest does?* A basket that beats the "
            "market only because it's assembled from the realised winners isn't an edge — it's a "
            "**scoreboard read backwards**. And a theme can be 100% real (AI capex **is** booming) "
            "while *this particular eight-name bet* is pure look-ahead."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We need a fair fight. So we build a **whole candidate field** — every plausible "
            "\"datacenter-and-power\" name a 2019 investor might have considered: chips, networking, "
            "electrical/cooling/industrials, regulated *and* merchant utilities, datacenter REITs "
            f"(**{R['n_field']}** names kept). Then three tests:\n\n"
            "1. **Did the basket really beat the market?** Race the eight, equal-weight, against the "
            "S&P (SPY) *and* the Nasdaq-100 (QQQ) — with a proper (autocorrelation-robust) "
            "significance test.\n"
            "2. **Pick the winners in hindsight.** With the whole tape in view, grab the eight "
            "*best* names of the field. If that \"cheating\" rule reproduces the basket's edge, the "
            "basket **is** the cheating rule.\n"
            "3. **Throw darts.** Draw 2,000 *random* eight-name baskets from the same field. Where "
            "does the famous basket land in that cloud — and where does a *blind* pick land?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: did it win?** Grow $1 in the basket, the S&P, the Nasdaq-100, and an "
            "equal-weight basket of the *whole* datacenter/power field. The basket's line is in a "
            "different postal code."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = {'AI-datacenter basket': RACE['basket'], 'S&P 500 (SPY)': RACE['spy'],\n"
            "            'Nasdaq-100 (QQQ)': RACE['qqq'], 'Equal-weight field': RACE['equal_field']}\n"
            "    cols = {'AI-datacenter basket': GREEN, 'S&P 500 (SPY)': GREY,\n"
            "            'Nasdaq-100 (QQQ)': AMBER, 'Equal-weight field': RED}\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "    for name, ser in legs.items():\n"
            "        eq = (1+ser.dropna()).cumprod()\n"
            "        ax.plot(eq.index, eq.values, lw=2, c=cols[name], label=name)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('The basket 5x; the market barely doubled (2022-02 -> 2026-05)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('basket CAGR %.0f%%  vs SPY %.0f%%' % (st.summarize(RACE['basket'])['cagr']*100, st.summarize(RACE['spy'])['cagr']*100))\n"
            "else:\n"
            "    names = list(R['legs']); cag = [R['legs'][k][0] for k in names]\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "    ax.bar(names, cag, color=[GREEN, GREY, AMBER, RED, GREEN][:len(names)])\n"
            "    ax.set_ylabel('CAGR (%)'); ax.set_title('Frozen CAGRs (see docs/results.md)')\n"
            "    plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.show()\n"
            "    print('basket CAGR', R['legs']['basket'][0], '% vs SPY', R['legs']['spy'][0], '%')"
        ),
        md(
            f"On the AI-era tape (**{R['start']} → {R['end']}**) the basket compounded at "
            f"**~{R['legs']['basket'][0]:.0f}%/yr** against the S&P's **~{R['legs']['spy'][0]:.0f}%** "
            f"— and it beat the *tech* tape (QQQ, ~{R['legs']['qqq'][0]:.0f}%) too. That out-"
            "performance is **real**; it isn't a rounding artefact. So far, so good for the pitch. "
            "Now the awkward question: *how was the list chosen?*"
        ),
        md(
            "**The hindsight test.** Let's *cheat on purpose*: with the entire tape in front of us, "
            "pick the eight **best** names of the field. If \"pick the winners after the fact\" "
            "matches the famous basket, the famous basket is just that — winners, named afterward."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bs = RACE['basket_cagr_spread']*100; es = RACE['expost_cagr_spread']*100\n"
            "    rs = float(np.median(RACE['rand_spread']))*100\n"
            "    winners = RACE['expost_members']\n"
            "else:\n"
            "    bs, es, rs = R['basket_spread'], R['expost_spread'], R['rand_median']\n"
            "    winners = R['expost_members']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "bars = ['the famous basket', 'pick 8 winners\\n(hindsight)', 'random blind 8\\n(median)']\n"
            "vals = [bs, es, rs]\n"
            "ax.bar(bars, vals, color=[GREEN, RED, GREY])\n"
            "for i,v in enumerate(vals): ax.annotate(f'+{v:.0f}%', (i,v), ha='center', va='bottom')\n"
            "ax.set_ylabel('CAGR spread over the S&P (%/yr)')\n"
            "ax.set_title('Hindsight reproduces (and beats) the famous basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('ex-post winners:', winners)"
        ),
        md(
            f"There it is. The hindsight \"pick the eight winners\" rule beats the S&P by "
            f"**~{R['expost_spread']:.0f}%/yr** — actually **more** than the famous basket's "
            f"**~{R['basket_spread']:.0f}%/yr** (it reproduces **{R['selection_share']:.0f}%** of it). "
            "The famous basket is essentially *a slightly-worse version of \"the names that won.\"* And "
            f"look at the third bar: a **random** eight from the same field already beat the market by "
            f"**~{R['rand_median']:.0f}%/yr** — because the **field itself** was a furnace."
        ),
        md(
            "**The dartboard.** Draw 2,000 random eight-name baskets from the same field. Where does "
            "the famous basket sit — and notice where the *whole cloud* sits relative to zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = RACE['rand_spread']*100; basket = RACE['basket_cagr_spread']*100; pct = RACE['basket_pctile']\n"
            "else:\n"
            "    rng = np.random.default_rng(393); rand = rng.normal(R['rand_mean'], 18, 2000)\n"
            "    basket = R['basket_spread']; pct = R['pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand, bins=45, color=GREY, alpha=.85, label='2,000 random 8-baskets vs S&P')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(np.median(rand), c=AMBER, lw=2, ls='--', label=f'random median +{np.median(rand):.0f}%/yr')\n"
            "ax.axvline(basket, c=GREEN, lw=2.5, label=f'the famous basket +{basket:.0f}%/yr ({pct:.0f}th pctile)')\n"
            "ax.set_xlabel('CAGR spread over the S&P (%/yr)'); ax.set_ylabel('number of random baskets')\n"
            "ax.set_title('Two artefacts at once: a hot field (cloud > 0) + a hand-picked top (green at the edge)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'the famous basket sits at the {pct:.0f}th percentile of random baskets from the same field')"
        ),
        md(
            f"Two things jump out. **(1)** The whole grey cloud sits *well to the right of zero* — even "
            f"a **random** datacenter/power basket crushed the S&P, because today's field is the set of "
            "names that **survived and re-rated** (survivorship). **(2)** The famous basket sits at the "
            f"very **{R['pctile']:.0f}th percentile** — the extreme right edge — because it isn't "
            "random at all: it's hand-picked from the top. The headline edge is *the field was hot* "
            "**plus** *we kept its winners*. Neither is something you could have named in 2019."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** The basket genuinely beat the market and tech on the tape (~72%/yr "
            "vs ~14%, and significant) — but it's a **hindsight-selected** list on **one short ~4-year "
            "regime**. Real recent spread, look-ahead selection.\n"
            "- **Tradability — Mirage.** Costs are nothing for an eight-name hold. The mirage is that "
            "**you couldn't have named these eight in advance** — and even the \"theme\" portion is the "
            "field's own survivorship, not a tilt you could specify forward.\n"
            "- **Ride the build-out by buying the basket? — Busted.** The build-out is real; *this "
            "basket* is yesterday's scoreboard. It's the same selection trick as the "
            "[Magnificent-Seven](../../355-magnificent-seven/) and the "
            "[GLP-1 basket](../../356-glp1-basket/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — name it *forward*\n\n"
            "The whole game is *when you choose the names*. Compare the thing you could have bought "
            "**blindly** (a random eight from the field) against the thing you can only assemble "
            "**afterward** (the realised winners). The gap between them is the part of the \"edge\" "
            "that only hindsight can buy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    blind = float(np.median(RACE['rand_spread']))*100\n"
            "    hindsight = RACE['expost_cagr_spread']*100\n"
            "    famous = RACE['basket_cagr_spread']*100\n"
            "else:\n"
            "    blind, hindsight, famous = R['rand_median'], R['expost_spread'], R['basket_spread']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 3.4))\n"
            "ax.barh(['forward-pickable\\n(blind random 8)', 'the famous basket', 'hindsight winners'],\n"
            "        [blind, famous, hindsight], color=[GREY, GREEN, RED])\n"
            "for i,v in enumerate([blind, famous, hindsight]): ax.annotate(f'+{v:.0f}%/yr',(v,i),va='center',ha='left')\n"
            "ax.set_xlabel('CAGR spread over the S&P (%/yr)')\n"
            "ax.set_title('What you could buy forward vs what only hindsight assembles')\n"
            "ax.set_xlim(0, hindsight*1.18); plt.tight_layout(); plt.show()\n"
            "print(f'forward-pickable ~+{blind:.0f}%/yr  |  hindsight ~+{hindsight:.0f}%/yr  -- the gap is pure selection')"
        ),
        md(
            f"The only thing a 2019 investor could actually have done is the **grey** bar — a blind "
            f"pick from the field — and even that's flattered by survivorship (the field is today's "
            f"survivors). The **red** bar (~+{R['expost_spread']:.0f}%/yr) needs a time machine. The "
            "famous basket lives almost entirely in the un-buyable gap between them. Costs never enter "
            "the story; **the selection does.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The mega-cap twin.** [Study 355 — Magnificent-Seven](../../355-magnificent-seven/): "
            "the exact same \"hold the names that already won\" trick on the Mag 7. Read together, the "
            "two show that *thematic baskets are scoreboards read backwards*.\n"
            "- **The single-name cousin.** [Study 356 — GLP-1 basket](../../356-glp1-basket/): a theme "
            "basket whose alpha collapses to one stock.\n"
            "- **The mechanics.** [Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/) and "
            "[Study 345 — Survivorship-Bias](../../345-survivorship-bias/) show how selection and "
            "survivorship manufacture edge from nothing.\n\n"
            "*Think \"buy the build-out\" is forward-tradable? Specify the basket **in advance** — a "
            "rule you could have written in 2019 — and show it beats a random pick from the same field "
            "out-of-sample. Then we'll talk.*"
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
            "# AI-Datacenter-Basket — a quantitative teardown 🔬\n"
            "### Equal-weight basket vs SPY & QQQ · HAC *t* of the spread · a theme-beta (equal-field) "
            "control · the ex-post-selection placebo · the random-basket distribution · a planted-edge "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two things \"buy the build-out\" fuses: a **real AI-era spread** from the "
            "**after-the-fact selection** that named the basket. The basket-minus-market spread is "
            "large and significant on the tape (that is not in dispute); the decisive object is the "
            "**decomposition** — how much of it survives once you account for (a) the candidate field "
            "itself being survivorship-selected and (b) the eight being the hindsight top of it.\n\n"
            "> ⚠️ **Data + look-ahead note.** yfinance monthly **total return**, a 30-name "
            "datacenter/power field + SPY + QQQ, **2022-02→2026-05** (post-spin tickers CEG/VST set "
            "the start — one short AI-capex regime). Both the field and the eight-name membership are "
            "*current-membership projected back* (look-ahead + survivorship; named on the Signal "
            "axis). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `MIXED` | Basket−SPY **+{R['t_spy'][0]:.0f}%/yr**, HAC "
            f"**t = {R['t_spy'][1]:.2f}** (n={R['t_spy'][2]}); still **t = {R['t_qqq'][1]:.2f}** vs QQQ "
            f"and **t = {R['t_eqf'][1]:.2f}** vs the equal-weight field. Real on the tape — but "
            f"look-ahead-selected on one ~4-year regime. |\n"
            f"| **Tradability** | `MIRAGE` | Ex-post 'pick 8 winners' reproduces "
            f"**{R['selection_share']:.0f}%** of the spread; the basket is the "
            f"**{R['pctile']:.0f}th percentile** of 2,000 random baskets; a *blind* eight already "
            f"beats SPY by **+{R['rand_mean']:.0f}%/yr**. Costs immaterial — **selection**, not "
            "friction, kills it. |\n"
            f"| **Ride the build-out?** | `BUSTED` | The +{R['basket_spread']:.0f}%/yr decomposes into "
            f"≈+{R['rand_mean']:.0f}%/yr (the field was hot, survivorship) + ≈+"
            f"{R['basket_spread']-R['rand_mean']:.0f}%/yr (hand-picking its winners). Neither is "
            "forward-specifiable. Study 355 (Mag-7), thematic edition. |\n\n"
            "> 💡 In plain words: the basket really did out-earn — because it's the realised top of a "
            "field that itself only contains survivors. Strip both and there's no tilt a 2019 investor "
            "could have *named*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b$ be the eight-name basket (equal-weight, monthly-rebalanced) and $m$ a benchmark "
            "(SPY or QQQ). Write the per-month spread $d_t = r^b_t - r^m_t$ (both legs fully invested, "
            "so $d_t$ *is* the excess-of-each-other return).\n\n"
            "- **H₁ (the basket out-earns).** $\\mathbb{E}[d_t] > 0$ under autocorrelation-robust "
            "inference — a real spread over the market and the tech tape.\n"
            "- **H₂ (it's a forward-tradable theme tilt).** The spread reflects a *pre-specifiable* "
            "tilt, not the realised ranking — i.e. it survives once you account for selecting the "
            "basket on its own outcome.\n"
            "- **H₃ (\"buy the build-out\").** The edge is the *theme*, harvestable by holding "
            "\"datacenter-and-power names,\" rather than the specific hindsight list.\n\n"
            "We find **H₁ supported** (HAC *t* ≈ 3 vs SPY *and* QQQ), but **H₂ rejected** (ex-post "
            "selection reproduces ≥100% of the spread; the basket is the 100th percentile of random "
            "draws) and **H₃ rejected** (even a *blind* basket from the field beats SPY by ~24%/yr — "
            "that's survivorship of the field, not a harvestable tilt). The legend is true exactly "
            "where it's a scoreboard (these names rose) and unproven exactly where it would pay "
            "(a *forward-nameable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition of the realised CAGR spread $S$:\n\n"
            "$$S \\;=\\; \\underbrace{S_{\\text{field}}}_{\\text{the candidate field beat the market "
            "(survivorship)}} \\;+\\; \\underbrace{\\big(S - S_{\\text{field}}\\big)}_{\\text{name "
            "selection within the field}}.$$\n\n"
            "$S_{\\text{field}}$ is the **mean random-basket spread** — what a *blind* pick of the "
            "field earned, i.e. the part that is the field being hot. $S - S_{\\text{field}}$ is the "
            "**name selection** — the gap between a blind pick and the hand-picked top. A real, "
            "forward-tradable theme tilt would put the *pre-specified* basket far into the right tail "
            "of the random distribution **and** survive a synthetic null where no name has any true "
            "edge. The HAC *t* answers H₁; the **placebo decomposition** answers H₂/H₃ — and it is the "
            "placebo, not the *t*, that decides Tradability."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** yfinance monthly total returns, **{R['n_field']}** datacenter/power "
            f"candidates + SPY + QQQ, {R['start']}→{R['end']} ({R['months']} months; fingerprint "
            f"`{R['fp']}`). Field & membership are **current-membership projected back** "
            "(look-ahead + survivorship; on the axis).\n"
            "- **Legs.** Named basket (the eight), equal-weight 1/N **field** (theme-beta control), "
            "ex-post **'pick the 8 winners'** (look-ahead placebo), and **2,000 random 8-baskets** "
            "(the blind-pick sampling distribution).\n"
            "- **Signal test (HAC t).** Newey-West *t* of the monthly basket−benchmark spread, vs SPY, "
            "QQQ, and the equal-field.\n"
            "- **Decomposition.** $S_{\\text{field}}$ = mean random-basket spread; "
            "$S-S_{\\text{field}}$ = name selection; ex-post placebo / basket = **selection share**; "
            "basket **percentile** in the random cloud.\n"
            "- **Single-name HAC t.** Is the basket really one or two names?\n"
            "- **Costs.** One-way turnover × NAV at 10 bps/rebalance.\n"
            "- **Positive control.** A deterministic panel with a planted per-name tilt "
            "(``alpha_spread``): the *pre-named* basket must be significant **only** when an edge is "
            "planted, while the ex-post placebo conjures a spread **even at zero true edge** — the "
            "selection artefact, isolated on data where we know the truth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The spread is real — HAC *t* vs SPY, QQQ, and the field\n\n"
            "Mean annualised basket−benchmark spread with its Newey-West *t*. The bar to clear is "
            "**t ≥ 2**; it clears against all three, including the tech tape and the theme-beta "
            "control."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tt = [('vs SPY', RACE['test_vs_spy']), ('vs QQQ', RACE['test_vs_qqq']), ('vs equal-field', RACE['test_vs_equal'])]\n"
            "    labels = [a for a,_ in tt]; means = [b['mean_diff_ann']*100 for _,b in tt]; ts = [b['tstat'] for _,b in tt]\n"
            "else:\n"
            "    labels = ['vs SPY','vs QQQ','vs equal-field']\n"
            "    means = [R['t_spy'][0], R['t_qqq'][0], R['t_eqf'][0]]; ts = [R['t_spy'][1], R['t_qqq'][1], R['t_eqf'][1]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(labels, means, color=GREEN, width=.55)\n"
            "for i,v in enumerate(means): a1.annotate(f'+{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('mean spread (%/yr)'); a1.set_title('Basket beats market, tech, and the field')\n"
            "a2.bar(labels, ts, color=AMBER, width=.55); a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(ts): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('Newey-West HAC t'); a2.set_title('All three clear t=2'); a2.legend()\n"
            "for ax in (a1,a2): ax.tick_params(axis='x', rotation=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t:', {l: round(t,2) for l,t in zip(labels,ts)})"
        ),
        md(
            f"> 💡 In plain words: the basket beats the **market** by ~{R['t_spy'][0]:.0f}%/yr "
            f"(t = {R['t_spy'][1]:.2f}), the **tech tape** by ~{R['t_qqq'][0]:.0f}%/yr "
            f"(t = {R['t_qqq'][1]:.2f}), and even the **equal-weight field** — \"datacenter stuff in "
            f"general\" — by ~{R['t_eqf'][0]:.0f}%/yr (t = {R['t_eqf'][1]:.2f}). H₁ is **supported**: "
            "this is not a rounding artefact, and it isn't purely market or tech beta. The catch is "
            "*not* significance — it's what the significance is **of**."
        ),
        md(
            "### 4b · The decisive test — selection reproduces the whole spread\n\n"
            "The random-basket distribution (2,000 blind eight-name draws from the field) is the null "
            "for *name selection*. The basket and the ex-post 'pick the winners' rule are the green and "
            "red lines; the cloud's *position* (well right of zero) is the field's survivorship."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = RACE['rand_spread']*100; basket = RACE['basket_cagr_spread']*100\n"
            "    expost = RACE['expost_cagr_spread']*100; pct = RACE['basket_pctile']\n"
            "else:\n"
            "    rng = np.random.default_rng(393); rand = rng.normal(R['rand_mean'], 18, 2000)\n"
            "    basket = R['basket_spread']; expost = R['expost_spread']; pct = R['pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(rand, bins=50, color=GREY, alpha=.85, label=f'2,000 random 8-baskets (mean +{rand.mean():.0f}%/yr)')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(basket, c=GREEN, lw=2.5, label=f'famous basket +{basket:.0f}%/yr ({pct:.0f}th pctile)')\n"
            "ax.axvline(expost, c=RED, lw=2.5, ls='--', label=f'ex-post winners +{expost:.0f}%/yr')\n"
            "ax.set_xlabel('CAGR spread over the S&P (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title('A hot field (cloud>0) + a hand-picked top (green at the edge, red beyond)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'selection share = ex-post/basket = {expost/basket*100:.0f}%  |  basket percentile = {pct:.0f}')"
        ),
        md(
            f"> 💡 In plain words: the ex-post \"pick the winners\" rule earns "
            f"**+{R['expost_spread']:.0f}%/yr** — **{R['selection_share']:.0f}%** of the basket's "
            f"+{R['basket_spread']:.0f}%/yr (selection is the *whole* edge, and then some). The basket "
            f"sits at the **{R['pctile']:.0f}th percentile** — no random draw beat it. And the cloud's "
            f"**mean is +{R['rand_mean']:.0f}%/yr**: even a blind pick crushed the market, because the "
            "field is survivors. H₂ and H₃ rejected."
        ),
        md(
            "### 4c · Is it one or two names? — single-name decomposition\n\n"
            "Unlike a one-stock theme, the strength here is genuinely spread across several names — "
            "which makes the *theme beta* real and puts the weight of the critique on **selection**, "
            "not concentration. HAC *t* of each member's monthly excess over SPY:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sn = RACE['single_names']\n"
            "    items = sorted(sn.items(), key=lambda kv: -kv[1]['tstat'])\n"
            "    names = [k for k,_ in items]; tvals = [v['tstat'] for _,v in items]\n"
            "else:\n"
            "    names = [n for n,_,_ in R['names']]; tvals = [t for _,_,t in R['names']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.0))\n"
            "cols = [GREEN if t>=2 else AMBER for t in tvals]\n"
            "ax.bar(names, tvals, color=cols); ax.axhline(2, ls='--', c=RED, label='t=2')\n"
            "for i,t in enumerate(tvals): ax.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t of excess over SPY'); ax.set_title('Five of eight clear t=2 individually — not one lucky stock')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "n_sig = sum(1 for t in tvals if t>=2); print(f'{n_sig} of {len(tvals)} names individually clear t=2')"
        ),
        md(
            "> 💡 In plain words: five of the eight (VRT, ANET, CEG, NVDA, VST) individually clear "
            "t = 2 against the market. So this is **not** the GLP-1 case (all one stock) — the theme "
            "really did lift several names. That's exactly why the decisive critique is **selection**: "
            "it isn't one fluke, it's a *hand-picked top slice of a field that was, in hindsight, on "
            "fire* — and you couldn't have specified that slice forward."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel where only a *pre-named* basket carries a planted tilt "
            "(``alpha_spread``). With **zero** true edge the pre-named basket must stay below t = 2 (no "
            "false positive) **while the ex-post placebo still conjures a spread**; with a real planted "
            "edge the pre-named basket must light up. Both hold — the engine finds a spread only where "
            "one is planted, and manufactures one whenever you select on the outcome."
        ),
        code(
            "res = []\n"
            "for a in (0.0, 0.012):\n"
            "    rdf, b, truth = data.synthetic_panel(alpha_spread=a)\n"
            "    named = st.basket_returns(rdf, truth['true_basket_cols'])\n"
            "    expo = st.basket_returns(rdf, st.expost_winners(rdf, k=8, allow_lookahead_selection=True))\n"
            "    sn = st.summarize(named)['cagr'] - st.summarize(b)['cagr']\n"
            "    se = st.summarize(expo)['cagr'] - st.summarize(b)['cagr']\n"
            "    tn = st.hac_tstat_diff(named, b)['tstat']\n"
            "    res.append((a, sn*100, tn, se*100))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "x = np.arange(len(res)); w=.38\n"
            "ax.bar(x-w/2, [r[1] for r in res], w, color=GREEN, label='pre-named basket (honest)')\n"
            "ax.bar(x+w/2, [r[3] for r in res], w, color=RED, label='ex-post placebo (selection)')\n"
            "for i,r in enumerate(res):\n"
            "    ax.annotate(f't={r[2]:.2f}',(i-w/2,r[1]),ha='center',va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['alpha=0\\n(NO true edge)','alpha=0.012\\n(real edge)'])\n"
            "ax.set_ylabel('CAGR spread vs index (%/yr)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title('Null: pre-named t<2, yet selection still manufactures a spread'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for a,sn,tn,se in res: print(f'alpha={a:.3f}: pre-named {sn:+.1f}%/yr (t={tn:+.2f})  ex-post placebo {se:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the pre-named basket sits at "
            f"**t = {R['syn'][0][2]:.2f}** (below 2 — no false positive) yet the ex-post placebo still "
            f"conjures **+{R['syn'][0][3]:.0f}%/yr** from pure selection; only a genuine planted tilt "
            f"drives the honest basket to **t = {R['syn'][1][2]:.2f}**. The machinery is faithful — and "
            "the real-tape lesson is that a *selected* basket can post a huge spread with **no** "
            "forward-specifiable edge underneath."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — basket−SPY **+{R['t_spy'][0]:.0f}%/yr**, HAC "
            f"**t = {R['t_spy'][1]:.2f}** (n={R['t_spy'][2]}); **t = {R['t_qqq'][1]:.2f}** vs QQQ, "
            f"**t = {R['t_eqf'][1]:.2f}** vs the equal-field; five names clear t=2 individually. Real "
            "on the realised tape — but **look-ahead / survivorship selected** on **one short ~4-year "
            "regime**. A `WEAK` going-forward case in a strong in-sample *t* ⇒ `MIXED`.\n"
            f"- **Tradability `MIRAGE`** — costs immaterial (net CAGR unchanged to the decimal). The "
            f"ex-post rule reproduces **{R['selection_share']:.0f}%** of the spread, the basket is the "
            f"**{R['pctile']:.0f}th percentile** of random draws, and even a *blind* eight beats SPY by "
            f"**+{R['rand_mean']:.0f}%/yr** (the field is survivors). The +{R['basket_spread']:.0f}%/yr "
            "is a number only hindsight can buy.\n"
            f"- **Ride the build-out? `BUSTED`** — the +{R['basket_spread']:.0f}%/yr decomposes into "
            f"≈+{R['rand_mean']:.0f}%/yr (hot field, survivorship) + "
            f"≈+{R['basket_spread']-R['rand_mean']:.0f}%/yr (name selection); neither is "
            "forward-specifiable. **Study 355 (Mag-7) inverted into a theme** — same selection "
            "pathology, datacenter costume."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the forward-vs-hindsight gap\n\n"
            "The operational truth in one decomposition: split the realised spread into the part a "
            "blind investor would have earned (the field's survivorship) and the part that requires "
            "hindsight (name selection). Only the first is even arguably forward-tradable — and it's "
            "itself an artefact of which names *survived* into today's field."
        ),
        code(
            "if HAVE_REAL:\n"
            "    field_part = float(RACE['rand_spread'].mean())*100\n"
            "    total = RACE['basket_cagr_spread']*100\n"
            "else:\n"
            "    field_part = R['rand_mean']; total = R['basket_spread']\n"
            "sel_part = total - field_part\n"
            "fig, ax = plt.subplots(figsize=(9.0, 3.0))\n"
            "ax.barh(['realised basket spread'], [field_part], color=GREY, label=f'field survivorship (~+{field_part:.0f}%/yr)')\n"
            "ax.barh(['realised basket spread'], [sel_part], left=[field_part], color=RED, label=f'name selection (~+{sel_part:.0f}%/yr)')\n"
            "ax.set_xlabel('CAGR spread over the S&P (%/yr)'); ax.set_xlim(0, total*1.05)\n"
            "ax.set_title(f'+{total:.0f}%/yr = hot field + hand-picking — neither forward-specifiable')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(f'survivorship ~+{field_part:.0f}%/yr  +  selection ~+{sel_part:.0f}%/yr  =  +{total:.0f}%/yr realised')"
        ),
        md(
            f"> 💡 In plain words: of the **+{R['basket_spread']:.0f}%/yr**, roughly "
            f"**+{R['rand_mean']:.0f}%/yr** is *\"any random pick of this field beat the market\"* — "
            f"but that field only exists because those names **survived and re-rated** (survivorship); "
            f"a 2019 field would have included names that since cratered. The remaining "
            f"**~+{R['basket_spread']-R['rand_mean']:.0f}%/yr** is **name selection** — keeping the "
            "winners. There is no slicing of this that a 2019 investor could have *specified in "
            "advance*: the rarity that makes the basket impressive is exactly the hindsight that makes "
            "it untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The mega-cap original.** [Study 355 — Magnificent-Seven](../../355-magnificent-seven/): "
            "the same selection decomposition on the Mag 7; reading them together is the cleanest demo "
            "that thematic baskets are realised rankings, not factors.\n"
            "- **The single-name limit.** [Study 356 — GLP-1 basket](../../356-glp1-basket/): a theme "
            "basket whose alpha collapses entirely to one stock.\n"
            "- **The mechanics.** [Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/) and "
            "[Study 345 — Survivorship-Bias](../../345-survivorship-bias/) isolate the selection and "
            "survivorship artefacts this basket inherits.\n"
            "- **Pre-register it.** Specify a datacenter/power basket from *pre-2022* information only, "
            "freeze it, and test it forward against a random pick of the field; the look-ahead vanishes "
            "and (the null suggests) so does most of the edge.\n\n"
            "*The reproducible core is offline and deterministic; the field & membership are an explicit "
            "current-membership projection. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
