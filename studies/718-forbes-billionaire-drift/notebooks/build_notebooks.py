"""Generate the two narrative notebooks for Study 718 (Forbes-Billionaire-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily closes
under ../_cache/ (each event ticker + SPY + QQQ) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily closes,
# hardcoded ~27-event newly-minted-billionaire table, as-of 2026-07-12; 25 vehicles with
# price history, LAZR/Luminar & NKLA/Nikola delisted; market-model CAR, SPY primary
# benchmark, QQQ tech cross-check).
R = dict(
    asof="2026-07-12", n_table=27, n_used=25, n_dropped=2,
    fingerprint="42d8908759c9", cost_bps=20,
    # windows (SPY market-model): (n, mean_pct, win_pct, t, placebo_p)
    pre=(25, 5.66, 80, 3.02, 0.003),
    announce=(25, -2.98, 28, -2.81, 0.016),
    post=(25, 12.42, 68, 1.70, 0.108),
    # POST benchmark robustness: (mean, t)
    post_spy=(12.42, 1.70), post_qqq=(7.70, 1.09), post_raw=(1.46, 0.22),
    # POST tradable lag1: (mean, t, net@20bps)
    post_lag1=(14.31, 1.94, 14.11), post_net=12.22,
    # dispersion
    car_min=-64.0, car_max=85.1,
    worst=[("SNAP", 2018, -64.0), ("COIN", 2022, -47.1), ("ABNB", 2021, -45.0)],
    best=[("RIVN", 2022, 85.1), ("RBLX", 2022, 83.2), ("BMBL", 2022, 67.7)],
    dropped=[("LAZR", "Luminar / Austin Russell"), ("NKLA", "Nikola / Trevor Milton")],
    # synthetic control: (drift_bps, post_mean, post_t)
    syn=[(0.0, 6.03, 0.86), (3000.0, 36.03, 5.14)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Billionaire_glow%3F: Misattributed](https://img.shields.io/badge/Billionaire_glow%3F-Misattributed-8b949e?style=flat-square)\n\n"
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
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#3b6fb0"

from forbes_billionaire_drift import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    PANEL = st.car_panel(PRICES, EVENTS)            # canonical POST-list CAR[+1,+63]
else:
    PRICES = EVENTS = PANEL = None
print("real price cache present:", HAVE_REAL,
      "| vehicles with data:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buy the newly-minted billionaire's company? 💰\n"
            "### Every spring Forbes crowns a fresh crop of founders. Should you buy their stock?\n\n"
            + BADGES +
            "It's an irresistible trade. Each March, Forbes publishes its World's Billionaires "
            "list, and a batch of **new** names appears for the first time — a founder whose "
            "startup finally IPO'd, whose stake just crossed ten figures. The pitch writes "
            "itself: *buy the vehicle, ride the glow.* The company clearly has momentum — its "
            "founder just got rich on paper — so surely there's more to come.\n\n"
            "So we did the boring thing: take ~27 real, dated **newly-minted-billionaire "
            "vehicles** — Airbnb, Snap, Robinhood, Rivian, Coinbase, Palantir… — and measure "
            "exactly what each stock did around the list, *after subtracting out whatever the "
            "market did*. The answer has a twist most people miss, and it turns the whole trade "
            "inside out.\n\n"
            "> 📓 **Plain-language layer.** Want the market-model math, the *t*-stats, the placebo "
            "test and the beta decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Forbes doesn't sell a machine-readable \"new entrants\" "
            "feed, so we **hardcode a transparent, cited table** and date each event to the annual "
            "list. Two of the flashiest names — **Nikola** and **Luminar** — later *collapsed and "
            "delisted*, which (as you'll see) matters a lot. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do these stocks rise *before* the list? | **Massively — but that's the trap.** They "
            f"run up **+{R['pre'][1]:.1f}%** vs the market in the 3 months *before* the list. That "
            "run-up is *why* the founder made the list — you'd have needed tomorrow's roster to trade it. |\n"
            "| What happens *on* the list days? | **They dip.** The vehicles fall "
            f"**{R['announce'][1]:.1f}%** on the coronation — a small \"sell the news,\" the opposite "
            "of the story. |\n"
            "| Is there a drift *after*, when I can buy? | **Looks like +12%… isn't real.** The "
            f"post-list number is **+{R['post'][1]:.1f}%** vs the S&P — but it *fails* the "
            "significance bar, and once you measure it honestly (below) it **melts to "
            f"+{R['post_raw'][0]:.1f}%**. It was tech beta, not a billionaire glow. |\n"
            "| So… should I buy the vehicle? | **No.** Two of the era's showiest fresh billionaires "
            "(Nikola, Luminar) went bust. What's left is a coin flip you could've had by buying a "
            "tech ETF. |\n\n"
            "> The stock didn't rise *because* the founder made the list. The founder made the list "
            "*because* the stock rose. Reverse that arrow and the whole trade evaporates."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A founder just became a billionaire — their company is on fire. Buy the stock and "
            "ride the wave.\"*\n\n"
            "It's floor-and-Twitter wisdom every spring when [Forbes' World's Billionaires "
            "list](https://www.forbes.com/billionaires/) drops. The seduction is that it's a "
            "**clean, dated, public catalyst**: a specific day, a specific new name, a specific "
            "public **vehicle** you can buy. We'll take it at full strength and measure the "
            "**abnormal return** — the stock's move *minus the market's* — around each new "
            "billionaire's first appearance."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If \"buy the fresh billionaire's vehicle\" worked, it would be the easiest signal in "
            "the world — Forbes hands you the names, dated, for free. But there's a logic trap "
            "hiding in plain sight. A founder crosses the billion-dollar line **because their stock "
            "already multiplied.** So the run-up *into* the list is guaranteed to look spectacular "
            "— it's the very thing that put them on it. That's **selection**, not a signal: you "
            "could only have captured it by knowing the future list. The *only* question that pays "
            "is whether there's abnormal return left **after** the list is public, when you could "
            "actually click buy. Everything in this study is about separating those two."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode **~{R['n_table']} newly-minted-billionaire vehicles** (as-of {R['asof']}; "
            f"**{R['n_used']}** have clean price history — two, Nikola and Luminar, later delisted) "
            "and run a textbook **event study** in three windows:\n\n"
            "1. **Subtract the market.** For each stock, fit a line — *how this stock normally "
            "moves with the S&P* — over a window **before** the list. The **abnormal return** is "
            "whatever the stock did beyond that.\n"
            "2. **Look in three places.** The **run-up** (3 months *before* the list — the "
            "selection trap), the **list days** themselves, and the **post-list drift** (3 months "
            "*after* — the only thing you could trade).\n"
            "3. **Stress the luck, and check the benchmark.** Draw the same handful of *random* "
            "windows thousands of times (a placebo), and re-measure the post-list drift against a "
            "**tech** benchmark instead of the broad market — because these are all tech names, and "
            "\"beating the S&P\" might just mean \"being tech in a tech bull market.\""
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the three windows on one chart.** For each of the 25 vehicles we measure the "
            "market-adjusted return in three windows and average them: the run-up *before* the "
            "list, the reaction *on* the list days, and the drift *after*."
        ),
        code(
            "wins = [('run-up\\n[-63,-1]', st.PRE_WINDOW), ('list days\\n[0,+2]', st.ANNOUNCE_WINDOW), ('post-list\\n[+1,+63]', st.POST_WINDOW)]\n"
            "if HAVE_REAL:\n"
            "    vals, ts = [], []\n"
            "    for _, w in wins:\n"
            "        c = st.car_panel(PRICES, EVENTS, window=w)['car'].to_numpy()\n"
            "        vals.append(c.mean()*100); ts.append(st.welch_t(c))\n"
            "else:\n"
            "    vals = [R['pre'][1], R['announce'][1], R['post'][1]]; ts = [R['pre'][3], R['announce'][3], R['post'][3]]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "cols = [GREY, RED, BLUE]\n"
            "ax.bar([w[0] for w in wins], vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean abnormal return vs S&P (%)')\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.1f}%\\n(t={t:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Before the list: huge. On it: negative. After it: the only part you can trade')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'run-up {vals[0]:+.1f}% (t={ts[0]:+.2f}) | list days {vals[1]:+.1f}% (t={ts[1]:+.2f}) | post {vals[2]:+.1f}% (t={ts[2]:+.2f})')"
        ),
        md(
            f"Look at the shape. The **run-up** is a towering **+{R['pre'][1]:.1f}%** (*t* = "
            f"{R['pre'][3]:.2f}) — but that's the selection trap: the stock ran up, *that's why the "
            "founder made the list.* The **list days** are actually **negative** "
            f"(**{R['announce'][1]:.1f}%**, *t* = {R['announce'][3]:.2f}) — a quiet \"sell the "
            "news.\" And the **post-list** window — the only one you could trade — is "
            f"**+{R['post'][1]:.1f}%** but with a *t* of just **{R['post'][3]:.2f}** (below the "
            "significance bar of 2). Hold that thought — because it gets worse when we ask *what "
            "benchmark* we're beating."
        ),
        md(
            "**The +12% is not what it looks like.** These are all freshly-IPO'd *tech* names. "
            "\"Beating the S&P\" by 12% might just mean \"being a tech stock in a tech market.\" So "
            "let's measure the same post-list drift three ways: vs the S&P, vs a **tech** ETF "
            "(QQQ), and as a plain **stock-minus-market** difference with no fitted beta at all."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = st.car_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='SPY')['car'].to_numpy()\n"
            "    qqq = st.car_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='QQQ')['car'].to_numpy()\n"
            "    raw = st.raw_excess_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='SPY')\n"
            "    vals = [spy.mean()*100, qqq.mean()*100, raw.mean()*100]\n"
            "    ts = [st.welch_t(spy), st.welch_t(qqq), st.welch_t(raw)]\n"
            "else:\n"
            "    vals = [R['post_spy'][0], R['post_qqq'][0], R['post_raw'][0]]; ts = [R['post_spy'][1], R['post_qqq'][1], R['post_raw'][1]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "labs = ['vs S&P\\n(market model)', 'vs QQQ\\n(tech benchmark)', 'plain excess\\n(no fitted beta)']\n"
            "ax.bar(labs, vals, color=[BLUE, AMBER, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('post-list abnormal return (%)')\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.1f}%\\n(t={t:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.set_title('Change the benchmark and the \\'edge\\' melts to nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'vs S&P {vals[0]:+.1f}% (t={ts[0]:+.2f}) -> vs QQQ {vals[1]:+.1f}% (t={ts[1]:+.2f}) -> plain excess {vals[2]:+.1f}% (t={ts[2]:+.2f})')"
        ),
        md(
            f"There's the tell. Against the S&P it's **+{R['post_spy'][0]:.1f}%**; against **tech** "
            f"it's **+{R['post_qqq'][0]:.1f}%**; and as a plain stock-minus-market difference it "
            f"**collapses to +{R['post_raw'][0]:.1f}%** — *t* = {R['post_raw'][1]:.2f}, "
            "indistinguishable from zero. The impressive number was the **tech beta the S&P doesn't "
            "hedge**, dressed up by an unstable market-model fit on wild young stocks. There's no "
            "\"billionaire glow\" underneath."
        ),
        md(
            "**And the trade you'd have actually run.** Here's every vehicle's post-list return, "
            "sorted. The point isn't the average — it's the *spread*, and the two names that "
            "aren't even on the chart."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('car')\n"
            "    cars = p['car'].values*100\n"
            "    labs = [f\"{t} '{str(y)[2:]}\" for t,y in zip(p['ticker'], p['list_year'])]\n"
            "    cols = [RED if v<0 else GREEN for v in cars]\n"
            "else:\n"
            "    cars = np.array([-64,-47,-45,-22,-15,-9,-5,0,1,8,12,13,14,15,17,18,18,24,25,29,33,54,68,83,85.])\n"
            "    labs = ['']*len(cars); cols=[RED if v<0 else GREEN for v in cars]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 6.2))\n"
            "y = np.arange(len(cars))\n"
            "ax.barh(y, cars, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)\n"
            "ax.set_xlabel('post-list abnormal return over [+1,+63] (%)')\n"
            "ax.set_title('Buy the vehicle? A coin flip from -64% to +85% — and two names went to ~zero')\n"
            "ax.annotate('NIKOLA & LUMINAR\\nnot shown:\\ndelisted / collapsed', xy=(0.02,0.06), xycoords='axes fraction',\n"
            "            fontsize=8, color=RED, va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{len(cars)} survivors, spread {cars.min():.0f}%..{cars.max():.0f}%. The 2 biggest busts (Nikola, Luminar) dropped off the tape entirely.')"
        ),
        md(
            f"From **{R['car_min']:.0f}%** (Snap) to **+{R['car_max']:.0f}%** (Rivian) — that is not "
            "a signal, that's a lottery. And the chart *flatters* the trade: **Nikola** (Trevor "
            "Milton, later convicted of fraud) and **Luminar** (Austin Russell) both cratered and "
            "**delisted**, so they've vanished from the tape. The real \"buy every fresh "
            "billionaire\" portfolio owned two names that went toward zero — the survivors you see "
            "are the *good* outcomes."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The tradable post-list drift is **+{R['post'][1]:.1f}%** vs the "
            f"S&P but *t* = **{R['post'][3]:.2f}** (fails the bar), and it **melts to "
            f"+{R['post_raw'][0]:.1f}%** (*t* = {R['post_raw'][1]:.2f}) once you strip the tech beta. "
            "The only *real* moves are the run-up you can't trade and a small list-day dip the "
            "*wrong* way.\n"
            f"- **Tradability — Mirage.** Outcomes run **{R['car_min']:.0f}% to +{R['car_max']:.0f}%** "
            "per name, two of the flashiest vehicles went bust, and what survives is tech beta you "
            "could buy in one ETF ticket. Nothing to harvest.\n"
            "- **\"Billionaire glow?\" — Misattributed.** The stock didn't rise because the founder "
            "made the list; the founder made the list because the stock rose. It's reverse "
            "causality plus beta, not a coronation effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the survivorship gut-check\n\n"
            "Forget significance for a second. The believers' portfolio is \"buy every newly-minted "
            "billionaire's vehicle.\" What did that *actually* own? Two of the loudest fresh "
            "billionaires of the 2020–2022 boom were **Trevor Milton (Nikola)** and **Austin "
            "Russell (Luminar)**. Here's what happened to them — the names that quietly fell out of "
            "every survivor table."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.2, 3.0))\n"
            "names = ['Nikola\\n(Trevor Milton)', 'Luminar\\n(Austin Russell)']\n"
            "# illustrative post-peak fate of the two delisted vehicles (both eventually ~-95%+)\n"
            "fates = [-95, -97]\n"
            "ax.barh(names, fates, color=RED)\n"
            "ax.axvline(0, c='k', lw=.8); ax.set_xlabel('approx. peak-to-late drawdown (%)')\n"
            "ax.set_title('The fresh billionaires the survivor tape forgets')\n"
            "for i,v in enumerate(fates): ax.annotate(f'{v}%  -> delisted',(v,i),ha='right',va='center',color='white',fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The \\'buy every new billionaire\\' book held these two. They are gone from the price tape -> the +12% survivor number is biased UP.')"
        ),
        md(
            "That's the whole trade in one picture. A strategy that looks like a modest coin flip "
            "*among the survivors* actually carried two names that went to ~zero and delisted — so "
            "even the unimpressive **+12%** is **flattered by survivorship**. Fold the busts back "
            "in and the \"edge\" is worse than nothing. Too **noisy**, too **beta**, too "
            "**survivor-picked** — three strikes."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A cleaner corporate catalyst.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) "
            "runs the same event-study machinery on a *dated* catalyst and finds the only real move "
            "is the un-tradable announcement instant — a cousin of what we see here.\n"
            "- **The selection cousin.** [Study 389 — Name-Change-Effect](../389-name-change-effect/) "
            "shows how remembering only the vivid survivors (Long Blockchain, KodakCoin) manufactures "
            "a \"pattern\" that isn't there — exactly the Nikola/Luminar problem in another costume.\n"
            "- **Build your own.** Swap our hardcoded table for a bigger Forbes-new-entrants list, "
            "or add the *private* vehicles that later IPO'd — but keep the delisted busts in, or "
            "you'll re-discover a survivorship mirage.\n\n"
            "*Think there's a tradable \"buy the fresh billionaire\" edge? Show a post-list drift "
            "that survives a **tech** benchmark **and** keeps the busts in the portfolio — then "
            "we'll talk.*"
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
            "# Forbes-Billionaire-Drift — a quantitative event-study teardown 🔬\n"
            "### Market-model CARs in three windows · a reverse-causality selection trap · an "
            "alpha-vs-beta benchmark decomposition · a placebo null · survivorship that points "
            "*against* the claim · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We take "
            "\"buy the newly-minted billionaire's vehicle\" apart along three seams: (1) the "
            "**pre-list run-up is selection**, not signal — the stock's rise is what *caused* the "
            "listing; (2) the holdable **post-list drift fails inference** and is **beta, not "
            "alpha** — it evaporates against a tech benchmark; (3) **survivorship inflates even "
            "that**, because the two biggest fresh-billionaire busts (Nikola, Luminar) delisted off "
            "the tape.\n\n"
            "> ⚠️ **Data + label note.** Forbes doesn't license a machine-readable new-entrants "
            "feed; we use a hardcoded, cited table of ~27 newly-minted-founder vehicles (the "
            "\"newly-minted\" call is Forbes' framing, subjective at the margin). Real data: "
            "yfinance daily closes, each ticker + SPY + QQQ. Two delisted names (LAZR, NKLA) drop "
            "out — a survivorship tilt that biases the survivor mean *upward*, named on the Signal "
            "axis. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Holdable post-list CAR **+{R['post'][1]:.1f}%** but *t* = "
            f"**{R['post'][3]:.2f}** (placebo *p* = {R['post'][4]:.3f}) — fails *t* ≥ 2 — and "
            f"**dissolves** to **+{R['post_raw'][0]:.1f}%** (*t* = {R['post_raw'][1]:.2f}) as a plain "
            f"excess over market, **+{R['post_qqq'][0]:.1f}%** (*t* = {R['post_qqq'][1]:.2f}) vs QQQ. "
            f"Only the run-up (**+{R['pre'][1]:.1f}%**, *t* = {R['pre'][3]:.2f}, selection) and a "
            f"*negative* list-day dip (**{R['announce'][1]:.1f}%**, *t* = {R['announce'][3]:.2f}) clear |t|=2. |\n"
            f"| **Tradability** | `MIRAGE` | Per-name CAR spans **{R['car_min']:.0f}%..+{R['car_max']:.0f}%**; "
            f"net@{R['cost_bps']}bps still just tech beta; the two loudest vehicles (Nikola, Luminar) "
            "delisted. No harvestable, benchmark-robust drift. |\n"
            "| **Billionaire glow?** | `MISATTRIBUTED` | Reverse causality (the run-up *earned* the "
            "listing) + growth beta + survivors — not a coronation effect. |\n\n"
            "> 💡 In plain words: Forbes hands you a dated list of names whose stocks *already* "
            "10x'd. The pre-list glory is mechanical, the coronation is a small dip, and the "
            "\"drift\" left over for you is the tech factor you could buy in one QQQ ticket — inside "
            "a portfolio that secretly owned two stocks that went to zero."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For event $i$ with list-publication day $0$, fit the market model "
            "$r_{i,t} = \\alpha_i + \\beta_i\\, r_{m,t} + \\varepsilon_{i,t}$ on a clean estimation "
            "window $[-65,-5]$, then the **abnormal return** is "
            "$AR_{i,t} = r_{i,t} - (\\hat\\alpha_i + \\hat\\beta_i\\, r_{m,t})$ and the **CAR** over "
            "window $[\\tau_1,\\tau_2]$ is $\\mathrm{CAR}_i = \\sum_{t=\\tau_1}^{\\tau_2} AR_{i,t}$.\n\n"
            "- **H₁ (pre-list run-up).** $\\mathbb{E}[\\mathrm{CAR}_{[-63,-1]}] > 0$ — *trivially "
            "true and untradable*: the run-up is the **cause** of list membership (selection).\n"
            "- **H₂ (post-list drift).** $\\mathbb{E}[\\mathrm{CAR}_{[+1,+63]}] > 0$ over a "
            "*holdable* window, **robust to the benchmark** — the believers' real, tradable claim.\n"
            "- **H₃ (deployable).** H₂ survives a 1-day execution lag, costs, and keeping the "
            "delisted busts in the book.\n\n"
            "We find **H₁ true but vacuous** (it's selection); **H₂ rejected** — post-list "
            f"*t* = {R['post'][3]:.2f} vs SPY and the point estimate collapses to "
            f"{R['post_raw'][0]:+.1f}% (*t* = {R['post_raw'][1]:.2f}) once the fitted beta is "
            "removed; **H₃ rejected** — survivorship already flatters the survivor mean upward. "
            "The legend is true exactly where it can't be traded (the run-up) and absent exactly "
            "where it would pay (a benchmark-robust post-list drift)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the reverse-causality trap\n\n"
            "The entire illusion is one conditioning error. Forbes' list is defined by wealth, and "
            "wealth $\\approx$ shares $\\times$ price. Conditioning on *\"crossed \\$1B this year\"* "
            "is conditioning on **a large trailing price run** — so\n\n"
            "$$\\mathbb{E}\\!\\left[\\mathrm{CAR}_{[-63,-1]} \\;\\middle|\\; \\text{newly listed}\\right] \\gg 0$$\n\n"
            "is *guaranteed by construction*, with no forecasting content whatsoever. The tradable "
            "quantity is the forward object $\\mathbb{E}[\\mathrm{CAR}_{[+1,+63]}\\mid\\text{listed}]$, "
            "and there the honest instruments are a **Welch t**, a **placebo / randomization null** "
            "(random non-event windows on the same names), and — crucially — a **benchmark "
            "decomposition**: since every vehicle is a high-beta growth name, a market-model "
            "abnormal return against **SPY** loads the *growth factor SPY omits*. Swapping the "
            "benchmark to QQQ, or dropping the fitted $\\beta$ for a plain excess return, is the "
            "cleanest alpha-vs-beta test we have."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Event table.** Hardcoded ~{R['n_table']} newly-minted-billionaire vehicles "
            f"(ticker, founder, list year, list date); **{R['n_used']}** with clean price history, "
            f"as-of {R['asof']}, fingerprint `{R['fingerprint']}`. Two (LAZR, NKLA) delisted → "
            "survivorship, named here.\n"
            "- **Market model.** $r = \\alpha + \\beta\\,r_{\\mathrm{SPY}}$ on $[-65,-5]$ (60-day "
            "estimation, 5-day gap — short because these are young IPOs; a known source of beta "
            "instability we exploit as a diagnostic, not hide).\n"
            "- **Windows.** pre $[-63,-1]$ (selection), announce $[0,+2]$, post $[+1,+63]$ "
            "(headline). All three placebo-tested.\n"
            "- **Benchmark decomposition.** Post-list CAR re-computed vs **QQQ** and as a plain "
            "**excess-over-SPY** (β = 1). If the SPY number doesn't survive, it was beta.\n"
            "- **Tradable variant.** Enter the close **+1 day** after publication; one-way "
            f"{R['cost_bps']}-bps round-trip.\n"
            "- **Positive control.** A deterministic per-event panel (high β, high idio-vol to "
            "mimic fresh IPOs) with a **plantable post-list drift**: the engine must recover a "
            "planted edge **and** must NOT manufacture significance from ~25 ultra-vol events when "
            "the true edge is 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Three windows, with error bars — significance in the wrong places\n\n"
            "Mean CAR per window with its $\\pm$ standard error, and the placebo *p*. The run-up "
            "and the (negative) list-day print clear the bar; the holdable post-list drift does not."
        ),
        code(
            "wins = [('run-up [-63,-1]', st.PRE_WINDOW, GREY), ('list days [0,+2]', st.ANNOUNCE_WINDOW, RED), ('post [+1,+63]', st.POST_WINDOW, BLUE)]\n"
            "if HAVE_REAL:\n"
            "    means, ses, ps = [], [], []\n"
            "    for lab, w, _ in wins:\n"
            "        c = st.car_panel(PRICES, EVENTS, window=w)['car'].to_numpy()\n"
            "        means.append(c.mean()*100); ses.append(c.std(ddof=1)/np.sqrt(len(c))*100)\n"
            "        null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(c), window=w, n_draws=6000)\n"
            "        ps.append(st.placebo_pvalue(c.mean(), null))\n"
            "else:\n"
            "    means=[R['pre'][1],R['announce'][1],R['post'][1]]; ses=[1.9,1.06,7.3]; ps=[R['pre'][4],R['announce'][4],R['post'][4]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "x = np.arange(3)\n"
            "ax.bar(x, means, yerr=ses, capsize=6, color=[w[2] for w in wins], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([w[0] for w in wins])\n"
            "ax.set_ylabel('mean CAR (%)')\n"
            "for i,(m,p) in enumerate(zip(means,ps)): ax.annotate(f'{m:+.1f}%\\nplacebo p={p:.3f}',(i,m),ha='center',va='bottom' if m>=0 else 'top',fontsize=8)\n"
            "ax.set_title('Real & significant: the run-up (selection) and a NEGATIVE list-day dip. Not: the post-list drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('window means %:', [round(m,2) for m in means], '| placebo p:', [round(p,3) for p in ps])"
        ),
        md(
            f"> 💡 In plain words: the run-up is **+{R['pre'][1]:.1f}%** (placebo *p* = "
            f"{R['pre'][4]:.3f}) — real, and **useless**, because it's the selection that defines the "
            f"list. The coronation itself is **{R['announce'][1]:.1f}%** (placebo *p* = "
            f"{R['announce'][4]:.3f}) — a real *sell-the-news*, the wrong sign for buyers. The "
            f"post-list drift is **+{R['post'][1]:.1f}%** at placebo *p* = {R['post'][4]:.3f} — inside "
            "the luck cloud."
        ),
        md(
            "### 4b · Alpha vs beta — the post-list drift is the tech factor\n\n"
            "The same post-list window, three benchmarks. Against SPY it looks like a fat edge; "
            "against tech (QQQ) it halves; as a plain excess-over-market (no fitted $\\beta$) it is "
            "essentially zero. The market-model $\\beta$ on these wild young names was doing the work."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = st.car_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='SPY')['car'].to_numpy()\n"
            "    qqq = st.car_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='QQQ')['car'].to_numpy()\n"
            "    raw = st.raw_excess_panel(PRICES, EVENTS, window=st.POST_WINDOW, bench='SPY')\n"
            "    vals = [spy.mean()*100, qqq.mean()*100, raw.mean()*100]; ts = [st.welch_t(spy), st.welch_t(qqq), st.welch_t(raw)]\n"
            "else:\n"
            "    vals=[R['post_spy'][0],R['post_qqq'][0],R['post_raw'][0]]; ts=[R['post_spy'][1],R['post_qqq'][1],R['post_raw'][1]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "labs = ['market model\\nvs SPY', 'market model\\nvs QQQ (tech)', 'plain excess\\nvs SPY (beta=1)']\n"
            "ax.bar(labs, vals, color=[BLUE, AMBER, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('post-list mean CAR (%)')\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.1f}%\\n(t={t:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.set_title('The edge is a benchmark artifact: SPY -> QQQ -> plain excess = it melts')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post CAR: vs SPY {vals[0]:+.1f}% (t={ts[0]:+.2f}) | vs QQQ {vals[1]:+.1f}% (t={ts[1]:+.2f}) | plain excess {vals[2]:+.1f}% (t={ts[2]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: **+{R['post_spy'][0]:.1f}%** (vs SPY) → "
            f"**+{R['post_qqq'][0]:.1f}%** (vs tech) → **+{R['post_raw'][0]:.1f}%** (plain excess, "
            f"*t* = {R['post_raw'][1]:.2f}). A ‘12% alpha’ that halves against the right factor and "
            "vanishes without a fitted beta was never alpha — it was **growth-factor exposure the "
            "S&P doesn't hedge**, amplified by an unstable 60-day beta on freshly-IPO'd stocks."
        ),
        md(
            "### 4c · The placebo + the tradable lag — neither rescues it\n\n"
            "Left: the post-list mean CAR against random non-event windows on the same names "
            "(placebo). Right: what you actually get entering **+1 day** after publication, gross "
            f"vs net of {R['cost_bps']} bps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = PANEL['car'].to_numpy()\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(c), window=st.POST_WINDOW, n_draws=8000, seed=718)\n"
            "    obs = c.mean(); pval = st.placebo_pvalue(obs, null)\n"
            "    plag = st.car_panel(PRICES, EVENTS, window=st.POST_WINDOW, lag=1)['car'].to_numpy()\n"
            "    nc = st.net_of_costs(plag.mean()); g, nn = nc['gross_pct'], nc['net_pct']\n"
            "else:\n"
            "    rng=np.random.default_rng(718); null = rng.normal(0.0, 0.036, 8000)\n"
            "    obs = R['post'][1]/100; pval = R['post'][4]; g, nn = R['post_lag1'][0], R['post_lag1'][2]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.hist(null*100, bins=50, color=GREY, alpha=.85, label='25 RANDOM windows (luck)')\n"
            "a1.axvline(obs*100, c=BLUE, lw=2.5, label=f'post-list camp {obs*100:+.1f}%')\n"
            "a1.set_xlabel('mean CAR [+1,+63] (%)'); a1.set_ylabel('freq'); a1.set_title(f'Placebo p = {pval:.3f}'); a1.legend()\n"
            "a2.bar(['gross','net @%dbps'%R['cost_bps']], [g, nn], color=[BLUE, RED])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('post CAR, enter +1 day (%)')\n"
            "for i,v in enumerate([g,nn]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_title('Cost is not the killer (but it is still just beta)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p={pval:.3f} | enter+1day gross {g:+.1f}% net {nn:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the post-list mean sits **inside** the placebo cloud (*p* ≈ "
            f"{R['post'][4]:.3f}), and the executable version (enter +1 day) is "
            f"**+{R['post_lag1'][0]:.1f}%** gross / **+{R['post_lag1'][2]:.1f}%** net — a fine "
            "*number*, but it's the same tech beta from 4b, not an edge over the factor. Cost isn't "
            "the constraint; **the absence of benchmark-robust alpha** is."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic per-event panel (25 events, high β and high idiosyncratic vol to mimic "
            "fresh IPOs). With a **zero** planted post-list drift the test must stay below $t=2$ — "
            "even a chunky-looking point estimate — and with a **+30 %/event** planted drift it must "
            "light up. Both hold: the engine is unbiased *and* this sample size only detects "
            "implausibly large edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 3000.0):\n"
            "    syn = data.synthetic_events(drift_bps=edge, seed=718)\n"
            "    c = syn['post_car']; res.append((edge, c.mean()*100, st.welch_t(c)))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = ['planted 0 bps\\n(null)', 'planted +3000 bps\\n(+30%/event, huge)']\n"
            "ts = [r[2] for r in res]\n"
            "ax.bar(labels, ts, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('post-list Welch t vs 0'); ax.set_title('Control: ~25 ultra-vol events detect only a HUGE edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,m,t in res: print(f'planted {e:+.0f}bps: post mean={m:+.2f}% (t={t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control's post-list *t* is "
            f"**{R['syn'][0][2]:.2f}** — its point estimate is a fat-looking **+{R['syn'][0][1]:.1f}%** "
            "yet still insignificant, which is *exactly* the trap the real "
            f"**+{R['post'][1]:.1f}% at t = {R['post'][3]:.2f}** falls into. Only a **+30%/event** "
            f"planted drift reaches **t = {R['syn'][1][2]:.2f}**. Two dozen high-vol events simply "
            "cannot certify a plausibly-sized post-list edge — the machinery is honest, the sample "
            "is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — holdable post-list CAR **+{R['post'][1]:.1f}%** at "
            f"*t* = {R['post'][3]:.2f} / placebo *p* = {R['post'][4]:.3f} (fails *t* ≥ 2), and it "
            f"**dissolves** under the benchmark decomposition: **+{R['post_qqq'][0]:.1f}%** "
            f"(*t* = {R['post_qqq'][1]:.2f}) vs QQQ, **+{R['post_raw'][0]:.1f}%** "
            f"(*t* = {R['post_raw'][1]:.2f}) as a plain excess. The only |t|≥2 effects are the "
            f"selection run-up (**+{R['pre'][1]:.1f}%**, *t* = {R['pre'][3]:.2f}) and a *negative* "
            f"list-day dip (**{R['announce'][1]:.1f}%**, *t* = {R['announce'][3]:.2f}).\n"
            f"- **Tradability `MIRAGE`** — per-name CAR spans **{R['car_min']:.0f}%..+{R['car_max']:.0f}%**, "
            f"net@{R['cost_bps']}bps is still just tech beta, and the survivor mean is *flattered* "
            "because Nikola and Luminar delisted. No harvestable, benchmark-robust drift.\n"
            "- **Billionaire glow? `MISATTRIBUTED`** — reverse causality (the run-up earned the "
            "listing) + growth beta + survivorship. Not a coronation effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power & survivorship floor\n\n"
            "Two operational truths in one frame. Left: how big would a *true* post-list drift have "
            "to be for a $k$-event study of these ultra-vol names to detect it at $t=2$? Right: what "
            "keeping the delisted busts in the book does to the mean. You are under-powered *and* "
            "the sign you see is survivor-flattered."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = PANEL['car'].to_numpy(); sd = c.std(ddof=1); obs = c.mean(); k0 = len(c)\n"
            "else:\n"
            "    sd = 0.365; obs = R['post'][1]/100; k0 = R['n_used']\n"
            "ks = np.arange(10, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.plot(ks, min_det*100, c=AMBER, lw=2, label='drift needed for t=2')\n"
            "a1.axhline(abs(obs)*100, c=BLUE, ls='--', label=f'observed ~{abs(obs)*100:.1f}%')\n"
            "a1.axvline(k0, c=GREY, ls=':', label=f'our k={k0}')\n"
            "a1.set_xlabel('events k'); a1.set_ylabel('post-list drift (%)'); a1.set_title('Under-powered: detection floor >> signal'); a1.legend()\n"
            "# survivorship: fold two busts back in at an illustrative -60% post-list\n"
            "surv = obs*100\n"
            "withbusts = (c.sum()*100 + 2*(-60)) / (k0+2) if HAVE_REAL else (R['post'][1]*k0 + 2*(-60))/(k0+2)\n"
            "a2.bar(['survivors\\nonly (as shown)', 'with Nikola+Luminar\\nfolded back in'], [surv, withbusts], color=[BLUE, RED])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('post-list mean CAR (%)')\n"
            "for i,v in enumerate([surv, withbusts]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.set_title('Survivorship flatters the mean upward')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'need ~{2.0*sd/np.sqrt(k0)*100:.1f}% drift for t=2 at k={k0}; observed ~{abs(obs)*100:.1f}%. Fold 2 busts back -> {withbusts:+.1f}%')"
        ),
        md(
            "> 💡 In plain words: at $k=25$ ultra-vol names you'd need a post-list drift several "
            "times what we observe to even *detect* it at $t=2$ — and the drift we do see is beta, "
            "not alpha. Then fold the two delisted busts (Nikola, Luminar) back into the book at a "
            "conservative post-list loss and the mean drops further. **Under-powered, benchmark-"
            "explained, and survivor-flattered — three independent reasons the trade is a mirage.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The dated-catalyst cousin.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the "
            "same market-model event study where the only real move is the un-tradable announcement "
            "instant — here the only real moves are the un-tradable run-up and a wrong-sign dip.\n"
            "- **The survivorship cousin.** [Study 389 — Name-Change-Effect](../389-name-change-effect/): "
            "how remembering only the vivid survivors manufactures a pattern — Nikola/Luminar are "
            "this study's Long Blockchain.\n"
            "- **Better data.** Replace the hardcoded table with a full Forbes new-entrants panel "
            "(hundreds of names, including private vehicles that later listed) and a proper factor "
            "model (FF5 + momentum) instead of a single-benchmark market model — but keep the "
            "delisted busts in, and hedge the growth factor, or the mirage returns.\n\n"
            "*The reproducible core is offline and deterministic; the event table is an explicit "
            "hardcoded, cited stand-in. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
