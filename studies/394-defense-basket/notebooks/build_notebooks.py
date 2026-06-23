"""Generate the two narrative notebooks for Study 394 (Defense-Basket).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached defense prices
under ../_cache/ (the basket + SPY) and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance defense basket
# LMT/RTX/NOC/GD + ITA + SPY, 1995-01-04 -> 2026-06-18, 7,917 days, 31.5 years, 20 shocks).
R = dict(
    start="1995-01-04", end="2026-06-18", days=7917, years=31.5, n_events=20,
    # per-window: (window, n, cond_mean%, cond_win%, base_mean%, base_win%, t, t_hac, p_placebo)
    w5=(5, 20, -0.14, 45, 0.08, 52, -0.19, -0.19, 0.661),
    w21=(21, 20, -1.32, 50, 0.33, 54, -1.22, -1.13, 0.938),
    w63=(63, 20, -2.25, 30, 1.06, 55, -1.33, -1.20, 0.961),
    net=[(5, -0.14, -0.24), (21, -1.32, -1.42), (63, -2.25, -2.35)],
    robust=[(5, -0.14, -0.19, 0.661), (10, -0.67, -0.77, 0.878), (21, -1.32, -1.22, 0.938),
            (42, -1.58, -1.01, 0.927), (63, -2.25, -1.33, 0.961)],
    ita=(14, -0.87, 43, 0.19, -1.32, 0.862),       # n, cond%, win%, base%, t, p
    syn=[(0.0, 20, 0.84, -0.53, 0.61, 0.151), (0.06, 20, 6.76, -0.16, 4.63, 0.000)],
    # per-event 21-day excess (date, excess%) — the wide spread that averages to noise
    events=[("1998-08-20", 2.54), ("1998-12-16", -12.92), ("1999-03-24", 0.14),
            ("2001-09-17", 4.33), ("2001-10-08", -8.52), ("2003-03-20", -1.64),
            ("2008-08-08", 3.11), ("2011-03-21", -3.98), ("2014-03-03", -1.49),
            ("2014-09-23", 1.14), ("2015-09-30", 1.06), ("2017-04-07", 2.60),
            ("2018-04-13", -8.03), ("2019-09-16", -3.17), ("2020-01-03", -1.91),
            ("2021-08-16", -2.59), ("2022-02-24", 6.24), ("2023-10-09", 3.04),
            ("2024-04-15", 0.44), ("2024-10-01", -6.87)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reliable_rally%3F: Busted](https://img.shields.io/badge/Reliable_rally%3F-Busted-8b949e?style=flat-square)\n\n"
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

from defense_basket import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
EVENTS = data.shock_dates()
print("real defense cache present:", HAVE_REAL, "| shock dates:", len(EVENTS))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Buy defense when war breaks out\" — does it actually work? 🛡️\n"
            "### The reflex trade everyone reaches for on a war headline, in plain English\n\n"
            + BADGES +
            "Every time conflict hits the tape — an invasion, a missile barrage, a new rearmament "
            "push — the same reflex fires across financial media: **buy defense stocks.** Lockheed, "
            "Raytheon (RTX), Northrop, General Dynamics — \"they always rally on war news.\"\n\n"
            "It *sounds* airtight: war → bigger defense budgets → more orders → higher prices. But "
            "\"they always rally\" is a claim you can actually **check** — and when you line up the "
            "shocks honestly, the famous rallies turn out to be matched, draw for draw, by forgotten "
            "double-digit *losers*. This notebook shows why the reflex is a great story and a losing "
            "trade at the same time.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **What \"defense\" means here.** We use a transparent **4-name equal-weight basket** "
            "(LMT, RTX, NOC, GD) and cross-check the **ITA** sector ETF, measuring each against the "
            "broad market (SPY). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a war shock, does defense beat the market? | **No — it slightly *lags*.** Over "
            f"the next month the basket returns about **{R['w21'][2]:.1f}%** *less* than SPY, on "
            f"average across **{R['n_events']} shocks**. |\n"
            "| But Ukraine 2022! Post-9/11! | **Real — and cherry-picked.** Those popped (+6%, +4%). "
            "But Desert Fox (**−13%**), the Afghanistan invasion (−9%) and the 2024 Iran barrage "
            "(−7%) fell just as hard. You only remember the winners. |\n"
            "| Is the lag a real, bettable signal? | **No.** With only "
            f"**{R['n_events']} events ever** the average is statistically indistinguishable from a "
            f"random handful of dates — a coin beats it ~**{R['w21'][8]*100:.0f}%** of the time. |\n"
            "| So why does \"buy defense on war\" feel so true? | **Vivid memories + an up-drifting "
            "market.** A couple of unforgettable rallies get generalised into a law. It's the same "
            "trick as every \"buy the headline\" reflex. |\n\n"
            "> Defense earnings really do rise with budgets over *years*. The fast, reliable "
            "*same-month stock rally on the headline* is the myth — and in this sample it points the "
            "wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When war or rearmament hits the news, defense contractors rally — reliably enough to "
            "trade. See the conflict, buy the basket.\"*\n\n"
            "It's one of the most repeated reflexes in markets, and the causal story is genuinely "
            "sound *over the long run*: conflict drives multi-year procurement, appropriations and "
            "order books. The leap the folklore makes is from that slow fundamental truth to a "
            "**fast, reliable, same-month stock rally you can place a trade on**. We'll line up the "
            "shocks and put that exact claim under a microscope."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If defense *reliably* beat the market in the month after a shock, that would be a "
            "rare thing: a **dated, repeatable** edge you could systematise — see the headline, buy "
            "the basket, short the market against it, collect. That's the dream the reflex sells. But "
            "two things have to be true for it to pay: (1) the move has to be **positive on average**, "
            "not just memorable in a couple of cases; and (2) it has to be **reliable** enough — and "
            "frequent enough — that you'd actually allocate capital to it. A trade that fires a "
            "handful of times a *decade*, on a move you can't distinguish from luck, is a story, not a "
            "strategy."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We run a clean **event study**. Fix a transparent list of **{R['n_events']} "
            "geopolitical-shock dates** (US strikes, invasions, Ukraine, Israel–Iran…), chosen on the "
            "*headline* — never on the outcome. Then:\n\n"
            "1. **Enter the day after.** Buy the defense basket one day after each shock (no peeking "
            "at the shock-day move you couldn't have caught), and measure its return *in excess of "
            "SPY* over the next **month** (and at 1 week / 1 quarter).\n"
            "2. **Compare to a normal stretch.** Stack that against what the basket does over *any* "
            "random window — its natural drift relative to the market (the base rate).\n"
            "3. **Stress the luck.** Draw the same handful of *random* dates thousands of times and "
            "ask how often pure chance looks as good as the shocks. That's the honest test for a "
            "20-event signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline number.** After each shock, how does the defense basket do versus "
            "the market over the next month — and how often is it actually *up* relative to SPY?"
        ),
        code(
            "wins = [5, 21, 63]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, EVENTS, window=w) for w in wins]\n"
            "    cwin = [r['cond_win']*100 for r in rows]; bwin = [r['base_win']*100 for r in rows]\n"
            "else:\n"
            "    cwin = [R['w5'][3], R['w21'][3], R['w63'][3]]\n"
            "    bwin = [R['w5'][5], R['w21'][5], R['w63'][5]]\n"
            "x = np.arange(len(wins))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, cwin, .4, color=RED, label='after a war shock')\n"
            "ax.bar(x+.2, bwin, .4, color=GREY, label='any random window (base rate)')\n"
            "ax.axhline(50, ls=':', c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1 week', '1 month', '1 quarter'])\n"
            "ax.set_ylim(0, 70); ax.set_ylabel('% of shocks where defense BEATS the market')\n"
            "ax.set_title('\"Defense beats the market on war news\" — at best a coin flip, often worse')\n"
            "for i,(c,b) in enumerate(zip(cwin,bwin)):\n"
            "    ax.annotate(f'{c:.0f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.0f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1-month: defense beats SPY', f'{cwin[1]:.0f}%', 'of shocks vs', f'{bwin[1]:.0f}%', 'baseline')"
        ),
        md(
            f"There's the first crack. After a shock the basket beats the market only about **half** "
            f"the time at one month (**{R['w21'][3]:.0f}%**) and a dismal **{R['w63'][3]:.0f}%** at a "
            "quarter — *worse* than picking a random window. \"Reliably rallies\" was supposed to mean "
            "the red bars tower over grey. They don't."
        ),
        md(
            "**Now look at the events one by one.** The whole illusion lives here: a few vivid "
            "rallies you remember, and just as many big losses you don't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ee = st.event_excess(F, EVENTS, window=21)\n"
            "    labels = [d.date().isoformat() for d in EVENTS]\n"
            "    vals = ee*100\n"
            "else:\n"
            "    labels = [d for d,_ in R['events']]; vals = np.array([v for _,v in R['events']])\n"
            "order = np.argsort(vals)\n"
            "vals = vals[order]; labels = [labels[i] for i in order]\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 6.2))\n"
            "ax.barh(range(len(vals)), vals, color=cols)\n"
            "ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.axvline(vals.mean(), c=GREY, ls='--', lw=2, label=f'average {vals.mean():.2f}%')\n"
            "ax.set_xlabel('defense basket return vs SPY, the month after (%)')\n"
            "ax.set_title('Every shock: the famous winners are matched by forgotten losers')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('biggest winner:', labels[-1], f'{vals[-1]:+.1f}%', '| biggest loser:', labels[0], f'{vals[0]:+.1f}%')"
        ),
        md(
            "Ukraine 2022 (**+6%**) and post-9/11 (**+4%**) are right there at the top — the trades "
            "everyone cites. But look at the bottom: Desert Fox 1998 (**−13%**), the 2001 Afghanistan "
            "invasion (**−9%**), the 2024 Iran barrage (**−7%**). Average all 20 honestly and you land "
            f"slightly **below zero** (**{R['w21'][2]:.1f}%**). The reflex survives because memory is "
            "selective, not because the trade works."
        ),
        md(
            "**Could 20 random dates look this good?** The honest test: draw "
            f"**{R['n_events']}** *random* dates over and over, and see where the shocks' average "
            "month-after excess lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, EVENTS, window=21)\n"
            "    obs = pl['shock_mean']; pmean = pl['placebo_mean']; pval = pl['p_value']\n"
            "    one = 1.0 + F['excess'].values; n=len(one); w=21; lag=1\n"
            "    valid = np.array([one[s:s+w].prod()-1 for s in range(lag, n-w) if not np.isnan(one[s:s+w]).any()])\n"
            "    rng = np.random.default_rng(394)\n"
            "    draws = np.array([valid[rng.integers(0,len(valid),size=len(EVENTS))].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['w21'][2]/100; pmean = R['w21'][4]/100; pval = R['w21'][8]\n"
            "    rng = np.random.default_rng(394); draws = rng.normal(pmean, 0.012, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85, label='20 RANDOM dates, month-after excess')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'the actual war shocks ({obs*100:.2f}%)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('average month-after return vs SPY (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The shocks sit on the LEFT of the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random 20-date draw beats the war shocks {pval*100:.0f}% of the time')"
        ),
        md(
            f"The red line — the actual shocks — sits on the **left** (worse) side of the grey luck "
            f"cloud. About **{R['w21'][8]*100:.0f}%** of random 20-date draws do *better* than buying "
            "defense on the headline. In plain terms: **the war trade underperforms a dartboard.** Not "
            "a missing edge — a slightly *negative* one, well inside the noise."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** After a shock the defense basket *trails* the market by about "
            f"**{abs(R['w21'][2]):.1f}%** over the month, beats it barely half the time, and the "
            "result isn't significant by any test. There's no reliable rally — if anything a mild "
            "post-headline lag.\n"
            f"- **Tradability — Mirage.** No positive edge to trade, and **{R['n_events']} shocks in "
            f"{R['years']:.0f} years** could never be a strategy anyway. Buying the headline loses "
            "money gross *and* net.\n"
            "- **Reliable rally? — Busted.** A vivid-event illusion: the unforgettable wins (Ukraine, "
            "9/11) are matched, draw for draw, by forgotten double-digit losers."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — buy the basket, short the market\n\n"
            "Let's build the trade the reflex implies: the day after each shock, buy the defense "
            "basket and short SPY against it (so you're betting on *defense beating the market*), hold "
            "a month, pay a realistic round-trip. Here's the running P&L across all 20 shocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ee = st.event_excess(F, EVENTS, window=21)\n"
            "    c = st.net_of_costs(F, EVENTS, window=21, cost_bps=10.0)\n"
            "    net_each = ee - 10/1e4\n"
            "else:\n"
            "    net_each = np.array([v/100 for _,v in R['events']]) - 10/1e4\n"
            "cum = np.cumsum(net_each)*100\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(range(1, len(cum)+1), cum, marker='o', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('war shock # (in time order)'); ax.set_ylabel('cumulative net return (%)')\n"
            "ax.set_title('\"Buy defense / short market on every war headline\": a sinking P&L')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after 20 trades, net of 10bps: {cum[-1]:+.1f}% cumulative — you LOSE money')"
        ),
        md(
            f"The line **drifts down**. Net of a 10-bps round-trip the average trade is "
            f"**{R['net'][1][2]:.2f}%** — and costs barely move it (gross was "
            f"**{R['net'][1][1]:.2f}%**), because **costs were never the problem**: there's no edge to "
            "erode. You can't run a fund on a rule that fires under once a year *and* loses on average "
            "when it does. The honest version isn't \"see the war, back up the truck\" — it's \"the "
            "rally you remember isn't in the average.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Change the names.** Swap the 4-name basket for ITA, or add Boeing / L3Harris / "
            "European primes — the dispersion widens, the average stays a coin flip. (We cross-check "
            "ITA in the quant notebook; same null result.)\n"
            "- **Change the dates.** Edit the shock list in `defense_basket/data.py` — add or drop "
            "headlines. As long as you pick on the *headline* (not the outcome), the lesson holds: 20 "
            "events can't certify a reflex.\n"
            "- **The fundamental long game.** The *budget* story is real over years; the *headline "
            "rally* is not. A fork worth doing: does multi-year defense-budget growth predict the "
            "sector at a 1–3 *year* horizon? (Different claim, different study.)\n\n"
            "*Think defense reliably beats the market after war news? Pick your shock dates ex-ante, "
            "enter a day late, draw the same number of random dates, and show the trade landing "
            "**right** of the luck cloud — then we'll talk.*"
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
            "# Defense-Basket — a quantitative teardown 🔬\n"
            "### Event study on 20 geopolitical shocks · excess-over-SPY · plain + HAC *t* · a placebo "
            "randomization null · costs · ETF cross-check · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two things \"defense rallies on war\" fuses: a couple of **memorable** "
            "rallies from the **average** behaviour of the basket relative to the market — and we "
            "confront both with the **sample size**. The decisive object is not the point estimate "
            "(it's actually *negative*, against the lore) but its **power**: with ~20 lifetime shocks, "
            "the event-window excess is statistically indistinguishable from drawing the same number "
            "of random dates.\n\n"
            "> ⚠️ **Data + basket note.** \"Defense\" = an explicit **4-name equal-weight basket** "
            "(LMT, RTX, NOC, GD), excess measured vs **SPY**; we cross-check the **ITA** sector ETF "
            "(lists from 2006). Real data: yfinance daily adjusted closes, 1995→2026. Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Month-after excess over SPY **{R['w21'][2]:.2f}%** (base "
            f"**+{R['w21'][4]:.2f}%**); plain **t = {R['w21'][6]:.2f}**, HAC **t = {R['w21'][7]:.2f}**, "
            f"placebo **p = {R['w21'][8]:.2f}** — *negative* and insignificant; ITA cross-check "
            f"**t = {R['ita'][4]:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Long-basket / short-SPY trade: gross **{R['net'][1][1]:.2f}%**, "
            f"net **{R['net'][1][2]:.2f}%** per trade; **{R['n_events']} events in {R['years']:.0f}y**. "
            "No positive edge to charge costs against — it loses money. |\n"
            f"| **Reliable rally?** | `BUSTED` | Beats-the-market rate **{R['w21'][3]:.0f}%** at 1mo "
            f"(base **{R['w21'][5]:.0f}%**); the spread runs **+6.2% (Ukraine)** to **−12.9% (Desert "
            "Fox)**. Vivid-event + base-rate illusion. |\n\n"
            "> 💡 In plain words: the war trade doesn't just fail to beat the market — in this sample "
            "it *trails* it, and the trailing is inside its own error bar. Strip the two famous "
            "rallies and 20 events leave nothing a random draw can't reproduce."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{D}_t$ be the equal-weight defense-basket daily return and $r^{M}_t$ the market "
            "(SPY) return; the **excess** is $x_t = r^{D}_t - r^{M}_t$. For shock date $\\tau$ the "
            "**event-window excess** is $X_\\tau(W) = \\prod_{s=\\tau+1}^{\\tau+W}(1+x_s) - 1$ "
            "(one-day entry lag, hold $W$ days).\n\n"
            "- **H₁ (the rally is real & positive).** $\\mathbb{E}[X_\\tau(W)] > 0$ and large — defense "
            "out-earns the market after a shock.\n"
            "- **H₂ (it's deployable).** The excess is reliable and frequent enough, net of costs, to "
            "run as a long/short trade.\n"
            "- **H₃ (\"reliably rallies\").** The post-shock behaviour reflects a shock-specific edge, "
            "not selective memory of a few vivid wins.\n\n"
            "We find **H₁ rejected in sign** ($\\bar X < 0$, $|t| < 2$), **H₂ rejected** (negative, "
            "~20 events / 31y), **H₃ rejected** (beats-market rate ≈ a coin, the average is dragged "
            "down by losers as big as the famous winners). The reflex is true exactly where it's "
            "untestable (a couple of anecdotes) and false where it would pay (the average)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one estimate and its standard error:\n\n"
            "$$\\bar X = \\frac1k\\sum_{\\tau} X_\\tau(W),\\qquad "
            "t = \\frac{\\bar X}{s_X/\\sqrt{k}}.$$\n\n"
            "With $k = 20$, $s_X/\\sqrt{k}$ is large: a one- or two-point $\\bar X$ drowns in its own "
            "SE whichever way it points. A raw **win-rate** is the wrong lens — defense carries a mild "
            "positive beta drift vs SPY *unconditionally*, so the right comparison is the **excess "
            "over that base rate**, not the sign. And because the daily in-window observations overlap "
            "and autocorrelate, we also report a **Newey-West (HAC)** *t* on the pooled daily excess. "
            "Finally, the honest small-sample instrument is a **randomization (placebo) test**: "
            "resample $k$ random entry dates and ask how often chance matches the shocks. That number "
            "decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Equal-weight daily return of LMT/RTX/NOC/GD; excess vs SPY (yfinance "
            f"adjusted closes, {R['start']}→{R['end']}, {R['days']:,} days). ITA ETF as a shorter-"
            "window cross-check.\n"
            f"- **Treatment.** **{R['n_events']}** hardcoded war/invasion/rearmament headline dates, "
            "chosen ex-ante on the *headline* (listed in `data.py`; sourced in references).\n"
            "- **Event window.** Cumulative excess over $W\\in\\{5,21,63\\}$ trading days, entering "
            "the close **1 day after** the shock (no look-ahead); drop events whose window overruns "
            "the tape.\n"
            "- **Null #1 (t / HAC-t).** Event-window excess vs zero (plain one-sample *t*); pooled "
            "in-window daily excess vs zero with a Bartlett-kernel **Newey-West** SE.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random valid entry dates; $p = "
            "\\Pr[\\text{random-draw mean} \\ge \\text{shock mean}]$ — the small-sample workhorse.\n"
            "- **Costs.** 1-day lag + a 10-bps all-in round-trip on the long-basket / short-SPY pair.\n"
            "- **Positive control.** A deterministic frame where the basket is $\\beta\\cdot$SPY + "
            "noise (zero true excess by construction) with **20 shocks injected** and a **known** "
            "window edge: the engine must recover a planted edge **and** must NOT manufacture "
            "significance when the true edge is zero.\n\n"
            "> **What would make us say \"mirage\":** a negative or insignificant $\\bar X$, a "
            "beats-market rate near the base rate, and a placebo $p$ far from the tails. That is "
            "exactly what we get."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — negative, small, growing with the window\n\n"
            "Event-window excess with its $\\pm$ standard error, against the unconditional base rate "
            "(diamonds). Negative at every horizon, and *more* negative the longer you hold."
        ),
        code(
            "wins = [5, 21, 63]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts = [], [], []\n"
            "    for w in wins:\n"
            "        s = st.summarize(F, EVENTS, window=w); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "    ses = [(lambda e: e.std(ddof=1)/np.sqrt(len(e)))(st.event_excess(F, EVENTS, window=w)) for w in wins]\n"
            "else:\n"
            "    cm = [R['w5'][2]/100, R['w21'][2]/100, R['w63'][2]/100]\n"
            "    bm = [R['w5'][4]/100, R['w21'][4]/100, R['w63'][4]/100]\n"
            "    ts = [R['w5'][6], R['w21'][6], R['w63'][6]]; ses = [0.0075, 0.011, 0.017]\n"
            "x = np.arange(len(wins))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=RED, width=.5, label='after a shock (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1 week', '1 month', '1 quarter']); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean excess over SPY (%)')\n"
            "ax.set_title('Excess is negative and inside its error bar at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('plain t by window:', {f'{w}d': round(t,2) for w,t in zip(wins,ts)})"
        ),
        md(
            f"> 💡 In plain words: at one month the basket runs **{R['w21'][2]:.2f}%** *vs* a base "
            f"**+{R['w21'][4]:.2f}%** — a ~1.6-point shortfall at **t = {R['w21'][6]:.2f}** (not "
            f"significant). At a quarter it's **{R['w63'][2]:.2f}%** (**t = {R['w63'][6]:.2f}**). H₁ is "
            "**rejected in sign**: not the positive edge the lore promises, but a small negative number "
            "you can't distinguish from zero."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            "Draw 20 *random* entry dates 20,000 times; the histogram is the null distribution of the "
            "month-after mean excess. The shocks are the red line; the *p*-value is the right-tail mass "
            "(the share of random draws that do *as well or better*)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, EVENTS, window=21); obs = pl['shock_mean']; pval = pl['p_value']\n"
            "    one = 1.0 + F['excess'].values; n=len(one); w=21; lag=1\n"
            "    valid = np.array([one[s:s+w].prod()-1 for s in range(lag, n-w) if not np.isnan(one[s:s+w]).any()])\n"
            "    rng = np.random.default_rng(394)\n"
            "    draws = np.array([valid[rng.integers(0,len(valid),size=len(EVENTS))].mean() for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['w21'][2]/100; pval = R['w21'][8]\n"
            "    rng = np.random.default_rng(394); draws = rng.normal(R['w21'][4]/100, 0.012, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: mean of {len(draws):,} random 20-date draws')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed shock mean {obs*100:.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('month-after mean excess over SPY (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the shocks are LEFT of the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random 20-date draw >= shocks] = {pval:.3f}  (a positive edge would push this <0.05)')"
        ),
        md(
            f"> 💡 In plain words: **{R['w21'][8]*100:.0f}%** of random 20-date samples *beat* the war "
            "shocks. A real edge would push the red line into the far **right** tail; instead it sits "
            "**left** of the cloud's centre. This is the crux — not a weak positive, but a point "
            "estimate a random dartboard reproduces (and improves on) nine times in ten. H₃ rejected."
        ),
        md(
            "### 4c · Robustness, ETF cross-check, and cost — nothing rescues it\n\n"
            "Vary the holding window, swap to the ITA sector ETF, and apply a round-trip cost. The "
            "excess is negative at *every* window, ITA agrees, and 10 bps barely moves a multi-week "
            "hold (cost is **not** the binding constraint — there's simply no edge)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for w in (5,10,21,42,63):\n"
            "        s = st.summarize(F, EVENTS, window=w); rob.append((w, s['cond_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F.copy(); F2['etf_excess'] = F2['etf'] - F2['mkt']\n"
            "    ev_ita = pd.DatetimeIndex([d for d in EVENTS if d >= pd.Timestamp('2006-05-05')])\n"
            "    si = st.summarize(F2, ev_ita, window=21, col='etf_excess')\n"
            "    ita = (si['n'], si['cond_mean']*100, si['cond_win']*100, si['base_mean']*100, si['t'], si['p_placebo'])\n"
            "    netr = [(w,)+(lambda c:(c['gross_mean']*100,c['net_mean']*100))(st.net_of_costs(F, EVENTS, window=w, cost_bps=10.0)) for w in (5,21,63)]\n"
            "else:\n"
            "    rob = R['robust']; ita = R['ita']; netr = [(w,g,nn) for (w,g,nn) in R['net']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "ws = [r[0] for r in rob]; cms = [r[1] for r in rob]\n"
            "a1.bar([f'{w}d' for w in ws], cms, color=RED, width=.55)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,(c,r) in enumerate(zip(cms,rob)): a1.annotate(f't={r[2]:.2f}',(i,c),ha='center',va='top' if c<0 else 'bottom', fontsize=8)\n"
            "a1.set_title('Excess is negative at every window'); a1.set_xlabel('holding window'); a1.set_ylabel('mean excess (%)')\n"
            "ms = [r[0] for r in netr]; g = [r[1] for r in netr]; nv = [r[2] for r in netr]; xx=np.arange(len(ms))\n"
            "a2.bar(xx-.2,g,.4,color=AMBER,label='gross'); a2.bar(xx+.2,nv,.4,color=GREY,label='net @10bps')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xticks(xx); a2.set_xticklabels([f'{w}d' for w in ms]); a2.set_ylabel('mean trade return (%)')\n"
            "a2.set_title('Cost barely matters — there is no edge to erode'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ITA cross-check (1mo): n={ita[0]} cond={ita[1]:.2f}% win={ita[2]:.0f}% base={ita[3]:.2f}% t={ita[4]:.2f} p={ita[5]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the single-name basket and the ITA ETF (**t = {R['ita'][4]:.2f}**, "
            f"placebo p = {R['ita'][5]:.2f}) tell the *same* story — a small negative excess, no "
            "significance. There is no window and no instrument at which the war trade turns positive, "
            "and a 10-bps round-trip shaves a tenth of a point off a months-long hold. **The constraint "
            "is the absence of an edge, not the cost of trading it.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic frame where the basket is $\\beta\\cdot$SPY + idiosyncratic noise (so "
            "the true excess is **zero** by construction) with **20 shocks injected**: with **zero** "
            "planted edge the test must stay below t=2 (20 events can't fake significance); with a "
            "**+6%/month** planted edge it must light up. Both hold — proving the engine is unbiased "
            "*and* that this sample size only detects implausibly large edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.06):\n"
            "    syn, evs = data.synthetic_defense(n_events=20, edge_window=edge, seed=394, window=21)\n"
            "    s = st.summarize(syn, evs, window=21)\n"
            "    res.append((edge, s['n'], s['cond_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('plain t (1-month)'); ax.set_title('Control: ~20 events detect only a HUGE edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/mo: detected={k} cond={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(below 2 — no false positive); only a **+6%/month** edge — enormous — reaches "
            f"**t = {R['syn'][1][4]:.2f}**. So the machinery is honest, and the real-tape *t* of "
            f"**{R['w21'][6]:.2f}** is exactly what an *absent* edge looks like through a 20-event "
            "keyhole. The sample size is half the verdict; the negative sign is the other half."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — month-after excess **{R['w21'][2]:.2f}%** at plain "
            f"**t = {R['w21'][6]:.2f}** / HAC **t = {R['w21'][7]:.2f}** / placebo **p = {R['w21'][8]:.2f}**; "
            f"ITA cross-check **t = {R['ita'][4]:.2f}**; negative at every window. Not REAL, not even "
            "WEAK — the point estimate has the **wrong sign** and is indistinguishable from zero.\n"
            f"- **Tradability `MIRAGE`** — long-basket / short-SPY loses **{R['net'][1][2]:.2f}%** per "
            f"trade net (gross **{R['net'][1][1]:.2f}%**); costs are negligible (no edge to erode); "
            f"**{R['n_events']} trades / {R['years']:.0f}y**. Un-allocatable and unprofitable.\n"
            f"- **Reliable rally? `BUSTED`** — beats-market rate **{R['w21'][3]:.0f}%** vs base "
            f"**{R['w21'][5]:.0f}%**; spread **+6.2% (Ukraine)** to **−12.9% (Desert Fox)**. A "
            "vivid-event + base-rate illusion: the wins you remember are cancelled by the losses you "
            "don't."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* month-after excess have to "
            "be for a $k$-event study to detect it at t=2? At $k=20$ you'd need a large positive "
            "excess; the real signal is **negative**, sitting below the axis — there is no edge to "
            "size, only a loss to avoid."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ee = st.event_excess(F, EVENTS, window=21); sd = ee.std(ddof=1)\n"
            "    obs_excess = R['w21'][2]/100\n"
            "else:\n"
            "    sd = 0.048; obs_excess = R['w21'][2]/100\n"
            "ks = np.arange(5, 150)\n"
            "min_detectable = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='positive excess needed for t=2')\n"
            "ax.plot(ks, -min_detectable*100, c=AMBER, lw=2, ls=':')\n"
            "ax.axhline(obs_excess*100, c=RED, ls='--', label=f'observed excess {obs_excess*100:.2f}% (negative!)')\n"
            "ax.axvline(R['n_events'], c=GREY, ls=':', label=f\"our k={R['n_events']}\")\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('1-month excess over SPY (%)')\n"
            "ax.set_title('The real signal is below zero — there is no edge to detect'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_events'])\n"
            "print(f'at k={R[\"n_events\"]} you would need ~+{need*100:.1f}% excess for t=2; observed is {obs_excess*100:.2f}% (wrong sign)')"
        ),
        md(
            "> 💡 In plain words: the amber band is the **minimum detectable excess** (positive or "
            "negative); the red line — what we actually observe — sits *below zero*, well inside the "
            "band. Even granting the lore a real edge, you'd need many times more shocks than history "
            "will ever supply to certify it — and the point estimate isn't even pointing the right "
            "way. **There is no sizing, no window, no cost assumption that turns a couple of "
            "remembered rallies into a strategy. The rarity that makes the trade feel obvious is "
            "exactly what makes it untestable, and the average is what makes it a loser.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Anticipation horizon.** The negative post-headline excess is consistent with the "
            "shock being priced *before/on* the event. A clean fork: scan the run-up window "
            "($\\tau-W$ to $\\tau$) for the rally the reflex is actually (mis)remembering.\n"
            "- **Wider universe / true sector cap-weight.** Replace the 4-name basket with the full "
            "ITA holdings or an XAR-style equal-weight; the dispersion and base rate shift, the power "
            "problem does not.\n"
            "- **The fundamental long game.** The *budget* thesis lives at a multi-**year** horizon "
            "(appropriations → orders → earnings), a different and more defensible claim than the "
            "same-month headline rally tested here.\n\n"
            "*The reproducible core is offline and deterministic; the basket and shock list are "
            "explicit and editable. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
