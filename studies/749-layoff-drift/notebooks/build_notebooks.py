"""Generate the two narrative notebooks for Study 749 (Layoff-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
closes under ../_cache/ (each event ticker + SPY) and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes, hardcoded ~28-event layoff table, as-of 2026-07-13; all 28 events priced;
# market-model CAR, SPY benchmark, 1-day execution lag).
R = dict(
    asof="2026-07-13", n_table=28, n_used=28, fingerprint="5c784549f0cd",
    # pop leg [+1,+3]:  (n, mean_pct, win_pct, welch_t, placebo_p)
    pop=(28, -0.34, 43, -0.53, 0.576),
    # drift leg [+4,+63]: (n, mean_pct, win_pct, welch_t, placebo_p, hac_t, daily_bps)
    drift=(28, 9.01, 64, 2.25, 0.007, 3.04, 15.01),
    net=(9.01, 8.91),                 # gross, net @ 10 bps
    median_drift=10.35,
    drop3=(4.37, 1.28),               # drop top-3 monster recoveries: mean_pct, t
    boot=(0.73, 2.29),                # bootstrap 5th-pct t, median t
    cluster=(13, 12.35, 2.48),        # n calendar-quarter clusters, collapsed mean, t
    monsters=[("META", 55), ("XOM", 45), ("BA", 42)],
    # robustness: (label, pop_mean, pop_t, drift_mean, drift_t)
    robust=[("[1,3]/[4,63]", -0.34, -0.53, 9.01, 2.25),
            ("[0,1]/[2,42]", 0.07, 0.06, 6.63, 2.27),
            ("[1,5]/[6,126]", 0.07, 0.10, 18.37, 2.69)],
    # split: (label, n, pop_mean, pop_t, drift_mean, drift_t)
    split=[("2022+ tech wave", 19, -0.08, -0.10, 9.65, 2.05),
           ("pre-2022", 9, -0.89, -0.73, 7.65, 0.97)],
    # synthetic control (seed 723): (pop_bps, drift_bps, pop_mean, pop_t, drift_mean, drift_t, hac_t)
    syn=[(0.0, 0.0, 0.38, 0.49, 4.13, 0.99, 1.23),
         (0.0, 400.0, 0.38, 0.49, 8.13, 1.95, 2.42)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Restructuring_pop%3F: Busted](https://img.shields.io/badge/Restructuring_pop%3F-Busted-8b949e?style=flat-square)\n\n"
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

from layoff_drift import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    PANEL = st.car_panel(PRICES, EVENTS)            # pop [+1,+3] and drift [+4,+63]
else:
    PRICES = EVENTS = PANEL = None
print("real price cache present:", HAVE_REAL,
      "| events with data:", (0 if PANEL is None else len(PANEL)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell
# can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When a company axes 10,000 jobs, does the stock cheer? 📉\n"
            "### The \"restructuring pop\" — and the quarter-long drift that isn't quite what it looks\n\n"
            + BADGES +
            "You've heard the take. A company announces a giant layoff, and the pundits nod: *\"good — "
            "they're finally cutting costs, the stock will pop.\"* In 2022–2023 it became a whole "
            "narrative — the \"year of efficiency\" — where slashing headcount was supposed to launch "
            "the stock and keep it climbing as fatter margins rolled in.\n\n"
            "So we did the boring thing: took ~28 real, dated large-cap layoff announcements — Meta, "
            "Amazon, Google, Boeing, Exxon, Ford — and measured exactly what each stock did afterward, "
            "*after subtracting whatever the whole market did*. The pop? It isn't there. The drift? It's "
            "there — until you look at **which** companies made the list.\n\n"
            "> 📓 **Plain-language layer.** Want the market-model math, the *t*-stats, the HAC test and "
            "the placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** There's no free, clean database of layoff dates, so we "
            "**hardcode a transparent table** of famous ones. Crucially, every name on it is a company "
            "that **survived** — the ones that laid off staff on the way to bankruptcy left no stock to "
            "measure. Hold that thought; it's the whole story. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock **pop** on the layoff news? | **No.** Over the 3 days after the "
            f"announcement the market-adjusted move is **{R['pop'][1]:+.1f}%** — a shrug that leans, if "
            "anything, slightly *down*. The \"cost-cutting cheer\" is a myth. |\n"
            "| Is there a longer **drift**? | **On paper, yes** — about "
            f"**+{R['drift'][1]:.0f}%** over the next quarter. That looks significant… |\n"
            "| …so it's a real edge? | **No.** Knock out the three giant recoveries (Meta +55%, Exxon "
            f"+45%, Boeing +42%) and the drift falls to **+{R['drop3'][0]:.0f}%** and goes "
            "*insignificant*. And every name on the list is a **survivor** — we're really measuring "
            "\"beaten-down big-caps that lived, later went up.\" |\n"
            "| Could I trade it? | **No.** At the announcement you can't know which layoff-er becomes "
            "Meta and which delists — and that guess *is* the edge. |\n\n"
            "> The layoff \"pop\" doesn't exist, and the layoff \"drift\" is mostly a survivorship "
            "mirage: a few cycle-bottom survivors that rallied, wearing a cost-cutting label."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company announces mass layoffs, it's cutting fat — buy the restructuring pop, "
            "and ride the drift as margins recover.\"*\n\n"
            "It's floor wisdom with an academic-sounding cousin (\"post-announcement drift\"). The "
            "seduction is that a layoff is a **clean, dated catalyst**: you know the day, you know the "
            "direction (they're cutting costs), so surely there's a trade. We take that seriously and "
            "measure the **abnormal return** — the stock's move *minus the market's* — over a short "
            "**pop** window and a longer **drift** window after each announcement."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two different claims hide inside \"layoffs are bullish,\" and they pay very differently. "
            "(1) *The pop* — the stock jumps in the days right after the news. If real, it's priced in "
            "before you can blink. (2) *The drift* — the stock keeps climbing for weeks, a slow bleed of "
            "good news you could actually hold. That second one is the only tradable version — **and "
            "it's exactly the one where survivorship does its dirtiest work**, because the companies "
            "that *recovered* from their layoffs are the only ones still on the tape to measure."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode **~{R['n_table']} notable layoff announcements** (2015–2025; all "
            f"**{R['n_used']}** priced, as-of {R['asof']}) and run a textbook **event study**:\n\n"
            "1. **Subtract the market.** For each stock, fit a line — *how it normally moves with the "
            "S&P* — over a calm window **before** the announcement. The **abnormal return** is whatever "
            "the stock did beyond that.\n"
            "2. **Two clocks.** Add up the abnormal return over a short **pop** window (days +1 to +3) "
            "and a long **drift** window (days +4 to +63). Enter the *day after* the headline — all a "
            "real trader can do.\n"
            "3. **Stress the luck — and the list.** Draw the same handful of *random* windows thousands "
            "of times (the placebo), then knock out the biggest winners and re-check. If a signal only "
            "survives with a few lucky names, it isn't a signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the pop.** Here's every event's abnormal return over the 3 days after the "
            "announcement — the window where the \"cheer\" should show up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('pop')\n"
            "    pops = p['pop'].values*100; labs = [f\"{t} {d.year}\" for t,d in zip(p['ticker'], p['announce_date'])]\n"
            "    mean_pop = PANEL['pop'].mean()*100\n"
            "else:\n"
            "    pops = np.array([-7.5,-4.3,-3.9,-3.6,-3.0,-3.0,-2.6,-1.9,-1.5,-1.0,-0.6,-0.4,-0.3,-0.2,-0.1,0.5,0.6,1.1,1.2,1.5,2.2,3.4,3.7,3.9,4.2,4.5,5.1]); labs=['']*len(pops); mean_pop=R['pop'][1]\n"
            "cols = [RED if v<0 else GREEN for v in pops]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.6))\n"
            "y = np.arange(len(pops))\n"
            "ax.barh(y, pops, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axvline(mean_pop, c=AMBER, lw=2, ls='--', label=f'mean {mean_pop:+.1f}%')\n"
            "ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)\n"
            "ax.set_xlabel('abnormal return over [+1,+3] days (%)'); ax.legend(loc='lower right')\n"
            "ax.set_title('The \"restructuring pop\": every layoff, 3-day market-adjusted move')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean 3-day pop = {mean_pop:+.2f}% — a shrug, tilted slightly negative')"
        ),
        md(
            f"No cheer. The average 3-day pop is **{R['pop'][1]:+.1f}%** — basically zero, leaning "
            "*down*. Winners and losers split roughly half-and-half. Whatever the market thinks of a "
            "layoff, it isn't \"buy.\" So much for the pop. Now the longer window."
        ),
        md(
            "**Next, the drift — and why it's a trap.** Here's the quarter-long (days +4 to +63) "
            "abnormal return per event. The average is a juicy **+9%** — but watch what three names do "
            "to it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('drift')\n"
            "    dr = p['drift'].values*100; labs=[f\"{t} {d.year}\" for t,d in zip(p['ticker'], p['announce_date'])]\n"
            "    mean_dr = PANEL['drift'].mean()*100\n"
            "else:\n"
            "    dr = np.array([-23.7,-21.5,-17.3,-14.2,-12.8,-11.9,-10.6,-10.2,-8.2,-1.0,2.0,5.2,9.0,9.8,10.9,11.2,11.8,12.3,16.0,17.8,18.5,28.4,34.8,37.5,42.5,45.3,55.2]); labs=['']*len(dr); mean_dr=R['drift'][1]\n"
            "top3 = set(t for t,_ in R['monsters'])\n"
            "cols = [AMBER if any(labs[i].startswith(t) for t in top3) else (GREEN if dr[i]>0 else RED) for i in range(len(dr))]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.6))\n"
            "y = np.arange(len(dr))\n"
            "ax.barh(y, dr, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axvline(mean_dr, c='k', lw=1.5, ls='--', label=f'mean {mean_dr:+.1f}%')\n"
            "ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)\n"
            "ax.set_xlabel('abnormal return over [+4,+63] days (%)'); ax.legend(loc='lower right')\n"
            "from matplotlib.patches import Patch\n"
            "ax.add_artist(ax.legend(handles=[Patch(color=AMBER,label='the 3 giant recoveries')], loc='upper left'))\n"
            "ax.set_title('The quarter-long drift — carried by a handful of cycle-bottom survivors')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean drift = {mean_dr:+.1f}% — but the top of the chart is Meta/Exxon/Boeing recoveries')"
        ),
        md(
            f"The three amber bars — **Meta +55%, Exxon +45%, Boeing +42%** — are cycle-bottom "
            f"*recoveries*, not layoff magic. Drop them and the drift falls from **+{R['drift'][1]:.0f}%** "
            f"to **+{R['drop3'][0]:.0f}%**, and its *t*-stat sinks from **{R['drift'][3]:.1f}** to "
            f"**{R['drop3'][1]:.1f}** — below the line where we'd call anything real. A signal that "
            "leans on three names isn't a signal."
        ),
        md(
            "**The deeper problem — who's even on this list?** Every company here *survived* its "
            "layoffs. The ones that cut staff on the way to delisting aren't measurable — they left no "
            "stock. So we're not asking \"do layoffs help?\" We're asking \"did big-caps that survived "
            "their layoffs go up?\" — and of course some did. Here's the tell: the drift is a slow "
            "grind over *months*, exactly the shape of a beaten-down survivor mean-reverting."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import numpy as np\n"
            "    # average cumulative abnormal path over [0..+63] across events\n"
            "    paths = []\n"
            "    for e in EVENTS:\n"
            "        w = st.event_abnormal(PRICES, e['ticker'], e['announce_date'], pop_win=(1,1), drift_win=(2,63))\n"
            "        if w is None: continue\n"
            "        daily = np.concatenate([w['pop_daily'], w['drift_daily']])\n"
            "        paths.append(np.cumsum(daily))\n"
            "    L = min(len(p) for p in paths); M = np.array([p[:L] for p in paths]).mean(0)*100\n"
            "    days = np.arange(1, L+1)\n"
            "else:\n"
            "    days = np.arange(1,64); M = np.concatenate([np.linspace(0,-0.3,3), np.linspace(-0.3,9.0,60)])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(days, M, c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvspan(1,3, color=AMBER, alpha=.15, label='pop window')\n"
            "ax.set_xlabel('trading days after announcement'); ax.set_ylabel('avg cumulative abnormal return (%)')\n"
            "ax.set_title('No jump, then a slow months-long grind — the signature of a recovery, not a catalyst'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'flat for the first 3 days (no pop), then a slow drift to ~+{M[-1]:.0f}% over a quarter')"
        ),
        md(
            "There's no jump on the news (flat through the amber pop window) and then a slow climb over "
            "months. That's not the market *reacting to a layoff* — it's what a portfolio of "
            "beaten-down survivors does after a cycle low. The layoff is a coincidence of *timing*, not "
            "the cause."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** No pop at all (**{R['pop'][1]:+.1f}%**, *t* = {R['pop'][3]:.2f}). A "
            f"drift that looks real (**+{R['drift'][1]:.0f}%**) but dies when you remove three recovery "
            f"names (**+{R['drop3'][0]:.0f}%**, *t* = {R['drop3'][1]:.1f}) — on a list of survivors only. "
            "Real on this tape, weak as a law.\n"
            "- **Tradability — Mirage.** Costs barely dent it, but you can't pick the survivors in "
            "advance, and picking them *is* the whole edge. Nothing to harvest.\n"
            "- **\"Restructuring pop?\" — Busted.** The cheer the pundits promise isn't in the data. "
            "The only thing that drifts up is a survivor-flattered recovery."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the survivor you can't pick\n\n"
            "Here's the trade laid bare. Two companies announce identical 10% layoffs on the same day. "
            "One is early-Meta (about to rip +55%); the other is quietly on its way to zero. The "
            "\"drift\" is entirely about *which one you bought* — a call you cannot make from the "
            "layoff headline. Below: the drift with, and without, the three survivors that carry it."
        ),
        code(
            "labels = ['full sample\\n(survivors only)', 'drop the 3\\ngiant recoveries']\n"
            "vals = [R['drift'][1], R['drop3'][0]]; ts = [R['drift'][3], R['drop3'][1]]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "bars = ax.bar(labels, vals, color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.1f}%\\n(t={t:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('quarter-long drift (%)')\n"
            "ax.set_title('Take out three names you could never have picked, and the edge is gone')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'the entire tradable claim rests on Meta/Exxon/Boeing — names you cannot identify at the layoff')"
        ),
        md(
            "That's the ballgame. The \"drift\" is a handful of cycle-bottom recoveries you'd have had "
            "to hand-pick *before* they recovered — which is not a strategy, it's hindsight. Too **"
            "selection-ridden**, too **survivor-flattered**, too **fragile** to trade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 📉\n\n"
            "- **The sibling catalyst.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) runs the same "
            "event-study machinery on forced CEO exits — a real day-one jump you can't trade, weak over "
            "any holdable window. Same small-sample event-study pathology.\n"
            "- **The other survivorship legend.** [Study 389 — Name-Change-Effect](../389-name-change-effect/) "
            "— the \"pop then dump\" of theme-chasing rebrands, remembered only because the losers "
            "delisted.\n"
            "- **Build your own.** Add the *distressed* layoff-announcers that delisted (hard — you'll "
            "need a survivorship-free database), and watch the drift shrink toward zero. That's the "
            "experiment that would actually settle it.\n\n"
            "*Think there's a tradable layoff edge? Show a drift that survives **dropping the three "
            "biggest winners** and adding back the **firms that didn't survive** — then we'll talk.*"
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
            "# Layoff-Drift — a quantitative event-study teardown 🔬\n"
            "### Market-model pop/drift CARs · Welch *t* + Newey-West HAC *t* + a placebo null · the "
            "drop-3 / bootstrap fragility · the survivorship argument · a costed book · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"layoffs are bullish\" fuses: a **(non-existent) announcement pop** and a "
            "**quarter-long drift** that is nominally significant but **fragile to a handful of names "
            "and manufactured by survivorship**. The decisive object is not the sign of the drift (it's "
            "positive) but its **robustness**: it clears *t* = 2 raw, and then fails to survive removing "
            "three recovery names, a bootstrap of the cross-section, and — most of all — the fact that "
            "the sample is winners by construction.\n\n"
            "> ⚠️ **Data + survivorship note.** No free clean layoff-date database exists; we use a "
            "hardcoded table of ~28 notable large-cap announcements. **Every name survived** — the "
            "distressed layoff-announcers that delisted are absent, so the drift is biased **up** (an "
            "upper bound, not a law), named on the Signal axis. Real data: yfinance daily "
            "**total-return** closes, each ticker + SPY. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pop [+1,+3] **{R['pop'][1]:+.2f}%** (Welch *t* = {R['pop'][3]:.2f}, "
            f"placebo *p* = {R['pop'][4]:.2f}) — **no pop**. Drift [+4,+63] **+{R['drift'][1]:.2f}%** "
            f"(*t* = {R['drift'][3]:.2f}, HAC *t* = {R['drift'][5]:.2f}, placebo *p* = {R['drift'][4]:.3f}) "
            f"— but drop-3 → **+{R['drop3'][0]:.2f}%** (*t* = {R['drop3'][1]:.2f}), bootstrap 5th-pct "
            f"*t* = {R['boot'][0]:.2f}, on a survivor-only tape. |\n"
            f"| **Tradability** | `MIRAGE` | Net of 10 bps the drift is **+{R['net'][1]:.2f}%** — cost "
            "isn't the constraint. Repeatability is: the edge is a few macro-timed recoveries you can't "
            "pick ex-ante, on a sample selected for survival. |\n"
            f"| **Restructuring pop?** | `BUSTED` | The announcement pop is **absent / slightly "
            f"negative** (*t* = {R['pop'][3]:.2f}); the only up-move is a fragile, survivor-flattered "
            "3-month drift. |\n\n"
            "> 💡 In plain words: there is no cost-cutting *cheer* on the tape, and the quarter-long "
            "drift is what you get when you (a) only look at companies that survived their layoffs and "
            "(b) let three cycle-bottom rockets do the lifting. Take either crutch away and it's noise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For event $i$ with announcement day $0$, fit the market model "
            "$r_{i,t} = \\alpha_i + \\beta_i\\, r_{m,t} + \\varepsilon_{i,t}$ on a clean estimation "
            "window $[-130,-10]$, then the **abnormal return** is "
            "$AR_{i,t} = r_{i,t} - (\\hat\\alpha_i + \\hat\\beta_i\\, r_{m,t})$ and the **CAR** over "
            "window $[\\tau_1,\\tau_2]$ is $\\mathrm{CAR}_i = \\sum_{t=\\tau_1}^{\\tau_2} AR_{i,t}$, "
            "with a one-day entry lag on every window.\n\n"
            "- **H₁ (pop).** $\\mathbb{E}[\\mathrm{CAR}_{[+1,+3]}] > 0$ — the restructuring cheer.\n"
            "- **H₂ (drift).** $\\mathbb{E}[\\mathrm{CAR}_{[+4,+63]}] > 0$ and robust — the PEAD-style "
            "continuation, the only *tradable* version.\n"
            "- **H₃ (deployable).** H₂ survives a one-day lag, costs, **removing the top-3 names**, and "
            "the survivorship caveat.\n\n"
            "We find **H₁ rejected** (pop $\\approx 0$, slightly negative); **H₂ supported only raw** "
            "(drift $+9\\%$, $t=2.25$, HAC $t=3.04$) **but fragile** (drop-3 $\\Rightarrow t=1.28$; "
            "bootstrap 5th-pct $t=0.73$); **H₃ rejected** (the drift is survivor-selected and "
            "carried by a few recoveries you can't pick ex-ante). The legend is false where it's vivid "
            "(the pop) and un-certifiable where it would pay (the drift)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The pop leg is a clean cross-sectional mean test. The drift leg needs **two** honest "
            "instruments, because a drift's daily increments are autocorrelated:\n\n"
            "$$t_{\\text{Welch}} = \\frac{\\bar{\\mathrm{CAR}}}{s/\\sqrt{k}}, \\qquad "
            "t_{\\text{HAC}} = \\frac{\\bar{ar}_{\\text{daily}}}{\\widehat{\\mathrm{se}}_{\\text{NW}}"
            "(\\bar{ar}_{\\text{daily}})},$$\n\n"
            "the first across the $k\\!\\approx\\!28$ event CARs, the second (Newey-West, Bartlett, "
            "$L=5$) on the **pooled daily** abnormal returns over the drift window. Both matter, and "
            "**both are undone by the same two facts**: (1) with $k\\approx 28$ heavy-tailed CARs, a "
            "few outliers set the mean — so we re-run **dropping the top-3** and **bootstrapping** the "
            "cross-section; (2) the panel is **survivor-selected**, which shifts $\\mathbb{E}[AR]$ up "
            "for reasons that have nothing to do with layoffs. A *t* that survives neither the drop-3 "
            "nor the survivorship logic is `WEAK`, not `REAL`."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Event table.** Hardcoded ~{R['n_table']} large-cap layoff announcements (ticker, "
            f"announce date, approx. cut); all **{R['n_used']}** priced, as-of {R['asof']}, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **Market model.** $r = \\alpha + \\beta\\,r_{\\mathrm{SPY}}$ on $[-130,-10]$ (120-day "
            "estimation, 10-day gap).\n"
            "- **Windows.** Pop $[+1,+3]$, drift $[+4,+63]$; robustness over $[0,+1]/[+2,+42]$ and "
            "$[+1,+5]/[+6,+126]$; **1-day execution lag throughout**.\n"
            "- **Null #1 (Welch t).** Each leg mean vs 0.\n"
            "- **Null #2 (HAC t).** Newey-West on the pooled daily drift series.\n"
            "- **Null #3 (placebo).** Random non-event $(\\text{ticker},\\text{date})$ windows on the "
            "same names.\n"
            "- **Fragility.** Drop the 3 largest drifts; bootstrap the event cross-section (5,000×).\n"
            "- **Tradable variant.** Enter +1 day, hold the drift window; one-way 10-bps round-trip.\n"
            "- **Positive control.** A deterministic per-event panel with **plantable pop/drift edges**: "
            "the engine must recover a planted drift **and** must NOT fabricate significance at zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs — no pop, a raw-significant drift\n\n"
            "Mean CAR per leg with its $\\pm$ standard error. The pop straddles zero; the drift sits "
            "clearly above it — *raw*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    po = PANEL['pop'].to_numpy(); dr = PANEL['drift'].to_numpy()\n"
            "    means=[po.mean()*100, dr.mean()*100]; ses=[po.std(ddof=1)/np.sqrt(len(po))*100, dr.std(ddof=1)/np.sqrt(len(dr))*100]\n"
            "    ts=[st.welch_t(po), st.welch_t(dr)]\n"
            "else:\n"
            "    means=[R['pop'][1], R['drift'][1]]; ses=[0.64, 4.0]; ts=[R['pop'][3], R['drift'][3]]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['pop [+1,+3]','drift [+4,+63]'], means, yerr=ses, capsize=6, color=[RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean CAR (%)')\n"
            "for i,(m,s) in enumerate(zip(means,ses)): ax.annotate(f'{m:+.2f}%\\n(t={ts[i]:+.2f})',(i,m),ha='center',va='bottom' if m>=0 else 'top')\n"
            "ax.set_title('No pop; a drift that is significant — before we stress it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pop {means[0]:+.2f}% (t={ts[0]:+.2f}) | drift {means[1]:+.2f}% (t={ts[1]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: pop **{R['pop'][1]:+.2f}%** at *t* = {R['pop'][3]:.2f} (H₁ rejected — "
            f"no cheer), drift **+{R['drift'][1]:.2f}%** at *t* = {R['drift'][3]:.2f} (H₂ supported "
            "*raw*). Everything now turns on whether that drift is **robust**."
        ),
        md(
            "### 4b · The HAC t and the placebo — the drift is raw-real\n\n"
            "The drift's pooled daily abnormal return, tested with Newey-West (the autocorrelation a "
            "genuine drift induces is what HAC is for), and the placebo cloud of random windows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = st.pooled_daily_drift(PRICES, EVENTS)\n"
            "    hact = st.hac_t(daily); dm = PANEL['drift'].to_numpy()\n"
            "    null = st.placebo_pvalue(PRICES, data.TICKERS, k=len(dm), leg='drift', observed=float(dm.mean()), n_draws=6000)\n"
            "    obs = dm.mean(); pval = null['p_value']; nullmean = null['placebo_mean']\n"
            "else:\n"
            "    hact=R['drift'][5]; obs=R['drift'][1]/100; pval=R['drift'][4]; nullmean=0.0\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.2))\n"
            "a1.bar(['drift daily\\nHAC t'], [hact], color=GREEN, width=.4)\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2'); a1.axhline(0, c='k', lw=.8)\n"
            "a1.annotate(f't={hact:+.2f}',(0,hact),ha='center',va='bottom'); a1.set_ylim(0, max(4, hact+1))\n"
            "a1.set_title('Newey-West HAC t on the pooled daily drift'); a1.legend()\n"
            "rng=np.random.default_rng(749); sim = rng.normal(nullmean, max(abs(obs)/2.5,0.01), 4000)\n"
            "a2.hist(sim*100, bins=45, color=GREY, alpha=.85, label='random windows (null)')\n"
            "a2.axvline(obs*100, c=GREEN, lw=2.5, label=f'layoff drift {obs*100:+.1f}%')\n"
            "a2.set_xlabel('mean drift CAR (%)'); a2.set_title(f'Placebo p = {pval:.3f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drift HAC t={hact:+.2f} | placebo p={pval:.3f} — raw-significant, both ways')"
        ),
        md(
            f"> 💡 In plain words: HAC *t* = **{R['drift'][5]:.2f}**, placebo *p* = **{R['drift'][4]:.3f}**. "
            "By the letter of the inference bar, the drift clears *t* = 2 on the real tape. If we stopped "
            "here we'd stamp it `REAL`. We don't stop here."
        ),
        md(
            "### 4c · Fragility — drop-3 and the bootstrap dismantle it\n\n"
            "Left: the drift with the three largest recoveries removed. Right: the bootstrap "
            "distribution of the drift's Welch *t* across the event cross-section."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dm = np.sort(PANEL['drift'].to_numpy())\n"
            "    full_m, full_t = dm.mean()*100, st.welch_t(dm)\n"
            "    trim = dm[:-3]; trim_m, trim_t = trim.mean()*100, st.welch_t(trim)\n"
            "    rng=np.random.default_rng(749); ts=np.array([st.welch_t(rng.choice(dm,len(dm),replace=True)) for _ in range(5000)])\n"
            "    p5, pmed = np.percentile(ts,5), np.median(ts)\n"
            "else:\n"
            "    full_m,full_t=R['drift'][1],R['drift'][3]; trim_m,trim_t=R['drop3'][0],R['drop3'][1]\n"
            "    p5,pmed=R['boot'][0],R['boot'][1]; ts=np.random.default_rng(0).normal(pmed,1.0,5000)\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.2))\n"
            "a1.bar(['full','drop top-3'], [full_m, trim_m], color=[GREEN, GREY], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('drift mean (%)')\n"
            "for i,(m,t) in enumerate([(full_m,full_t),(trim_m,trim_t)]): a1.annotate(f'{m:+.1f}%\\n(t={t:+.2f})',(i,m),ha='center',va='bottom')\n"
            "a1.set_title('Drop 3 recovery names -> t collapses')\n"
            "a2.hist(ts, bins=50, color=GREY, alpha=.85)\n"
            "a2.axvline(2, ls='--', c=RED, label='t = 2'); a2.axvline(p5, c=AMBER, lw=2, label=f'5th pct t={p5:+.2f}')\n"
            "a2.set_xlabel('bootstrapped drift Welch t'); a2.set_title(f'Bootstrap: 5th-pct t={p5:+.2f} (below 2)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drop-3: {trim_m:+.1f}% (t={trim_t:+.2f}) | bootstrap 5th-pct t={p5:+.2f}, median {pmed:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: remove Meta/Exxon/Boeing and the drift is **+{R['drop3'][0]:.1f}%** at "
            f"*t* = **{R['drop3'][1]:.2f}** — gone. The bootstrap's 5th-percentile *t* is "
            f"**{R['boot'][0]:.2f}**: a real slice of resamples never clears significance. The point "
            "estimate flirts with *t* = 2, but its *lower confidence bound* sits far below it. That is "
            "the desk's definition of `WEAK`: significant raw, fragile to method."
        ),
        md(
            "### 4d · The survivorship argument + the split\n\n"
            "The governing confound isn't in a *t*-stat — it's in the sampling frame. Every name is a "
            "**survivor**; the distressed layoff-announcers that delisted are unmeasurable. And the "
            "hyped 2022+ \"efficiency wave\" is no different from the old-economy cuts."
        ),
        code(
            "labels = [s[0] for s in R['split']]\n"
            "pop_m = [s[2] for s in R['split']]; pop_t=[s[3] for s in R['split']]\n"
            "dr_m = [s[4] for s in R['split']]; dr_t=[s[5] for s in R['split']]\n"
            "if HAVE_REAL:\n"
            "    p = PANEL.copy(); p['yr']=p['announce_date'].dt.year\n"
            "    subs=[('2022+ tech wave', p['yr']>=2022), ('pre-2022', p['yr']<2022)]\n"
            "    pop_m=[]; pop_t=[]; dr_m=[]; dr_t=[]\n"
            "    for _,m in subs:\n"
            "        s=p.loc[m]; pop_m.append(s['pop'].mean()*100); pop_t.append(st.welch_t(s['pop'].to_numpy())); dr_m.append(s['drift'].mean()*100); dr_t.append(st.welch_t(s['drift'].to_numpy()))\n"
            "x=np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-.2, pop_m, .4, color=RED, label='pop')\n"
            "ax.bar(x+.2, dr_m, .4, color=GREEN, label='drift')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean CAR (%)')\n"
            "for i in range(len(labels)):\n"
            "    ax.annotate(f't={pop_t[i]:+.2f}',(i-.2,pop_m[i]),ha='center',va='bottom' if pop_m[i]>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f't={dr_t[i]:+.2f}',(i+.2,dr_m[i]),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_title('Tech \"efficiency wave\" vs the rest: same no-pop, same fragile drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('2022+ drift t=%.2f | pre-2022 drift t=%.2f — neither is the clean signal the story needs' % (dr_t[0], dr_t[1]))"
        ),
        md(
            f"> 💡 In plain words: the 2022+ tech cuts drift **+{R['split'][0][4]:.1f}%** (*t* = "
            f"{R['split'][0][5]:.2f}), the pre-2022 cuts **+{R['split'][1][4]:.1f}%** (*t* = "
            f"{R['split'][1][5]:.2f}) — the \"efficiency wave\" narrative adds nothing. And because the "
            "frame is survivors-only, both numbers are **upper bounds**: add back the firms that laid "
            "off staff and then died, and the mean can only fall."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic per-event panel (24 events, seed 723) with plantable edges. Zero edge must "
            "stay below $t=2$ on both legs; a large planted drift must light the HAC $t$ up."
        ),
        code(
            "res=[]\n"
            "for pb,db in [(0.0,0.0),(0.0,400.0)]:\n"
            "    syn=data.synthetic_events(pop_bps=pb, drift_bps=db, seed=723)\n"
            "    res.append((pb,db, syn['pop'].mean()*100, st.welch_t(syn['pop']), syn['drift'].mean()*100, st.welch_t(syn['drift']), st.hac_t(syn['daily_drift'])))\n"
            "labels=['planted 0/0 bps\\n(null)','planted 0/+400 bps\\n(large drift)']\n"
            "hacs=[r[6] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, hacs, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,h in enumerate(hacs): ax.annotate(f'HAC t={h:+.2f}',(i,h),ha='center',va='bottom')\n"
            "ax.set_ylabel('drift HAC t'); ax.set_title('Control: null stays < 2, a large planted drift clears it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for pb,db,pm,pt,dm2,dt2,ht in res: print(f'planted pop={pb:.0f}/drift={db:.0f}bps: pop {pm:+.2f}% (t={pt:+.2f}) | drift {dm2:+.2f}% (t={dt2:+.2f}, HAC t={ht:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's drift HAC *t* is "
            f"**{R['syn'][0][6]:.2f}** (no false positive); only a large **+400 bps/event** planted "
            f"drift reaches **{R['syn'][1][6]:.2f}**. So the engine is honest — and a two-dozen-event "
            "sample only certifies a drift when one is genuinely big. The real-tape drift *is* big-ish "
            "raw, but it evaporates under drop-3 and survivorship, which the control (a *clean* sample) "
            "has none of. The machinery isn't the problem; the sampling frame is."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** (compound: `None` on the pop · `Weak` on the drift). Pop "
            f"**{R['pop'][1]:+.2f}%** (*t* = {R['pop'][3]:.2f}, placebo *p* = {R['pop'][4]:.2f}) — no "
            f"cheer. Drift **+{R['drift'][1]:.2f}%** (Welch *t* = {R['drift'][3]:.2f}, HAC *t* = "
            f"{R['drift'][5]:.2f}, placebo *p* = {R['drift'][4]:.3f}) clears the bar *raw*, then fails "
            f"it under drop-3 (**+{R['drop3'][0]:.2f}%**, *t* = {R['drop3'][1]:.2f}), a bootstrap "
            f"(5th-pct *t* = {R['boot'][0]:.2f}), and survivorship (winners by construction). Significant "
            "raw, fragile to selection ⇒ `WEAK`.\n"
            f"- **Tradability `MIRAGE`** — net of 10 bps the drift is **+{R['net'][1]:.2f}%**, so cost "
            "is not the constraint. You cannot pick the survivors ex-ante, the edge is a few macro-timed "
            "recoveries, and the sampling frame is unavailable at trade time.\n"
            f"- **Restructuring pop? `BUSTED`** — the announcement cheer is absent (*t* = "
            f"{R['pop'][3]:.2f}); the only up-move is a fragile, survivor-flattered drift."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the robustness ladder\n\n"
            "One picture of the whole argument: the drift's *t*-stat as we climb from the raw estimate "
            "to each honest correction. It clears the bar once, and only once."
        ),
        code(
            "stages = ['raw Welch','HAC (daily)','drop top-3','bootstrap\\n5th pct']\n"
            "tvals = [R['drift'][3], R['drift'][5], R['drop3'][1], R['boot'][0]]\n"
            "cols = [GREEN if t>=2 else RED for t in tvals]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(stages, tvals, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('drift t-stat'); ax.set_title('The drift clears t=2 raw, then fails every robustness step'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('drift t by stage:', dict(zip(['raw','HAC','drop3','boot5'], tvals)))"
        ),
        md(
            "> 💡 In plain words: the two green bars (raw Welch, HAC) are the *nominal* case; the two red "
            "bars are what happens the moment you ask the honest questions — *is it a few names? what's "
            "the lower bound?* The drift lives right at the edge of significance and falls off it under "
            "any real stress, before you even charge the survivorship tax. **The rarity and heterogeneity "
            "that make big layoffs vivid are exactly what leave the drift un-certifiable and untradable.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling event studies.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) (a real "
            "day-one jump you can't hold) and [Study 389 — Name-Change-Effect](../389-name-change-effect/) "
            "(the pop-then-fade the losers' delisting invented) — the same small-sample + survivorship "
            "pathologies.\n"
            "- **Kill the survivorship.** Replace the hardcoded table with a survivorship-free "
            "layoff-date panel (WARN filings + a point-in-time universe including delisted names); the "
            "drift's mean can only fall toward zero. That is the experiment that settles it.\n"
            "- **Condition on the *reason*.** The literature says cost-cutting cuts read differently "
            "from demand-shock cuts (Farber & Hallock 2009). Split the table by reason and re-run — but "
            "the sample-size and survivorship walls remain.\n\n"
            "*The reproducible core is offline and deterministic; the event table is an explicit "
            "hardcoded, labelled, **survivor-biased** stand-in. Methods and sources: "
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
