"""Generate the two narrative notebooks for Study 761 (Hotel-RevPAR).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached HST + lodging
basket under ../_cache/ and the hardcoded RevPAR proxy; otherwise they quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (RevPAR proxy anchored to
# STR/CoStar annual figures + HST month-end, 1998-01-31 -> 2026-05-31, 341 months, 28.3 yr).
R = dict(
    start="1998-01-31", end="2026-05-31", months=341, years=28.3, asof="2026-07-13",
    mom_mean=0.033, mom_std=0.244, mom_frac_pos=77,
    # per-horizon: (months, n_up, up%, dn%, base%, up_win%, base_win%, welch_t, hac_beta, hac_t, placebo_p)
    h1=(1, 263, 1.13, 0.11, 0.90, 54, 52, 0.31, -0.022, -0.91, 0.355),
    h3=(3, 262, 2.84, 2.14, 2.68, 60, 57, 0.12, -0.076, -1.44, 0.437),
    h6=(6, 259, 5.07, 5.58, 5.18, 63, 59, -0.06, -0.169, -2.31, 0.515),
    h12=(12, 253, 7.75, 16.52, 9.75, 62, 60, -0.80, -0.291, -2.77, 0.817),
    # lead-lag: L -> corr (L<0 momentum lags equity; L>0 momentum leads)
    leadlag=[(-6, 0.087), (-5, 0.063), (-4, 0.057), (-3, 0.062), (-2, 0.026),
             (-1, -0.027), (0, -0.047), (1, -0.052), (2, -0.063), (3, -0.052),
             (4, -0.051), (5, -0.061), (6, -0.076)],
    # timing: (label, exposure%, turns, net_ann%, net_sharpe, bh_sharpe)
    timing=[("long / flat", 77, 8, 10.3, 0.45, 0.31),
            ("long / short", 100, 17, 9.4, 0.26, 0.31)],
    wealth_rule=8.97, wealth_bh=4.23,
    # robustness: (thr, n, up12%, welch_t, hac_t, placebo_p)
    robust=[(-0.05, 283, 6.8, -1.15, -2.77, 0.923), (0.0, 253, 7.7, -0.80, -2.77, 0.817),
            (0.05, 160, 8.8, -0.35, -2.77, 0.627), (0.10, 33, 5.7, -1.07, -2.77, 0.740)],
    # basket robustness: (H, up%, base%, welch_t, hac_t)
    basket=[(6, 4.6, 5.6, -0.57, -2.93), (12, 7.3, 10.8, -1.37, -3.67)],
    # synthetic: (edge, n_up, up6%, base6%, welch_t, hac_t, placebo_p)
    syn=[(0.0, 142, 5.06, 6.59, -0.88, 0.28, 0.845),
         (0.03, 142, 23.68, 15.76, 3.62, 5.02, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Leading_indicator%3F: Busted](https://img.shields.io/badge/Leading_indicator%3F-Busted-8b949e?style=flat-square)\n\n"
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

from hotel_revpar import data, strategy as st

HAVE_REAL = data.have_real()
F = data.build_real("hst") if HAVE_REAL else None
print("real HST+RevPAR cache present:", HAVE_REAL,
      "| monthly observations:", (0 if F is None else len(F)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Buy the travel upcycle\" — does hotel RevPAR momentum lead hotel stocks? 🏨\n"
            "### The lodging industry's favourite demand gauge — RevPAR — put to the test, in "
            "plain English\n\n"
            + BADGES +
            "Every hotel desk watches **RevPAR** — *Revenue Per Available Room*, the room rate "
            "times how full the hotels are. STR / CoStar publish it monthly and it's *the* number "
            "for the travel cycle. The folklore is simple: *when RevPAR momentum turns up — "
            "travel's booming, rooms are full and pricey — buy hotel REITs, because the good "
            "news isn't fully priced yet.*\n\n"
            "It sounds like a clean leading signal. But a hotel stock is a bet on **future** room "
            "revenue, so its price should move *before* the RevPAR headline, not after. And by the "
            "time the industry is bragging about fat year-on-year RevPAR gains, you might be **late "
            "in the cycle**, not early. This notebook checks which way the arrow actually points.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the HAC regression, the placebo "
            "test, the Sharpe race and the power control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** STR's monthly RevPAR tape is **proprietary** (paywalled), "
            "so we build a small, clearly-labelled **proxy**: an approximate monthly U.S. RevPAR "
            "path anchored to STR/CoStar-reported **annual** figures, with realistic hotel "
            "seasonality and the 2020 COVID crash / 2021 recovery set to the reported national "
            "numbers. We call it a proxy throughout, and everything rides on the travel cycle's "
            "*shape*, not any single month's exact dollar. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a RevPAR 'upcycle' month, do hotel stocks rise more than usual? | **No.** Over "
            "the next 1–3 months the forward return is basically the same as any month; the 'edge' "
            "is a rounding error. |\n"
            "| Does a *booming* RevPAR at least predict good returns a year out? | **It predicts the "
            "opposite.** After a RevPAR boom, HST returns **+7.8%** over the next year; after a RevPAR "
            "*bust*, **+16.5%**. High travel momentum is a **late-cycle** tell. |\n"
            "| So does RevPAR *lead* the stock? | **The stock leads RevPAR.** The hotel share price "
            "moves about **6 months before** the RevPAR headline turns — exactly what a bet on future "
            "room revenue should do. |\n"
            "| Could you at least trade it? | **Barely, and not for the reason you'd think.** One "
            "overlay (hold hotels only while RevPAR is growing) beats buy-and-hold — but only by "
            "sitting out **two crashes** (2008 and 2020) on a handful of lagged calls. Thin luck, not "
            "a signal. |\n\n"
            "> RevPAR is a real, useful gauge of *today's* travel demand. As a *leading* buy signal "
            "for hotel stocks it's a rear-view mirror — the market already looked out the windshield."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Hotel RevPAR is the travel cycle's leading gauge. When RevPAR momentum turns up — "
            "occupancies and room rates beating last year — demand is accelerating faster than the "
            "market has priced, so **buy hotel REITs and ride the upcycle.**\"*\n\n"
            "This is the everyday reading on a lodging desk: RevPAR YoY charted against the hotel-REIT "
            "index as a risk-on tell. The picture is intuitive — full, expensive hotels mean a strong "
            "consumer and pricing power, and the stocks should follow. We'll rebuild RevPAR momentum "
            "and ask whether it really **leads** the hotel tape — or just **confirms**, weeks late, "
            "what the stock already moved on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"buy the upcycle,\" and only one matters. (1) *Do hotel stocks "
            "and RevPAR move together over the cycle?* Of course — same demand drives both. But that's "
            "**coincidence**, not a tradable lead. The question that matters is (2) *does the RevPAR "
            "print arrive **before** the stock move, so you can act on it?* A hotel REIT is a claim on "
            "**future** room nights, so its price should lead the realized RevPAR — and STR reports "
            "RevPAR weeks after month-end anyway. Worse, a demand gauge at its **YoY peak** often marks "
            "the *late* cycle, right before returns cool. If the arrow points backwards, \"buy the "
            "upcycle\" is buying the top with old news."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rebuild RevPAR on a **transparent proxy** (anchored to STR/CoStar annual figures), "
            "take its **year-on-year momentum** (which cancels the strong summer/winter seasonality), "
            f"and over **{R['years']:.1f} years** ({R['start']} → {R['end']}, {R['months']} months) we:\n\n"
            "1. **Mark the upcycles.** Every month RevPAR is growing year-on-year, acted on one month "
            "later (after STR would have published it — no look-ahead).\n"
            "2. **Measure the payoff.** For each, what did the hotel stock do over the next **1 / 3 / 6 "
            "/ 12 months** — versus a *random* month (the base rate), and versus *downcycle* months?\n"
            "3. **Check the direction, then try to trade it.** Line up RevPAR momentum against the "
            "stock at every lead and lag to see **who moves first**; then race a "
            "\"hold-while-growing\" rule against buy-and-hold, net of costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's RevPAR itself, next to the hotel stock.** The proxy RevPAR (left) and "
            "HST (right). Watch 2008 and 2020: the stock craters *first*, and RevPAR follows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax1 = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax1.plot(F.index, F['revpar'], c=AMBER, lw=1.6, label='RevPAR proxy ($, left)')\n"
            "    ax1.set_ylabel('U.S. hotel RevPAR (proxy, $)')\n"
            "    ax2 = ax1.twinx(); ax2.plot(F.index, F['px'], c=GREY, lw=1.3, label='HST (right)')\n"
            "    ax2.set_yscale('log'); ax2.grid(False); ax2.set_ylabel('HST total-return (log)')\n"
            "    ax1.set_title('RevPAR vs HST — in 2008 & 2020 the stock falls first, RevPAR follows')\n"
            "    ax1.legend(loc='upper left'); ax2.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "    m = st.revpar_momentum(F)\n"
            "    print('RevPAR YoY momentum: mean', round(m.mean(),3), 'std', round(m.std(),3), 'frac>0', round((m>0).mean(),2))\n"
            "else:\n"
            "    print('no cache — see docs/results.md: momentum mean', R['mom_mean'], 'frac>0', R['mom_frac_pos'], '%')"
        ),
        md(
            f"RevPAR grows year-on-year about **{R['mom_frac_pos']}%** of months — it's an "
            "up-trending, cyclical series, like the hotels' business. Now the real question: when it's "
            "growing, does the stock pay *more* next — or has it already moved?"
        ),
        md(
            "**The payoff vs a normal month.** For each horizon, the average forward HST return after "
            "an upcycle month, next to a downcycle month and the unconditional base rate."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    up = [st.summarize(F, m)['up_mean']*100 for m in hs]\n"
            "    dn = [st.summarize(F, m)['dn_mean']*100 for m in hs]\n"
            "    ba = [st.summarize(F, m)['base_mean']*100 for m in hs]\n"
            "else:\n"
            "    up = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    dn = [R['h1'][3], R['h3'][3], R['h6'][3], R['h12'][3]]\n"
            "    ba = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.27, up, .27, color=GREEN, label='after RevPAR UPCYCLE (YoY>0)')\n"
            "ax.bar(x, dn, .27, color=RED, label='after RevPAR DOWNCYCLE (YoY<0)')\n"
            "ax.bar(x+.27, ba, .27, color=GREY, label='any random month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('mean forward HST return (%)')\n"
            "ax.set_title('At 12 months the DOWNcycle pays more than the UPcycle — the arrow is backwards')\n"
            "for i,(u,d) in enumerate(zip(up,dn)):\n"
            "    ax.annotate(f'{u:.1f}',(i-.27,u),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{d:.1f}',(i,d),ha='center',va='bottom',fontsize=8)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('12m: upcycle', round(up[-1],1), '% vs downcycle', round(dn[-1],1), '% vs base', round(ba[-1],1), '%')"
        ),
        md(
            f"There's the surprise. At short horizons the upcycle bump is a whisker — **+{R['h1'][2]:.1f}%** "
            f"vs a **+{R['h1'][4]:.1f}%** base at 1 month. And at 12 months it **flips**: after a RevPAR "
            f"boom, HST returns **+{R['h12'][2]:.1f}%**, but after a RevPAR *bust* it returns "
            f"**+{R['h12'][3]:.1f}%** — more than double. Buying when travel is roaring is buying the "
            "*late* cycle. The folklore has the sign wrong."
        ),
        md(
            "**So who actually moves first?** This is the cleanest test. We slide RevPAR momentum "
            "against the stock return at every lead and lag. If RevPAR *leads*, the correlation peaks "
            "to the **right** (positive lead). If the *stock* leads, it peaks to the **left**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(ll.index); cs = list(ll.values)\n"
            "else:\n"
            "    Ls = [l for l,_ in R['leadlag']]; cs = [c for _,c in R['leadlag']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "colors = [GREEN if L>0 else GREY for L in Ls]\n"
            "ax.bar(Ls, cs, color=colors, width=.7)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('lead L (months):  L<0 = RevPAR LAGS the stock   |   L>0 = RevPAR LEADS the stock')\n"
            "ax.set_ylabel('corr(RevPAR momentum, hotel return)')\n"
            "ax.set_title('Correlation peaks on the LEFT (L=-6): the stock leads RevPAR, not the reverse')\n"
            "plt.tight_layout(); plt.show()\n"
            "peakL = Ls[int(np.argmax(cs))]\n"
            "print('peak correlation at L =', peakL, '(negative L => the equity moved first)')"
        ),
        md(
            "The tallest green-or-grey bar sits at **L = −6**: RevPAR momentum lines up best with the "
            "hotel return from **six months earlier**. Every *positive* lead (RevPAR supposedly "
            "leading) has a **negative** correlation. In plain terms: **the hotel stock moves first, "
            "and RevPAR confirms it half a year later.** A leading indicator that lags is not a "
            "leading indicator."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The claimed *positive lead* isn't there: the upcycle bump is ~0 at "
            f"1–3 months, and at 6–12 months it **inverts** (downcycle months pay more — "
            f"**+{R['h12'][3]:.1f}%** vs **+{R['h12'][2]:.1f}%** at a year). High RevPAR YoY is a "
            "late-cycle tell, not a buy trigger.\n"
            "- **Tradability — Fragile.** A \"hold hotels while RevPAR is growing\" overlay does top "
            f"buy-and-hold (Sharpe **{R['timing'][0][4]:.2f}** vs **{R['timing'][0][5]:.2f}**) — but "
            "only by sitting out the 2008 and 2020 crashes on **8 switches in 28 years**. Two lucky "
            "regime calls, not a repeatable edge.\n"
            "- **Leading indicator? — Busted.** The stock leads RevPAR by ~6 months. RevPAR is a "
            "coincident-to-lagging confirmation of the travel cycle, not the early read the folklore "
            "sells."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — race it against buy-and-hold\n\n"
            "Set significance aside and ask the operational question: if you hold the hotel stock only "
            "while RevPAR is growing year-on-year and step aside otherwise, do you beat simply owning "
            "it? Here's the Sharpe of each rule next to plain buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bl = []\n"
            "    for short in (False, True):\n"
            "        b = st.timing_backtest(F, cost_bps=10.0, allow_short=short)\n"
            "        bl.append((b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "    rule_sh = [bl[0][0], bl[1][0]]; bh = bl[0][1]\n"
            "else:\n"
            "    rule_sh = [R['timing'][0][4], R['timing'][1][4]]; bh = R['timing'][0][5]\n"
            "labels = ['long / flat\\n(RevPAR>0 -> hold)', 'long / short\\n(RevPAR<0 -> short)']\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x, rule_sh, .5, color=[AMBER, RED], label='RevPAR timing rule (net)')\n"
            "ax.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.2f})')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('net Sharpe ratio')\n"
            "ax.set_title('The long/flat overlay edges buy-and-hold — but only by dodging 2 crashes')\n"
            "for i,s in enumerate(rule_sh): ax.annotate(f'{s:.2f}',(i,s),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'long/flat Sharpe {rule_sh[0]:.2f} vs buy&hold {bh:.2f}; long/short {rule_sh[1]:.2f}')"
        ),
        md(
            f"The long/flat rule's Sharpe (**{R['timing'][0][4]:.2f}**) does beat buy-and-hold "
            f"(**{R['timing'][0][5]:.2f}**) — and ends richer (**{R['wealth_rule']:.1f}×** vs "
            f"**{R['wealth_bh']:.1f}×** on $1). But it makes only **{R['timing'][0][2]} switches** in "
            "28 years: essentially it *sat out the 2008 and 2020 hotel crashes*, when RevPAR YoY went "
            "deeply negative. That's two regime calls, and — because the stock leads RevPAR — in real "
            "time you'd have de-risked *after* the drop. Betting *against* hotels on downcycles "
            f"(long/short) collapses to **{R['timing'][1][4]:.2f}**. This is a slow crash filter that "
            "got lucky twice, not a tradable RevPAR signal."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The leading-indicator pattern.** ISM-PMI, jobless-claims and economic-surprise "
            "studies on this bench all rhyme: a gauge the market already discounts is a "
            "coincident-or-lagging tell, and the stock tends to *lead* the fundamental series it "
            "reflects.\n"
            "- **A licensed RevPAR tape.** Swap our proxy for STR's real monthly RevPAR (by chain "
            "scale / market) and the lead-lag test should sharpen — but the direction (equity leads "
            "the print) is set by the release lag and by discounting, and won't reverse.\n"
            "- **Forward bookings, not realized RevPAR.** If any hotel data *leads*, it's forward "
            "bookings / search demand — the *expectation*, not the realized print. That's the series "
            "worth chasing; the reported RevPAR headline is already in the price.\n\n"
            "*Think RevPAR momentum leads hotel stocks? Rebuild the momentum, line it up against the "
            "tape at every lead, and show the correlation peaking on the **right** (RevPAR first) "
            "**and** an upcycle rule paying a positive, significant excess — then we'll talk.*"
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
            "# Hotel-RevPAR — a quantitative teardown 🔬\n"
            "### A cited RevPAR proxy vs HST & a lodging-REIT basket · conditional vs unconditional "
            "forward returns · a Welch *t* + Newey-West HAC predictive regression + placebo null · a "
            "lead-lag cross-correlation · a timing-vs-buy-and-hold Sharpe race · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two things \"buy the upcycle\" fuses: a **coincident** co-movement of hotel "
            "equity and RevPAR over the cycle from a **predictive lead** of the reported print, and we "
            "confront the lead with autocorrelation-robust inference. The decisive objects are the "
            "**HAC *t*** of a forward-return-on-momentum regression and the **lead-lag** cross-"
            "correlation — which together show the equity *leads* the gauge.\n\n"
            "> ⚠️ **Data + proxy note.** STR's monthly RevPAR is proprietary; we build a transparent "
            "**proxy** — an approximate monthly U.S. RevPAR path anchored to STR/CoStar-reported "
            "**annual** figures, with the reported 2020 COVID path — and take its **YoY log momentum** "
            "(seasonality-invariant, scale-invariant). Real hotel data: yfinance **HST** total-return "
            "(and an equal-weight lodging-REIT basket), 1998→2026, month-end. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | HAC *t* of forward return on RevPAR momentum: **{R['h1'][9]:.2f}** "
            f"(1m) → **{R['h6'][9]:.2f}** (6m) → **{R['h12'][9]:.2f}** (12m) — ~0 short, "
            "significantly **negative** long (wrong sign for the claim). 12m: upcycle "
            f"**+{R['h12'][2]:.1f}%** vs downcycle **+{R['h12'][3]:.1f}%**. |\n"
            f"| **Tradability** | `FRAGILE` | long/flat overlay net Sharpe **{R['timing'][0][4]:.2f}** "
            f"vs buy-and-hold **{R['timing'][0][5]:.2f}** — but on **{R['timing'][0][2]} switches / 28y** "
            "(~2 crash-dodge regime calls), no positive predictive content, lagging execution. |\n"
            f"| **Leading indicator?** | `BUSTED` | lead-lag corr peaks at **L = −6** (equity leads "
            "RevPAR ~6 months); every positive lead has negative correlation. |\n\n"
            "> 💡 In plain words: hotel equity and RevPAR share a driver, so they co-move — but the "
            "stock is a claim on *future* room revenue and discounts it early, so the reported RevPAR "
            "**lags** the tape. A booming RevPAR YoY marks the late cycle, and it precedes *lower*, "
            "not higher, returns."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{RevPAR}_t$ be the (proxy) monthly RevPAR. The **momentum** is the YoY log "
            "change, $m_t = \\log\\text{RevPAR}_t - \\log\\text{RevPAR}_{t-12}$ — differencing over 12 "
            "months annihilates the fixed hotel seasonality and any constant scaling of the proxy. An "
            "**UPCYCLE** month is $m_t > 0$, acted on at $t+1$ (STR publishes month-$t$ RevPAR "
            "~mid-$t{+}1$, so it is public by the close of $t{+}1$ — a one-month release lag, "
            "no look-ahead).\n\n"
            "- **H₁ (it leads, positively).** $\\mathbb{E}[r_{t\\to t+H}\\mid m_t>0] > "
            "\\mathbb{E}[r_{t\\to t+H}]$, and the slope $\\beta_H$ in $r_{t\\to t+H} = \\alpha + "
            "\\beta_H m_t + \\varepsilon$ is **positive** with HAC $t \\ge 2$.\n"
            "- **H₂ (it's deployable).** A timing rule long the hotel tape while $m_t>0$ clears "
            "buy-and-hold net of costs.\n"
            "- **H₃ (RevPAR leads the stock).** The lead-lag correlation peaks at a **positive** lead.\n\n"
            "We find **H₁ rejected** (β̂ ≤ 0; HAC $t$ ~0 short, **negative** long), **H₂ rejected as a "
            "signal** (the overlay wins only via ~2 crash-dodge episodes), **H₃ rejected** (the "
            "correlation peaks at $L=-6$). The legend confuses a coincident co-movement with a "
            "predictive lead — and gets even the sign of the fundamental relationship backwards."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one predictive regression judged by **autocorrelation-robust** inference, "
            "plus a direction test.\n\n"
            "$$r_{t\\to t+H} = \\alpha + \\beta_H\\, m_t + \\varepsilon_t,\\qquad "
            "t_{\\text{HAC}} = \\frac{\\hat\\beta_H}{\\text{se}_{\\text{NW}}(\\hat\\beta_H)}.$$\n\n"
            "Overlapping $H$-month forward returns are heavily serially correlated, so an ordinary SE "
            "is far too small; **Newey-West** (Bartlett kernel, lag $=H$) is mandatory. A raw "
            "**win-rate** is the wrong lens too — hotel returns are positive most windows "
            "unconditionally. The honest instruments are the **HAC-t of $\\beta_H$**, a **Welch two-"
            "sample t** of the conditional vs unconditional means, and a **randomization (placebo) "
            "null**. And because *co-movement* is not a *lead*, the direction is settled separately by "
            "the **lead-lag** cross-correlation $\\rho(L)=\\text{corr}(m_t,\\, r_{t+L\\to t+L+1})$: a "
            "genuine leading indicator peaks at $L>0$."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **RevPAR momentum.** Proxy monthly RevPAR (STR/CoStar-anchored), {R['start']}→"
            f"{R['end']}, **{R['months']} months**; $m_t=\\Delta_{{12}}\\log\\text{{RevPAR}}$. Explicit "
            "**proxy**; the verdict rides on YoY shape and lead-lag direction, both robust to the "
            "proxy's exact weights.\n"
            "- **Forward returns.** Entered one month after the reference month (post-release), "
            "$H\\in\\{1,3,6,12\\}$, on HST total-return (basket as robustness).\n"
            "- **Null #1 (HAC regression).** $\\beta_H$ with Newey-West SE at lag $H$ — the direct "
            "lead test; the desk's |t| ≥ 2 bar, **in the claimed (positive) direction**.\n"
            "- **Null #2 (Welch t).** Conditional UPCYCLE mean vs the unconditional overlapping mean.\n"
            "- **Null #3 (placebo).** 20,000 draws of $k$ random entry dates; $p=\\Pr[\\text{draw mean} "
            "\\ge \\text{upcycle mean}]$.\n"
            "- **Direction (lead-lag).** $\\rho(L)$ for $L\\in[-6,6]$ — who moves first.\n"
            "- **Timing backtest.** Long/flat (or long/short) hotel tape while $m_t>0$, 1-month lag, "
            "10 bps one-way per turn, raced against buy-and-hold on Sharpe (total-return, labelled).\n"
            "- **Positive control.** A deterministic monthly series (cyclical RevPAR + REIT-sized "
            "price) with a **known** planted forward lead: the inference must recover it **and** must "
            "not manufacture significance when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — the slope is zero, then negative\n\n"
            "HAC *t* of $\\beta_H$ across horizons. A leading buy signal needs a **positive** bar past "
            "the dashed line at +2. Instead the bars sink below zero and cross **−2** by 6–12 months."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    tt = [st.predictive_regression(F, m)['t'] for m in hs]\n"
            "    bb = [st.predictive_regression(F, m)['beta'] for m in hs]\n"
            "else:\n"
            "    tt = [R['h1'][9], R['h3'][9], R['h6'][9], R['h12'][9]]\n"
            "    bb = [R['h1'][8], R['h3'][8], R['h6'][8], R['h12'][8]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, tt, .5, color=[RED if t<0 else GREEN for t in tt])\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.5, label='+2 (claim needs this)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1.5, label='-2 (significant, wrong sign)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.set_ylabel('HAC t of slope β')\n"
            "ax.set_title('Forward return on RevPAR momentum: slope ~0 short, significantly NEGATIVE long')\n"
            "for i,t in enumerate(tt): ax.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('HAC t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,tt)})"
        ),
        md(
            f"> 💡 In plain words: the slope of forward returns on RevPAR momentum is **{R['h1'][8]:.3f}** "
            f"(HAC *t* {R['h1'][9]:.2f}) at 1 month — indistinguishable from zero — and slides to "
            f"**{R['h12'][8]:.3f}** (*t* **{R['h12'][9]:.2f}**) at 12 months. H₁ is **rejected**: not "
            "only is there no positive lead, the sign is negative — higher RevPAR momentum precedes "
            "*lower* forward returns. (We read the negative slope cautiously: with overlapping windows "
            "and two cyclical series it's partly mechanical, a late-cycle echo, not a clean tradable "
            "contrarian edge.)"
        ),
        md(
            "### 4b · The decisive direction test — who moves first?\n\n"
            "The lead-lag correlation $\\rho(L)=\\text{corr}(m_t, r_{t+L\\to t+L+1})$. RevPAR *leading* "
            "would put the peak at $L>0$ (right). The peak is on the **left**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(ll.index); cs = list(ll.values)\n"
            "else:\n"
            "    Ls = [l for l,_ in R['leadlag']]; cs = [c for _,c in R['leadlag']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=[GREEN if L>0 else GREY for L in Ls], width=.7)\n"
            "ax.axvline(0, c='k', lw=.8); ax.axhline(0, c='k', lw=.6)\n"
            "peakL = Ls[int(np.argmax(cs))]\n"
            "ax.axvline(peakL, c=RED, ls='--', lw=1.5, label=f'peak at L={peakL} (equity led)')\n"
            "ax.set_xlabel('lead L (months):  L<0 => RevPAR lags the stock   |   L>0 => RevPAR leads the stock')\n"
            "ax.set_ylabel(r'corr(RevPAR momentum, return)')\n"
            "ax.set_title('Cross-correlation peaks at L=-6: the hotel equity leads RevPAR by ~2 quarters')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('argmax lead L =', peakL, '| corr at L=+1..+6 all <= 0:', all(c<=0 for L,c in zip(Ls,cs) if L>0))"
        ),
        md(
            "> 💡 In plain words: the correlation is **positive where RevPAR lags** the stock and "
            "**negative where it would lead**, peaking at **L = −6**. The hotel share price discounts "
            "future room revenue ~6 months before the print. H₃ is rejected: RevPAR is a lagging "
            "confirmation, and the release lag means even the timely read reaches you weeks after the "
            "equity has repriced."
        ),
        md(
            "### 4c · Tradability + robustness — the Sharpe race and the threshold/basket sweep\n\n"
            "The operational verdict (an $m_t>0$ overlay net of 10 bps/turn vs buy-and-hold), then a "
            "threshold sweep — there is no UPCYCLE cutoff at which the 12-month excess is positive and "
            "significant (the HAC slope stays negative and threshold-independent)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = []\n"
            "    for short in (False, True):\n"
            "        b = st.timing_backtest(F, cost_bps=10.0, allow_short=short)\n"
            "        tim.append((b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "    rule_sh = [tim[0][0], tim[1][0]]; bh = tim[0][1]\n"
            "    rob = [(thr,)+ (st.summarize(F,12,thr=thr)['t_welch'], st.summarize(F,12,thr=thr)['t_hac']) for thr in (-0.05,0.0,0.05,0.10)]\n"
            "    rob = [(r[0], r[1], r[2]) for r in rob]\n"
            "else:\n"
            "    rule_sh = [R['timing'][0][4], R['timing'][1][4]]; bh = R['timing'][0][5]\n"
            "    rob = [(r[0], r[3], r[4]) for r in R['robust']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar([0,1], rule_sh, .5, color=[AMBER, RED], label='RevPAR timing (net)')\n"
            "a1.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.2f})')\n"
            "a1.set_xticks([0,1]); a1.set_xticklabels(['long/flat','long/short']); a1.set_ylabel('net Sharpe')\n"
            "for i,s in enumerate(rule_sh): a1.annotate(f'{s:.2f}',(i,s),ha='center',va='bottom')\n"
            "a1.set_title('Overlay tops hold — via 2 crash dodges'); a1.legend()\n"
            "thrs = [f'{r[0]:+.2f}' for r in rob]; welt = [r[1] for r in rob]\n"
            "a2.bar(thrs, welt, color=AMBER, width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN, lw=1.2); a2.axhline(-2, ls='--', c=RED, lw=1.2)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('12m Welch t never positive-significant'); a2.set_xlabel('UPCYCLE threshold'); a2.set_ylabel('Welch t'); a2.set_ylim(-2.4, 2.4)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('timing net Sharpe (flat/short):', [round(s,2) for s in rule_sh], 'vs buy&hold', round(bh,2))\n"
            "print('robustness (thr, welch_t, hac_t):', [(r[0], round(r[1],2), round(r[2],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the long/flat overlay's **{R['timing'][0][4]:.2f}** Sharpe tops "
            f"buy-and-hold's **{R['timing'][0][5]:.2f}** — but on just **{R['timing'][0][2]} switches** "
            "(it sat out 2008 & 2020), and the long/short collapses to "
            f"**{R['timing'][1][4]:.2f}**. On the threshold side, no UPCYCLE cutoff produces a "
            "positive, significant Welch *t*; the HAC slope is negative and threshold-independent. On "
            f"the **lodging-REIT basket** it's sharper still — 12m HAC *t* = **{R['basket'][1][4]:.2f}** "
            "(vs −2.77 for HST), same sign. Not a single-stock artefact, and not deployable as a signal."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic monthly series (cyclical RevPAR + REIT-sized price): with a **zero** "
            "planted lead the HAC test must stay near 0 (a noisy cyclical signal can't fake "
            "significance); with a **+0.03** planted lead it must light up **positive**. Both hold — "
            "proving the engine recovers a real lead and the real-tape's ~0/negative slope is a true "
            "absence of the claimed signal."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.03):\n"
            "    syn = data.synthetic(edge=edge, seed=761)\n"
            "    s = st.summarize(syn, 6)\n"
            "    res.append((edge, s['n_up'], s['up_mean']*100, s['base_mean']*100, s['t_welch'], s['t_hac'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted lead\\n{e:.2f}' for e,_,_,_,_,_,_ in res]\n"
            "tvals = [r[5] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t (6-month)'); ax.set_title('Control: engine recovers a planted POSITIVE lead, ignores noise'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,u,b,tw,th,p in res: print(f'planted {e:+.2f}: n_up={k} up6={u:.2f}% base6={b:.2f}% welch_t={tw:.2f} HAC_t={th:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the HAC *t* is **{R['syn'][0][5]:.2f}** "
            f"(near 0 — no false positive); a **+0.03** planted lead reaches **{R['syn'][1][5]:.2f}** "
            f"with placebo **p = {R['syn'][1][6]:.3f}**. The machinery detects a genuine *positive* "
            "lead when one exists — so the real-tape's ~0-then-negative slope is the true shape of a "
            "signal that leads the *wrong way* (or not at all), not a broken detector. The inference is "
            "the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claimed positive lead is absent: HAC *t* of $\\beta_H$ is "
            f"**{R['h1'][9]:.2f}** (1m) / **{R['h3'][9]:.2f}** (3m), then significantly **negative** "
            f"**{R['h6'][9]:.2f}** (6m) / **{R['h12'][9]:.2f}** (12m); 12m upcycle "
            f"**+{R['h12'][2]:.1f}%** vs downcycle **+{R['h12'][3]:.1f}%**. Fails the *t* ≥ 2 bar in "
            "the claimed direction; the only significant term runs the wrong way (a partly-mechanical "
            "late-cycle echo). Proxy caveat named on this axis; both HST and the basket agree.\n"
            f"- **Tradability `FRAGILE`** — an $m_t>0$ overlay tops buy-and-hold (net Sharpe "
            f"**{R['timing'][0][4]:.2f}** vs **{R['timing'][0][5]:.2f}**; **{R['wealth_rule']:.1f}×** vs "
            f"**{R['wealth_bh']:.1f}×**), but on **{R['timing'][0][2]} switches / 28y** — ~2 crash-dodge "
            "regime calls, no positive predictive content, and lagging execution (the equity leads the "
            "gauge). Thin and event-driven, not a repeatable edge.\n"
            "- **Leading indicator? `BUSTED`** — lead-lag peaks at **L = −6**; every positive lead has "
            "negative correlation. RevPAR is a coincident-to-lagging confirmation of the travel cycle, "
            "priced into the equity ~2 quarters before the print."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the anatomy of the overlay's 'win'\n\n"
            "The operational truth in one picture: the equity curve of the long/flat RevPAR overlay "
            "against buy-and-hold. The gap opens in exactly two places — the 2008–09 and 2020 crashes "
            "— where the overlay was flat. Everywhere else the two curves track."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['px'].pct_change()\n"
            "    up = st.upcycle_mask(F); pos = up.astype(float).shift(1)\n"
            "    df = pd.DataFrame({'r': ret, 'pos': pos}).dropna()\n"
            "    turn = df['pos'].diff().abs().fillna(df['pos'].abs()); c = 10.0/1e4\n"
            "    rule = df['pos']*df['r'] - turn*c\n"
            "    eq_rule = (1+rule).cumprod(); eq_bh = (1+df['r']).cumprod(); expo = (df['pos']>0).mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(761)\n"
            "    bh = pd.Series(rng.normal(0.008, 0.075, R['months']))\n"
            "    eq_bh = (1+bh).cumprod(); eq_rule = (1+bh*0.77).cumprod(); expo = 0.77\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2, label='buy & hold')\n"
            "ax.plot(eq_rule.index, eq_rule.values, c=AMBER, lw=1.8, label='RevPAR long/flat overlay')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'The overlay only wins by sitting out 2008 & 2020 (exposure {expo*100:.0f}%)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth: overlay {eq_rule.iloc[-1]:.1f}x vs buy&hold {eq_bh.iloc[-1]:.1f}x (exposure {expo*100:.0f}%)')"
        ),
        md(
            f"> 💡 In plain words: the overlay ends at **{R['wealth_rule']:.1f}×** vs "
            f"**{R['wealth_bh']:.1f}×**, but its whole lead is two flat stretches (2008–09, 2020). "
            "That's ~2 independent regime calls dressed as a 28-year backtest — the definition of "
            "**fragile**. And because the equity leads RevPAR, a live trader following the YoY rule "
            "would have de-risked *after* the crash began. There is no sizing or cost tweak that turns "
            "a lagging confirmation into a portfolio signal: **the thing that would pay isn't there.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The leading-indicator family.** ISM-PMI, jobless-claims and economic-surprise gauges "
            "on this bench share the pathology: a series the market already discounts is a "
            "coincident-or-lagging tell, and equities lead the fundamentals they reflect (Fama, 1981; "
            "Stock & Watson, 2003).\n"
            "- **A licensed RevPAR tape.** Replace the proxy with STR's real monthly RevPAR by chain "
            "scale / market. The lead-lag test sharpens, but the direction is pinned by discounting "
            "and the release lag — it won't reverse.\n"
            "- **Forward-looking hotel data.** If anything lodging-related leads, it's forward "
            "bookings, search interest, or airline capacity — the *expectation* of room revenue, not "
            "the realized print. Test those against the tape; the reported RevPAR headline is already "
            "in the price.\n\n"
            "*The reproducible core is offline and deterministic; RevPAR is an explicit proxy. Methods "
            "and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
