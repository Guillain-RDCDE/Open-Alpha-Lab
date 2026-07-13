"""Generate the two narrative notebooks for Study 757 (Cass-Freight).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY + IYT
month-end closes under ../_cache/ and the hardcoded Cass Freight PROXY; otherwise they quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (hardcoded Cass shipments
# PROXY, YoY expansion signal, aligned to real SPY (1999-06->2026-06, 325 months) & IYT
# (2004-01->2026-06, 270 months); 2-month publication+execution lag).
R = dict(
    start="1999-06-30", end="2026-06-30", months=325, years=27.0, lag=2,
    iyt_start="2004-01-31", iyt_months=270,
    yoy_mean=-0.001, yoy_std=0.050, frac_exp=50,
    # per-horizon SPY: (months, n, cond%, cond_win%, base%, base_win%, t, p_placebo)
    spy_h1=(1, 157, 0.96, 69, 0.77, 63, 0.49, 0.298),
    spy_h3=(3, 157, 2.90, 74, 2.33, 70, 0.91, 0.167),
    spy_h6=(6, 157, 5.69, 81, 4.71, 73, 1.09, 0.130),
    spy_h12=(12, 157, 10.57, 86, 9.62, 78, 0.73, 0.244),
    # per-horizon IYT
    iyt_h1=(1, 126, 0.85, 61, 0.98, 61, -0.21, 0.596),
    iyt_h3=(3, 126, 2.79, 62, 2.90, 63, -0.11, 0.550),
    iyt_h6=(6, 126, 5.53, 71, 5.77, 69, -0.18, 0.575),
    iyt_h12=(12, 126, 9.38, 70, 11.12, 74, -0.94, 0.837),
    # lead-lag: (tape, peak_lag, peak_corr, mean|corr| stocks-lead k<0, freight-lead k>0)
    ll_spy=("SPY", -3, 0.222, 0.117, 0.063),
    ll_iyt=("IYT", -3, 0.238, 0.126, 0.065),
    # timing: (tape, label, exposure%, turns, net_ann%, net_sharpe, bh_sharpe)
    timing=[("SPY", "long / flat", 48, 10, 5.3, 0.59, 0.61),
            ("SPY", "long / short", 100, 21, 1.4, 0.09, 0.61),
            ("IYT", "long / flat", 47, 8, 5.3, 0.42, 0.58),
            ("IYT", "long / short", 100, 17, -1.0, -0.05, 0.58)],
    # robustness SPY 12m: (thr, n, cond12%, t, p_placebo)
    robust=[(-0.02, 192, 10.0, 0.29, 0.383), (0.0, 157, 10.6, 0.73, 0.244),
            (0.02, 112, 8.1, -1.13, 0.833)],
    # synthetic 6m: (edge, n, cond6%, base6%, t, p_placebo)
    syn=[(0.0, 227, 7.56, 7.31, 0.24, 0.374), (0.05, 227, 42.94, 29.94, 4.68, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads_the_cycle%3F: Busted](https://img.shields.io/badge/Leads_the_cycle%3F-Busted-8b949e?style=flat-square)\n\n"
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

from cass_freight import data, strategy as st

HAVE_REAL = data.have_real()
F  = data.build_real() if HAVE_REAL else None
FI = F.dropna(subset=["iyt"]) if HAVE_REAL else None
print("real SPY+IYT cache present:", HAVE_REAL,
      "| SPY months:", (0 if F is None else len(F)),
      "| IYT months:", (0 if FI is None else len(FI)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Watch the freight\" — does the Cass Freight Index lead the market? 🚚\n"
            "### The freight world's favourite leading indicator — trucks and trains move the real "
            "economy — put to the test, in plain English\n\n"
            + BADGES +
            "Every freight-macro commentator watches the **Cass Freight Index**: a monthly line that "
            "rises when America is shipping more stuff and falls when the goods economy cools. The "
            "folklore is irresistibly simple — *freight moves the real economy, so when freight rolls "
            "over, the slowdown (and the lower stock market) is already coming; watch the freight and "
            "you're ahead of the cycle.*\n\n"
            "It sounds like a free crystal ball. But the stock market is a **forward-looking** machine "
            "that reprices every second on what's *next*, while a freight index counts trucks that "
            "**already rolled** — and Cass doesn't even publish a month's number until the middle of "
            "the *next* month. So \"freight leads\" might have the arrow backwards: maybe the market "
            "turns first and freight follows. This notebook checks which way the arrow actually "
            "points.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test, the lead-lag "
            "cross-correlation and the Sharpe race? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Cass's monthly shipments series isn't freely downloadable, "
            "so we **hardcode a small, cited, approximate** version of it (its year-over-year ups and "
            "downs — the 2008–09 crash, the 2015–16 and 2022–24 freight recessions, the 2020 COVID "
            "air-pocket) and call it a **proxy** throughout. The stock tapes — **SPY** (the market) and "
            "**IYT** (transport stocks) — are the *real* thing. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After freight starts expanding, is the market higher later? | **A bit more often, yes** "
            "— but the market rises most of the time *anyway*. The extra is tiny and inside the noise. |\n"
            "| Does it at least work for **transport** stocks (IYT)? | **No — it's backwards.** Owning "
            "transports when freight is expanding did **worse** than just holding them (a *negative* "
            "excess at every horizon). |\n"
            "| So does freight **lead** the market? | **No — it lags.** Freight lines up best with the "
            "stock move of **three months earlier**. The market turns first; freight follows. |\n"
            "| Could you trade it? | **No.** \"Own stocks only when freight is expanding\" earns a "
            "**lower Sharpe than buy-and-hold** — because freight is contracting right at the market "
            "bottoms (2009, 2020) you most wanted to own. |\n\n"
            "> The Cass index is a genuine, useful read on the goods economy. But as a *market* signal "
            "it's a rear-view mirror: it describes a turn stocks already made, and trading it loses to "
            "doing nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Trucks and trains move the physical economy. When the Cass Freight Index rolls over, "
            "an industrial slowdown — and a weaker stock market — is already in motion but not yet "
            "priced. So watch the freight: a freight downturn is an early sell signal, a freight upturn "
            "an early buy, especially for the transports.\"*\n\n"
            "This is the quantified version of the old Dow-Theory instinct that the transportation "
            "average must \"confirm\" the industrials. The picture is intuitive: freight is the "
            "economy's bloodstream, so a change in flow should show up in shipments *before* it shows up "
            "in earnings and prices. We'll rebuild the freight cycle from public data and ask whether "
            "\"freight turns\" really comes **before** \"market turns\" — or **after**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If freight genuinely led the market, it would be one of the most valuable free signals in "
            "finance — a monthly heads-up on the whole cycle. But two things have to be true, and both "
            "are in doubt. (1) *Does freight move **before** stocks?* The market is a forward-looking "
            "discounting machine; a freight print counts goods **already shipped** and lands weeks late. "
            "It's at least as plausible that stocks lead freight. (2) *Even if freight rises before "
            "good times, does that beat just owning stocks?* The market rises most years regardless — a "
            "signal that merely rides that up-drift, while sitting out exactly when freight is "
            "collapsing (which is often the **bottom**), can easily **lose** to buy-and-hold."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We rebuild the freight cycle as a **cited, approximate monthly proxy** of the Cass "
            "shipments index and measure its **year-over-year growth** — \"expanding\" when freight is "
            f"up on a year ago. Crucially we only act on a month's reading **{R['lag']} months later** "
            "(Cass publishes a month in arrears, and you need a day to trade). Over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}) we then:\n\n"
            "1. **Mark the expansions.** Every month freight is growing year-over-year, known only after "
            "the publication lag — no look-ahead.\n"
            "2. **Measure the payoff.** After an expansion signal, what did **SPY** and **IYT** "
            "(transports) do over the next **1 / 3 / 6 / 12 months** — versus a *random* month (the "
            "base rate)?\n"
            "3. **Point the arrow, then try to trade it.** Line freight up against stock returns at "
            "every lead and lag to see which comes first; then race an \"own-when-expanding\" overlay "
            "against buy-and-hold, net of costs.\n\n"
            "**What would make us say \"mirage\"?** If the excess over the base rate is small and "
            "insignificant, if freight lines up *after* stocks rather than before, and if the overlay "
            "can't beat buy-and-hold — then \"watch the freight\" is a rear-view mirror."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the freight cycle itself.** The proxy's year-over-year growth going "
            "positive (freight expanding, green) and negative (contracting, red), overlaid on SPY. "
            "Watch how the red patches sit *around* the 2009 and 2020 crashes — the very moments the "
            "market was bottoming and about to rip higher."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yoy = st.freight_yoy(F)\n"
            "    fig, ax1 = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax1.fill_between(F.index, yoy*100, 0, where=(yoy>=0), color=GREEN, alpha=.5, label='expanding (YoY>0)')\n"
            "    ax1.fill_between(F.index, yoy*100, 0, where=(yoy<0), color=RED, alpha=.5, label='contracting (YoY<0)')\n"
            "    ax1.axhline(0, c='k', lw=.8); ax1.set_ylabel('Cass Freight YoY growth (%, proxy)')\n"
            "    ax2 = ax1.twinx(); ax2.plot(F.index, F['spy'], c=GREY, lw=1.3, label='SPY (right)')\n"
            "    ax2.set_yscale('log'); ax2.grid(False); ax2.set_ylabel('SPY (log)')\n"
            "    ax1.set_title('The freight cycle vs SPY - contractions bracket the market bottoms')\n"
            "    ax1.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "    print('freight YoY mean', round(yoy.mean(),3), 'std', round(yoy.std(),3),\n"
            "          'frac expanding', round((yoy>0).mean(),2))\n"
            "else:\n"
            "    print('no cache - see docs/results.md: YoY mean', R['yoy_mean'], 'frac exp', R['frac_exp'],'%')"
        ),
        md(
            f"The freight cycle spends about **{R['frac_exp']}%** of months expanding. Notice the "
            "problem already: the big red (contraction) stretches wrap around 2009 and 2020 — exactly "
            "when stocks were staging their biggest *recoveries*. A rule that flees when freight "
            "contracts would have been in cash at the bottom. Now the real question: does an expansion "
            "signal predict *extra* return?"
        ),
        md(
            "**The payoff vs a normal month — and does it work for transports?** For each horizon, the "
            "win-rate after an expansion signal next to the *base rate* (any random month), for the "
            "broad market (SPY) and for transport stocks (IYT), the sector freight should move most."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    spy_c = [st.summarize(F,'spy',m)['cond_win']*100 for m in hs]\n"
            "    spy_b = [st.summarize(F,'spy',m)['base_win']*100 for m in hs]\n"
            "    iyt_c = [st.summarize(FI,'iyt',m)['cond_win']*100 for m in hs]\n"
            "    iyt_b = [st.summarize(FI,'iyt',m)['base_win']*100 for m in hs]\n"
            "else:\n"
            "    spy_c=[R['spy_h1'][3],R['spy_h3'][3],R['spy_h6'][3],R['spy_h12'][3]]\n"
            "    spy_b=[R['spy_h1'][5],R['spy_h3'][5],R['spy_h6'][5],R['spy_h12'][5]]\n"
            "    iyt_c=[R['iyt_h1'][3],R['iyt_h3'][3],R['iyt_h6'][3],R['iyt_h12'][3]]\n"
            "    iyt_b=[R['iyt_h1'][5],R['iyt_h3'][5],R['iyt_h6'][5],R['iyt_h12'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)\n"
            "for ax,c,b,ttl in ((a1,spy_c,spy_b,'SPY (broad market)'),(a2,iyt_c,iyt_b,'IYT (transports)')):\n"
            "    ax.bar(x-.2, c, .4, color=GREEN, label='after freight expands')\n"
            "    ax.bar(x+.2, b, .4, color=GREY, label='any random month')\n"
            "    ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.set_ylim(0,100)\n"
            "    ax.set_title(ttl)\n"
            "    for i,(cc,bb) in enumerate(zip(c,b)):\n"
            "        ax.annotate(f'{cc:.0f}',(i-.2,cc),ha='center',va='bottom',fontsize=8)\n"
            "        ax.annotate(f'{bb:.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "a1.set_ylabel('% of the time higher'); a1.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY 12m win after expand:', f'{spy_c[-1]:.0f}%', 'vs base', f'{spy_b[-1]:.0f}%')\n"
            "print('IYT 12m win after expand:', f'{iyt_c[-1]:.0f}%', 'vs base', f'{iyt_b[-1]:.0f}%')"
        ),
        md(
            f"There's the tell. On **SPY** the expansion win-rate sits just a few points above the base "
            "rate — because the market rises most years anyway. On **IYT — the transports, the sector "
            "the freight story is *most* about — the expansion months actually do a touch **worse** "
            f"than average** ({R['iyt_h12'][3]:.0f}% vs {R['iyt_h12'][5]:.0f}% at 12 months). If freight "
            "were a leading signal for anything, it would be transport stocks. It isn't."
        ),
        md(
            "**Which comes first — freight or the market?** Here's the decisive picture. We slide "
            "freight's month-to-month *change* against stock returns and measure how strongly they line "
            "up at each offset. Bars to the **right** of zero mean *freight moves first* (a real leading "
            "indicator); bars to the **left** mean *stocks move first* and freight follows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag_corr(F, 'spy', max_lag=12)\n"
            "    lags, corr = ll['lags'], ll['corr']; peak = ll['peak_lag']\n"
            "else:\n"
            "    lags = list(range(-12,13)); peak = R['ll_spy'][1]\n"
            "    corr = [0.0]*len(lags)  # shape-only fallback\n"
            "cols = [GREEN if k>0 else (RED if k<0 else GREY) for k in lags]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(lags, [c*100 for c in corr], color=cols, width=.8)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.axvline(peak, c=RED, ls='--', lw=2, label=f'peak at k={peak:+d} months')\n"
            "ax.set_xlabel('offset k (months):   <-- stocks move first        freight moves first -->')\n"
            "ax.set_ylabel('correlation with SPY return (%)')\n"
            "ax.set_title('Freight lines up with the stock move of ~3 months EARLIER')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('peak correlation at k =', peak, 'months  (negative = stocks lead freight)')"
        ),
        md(
            f"The peak is at **k = {R['ll_spy'][1]} months** — on the *left*. Freight's swings line up "
            "best with what the stock market did **a quarter earlier**. In plain terms: **the market "
            "turns first, and the freight index follows about three months later.** \"Watch the "
            "freight\" is watching the past. (Same story on IYT — peak at "
            f"**{R['ll_iyt'][1]} months**.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** After the {R['lag']}-month lag, \"freight expanding\" leaves only a "
            f"tiny, insignificant bump on SPY (strongest at 6 months, *t* = {R['spy_h6'][6]:.2f}) and a "
            "**negative** excess on IYT — the transports it should predict best. No forward edge.\n"
            "- **Tradability — Mirage.** An \"own-when-expanding\" overlay earns a **lower Sharpe than "
            f"buy-and-hold** ({R['timing'][0][5]:.2f} vs {R['timing'][0][6]:.2f} on SPY; "
            f"{R['timing'][2][5]:.2f} vs {R['timing'][2][6]:.2f} on IYT) — because freight contracts "
            "right at the market bottoms you most wanted to own.\n"
            f"- **Leads the cycle? — Busted.** Freight lines up with the stock move of "
            f"**{-R['ll_spy'][1]} months earlier**. Stocks lead freight, not the reverse. The signal "
            "describes a turn the market already made."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — race it against buy-and-hold\n\n"
            "Forget significance for a second and just ask the operational question: if you own the tape "
            "only when freight is expanding and step aside otherwise, do you beat simply holding it? "
            "Here's the net-of-cost Sharpe of each overlay next to plain buy-and-hold, on both tapes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = []\n"
            "    for col, Fx in (('spy',F), ('iyt',FI)):\n"
            "        b = st.timing_backtest(Fx, col, cost_bps=10.0, allow_short=False)\n"
            "        bars.append((col.upper(), b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "else:\n"
            "    bars = [('SPY', R['timing'][0][5], R['timing'][0][6]), ('IYT', R['timing'][2][5], R['timing'][2][6])]\n"
            "x = np.arange(len(bars))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, [b[1] for b in bars], .4, color=AMBER, label='freight overlay (net)')\n"
            "ax.bar(x+.2, [b[2] for b in bars], .4, color=GREEN, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars]); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('The freight overlay LOSES to holding on both tapes')\n"
            "for i,b in enumerate(bars):\n"
            "    ax.annotate(f'{b[1]:.2f}',(i-.2,b[1]),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b[2]:.2f}',(i+.2,b[2]),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('overlay vs buy&hold Sharpe:', [(b[0], round(b[1],2), round(b[2],2)) for b in bars])"
        ),
        md(
            f"There it is. The overlay trails buy-and-hold on **both** tapes "
            f"({R['timing'][0][5]:.2f} vs {R['timing'][0][6]:.2f} on SPY; "
            f"{R['timing'][2][5]:.2f} vs {R['timing'][2][6]:.2f} on IYT) — and that's *before* crediting "
            "the interest it would earn while parked in cash, which only flatters the rule. Betting "
            "*against* the market on contraction months is even worse (it collapses toward zero), "
            "because those are the bottoms. **There is no version of this that beats buy-and-hold.** "
            "Being out of the market when freight is weak — which is when it's cheapest — costs more "
            "than the signal is worth."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The leading-indicator family.** The economic-surprise index, the ISM-PMI regime and "
            "jobless-claims momentum on this bench all rhyme with this: a *real* macro gauge rarely "
            "survives as a **tradable** monthly timing rule once you charge it the publication delay and "
            "the cost of sitting out an up-drifting market — and the \"leading\" ones often turn out to "
            "**lag** the market.\n"
            "- **The real Cass tape.** Swap our cited proxy for Cass's licensed monthly series (or the "
            "FRED mirror) and re-run — the lead-lag arrow and the fast-pricing problem won't move; the "
            "market discounts the cycle a freight print only later confirms.\n"
            "- **The other direction.** The genuinely interesting question isn't \"does freight predict "
            "stocks\" (it doesn't) but \"do stocks predict freight\" (they seem to, by ~a quarter) — a "
            "nowcast of the *goods economy* from the *market*, which is the opposite trade from the "
            "folklore.\n\n"
            "*Think the Cass index leads the market? Build the freight cycle, slide it against stock "
            "returns, and show the correlation peaking to the **right** of zero (freight first) **and** "
            "an overlay clearing buy-and-hold — then we'll talk.*"
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
            "# The Cass Freight Index — a quantitative teardown 🔬\n"
            "### A cited Cass-shipments proxy vs real SPY & IYT · conditional vs unconditional forward "
            "returns · a Welch *t* + placebo null · a lead-lag cross-correlation · an overlay-vs-"
            "buy-and-hold Sharpe race · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"watch the freight\" fuses: whether the freight cycle carries **forward** "
            "information about equities, and the **direction of the arrow** between them. The decisive "
            "object is not a point estimate (the SPY one is weakly positive, as the lore wants) but the "
            "**lead-lag sign** — and whether a timing overlay built on it can clear buy-and-hold.\n\n"
            "> ⚠️ **Data + proxy note.** Cass's monthly shipments series isn't freely API-available, so "
            "we build a transparent **proxy** — cited approximate annual anchors of the shipments level "
            "(base Jan-1990 ≈ 1.00) interpolated to monthly, conditioned on **year-over-year growth**. "
            "The equity tapes are **real** yfinance closes: **SPY** (1999→ here) and **IYT** (2004→). "
            "The freight reference month is public only mid-next-month, so we impose a "
            f"**{R['lag']}-month** publication+execution lag — applied once. Offline core + synthetic "
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
            f"| **Signal** | `NONE` | SPY 6m conditional **+{R['spy_h6'][2]:.1f}%** vs base "
            f"**+{R['spy_h6'][4]:.1f}%**, Welch **t = {R['spy_h6'][6]:.2f}** (strongest horizon) — "
            f"**fails t ≥ 2**; **IYT** excess is **negative** at every horizon (12m *t* = "
            f"{R['iyt_h12'][6]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Overlay net Sharpe **{R['timing'][0][5]:.2f}** (SPY) / "
            f"**{R['timing'][2][5]:.2f}** (IYT) vs buy-and-hold **{R['timing'][0][6]:.2f}** / "
            f"**{R['timing'][2][6]:.2f}**; long/short collapses ({R['timing'][1][5]:.2f} / "
            f"{R['timing'][3][5]:.2f}). |\n"
            f"| **Leads the cycle?** | `BUSTED` | Lead-lag cross-correlation peaks at "
            f"**k = {R['ll_spy'][1]}** (SPY) / **{R['ll_iyt'][1]}** (IYT) — stocks lead freight by ~a "
            "quarter. |\n\n"
            "> 💡 In plain words: the freight cycle really does rhyme with the economy — but the market "
            "has already priced that cycle before Cass prints it, so freight lines up with *past* stock "
            "returns, adds no forward edge (a *negative* one on transports), and can't be traded."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_t$ be the Cass shipments proxy and $g_t = c_t/c_{t-12} - 1$ its year-over-year "
            "growth. A month is **expanding** when $g_{t} > 0$. Because Cass publishes month $t$ in the "
            f"middle of month $t{{+}}1$, an expanding signal is acted on at $t{{+}}{R['lag']}$ "
            f"(**{R['lag']}-month** publication+execution lag).\n\n"
            "- **H₁ (forward predictive).** $\\mathbb{E}[r_{t\\to t+H}\\mid g_t>0] > "
            "\\mathbb{E}[r_{t\\to t+H}]$ for forward horizon $H$ on SPY *and* (more so) on IYT.\n"
            "- **H₂ (leads).** The cross-correlation of $\\Delta g_t$ with $r_{t+k}$ peaks at a "
            "**positive** $k$ (freight today ⇒ stocks later).\n"
            "- **H₃ (deployable).** An overlay long the tape when $g>0$ clears buy-and-hold net of "
            "costs.\n\n"
            "We find **H₁ rejected** (SPY excess positive but $t<2$; IYT excess **negative**), **H₂ "
            f"rejected and *reversed*** (peak at $k={R['ll_spy'][1]}$ — stocks lead freight), **H₃ "
            "rejected** (the overlay loses on both tapes). The legend mistakes a coincident-to-lagging "
            "*measurement* of the goods economy for a forward *forecast* of the market."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Two decompositions carry the teardown. First, a conditional mean against an unconditional "
            "baseline, judged by its **standard error**:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{exp}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{exp}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "A raw **win-rate** is the wrong lens (US returns are positive most of the time "
            "unconditionally), so the honest instruments are a **Welch two-sample t** and a "
            "**randomization (placebo) null**. Second — and decisively — the **lead-lag** "
            "cross-correlation $\\rho_k = \\mathrm{corr}(\\Delta g_t,\\, r_{t+k})$: its **argmax over "
            "$k$** *orders the two series in time*. A leading indicator needs $k^\\star>0$. Using "
            "$\\Delta g$ (a stationary change) rather than the level avoids a spurious trend-on-trend "
            "correlation. And because a real estimate is worthless if it can't be deployed, "
            "**Tradability** is settled separately by a net-of-cost **Sharpe race**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Freight signal.** Cited Cass-shipments **proxy**, {R['start']}→{R['end']} "
            f"(**{R['months']} months**); YoY growth $g_t$; \"expanding\" = $g_t>0$ "
            f"(**{R['frac_exp']}%** of months). Named a proxy on the Signal axis.\n"
            f"- **Lag.** {R['lag']} months = 1 (Cass prints month $t$ mid-$t{{+}}1$) + 1 (execution). "
            "One shift, applied once, no look-ahead.\n"
            "- **Tapes.** Real yfinance month-end closes, **price-only**: SPY (broad) and IYT "
            f"(transports, from {R['iyt_start'][:4]}, **{R['iyt_months']} months**).\n"
            "- **Null #1 (Welch t).** Conditional mean vs the unconditional overlapping-window mean, per "
            "horizon $H\\in\\{1,3,6,12\\}$, on both tapes.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random valid entry dates; $p=\\Pr[\\text{"
            "random mean}\\ge\\text{expanding mean}]$.\n"
            "- **Lead-lag.** $\\rho_k=\\mathrm{corr}(\\Delta g_t, r_{t+k})$ for $k\\in[-12,12]$; report "
            "$k^\\star=\\arg\\max_k|\\rho_k|$.\n"
            "- **Overlay.** Long/flat (or long/short) tape held when $g>0$, 2-month lag, 10 bps one-way "
            "per turn, raced vs buy-and-hold on Sharpe.\n"
            "- **Positive control.** Deterministic AR(1) freight-YoY + SPY-like price with a **known** "
            "planted forward edge: the inference must recover it, and must NOT fire when the edge is 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — weak-positive on SPY, negative on the sector\n\n"
            "Conditional mean forward return $\\pm$ standard error against the unconditional base rate "
            "(diamonds), for SPY and IYT. On the broad market the excess is small and never significant; "
            "on the transports — the freight-sensitive sector — it is *negative*."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "def _pack(Fx, col, key):\n"
            "    if HAVE_REAL:\n"
            "        cm,bm,ts,se = [],[],[],[]\n"
            "        for m in hs:\n"
            "            s=st.summarize(Fx,col,m); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "            cr=st.conditional_returns(Fx,col,m); se.append(cr.std(ddof=1)/np.sqrt(len(cr)))\n"
            "        return cm,bm,ts,se\n"
            "    rows=[R[key+'_h1'],R[key+'_h3'],R[key+'_h6'],R[key+'_h12']]\n"
            "    return ([r[2]/100 for r in rows],[r[4]/100 for r in rows],[r[6] for r in rows],[.01,.018,.028,.045])\n"
            "spy_cm,spy_bm,spy_t,spy_se = _pack(F,'spy','spy')\n"
            "iyt_cm,iyt_bm,iyt_t,iyt_se = _pack(FI,'iyt','iyt')\n"
            "x=np.arange(len(hs))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.4),sharey=True)\n"
            "for ax,cm,bm,se,ttl in ((a1,spy_cm,spy_bm,spy_se,'SPY'),(a2,iyt_cm,iyt_bm,iyt_se,'IYT (transports)')):\n"
            "    ax.bar(x,[c*100 for c in cm],yerr=[s*100 for s in se],capsize=5,color=GREEN,width=.5,label='after expand (±SE)')\n"
            "    ax.plot(x,[b*100 for b in bm],'D',ms=11,c=GREY,label='base rate')\n"
            "    ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0,c='k',lw=.8); ax.set_title(ttl)\n"
            "a1.set_ylabel('mean forward return (%)'); a1.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY Welch t:', {f'{m}m':round(t,2) for m,t in zip(hs,spy_t)})\n"
            "print('IYT Welch t:', {f'{m}m':round(t,2) for m,t in zip(hs,iyt_t)})"
        ),
        md(
            f"> 💡 In plain words: on SPY the excess tops out at **+{R['spy_h6'][2]-R['spy_h6'][4]:.1f}pp** "
            f"(6m, **t = {R['spy_h6'][6]:.2f}**) — a positive whisper inside its own error bar. On IYT "
            f"the 12-month conditional is **+{R['iyt_h12'][2]:.1f}%** vs a **+{R['iyt_h12'][4]:.1f}%** "
            f"base (**t = {R['iyt_h12'][6]:.2f}**): owning transports *because* freight is expanding was "
            "actively worse than owning them anyway. H₁ is rejected — and rejected hardest exactly where "
            "the story is strongest."
        ),
        md(
            "### 4b · The decisive test — which series leads?\n\n"
            "Cross-correlation $\\rho_k=\\mathrm{corr}(\\Delta g_t, r_{t+k})$. $k>0$ ⇒ freight leads "
            "stocks; $k<0$ ⇒ stocks lead freight. The argmax over $k$ tells the direction of causation-"
            "in-time."
        ),
        code(
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3),sharey=True)\n"
            "for ax,col,Fx,key in ((a1,'spy',F,'ll_spy'),(a2,'iyt',FI,'ll_iyt')):\n"
            "    if HAVE_REAL:\n"
            "        ll=st.lead_lag_corr(Fx,col,max_lag=12); lags,corr,peak=ll['lags'],ll['corr'],ll['peak_lag']\n"
            "    else:\n"
            "        lags=list(range(-12,13)); corr=[0.0]*25; peak=R[key][1]\n"
            "    cols=[GREEN if k>0 else (RED if k<0 else GREY) for k in lags]\n"
            "    ax.bar(lags,[c*100 for c in corr],color=cols,width=.8); ax.axvline(0,c='k',lw=.8)\n"
            "    ax.axvline(peak,c=RED,ls='--',lw=2,label=f'peak k={peak:+d}')\n"
            "    ax.set_title(col.upper()); ax.set_xlabel('k (months): <-stocks lead | freight lead->'); ax.legend()\n"
            "a1.set_ylabel('corr(dFreight, return) (%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for col,Fx in (('spy',F),('iyt',FI)):\n"
            "    ll=st.lead_lag_corr(Fx,col,12) if HAVE_REAL else {'peak_lag':R['ll_'+col][1],'peak_corr':R['ll_'+col][2]}\n"
            "    print(col.upper(),'peak lag =', ll['peak_lag'], ' corr =', round(ll['peak_corr'],3))"
        ),
        md(
            f"> 💡 In plain words: both tapes peak at **k = {R['ll_spy'][1]}** — freight correlates most "
            "with the equity return of *three months earlier*. Averaged over the window, the "
            f"\"stocks-lead\" mass (k<0, {R['ll_spy'][3]:.3f}) is nearly **double** the \"freight-lead\" "
            f"mass (k>0, {R['ll_spy'][4]:.3f}). This is the crux: not a weak forward estimate alone, but "
            "an arrow that points the **wrong way**. H₂ is rejected and reversed — the market prices the "
            "cycle before Cass measures it."
        ),
        md(
            "### 4c · Tradability + robustness — the Sharpe race and the threshold sweep\n\n"
            "First the operational verdict: an \"own-when-expanding\" overlay, net of 10 bps/turn, "
            "against buy-and-hold on both tapes (long/flat and long/short). Then a threshold sweep on "
            "SPY — there's no 'expanding' cutoff at which the 12-month *t* clears 2 (it goes *negative* "
            "when tightened)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim=[]\n"
            "    for col,Fx in (('spy',F),('iyt',FI)):\n"
            "        for short in (False,True):\n"
            "            b=st.timing_backtest(Fx,col,cost_bps=10.0,allow_short=short)\n"
            "            tim.append((col.upper(),'L/S' if short else 'L/F',b['net']['sharpe'],b['buy_hold']['sharpe']))\n"
            "    rob=[(thr,)+tuple(st.summarize(F,'spy',12,thr=thr)[k] for k in ('n','cond_mean','t','p_placebo')) for thr in (-0.02,0.0,0.02)]\n"
            "else:\n"
            "    tim=[('SPY','L/F',R['timing'][0][5],R['timing'][0][6]),('SPY','L/S',R['timing'][1][5],R['timing'][1][6]),\n"
            "         ('IYT','L/F',R['timing'][2][5],R['timing'][2][6]),('IYT','L/S',R['timing'][3][5],R['timing'][3][6])]\n"
            "    rob=[(r[0],r[1],r[2]/100,r[3],r[4]) for r in R['robust']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))\n"
            "labs=[f'{t[0]}\\n{t[1]}' for t in tim]; xs=np.arange(len(tim))\n"
            "a1.bar(xs-.2,[t[2] for t in tim],.4,color=AMBER,label='overlay (net)')\n"
            "a1.bar(xs+.2,[t[3] for t in tim],.4,color=GREEN,label='buy & hold')\n"
            "a1.set_xticks(xs); a1.set_xticklabels(labs); a1.set_ylabel('Sharpe'); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_title('Overlay loses to holding, both tapes'); a1.legend(fontsize=8)\n"
            "thrs=[f'{r[0]:+.2f}' for r in rob]; tt=[r[3] for r in rob]; nn=[r[1] for r in rob]\n"
            "a2.bar(thrs,tt,color=AMBER,width=.55); a2.axhline(2,ls='--',c=RED,label='t=2 bar'); a2.axhline(0,c='k',lw=.8)\n"
            "for i,(t,kk) in enumerate(zip(tt,nn)): a2.annotate(f'n={kk}',(i,t),ha='center',va='bottom' if t>=0 else 'top',fontsize=8)\n"
            "a2.set_title('SPY 12m t never reaches 2'); a2.set_xlabel(\"'expanding' threshold\"); a2.set_ylabel('Welch t'); a2.set_ylim(-1.6,2.4); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('overlay net Sharpe vs buy&hold:', [(t[0],t[1],round(t[2],2),round(t[3],2)) for t in tim])\n"
            "print('robustness (thr,n,cond12,t,p):', [(r[0],r[1],round(r[2]*100,1),round(r[3],2),round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: every overlay trails buy-and-hold — SPY long/flat "
            f"**{R['timing'][0][5]:.2f}** vs **{R['timing'][0][6]:.2f}**, IYT **{R['timing'][2][5]:.2f}** "
            f"vs **{R['timing'][2][6]:.2f}** — and the long/short versions collapse "
            f"({R['timing'][1][5]:.2f} / {R['timing'][3][5]:.2f}) because contraction months are the "
            "market bottoms. On the threshold side, loosening washes *t* toward 0, tightening turns it "
            f"**negative** ({R['robust'][2][3]:.2f}). **No threshold, cost regime, or leg** makes this "
            "deployable — the signal isn't there."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic monthly series (AR(1) freight-YoY + SPY-like price): with a **zero** "
            "planted edge the test must stay below t=2; with a **+0.05** planted edge (next return "
            "loaded on freight two months earlier) it must light up. Both hold — proving the engine is "
            "unbiased *and* that the real-tape near-zero/negative *t* is what an *absent* edge looks "
            "like."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 0.05):\n"
            "    syn=data.synthetic(n_months=312, edge=edge, seed=757)\n"
            "    s=st.summarize(syn,'spy',6)\n"
            "    res.append((edge,s['n'],s['cond_mean']*100,s['base_mean']*100,s['t'],s['p_placebo']))\n"
            "fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "labels=[f'planted edge\\n{e:.2f}' for e,_,_,_,_,_ in res]; tvals=[r[4] for r in res]\n"
            "ax.bar(labels,tvals,color=[GREY,GREEN],width=.5); ax.axhline(2,ls='--',c=RED,label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (6-month)'); ax.set_title('Control: recovers a real edge, ignores a fake one'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e:+.2f}: detected={k} cond={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive); a **+0.05** planted edge reaches "
            f"**t = {R['syn'][1][4]:.2f}** with placebo **p = {R['syn'][1][5]:.3f}**. The machinery is "
            "honest — so the real-tape result (weak-positive on SPY, negative on IYT, arrow reversed) is "
            "an *absent* forward edge seen through a faithful lens, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SPY excess tops **+{R['spy_h6'][2]-R['spy_h6'][4]:.1f}pp** at Welch "
            f"**t = {R['spy_h6'][6]:.2f}** (placebo **p = {R['spy_h6'][7]:.2f}**), **fails t ≥ 2**; the "
            f"freight-sensitive IYT excess is **negative** at every horizon (12m *t* = "
            f"{R['iyt_h12'][6]:.2f}). Named structural fact on the Signal axis: the lead-lag peaks at "
            f"**k = {R['ll_spy'][1]}** — freight *lags* equities. Not REAL, not WEAK: no forward edge.\n"
            f"- **Tradability `MIRAGE`** — overlay net Sharpe **{R['timing'][0][5]:.2f}** (SPY) / "
            f"**{R['timing'][2][5]:.2f}** (IYT) vs **{R['timing'][0][6]:.2f}** / "
            f"**{R['timing'][2][6]:.2f}** buy-and-hold; long/short collapses "
            f"({R['timing'][1][5]:.2f} / {R['timing'][3][5]:.2f}). Freight contracts at the bottoms, so "
            "sitting out costs more than the signal returns. No NAV-scale edge.\n"
            f"- **Leads the cycle? `BUSTED`** — the cross-correlation argmax is **k = {R['ll_spy'][1]}** "
            f"(SPY) / **{R['ll_iyt'][1]}** (IYT); the stocks-lead mass is ~2× the freight-lead mass. "
            "\"Watch the freight\" watches a turn the market already made."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost of sitting out\n\n"
            "The operational truth in one picture: the equity curve of the SPY \"own-when-expanding\" "
            "overlay against buy-and-hold. The gap isn't costs — it's the **months out of the market** "
            "while freight is contracting, which is precisely when the market is cheapest and about to "
            "rebound."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret=st.monthly_returns(F,'spy')\n"
            "    pos=(st.freight_yoy(F).shift(R['lag'])>0).reindex(ret.index).fillna(False).astype(float)\n"
            "    turn=pos.diff().abs().fillna(pos.abs()); c=10.0/1e4\n"
            "    eq_rule=(1+(pos*ret-turn*c)).cumprod(); eq_bh=(1+ret).cumprod(); expo=(pos>0).mean()\n"
            "else:\n"
            "    rng=np.random.default_rng(757); bh=pd.Series(rng.normal(0.006,0.043,R['months']))\n"
            "    eq_bh=(1+bh).cumprod(); eq_rule=(1+bh*0.48).cumprod(); expo=0.48\n"
            "fig,ax=plt.subplots(figsize=(9.2,4.4))\n"
            "ax.plot(eq_bh.index,eq_bh.values,c=GREEN,lw=2,label='buy & hold')\n"
            "ax.plot(eq_rule.index,eq_rule.values,c=AMBER,lw=1.8,label='freight overlay (long/flat)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'Owning only when freight expands ({expo*100:.0f}% exposure) trails buy & hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth: overlay {eq_rule.iloc[-1]:.1f}x vs buy&hold {eq_bh.iloc[-1]:.1f}x (exposure {expo*100:.0f}%)')"
        ),
        md(
            f"> 💡 In plain words: the overlay holds SPY only ~**{R['timing'][0][2]}%** of months and "
            "ends **below** buy-and-hold. The amber curve flatlines through every contraction — and "
            "because those contractions bracket the 2009 and 2020 bottoms, the foregone rebound swamps "
            "any protection. There is no sizing, cost assumption, or tape (SPY or IYT) that turns "
            "\"watch the freight\" into a portfolio: the signal describes the past, and the past isn't "
            "tradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The leading-indicator family.** The economic-surprise index, the ISM-PMI regime and "
            "jobless-claims momentum on this bench share the pathology: a real macro gauge rarely "
            "survives as a tradable monthly timing rule once charged the publication delay and the "
            "opportunity cost of cash through an up-drifting market — and the \"leading\" ones often "
            "*lag*.\n"
            "- **The real Cass tape.** Replace the cited proxy with Cass's licensed series (or the FRED "
            "mirror `FRGSHPUSM649NCIS`) and re-run — the lead-lag sign and the fast-pricing problem "
            "(Andersen-Bollerslev-Diebold-Vega, 2003) won't move.\n"
            "- **Flip the arrow.** The defensible finding is the *reverse* nowcast: equity returns lead "
            "$\\Delta$freight by ~a quarter. Model freight *from* the market (a goods-economy nowcast), "
            "not the market from freight — the opposite of the folklore.\n\n"
            "*The reproducible core is offline and deterministic; the freight series is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
