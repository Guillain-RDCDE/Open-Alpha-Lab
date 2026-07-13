"""Generate the two narrative notebooks for Study 740 (Drought-Monitor).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
SPY/DE/MOS/ADM/MOO/DBA/CORN/WEAT tapes under ../_cache/ and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + DE/MOS/ADM/MOO
# + DBA/CORN/WEAT, 2000-01-03 -> 2026-06-30; 21 hardcoded US drought escalations 2000-08 -> 2025-05).
R = dict(
    n_events=21, n_grain=16, cal_lo="2000-08-03", cal_hi="2025-05-15",
    fp="9261c00637a8",
    # print-day (day 0) abnormal return vs SPY
    eq_day0_bps=-24.28, eq_day0_t=-0.84, eq_up=9, eq_n=21, eq_wilson=(24.5, 63.5),
    gr_day0_bps=-71.01, gr_day0_t=-1.30, gr_up=8, gr_n=16, gr_wilson=(28.0, 72.0),
    # random-calendar placebo (right tail)
    eq_pl_mean_bps=2.10, eq_pl_sd_bps=27.63, eq_pl_p=0.837,
    gr_pl_mean_bps=-4.69, gr_pl_sd_bps=39.24, gr_pl_p=0.958, pl_draws=20000,
    # event window: offset -> (mean_bps, car_bps, t)
    event={-1: (-32.54, 0.00, -1.37), 0: (-24.28, -24.28, -0.84), 1: (-2.66, -26.94, -0.16),
           2: (27.66, 0.72, 0.91), 3: (29.84, 30.56, 1.46), 4: (-18.37, 12.18, -0.74),
           5: (40.80, 52.99, 2.17)},
    # post-print drift [+1..+5]
    drift_bps=77.27, drift_t=1.56, drift_pl_p=0.137, drift_ci=(-9.2, 177.8),
    # third axis: grain vs ag-equity, paired day-0 difference
    grain_mean_bps=-71.01, equity_mean_bps=-33.64, diff_bps=-37.36, diff_t=-0.83, diff_n=16,
    # buy-the-drought timer: hold -> (gross_bps, net5_bps, t_net5, win_pct, baseline_bps)
    timer={1: (-2.66, -12.66, -0.74, 48, 2.05), 5: (77.27, 67.27, 1.36, 67, 10.06),
           10: (84.29, 74.29, 1.14, 62, 19.77), 21: (73.80, 63.80, 0.49, 62, 41.39)},
    timer_net10_5d=57.27, timer_net10_5d_t=1.15,
    # regime test (labelled monthly proxy)
    reg_n=305, reg_thr=20, reg_hi_n=115, reg_lo_n=190,
    reg_hi_bps=42.85, reg_lo_bps=50.46, reg_welch_t=-0.11, reg_hi_t=0.82,
    reg_timer_net_bps=32.85,
    # synthetic control
    syn_null_mean=0.02, syn_null_sd=1.07, syn_null_fire=0, syn_null_seeds=20,
    syn_planted1_t=2.93, syn_planted2_t=7.59, syn_planted2_bps=160.9,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Drought_moves_ag%3F: Busted](https://img.shields.io/badge/Drought_moves_ag%3F-Busted-8b949e?style=flat-square)\n\n"
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

from drought_monitor import data, strategy as st

PRE, POST = 1, 5
EVENTS = data.drought_events()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, NAMES = data.load_real()
    SPY_RET = st.daily_returns(SPY)
    EQ_RET, EQ_COV = st.basket_returns(NAMES, data.AG_EQUITY_TICKERS)
    GR_RET, GR_COV = st.basket_returns(NAMES, data.GRAIN_TICKERS)
    AR_EQ = st.abnormal_vs_bench(EQ_RET, SPY_RET)
    AR_GR = st.abnormal_vs_bench(GR_RET, SPY_RET)
else:
    SPY = NAMES = SPY_RET = EQ_RET = GR_RET = AR_EQ = AR_GR = None
print("real cache present:", HAVE_REAL, "| drought escalations:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you trade the drought? 🏜️🌽\n"
            "### The US Drought Monitor prints every Thursday — and the ag complex "
            "shrugs\n\n"
            + BADGES +
            "Every Thursday morning since 2000, the **US Drought Monitor** publishes a "
            "map of how much of the country is baking in severe drought. The trade "
            "writes itself: drought means a smaller harvest, means pricier grain, means "
            "good times for the people who sell into the shortage — Deere (tractors), "
            "Mosaic (fertilizer), ADM (grain trading), and the ag/grain ETFs. So when "
            "the Monitor prints a worse number, *buy the drought.*\n\n"
            "We tested it properly — the 21 biggest US drought escalations since 2000, "
            "against a basket of the exact names the story names.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "regime split? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 21 drought escalations hardcoded to their Thursday "
            "Monitor release; each mapped to an ag-equity basket (DE/MOS/ADM/MOO) and a "
            "grain basket (DBA/CORN/WEAT), measured *against SPY*. Every chart is drawn "
            "by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the ag complex pop on the drought print? | **No.** It drifts "
            f"*down* **{R['eq_day0_bps']:+.0f} bps** vs the market — the wrong sign, and "
            "statistically nothing (*t* = −0.84). |\n"
            f"| Does grain at least react? | **No — and if anything, worse** "
            f"(**{R['gr_day0_bps']:+.0f} bps**, *t* = −1.30). Grain markets priced the "
            "weather forecast weeks before the Monitor confirmed it. |\n"
            f"| Could you \"buy the drought\"? | **Not reliably.** The best horizon "
            f"(5 days) nets **{R['timer'][5][1]:+.0f} bps** — but at *t* = "
            f"{R['timer'][5][2]:+.2f}, short of the bar, and it fades away by 21 days. |\n"
            f"| Do high-drought *months* favour ag? | **No.** Ag is a hair *behind* the "
            f"market in the driest third of months (Welch *t* = {R['reg_welch_t']:+.2f}). |\n\n"
            "> The story is intuitive and completely undetectable. The weekly Monitor is "
            "old news to a market that prices the forecast in real time."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the US Drought Monitor shows severe drought spreading across the "
            "crop belt, the harvest outlook worsens, grain gets scarcer and pricier, "
            "and the ag names — farm equipment, fertilizer, grain traders, the ag "
            "ETFs — get a tailwind. Read the Thursday print and buy the drought.\"*\n\n"
            "It rides on something real: weather genuinely moves agricultural commodity "
            "*prices* (the 2012 Corn Belt drought sent corn futures to record highs). "
            "The open question is narrower — is the **public, weekly, pre-scheduled "
            "Monitor print itself** a tradable event for US-listed ag stocks and ETFs? "
            "Or is the drought already fully in prices by the time the Monitor confirms "
            "it?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a clean, mechanical, weather-driven edge: a free "
            "government data release every Thursday, a basket of liquid names to express "
            "it, a supply-shock story any commodity desk would nod along to. Weather "
            "trades are a whole cottage industry. We wanted to know: does *this* one — "
            "the most-watched public drought index in the world — actually pay?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know\n\n"
            f"- **The calendar.** **{R['n_events']}** major US drought escalations "
            "2000→2025, each dated to a representative **Thursday** Drought Monitor "
            "release during the rapid-worsening phase (2011 Texas, 2012 Corn Belt, "
            "2012–15 California, 2021–22 Western megadrought, 2023 Midwest…).\n"
            "- **The names.** An ag-equity basket (Deere `DE`, Mosaic `MOS`, ADM, "
            "agribusiness ETF `MOO`) and a grain basket (broad-ag `DBA`, corn `CORN`, "
            "wheat `WEAT`), each measured **against SPY** — so a result means \"ag beat "
            "the market\", not \"stocks went up\".\n"
            "- **The lag.** The Monitor is public Thursday ~8:30am, before the close, "
            "so we enter at that **Thursday's close** — zero look-ahead.\n"
            "- **The honesty check.** A random-calendar placebo (does a random Thursday "
            "produce the same move just as often?), a grain-vs-equity test, a costed "
            "\"buy the drought\" timer, and a drought-*regime* split on a labelled "
            "monthly severity proxy."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline: what does the ag complex do on the print day?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0e = st.day0_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    d0g = st.day0_stats(AR_GR, EVENTS['date'], PRE, POST)\n"
            "    eq, gr = d0e['mean']*1e4, d0g['mean']*1e4\n"
            "else:\n"
            "    eq, gr = R['eq_day0_bps'], R['gr_day0_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.6))\n"
            "ax.bar(['ag-equity\\n(DE/MOS/ADM/MOO)', 'grain\\n(DBA/CORN/WEAT)'], [eq, gr],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([eq, gr]): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('print-day abnormal return vs SPY (bps)')\n"
            "ax.set_title('The drought prints — and the ag complex drifts the WRONG way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ag-equity {eq:+.2f} bps   grain {gr:+.2f} bps')"
        ),
        md(
            f"On the day the bad news is confirmed, the ag-equity basket is "
            f"**{R['eq_day0_bps']:+.0f} bps** vs the market and grain is "
            f"**{R['gr_day0_bps']:+.0f} bps** — both *negative*, the opposite of the "
            "claim, and both statistically indistinguishable from zero "
            f"(*t* = {R['eq_day0_t']:.2f} / {R['gr_day0_t']:.2f}). The up-rate is "
            f"{R['eq_up']}/{R['eq_n']} = 42.9% for the equities — a coin flip.\n\n"
            "**Is that just a small, noisy sample? Compare it to random Thursdays.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0e = st.day0_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    draws = np.concatenate([st.placebo_distribution(AR_EQ, d0e['n'], PRE, POST,\n"
            "                            n_draws=1000, seed=740+s, stat='day0') for s in range(20)])*1e4\n"
            "    obs = d0e['mean']*1e4\n"
            "else:\n"
            "    rng = np.random.default_rng(740)\n"
            "    draws = rng.normal(R['eq_pl_mean_bps'], R['eq_pl_sd_bps'], R['pl_draws'])\n"
            "    obs = R['eq_day0_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='random-calendar null (20k draws)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed drought prints {obs:+.0f} bps')\n"
            "ax.axvline(float(np.mean(draws)), c='k', lw=1, ls=':', label='null mean')\n"
            "ax.set_xlabel('mean print-day abnormal return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A random calendar beats the drought calendar {R[\"eq_pl_p\"]*100:.0f}% of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['eq_day0_bps']:+.2f} bps, placebo p = {R['eq_pl_p']:.3f}\")"
        ),
        md(
            f"The observed drought-print move sits **inside the bulk** of the "
            f"random-calendar cloud — right-tail *p* = **{R['eq_pl_p']:.3f}** for the "
            f"equities, **{R['gr_pl_p']:.3f}** for grain. Picking 21 random Thursdays "
            "does at least as well as picking the 21 worst drought weeks. There is no "
            "print-day reaction to find.\n\n"
            "**Does the reaction just show up a bit later? Here's the whole week.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    days = list(cp.index); car = list(cp['car']*1e4); ts = list(cp['t'])\n"
            "else:\n"
            "    days = sorted(R['event']); car = [R['event'][k][1] for k in days]\n"
            "    ts = [R['event'][k][2] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts]\n"
            "ax.plot(days, car, color=AMBER, lw=2.2, marker='o')\n"
            "ax.scatter(days, car, c=cols, s=70, zorder=5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the drought print (day -1 = 0)')\n"
            "ax.set_ylabel('cumulative abnormal return vs SPY (bps)')\n"
            "ax.set_title('A wobble around zero — the one red dot (day +5) is a fluke among 7 offsets')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAR by day (bps):', [round(c,1) for c in car])"
        ),
        md(
            "The cumulative path just **wanders around zero**. Exactly one bar (day +5) "
            "crosses the significance line — but with 7 offsets tested, roughly one "
            "crossing by pure luck is *expected*, and there's no story for why the ag "
            "complex would wake up precisely five sessions after a print and not before. "
            "That's a look-elsewhere fluke, not a delayed reaction.\n\n"
            "**Finally — could you have banked it? \"Buy the drought\" and hold.**"
        ),
        code(
            "holds = [1, 5, 10, 21]\n"
            "if HAVE_REAL:\n"
            "    net, base = [], []\n"
            "    for h in holds:\n"
            "        ln = st.trade_it(EQ_RET, SPY_RET, EVENTS['date'], hold=h, cost_bps=5.0)\n"
            "        net.append(st.summarize_trade(ln, 'ret_net')['mean_bps'])\n"
            "        fwd = st.abnormal_vs_bench(EQ_RET, SPY_RET).rolling(h).sum().shift(-h)\n"
            "        base.append(float(fwd.mean()*1e4))\n"
            "else:\n"
            "    net = [R['timer'][h][1] for h in holds]; base = [R['timer'][h][4] for h in holds]\n"
            "x = np.arange(len(holds)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - w/2, net, w, color=AMBER, label='buy the drought (net of costs)')\n"
            "ax.bar(x + w/2, base, w, color=GREY, label='just hold ag-vs-SPY (unconditional)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return, ag excess of SPY (bps)')\n"
            "ax.set_title('The drought trade decays into the always-hold baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net (bps):', [round(n,1) for n in net])"
        ),
        md(
            f"The best case is a 5-day hold: **{R['timer'][5][1]:+.0f} bps** net, above "
            f"the **{R['timer'][5][4]:+.0f} bps** you'd earn just holding ag-vs-SPY "
            f"unconditionally. But it's only *t* = {R['timer'][5][2]:+.2f} — nowhere near "
            "the desk's bar — and by 21 days the \"drought\" trade "
            f"(**{R['timer'][21][1]:+.0f} bps**) has decayed almost into the "
            f"**{R['timer'][21][4]:+.0f} bps** unconditional drift. You're not being paid "
            "for the drought; you're being paid for owning ag stocks, drought or no drought."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The print-day move is the wrong sign and "
            "insignificant for both baskets; a random calendar does as well "
            f"(*p* = {R['eq_pl_p']:.2f} / {R['gr_pl_p']:.2f}); and high-drought months "
            f"carry no forward ag edge (Welch *t* = {R['reg_welch_t']:+.2f}).\n"
            "- **Tradability — Mirage.** No horizon of \"buy the drought\" clears the "
            "bar net of costs; the best (5-day) is *t* = 1.36 and decays toward the "
            "always-hold baseline.\n"
            "- **\"Does a drought print move the ag complex?\" — Busted.** The weekly "
            "Monitor is old news to a market that prices the weather forecast weeks "
            "earlier."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is what an *efficient* public signal looks like.** The Drought "
            "Monitor is famous, free, and slow-moving — exactly the kind of information "
            "the market has already digested from weather models long before Thursday. "
            "A real weather edge, if one exists, lives in the *forecast surprise*, not "
            "the *confirmation*.\n"
            "- **The natural sequel** would test grain *futures* (not equity-diluted "
            "ETFs) around USDA WASDE crop-report *surprises* — a genuine scheduled "
            "information shock — rather than the already-anticipated Monitor. That's "
            "where a supply-shock reaction, if it survives anywhere, should show up.\n"
            "- **Sibling studies:** the [plane-crash effect](../../707-plane-crash-effect/) "
            "(a news-shock event study on the broad market), the "
            "[geopolitical-shock study](../../313-geopolitical-shock/), and the "
            "[FOMC vol-crush](../../637-fomc-vol-crush/) (a *scheduled public release* "
            "event study) — same machinery, different triggers.\n\n"
            "*Think the Drought Monitor really is tradable? Bring grain futures, a "
            "forecast-surprise measure, or an intraday window around the 8:30 release, "
            "and show a net, placebo-surviving edge. We'll publish the teardown.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Drought-Monitor — a quantitative teardown 🔬\n"
            "### An event study on the ag complex around 21 US Drought Monitor "
            "escalations · a random-calendar placebo · a grain-vs-equity paired test · "
            "a costed timer · a drought-regime split · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **a worsening US Drought Monitor "
            "print is tradable supply-shock news for the ag complex** — is tested on the "
            "instruments a reader could actually buy: an ag-equity basket (DE/MOS/ADM/"
            "MOO) and a grain basket (DBA/CORN/WEAT), each measured abnormal of SPY, "
            "around the Thursday release.\n\n"
            "> ⚠️ **Data note.** SPY + 7 ag tickers, yfinance, total-return daily closes, "
            "2000-01-03→2026-06-30. 21 drought escalations hardcoded to their Thursday "
            "Monitor release. Grain-ETF coverage names **16 of 21** events honestly "
            "(DBA 2007, CORN 2010, WEAT 2011). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | ag-equity print-day AR **{R['eq_day0_bps']:+.2f} bps** "
            f"(*t* = {R['eq_day0_t']:.2f}, placebo *p* = {R['eq_pl_p']:.3f}); grain "
            f"**{R['gr_day0_bps']:+.2f} bps** (*t* = {R['gr_day0_t']:.2f}, *p* = "
            f"{R['gr_pl_p']:.3f}); regime Welch *t* = {R['reg_welch_t']:+.2f} |\n"
            f"| **Tradability** | `MIRAGE` | best \"buy the drought\" horizon (5d) net "
            f"**{R['timer'][5][1]:+.0f} bps**, *t* = {R['timer'][5][2]:+.2f}; decays to "
            f"*t* = {R['timer'][21][2]:+.2f} at 21d |\n"
            f"| **Drought moves ag?** | `BUSTED` | event study + placebo + grain-vs-equity "
            f"(diff *t* = {R['diff_t']:.2f}) + regime split all null |\n\n"
            "> 💡 In plain words: a famous, free, weekly public signal with zero tradable "
            "content on the ag names — the market prices the forecast, not the "
            "confirmation."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{basket}_t$ be an equal-weight ag basket's daily return and "
            "$r^{SPY}_t$ the benchmark's. The abnormal return is a beta-1 market model,\n\n"
            "$$AR_t = r^{basket}_t - r^{SPY}_t,$$\n\n"
            "so a positive $AR$ means the basket **outperformed the market**, not merely "
            "\"went up\". For each drought escalation, day 0 is the first NYSE session on/"
            "after the Thursday Monitor release (public ~8:30 ET, before the close — zero "
            "look-ahead). Because each print is a single, non-overlapping, independent "
            "event, the **one-sample t** of $AR$ across events is the correct primary "
            "statistic — not a daily panel. Claims:\n\n"
            "- **H1 (print-day pop).** $E[AR_0] > 0$ for the ag-equity and grain baskets.\n"
            "- **H2 (post-print drift).** $E[\\sum_{k=1}^{5} AR_k] > 0$ (the reaction "
            "builds over the week).\n"
            "- **H3 (grain reacts harder).** grain's day-0 $AR$ > the equities' day-0 $AR$.\n"
            "- **H4 (regime).** ag beats the market in high-drought months.\n\n"
            "We find **H1 rejected** (wrong sign, insignificant, placebo-confirmed null); "
            "**H2 not supported** (drift positive but sub-2, placebo *p* = 0.14, CI "
            "straddles zero); **H3 rejected** (grain reacts, if anything, *less*); "
            "**H4 rejected** (high-drought months slightly *behind*)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is fixed by the drought record: **{R['n_events']}** escalations for the "
            f"ag-equity basket, **{R['n_grain']}** for the grain basket (the 5 earliest "
            "predate every grain ETF). The plan is a **one-sample t** per basket on the "
            "print-day AR, a **Wilson interval** on the up-rate, a **20-seed × 1,000-draw "
            "random-calendar placebo** (redraw N random non-drought Thursdays from the "
            "same tape and see how often the null matches or beats the observed mean), a "
            "**paired grain−equity** day-0 test, a **costed timer** across holds, and a "
            "**drought-regime split** on the labelled monthly proxy with the regime known "
            "at the month's start (one shift)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} escalations 2000→2025, hardcoded Thursday "
            "USDM releases.\n"
            "- **Baskets.** ag-equity DE/MOS/ADM/MOO; grain DBA/CORN/WEAT; abnormal of "
            "SPY.\n"
            "- **Headline.** One-sample *t* on the print-day AR (both baskets) + Wilson "
            "up-rate.\n"
            "- **Robustness.** 20×1,000-draw random-calendar placebo; event window "
            "[−1..+5] with per-offset *t* (multiple-comparison caveat); bootstrap CI on "
            "the post-print drift.\n"
            "- **Third axis.** Paired grain−equity day-0 difference.\n"
            "- **Execution.** Timer = long the basket (excess of SPY) at day-0 close, "
            "hold {1,5,10,21}, 2× one-way cost × NAV; gross AND net; vs unconditional "
            "baseline.\n"
            "- **Regime.** Labelled monthly D2+ proxy, high-third vs rest, regime lagged "
            "one month (no look-ahead).\n"
            "- **Control.** Synthetic tape, planted-bump knob; the null must not fire "
            "across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Coverage funnel — the grain ETFs don't reach the early droughts\n\n"
            "All 21 escalations land on the ag-equity tape (DE/ADM span the whole "
            "sample; MOS from 2004, MOO from 2007 — the earliest events run on the names "
            "that existed). The grain basket exists only from DBA's 2007 launch, so its "
            "test runs on the 16 post-2007 events — named, not zero-filled."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0e = st.day0_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    d0g = st.day0_stats(AR_GR, EVENTS['date'], PRE, POST)\n"
            "    n_eq, n_gr = d0e['n'], d0g['n']\n"
            "else:\n"
            "    n_eq, n_gr = R['eq_n'], R['gr_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 3.2))\n"
            "ax.barh(['ag-equity (DE/MOS/ADM/MOO)', 'grain (DBA/CORN/WEAT)'], [n_eq, n_gr],\n"
            "        color=[GREY, AMBER])\n"
            "for i, v in enumerate([n_eq, n_gr]): ax.annotate(f'{v} events', (v, i), va='center', ha='right', color='w')\n"
            "ax.set_xlabel('drought escalations covered (of 21)')\n"
            "ax.set_title('Grain-ETF era covers 16 of 21 escalations'); plt.tight_layout(); plt.show()\n"
            "print('ag-equity n =', n_eq, '| grain n =', n_gr)"
        ),
        md(
            "> 💡 In plain words: coverage is named on the Signal axis. The 5 pre-2007 "
            "escalations (2000, 2002, 2003, 2005, 2006) have no grain ETF and are dropped "
            "from the grain test; the ag-equity basket runs on DE/ADM for the very "
            "earliest, gaining MOS (2004) and MOO (2007) as they list."
        ),
        md(
            "### 4b · The headline — print-day AR, one-sample t, both baskets"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0e = st.day0_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    d0g = st.day0_stats(AR_GR, EVENTS['date'], PRE, POST)\n"
            "    means = [d0e['mean']*1e4, d0g['mean']*1e4]; ts = [d0e['t'], d0g['t']]\n"
            "else:\n"
            "    means = [R['eq_day0_bps'], R['gr_day0_bps']]; ts = [R['eq_day0_t'], R['gr_day0_t']]\n"
            "labels = ['ag-equity', 'grain']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(labels, means, color=[RED if t <= -2 else GREY for t in ts], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('print-day AR vs SPY (bps)')\n"
            "a1.set_title('Wrong sign, both baskets')\n"
            "a2.bar(labels, ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('one-sample t'); a2.set_title('Neither near |t|=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"ag-equity {R['eq_day0_bps']:+.2f} bps t={R['eq_day0_t']:.2f} | \"\n"
            "      f\"grain {R['gr_day0_bps']:+.2f} bps t={R['gr_day0_t']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: ag-equity **{R['eq_day0_bps']:+.2f} bps** "
            f"(*t* = {R['eq_day0_t']:.2f}, up-rate {R['eq_up']}/{R['eq_n']}), grain "
            f"**{R['gr_day0_bps']:+.2f} bps** (*t* = {R['gr_day0_t']:.2f}). Both negative, "
            "both a coin flip. The drought print is not a positive catalyst for the ag "
            "complex — it is not a catalyst at all."
        ),
        md(
            "### 4c · The random-calendar placebo — is the print calendar special?\n\n"
            "For each basket, redraw N random non-drought sessions, 20 seeds × 1,000 "
            "draws, and compare the observed print-day mean to that null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0e = st.day0_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    draws = np.concatenate([st.placebo_distribution(AR_EQ, d0e['n'], PRE, POST,\n"
            "                            n_draws=1000, seed=740+s, stat='day0') for s in range(20)])*1e4\n"
            "    obs = d0e['mean']*1e4\n"
            "else:\n"
            "    rng = np.random.default_rng(740)\n"
            "    draws = rng.normal(R['eq_pl_mean_bps'], R['eq_pl_sd_bps'], R['pl_draws'])\n"
            "    obs = R['eq_day0_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='random-calendar null (20k)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean print-day AR of a random N-Thursday draw (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'ag-equity placebo p = {R[\"eq_pl_p\"]:.3f} (grain p = {R[\"gr_pl_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"ag-equity: observed {R['eq_day0_bps']:+.2f} vs null {R['eq_pl_mean_bps']:+.2f} \"\n"
            "      f\"(sd {R['eq_pl_sd_bps']:.2f}) -> p={R['eq_pl_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed print-day move sits deep in the bulk of "
            f"the null (right-tail *p* = {R['eq_pl_p']:.3f} equities, {R['gr_pl_p']:.3f} "
            "grain). A random set of Thursdays does *better* than the drought calendar "
            "most of the time — the calendar carries no information."
        ),
        md(
            "### 4d · Event anatomy — the [−1..+5] window and the look-elsewhere trap"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    days = list(cp.index); mean_ar = list(cp['mean_ar']*1e4); ts = list(cp['t'])\n"
            "else:\n"
            "    days = sorted(R['event']); mean_ar = [R['event'][k][0] for k in days]\n"
            "    ts = [R['event'][k][2] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts]\n"
            "ax.bar(days, mean_ar, color=cols)\n"
            "for d, m, t in zip(days, mean_ar, ts):\n"
            "    ax.annotate(f't={t:+.1f}', (d, m), ha='center',\n"
            "                va='bottom' if m >= 0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('offset from print (day 0 = release session)')\n"
            "ax.set_ylabel('mean abnormal return (bps)')\n"
            "ax.set_title('7 offsets, one crosses |t|=2 (day +5) — a textbook multiple-comparison fluke')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: of 7 offsets, exactly one (day +5, *t* = "
            f"{R['event'][5][2]:+.2f}) crosses the bar — the ~1-in-7 you'd expect by "
            "chance testing 7 things, with no mechanism for a five-day-delayed "
            "one-session pop. It fails the day-0 headline, the placebo, and the timer. "
            "This is exactly the look-elsewhere artifact study 707 flagged at its "
            "offset +2 — a warning about eyeballing event windows, not a signal."
        ),
        md(
            "### 4e · Post-print drift + bootstrap CI"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pe = st.post_event_car(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    w, kept = st.stack_windows(AR_EQ, EVENTS['date'], PRE, POST)\n"
            "    per = w[:, PRE+1:PRE+1+POST].sum(axis=1)\n"
            "    lo, hi = st.block_bootstrap_ci(per)\n"
            "    drift, dt = pe['mean']*1e4, pe['t']; lo, hi = lo*1e4, hi*1e4\n"
            "else:\n"
            "    drift, dt = R['drift_bps'], R['drift_t']; lo, hi = R['drift_ci']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 3.0))\n"
            "ax.errorbar([drift], [0], xerr=[[drift-lo], [hi-drift]], fmt='o', color=AMBER,\n"
            "            capsize=6, lw=2, ms=10, label='5-day post-print drift, 95% bootstrap CI')\n"
            "ax.axvline(0, c=RED, lw=1.2, ls='--')\n"
            "ax.set_yticks([]); ax.set_xlabel('cumulative abnormal return, [+1..+5] (bps)')\n"
            "ax.set_title(f'Drift {drift:+.0f} bps (t={dt:.2f}, placebo p={R[\"drift_pl_p\"]:.3f}) — CI straddles 0')\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print(f'drift {R[\"drift_bps\"]:+.2f} bps  t={R[\"drift_t\"]:.2f}  '\n"
            "      f'CI [{R[\"drift_ci\"][0]:.1f}, {R[\"drift_ci\"][1]:.1f}] bps')"
        ),
        md(
            f"> 💡 In plain words: the 5-day drift is mildly positive "
            f"(**{R['drift_bps']:+.0f} bps**) but *t* = {R['drift_t']:.2f}, placebo "
            f"*p* = {R['drift_pl_p']:.3f}, and the bootstrap CI **[{R['drift_ci'][0]:.0f}, "
            f"{R['drift_ci'][1]:.0f}] bps** straddles zero. Suggestive, not certified — "
            "and the timer below shows it isn't bankable."
        ),
        md(
            "### 4f · Third axis — does grain react harder than the equities?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex = st.basket_extra_move(AR_EQ, AR_GR, EVENTS['date'], PRE, POST)\n"
            "    gm, em, diff, dt, n = ex['grain_mean']*1e4, ex['equity_mean']*1e4, ex['mean_diff']*1e4, ex['t'], ex['n']\n"
            "else:\n"
            "    gm, em, diff, dt, n = R['grain_mean_bps'], R['equity_mean_bps'], R['diff_bps'], R['diff_t'], R['diff_n']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['grain day-0', 'ag-equity day-0', 'grain - equity'], [gm, em, diff],\n"
            "       color=[AMBER, GREY, RED], width=.55)\n"
            "for i, v in enumerate([gm, em, diff]): ax.annotate(f'{v:+.0f}', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('bps (same 16 grain-era events)')\n"
            "ax.set_title(f'Grain reacts LESS, not more (diff t={dt:.2f})'); plt.tight_layout(); plt.show()\n"
            "print(f'grain {R[\"grain_mean_bps\"]:+.2f}  equity {R[\"equity_mean_bps\"]:+.2f}  '\n"
            "      f'diff {R[\"diff_bps\"]:+.2f} (t={R[\"diff_t\"]:.2f}, n={R[\"diff_n\"]})')"
        ),
        md(
            f"> 💡 In plain words: the claim's cleanest prediction — the weather-exposed "
            f"grain vehicles react harder — has the **wrong sign**: grain "
            f"(**{R['grain_mean_bps']:+.0f} bps**) moves *more negatively* than the "
            f"equities (**{R['equity_mean_bps']:+.0f} bps**) on the print, difference "
            f"*t* = {R['diff_t']:.2f}. Grain futures/ETFs price the weather forecast "
            "continuously; the weekly Monitor tells them nothing new. **H3 rejected.**"
        ),
        md(
            "### 4g · The costed timer — \"buy the drought\""
        ),
        code(
            "holds = [1, 5, 10, 21]\n"
            "if HAVE_REAL:\n"
            "    gross, net5, base = [], [], []\n"
            "    for h in holds:\n"
            "        lg = st.trade_it(EQ_RET, SPY_RET, EVENTS['date'], hold=h, cost_bps=0.0)\n"
            "        ln = st.trade_it(EQ_RET, SPY_RET, EVENTS['date'], hold=h, cost_bps=5.0)\n"
            "        gross.append(st.summarize_trade(lg, 'ret_gross')['mean_bps'])\n"
            "        net5.append(st.summarize_trade(ln, 'ret_net')['mean_bps'])\n"
            "        fwd = st.abnormal_vs_bench(EQ_RET, SPY_RET).rolling(h).sum().shift(-h)\n"
            "        base.append(float(fwd.mean()*1e4))\n"
            "else:\n"
            "    gross = [R['timer'][h][0] for h in holds]; net5 = [R['timer'][h][1] for h in holds]\n"
            "    base = [R['timer'][h][4] for h in holds]\n"
            "x = np.arange(len(holds)); w = 0.27\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x - w, gross, w, color=GREEN, label='gross')\n"
            "ax.bar(x, net5, w, color=AMBER, label='net @ 5 bps')\n"
            "ax.bar(x + w, base, w, color=GREY, label='unconditional baseline')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('ag excess of SPY (bps)'); ax.legend()\n"
            "ax.set_title('Every horizon sub-2 net; the edge decays into the baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h in holds: print(f\"hold {h:>2d}d net {R['timer'][h][1]:+7.2f} bps t={R['timer'][h][2]:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the best net case is the 5-day hold at "
            f"**{R['timer'][5][1]:+.0f} bps** (*t* = {R['timer'][5][2]:+.2f}, "
            f"{R['timer'][5][3]}% win) — above its {R['timer'][5][4]:+.0f} bps "
            f"unconditional baseline but well short of *t* = 2, and at 10 bps one-way "
            f"cost it slips to {R['timer_net10_5d']:+.0f} bps (*t* = "
            f"{R['timer_net10_5d_t']:.2f}). By 21 days the 'drought' net "
            f"({R['timer'][21][1]:+.0f} bps) is barely above the {R['timer'][21][4]:+.0f} "
            f"bps you'd earn holding ag-vs-SPY unconditionally. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4h · The drought-regime test (labelled monthly proxy)\n\n"
            "Split months by the labelled D2+ severity proxy **known at the month's "
            "start** (one shift — no look-ahead); compare that month's forward "
            "ag-equity-minus-SPY return, high-drought third vs the rest. *(The proxy is "
            "an approximate labelled series — used only here, never under a real-tape "
            "banner.)*"
        ),
        code(
            "if HAVE_REAL:\n"
            "    proxy = data.drought_proxy(); m_ar = st.monthly_abnormal(EQ_RET, SPY_RET)\n"
            "    reg = st.regime_stats(proxy, m_ar, hi_pct=66.0, cost_bps=5.0)\n"
            "    hi_m, lo_m, wt = reg['hi_mean']*1e4, reg['lo_mean']*1e4, reg['welch_t']\n"
            "    n_hi, n_lo = reg['n_hi'], reg['n_lo']\n"
            "else:\n"
            "    hi_m, lo_m, wt = R['reg_hi_bps'], R['reg_lo_bps'], R['reg_welch_t']\n"
            "    n_hi, n_lo = R['reg_hi_n'], R['reg_lo_n']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar([f'high-drought\\n(n={n_hi})', f'other months\\n(n={n_lo})'], [hi_m, lo_m],\n"
            "       color=[AMBER, GREY], width=.5)\n"
            "for i, v in enumerate([hi_m, lo_m]): ax.annotate(f'{v:+.0f} bps/mo', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean forward ag abnormal return (bps/mo)')\n"
            "ax.set_title(f'High-drought months are a hair BEHIND (Welch t={wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"high {R['reg_hi_bps']:+.2f} vs low {R['reg_lo_bps']:+.2f} bps/mo, \"\n"
            "      f\"Welch t={R['reg_welch_t']:+.2f}, n={R['reg_n']} months\")"
        ),
        md(
            f"> 💡 In plain words: being in the driest third of months buys you nothing — "
            f"ag is **{R['reg_hi_bps']:+.0f} bps/mo** vs the market in high-drought "
            f"months, actually *behind* the **{R['reg_lo_bps']:+.0f} bps/mo** of the "
            f"rest (Welch *t* = {R['reg_welch_t']:+.2f}). No level effect, no event "
            "effect. **H4 rejected.**"
        ),
        md(
            "### 4i · Faithful-engine & power control\n\n"
            "Deterministic random-walk tape with 21 scheduled synthetic 'print' dates "
            "and a TUNABLE planted day-0 bump that fades over 5 sessions. Null (bump=0) "
            "checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(*data.synthetic_world(bump=0.0, seed=740+s))['t']\n"
            "                    for s in range(20)])\n"
            "p1 = st.synthetic_detect(*data.synthetic_world(bump=0.01, seed=740))\n"
            "p2 = st.synthetic_detect(*data.synthetic_world(bump=0.02, seed=740))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40, label='null (bump=0) x20')\n"
            "ax.scatter([1], [p1['t']], color=AMBER, s=90, zorder=5, label='planted +1%')\n"
            "ax.scatter([2], [p2['t']], color=RED, s=90, zorder=5, label='planted +2%')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted +1%', 'planted +2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Quiet null, planted bumps light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 | planted +1% t={p1[\"t\"]:+.2f} +2% t={p2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|t|≥2 in {R['syn_null_fire']}/{R['syn_null_seeds']} seeds; a planted +1% "
            f"bump reads t={R['syn_planted1_t']:.2f}, +2% reads t={R['syn_planted2_t']:.2f}. "
            "The machinery detects a real print-day effect when one is planted — the "
            "real-tape null is genuine. *(A faithful-engine / power check only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — print-day ag-equity AR **{R['eq_day0_bps']:+.2f} bps** "
            f"(*t* = {R['eq_day0_t']:.2f}, placebo *p* = {R['eq_pl_p']:.3f}) and grain "
            f"**{R['gr_day0_bps']:+.2f} bps** (*t* = {R['gr_day0_t']:.2f}, *p* = "
            f"{R['gr_pl_p']:.3f}) — both the wrong sign, both indistinguishable from a "
            f"random calendar. The 5-day drift (+{R['drift_bps']:.0f} bps) is sub-2 "
            f"(*t* = {R['drift_t']:.2f}, CI straddles 0); the one |t|≥2 offset is a "
            "look-elsewhere fluke; the regime split is Welch *t* = "
            f"{R['reg_welch_t']:+.2f}.\n"
            f"- **Tradability `MIRAGE`** — no \"buy the drought\" horizon clears *t* ≥ 2 "
            f"net of costs; best case 5-day net **{R['timer'][5][1]:+.0f} bps** "
            f"(*t* = {R['timer'][5][2]:.2f}), decaying to *t* = {R['timer'][21][2]:.2f} at "
            "21 days as the edge merges into the always-hold baseline.\n"
            f"- **\"Does a drought print move the ag complex?\" `BUSTED`** — event study, "
            "random-calendar placebo, grain-vs-equity paired test (diff *t* = "
            f"{R['diff_t']:.2f}) and drought-regime split all agree: the weekly Monitor "
            "carries no tradable content for the ag names, because the market has already "
            "priced the weather forecast it confirms."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The lesson is about *anticipated* public information.** The Drought "
            "Monitor is a slow-moving, continuously-forecast physical condition confirmed "
            "on a known schedule — exactly the kind of release efficient-markets priors "
            "say should carry near-zero surprise. A genuine weather edge lives in the "
            "forecast *surprise*, not the Monitor's *confirmation*.\n"
            "- **The natural sequel** swaps the anticipated Monitor for a real scheduled "
            "*shock*: USDA WASDE crop-report surprises on grain **futures** (not "
            "equity-diluted ETFs), where the announcement-effect literature (Adjemian "
            "2012) finds genuine, if fleeting, reactions. That is where a supply-shock "
            "edge, if it survives anywhere, should appear.\n"
            "- **Dedup map:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(broad-market news-shock event study), "
            "[313-geopolitical-shock](../../313-geopolitical-shock/) (geopolitical shock "
            "calendar), [637-fomc-vol-crush](../../637-fomc-vol-crush/) (a *scheduled "
            "public release* event study on index vol). None test whether a scheduled "
            "weekly public drought index is tradable news for the ag equity + grain "
            "complex — this study's own axis.\n\n"
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
