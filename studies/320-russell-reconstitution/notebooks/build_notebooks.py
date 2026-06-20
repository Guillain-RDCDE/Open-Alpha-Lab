"""Generate (and execute) the two narrative notebooks for Study 320 (Russell-Reconstitution).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The reconstitution
calendar is hardcoded and the IWM tape uses the study-level cache
(``_cache/bars_IWM_daily.parquet``) if present; otherwise the cells fall back to the
deterministic synthetic tape and clearly banner that the headline numbers are quoted from
``R`` (mirroring docs/results.md). So the notebooks re-run for any reader, offline.

``build_notebooks.py`` writes each notebook then executes it in place with nbconvert (no
``--allow-errors``); every code cell ends up with a real execution_count and no error output.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17, IWM).
R = dict(
    n=26,
    runup_days=5,
    runup_mean_pct=0.5788,
    runup_std_pct=2.7477,
    base_mean_pct=0.2150,
    excess_bps=36.39,
    up_rate_pct=61.54,
    base_up_rate_pct=55.25,
    t_vs_base=0.675,
    hac_t_excess=0.894,
    perm_p=0.5474,
    june_excess_bps=35.31,
    june_hac_t=0.868,
    event_day_mean_pct=0.0353,
    event_day_hac_t=-0.05,
    post_mean_pct=0.0175,
    post_hac_t=-0.325,
    gross_per_event_bps=57.88,
    net_per_event_bps=55.88,
    net_cum=1.145224,
    net_cagr_pct=0.5229,
    # window-length sweep (excess bps, hac_t)
    rd1_excess=-1.0, rd1_t=-0.05,
    rd3_excess=57.0, rd3_t=1.743,
    rd5_excess=36.4, rd5_t=0.894,
    rd10_excess=-41.5, rd10_t=-0.857,
    # sub-periods (excess bps, hac_t)
    sp1_excess=45.89, sp1_t=0.719,
    sp2_excess=-39.92, sp2_t=-0.896,
    sp3_excess=94.71, sp3_t=1.499,
    fp="ec8b35339b52",
    year_start=2000,
    year_end=2025,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from russell_reconstitution import data, strategy as st

def _have_cache():
    try:
        data.fetch_daily("IWM", fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("IWM cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17, IWM)
R = dict(
    n=26, runup_days=5, runup_std_pct=2.7477,
    runup_mean_pct=0.5788, base_mean_pct=0.2150, excess_bps=36.39,
    up_rate_pct=61.54, base_up_rate_pct=55.25,
    t_vs_base=0.675, hac_t_excess=0.894, perm_p=0.5474,
    june_excess_bps=35.31, june_hac_t=0.868,
    event_day_mean_pct=0.0353, event_day_hac_t=-0.05,
    post_mean_pct=0.0175, post_hac_t=-0.325,
    gross_per_event_bps=57.88, net_per_event_bps=55.88,
    net_cum=1.145224, net_cagr_pct=0.5229,
    rd1_excess=-1.0, rd1_t=-0.05, rd3_excess=57.0, rd3_t=1.743,
    rd5_excess=36.4, rd5_t=0.894, rd10_excess=-41.5, rd10_t=-0.857,
    sp1_excess=45.89, sp1_t=0.719, sp2_excess=-39.92, sp2_t=-0.896,
    sp3_excess=94.71, sp3_t=1.499,
    year_start=2000, year_end=2025,
)

# Build the real event windows + baseline if the cache is present, else the synthetic NULL.
TAPE = "REAL (IWM, cached)" if HAVE_REAL else "SYNTHETIC null (offline fallback)"
if HAVE_REAL:
    BARS = data.fetch_daily("IWM", fetch=False)
else:
    BARS, _ = data.synthetic_daily(start_year=R['year_start'], end_year=R['year_end'],
                                   premium_bps=0.0, seed=320)
WIN = st.build_windows(BARS, runup_days=R['runup_days'])
BASE = st.runup_windows_baseline(BARS, runup_days=R['runup_days'])
print("tape:", TAPE, "| events:", len(WIN))
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Russell-Reconstitution -- can you front-run the June reshuffle?\n"
            "### The most predictable forced trade of the year -- and why you still can't get paid for it\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Front--runnable%3F: No](https://img.shields.io/badge/Front--runnable%3F-No-8b949e?style=flat-square)\n\n"
            "Once a year, FTSE Russell rebuilds its US indices, and on the **reconstitution "
            "Friday** in late June every index fund must trade to the new membership -- one of "
            "the highest-volume closing auctions of the entire US year. The date is published "
            "*years* in advance and the flow only goes one way. Surely you can stand in front of "
            "that freight train and collect? That's the folklore. This notebook checks whether "
            "the small-cap *index* actually moves when you'd need it to.\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat, the same-month "
            "control, the window-length sweep and the n=26 power calculation? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a real run-up into the Russell reshuffle? | **No.** The small-cap index "
            "(IWM) drifts up **+0.58%** over the five sessions before reconstitution Friday -- but "
            "the index drifts up on *any* random five-session stretch (+0.22%). The **+36 bps** "
            "extra is noise. |\n"
            "| Is it statistically real? | **No.** HAC *t* = **+0.89**, permutation *p* = **0.55** "
            "-- nowhere near the desk's |t| >= 2 bar. |\n"
            "| Does the reconstitution day itself pop? | **No.** The famous high-volume Friday "
            "moves ~**+0.04%** -- a coin flip. Huge *volume*, no *price*. |\n"
            "| Could you trade it? | **No.** Even taken at face value it's ~**+0.5%/yr** while you "
            "sit in cash 360 days -- and there's no significant edge underneath it anyway. |\n\n"
            "> A textbook **None / Mirage**: the forced flow is real and enormous, but it's so "
            "predictable that the *index price* is already efficient by the time you could trade "
            "it. Big volume is not the same thing as a tradable move."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The Russell reconstitution forces billions of dollars of one-directional "
            "small-cap buying into one closing auction in late June. The date is on the calendar "
            "years ahead. Buy the index before it and ride the wave.\"*\n\n"
            "Every June FTSE Russell recomputes the Russell 1000 / 2000 / 3000 from a late-May "
            "snapshot, publishes the new lists in mid-June, and makes them **effective after the "
            "close of the reconstitution Friday** (traditionally the last Friday of June). Index "
            "funds that track these benchmarks *must* hold the new weights by that close, so a "
            "giant, mechanical, pre-announced trade lands in that one session."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the index reliably popped into a date you can mark on a calendar, that would be a "
            "free lunch: no forecasting, no stock-picking, just show up once a year. It would also "
            "tell us something deep -- that even *fully transparent* forced flow moves prices. The "
            "more likely story is the opposite: because everyone can see it coming, the run-up is "
            "exactly the thing that gets competed away. This study is about which world we live in."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong null.** Small-caps drift up over time, so *any* five-day window tends "
            "to be positive. The honest question is whether the run-up beats a **random five-day "
            "window**, not zero.\n"
            "2. **Is it Russell, or is it June?** June might just be a good month for small-caps. "
            "So we also compare against a **same-month (June) baseline**.\n"
            "3. **Tiny n.** One reconstitution a year means only **26 events** (2000-2025). With "
            "small-caps that volatile, an effect has to be *huge* to show up as real -- and a "
            "handful of years can look impressive by pure chance.\n\n"
            "We test on the IWM (iShares Russell 2000) daily tape, with the reconstitution Fridays "
            "hardcoded from the FTSE Russell calendar in this study's `data.py`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the five-session run-up into each reconstitution Friday, "
            "compared with the everyday five-session drift."
        ),
        code(
            "et = data.recon_table()\n"
            "print('Reconstitution events:', len(et), f\"({et['year'].min()}-{et['year'].max()})\")\n"
            "print('Every event weekday:', sorted(et['weekday'].unique()), '(4 = Friday)')\n"
            "if HAVE_REAL:\n"
            "    rm = float(WIN['runup_ret'].mean()*100)\n"
            "    bm = float(BASE.mean()*100)\n"
            "    up = float((WIN['runup_ret']>0).mean()*100)\n"
            "else:\n"
            "    rm, bm, up = R['runup_mean_pct'], R['base_mean_pct'], R['up_rate_pct']\n"
            "print()\n"
            "print(f'5-session run-up mean:      {rm:+.4f}%')\n"
            "print(f'Everyday 5-session drift:   {bm:+.4f}%  <- the honest baseline')\n"
            "print(f'Run-up up-rate:             {up:.1f}% of years')\n"
            "print(f'[tape: {TAPE}]')"
        ),
        md("**The run-up vs the everyday drift -- is the extra bit real, or just the usual drift?**"),
        code(
            "if HAVE_REAL:\n"
            "    rm = float(WIN['runup_ret'].mean()*100); bm = float(BASE.mean()*100)\n"
            "else:\n"
            "    rm, bm = R['runup_mean_pct'], R['base_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Run-up into\\nreconstitution', 'Any random\\n5 sessions'], [rm, bm],\n"
            "              color=[AMBER, GREY], width=0.5)\n"
            "ax.axhline(bm, ls='--', c=GREY, lw=1.5)\n"
            "ax.set_ylabel('Mean cumulative return (%)')\n"
            "ax.set_title('The run-up is barely above the everyday drift -- and the gap is noise')\n"
            "for b, v in zip(bars, [rm, bm]):\n"
            "    ax.annotate(f'{v:+.3f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Excess over the everyday drift: {(rm-bm)*100:+.1f} bps -- looks like something...')\n"
            "print(f'[tape: {TAPE}]')"
        ),
        md(
            "There *is* a small positive gap -- about **+36 bps** over the everyday drift, up in "
            "**62%** of years. It looks like something. But with only 26 events and small-caps "
            "this jumpy, the real test is whether that gap is bigger than what pure chance throws "
            "up. (The quants notebook puts a t-stat on it: **+0.89** -- not even close.)"
        ),
        md("**The reconstitution day itself: all that volume, where's the price move?**"),
        code(
            "if HAVE_REAL:\n"
            "    ev = float(WIN['event_day_ret'].mean()*100)\n"
            "    po = float(WIN['post_ret'].mean()*100)\n"
            "else:\n"
            "    ev, po = R['event_day_mean_pct'], R['post_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Run-up\\n(5 sessions)', 'Reconstitution\\nFriday', 'Give-back\\n(next 5)'],\n"
            "              [rm if HAVE_REAL else R['runup_mean_pct'], ev, po],\n"
            "              color=[AMBER, GREY, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return (%)')\n"
            "ax.set_title('Huge volume on the Friday -- but the price move is a rounding error')\n"
            "for b, v in zip(bars, [rm if HAVE_REAL else R['runup_mean_pct'], ev, po]):\n"
            "    ax.annotate(f'{v:+.3f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Reconstitution-Friday mean: {ev:+.4f}% -- a coin flip.')\n"
            "print(f'[tape: {TAPE}]')"
        ),
        md(
            "The reconstitution Friday -- the day of the famous closing-auction surge -- moves the "
            "*index* about **+0.04%**, and the week after gives back nothing in particular. The "
            "enormous volume reshuffles *which* small-caps are held; it does not push the basket "
            "up or down in a way you could have bet on."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Run-up excess +36 bps, HAC *t* = **+0.89**, perm *p* = 0.55; "
            "the reconstitution day and the give-back week are coin flips; the same-month June "
            "control doesn't change the story. Indistinguishable from noise.\n"
            "- **Tradability -- Mirage.** Even at face value the front-run rule is ~+0.5%/yr while "
            "sitting in cash 360 days -- and there's no significant edge to harvest in the first "
            "place.\n"
            "- **Front-runnable? -- No.** The flow is real and gigantic, but it's the most "
            "telegraphed trade of the year, so the index price is already efficient. Volume is not "
            "a price move."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy': buy IWM at the close five sessions before reconstitution Friday, sell "
            "on the event close, once a year. Gross, and net of a modest 1 bp one-way (2 bps "
            "round-trip) cost on NAV."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tradability(WIN['runup_ret'], cost_bps_oneway=1.0)\n"
            "    g, nbps, ncum = tr['gross_mean_per_event']*1e4, tr['net_mean_per_event']*1e4, tr['net_cum_growth']\n"
            "else:\n"
            "    g, nbps, ncum = R['gross_per_event_bps'], R['net_per_event_bps'], R['net_cum']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.0))\n"
            "bars = ax.bar(['Gross\\nper event', 'Net\\n(2 bps cost)'], [g, nbps], color=[AMBER, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean return per event (bps)')\n"
            "ax.set_title('Survives costs -- but it is a once-a-year sliver of noise')\n"
            "for b, v in zip(bars, [g, nbps]):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Net per event: {nbps:+.1f} bps | compounded over 26 yrs: x{ncum:.3f} (~{(ncum**(1/26)-1)*100:.2f}%/yr)')\n"
            "print('Costs are not the killer -- the absence of a real edge is. [tape: ' + TAPE + ']')"
        ),
        md(
            "Costs barely register (one round-trip a year), so the rule 'survives costs' on paper "
            "-- but that's the wrong frame: there's no statistically real edge to begin with. A "
            "~0.5%/yr contribution from a coin-flip signal, while parked in cash 360 days, is "
            "dominated by simply holding the index. **No signal, no vehicle.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a run-up premium into a "
            "synthetic tape and shows the machinery only detects it once it's *enormous* (~40 "
            "bps/day) -- because n=26 is so low-powered. The real tape's +36 bps cumulative is far "
            "below that floor.\n"
            "- **The single-name cousin:** [Study 249 -- Index-Inclusion](../../249-index-inclusion/) "
            "tests the *individual stocks* added to the S&P 500 (the inclusion pop). That's a "
            "cross-section; this is the whole index on a calendar date -- two different angles on "
            "forced index flow.\n"
            "- **A real calendar signal for contrast:** [Study 287 -- Easter-Effect](../../287-easter-effect/) "
            "runs the *same* event-study machinery and lands Real -- which is exactly why running "
            "the identical apparatus here and getting None is meaningful.\n\n"
            "*Think the action is in the single names, not the index? Fork this and test the "
            "additions vs deletions spread around reconstitution -- but mind the survivorship and "
            "the borrow on the short leg.*"
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
            "# Russell-Reconstitution -- a quantitative teardown\n"
            "### 26 reconstitutions * IWM daily * run-up / event-day / give-back windows * "
            "excess vs unconditional & same-month * one-sample t * Newey-West HAC t * permutation "
            "* window & sub-period sweeps * n=26 power * synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Front--runnable%3F: No](https://img.shields.io/badge/Front--runnable%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test the five-session "
            "run-up into each late-June Russell reconstitution Friday (plus the event day and the "
            "give-back week) against the unconditional rolling-window baseline **and** a same-month "
            "June control, with a one-sample t-test, a Newey-West HAC t on the excess, a "
            "permutation test, a window-length sweep, a sub-period split, and a synthetic positive "
            "control.\n\n"
            "> **Not investment advice.** Real data: IWM (iShares Russell 2000) daily close-to-"
            "close, split/dividend-adjusted (auto_adjust), 2000-2025; reconstitution calendar "
            "hardcoded in `data.py`. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Signal-axis caveat:** this is an *index ETF*, so survivorship is moot at the index "
            "level (no cross-section is picked). The binding limitation is **n = 26** -- a "
            "low-power sample where only a large effect could ever clear |t| = 2.\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Run-up excess **+36 bps**, one-sample t vs baseline **+0.68**, "
            "Newey-West HAC t **+0.89**, perm p **0.55**. Event-day & give-back ≈ 0; same-month "
            "June control unchanged (HAC t +0.87); sign flips across thirds. |\n"
            "| **Tradability** | `MIRAGE` | Net +56 bps/event survives costs, but ~**+0.5%/yr** in "
            "cash 360 days -- and no significant edge underneath. The forced flow is real; the "
            "harvestable price move is not. |\n"
            "| **Front-runnable?** | `NOT SUPPORTED` | The most telegraphed trade of the year; the "
            "index ETF is efficiently priced by the time you could trade it. |\n\n"
            "> 💡 **In plain words:** the run-up looks slightly positive, but it's the kind of "
            "positive any random five days of a drifting small-cap index shows. With 26 shots, "
            "that's exactly what noise looks like."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the cumulative IWM return over the five sessions ending on the "
            "reconstitution Friday of year $t$, and $\\mu_5$ the unconditional mean of *all* "
            "rolling five-session block returns over the same window. The hypotheses:\n\n"
            "- **H1 (run-up excess).** $E[r_t] > \\mu_5$ -- the run-up beats the everyday "
            "five-session drift.\n"
            "- **H1' (not just June).** $E[r_t] > \\mu_5^{\\text{June}}$ -- it also beats a "
            "five-session window that *ends in June*, ruling out generic June seasonality.\n"
            "- **H2 (event-day pop).** The reconstitution Friday's own return is positive on "
            "average.\n\n"
            "We **reject** H1 (excess +36 bps, t = +0.68, HAC t = +0.89), find H1' equally null "
            "(June excess +35 bps, HAC t = +0.87), and find no H2 event-day pop (+0.04%, HAC t = "
            "-0.05)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "A tradable index-level run-up would be a rare thing: a calendar-dated, "
            "stock-picking-free edge from *transparent* forced flow. But the single-name "
            "reconstitution literature (Madhavan 2003; Chen-Noronha-Singal 2006) already notes the "
            "predictability invites front-running that competes the effect away, and the "
            "index-level move is diluted by ~2000 names where adds and drops roughly offset. The "
            "interesting question is whether *anything* survives at the index. It doesn't -- and "
            "the n=26 power calc shows we couldn't even have detected a modest one."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** IWM daily close-to-close, auto-adjusted, 2000-2025; the run-up = close 5 "
            "sessions before the reconstitution Friday → event close (calendar-known, **no "
            "execution lag**).\n"
            "- **Statistic.** Mean of the 26 run-up returns vs the unconditional mean of all "
            "rolling 5-session block returns $\\mu_5$ (the excess) -- and vs the June-only blocks.\n"
            "- **Correct null.** One-sample t against $\\mu_5$ (not 0); plus a permutation null of "
            "random 26-window draws from the baseline.\n"
            "- **Inference.** `ttest_1samp`; Newey-West HAC t (Bartlett, lag 2) on the excess; "
            "permutation test (10,000 draws); window-length sweep (1/3/5/10); three-way sub-period "
            "split; n=26 power calculation.\n"
            "- **Positive control.** Synthetic daily tape with a planted run-up premium.\n\n"
            "> 💡 **In plain words:** we ask not 'did it go up?' but 'did it go up *more than a "
            "random five days would*, by more than chance can explain across 26 tries?'"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The correct baseline: run-up excess and HAC t\n\n"
            "Test the run-up against the everyday five-session drift $\\mu_5$ and report the "
            "Newey-West HAC t on the excess (the |t| >= 2 hurdle)."
        ),
        code(
            "es = st.event_stats(WIN['runup_ret'], BASE, n_permutations=10_000, seed=320)\n"
            "print(f\"n run-up windows:     {es['n']}\")\n"
            "print(f\"Run-up mean:          {es['ev_mean']*100:+.4f}%\")\n"
            "print(f\"Unconditional 5-sess: {es['base_mean']*100:+.4f}%  (the honest baseline)\")\n"
            "print(f\"Excess:               {es['excess_mean']*1e4:+.2f} bps\")\n"
            "print(f\"Up-rate:              {es['up_rate']*100:.1f}%  (vs {es['base_up_rate']*100:.1f}% base rate)\")\n"
            "print()\n"
            "print(f\"One-sample t vs base: {es['t_vs_base']:+.3f}  (p={es['p_vs_base']:.4f})\")\n"
            "print(f\"Newey-West HAC t:     {es['hac_t_excess']:+.3f}   <- below the |t|>=2 bar\")\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "> The run-up beats the everyday drift by +36 bps -- but with HAC t = **+0.89** and "
            "perm p = 0.55 that is squarely inside the noise band. Unlike a real calendar effect "
            "(e.g. Study 287), this never clears the hurdle."
        ),
        md(
            "### 4b - Is it Russell, or is it June?\n\n"
            "The run-up always ends in late June. Compare against a five-session window that *also* "
            "ends in June -- the same-month control."
        ),
        code(
            "june_base = st.runup_windows_baseline(BARS, runup_days=R['runup_days'], month=6)\n"
            "es_j = st.event_stats(WIN['runup_ret'], june_base, n_permutations=10_000, seed=320)\n"
            "print(f\"June 5-session baseline: {es_j['base_mean']*100:+.4f}%\")\n"
            "print(f\"Excess vs June:          {es_j['excess_mean']*1e4:+.2f} bps\")\n"
            "print(f\"HAC t (excess vs June):  {es_j['hac_t_excess']:+.3f}\")\n"
            "print(f\"Permutation p (vs June): {es_j['perm_p']:.4f}\")\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "Against the **June-only** baseline the excess is essentially unchanged (+35 bps, HAC "
            "t = **+0.87**). So it is not even a 'June is good for small-caps' effect dressed up as "
            "reconstitution -- there is just nothing there. *(A same-month control can support "
            "existence; here it confirms non-existence.)*"
        ),
        md(
            "### 4c - The permutation test: where does the run-up mean land?\n\n"
            "Draw 26 random 5-session windows from the baseline 10,000 times and see where the "
            "observed run-up mean sits in the null."
        ),
        code(
            "perm_means = es['perm_means'] * 100\n"
            "obs = float(WIN['runup_ret'].mean()*100)\n"
            "mu = float(BASE.mean()*100)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=45, color=GREY, alpha=0.7, label='Random 26-window means')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Run-up mean: {obs:+.3f}%')\n"
            "ax.axvline(mu, c=AMBER, lw=2, ls='--', label=f'Everyday drift: {mu:+.3f}%')\n"
            "ax.set_xlabel('Mean 5-session return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The run-up mean sits comfortably inside the null -- not in any tail')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Permutation p (two-sided): {es['perm_p']:.4f}\")\n"
            "print(f\"Percentile of run-up mean: {(perm_means < obs).mean():.0%} of draws below\")\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "The observed run-up mean lands near the **middle** of the null distribution -- the "
            "opposite of the right-tail placement a real effect produces. perm p = 0.55."
        ),
        md(
            "### 4d - Window-length sweep: no horizon clears the bar\n\n"
            "Maybe five sessions is the wrong window. Sweep 1, 3, 5, 10 -- if the edge were real it "
            "would be robust to the exact horizon."
        ),
        code(
            "rows = []\n"
            "for rd in (1, 3, 5, 10):\n"
            "    w = st.build_windows(BARS, runup_days=rd)\n"
            "    b = st.runup_windows_baseline(BARS, runup_days=rd)\n"
            "    e = st.event_stats(w['runup_ret'], b, n_permutations=3000, seed=320)\n"
            "    rows.append({'runup_days': rd, 'excess_bps': round(e['excess_mean']*1e4, 1),\n"
            "                 'hac_t': e['hac_t_excess'], 'perm_p': e['perm_p']})\n"
            "sweep = pd.DataFrame(rows)\n"
            "print(sweep.to_string(index=False))\n"
            "colors = [GREEN if abs(t) >= 2 else RED for t in sweep['hac_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.bar(sweep['runup_days'].astype(str), sweep['hac_t'], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2 hurdle'); ax.axhline(-2, c='k', ls=':', lw=1)\n"
            "ax.set_xlabel('Run-up length (sessions)'); ax.set_ylabel('HAC t (excess)')\n"
            "ax.set_title('No run-up length clears |t| = 2 (the 3-day is the best, still short)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "> 💡 **In plain words:** the best-looking window (3 sessions, t = +1.74) is itself a "
            "warning -- in a tiny sample you can always find the most flattering horizon, and even "
            "the cherry-picked one doesn't clear the bar. The 1- and 10-session windows are flat or "
            "negative."
        ),
        md(
            "### 4e - Sub-periods: the sign flips\n\n"
            "Split the 26 events into thirds and recompute the excess HAC t in each."
        ),
        code(
            "sp = st.subperiod_stats(WIN, BASE,\n"
            "                        breaks=((2000, 2008), (2009, 2016), (2017, 2025)))\n"
            "print(sp.to_string(index=False))\n"
            "colors = [GREEN if abs(t) >= 2 else AMBER for t in sp['hac_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.bar(sp['period'], sp['hac_t'], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1); ax.axhline(-2, c='k', ls=':', lw=1, label='|t| = 2')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('HAC t (excess)')\n"
            "ax.set_title('No stable effect -- the middle third is negative, none clears |t| = 2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "The middle third (2009-2016) is **negative**; the outer thirds are weakly positive; "
            "none clears |t| = 2. There is no durable, repeatable run-up -- just three small noisy "
            "samples."
        ),
        md(
            "### 4f - Power: what could n = 26 even detect?\n\n"
            "The minimum detectable excess at 80% power, two-sided, alpha = 0.05."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vol = float(WIN['runup_ret'].std(ddof=1))\n"
            "else:\n"
            "    vol = R['runup_std_pct']/100 if 'runup_std_pct' in R else 0.0275\n"
            "n = len(WIN)\n"
            "from scipy.stats import norm\n"
            "z_a, z_b = norm.ppf(1 - 0.05/2), norm.ppf(0.80)\n"
            "se = vol/np.sqrt(n); mde = (z_a + z_b)*se\n"
            "print(f'Run-up window vol: {vol*100:.3f}%, n = {n}')\n"
            "print(f'Standard error of the run-up mean: {se*100:.4f}% ({se*1e4:.0f} bps)')\n"
            "print(f'Minimum detectable excess (80% power): {mde*100:.3f}% ({mde*1e4:.0f} bps)')\n"
            "print()\n"
            "print(f\"Observed excess: ~{R['excess_bps']:+.0f} bps -- well below the ~{mde*1e4:.0f} bps detection floor.\")\n"
            "print('Even a moderately real run-up would be invisible at this sample size.')\n"
            "print(f\"[tape: {TAPE}]\")"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- run-up excess +36 bps, t vs baseline +0.68, HAC t +0.89, perm "
            "p 0.55; event-day +0.04% (HAC t -0.05); give-back ≈ 0; same-month June control "
            "unchanged (HAC t +0.87); the sign flips across thirds and no window length clears "
            "|t| = 2. Indistinguishable from noise.\n"
            "- **Tradability `MIRAGE`** -- net +56 bps/event survives costs, but ~+0.5%/yr in cash "
            "360 days, on top of a signal that isn't statistically there.\n"
            "- **Front-runnable? `NOT SUPPORTED`** -- the forced flow is the most predictable of "
            "the year; the index ETF is efficiently priced by the time you could act.\n\n"
            "*Signal-axis note:* survivorship is moot for an index ETF; the binding limitation is "
            "n = 26, and the power calc shows even a moderate effect would be undetectable here. "
            "So 'None' means 'no detectable index-level run-up', which is the honest claim."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- gross vs net\n\n"
            "Buy 5 sessions before reconstitution Friday, sell on the event close, once a year; "
            "1 bp one-way (2 bps round-trip) on NAV."
        ),
        code(
            "tr = st.tradability(WIN['runup_ret'], cost_bps_oneway=1.0)\n"
            "print(f\"Events:               {tr['n_events']}\")\n"
            "print(f\"Gross mean per event: {tr['gross_mean_per_event']*1e4:+.2f} bps\")\n"
            "print(f\"Cost drag (round-trip): {tr['cost_drag_per_event']*1e4:.1f} bps\")\n"
            "print(f\"Net mean per event:   {tr['net_mean_per_event']*1e4:+.2f} bps\")\n"
            "print()\n"
            "print(f\"Compounded (net) over {tr['n_events']} yrs: x{tr['net_cum_growth']:.4f}  ->  ~{tr['net_cagr']*100:.2f}%/yr\")\n"
            "print('Survives costs per event -- but it is a once-a-year sliver, and the signal is noise.')\n"
            "print(f\"[tape: {TAPE}]\")"
        ),
        md(
            "> 💡 **In plain words:** you 'survive costs' only because you trade once a year. That's "
            "not a feature -- it means the whole annual contribution is a single coin flip dressed "
            "up as a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a run-up premium *when one exists*, and stay quiet at zero? "
            "Plant a premium into a synthetic tape (same 26-event calendar) and sweep its size. The "
            "key reading is **how big** the premium must be -- because n = 26 is low-powered."
        ),
        code(
            "planted = [0, 5, 10, 20, 40]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_daily(start_year=2000, end_year=2025,\n"
            "                                   premium_bps=float(bps), seed=320)\n"
            "    w = st.build_windows(df_s, runup_days=5)\n"
            "    b = st.runup_windows_baseline(df_s, runup_days=5)\n"
            "    r = st.event_stats(w['runup_ret'], b, n_permutations=2000, seed=320)\n"
            "    results.append({'planted_bps_per_day': bps, 'excess_bps': round(r['excess_mean']*1e4, 1),\n"
            "                    'hac_t': r['hac_t_excess'], 'perm_p': r['perm_p']})\n"
            "res = pd.DataFrame(results)\n"
            "print(res.to_string(index=False))\n"
            "colors = [GREEN if (row['hac_t'] > 2 and row['perm_p'] < 0.05) else RED\n"
            "          for _, row in res.iterrows()]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(res['planted_bps_per_day'].astype(str), res['hac_t'], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2 hurdle')\n"
            "ax.set_xlabel('Planted run-up premium (bps/day)'); ax.set_ylabel('HAC t (excess)')\n"
            "ax.set_title('The engine only fires once the premium is large (~40 bps/day) -- n=26 is low-powered')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Quiet at 0; only detectable from ~40 bps/day (~200 bps cumulative). The real')\n"
            "print('+36 bps cumulative is an order of magnitude below that floor -> NONE is honest.')"
        ),
        md(
            "The engine stays silent at premium = 0 and only clears the bar once the planted run-up "
            "reaches ~40 bps/day (~200 bps cumulative over five sessions). That is the whole story "
            "of the n = 26 sample: a front-run edge would have to be *enormous* to be detectable, "
            "and the real tape's +36 bps cumulative excess is far below the floor. The synthetic is "
            "a **machinery proof only** -- it confirms the apparatus works, and it never backs the "
            "Signal stamp."
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


def _execute(name):
    """Execute a notebook in place with nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    path = os.path.join(HERE, name)
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
