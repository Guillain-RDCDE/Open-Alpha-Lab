"""Generate the two narrative notebooks for Study 637 (FOMC Vol Crush).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^VIX/SPY/SVXY
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^VIX/SPY 1994-01-03
# -> 2026-06-30, SVXY 2011-10-04 -> 2026-06-30; 261 hardcoded scheduled FOMC decisions).
R = dict(
    start="1994-01-03", end="2026-06-30", n_fomc=261, n_rest=7916,
    cal_lo="1994-02-04", cal_hi="2026-06-17",
    fomc_pts=-0.415, rest_pts=+0.014, gap_pts=-0.429,
    fomc_logpct=-1.97, rest_logpct=+0.07,
    welch_pts=-3.51, welch_log=-3.94, nw_t=-3.47,
    hit=172, hit_pct=65.9, wilson=(60.0, 71.4),
    placebo_p=0.00005, placebo_mean=+0.016, placebo_sd=0.105, placebo_draws=20000,
    # event window: offset -> (mean pts, welch t)
    event={-5: (-0.050, -0.53), -4: (-0.079, -0.79), -3: (-0.014, -0.18),
           -2: (+0.301, +2.27), -1: (+0.050, +0.35), 0: (-0.415, -3.45),
           1: (+0.020, +0.10), 2: (-0.105, -1.22), 3: (+0.120, +0.82)},
    runup_pts=+0.209, runup_t=+0.90,
    spy_fomc=1.557, spy_rest=1.296, spy_t=+4.04,
    era_early=-0.549, era_early_n=139, era_early_t=-4.81,
    era_late=-0.262, era_late_n=122, era_late_t=-1.21, era_diff_t=+1.15,
    # SVXY third axis
    sv_gross=58.9, sv_net5=48.9, sv_net10=38.9, sv_rest=10.9, sv_t=+1.58,
    sv_n=118, sv_hit=61.9, sv_worst=-10.5,
    sv_ann5=3.91, sv_ann10=3.11,
    sv_post_gross=14.7, sv_post_t=+0.24, sv_post_n=67,
    # synthetic control
    syn_null_mean=-0.04, syn_null_sd=1.04, syn_null_fire=0, syn_planted_t=-15.67,
    fp_vix="e876dcb8d22b", fp_spy="d0ce62443ce7", fp_svxy="a876bb11537c",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Retail_capture%3F: Mixed](https://img.shields.io/badge/Retail_capture%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from fomc_vol_crush import data, strategy as st

FOMC = data.fomc_calendar()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    VIX, SPY, SVXY = data.load_real()
    DF = st.day_frame(VIX, SPY, FOMC)
else:
    VIX = SPY = SVXY = DF = None
print("real cache present:", HAVE_REAL, "| scheduled FOMC decisions:", len(FOMC),
      "| tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the fear index really collapse the afternoon the Fed speaks? 📉🕑\n"
            "### The FOMC vol crush — a Wall Street legend that turns out to be **true**, "
            "and still unbankable\n\n"
            + BADGES +
            "Eight times a year, on a date published **years in advance**, the Federal Reserve "
            "announces what it will do with interest rates. At 2:00 pm Eastern the statement hits "
            "the wires — and options traders swear the VIX, the market's \"fear gauge\", **deflates "
            "on the spot**: all the insurance bought against a Fed surprise expires worthless the "
            "moment the surprise is no longer possible.\n\n"
            "That's the claim we test: *uncertainty resolution you can set your watch to.* Most "
            "market folklore dies on contact with data. This one **doesn't** — and the interesting "
            "part is why you *still* can't get paid for it.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 261 scheduled decision dates (1994→2026) hardcoded from the Fed's "
            "own calendar archive — *scheduled* meetings only, because emergency cuts are surprises, "
            "the exact opposite of the claim. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the VIX drop on Fed decision days? | **Yes — sharply.** About "
            f"**{R['fomc_pts']:.1f} points (≈ {R['fomc_logpct']:.0f}%)** on the average decision "
            f"day, versus roughly zero on a normal day. It fell on **{R['hit_pct']:.0f}%** of the "
            f"261 decisions since 1994. Odds it's a calendar fluke: about **1 in 20,000**. |\n"
            "| Is the market actually *calm* that afternoon? | **No — that's the beautiful part.** "
            "The S&P's actual intraday swing is *bigger* than average on Fed days. Implied "
            "(expected) volatility collapses while realized volatility is loud — the definition of "
            "*uncertainty being resolved*, not of a quiet day. |\n"
            "| Does the drop bounce back tomorrow? | **No.** Day +1 is flat. The fear premium is "
            "gone for good — the event is over. |\n"
            f"| Can you trade it? | **Not really.** The VIX itself isn't buyable. The retail "
            f"short-vol ETF (SVXY) held for just the decision day made **+{R['sv_gross']:.0f} bps "
            f"per event** on paper — but that's statistically uncertifiable, nearly zero since "
            f"2018, and one bad Fed afternoon cost **{R['sv_worst']:.0f}%**. |\n\n"
            "> The physics is real. The paycheck is not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Ahead of every Fed meeting, option prices carry an extra premium for the chance "
            "of a shock. At 2 pm on decision day the shock either happens or it doesn't — either "
            "way the **uncertainty premium evaporates**, and the VIX crushes into the close.\"*\n\n"
            "It's not just floor talk: academics found that a striking share of all *sudden "
            "downward volatility jumps* land on FOMC afternoons (Amengual & Xiu 2018). And unlike "
            "most calendar stories, this one has a **mechanism you can state in one sentence**: "
            "scheduled uncertainty must leave the option surface the moment it resolves."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this is the cleanest possible demonstration that **implied volatility prices "
            "events, not calm** — and it should tempt every retail trader with the same idea: "
            "*short volatility into the meeting, collect the crush at 2 pm.* Entire products exist "
            "for exactly that trade (SVXY — and its late twin XIV, which died trying in 2018).\n\n"
            "So we ask three things: is the crush real, does the day around it behave the way the "
            "story says, and can the obvious one-day retail trade actually bank it after costs?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_fomc']}** scheduled FOMC decision dates from "
            f"{R['cal_lo']} to {R['cal_hi']}, hardcoded from the Fed's calendar archive. The "
            "statement lands at 2:00/2:15 pm ET; the VIX closes at 4:15 pm — so the daily close "
            "already contains the crush.\n"
            "- **The comparison.** The VIX's daily change on those 261 days vs the other "
            f"**{R['n_rest']:,}** trading days since 1994.\n"
            "- **The luck check.** Draw 261 random days instead, 20,000 times — how often does a "
            "random calendar produce a drop that big?\n"
            "- **The trade check.** Buy SVXY at the close *before* the decision (the date is public "
            "years ahead — no crystal ball needed), sell at the decision-day close, pay costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average daily VIX change on decision days vs every other day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.decision_day_stats(DF)\n"
            "    fp, rp = s['fomc_pts'], s['rest_pts']\n"
            "else:\n"
            "    fp, rp = R['fomc_pts'], R['rest_pts']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['Fed decision days\\n(n=261)','all other days\\n(n=7,916)'], [fp, rp],\n"
            "       color=[RED, GREY], width=.6)\n"
            "for i,v in enumerate([fp, rp]): ax.annotate(f'{v:+.3f} pts',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average daily VIX change (points)')\n"
            "ax.set_title('The fear gauge deflates the afternoon the Fed speaks')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'decision days {fp:+.3f} pts   other days {rp:+.3f} pts')"
        ),
        md(
            f"There it is: **{R['fomc_pts']:.3f} points** on the average decision day (about "
            f"**{R['fomc_logpct']:.0f}%** of the VIX level) against **{R['rest_pts']:+.3f}** — i.e. "
            f"nothing — on a normal day. The VIX fell on **{R['hit']}/261 = {R['hit_pct']:.1f}%** of "
            "decisions. The quants notebook shows a random calendar beats this about **1 time in "
            "20,000** — this is not luck.\n\n"
            "**Next, the anatomy.** What does the whole week around the decision look like?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, FOMC)\n"
            "    ks, ms = list(ev.index), list(ev['mean_pts'])\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "cols = [RED if k==0 else (AMBER if k==-2 else GREY) for k in ks]\n"
            "ax.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "for i,v in enumerate(ms): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days relative to the Fed decision (0 = decision day)')\n"
            "ax.set_ylabel('average VIX change (points)')\n"
            "ax.set_title('A faint build-up, one big crush, no bounce-back')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('offsets:', {k: round(m,3) for k,m in zip(ks, ms)})"
        ),
        md(
            "Three things to read off this chart:\n\n"
            f"1. **The build-up is faint.** The five days before the meeting add only about "
            f"**{R['runup_pts']:+.2f} points in total** — statistically indistinguishable from "
            "nothing. The \"VIX always ramps into the Fed\" half of the folklore barely exists.\n"
            f"2. **The crush is the event.** Day 0: **{R['event'][0][0]:+.2f} points**, by far the "
            "largest bar.\n"
            f"3. **It doesn't come back.** Day +1 is **{R['event'][1][0]:+.2f}** — flat. The fear "
            "premium isn't *displaced*, it's *extinguished*.\n\n"
            "**And here's the elegant bit** — the market isn't calm that day. It's loud:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.spy_range_stats(DF)\n"
            "    a, b = rg['fomc_range_pct'], rg['rest_range_pct']\n"
            "else:\n"
            "    a, b = R['spy_fomc'], R['spy_rest']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(['Fed days','other days'], [R['fomc_logpct'], R['rest_logpct']], color=[RED, GREY], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('IMPLIED vol: collapses')\n"
            "a1.set_ylabel('avg daily VIX change (%)')\n"
            "for i,v in enumerate([R['fomc_logpct'], R['rest_logpct']]):\n"
            "    a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a2.bar(['Fed days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "a2.set_title('REALIZED range: louder than average')\n"
            "a2.set_ylabel('avg SPY high-low range (% of prev close)')\n"
            "for i,v in enumerate([a, b]): a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: Fed days {a:.3f}%  vs other {b:.3f}%')"
        ),
        md(
            f"The S&P swings **{R['spy_fomc']:.2f}%** high-to-low on Fed days vs "
            f"**{R['spy_rest']:.2f}%** normally — the decision day is one of the *noisier* days of "
            "the calendar. Yet implied volatility drops ~2% into that same close. That combination "
            "— *loud day, deflating fear gauge* — is exactly what \"uncertainty resolution\" looks "
            "like, and it's why this isn't some artefact of quiet afternoons.\n\n"
            "**Finally, the trade.** Can you actually collect the crush with the retail short-vol "
            "ETF, held for just that one day?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sv = st.svxy_capture(SVXY, FOMC, cost_bps=5.0)\n"
            "    sv2 = st.svxy_capture(SVXY, FOMC, cost_bps=5.0, start=data.SVXY_DELEV)\n"
            "    g, g2 = sv['gross_bps'], sv2['gross_bps']\n"
            "else:\n"
            "    g, g2 = R['sv_gross'], R['sv_post_gross']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['full sample\\n2011-2026 (n=118)','since the 2018\\nde-leverage (n=67)'],\n"
            "       [g, g2], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([g, g2]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg SVXY return per decision day (bps, gross)')\n"
            "ax.set_title('The retail echo: positive on paper, fading, and never certifiable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SVXY per decision day: full {g:+.1f} bps  |  post-2018 {g2:+.1f} bps')"
        ),
        md(
            f"On paper: **+{R['sv_gross']:.0f} bps per event** gross, ~**+{R['sv_net5']:.0f} bps** "
            f"after costs, ≈ **+{R['sv_ann5']:.1f}%/yr** from eight afternoons of \"work\". But — \n\n"
            f"- the statistics **can't certify it** (it's within luck's reach at *t* = {R['sv_t']:.2f}; "
            "the bar is 2);\n"
            f"- since SVXY halved its exposure after the 2018 \"Volmageddon\", the edge is "
            f"**+{R['sv_post_gross']:.0f} bps at t = {R['sv_post_t']:.2f}** — statistically nothing;\n"
            f"- the worst single decision day cost **{R['sv_worst']:.1f}%** — one bad afternoon "
            "erases a year of edge — and this product's twin (XIV) went to zero in a single day.\n\n"
            "Why doesn't the obvious trade work when the crush is so real? Because SVXY doesn't hold "
            "the VIX — nobody can. It holds VIX **futures**, and futures traders *also* know the Fed "
            "calendar: they price the crush in ahead of time. What's left for you is only the "
            "*surprise* part — which is, by definition, a coin flip."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The VIX drops **~{abs(R['fomc_pts']):.1f} points (~2%)** into the "
            f"decision-day close, fell on **{R['hit_pct']:.0f}%** of 261 scheduled decisions since "
            "1994, the drop survives every robustness check we throw at it, and it *sticks*. One of "
            "the rare pieces of market folklore that is simply true.\n"
            "- **Tradability — Mirage.** The index isn't buyable, futures pre-price the event, and "
            "the retail echo is thin, uncertifiable, and carries gap-to-zero tail risk.\n"
            "- **\"Retail can bank it with a 1-day SVXY hold\"? — Mixed.** Right direction, real "
            "positive average in sample — but statistically unprovable, roughly zero since 2018, and "
            "one −10.5% afternoon away from a very bad year."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The event premium is the general lesson.** Earnings, CPI mornings, elections — "
            "implied vol prices *scheduled events*, and the premium leaves the surface the moment "
            "the event resolves. The Fed is just the cleanest, most metronomic case.\n"
            "- **Where the pros actually play it** is in the *options on the day* (straddles, "
            "calendars) and the *futures basis* into the meeting — instruments that isolate the "
            "event premium instead of riding a leveraged ETP through it.\n"
            "- **Sibling studies:** the [pre-FOMC equity drift](../../517-pre-fomc-drift/) (returns "
            "*before* the statement), the [FOMC cycle](../../135-fomc-cycle/) (week-parity returns) "
            "and the [blackout window](../../322-fomc-blackout/) — none of them is this study: the "
            "VIX **on** the day.\n\n"
            "*Think you can catch the crush better than SVXY does? Show a net, certifiable edge in "
            "the options themselves — after the market-maker's spread on Fed afternoons — then "
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
            "# The FOMC Vol Crush — a quantitative teardown 🔬\n"
            "### Decision-day Welch/HAC splits · a 20-seed random-calendar placebo · the [−5..+3] "
            "event anatomy · the realized-vs-implied cross-check · the press-conference-era "
            "contrast · an honest SVXY capture test · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **implied vol collapses into the scheduled FOMC decision close** — is one of "
            "the rare folklore claims with a stated mechanism (event-premium expiry), an academic "
            "anchor (Amengual-Xiu 2018) and a hard calendar. The job here is to measure it "
            "honestly, then ask the only question that pays: *is any of it tradable?*\n\n"
            "> ⚠️ **Data note.** ^VIX daily OHLC + SPY raw OHLC (1994→2026) + SVXY adjusted closes "
            "(2011→2026), yfinance, cached; **261 hardcoded scheduled decision dates** from the "
            "Fed calendar archive (scheduled only — emergency actions are surprises, excluded by "
            "construction; the pre-empted 2020-03-18 date is kept, named quirk). No survivorship "
            "on the Signal axis (indices); **SVXY's survivor status is named on the third axis** "
            "(XIV died 2018-02-05). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_vix"] +
            "` / `" + R["fp_spy"] + "` / `" + R["fp_svxy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | decision-day ΔVIX **{R['fomc_pts']:+.3f} pts "
            f"({R['fomc_logpct']:+.2f}%)** vs {R['rest_pts']:+.3f}: Welch **t = {R['welch_pts']:.2f}** "
            f"(pts) / **{R['welch_log']:.2f}** (log), NW **t = {R['nw_t']:.2f}**, hit "
            f"{R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]), placebo "
            f"**p = {R['placebo_p']:.5f}** |\n"
            f"| **Tradability** | `MIRAGE` | spot ^VIX untradable; SVXY echo t = {R['sv_t']:.2f} "
            f"full-sample, **t = {R['sv_post_t']:.2f}** post-2018, worst day {R['sv_worst']:.1f}% |\n"
            f"| **Retail capture?** | `MIXED` | +{R['sv_net5']:.0f} bps net/event in sample, "
            f"uncertifiable; ≈ 0 since the −0.5× de-leverage |\n\n"
            "> 💡 In plain words: the crush is real physics on the tape; the tradable echo is a "
            "rumor the statistics decline to confirm."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\Delta v_t$ be the daily ^VIX close-to-close change and $D_t \\in \\{0,1\\}$ the "
            "scheduled-decision-day flag (known *ex ante* — the calendar is published years "
            "ahead). The statement (14:00/14:15 ET) predates the ^VIX close (16:15 ET), so the "
            "decision-day bar contains the resolution. The claims:\n\n"
            "- **H₁ (crush).** $E[\\Delta v_t \\mid D_t=1] \\ll E[\\Delta v_t \\mid D_t=0]$ — large, "
            "systematic, not a calendar coincidence.\n"
            "- **H₂ (anatomy).** A run-up before ($\\Delta v > 0$ pre-meeting) and no bounce-back "
            "after (the premium is extinguished, not displaced).\n"
            "- **H₃ (resolution, not calm).** Realized SPY range on decision days is *not* below "
            "average — implied falls while realized is loud.\n"
            "- **H₄ (capture).** A retail short-vol expression (SVXY, decision day only) banks it "
            "net of costs.\n\n"
            "We find **H₁ strongly supported**, **H₂ half-supported** (crush sticks; the run-up is "
            "faint, t = 0.90), **H₃ supported** (range t = +4.04), **H₄ not certified** (t = 1.58; "
            "t = 0.24 post-2018)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Decision days are **single, non-overlapping events**, so the planned primary is a "
            "**Welch t** on the group split (points and logs — logs tame the level dependence of "
            "vol-point moves). Because daily ΔVIX is serially correlated, we cross-check with a "
            "**Newey-West (5-lag) t** on the dummy regression $\\Delta v_t = a + b D_t$ — the slope "
            "*is* the mean gap. The hit rate carries a **Wilson interval** (no conditional claim "
            "without uncertainty), the placebo draws 261 random non-FOMC days **20,000 times "
            "(20 seeds × 1,000)**, and the era split (2011-04-27, first post-meeting press "
            "conference) is justified *ex ante*, not snooped — and tested as a **difference**, not "
            "eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_fomc']} scheduled decisions {R['cal_lo']} → {R['cal_hi']}, "
            "hardcoded (Fed calendar archive; scheduled only). All 261 sit on the tape.\n"
            f"- **Tape.** ^VIX OHLC + SPY raw OHLC {R['start']} → {R['end']}; SVXY total-return "
            "closes from 2011-10 inception. As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** Welch t (pts + log) + NW(5) t + Wilson hit rate + 20-seed placebo.\n"
            "- **Anatomy.** Event window [−5..+3], per-offset Welch t vs far days; cumulative "
            "run-up per meeting, one-sample t.\n"
            "- **Cross-check.** SPY (H−L)/prev-close split on the same days.\n"
            "- **Execution (third axis).** Enter SVXY at the prior close (calendar public — zero "
            "look-ahead), exit the decision close; 2 × one-way cost × NAV per event; long-only, "
            "no borrow.\n"
            "- **Control.** Synthetic OU vol index, planted crush knob; the null must not fire "
            "across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo\n\n"
            "Welch t on ΔVIX (points and logs), NW t on the dummy regression, and the "
            "random-calendar null. In the notebook we run a lighter placebo (4 seeds × 500 draws) "
            "and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.decision_day_stats(DF)\n"
            "    print(f\"FOMC-day ΔVIX {s['fomc_pts']:+.3f} pts ({s['fomc_logpct']:+.2f}%)  vs  \"\n"
            "          f\"other {s['rest_pts']:+.3f} pts ({s['rest_logpct']:+.2f}%)\")\n"
            "    print(f\"Welch t = {s['welch_t_pts']:+.2f} (pts) / {s['welch_t_log']:+.2f} (log)   \"\n"
            "          f\"NW(5) t = {s['nw_t_pts']:+.2f}\")\n"
            "    print(f\"hit {s['hit_down']}/{s['n_fomc']} = {s['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    pl = st.placebo_pvalue(DF, n_draws_per_seed=500, n_seeds=4)\n"
            "    obs, draws = pl['obs'], pl['draws']\n"
            "else:\n"
            "    obs = R['fomc_pts']\n"
            "    rng = np.random.default_rng(637)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 261 days (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed FOMC-day mean {obs:+.3f} pts')\n"
            "ax.set_xlabel('mean ΔVIX of a random 261-day calendar (points)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Far outside the luck cloud: canonical p = {R['placebo_p']:.5f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.3f}, \"\n"
            "      f\"sd {R['placebo_sd']:.3f}, p = {R['placebo_p']:.5f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['fomc_pts']:+.3f} pts** sits about four "
            f"placebo-sigmas below the null's center ({R['placebo_mean']:+.3f} ± "
            f"{R['placebo_sd']:.3f}); **p = {R['placebo_p']:.5f}**. With Welch t = "
            f"**{R['welch_log']:.2f}** (log) and NW t = **{R['nw_t']:.2f}**, H₁ clears the desk bar "
            "three separate ways."
        ),
        md(
            "### 4b · Anatomy — run-up, crush, persistence\n\n"
            "Per-offset means with Welch t vs far-from-meeting days; the run-up is tested as a "
            "**cumulative per-meeting** quantity (one-sample t across 261 meetings)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, FOMC)\n"
            "    ks = list(ev.index); ms = list(ev['mean_pts']); ts = list(ev['welch_t'])\n"
            "    ru = st.runup_stats(DF, FOMC)\n"
            "    ru_m, ru_t = ru['mean_runup_pts'], ru['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][1] for k in ks]; ru_m, ru_t = R['runup_pts'], R['runup_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k==0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean ΔVIX (pts)')\n"
            "a1.set_title('Event anatomy: faint build-up, one crush, no bounce-back')\n"
            "a2.bar([str(k) for k in ks], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from decision)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cumulative run-up [-5..-1]: {ru_m:+.3f} pts/meeting (t = {ru_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: only two bars matter — day 0 (**{R['event'][0][0]:+.3f}**, "
            f"t = {R['event'][0][1]:+.2f}) and, mildly, day −2 (**{R['event'][-2][0]:+.3f}**, "
            f"t = {R['event'][-2][1]:+.2f}). The cumulative pre-meeting run-up is "
            f"**{R['runup_pts']:+.2f} pts at t = {R['runup_t']:+.2f}** — folklore says the VIX "
            "\"always ramps into the Fed\"; the tape says *barely, if at all*. And day +1 is "
            f"**{R['event'][1][0]:+.3f}** (t = {R['event'][1][1]:+.2f}): the crush is a level "
            "reset, not a dip that mean-reverts — exactly what premium-expiry predicts and what a "
            "liquidity glitch would not."
        ),
        md(
            "### 4c · Resolution, not calm — the realized cross-check\n\n"
            "If decision days were simply *quiet*, a VIX drop would be trivial. They are the "
            "opposite:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.spy_range_stats(DF)\n"
            "    a, b, t = rg['fomc_range_pct'], rg['rest_range_pct'], rg['welch_t']\n"
            "else:\n"
            "    a, b, t = R['spy_fomc'], R['spy_rest'], R['spy_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(['decision days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('SPY (H-L)/prev close, mean (%)')\n"
            "ax.set_title(f'Realized range is LOUDER on decision days (Welch t = {t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: FOMC {a:.3f}%  vs other {b:.3f}%   Welch t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the S&P moves **{R['spy_fomc']:.2f}%** high-to-low on decision "
            f"days vs **{R['spy_rest']:.2f}%** otherwise (t = {R['spy_t']:+.2f}) — among the louder "
            "days on the calendar — while implied vol falls ~2% into the same close. Implied prices "
            "the *future*: once the statement is out, tomorrow has less uncertainty in it, however "
            "wild today was. H₃ holds; the crush is resolution, not calm."
        ),
        md(
            "### 4d · The era contrast — justified split, tested as a difference\n\n"
            "Split at **2011-04-27** (the first post-meeting press conference — the structural "
            "change in decision communication; chosen ex ante, not snooped)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(DF, data.PRESSER_SPLIT)\n"
            "    e, l = ec['early_pts'], ec['late_pts']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_early'], R['era_late']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['1994 - 2011-04\\n(n=139)','2011-04 - 2026\\n(n=122)'], [e, l],\n"
            "       color=[RED, AMBER], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    ax.annotate(f'{v:+.3f} pts\\n(within-era t={t_:+.2f})',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('FOMC-day mean ΔVIX (pts)')\n"
            "ax.set_title(f'Smaller since the press-conference era - but the decay is NOT certified '\n"
            "             f'(diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.3f} (t={et:+.2f})  late {l:+.3f} (t={lt:+.2f})  diff t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the crush has roughly halved ({R['era_early']:+.3f} → "
            f"{R['era_late']:+.3f} pts) since the Fed started explaining itself on camera — "
            "plausible (dot plots, forward guidance and VIX-futures pre-pricing all move resolution "
            f"*ahead* of the day) — but the difference is only t = {R['era_diff_t']:+.2f}, so we "
            "may not claim decay. Honest reading: full-tape effect decisive; the recent slice alone "
            f"(t = {R['era_late_t']:+.2f}) could not certify it by itself. Said out loud on the "
            "front card."
        ),
        md(
            "### 4e · The third axis — the honest SVXY capture test\n\n"
            "Enter the prior close (zero look-ahead — the calendar is public years ahead), exit the "
            "decision close, pay 2 × one-way costs per event. SVXY is the *surviving* short-vol ETP "
            "— XIV died 2018-02-05 — and cut its exposure to −0.5× on 2018-02-27 (both named)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.svxy_capture(SVXY, FOMC, cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    sv2 = st.svxy_capture(SVXY, FOMC, cost_bps=5.0, start=data.SVXY_DELEV)\n"
            "    g = rows[0]['gross_bps']; n5, n10 = rows[0]['net_bps'], rows[1]['net_bps']\n"
            "    tv, g2, t2 = rows[0]['welch_t'], sv2['gross_bps'], sv2['welch_t']\n"
            "    worst = rows[0]['worst_day_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['sv_gross'], R['sv_net5'], R['sv_net10']\n"
            "    tv, g2, t2, worst = R['sv_t'], R['sv_post_gross'], R['sv_post_t'], R['sv_worst']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['gross','net 5 bps','net 10 bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('bps per decision day'); a1.set_title(f'Positive on paper (Welch t = {tv:+.2f})')\n"
            "a2.bar(['full sample\\n(n=118)','post-2018\\n-0.5x (n=67)'], [g, g2], color=[AMBER, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(g,tv),(g2,t2)]):\n"
            "    a2.annotate(f'{v:+.1f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('gross bps per decision day')\n"
            "a2.set_title('...and gone since the de-leverage')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f} -> net {n5:+.1f} / {n10:+.1f} bps  (t={tv:+.2f});  '\n"
            "      f'post-2018 {g2:+.1f} bps (t={t2:+.2f});  worst day {worst:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: +{R['sv_net5']:.0f} bps net per event, ~+{R['sv_ann5']:.1f}%/yr "
            f"from eight afternoons — *if* you believe a t of {R['sv_t']:.2f}. We don't (the bar is "
            f"2), and the post-de-leverage slice (+{R['sv_post_gross']:.0f} bps, "
            f"t = {R['sv_post_t']:.2f}) says the modern edge is ≈ 0. Structurally: SVXY holds VIX "
            "**futures**, whose basis prices the scheduled crush *before* the day — the holder "
            "collects only the surprise term. Add a {:.1f}% worst day and a product class that "
            "has already gapped to zero (XIV) — **H₄ not certified; Tradability = MIRAGE, "
            "third axis = MIXED**.".format(R["sv_worst"])
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting vol index, scheduled pseudo-decisions every 32nd business "
            "day, TUNABLE planted crush. The null (crush = 0) is checked over **20 seeds** — never "
            "a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, dec = data.synthetic_world(crush=0.0, seed=637 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, dec)['welch_t_pts'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, dec = data.synthetic_world(crush=1.0, seed=637)\n"
            "planted_t = st.synthetic_detect(close, dec)['welch_t_pts']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (crush=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted crush = 1.0 pt')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (decision vs rest)')\n"
            "ax.set_title('Control: no null fires; a planted crush lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted 1-point crush reads t = {R['syn_planted_t']:.2f}. The machinery is "
            "unbiased — the real-tape t ≈ −3.9 is the genuine article. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — decision-day ΔVIX **{R['fomc_pts']:+.3f} pts "
            f"({R['fomc_logpct']:+.2f}%)** vs {R['rest_pts']:+.3f}: Welch t = "
            f"**{R['welch_pts']:.2f} / {R['welch_log']:.2f}** (pts/log), NW t = **{R['nw_t']:.2f}**, "
            f"hit {R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]), "
            f"placebo p = {R['placebo_p']:.5f}; the crush *sticks* (day +1 t = "
            f"{R['event'][1][1]:+.2f}) and lands on realized-loud days (range t = {R['spy_t']:+.2f}). "
            f"Caveat named: post-2011 slice alone t = {R['era_late_t']:+.2f}; decay uncertified "
            f"(diff t = {R['era_diff_t']:+.2f}).\n"
            f"- **Tradability `MIRAGE`** — spot ^VIX untradable; the SVXY echo is "
            f"+{R['sv_net5']:.0f} bps net/event at t = {R['sv_t']:.2f} (full) and "
            f"+{R['sv_post_gross']:.0f} bps at t = {R['sv_post_t']:.2f} (post-2018), with a "
            f"{R['sv_worst']:.1f}% worst day and gap-to-zero class risk (XIV). Futures pre-price "
            "the event; what's left is the surprise, i.e. a coin flip.\n"
            f"- **Retail capture? `MIXED`** — direction right, magnitude positive in sample, "
            "certification failed, modern edge ≈ 0, tails fat. Not *wrong* — just not *bankable*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is the announcement premium.** Amengual-Xiu (2018) show "
            "downward vol jumps cluster on FOMC afternoons; Fernandez-Perez et al. (2017) time the "
            "intraday drop. The same physics runs CPI mornings, ECB Thursdays and earnings — the "
            "professional expressions live in event straddles and the futures basis, not in ETPs.\n"
            "- **Why the futures leak the edge:** the VIX futures basis into a meeting *is* the "
            "market's estimate of the crush; holding SVXY through the event collects only the "
            "realization-minus-expectation residual. Measuring that basis directly (VX term "
            "structure around meetings) is the natural sequel study.\n"
            "- **Dedup map:** [517-pre-fomc-drift](../../517-pre-fomc-drift/) (equity returns "
            "*before*), [67-fed-drift](../../67-fed-drift/) (its decay), "
            "[135-fomc-cycle](../../135-fomc-cycle/) (week parity), "
            "[322-fomc-blackout](../../322-fomc-blackout/) (the pre-meeting window), "
            "[605-vix-settlement-day](../../605-vix-settlement-day/) (settlement mornings, which "
            "*controls out* FOMC days — here promoted from confounder to subject).\n\n"
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
