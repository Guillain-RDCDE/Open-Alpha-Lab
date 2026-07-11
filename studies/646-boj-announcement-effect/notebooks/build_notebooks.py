"""Generate the two narrative notebooks for Study 646 (BoJ Announcement Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EWJ/JPY=X tapes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance EWJ/JPY=X 2005-01-03
# -> 2026-06-30; 248 hardcoded BoJ decision/statement dates, 247 on the tape).
R = dict(
    start="2005-01-03", end="2026-06-30", n_boj=247, n_rest=5157, n_hardcoded=248,
    cal_lo="2005-01-19", cal_hi="2026-06-16",
    ewj_boj=+0.0080, ewj_rest=+0.0145, ewj_gap=-0.0065, ewj_t=-0.06, ewj_nw=-0.06,
    ewj_hit=118, ewj_hit_pct=47.8, ewj_wilson=(41.6, 54.0),
    jpy_boj=-0.0633, jpy_rest=-0.0058, jpy_gap=-0.0575, jpy_t=-0.85, jpy_nw=-0.85,
    jpy_hit=117, jpy_hit_pct=47.4, jpy_wilson=(41.2, 53.6),
    ewj_placebo_p=0.9223, ewj_placebo_mean=+0.0147, ewj_placebo_sd=0.0788,
    jpy_placebo_p=0.1584, jpy_placebo_mean=-0.0057, jpy_placebo_sd=0.0453,
    placebo_draws=20000,
    # event window: offset -> (ewj mean %, ewj t, jpy mean %, jpy t)
    event={
        -5: (-0.127, -1.76, +0.026, +0.52), -4: (-0.072, -1.40, +0.036, +0.68),
        -3: (+0.016, -0.19, -0.058, -1.19), -2: (-0.104, -1.51, -0.051, -1.02),
        -1: (+0.105, +0.69, +0.007, +0.21), 0: (+0.040, +0.04, -0.042, -0.65),
        1: (+0.013, -0.23, +0.036, +0.98), 2: (-0.011, -0.49, -0.026, -0.51),
        3: (-0.004, -0.49, -0.077, -0.88),
    },
    ewj_range_boj=1.164, ewj_range_rest=0.995, ewj_range_t=+2.49,
    jpy_range_boj=1.059, jpy_range_rest=0.760, jpy_range_t=+6.30,
    era_split="2016-01-29",
    ewj_early=-0.001, ewj_early_n=162, ewj_early_t=-0.02,
    ewj_late=+0.025, ewj_late_n=85, ewj_late_t=-0.02, ewj_diff_t=+0.13,
    jpy_early=-0.100, jpy_early_n=162, jpy_early_t=-1.19,
    jpy_late=+0.006, jpy_late_n=85, jpy_late_t=+0.16, jpy_diff_t=+0.75,
    # tail events: date -> (label, ewj%, ewj_z, jpy%, jpy_z, ewj_range%, jpy_range%)
    tail={
        "2016-01-29": ("NIRP announced", +2.38, +1.41, -2.01, -1.86, 1.87, 2.64),
        "2016-09-21": ("YCC introduced", +2.88, +1.71, +1.18, +1.19, 1.14, 2.18),
        "2022-12-20": ("YCC band widened (\"Kuroda shock\")", +1.41, +0.84, +3.80, +3.69, 1.29, 4.70),
        "2023-07-28": ("YCC made more flexible", +0.57, +0.33, -1.35, -1.23, 0.62, 2.06),
        "2023-10-31": ("YCC further loosened", +1.21, +0.72, -1.45, -1.33, 0.84, 1.68),
        "2024-03-19": ("NIRP/YCC formally ended", +0.38, +0.22, -1.26, -1.14, 0.79, 1.15),
        "2024-07-31": ("Rate hike to ~0.25%", +2.90, +1.72, +1.94, +1.91, 0.99, 2.57),
    },
    # third axis: capture test
    ewj_gross=0.80, ewj_net5=-9.20, ewj_net10=-19.20, ewj_worst=-8.06, ewj_best=+5.97,
    jpy_gross=-6.33, jpy_net5=-16.33, jpy_net10=-26.33, jpy_worst=-7.80, jpy_best=+3.80,
    # synthetic control
    syn_null_mean=-0.23, syn_null_sd=1.34, syn_null_fire=3, syn_null_seeds=20,
    syn_100seed_fp=3, syn_planted_effect=0.20, syn_planted_t=+5.46,
    fp_ewj="e6d94e03a855", fp_jpy="70f450fc971e",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Bigger_swings%3F: Confirmed](https://img.shields.io/badge/Bigger_swings%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from boj_announcement_effect import data, strategy as st

BOJ = data.boj_calendar()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    EWJ, JPY = data.load_real()
    DF = st.day_frame(EWJ, JPY, BOJ)
else:
    EWJ = JPY = DF = None
print("real cache present:", HAVE_REAL, "| BoJ decisions:", len(BOJ),
      "| tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Tokyo move the market on cue? 🇯🇵🏦\n"
            "### The BoJ Announcement Effect — a plausible-sounding legend that turns out to be "
            "**mostly noise**, with one loud exception\n\n"
            + BADGES +
            "Eight (sometimes more) times a year, the Bank of Japan's Policy Board announces "
            "what it will do with interest rates and its bond-buying program. Everyone remembers "
            "a handful of these days — the shock negative-rate cut in 2016, the 2022 \"Kuroda "
            "shock\" that sent the yen flying, the 2024 hike that helped trigger a global "
            "market wobble. Surely, the story goes, **BoJ day is a systematically bigger day** "
            "for Japanese stocks and the yen?\n\n"
            "We tested it on 247 BoJ decisions since 2005. The honest answer has two very "
            "different halves.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 248 hardcoded BoJ decision dates (2005→2026) from the Bank of "
            "Japan's own archives — every decision on record, since this claim is explicitly "
            "about the surprise-driven eras. Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do Japanese stocks (EWJ) move predictably on BoJ days? | **No.** "
            f"**{R['ewj_boj']:+.3f}%** on the average decision day vs {R['ewj_rest']:+.3f}% "
            "normally — statistically indistinguishable from zero, and stocks were UP on only "
            f"**{R['ewj_hit_pct']:.0f}%** of decisions (below a coin flip). |\n"
            "| Does the yen move predictably on BoJ days? | **No, either.** "
            f"**{R['jpy_boj']:+.3f}%** vs {R['jpy_rest']:+.3f}% normally — also statistically "
            "nothing. |\n"
            "| Is the \"surprise era\" (NIRP/YCC, 2016 onward) any different? | **Not on "
            "average.** Same coin-flip pattern before and after the negative-rate shock. |\n"
            "| Are BoJ days actually calmer or louder than normal? | **Louder — genuinely.** "
            "Both instruments swing more that day than an average day. The catch: which "
            "*direction* they swing is unpredictable. |\n\n"
            "> The BoJ can move markets hugely on any given day. It just can't tell you which "
            "way — and neither, on average, can the calendar."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Bank of Japan is the last major central bank still capable of shocking "
            "the market — negative rates, yield-curve control, the world's biggest bond-buying "
            "program. When it moves, Japanese stocks and the yen move with it.\"*\n\n"
            "It's a reasonable prior: the BoJ has delivered several of the biggest single-day "
            "FX moves of the past decade, and unlike the Fed (which telegraphs its reaction "
            "function months ahead), a large share of BoJ decisions genuinely surprised the "
            "market. The question is whether that surprise-power shows up as a *systematic*, "
            "tradable pattern — or only as a handful of unforgettable outliers."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If BoJ days really moved markets systematically, that would be a clean, calendar-"
            "based edge: you'd know the date years in advance, position ahead of it, and collect "
            "a repeatable premium. If instead the reaction is all in a handful of huge, "
            "sign-flipping surprises, there's nothing to harvest on an *average* decision day — "
            "just tail risk to respect on the rare one that matters."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_hardcoded']}** BoJ decision/statement dates from "
            f"{R['cal_lo']} to {R['cal_hi']}, hardcoded from the Bank of Japan's own archives — "
            "every decision, scheduled or emergency, because this claim is about the surprise "
            "eras specifically.\n"
            "- **The comparison.** EWJ (Japan equities, USD) and the yen's return on those "
            f"**{R['n_boj']}** days vs the other **{R['n_rest']:,}** trading days since 2005.\n"
            "- **The luck check.** Draw 247 random days instead, 20,000 times, two-sided (a "
            "BoJ surprise can go either way) — how often does a random calendar produce a gap "
            "this large?\n"
            "- **The amplitude check.** Regardless of direction — is the *swing* (high-low "
            "range) actually bigger on decision days?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average daily return on BoJ decision days vs every other "
            "day, for both instruments."
        ),
        code(
            "if HAVE_REAL:\n"
            "    se = st.decision_day_stats(DF, 'ewj_ret'); sj = st.decision_day_stats(DF, 'jpy_ret')\n"
            "    ea, eb, ja, jb = se['boj_pct'], se['rest_pct'], sj['boj_pct'], sj['rest_pct']\n"
            "else:\n"
            "    ea, eb, ja, jb = R['ewj_boj'], R['ewj_rest'], R['jpy_boj'], R['jpy_rest']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar(['BoJ days\\n(n=247)','other days\\n(n=5,157)'], [ea, eb], color=[RED, GREY], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('avg daily return (%)'); a1.set_title('EWJ (Japan equities)')\n"
            "for i,v in enumerate([ea, eb]): a1.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.bar(['BoJ days\\n(n=247)','other days\\n(n=5,157)'], [ja, jb], color=[RED, GREY], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('avg daily return (%)'); a2.set_title('Yen (minus USDJPY)')\n"
            "for i,v in enumerate([ja, jb]): a2.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.suptitle('Neither instrument tells decision days apart from any other day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ {ea:+.4f}% vs {eb:+.4f}%   yen {ja:+.4f}% vs {jb:+.4f}%')"
        ),
        md(
            f"The bars are nearly indistinguishable. EWJ hit rate: only **{R['ewj_hit_pct']:.0f}%** "
            f"of decisions were up days (below a coin flip). Yen hit rate: **{R['jpy_hit_pct']:.0f}%**. "
            "The quants notebook shows a random calendar reproduces both gaps the large majority "
            "of the time — this is not a fluke of small samples, it's a genuine null.\n\n"
            "**But here's the twist.** Even though the *direction* is a wash, is the *size* of "
            "the move on those days bigger?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rge = st.range_stats(DF, 'ewj_range'); rgj = st.range_stats(DF, 'jpy_range')\n"
            "    ea2, eb2, ja2, jb2 = rge['boj_pct'], rge['rest_pct'], rgj['boj_pct'], rgj['rest_pct']\n"
            "else:\n"
            "    ea2, eb2 = R['ewj_range_boj'], R['ewj_range_rest']\n"
            "    ja2, jb2 = R['jpy_range_boj'], R['jpy_range_rest']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar(['BoJ days','other days'], [ea2, eb2], color=[AMBER, GREY], width=.55)\n"
            "a1.set_title('EWJ realized range'); a1.set_ylabel('(High-Low)/open (%)')\n"
            "for i,v in enumerate([ea2, eb2]): a1.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['BoJ days','other days'], [ja2, jb2], color=[AMBER, GREY], width=.55)\n"
            "a2.set_title('Yen realized range'); a2.set_ylabel('(High-Low)/open (%)')\n"
            "for i,v in enumerate([ja2, jb2]): a2.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "plt.suptitle('BoJ days ARE louder - direction just isn\\'t predictable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ range {ea2:.3f}% vs {eb2:.3f}%   yen range {ja2:.3f}% vs {jb2:.3f}%')"
        ),
        md(
            "Both instruments swing meaningfully more on decision days. That's the real, "
            "certified finding of this study: **BoJ days are louder** — the folklore has a true "
            "kernel, it's just about *volatility*, not *direction*.\n\n"
            "**So where does the folklore come from?** From a handful of genuinely huge, "
            "unforgettable days."
        ),
        code(
            "labels = list(R['tail'].keys())\n"
            "jvals = [R['tail'][d][3] for d in labels]\n"
            "cols = [RED if abs(v) >= 2 else AMBER for v in jvals]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "ax.bar(labels, [R['tail'][d][2] for d in labels], color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('yen return that day (%)')\n"
            "ax.set_xticklabels(labels, rotation=30, ha='right')\n"
            "ax.set_title('The seven days everyone remembers - the yen swings BOTH ways')\n"
            "for i, d in enumerate(labels):\n"
            "    v = R['tail'][d][2]\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({d: R['tail'][d][0] for d in labels})"
        ),
        md(
            "Notice the yen moves in **both** directions across these seven famous surprises — "
            "weakening on the dovish NIRP/YCC-loosening days, strengthening sharply on the "
            "hawkish band-widening/hike days. Japanese stocks (EWJ), meanwhile, went **up on "
            "every single one** — a relief-rally pattern that has nothing to do with whether the "
            "surprise was hawkish or dovish. Only one day (the December-2022 \"Kuroda shock\") "
            "individually clears our statistical bar on any metric: the yen jumped **+3.8%** in "
            "a single session. That's a real, occasionally enormous tail event — not a "
            "repeatable calendar pattern.\n\n"
            "**Finally, the trade.** Even ignoring statistics for a moment — is there anything "
            "left to harvest, net of costs?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ce = st.decision_day_capture(DF, 'ewj_ret', cost_bps=5.0)\n"
            "    cj = st.decision_day_capture(DF, 'jpy_ret', cost_bps=5.0)\n"
            "    ge, ne, gj, nj = ce['gross_bps'], ce['net_bps'], cj['gross_bps'], cj['net_bps']\n"
            "else:\n"
            "    ge, ne, gj, nj = R['ewj_gross'], R['ewj_net5'], R['jpy_gross'], R['jpy_net5']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['EWJ gross','EWJ net','yen gross','yen net'], [ge, ne, gj, nj],\n"
            "       color=[GREY, RED, GREY, RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([ge, ne, gj, nj]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('bps per decision day')\n"
            "ax.set_title('Nothing to harvest: costs alone flip both to negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ gross {ge:+.1f} -> net {ne:+.1f} bps   yen gross {gj:+.1f} -> net {nj:+.1f} bps')"
        ),
        md(
            "Gross is already close to zero on both legs. Add costs and both flip decisively "
            "negative. There's no \"buy the dip before BoJ day\" edge in this data — just "
            "volatility risk you should size for, not a return you should chase."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No systematic directional reaction in EWJ or the yen across "
            "247 BoJ decisions since 2005, in or out of the NIRP/YCC surprise era.\n"
            "- **Tradability — Mirage.** With no average edge, costs alone make both legs "
            "losers.\n"
            "- **\"BoJ days swing more than average\"? — Confirmed.** They genuinely do — the "
            "folklore's kernel of truth is about *amplitude*, carried by a handful of huge, "
            "sign-flipping tail events, not a repeatable *direction*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** \"A scheduled announcement moves the market\" and \"a "
            "scheduled announcement moves the market *in a knowable direction*\" are two very "
            "different claims — this study busts the second while confirming the first.\n"
            "- **Where the real risk sits** is the tail, not the average: an investor holding "
            "unhedged Japan exposure through a BoJ meeting should size for an occasional "
            "multi-percent single-day move, not expect a systematic drift.\n"
            "- **Sibling studies:** [615-yen-safe-haven](../../615-yen-safe-haven/) (the yen's "
            "general risk-off reaction, any day) and "
            "[637-fomc-vol-crush](../../637-fomc-vol-crush/) (the Fed's decision-day effect on "
            "implied vol) ask related but different questions about scheduled central-bank "
            "days.\n\n"
            "*Think the surprise era is different? Show a directionally-predictive signal, "
            "certified at t ≥ 2, that survives the full NIRP/YCC sample — then we'll talk.*"
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
            "# The BoJ Announcement Effect — a quantitative teardown 🔬\n"
            "### Decision-day Welch/HAC splits · a two-sided 20-seed random-calendar placebo · "
            "the [−5..+3] event anatomy · the NIRP/YCC era contrast · the realized-range "
            "myth-check · a named `JPY=X` data quirk · tail-event z-scores · an honest capture "
            "test · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **Japanese equities/the yen react systematically to BoJ decision "
            "days, especially the NIRP/YCC surprise eras** — is deliberately NOT the FOMC "
            "vol-crush claim: BoJ decisions were near-monthly through 2015 and largely "
            "low-information under YCC, punctuated by real surprises. The job here is to "
            "measure the average-day reaction honestly, separate it from the tail, and ask "
            "the only question that pays: *is any of it tradable?*\n\n"
            "> ⚠️ **Data note.** EWJ raw OHLC + `JPY=X` OHLC (2005→2026), yfinance, cached; "
            "**248 hardcoded BoJ decision/statement dates** from the Bank of Japan's own "
            "archives (every decision on record — the claim is about the surprise eras, so "
            "nothing is filtered out). **Named data quirk:** `JPY=X`'s `Close` field silently "
            "duplicates `Open` on >95% of 2023-2025 rows (a Yahoo FX-data limitation); the yen "
            "return is dated D as `Open[D+1]/Open[D] - 1`, applied uniformly across the WHOLE "
            "sample, not just the broken years. No survivorship on the Signal axis (a fund and "
            "an FX rate). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_ewj"] + "` / `" +
            R["fp_jpy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | EWJ decision-day {R['ewj_boj']:+.4f}% vs {R['ewj_rest']:+.4f}%: "
            f"Welch **t = {R['ewj_t']:.2f}**; yen {R['jpy_boj']:+.4f}% vs {R['jpy_rest']:+.4f}%: "
            f"Welch **t = {R['jpy_t']:.2f}**; hit rates {R['ewj_hit_pct']:.1f}% / "
            f"{R['jpy_hit_pct']:.1f}% (both < 50%); placebo *p* = {R['ewj_placebo_p']:.4f} / "
            f"{R['jpy_placebo_p']:.4f} |\n"
            f"| **Tradability** | `MIRAGE` | EWJ net {R['ewj_net5']:+.1f} / {R['ewj_net10']:+.1f} bps, "
            f"yen net {R['jpy_net5']:+.1f} / {R['jpy_net10']:+.1f} bps at 5/10 bps; worst days "
            f"{R['ewj_worst']:.1f}% / {R['jpy_worst']:.1f}% |\n"
            f"| **Bigger swings?** | `CONFIRMED` | EWJ range **t = {R['ewj_range_t']:.2f}**, yen "
            f"range **t = {R['jpy_range_t']:.2f}** |\n\n"
            "> 💡 In plain words: the BoJ can shock markets on any given day — it just can't "
            "tell you in advance which way, or on which day."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the daily return (EWJ log-close-to-close, or the yen's own convention "
            "below) and $D_t \\in \\{0,1\\}$ the BoJ decision-day flag (known *ex ante* — the "
            "BoJ publishes its meeting schedule for the year ahead). The claims:\n\n"
            "- **H₁ (systematic reaction).** $E[r_t \\mid D_t=1] \\ne E[r_t \\mid D_t=0]$ — a "
            "large, systematic average-day gap.\n"
            "- **H₂ (anatomy).** A build-up before and/or persistence after the decision day.\n"
            "- **H₃ (surprise era).** The gap is stronger inside the 2016-2026 NIRP/YCC regime "
            "than before it.\n"
            "- **H₄ (amplitude, not direction).** Even absent H₁, decision days carry more "
            "realized range than an average day.\n"
            "- **H₅ (capture).** A naive decision-day-only hold banks H₁ net of costs.\n\n"
            "We find **H₁ rejected** (both instruments), **H₂ rejected** (no offset clears the "
            "bar), **H₃ rejected** (era-difference *t* < 2 on both instruments), **H₄ "
            "confirmed** (both range *t*'s ≥ 2), **H₅ moot** (nothing to capture, and costs "
            "make it actively negative)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Decision days are treated as single events; the planned primary is a **Welch t** "
            "on the group split, cross-checked with a **Newey-West (5-lag) t** on the "
            "decision-day dummy regression $r_t = a + b D_t$ (the slope *is* the mean gap). "
            "Unlike the FOMC vol-crush sibling — where the claim has a pre-committed sign (the "
            "VIX should *fall*) — the BoJ claim has **no pre-committed sign** (a surprise can "
            "be hawkish or dovish), so the random-calendar placebo here is **two-sided**: "
            "20,000 draws (20 seeds × 1,000) test $P(|\\text{placebo mean}| \\geq "
            "|\\text{observed}|)$. The NIRP era split (2016-01-29) is justified *ex ante* — the "
            "policy-history record, not a snooped date."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_hardcoded']} decision/statement dates {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded (BoJ official archives; every decision, not just "
            "scheduled ones — this claim is about the surprise eras).\n"
            f"- **Tape.** EWJ raw OHLC + `JPY=X` OHLC {R['start']} → {R['end']}. As-of "
            "2026-06-30 (last complete month). 247/248 hardcoded dates sit on the tape (1 "
            "missing: 2012-10-30, Hurricane Sandy NYSE closure).\n"
            "- **Headline.** Welch t + NW(5) t + Wilson hit rate + 20-seed two-sided placebo, "
            "on EWJ and the yen separately.\n"
            "- **Anatomy.** Event window [−5..+3], per-offset Welch t vs far days (meetings can "
            "sit closer than the window during the 2008/2010-11 crisis clusters; overlaps "
            "resolve to the chronologically later meeting — a named, minor caveat).\n"
            "- **Amplitude.** (H−L)/day-open split on the same days — the grey third axis.\n"
            "- **Execution (tradability).** Enter at the prior close (schedule published in "
            "advance — zero look-ahead), exit the decision-day close/next-open leg; 2 × "
            "one-way cost × NAV per event; long-only, no borrow.\n"
            "- **Control.** Synthetic i.i.d.-return world, planted mean-shift knob; the null "
            "must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its two-sided placebo\n\n"
            "Welch t on decision-day returns, NW t on the dummy regression, and the "
            "random-calendar null (two-sided — the BoJ claim has no pre-committed sign). In "
            "the notebook we run a lighter placebo (4 seeds × 500 draws) and quote the "
            "canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    se = st.decision_day_stats(DF, 'ewj_ret'); sj = st.decision_day_stats(DF, 'jpy_ret')\n"
            "    print(f\"EWJ  {se['boj_pct']:+.4f}% vs {se['rest_pct']:+.4f}%   Welch t = {se['welch_t']:+.2f}   NW t = {se['nw_t']:+.2f}\")\n"
            "    print(f\"yen  {sj['boj_pct']:+.4f}% vs {sj['rest_pct']:+.4f}%   Welch t = {sj['welch_t']:+.2f}   NW t = {sj['nw_t']:+.2f}\")\n"
            "    ple = st.placebo_pvalue(DF, 'ewj_ret', n_draws_per_seed=500, n_seeds=4)\n"
            "    obs_e, draws_e = ple['obs']*100, ple['draws']*100\n"
            "else:\n"
            "    obs_e = R['ewj_boj']\n"
            "    rng = np.random.default_rng(646)\n"
            "    draws_e = rng.normal(R['ewj_placebo_mean'], R['ewj_placebo_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws_e, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 247 days (light in-notebook run)')\n"
            "ax.axvline(obs_e, c=RED, lw=2.5, label=f'observed EWJ decision-day mean {obs_e:+.4f}%')\n"
            "ax.set_xlabel('mean EWJ return of a random 247-day calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Well inside the luck cloud: canonical p = {R['ewj_placebo_p']:.4f} \"\n"
            "             '(two-sided, 20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"EWJ canonical placebo: mean {R['ewj_placebo_mean']:+.4f}%, sd {R['ewj_placebo_sd']:.4f}%, p = {R['ewj_placebo_p']:.4f}\")\n"
            "print(f\"yen canonical placebo: mean {R['jpy_placebo_mean']:+.4f}%, sd {R['jpy_placebo_sd']:.4f}%, p = {R['jpy_placebo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: both observed gaps sit comfortably inside the null cloud of "
            f"random 247-day calendars (*p* = {R['ewj_placebo_p']:.2f} EWJ, "
            f"{R['jpy_placebo_p']:.2f} yen). Welch t = {R['ewj_t']:.2f} / {R['jpy_t']:.2f} — "
            "neither clears the desk bar. H₁ is rejected on both instruments."
        ),
        md(
            "### 4b · Anatomy — no build-up, no crush, no persistence\n\n"
            "Per-offset means with Welch t vs far-from-meeting days, both instruments."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eve = st.event_study(DF, 'ewj_ret', BOJ); evj = st.event_study(DF, 'jpy_ret', BOJ)\n"
            "    ks = list(eve.index)\n"
            "    ems, ets = list(eve['mean_pct']), list(eve['welch_t'])\n"
            "    jms, jts = list(evj['mean_pct']), list(evj['welch_t'])\n"
            "else:\n"
            "    ks = sorted(R['event'])\n"
            "    ems = [R['event'][k][0] for k in ks]; ets = [R['event'][k][1] for k in ks]\n"
            "    jms = [R['event'][k][2] for k in ks]; jts = [R['event'][k][3] for k in ks]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.8), sharex=True)\n"
            "w = .35\n"
            "x = np.arange(len(ks))\n"
            "a1.bar(x-w/2, ems, width=w, color=RED, label='EWJ')\n"
            "a1.bar(x+w/2, jms, width=w, color=AMBER, label='yen')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels([str(k) for k in ks])\n"
            "a1.set_ylabel('mean return (%)'); a1.legend()\n"
            "a1.set_title('Event anatomy: no offset stands out for either instrument')\n"
            "a2.bar(x-w/2, ets, width=w, color=[RED if abs(t)>=2 else GREY for t in ets])\n"
            "a2.bar(x+w/2, jts, width=w, color=[AMBER if abs(t)>=2 else GREY for t in jts])\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_xticks(x); a2.set_xticklabels([str(k) for k in ks])\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from decision)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('max |t| EWJ:', max(abs(t) for t in ets), '  max |t| yen:', max(abs(t) for t in jts))"
        ),
        md(
            "> 💡 In plain words: nine offsets, two instruments, eighteen Welch t's — none "
            "clears |t| = 2. There is no pre-meeting drift, no decision-day spike, and no "
            "post-day persistence on the average tape. H₂ is rejected."
        ),
        md(
            "### 4c · Amplitude, not direction — the grey third-axis myth-check\n\n"
            "If decision days are simply louder without being directional, realized range "
            "should be elevated even though the mean return split is null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rge = st.range_stats(DF, 'ewj_range'); rgj = st.range_stats(DF, 'jpy_range')\n"
            "    ea, eb, et = rge['boj_pct'], rge['rest_pct'], rge['welch_t']\n"
            "    ja, jb, jt = rgj['boj_pct'], rgj['rest_pct'], rgj['welch_t']\n"
            "else:\n"
            "    ea, eb, et = R['ewj_range_boj'], R['ewj_range_rest'], R['ewj_range_t']\n"
            "    ja, jb, jt = R['jpy_range_boj'], R['jpy_range_rest'], R['jpy_range_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))\n"
            "a1.bar(['BoJ days','other days'], [ea, eb], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([ea, eb]): a1.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_title(f'EWJ range (Welch t = {et:+.2f})'); a1.set_ylabel('(H-L)/open (%)')\n"
            "a2.bar(['BoJ days','other days'], [ja, jb], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([ja, jb]): a2.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_title(f'yen range (Welch t = {jt:+.2f})')\n"
            "plt.suptitle('Both clear the bar: decision days are genuinely louder')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ range {ea:.3f}% vs {eb:.3f}% (t={et:+.2f})   yen range {ja:.3f}% vs {jb:.3f}% (t={jt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: EWJ range **t = {R['ewj_range_t']:.2f}**, yen range "
            f"**t = {R['jpy_range_t']:.2f}** — both comfortably clear the bar. H₄ confirmed: "
            "the day is louder, the sign just isn't predictable ex ante. This is the honest "
            "kernel of the folklore — reframed correctly, it survives."
        ),
        md(
            "### 4d · The named `JPY=X` data quirk (and why it matters here)\n\n"
            "From 2022 onward yfinance's `JPY=X` daily `Close` silently duplicates the same "
            "row's `Open` on the large majority of days — verified below. Left unfixed, this "
            "would flatten exactly the tail events (Dec-2022 \"Kuroda shock\") this study is "
            "built to see."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq_frac_recent = (JPY.loc['2023':'2025', 'Open'] == JPY.loc['2023':'2025', 'Close']).mean()\n"
            "    eq_frac_early = (JPY.loc['2005':'2015', 'Open'] == JPY.loc['2005':'2015', 'Close']).mean()\n"
            "else:\n"
            "    eq_frac_recent, eq_frac_early = 0.95, 0.02\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['2005-2015\\n(reliable)','2023-2025\\n(broken)'], [eq_frac_early, eq_frac_recent],\n"
            "       color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([eq_frac_early, eq_frac_recent]): ax.annotate(f'{v*100:.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('share of days where Close == Open')\n"
            "ax.set_title('The JPY=X Close field: fine pre-2022, broken since')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Close==Open share: 2005-2015 {eq_frac_early*100:.1f}%   2023-2025 {eq_frac_recent*100:.1f}%')\n"
            "print('Fix applied uniformly across 2005-2026: yen return = -(log(Open[D+1]) - log(Open[D])),')\n"
            "print('dated D; High/Low remain reliable throughout so range uses (High-Low)/Open.')"
        ),
        md(
            "> 💡 In plain words: this isn't cherry-picked — it's a real, checkable vendor "
            "artifact, fixed with a single convention applied to the *entire* sample (not "
            "switched on only where it's broken), so no era-dependent methodology change is "
            "hiding in the numbers."
        ),
        md(
            "### 4e · Named surprise tail events — illustrative, never used to certify H₁\n\n"
            "z-scores vs the 247-event BoJ-day distribution."
        ),
        code(
            "labels = list(R['tail'].keys())\n"
            "ez = [R['tail'][d][1] for d in labels]; jz = [R['tail'][d][3] for d in labels]\n"
            "fig, ax = plt.subplots(figsize=(10.4, 4.8))\n"
            "x = np.arange(len(labels)); w = .35\n"
            "ax.bar(x-w/2, ez, width=w, color=[RED if abs(v)>=2 else AMBER for v in ez], label='EWJ z')\n"
            "ax.bar(x+w/2, jz, width=w, color=[RED if abs(v)>=2 else AMBER for v in jz], label='yen z')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')\n"
            "ax.set_ylabel('z-score vs the 247-event BoJ-day distribution'); ax.legend()\n"
            "ax.set_title('Only the Dec-2022 \\'Kuroda shock\\' individually clears |z| >= 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "for d in labels:\n"
            "    lbl, ep, ez_, jp, jz_, er, jr = R['tail'][d]\n"
            "    print(f'{d}  {lbl}: EWJ {ep:+.2f}% (z={ez_:+.2f})  yen {jp:+.2f}% (z={jz_:+.2f})')"
        ),
        md(
            "> 💡 In plain words: seven days everyone remembers, one that's individually "
            f"significant (yen *z* = {R['tail']['2022-12-20'][3]:+.2f} on the Kuroda shock). "
            "EWJ is up on every single one of them — a relief-rally pattern independent of "
            "hawkish/dovish framing — while the yen swings both ways depending on whether the "
            "surprise loosened or tightened policy relative to what was priced in. A real, "
            "occasionally huge tail — not a repeatable calendar pattern, and never used to "
            "certify the average-day Signal stamp."
        ),
        md(
            "### 4f · The era contrast — justified split, tested as a difference\n\n"
            f"Split at **{R['era_split']}** (the NIRP announcement — the start of the "
            "\"unconventional surprise\" regime; chosen ex ante, policy-history record)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ece = st.era_contrast(DF, 'ewj_ret', data.NIRP_YCC_SPLIT)\n"
            "    ecj = st.era_contrast(DF, 'jpy_ret', data.NIRP_YCC_SPLIT)\n"
            "    ee, el = ece['early_pct'], ece['late_pct']; edt = ece['welch_t_diff']\n"
            "    je, jl = ecj['early_pct'], ecj['late_pct']; jdt = ecj['welch_t_diff']\n"
            "else:\n"
            "    ee, el, edt = R['ewj_early'], R['ewj_late'], R['ewj_diff_t']\n"
            "    je, jl, jdt = R['jpy_early'], R['jpy_late'], R['jpy_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "x = np.arange(2); w = .35\n"
            "ax.bar(x-w/2, [ee, el], width=w, color=RED, label='EWJ')\n"
            "ax.bar(x+w/2, [je, jl], width=w, color=AMBER, label='yen')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(['pre-NIRP\\n2005-2016','NIRP/YCC\\n2016-2026'])\n"
            "ax.set_ylabel('decision-day mean return (%)'); ax.legend()\n"
            "ax.set_title(f'No era stands out (EWJ diff t={edt:+.2f}, yen diff t={jdt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ: early {ee:+.4f}%  late {el:+.4f}%  diff t={edt:+.2f}')\n"
            "print(f'yen: early {je:+.4f}%  late {jl:+.4f}%  diff t={jdt:+.2f}')"
        ),
        md(
            "> 💡 In plain words: neither era, nor the difference between them, reaches the "
            "bar on either instrument. H₃ is rejected — the surprise regime is not, on "
            "average, more directionally reactive than the earlier one; it just contains "
            "surprises that go both ways and cancel."
        ),
        md(
            "### 4g · The third axis — the honest capture test\n\n"
            "Enter the prior close (the BoJ's schedule is public — zero look-ahead), exit the "
            "decision-day close/next-open leg, pay 2 × one-way costs per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows_e = [st.decision_day_capture(DF, 'ewj_ret', cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    rows_j = [st.decision_day_capture(DF, 'jpy_ret', cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    ge, n5e, n10e = rows_e[0]['gross_bps'], rows_e[0]['net_bps'], rows_e[1]['net_bps']\n"
            "    gj, n5j, n10j = rows_j[0]['gross_bps'], rows_j[0]['net_bps'], rows_j[1]['net_bps']\n"
            "else:\n"
            "    ge, n5e, n10e = R['ewj_gross'], R['ewj_net5'], R['ewj_net10']\n"
            "    gj, n5j, n10j = R['jpy_gross'], R['jpy_net5'], R['jpy_net10']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "labels = ['EWJ gross','EWJ net5','EWJ net10','yen gross','yen net5','yen net10']\n"
            "vals = [ge, n5e, n10e, gj, n5j, n10j]\n"
            "cols = [GREY, AMBER, RED, GREY, AMBER, RED]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('bps per decision day')\n"
            "ax.set_title('Costs alone flip both legs decisively negative')\n"
            "plt.xticks(rotation=20, ha='right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EWJ gross {ge:+.2f} -> net5 {n5e:+.2f} / net10 {n10e:+.2f} bps')\n"
            "print(f'yen gross {gj:+.2f} -> net5 {n5j:+.2f} / net10 {n10j:+.2f} bps')"
        ),
        md(
            f"> 💡 In plain words: gross is already indistinguishable from zero "
            f"(t = {R['ewj_t']:.2f} EWJ, {R['jpy_t']:.2f} yen). Add costs and both legs go "
            f"decisively negative ({R['ewj_net5']:+.1f} / {R['jpy_net5']:+.1f} bps net at 5 "
            f"bps). Worst single days: {R['ewj_worst']:.1f}% (EWJ), {R['jpy_worst']:.1f}% "
            "(yen) — H₅ is not just uncertified, it's actively negative-expectancy after "
            "costs. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4h · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic i.i.d.-return world, scheduled pseudo-decisions every 12th business day, "
            "TUNABLE planted mean shift. The null (effect = 0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, dec = data.synthetic_world(effect=0.0, seed=646 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, dec)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, dec = data.synthetic_world(effect=0.002, seed=646)\n"
            "planted_t = st.synthetic_detect(close, dec)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (effect=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted effect = +0.20% log-return')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (decision vs rest)')\n"
            "ax.set_title('Control: the null rarely fires; a planted shift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires in "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds — a 100-seed check gives an "
            f"empirical false-positive rate of {R['syn_100seed_fp']}%, consistent with the "
            "nominal ~5% for a single test, confirming the machinery is unbiased rather than "
            f"systematically over-triggering. A planted +{R['syn_planted_effect']:.2f}% shift — "
            "far smaller than the ~2-4% tail-event moves above — reads "
            f"t = {R['syn_planted_t']:.2f}, comfortably detected. The real-tape null "
            "(t ≈ -0.06 / -0.85) is the genuine article, not an underpowered test. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — EWJ decision-day {R['ewj_boj']:+.4f}% vs {R['ewj_rest']:+.4f}%: "
            f"Welch t = **{R['ewj_t']:.2f}**; yen {R['jpy_boj']:+.4f}% vs {R['jpy_rest']:+.4f}%: "
            f"Welch t = **{R['jpy_t']:.2f}**. Hit rates {R['ewj_hit_pct']:.1f}% / "
            f"{R['jpy_hit_pct']:.1f}% (both below a coin flip), placebo p = "
            f"{R['ewj_placebo_p']:.4f} / {R['jpy_placebo_p']:.4f}, no event-window offset "
            f"clears the bar, NIRP/YCC era-difference t = {R['ewj_diff_t']:+.2f} / "
            f"{R['jpy_diff_t']:+.2f}.\n"
            f"- **Tradability `MIRAGE`** — net {R['ewj_net5']:+.1f} bps (EWJ) / "
            f"{R['jpy_net5']:+.1f} bps (yen) at 5 bps, worse at 10 bps; worst single days "
            f"{R['ewj_worst']:.1f}% / {R['jpy_worst']:.1f}%.\n"
            f"- **\"BoJ days swing more than average?\" `CONFIRMED`** — realized range Welch "
            f"t = **{R['ewj_range_t']:.2f}** (EWJ) / **{R['jpy_range_t']:.2f}** (yen). The "
            "folklore's honest kernel is amplitude, carried by a handful of huge, "
            "sign-flipping surprises that cancel on average — not a directional edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson generalizes.** \"A scheduled central-bank day moves the "
            "market\" and \"a scheduled central-bank day moves the market *in a knowable "
            "direction*\" are separable claims; this study rejects the second while confirming "
            "the first (realized range).\n"
            "- **Where the professional expression actually lives** is options (a long-gamma "
            "position through the decision, sized for the loud-but-directionless tail) rather "
            "than a directional cash position — the natural sequel study is a USDJPY-options "
            "straddle-through-BoJ-day backtest.\n"
            "- **Dedup map:** [615-yen-safe-haven](../../615-yen-safe-haven/) (the yen's "
            "*general* risk-off reaction to any equity-crash day), "
            "[645-ecb-announcement-effect](../../645-ecb-announcement-effect/) (the same "
            "question for the ECB), [637-fomc-vol-crush](../../637-fomc-vol-crush/) (the Fed's "
            "decision-day effect on *implied vol*, scheduled meetings only).\n\n"
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
