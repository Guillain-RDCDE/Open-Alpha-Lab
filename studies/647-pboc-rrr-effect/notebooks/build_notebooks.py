"""Generate the two narrative notebooks for Study 647 (PBoC RRR Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FXI/MCHI tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance FXI 2008-01-02 ->
# 2026-06-30, MCHI 2011-03-31 -> 2026-06-30; 48 hardcoded broad-based PBoC RRR announcements,
# 31 cuts / 17 hikes).
R = dict(
    start="2008-01-01", end="2026-06-30", n_events=48, n_cut=31, n_hike=17,
    cal_lo="2008-01-16", cal_hi="2025-05-07",
    cut_pct=0.7145, cut_rest_pct=-0.0079, cut_gap=0.7225, cut_t=0.99, cut_nw=1.00,
    cut_hit=20, cut_n=31, cut_hit_pct=64.5, cut_wilson=(46.9, 78.9),
    hike_pct=0.6397, hike_rest_pct=-0.0055, hike_gap=0.6452, hike_t=1.36, hike_nw=1.40,
    hike_hit=13, hike_n=17, hike_hit_pct=76.5, hike_wilson=(52.7, 90.4),
    cvh_t=0.09,
    placebo_cut_p=0.0296, placebo_cut_mean=-0.0057, placebo_cut_sd=0.3682,
    placebo_hike_p=0.9132, placebo_hike_mean=-0.0090, placebo_hike_sd=0.4953,
    event={-5: (-0.1840, -0.36), -4: (-0.7631, -1.30), -3: (0.3660, 0.64), -2: (0.0092, 0.00),
           -1: (-0.6345, -1.08), 0: (0.7145, 0.96), 1: (0.4567, 0.77), 2: (-0.5126, -0.76),
           3: (1.0035, 1.58), 4: (-0.5204, -1.06), 5: (-0.4434, -0.88), 6: (-0.3196, -0.85),
           7: (-0.6130, -0.84), 8: (-0.6753, -1.32), 9: (1.3584, 1.87), 10: (-0.7219, -1.97)},
    runup_pct=-1.206, runup_t=-0.97, runup_n=31,
    post_pct=-0.943, post_t=-0.71, post_n=31,
    range_event=2.1772, range_rest=1.4900, range_t=2.45,
    mchi_cut_n=27, mchi_cut_pct=0.9768, mchi_cut_rest=0.0008, mchi_cut_t=1.98,
    mchi_hike_n=3, mchi_hike_pct=0.9219, mchi_hike_rest=0.0069, mchi_hike_t=2.59,
    mchi_cvh_t=0.09,
    era_early_pct=-0.3898, era_early_n=7, era_early_t=-0.13,
    era_late_pct=1.0366, era_late_n=24, era_late_t=2.10, era_diff_t=0.49,
    timer={1: (71.5, 61.5, 1.7, 0.95, 64.5, -16.1),
           3: (65.9, 55.9, 7.3, 0.57, 61.3, -15.9),
           5: (111.8, 101.8, 14.3, 0.87, 54.8, -15.9),
           10: (30.1, 20.1, 30.4, -0.00, 58.1, -29.8)},
    syn_null_mean=-0.15, syn_null_sd=1.22, syn_null_fire=2, syn_planted_t=4.13,
    fp_fxi="d41690523582", fp_mchi="460c227db0e1",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Buy_the_rumor%2C_sell_the_news%3F: Busted]"
    "(https://img.shields.io/badge/Buy_the_rumor%2C_sell_the_news%3F-Busted-8b949e?style=flat-square)\n\n"
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

from pboc_rrr_effect import data, strategy as st

RRR = data.rrr_frame()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    FXI, MCHI = data.load_real()
    MATCHED_FXI = st.match_trading_days(FXI.index, RRR)
    MATCHED_MCHI = st.match_trading_days(MCHI.index, RRR)
    DF = st.day_frame(FXI, MATCHED_FXI)
    MDF = st.day_frame(MCHI, MATCHED_MCHI)
    CUT_DAYS = pd.DatetimeIndex(MATCHED_FXI.loc[MATCHED_FXI["direction"] == "cut", "trading_date"])
else:
    FXI = MCHI = DF = MDF = MATCHED_FXI = CUT_DAYS = None
print("real cache present:", HAVE_REAL, "| RRR announcements:", len(RRR),
      "| tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Beijing's easing button actually push stocks up? 🇨🇳\n"
            "### The PBoC \"stimulus rally\" — a headline you've read fifty times, tested "
            "against the tape\n\n"
            + BADGES +
            "Every few months the People's Bank of China cuts the **Reserve Requirement "
            "Ratio (RRR)** — the share of deposits banks must lock away instead of lending out "
            "— and the financial press runs some version of the same headline: *\"China stocks "
            "rally as PBoC cuts reserve ratio.\"* It sounds like plain economics: freeing up "
            "lending capacity is stimulus, stimulus is bullish, so equities should pop.\n\n"
            "That's the claim we test — and, quietly, its unstated mirror: if a cut is "
            "bullish, a **hike** should be bearish. Steelmanning a folk claim means testing "
            "*both* halves, not just the flattering one.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 48 hardcoded, BROAD-BASED PBoC RRR announcements "
            "(2008 → 2025: 31 cuts, 17 hikes) — targeted relief for rural/SME lenders is "
            "excluded, because that's not the \"big lever\" the folklore is about. Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do Chinese equities pop on a cut day? | **On paper, yes** — FXI averages "
            f"**{R['cut_pct']:+.2f}%** on a cut day vs about zero normally. But that number "
            "alone can't tell you why. |\n"
            f"| Do they fall on a HIKE day, as the same logic would predict? | **No — they "
            f"ALSO pop**, by almost exactly the same amount (**{R['hike_pct']:+.2f}%**). "
            "That's the tell. |\n"
            f"| So does direction matter at all? | **No.** Cuts vs hikes, head to head: "
            f"statistically indistinguishable (Welch *t* = {R['cvh_t']:.2f} — the bar for "
            "\"real\" is 2). Whatever makes RRR announcement days run hot, it isn't the "
            "direction of the move. |\n"
            f"| Can you at least trade the timing — buy the rumor, sell the news? | **No.** "
            f"The days *before* a cut actually drift slightly DOWN ({R['runup_pct']:+.2f}%, "
            "not up), and a 10-day hold after the cut earns exactly what a random 10-day "
            "window earns. |\n\n"
            "> A headline you've read fifty times, and the tape says: not because of the "
            "direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The PBoC cutting the RRR frees up bank lending capacity across the whole "
            "system — that's stimulus, stimulus is bullish, and Chinese equities should rally "
            "on the news.\"*\n\n"
            "It's intuitive, it's repeated after nearly every cut by Reuters, Bloomberg and "
            "CNBC, and it has a real economic mechanism behind it (more lendable reserves *can* "
            "support credit growth). What it rarely gets is the honest mirror-image test: if "
            "cuts are bullish stimulus, hikes should be bearish tightening. We test both."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is about as simple an edge as macro trading gets: watch one "
            "official's calendar, buy the ETF on the announcement, collect the pop. China "
            "watchers already try to front-run State Council hints of an upcoming cut. If it "
            "doesn't hold up, that's worth knowing too — it means the \"stimulus rally\" "
            "headline you keep reading is describing a coincidence, not a mechanism you can "
            "set a calendar alert for."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_events']}** broad-based PBoC RRR announcements "
            f"{R['cal_lo']} → {R['cal_hi']} ({R['n_cut']} cuts, {R['n_hike']} hikes), "
            "hardcoded from the PBoC's own announcement archive.\n"
            "- **The comparison.** FXI's (China large-cap ETF) return on each announcement's "
            "trading day vs every other day — split separately for cuts and for hikes.\n"
            "- **The decisive test.** Cuts vs hikes, head to head. If direction matters, this "
            "gap should be large; if the folklore is really just \"RRR days are eventful "
            "days,\" this gap should be near zero.\n"
            "- **The trade check.** Buy FXI at the close before a cut, hold 1/3/5/10 trading "
            "days, pay costs — does any horizon actually pay?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline split** — cut days and hike days, each against an ordinary "
            "trading day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sc = st.decision_day_stats(DF, 'ret', 'cut')\n"
            "    sh = st.decision_day_stats(DF, 'ret', 'hike')\n"
            "    cp, hp, rp = sc['event_pct'], sh['event_pct'], sc['rest_pct']\n"
            "else:\n"
            "    cp, hp, rp = R['cut_pct'], R['hike_pct'], R['cut_rest_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['cut days\\n(n=31)', 'hike days\\n(n=17)', 'an ordinary day'],\n"
            "       [cp, hp, rp], color=[GREEN, RED, GREY], width=.6)\n"
            "for i, v in enumerate([cp, hp, rp]):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average FXI return')\n"
            "ax.set_title('Both cut days AND hike days run hot — that is the tell')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cut {cp:+.3f}%   hike {hp:+.3f}%   ordinary day {rp:+.3f}%')"
        ),
        md(
            f"Cut days average **{R['cut_pct']:+.2f}%**. That alone looks like the folklore, "
            f"until you check the mirror: hike days average **{R['hike_pct']:+.2f}%** — "
            "nearly identical, and in the SAME direction a tightening move is supposed to "
            "oppose. Both kinds of RRR day run far hotter than an ordinary day "
            f"({R['cut_rest_pct']:+.2f}%), but not because of what the RRR did.\n\n"
            "**The decisive number** makes this precise: cuts vs hikes, directly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cvh = st.cuts_vs_hikes(DF, 'ret')\n"
            "    t = cvh['welch_t']\n"
            "else:\n"
            "    t = R['cvh_t']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.barh(['cuts vs hikes\\n(Welch t)'], [t], color=GREY, height=.5)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.axvline(2, ls='--', c=RED, lw=1.2); ax.axvline(-2, ls='--', c=RED, lw=1.2)\n"
            "ax.annotate(f't = {t:+.2f}', (t, 0), ha='center', va='bottom', fontsize=13)\n"
            "ax.set_xlim(-4, 4)\n"
            "ax.set_title('If direction mattered, this bar would sit past the dashed line')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cuts vs hikes, Welch t = {t:+.2f}  (bar for \"real\" is 2)')"
        ),
        md(
            f"**Welch *t* = {R['cvh_t']:.2f}.** The bar for calling something real on this desk "
            "is *t* = 2 — and this is about as close to a coin flip as a statistic gets. Cut "
            "days and hike days are not distinguishable from each other. Whatever makes RRR "
            "announcement days run hot — and the market IS louder on those days, not just "
            "higher — it isn't the direction of the move.\n\n"
            "**So what does drift around the announcement look like?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, 'ret', CUT_DAYS)\n"
            "    ks, ms = list(ev.index), list(ev['mean_pct'])\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.6))\n"
            "cols = [RED if k == 0 else GREY for k in ks]\n"
            "ax.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "for i, v in enumerate(ms):\n"
            "    ax.annotate(f'{v:+.1f}', (i, v), ha='center',\n"
            "                va='bottom' if v >= 0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days relative to the cut (0 = cut day)')\n"
            "ax.set_ylabel('average FXI return (%)')\n"
            "ax.set_title('No shape at all: not a run-up, not a crush, not a fade')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('offsets:', {k: round(m, 2) for k, m in zip(ks, ms)})"
        ),
        md(
            "There's no recognizable pattern here — no ramp into the cut, no clean pop that "
            f"fades afterward. The five days BEFORE a cut actually drift down a little "
            f"(**{R['runup_pct']:+.2f}%** cumulative, not up — so much for \"buying the "
            "rumor\"), and the ten days after are flat-to-slightly-negative on average. "
            "**Finally, the trade:** does any holding period actually pay, after costs?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    caps = [st.capture_horizon(FXI['AdjClose'], MATCHED_FXI, 'cut', h, 5.0)\n"
            "            for h in (1, 3, 5, 10)]\n"
            "    nets = [c['net_bps'] for c in caps]\n"
            "else:\n"
            "    nets = [R['timer'][h][1] for h in (1, 3, 5, 10)]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['1 day', '3 days', '5 days', '10 days'], nets, color=AMBER, width=.55)\n"
            "for i, v in enumerate(nets):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net return per cut event (bps, 5 bps one-way costs)')\n"
            "ax.set_title('Positive on paper at every horizon — none of it is certifiable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net bps by hold:', {h: round(n, 1) for h, n in zip((1, 3, 5, 10), nets)})"
        ),
        md(
            "Every bar is positive — and every single one sits at a Welch *t* well under 2 "
            "(the best is 0.95). The 10-day hold, the horizon closest to \"did the pop hold up "
            "after the news cycle moved on,\" earns **exactly what a random 10-day window on "
            "the same tape earns** — zero excess. Add a worst-case single event of **-29.8%** "
            "over that horizon, and there's nothing here to bank."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Cut days and hike days are statistically indistinguishable "
            f"(Welch *t* = {R['cvh_t']:.2f}, confirmed on a second, independently constructed "
            "China ETF). RRR announcement days ARE louder than average — the market moves more "
            "on those days — but not in a direction the folklore predicts.\n"
            "- **Tradability — Mirage.** No holding period of a buy-the-cut timer clears the "
            "bar; the longest horizon shows zero excess over random chance.\n"
            "- **\"Buy the rumor, sell the news\"? — Busted.** There's no rumor phase (prices "
            "drift down before a cut, not up) and no clean news-selloff either. The story just "
            "isn't in the data."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **What might actually be going on:** RRR cuts don't arrive at random — they "
            "cluster in stretches when Beijing is already leaning supportive (property-market "
            "stress, growth wobbles, coordinated \"national team\" buying). The loud-but-"
            "directionless pattern here is consistent with RRR days being *correlated with* "
            "supportive periods, not *causing* them.\n"
            "- **The professional angle:** if there's a real edge in Chinese monetary policy "
            "for equities, it more plausibly lives in policy-RATE surprises (the 1yr/5yr LPR) "
            "or in reading State-Council pre-announcements days ahead — not in the RRR "
            "announcement itself, which this study finds carries no average directional "
            "content.\n"
            "- **Sibling studies:** [620-a-h-premium](../../620-a-h-premium/) (the structural "
            "A/H price gap, no calendar) and [313-geopolitical-shock](../../313-geopolitical-"
            "shock/) (the same event-study skeleton, a completely different market) — neither "
            "overlaps this study's claim.\n\n"
            "*Think the edge is real but mistimed? Show a net, certifiable directional gap "
            "between cuts and hikes — after costs, on out-of-sample announcements — then we'll "
            "talk.*"
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
            "# The PBoC RRR Effect — a quantitative teardown 🔬\n"
            "### Cut/hike Welch-HAC splits · a one-sided 20-seed placebo · the [-5..+10] "
            "event anatomy · an MCHI cross-check · the era contrast · a buy-the-cut timer · a "
            "20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **Chinese equities pop on a PBoC RRR cut** — is repeated financial-"
            "press folklore with a plausible mechanism (freed lending capacity) but, unlike "
            "the FOMC-vol-crush or BoJ siblings, no dedicated peer-reviewed event-study anchor. "
            "The job here is to measure it honestly, with its own unstated mirror (hikes "
            "should be bearish) tested just as hard.\n\n"
            "> ⚠️ **Data note.** FXI daily raw OHLC + adjusted close (2008 → 2026) + MCHI (2011 "
            "→ 2026, cross-check), yfinance, cached; **48 hardcoded broad-based PBoC RRR "
            "announcements** (31 cuts, 17 hikes) from the PBoC's official archive, cross-"
            "checked against financial-press coverage — targeted/structural relief excluded by "
            "construction. No survivorship on the Signal axis (fund tapes); **FXI's swap-heavy "
            "QFII-quota structure is named on Tradability**. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_fxi"] +
            "` / `" + R["fp_mchi"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | cut-day *t* = {R['cut_t']:.2f}, hike-day *t* = "
            f"{R['hike_t']:.2f}; **cuts vs hikes, Welch *t* = {R['cvh_t']:.2f}** "
            f"(MCHI confirms: {R['mchi_cvh_t']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | buy-the-cut timer never clears *t* = 2 (best "
            f"{max(v[3] for v in R['timer'].values()):.2f}); 10-day hold *t* = "
            f"{R['timer'][10][3]:.2f} vs a random window |\n"
            f"| **Rumor/news?** | `BUSTED` | pre-cut run-up *t* = {R['runup_t']:.2f} "
            f"(negative); post-cut window *t* = {R['post_t']:.2f} |\n\n"
            "> 💡 In plain words: RRR announcement days are louder than average, but the "
            "loudness has no sign attached to it — cuts and hikes look the same."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be FXI's daily log return and $C_t, H_t \\in \\{0,1\\}$ the scheduled "
            "cut-day / hike-day flags from the hardcoded, broad-based RRR calendar. The "
            "claims:\n\n"
            "- **H₁ (cut pop).** $E[r_t \\mid C_t=1] \\gg E[r_t \\mid C_t=0]$.\n"
            "- **H₂ (hike drop, the unstated mirror).** $E[r_t \\mid H_t=1] \\ll E[r_t \\mid "
            "H_t=0]$.\n"
            "- **H₃ (direction matters).** $E[r_t \\mid C_t=1] \\ne E[r_t \\mid H_t=1]$, tested "
            "*directly* — the cleanest single falsification available.\n"
            "- **H₄ (rumor/news timing).** A pre-cut run-up (rumor) and a post-cut fade "
            "(news) inside the [-5..+10] window.\n"
            "- **H₅ (capture).** A prior-close-to-close FXI hold over the cut day (or several "
            "days) banks it net of costs.\n\n"
            "We find **H₁ unsupported at the primary bar** (*t* = 0.99), **H₂ flatly rejected** "
            "(hike days are POSITIVE, *t* = 1.36, same sign as cuts), **H₃ decisively "
            "rejected** (*t* = 0.09, replicated on MCHI), **H₄ rejected** (run-up is negative, "
            "no clean fade), **H₅ not certified** at any horizon."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "RRR events are **single, non-overlapping-in-theory events** (median spacing is "
            "months), so the planned primary is a **Welch t** on each group split — but the "
            "single most informative statistic on this desk's own terms is the **direct "
            "cuts-vs-hikes Welch t**, which needs no assumption about what an \"ordinary day\" "
            "looks like: if the claim's own mechanism (direction) is doing the work, that gap "
            "should be large. A **Newey-West (5-lag) t** cross-checks the day-level splits; "
            "the placebo is **one-sided** with a pre-committed tail per direction (the claim "
            "has a sign, unlike a symmetric-surprise story); MCHI replicates every number on a "
            "differently constructed index to rule out FXI-specific microstructure; and the "
            "era split (2015-01-01, the RRR's regime pivot from hiking to a near-permanent "
            "cutting stance) is tested as a **difference**, not eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} broad-based announcements {R['cal_lo']} → "
            f"{R['cal_hi']} ({R['n_cut']} cuts, {R['n_hike']} hikes), hardcoded, all mapping "
            "onto the FXI tape.\n"
            f"- **Tape.** FXI raw OHLC + adjusted close {R['start']} → {R['end']}; MCHI from "
            "its 2011-03-29 inception (cross-check). As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** Welch t + NW(5) t + Wilson hit rate + one-sided 20-seed placebo, "
            "cut and hike each vs the rest; then the direct cuts-vs-hikes Welch t.\n"
            "- **Anatomy.** Event window [-5..+10] (cut days), per-offset Welch t vs far days; "
            "cumulative pre/post-window one-sample t's.\n"
            "- **Cross-check.** MCHI replicates the whole headline split, modern era only.\n"
            "- **Execution (third axis).** Enter FXI at the prior close relative to the "
            "announcement's mapped trading day (zero look-ahead — see docs/results.md), hold "
            "1/3/5/10 days, exit; 2 x one-way cost x NAV per event; long-only, no borrow.\n"
            "- **Control.** Synthetic i.i.d.-return world, planted cut-day mean-shift knob; "
            "the null must not (systematically) fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split, and the test that actually decides it\n\n"
            "Welch t + NW(5) t + Wilson hit rate for cuts and hikes, each vs an ordinary day — "
            "then the direct cuts-vs-hikes Welch t, the single most informative number here."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sc = st.decision_day_stats(DF, 'ret', 'cut')\n"
            "    sh = st.decision_day_stats(DF, 'ret', 'hike')\n"
            "    cvh = st.cuts_vs_hikes(DF, 'ret')\n"
            "else:\n"
            "    sc = dict(event_pct=R['cut_pct'], rest_pct=R['cut_rest_pct'], welch_t=R['cut_t'],\n"
            "              nw_t=R['cut_nw'], hit_up=R['cut_hit'], n_event=R['cut_n'],\n"
            "              hit_rate=R['cut_hit_pct']/100, hit_lo=R['cut_wilson'][0]/100,\n"
            "              hit_hi=R['cut_wilson'][1]/100)\n"
            "    sh = dict(event_pct=R['hike_pct'], rest_pct=R['hike_rest_pct'], welch_t=R['hike_t'],\n"
            "              nw_t=R['hike_nw'], hit_up=R['hike_hit'], n_event=R['hike_n'],\n"
            "              hit_rate=R['hike_hit_pct']/100, hit_lo=R['hike_wilson'][0]/100,\n"
            "              hit_hi=R['hike_wilson'][1]/100)\n"
            "    cvh = dict(cut_pct=R['cut_pct'], hike_pct=R['hike_pct'], welch_t=R['cvh_t'])\n"
            "for label, s in (('cut', sc), ('hike', sh)):\n"
            "    print(f\"{label:>4}: {s['event_pct']:+.4f}% vs {s['rest_pct']:+.4f}%   \"\n"
            "          f\"Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}  \"\n"
            "          f\"hit {s['hit_up']}/{s['n_event']}={s['hit_rate']*100:.1f}% \"\n"
            "          f\"[{s['hit_lo']*100:.1f}%,{s['hit_hi']*100:.1f}%]\")\n"
            "print(f\"cuts vs hikes, direct: cut {cvh['cut_pct']:+.4f}%  hike \"\n"
            "      f\"{cvh['hike_pct']:+.4f}%   Welch t = {cvh['welch_t']:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['cut days', 'hike days', 'ordinary day'],\n"
            "       [sc['event_pct'], sh['event_pct'], sc['rest_pct']],\n"
            "       color=[GREEN, RED, GREY], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean FXI return (%)')\n"
            "ax.set_title(f\"cuts vs hikes Welch t = {cvh['welch_t']:+.2f}  (bar = 2)\")\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: neither the cut split (*t* = {R['cut_t']:.2f}) nor the hike "
            f"split (*t* = {R['hike_t']:.2f}) individually clears the bar, but the number that "
            f"matters most is **cuts vs hikes = {R['cvh_t']:.2f}** — direction explains almost "
            "nothing. Both kinds of announcement day are hotter than an ordinary day; neither "
            "is hotter than the other."
        ),
        md(
            "### 4b · The one-sided placebo\n\n"
            "Pre-committed sign per direction: cuts should beat a random calendar from ABOVE "
            "(right tail), hikes from BELOW (left tail). Null pool excludes ALL RRR days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl_c = st.placebo_pvalue(DF, 'ret', 'cut', tail='right',\n"
            "                             n_draws_per_seed=250, n_seeds=4)\n"
            "    pl_h = st.placebo_pvalue(DF, 'ret', 'hike', tail='left',\n"
            "                             n_draws_per_seed=250, n_seeds=4)\n"
            "    obs_c, obs_h = pl_c['obs'], pl_h['obs']\n"
            "else:\n"
            "    rng = np.random.default_rng(647)\n"
            "    obs_c, obs_h = R['cut_pct']/100, R['hike_pct']/100\n"
            "    pl_c = dict(placebo_mean=R['placebo_cut_mean']/100, placebo_sd=R['placebo_cut_sd']/100)\n"
            "    pl_h = dict(placebo_mean=R['placebo_hike_mean']/100, placebo_sd=R['placebo_hike_sd']/100)\n"
            "print(f\"cut  (right tail): canonical p = {R['placebo_cut_p']:.4f} \"\n"
            "      f\"(obs {R['cut_pct']:+.4f}% vs placebo mean {R['placebo_cut_mean']:+.4f}%, \"\n"
            "      f\"sd {R['placebo_cut_sd']:.4f}%)\")\n"
            "print(f\"hike (left  tail): canonical p = {R['placebo_hike_p']:.4f} \"\n"
            "      f\"(obs {R['hike_pct']:+.4f}% vs placebo mean {R['placebo_hike_mean']:+.4f}%, \"\n"
            "      f\"sd {R['placebo_hike_sd']:.4f}%)\")"
        ),
        md(
            f"> 💡 In plain words: the cut placebo's canonical *p* = {R['placebo_cut_p']:.4f} "
            "looks more impressive than its Welch *t* — a small, slightly right-skewed sample "
            "can do that. It doesn't survive the cuts-vs-hikes wash test above (4a), which is "
            f"why the certified verdict is `NONE`, not `WEAK`. The hike placebo's "
            f"*p* = {R['placebo_hike_p']:.4f} flatly rejects the tightening half of the folklore "
            "— hike days are nowhere near the bottom of a random calendar."
        ),
        md(
            "### 4c · Event anatomy — is there a shape at all?\n\n"
            "Per-offset means with Welch t vs far-from-event days, cut days only "
            "[-5..+10]."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, 'ret', CUT_DAYS)\n"
            "    ks = list(ev.index); ms = list(ev['mean_pct']); ts = list(ev['welch_t'])\n"
            "    ru = st.runup_stats(DF, 'ret', CUT_DAYS); pe = st.postevent_stats(DF, 'ret', CUT_DAYS)\n"
            "    ru_m, ru_t, pe_m, pe_t = ru['mean_runup_pct'], ru['t'], pe['mean_postrun_pct'], pe['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][1] for k in ks]\n"
            "    ru_m, ru_t, pe_m, pe_t = R['runup_pct'], R['runup_t'], R['post_pct'], R['post_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k == 0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean FXI return (%)')\n"
            "a1.set_title('No coherent shape: not a build-up, not a crush, not a fade')\n"
            "a2.bar([str(k) for k in ks], ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from the cut)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-cut run-up [-5..-1]: {ru_m:+.3f}%/event (t={ru_t:+.2f})')\n"
            "print(f'post-cut window [+1..+10]: {pe_m:+.3f}%/event (t={pe_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the pre-cut run-up is **{R['runup_pct']:+.2f}%** at "
            f"*t* = {R['runup_t']:+.2f} — *negative*, the opposite of \"buy the rumor.\" The "
            f"post-cut window is **{R['post_pct']:+.2f}%** at *t* = {R['post_t']:+.2f} — mildly "
            "negative, not a clean decay from a certified pop (there's no certified pop to "
            "decay from). The one lone offset that individually crosses \\|t\\| ≥ 2 (day +10, "
            f"*t* = {R['event'][10][1]:+.2f}) runs the WRONG sign for a \"news gets sold\" "
            "story to claim credit for, and at 16 tested offsets one crossing 2 by chance is "
            "expected, not notable."
        ),
        md(
            "### 4d · MCHI cross-check — same result, a differently built index\n\n"
            "MCHI (broad China ETF, 2011 inception) replicates the whole headline split, "
            "modern-era events only."
        ),
        code(
            "if HAVE_REAL:\n"
            "    smc = st.decision_day_stats(MDF, 'ret', 'cut')\n"
            "    smh = st.decision_day_stats(MDF, 'ret', 'hike')\n"
            "    cvhm = st.cuts_vs_hikes(MDF, 'ret')\n"
            "    cp_m, hp_m, tm = smc['event_pct'], smh['event_pct'], cvhm['welch_t']\n"
            "else:\n"
            "    cp_m, hp_m, tm = R['mchi_cut_pct'], R['mchi_hike_pct'], R['mchi_cvh_t']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['MCHI cut\\n(n=27)', 'MCHI hike\\n(n=3)'], [cp_m, hp_m], color=[GREEN, RED], width=.5)\n"
            "for i, v in enumerate([cp_m, hp_m]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title(f'MCHI cuts-vs-hikes Welch t = {tm:+.2f} — the same non-result')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'MCHI cut {cp_m:+.3f}%  hike {hp_m:+.3f}%   cuts-vs-hikes t = {tm:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: MCHI's cuts-vs-hikes Welch *t* = {R['mchi_cvh_t']:.2f} — "
            "the identical non-result on an independently constructed index. The MCHI hike "
            f"split alone crosses *t* = {R['mchi_hike_t']:.2f}, but on only "
            f"**n = {R['mchi_hike_n']}** events (all from spring 2011, before MCHI had much "
            "history) — nowhere near enough to certify anything on its own, and it doesn't "
            "change the cuts-vs-hikes conclusion, which uses the same three hikes and still "
            "washes out against cuts."
        ),
        md(
            "### 4e · Era contrast — panic-easing (2008-2012) vs secular grind (2015-2025), "
            "cut days only\n\n"
            "Split at **2015-01-01**, justified ex ante: the RRR's last hike was June 2011; "
            "every cut since 2015 has been well telegraphed days ahead by State Council "
            "meetings, vs the more acute 2008-2012 crisis pivots."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(DF, 'ret', data.ZERO_RATE_ERA_SPLIT)\n"
            "    e, l = ec['early_pct'], ec['late_pct']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_early_pct'], R['era_late_pct']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['2008-2012\\npanic-easing (n=7)', '2015-2025\\nsecular grind (n=24)'], [e, l],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "for i, (v, t_) in enumerate([(e, et), (l, lt)]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t_:+.2f})', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('FXI cut-day mean return (%)')\n"
            "ax.set_title(f'The modern slice clears t=2 alone, but the DIFFERENCE does not '\n"
            "             f'(diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.3f}% (t={et:+.2f})  late {l:+.3f}% (t={lt:+.2f})  diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the 2015-2025 cut slice alone crosses the bar "
            f"(*t* = {R['era_late_t']:.2f}, n = {R['era_late_n']}) — the single closest thing "
            "to a \"real\" number in this whole study. But the difference between the two eras "
            f"is **not certified** (*t* = {R['era_diff_t']:.2f}), the early-era *n* is tiny "
            f"({R['era_early_n']}), and — decisively — the SAME modern era shows no directional "
            "signature at all once hikes are brought back into the comparison (4a/4d). Read "
            "honestly: the modern cut window overlaps several broad China-equity bull legs "
            "(2015 rally, 2019-2021 recovery) — consistent with clustering, not a certified "
            "announcement effect."
        ),
        md(
            "### 4f · The third axis — the buy-the-cut timer, with costs\n\n"
            "Enter FXI at the prior close, hold 1/3/5/10 trading days, exit; 2 x one-way cost "
            "(5 bps) x NAV per round trip. Rest-of-tape null: every same-horizon rolling window "
            "that doesn't touch any RRR event's [-5..+10] window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    caps = {h: st.capture_horizon(FXI['AdjClose'], MATCHED_FXI, 'cut', h, 5.0)\n"
            "            for h in (1, 3, 5, 10)}\n"
            "    net = [caps[h]['net_bps'] for h in (1, 3, 5, 10)]\n"
            "    tt = [caps[h]['welch_t'] for h in (1, 3, 5, 10)]\n"
            "    worst = [caps[h]['worst_pct'] for h in (1, 3, 5, 10)]\n"
            "else:\n"
            "    net = [R['timer'][h][1] for h in (1, 3, 5, 10)]\n"
            "    tt = [R['timer'][h][3] for h in (1, 3, 5, 10)]\n"
            "    worst = [R['timer'][h][5] for h in (1, 3, 5, 10)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['1d', '3d', '5d', '10d'], net, color=AMBER, width=.6)\n"
            "for i, v in enumerate(net): a1.annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('net bps/event'); a1.set_title('Positive on paper at every horizon...')\n"
            "a2.bar(['1d', '3d', '5d', '10d'], tt, color=[RED if abs(t) >= 2 else GREY for t in tt], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t vs a random window'); a2.set_title('...none of it is certifiable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net bps:', dict(zip((1, 3, 5, 10), [round(n, 1) for n in net])))\n"
            "print('Welch t:', dict(zip((1, 3, 5, 10), [round(t, 2) for t in tt])))\n"
            "print('worst %:', dict(zip((1, 3, 5, 10), [round(w, 1) for w in worst])))"
        ),
        md(
            f"> 💡 In plain words: every horizon nets positive on paper (best: "
            f"+{R['timer'][5][1]:.0f} bps at the 5-day hold) but every single Welch *t* sits "
            f"well under 2 (best: {max(v[3] for v in R['timer'].values()):.2f}). The 10-day "
            f"hold — long enough to ask \"did this survive the news cycle\" — nets "
            f"+{R['timer'][10][1]:.0f} bps at *t* = {R['timer'][10][3]:.2f}, statistically "
            f"identical to a random 10-day window, with a **{R['timer'][10][5]:.1f}%** "
            "worst-case single event. **H₅ not certified; Tradability = MIRAGE**."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic i.i.d.-return world, scheduled pseudo-cut days every 20th business day, "
            "TUNABLE planted mean shift. The null (effect = 0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, dec = data.synthetic_world(effect=0.0, seed=647 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, dec)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, dec = data.synthetic_world(effect=0.003, seed=647)\n"
            "planted_t = st.synthetic_detect(close, dec)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (effect=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted effect = +0.30%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (event vs rest)')\n"
            "ax.set_title('Control: the null mostly does not fire; a planted shift lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the null fires in {R['syn_null_fire']}/20 seeds — ordinary "
            "sampling noise at this seed count, not a biased detector — and a planted shift "
            f"about the same size as the real cut-day gap reads *t* = {R['syn_planted_t']:.2f}. "
            "The machinery works; the real tape's `NONE` verdict is a genuine finding, not a "
            "broken pipeline. *(A faithful-engine / power check only — never cited in support "
            "of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — cut-day *t* = {R['cut_t']:.2f}, hike-day *t* = "
            f"{R['hike_t']:.2f}, neither clears the bar; the decisive **cuts-vs-hikes Welch "
            f"*t* = {R['cvh_t']:.2f}** (MCHI: {R['mchi_cvh_t']:.2f}) shows the folklore's own "
            "distinguishing prediction fails outright. RRR days ARE louder than average "
            f"(range *t* = +{R['range_t']:.2f}) but direction carries no reliable sign. The "
            f"one number that clears the bar alone — the 2015-2025 cut-only slice, *t* = "
            f"{R['era_late_t']:.2f} — is not a certified regime shift (diff *t* = "
            f"{R['era_diff_t']:.2f}) and is itself undercut by the wash test.\n"
            f"- **Tradability `MIRAGE`** — no hold length (1/3/5/10 days) reaches *t* = 2; the "
            f"10-day hold shows zero excess over a random window (*t* = {R['timer'][10][3]:.2f}) "
            f"with a {R['timer'][10][5]:.1f}% worst-case tail; FXI's own swap/QFII structure "
            "adds fragility on top.\n"
            f"- **\"Buy the rumor, sell the news\"? `BUSTED`** — pre-cut run-up is *negative* "
            f"(*t* = {R['runup_t']:.2f}) and the post-cut window shows no clean decay "
            f"(*t* = {R['post_t']:.2f}). Not a subtler mechanism to uncover — it simply isn't "
            "there."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The likely confound.** RRR cuts cluster in stretches when policy is already "
            "leaning supportive — property-market stress, growth wobbles, coordinated equity-"
            "market support (\"national team\" buying). That's consistent with the loud-but-"
            "directionless pattern here: RRR days correlate with supportive macro windows "
            "without the RRR move itself doing the causal work.\n"
            "- **Where a real edge might live instead:** policy-RATE surprises (the 1yr/5yr "
            "LPR fixings) carry more information content per announcement than the RRR, which "
            "is largely mechanical and well-telegraphed by State Council pre-announcements — "
            "the natural sequel study.\n"
            "- **Dedup map:** [620-a-h-premium](../../620-a-h-premium/) (structural A/H price "
            "gap, no event calendar), [313-geopolitical-shock](../../313-geopolitical-shock/) "
            "(same event-study skeleton, a different market and claim entirely — both studies "
            "land on `None x Mirage` via the same protocol, a small proof the shared machinery "
            "isn't rigged to manufacture positives).\n\n"
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
