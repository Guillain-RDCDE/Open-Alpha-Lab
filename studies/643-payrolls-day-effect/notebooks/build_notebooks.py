"""Generate the two narrative notebooks for Study 643 (Payrolls-Day-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY tape
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY 1997-01-02
# -> 2026-06-30; 353 hardcoded actual NFP release dates).
R = dict(
    start="1997-01-02", end="2026-06-30", n_nfp=353, n_rest=7065,
    cal_lo="1997-01-10", cal_hi="2026-06-05", n_mapped=8,
    nfp_bps=+12.43, rest_bps=+3.37, gap_bps=+9.06,
    welch_t=+1.31, nw_t=+1.31,
    hit=206, hit_pct=58.4, wilson=(53.2, 63.4), rest_hit_pct=54.1,
    placebo_p=0.079, placebo_mean=+3.33, placebo_sd=6.31, placebo_draws=20000,
    # event window: offset -> (mean bps, welch t)
    event={-3: (+6.10, +0.30), -2: (+17.05, +2.08), -1: (-2.05, -0.93),
           0: (+12.43, +1.19), 1: (-7.16, -1.60), 2: (+0.67, -0.53), 3: (-4.90, -1.30)},
    runup_bps=+21.10, runup_t=+2.07,
    spy_fomc=1.474, spy_rest=1.342, spy_t=+2.50,   # NFP-day vs other-day realized range
    era_early=+13.58, era_early_n=192, era_early_t=+1.16,
    era_late=+11.06, era_late_n=161, era_late_t=+0.63, era_diff_t=-0.19,
    # naive timer third axis
    tm_gross=12.43, tm_net5=2.43, tm_net10=-7.57, tm_rest=3.37, tm_t=+1.31,
    tm_n=353, tm_hit=58.4, tm_worst=-6.0,
    tm_ann5=0.29, tm_ann10=-0.91,
    # synthetic control
    syn_null_mean=+0.16, syn_null_sd=1.20, syn_null_fire=2, syn_planted_t=+2.46,
    fp_spy="7eebd76787d8",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Louder%2C_not_directional%3F: Confirmed](https://img.shields.io/badge/Louder%2C_not_directional%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from payrolls_day_effect import data, strategy as st

NFP_CAL = data.nfp_calendar()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY = data.load_real()
    SESSIONS, N_MAPPED = data.map_to_sessions(SPY.index, NFP_CAL)
    DF = st.day_frame(SPY, SESSIONS)
else:
    SPY = SESSIONS = DF = None
print("real cache present:", HAVE_REAL, "| scheduled NFP releases:", len(NFP_CAL),
      "| tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the market really move on Payrolls Friday? 📰📆\n"
            "### The Nonfarm-Payrolls day effect — a legend that's *half* true, and "
            "the tradable half isn't the half you'd think\n\n"
            + BADGES +
            "Once a month, on a Friday morning published **months in advance**, the "
            "Bureau of Labor Statistics drops the Employment Situation report at 8:30 am "
            "Eastern — the single most-watched number in macro. Trading desks staff up for "
            "it, volatility markets price it in, and financial media treat it like a "
            "market-moving event unto itself.\n\n"
            "That's the claim we test: *does the S&P actually behave systematically on "
            "payrolls morning — and if so, can you get paid for knowing that?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 353 *actual* NFP release dates (1997→2026), hardcoded from "
            "BLS records — not a \"first Friday\" pattern guess. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does SPY move *more* on payrolls morning? | **Yes.** The high-low swing is "
            f"**{R['spy_fomc']:.2f}%** vs **{R['spy_rest']:.2f}%** on a normal day — a real, "
            "measurable bump in how loud the session is. |\n"
            f"| Does it move in a *knowable direction*? | **Not really.** The average "
            f"release-day return is **+{R['nfp_bps']:.1f} bps** vs +{R['rest_bps']:.1f} bps "
            "normally — bigger, but not statistically distinguishable from luck (a random "
            f"353-day calendar produces something this big about **{R['placebo_p']*100:.0f}% "
            "of the time**). |\n"
            "| Is there a build-up before the report? | **Maybe, faintly** — but it's one "
            "flagged hit among seven things we checked, not a certified pattern. |\n"
            f"| Can you trade it? | **No.** A naive \"own SPY only on payrolls day\" plan "
            f"nets essentially nothing after 5 bps of costs (**+{R['tm_ann5']:.2f}%/yr**) and "
            f"loses money at 10 bps — and a single bad release morning can cost "
            f"**{R['tm_worst']:.1f}%**. |\n\n"
            "> The morning is louder. The trade isn't there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Payrolls Friday is the biggest data point of the month — the market either "
            "breathes a sigh of relief once it's out, or builds up nervous energy beforehand. "
            "Either way, you can feel it in the tape.\"*\n\n"
            "Academics agree scheduled macro releases produce real, measurable spikes in "
            "realized volatility (Andersen & Bollerslev 1998; Balduzzi, Elton & Green 2001). "
            "The question isn't whether the report *matters* — it's whether that mattering "
            "shows up as a **direction you could have bet on**, not just noise."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If payrolls morning has a systematic tilt, it's one of the cleanest possible "
            "trades: a single, scheduled, public-calendar bet, twelve times a year, no "
            "guesswork about *when*. If it's just noise wearing a headline, the honest lesson "
            "is that \"the market moves on big news days\" doesn't automatically mean \"you "
            "can predict which way\" — loud and knowable are two different things."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_nfp']}** actual NFP release dates from "
            f"{R['cal_lo']} to {R['cal_hi']}, hardcoded from BLS records (not a weekday-"
            "pattern guess). The report lands at 8:30 am ET, before the 9:30 am open, so the "
            "daily close-to-close bar fully contains the reaction.\n"
            "- **The comparison.** SPY's release-day return vs the other "
            f"**{R['n_rest']:,}** trading days since 1997.\n"
            "- **The luck check.** Draw 353 random days instead, 20,000 times, two-sided — "
            "how often does a random calendar produce a gap this large in *either* "
            "direction?\n"
            "- **The trade check.** Buy SPY at the close *before* release day (the date is "
            "public months ahead), sell at the release-day close, pay costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average daily SPY return on release days vs every "
            "other day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.nfp_day_stats(DF)\n"
            "    fp, rp = s['nfp_bps'], s['rest_bps']\n"
            "else:\n"
            "    fp, rp = R['nfp_bps'], R['rest_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['NFP release days\\n(n=353)','all other days\\n(n=7,065)'], [fp, rp],\n"
            "       color=[AMBER, GREY], width=.6)\n"
            "for i,v in enumerate([fp, rp]): ax.annotate(f'{v:+.2f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average daily SPY return (bps)')\n"
            "ax.set_title('Payrolls-day return is bigger on paper - but is it real?')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'release days {fp:+.2f} bps   other days {rp:+.2f} bps')"
        ),
        md(
            f"**+{R['nfp_bps']:.2f} bps** on the average release day vs **+{R['rest_bps']:.2f} "
            f"bps** normally — about **3.7×** the baseline. SPY rose on **{R['hit']}/353 = "
            f"{R['hit_pct']:.1f}%** of release days, but ordinary days already rise "
            f"**{R['rest_hit_pct']:.1f}%** of the time (SPY drifts up over the long run) — so "
            "that hit rate is barely above the baseline. The quants notebook shows a random "
            f"calendar beats this gap about **{R['placebo_p']*100:.0f} times in 100** — not "
            "the 1-in-20,000 the FOMC study found for the VIX crush.\n\n"
            "**Next, the anatomy.** What does the week around release day look like?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, SESSIONS)\n"
            "    ks, ms = list(ev.index), list(ev['mean_bps'])\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "cols = [RED if k==0 else (AMBER if k==-2 else GREY) for k in ks]\n"
            "ax.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "for i,v in enumerate(ms): ax.annotate(f'{v:+.1f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days relative to the NFP release (0 = release day)')\n"
            "ax.set_ylabel('average SPY return (bps)')\n"
            "ax.set_title('A noisy week, one flagged pre-release hint, no clean pattern')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('offsets:', {k: round(m,2) for k,m in zip(ks, ms)})"
        ),
        md(
            "The bars bounce around without a clean story. Day −2 (**+17.05 bps**) and the "
            "cumulative 3-day run-up nominally clear the desk's statistical bar — but that's "
            "**one flagged result among seven offsets we checked**, exactly the kind of "
            "after-the-fact pattern honest research is supposed to be suspicious of. Day +1 "
            "is *negative* (−7.16 bps) — if the report were cleanly bullish, you'd expect "
            "follow-through, not a give-back.\n\n"
            "**And here's the interesting part** — the market IS louder that morning, even "
            "though it isn't more *predictable*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.spy_range_stats(DF)\n"
            "    a, b = rg['nfp_range_pct'], rg['rest_range_pct']\n"
            "else:\n"
            "    a, b = R['spy_fomc'], R['spy_rest']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['NFP days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg SPY high-low range (% of prev close)')\n"
            "ax.set_title('Payrolls mornings are objectively louder (Welch t = +2.50)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: NFP days {a:.3f}%  vs other {b:.3f}%')"
        ),
        md(
            f"SPY swings **{R['spy_fomc']:.2f}%** high-to-low on release days vs "
            f"**{R['spy_rest']:.2f}%** normally — this one *does* clear the statistical bar "
            "(Welch *t* = +2.50). Payrolls mornings really are noisier. It's the same "
            "\"scheduled uncertainty resolving\" signature the desk found for the VIX on Fed "
            "decision days ([637-fomc-vol-crush](../../637-fomc-vol-crush/)) — except there, "
            "the *direction* was also certified. Here, only the *loudness* is.\n\n"
            "**Finally, the trade.** Even taking the raw numbers at face value, does the "
            "naive \"own it only on release day\" plan actually pay?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc5 = st.timer_capture(DF, cost_bps=5.0)\n"
            "    tc10 = st.timer_capture(DF, cost_bps=10.0)\n"
            "    g, n5, n10 = tc5['gross_bps'], tc5['net_bps'], tc10['net_bps']\n"
            "else:\n"
            "    g, n5, n10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['gross','net @ 5 bps','net @ 10 bps'], [g, n5, n10],\n"
            "       color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bps per release event')\n"
            "ax.set_title('The naive timer: thin, then gone, once costs show up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer: gross {g:+.2f} -> net {n5:+.2f} (5bps) / {n10:+.2f} (10bps) bps')"
        ),
        md(
            f"Gross **+{R['tm_gross']:.2f} bps/event** becomes **+{R['tm_net5']:.2f} bps** "
            f"after 5 bps costs (~+{R['tm_ann5']:.2f}%/yr from twelve trades a year — not "
            f"worth the screen time) and **{R['tm_net10']:+.2f} bps** — negative — at 10 bps. "
            f"The worst single release day cost **{R['tm_worst']:.1f}%**. And remember: the "
            "gross number was never statistically certified to begin with (*t* = "
            f"{R['tm_t']:.2f}). There's no edge here to protect."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** SPY does move more on release day on average "
            f"(+{R['nfp_bps']:.1f} vs +{R['rest_bps']:.1f} bps), but the gap doesn't clear "
            "the statistical bar (Welch *t* = +1.31, placebo *p* = 0.079) and the hit rate "
            "barely beats the baseline.\n"
            "- **Tradability — Mirage.** The naive timer nets almost nothing after realistic "
            "costs and goes negative at a slightly higher cost, on top of an uncertified "
            "point estimate and a −6% worst day.\n"
            "- **\"Louder, not directional?\" — Confirmed.** The realized range genuinely "
            "clears the bar — payrolls mornings ARE noisier, mechanically. That loudness "
            "just never becomes a bankable direction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The pre-release hint deserves its own honest test.** A dedicated,\n"
            "  pre-registered study of the [−3..−1] window (with a multiple-comparison\n"
            "  correction built in from the start) is the natural sequel — this study flags\n"
            "  it, it doesn't certify it.\n"
            "- **Where the real edge (if any) probably lives** is in the *options market*\n"
            "  around the print — the implied-vol crush after the number, mirroring what\n"
            "  [637-fomc-vol-crush](../../637-fomc-vol-crush/) found for FOMC afternoons —\n"
            "  not in a directional long-SPY timer.\n"
            "- **Sibling studies:** [385-jobless-claims-momentum](../../385-jobless-claims-momentum/)\n"
            "  (weekly claims as a slow *leading* indicator) and\n"
            "  [602-macro-announcement-premium](../../602-macro-announcement-premium/) (the\n"
            "  pooled CPI+FOMC+NFP bundle, shown there to be an FOMC-only effect) ask\n"
            "  related but distinct questions — neither isolates the NFP morning itself.\n\n"
            "*Think the pre-release drift is real? Show a corrected, pre-registered *t* ≥ 2 "
            "on an out-of-sample window — then we'll talk.*"
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
            "# The Payrolls-Day-Effect — a quantitative teardown 🔬\n"
            "### Release-day Welch/HAC splits · a two-sided 20-seed random-calendar placebo · "
            "the [−3..+3] event anatomy (and its multiple-comparisons caveat) · the "
            "realized-range cross-check · the era contrast · an honest naive-timer cost sweep "
            "· a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **SPY behaves systematically on NFP release mornings** — has a real "
            "academic anchor for the *volatility* half (Andersen-Bollerslev 1998,\n"
            "Balduzzi-Elton-Green 2001) but no consensus anchor for a *directional* edge. The "
            "job here is to measure both halves honestly, name the actual (not\n"
            "weekday-pattern-guessed) release calendar, and ask the only question that pays: "
            "*is any of it tradable?*\n\n"
            "> ⚠️ **Data note.** SPY raw OHLC + adjusted close (1997→2026), yfinance, cached; "
            "**353 hardcoded actual NFP release dates** (BLS archived-news-release index, "
            "identical table to sibling study 602), 8 forward-mapped off market holidays. No "
            "survivorship on either axis (SPY is an index-tracking ETF). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | release-day return **{R['nfp_bps']:+.2f} bps** vs "
            f"{R['rest_bps']:+.2f}: Welch **t = {R['welch_t']:.2f}**, NW **t = "
            f"{R['nw_t']:.2f}**, placebo **p = {R['placebo_p']:.3f}** (two-sided) |\n"
            f"| **Tradability** | `MIRAGE` | naive timer net {R['tm_net5']:+.2f} bps @ 5 bps "
            f"cost, {R['tm_net10']:+.2f} bps @ 10 bps; worst day {R['tm_worst']:.1f}% |\n"
            f"| **Louder, not directional?** | `CONFIRMED` | realized range *t* = "
            f"{R['spy_t']:.2f} (clears bar) vs return *t* = {R['welch_t']:.2f} (does not) |\n\n"
            "> 💡 In plain words: the tape confirms the *mechanism* (scheduled news makes for "
            "a noisier morning) but not the *folklore's payoff* (a knowable direction you can "
            "trade)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be SPY's daily close-to-close log return and $D_t \\in \\{0,1\\}$ the "
            "actual-NFP-release-day flag (known *ex ante* — the BLS calendar is published "
            "months ahead). The report (08:30 ET) predates the SPY open (09:30 ET), so the "
            "release-day bar contains the full reaction. The claims:\n\n"
            "- **H₁ (systematic).** $E[r_t \\mid D_t=1] \\ne E[r_t \\mid D_t=0]$ — release-day "
            "returns differ systematically from ordinary days, not just by chance.\n"
            "- **H₂ (anatomy).** A pre-release drift and/or a persistence pattern shows up in "
            "the days around the release.\n"
            "- **H₃ (louder, not just biased).** Realized SPY range on release days exceeds "
            "the baseline — a resolution-of-scheduled-uncertainty signature independent of "
            "direction.\n"
            "- **H₄ (capture).** A naive directional timer (own SPY only on release day) "
            "banks the point estimate net of costs.\n\n"
            "We find **H₁ not certified** (Welch *t* = 1.31 < 2), **H₂ flagged but not "
            "certified** (one uncorrected hit among seven offsets), **H₃ supported** (range "
            "*t* = +2.50), **H₄ not certified and unprofitable** (gross uncertified, net "
            "negative at 10 bps)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Release days are **single, non-overlapping events**, so the planned primary is "
            "a **Welch t** on the group split. Because daily returns are weakly serially "
            "correlated, we cross-check with a **Newey-West (5-lag) t** on the dummy "
            "regression $r_t = a + b D_t$ — the slope *is* the mean gap. The hit rate carries "
            "a **Wilson interval**, benchmarked against the **baseline** up-day rate (not a "
            "naive 50%, since SPY already drifts up over time). The claim carries **no "
            "a-priori sign**, so the random-calendar placebo (20 seeds × 1,000 draws of 353 "
            "random days) is **two-sided**: $p = \\Pr(|\\bar{r}_{\\text{placebo}}| \\ge "
            "|\\bar{r}_{\\text{obs}}|)$."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_nfp']} actual NFP releases {R['cal_lo']} → {R['cal_hi']}, "
            "hardcoded from BLS records (source-verified, shared with sibling study 602), "
            f"{R['n_mapped']} forward-mapped off market holidays.\n"
            f"- **Tape.** SPY raw OHLC + adjusted close {R['start']} → {R['end']}. As-of "
            "2026-06-30 (last complete month).\n"
            "- **Headline.** Welch t + NW(5) t + Wilson hit rate (vs baseline) + two-sided "
            "20-seed placebo.\n"
            "- **Anatomy.** Event window [−3..+3], per-offset Welch t vs far days; cumulative "
            "run-up per event, one-sample t — reported with an explicit multiple-comparisons "
            "caveat.\n"
            "- **Cross-check.** SPY (H−L)/prev-close split on the same days.\n"
            "- **Execution (third axis).** Enter the naive timer at the prior close (calendar "
            "public — zero look-ahead), exit the release close; 2 × one-way cost × NAV per "
            "event; long-only, no borrow.\n"
            "- **Control.** Synthetic random-walk daily returns, planted release-day-effect "
            "knob; the null must not fire at an unusual rate across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo\n\n"
            "Welch t on release-day return, NW t on the dummy regression, and the two-sided "
            "random-calendar null. In the notebook we run a lighter placebo (4 seeds × 500 "
            "draws) and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.nfp_day_stats(DF)\n"
            "    print(f\"NFP-day return {s['nfp_bps']:+.2f} bps  vs  other {s['rest_bps']:+.2f} bps\")\n"
            "    print(f\"Welch t = {s['welch_t']:+.2f}   NW(5) t = {s['nw_t']:+.2f}\")\n"
            "    print(f\"hit {s['hit_up']}/{s['n_nfp']} = {s['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    pl = st.placebo_pvalue(DF, n_draws_per_seed=500, n_seeds=4)\n"
            "    obs, draws = pl['obs'] * 1e4, pl['draws'] * 1e4\n"
            "else:\n"
            "    obs = R['nfp_bps']\n"
            "    rng = np.random.default_rng(643)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 353 days (light in-notebook run)')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'observed NFP-day mean {obs:+.2f} bps')\n"
            "ax.set_xlabel('mean return of a random 353-day calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Inside the luck cloud: canonical p = {R['placebo_p']:.3f} \"\n"
            "             '(20 seeds x 1,000 draws, two-sided)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f}, \"\n"
            "      f\"sd {R['placebo_sd']:.2f}, p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['nfp_bps']:+.2f} bps** sits well within "
            f"the null's spread ({R['placebo_mean']:+.2f} ± {R['placebo_sd']:.2f} bps); "
            f"**p = {R['placebo_p']:.3f}** — about 1 in 13 random calendars beats it. With "
            f"Welch t = **{R['welch_t']:.2f}** and NW t = **{R['nw_t']:.2f}**, H₁ does **not** "
            "clear the desk bar."
        ),
        md(
            "### 4b · Anatomy — a scan, and an honest multiple-comparisons caveat\n\n"
            "Per-offset means with Welch t vs far-from-release days; the pre-release window "
            "is also tested as a **cumulative per-event** quantity (one-sample t across 353 "
            "events)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, SESSIONS)\n"
            "    ks = list(ev.index); ms = list(ev['mean_bps']); ts = list(ev['welch_t'])\n"
            "    ru = st.runup_stats(DF, SESSIONS)\n"
            "    ru_m, ru_t = ru['mean_runup_bps'], ru['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][1] for k in ks]; ru_m, ru_t = R['runup_bps'], R['runup_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k==0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean return (bps)')\n"
            "a1.set_title('Event anatomy: a noisy week, no certified pattern')\n"
            "a2.bar([str(k) for k in ks], ts, color=[AMBER if abs(t)>=2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from release)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cumulative pre-release run-up [-3..-1]: {ru_m:+.2f} bps/event (t = {ru_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: offset −2 (**{R['event'][-2][0]:+.2f} bps**, t = "
            f"{R['event'][-2][1]:+.2f}) and the cumulative run-up (**{R['runup_bps']:+.2f} "
            f"bps/event**, t = {R['runup_t']:+.2f}) both nominally clear *t* ≥ 2 — but this "
            "scan checked **seven offsets** with no correction applied. At a naive 5% level "
            "you'd expect roughly one false positive out of seven tests by chance alone; "
            "we're looking at exactly one. Honest reading: **flagged as a hint, not banked "
            f"as a finding.** The release day itself, day 0 ({R['event'][0][0]:+.2f} bps, t = "
            f"{R['event'][0][1]:+.2f}), and day +1 ({R['event'][1][0]:+.2f} bps, t = "
            f"{R['event'][1][1]:+.2f}, a *give-back* not a continuation) show no clean "
            "pattern at all."
        ),
        md(
            "### 4c · Louder, not necessarily biased — the realized-range cross-check\n\n"
            "If release days were simply *ordinary*, this split should be flat. It isn't:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.spy_range_stats(DF)\n"
            "    a, b, t = rg['nfp_range_pct'], rg['rest_range_pct'], rg['welch_t']\n"
            "else:\n"
            "    a, b, t = R['spy_fomc'], R['spy_rest'], R['spy_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(['release days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('SPY (H-L)/prev close, mean (%)')\n"
            "ax.set_title(f'Realized range IS elevated on release days (Welch t = {t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: NFP {a:.3f}%  vs other {b:.3f}%   Welch t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: SPY moves **{R['spy_fomc']:.2f}%** high-to-low on release "
            f"days vs **{R['spy_rest']:.2f}%** otherwise (t = {R['spy_t']:+.2f}) — this **does** "
            "clear the bar. Compare to study 637's FOMC finding: there, both the *level* "
            "(implied vol) and the *loudness* (realized range) cleared the bar together. "
            "Here, only the loudness does — the direction is genuinely unresolved by this "
            "tape. H₃ holds; H₁ doesn't."
        ),
        md(
            "### 4d · The era contrast — justified split, tested as a difference\n\n"
            "Split at **2013-01-01** (roughly the sample midpoint / the post-crisis-"
            "normalization, QE-taper-talk era; chosen ex ante as a round, structurally "
            "sensible date, not snooped)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(DF, '2013-01-01')\n"
            "    e, l = ec['early_bps'], ec['late_bps']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_early'], R['era_late']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['1997 - 2013\\n(n=192)','2013 - 2026\\n(n=161)'], [e, l],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    ax.annotate(f'{v:+.2f} bps\\n(within-era t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('NFP-day mean return (bps)')\n"
            "ax.set_title(f'Neither era certifies it alone (diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.2f} (t={et:+.2f})  late {l:+.2f} (t={lt:+.2f})  diff t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early']:+.2f} bps early vs {R['era_late']:+.2f} bps "
            f"late — a mild apparent decline, but neither slice is certified alone "
            f"(t = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}) and the difference itself "
            f"is statistically nothing (t = {R['era_diff_t']:+.2f}). No decay story to tell — "
            "there was nothing large enough to decay from."
        ),
        md(
            "### 4e · The third axis — the honest naive-timer cost sweep\n\n"
            "Enter the prior close (zero look-ahead — the calendar is public months ahead), "
            "exit the release close, pay 2 × one-way costs per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc5 = st.timer_capture(DF, cost_bps=5.0)\n"
            "    tc10 = st.timer_capture(DF, cost_bps=10.0)\n"
            "    g, n5, n10 = tc5['gross_bps'], tc5['net_bps'], tc10['net_bps']\n"
            "    tv, worst, hitr = tc5['welch_t'], tc5['worst_day_pct'], tc5['hit_rate']\n"
            "else:\n"
            "    g, n5, n10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "    tv, worst, hitr = R['welch_t'], R['tm_worst'], R['tm_hit'] / 100\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['gross','net 5 bps','net 10 bps'], [g, n5, n10],\n"
            "       color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bps per release event')\n"
            "ax.set_title(f'Thin even before costs (Welch t = {tv:+.2f}), gone after')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} -> net {n5:+.2f} (5bps) / {n10:+.2f} (10bps) bps;  '\n"
            "      f'hit rate {hitr*100:.1f}%  worst day {worst:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: +{R['tm_net5']:.2f} bps net per event at 5 bps "
            f"(~+{R['tm_ann5']:.2f}%/yr from twelve releases) sitting on top of an already-"
            f"uncertified gross estimate (*t* = {R['welch_t']:.2f}), and **{R['tm_net10']:+.2f} "
            f"bps — negative — at 10 bps**. A {abs(R['tm_worst']):.1f}% worst day is a normal "
            "bad session, not a tail event, and it alone would wipe out roughly two years of "
            "the 5-bps net edge. **H₄ not certified; Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk daily-return world, scheduled pseudo-releases every 21st "
            "business day, TUNABLE planted release-day effect. The null (edge = 0) is checked "
            "over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, rel = data.synthetic_world(edge=0.0, seed=643 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, rel)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, rel = data.synthetic_world(edge=0.0015, seed=643)\n"
            "planted_t = st.synthetic_detect(close, rel)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=AMBER, s=90, zorder=5,\n"
            "           label='planted edge = +15 bps/day')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (release vs rest)')\n"
            "ax.set_title('Control: the null rarely fires; a planted effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires "
            f"({R['syn_null_fire']}/20 seeds) at roughly the rate a two-sided *t* = 2 test "
            "should — no systematic bias in the harness. A planted +15 bps/day effect (about "
            f"the real observed gap's order of magnitude) reads t = {R['syn_planted_t']:.2f}, "
            "clearing the bar — the machinery *can* detect an effect this size; the real "
            "tape's own *t* = 1.31 says it just isn't there at the certified level. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — release-day return **{R['nfp_bps']:+.2f} bps** vs "
            f"{R['rest_bps']:+.2f}: Welch t = **{R['welch_t']:.2f}**, NW t = "
            f"**{R['nw_t']:.2f}**, placebo p = **{R['placebo_p']:.3f}** (two-sided); hit "
            f"{R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]) vs "
            f"a {R['rest_hit_pct']:.1f}% baseline. A nominal pre-release drift (t = "
            f"{R['runup_t']:+.2f}) is one uncorrected hit among seven offsets — flagged, "
            "not certified.\n"
            f"- **Tradability `MIRAGE`** — the naive timer nets {R['tm_net5']:+.2f} bps/event "
            f"at 5 bps (~{R['tm_ann5']:+.2f}%/yr) and {R['tm_net10']:+.2f} bps at 10 bps, on "
            f"top of an uncertified gross estimate, with a {R['tm_worst']:.1f}% worst day.\n"
            f"- **\"Louder, not directional?\" `CONFIRMED`** — realized range clears the bar "
            f"(t = {R['spy_t']:+.2f}) while the return itself does not (t = "
            f"{R['welch_t']:+.2f}). Scheduled news makes for a genuinely noisier morning; it "
            "does not hand you a certified direction to trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is scheduled-announcement volatility**, not a directional "
            "premium. Andersen-Bollerslev (1998) and Balduzzi-Elton-Green (2001) document the "
            "loudness half broadly; the professional expression is options straddles around "
            "the print, not a directional cash-equity timer.\n"
            "- **The pre-release hint is worth its own study** — pre-registered, with a "
            "multiple-comparisons correction stated *before* the scan, on data this study "
            "hasn't touched.\n"
            "- **Dedup map:** [385-jobless-claims-momentum](../../385-jobless-claims-momentum/) "
            "(weekly claims, a slow leading indicator, different series and clock), "
            "[602-macro-announcement-premium](../../602-macro-announcement-premium/) (the "
            "pooled CPI+FOMC+NFP bundle, shown there to be FOMC-driven), and "
            "[637-fomc-vol-crush](../../637-fomc-vol-crush/) (the VIX collapse on FOMC "
            "afternoons — the sibling study where *both* the loudness and the direction "
            "certify).\n\n"
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
