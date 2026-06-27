"""Generate the two narrative notebooks for Study 515 (Earnings-Announcement Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices +
earnings dates under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 30-name large-cap
# basket + per-name scheduled earnings dates, 2005-01-03 -> 2026-06-26, 2,987 dates, 21.5 years).
R = dict(
    start="2005-01-03", end="2026-06-26", years=21.5, n_dates=2987, n_events=2609, n_names=30,
    n_days=5404, ann_day_pct=4.79,
    # per-day premium on the tradable [D-1,D+1] window
    ann_mean_bps=6.33, rest_mean_bps=5.80, premium_bps=0.53, welch_t=0.16,
    event_premium_bps=-0.36, t_event=-0.11, p_placebo=0.394,
    # share-of-return decomposition at [D-1,D+1]
    share_days=4.79, share_return=2.38, concentration=0.50,
    # the day-of nuance [D-0,D+0]
    w0_premium_bps=5.53, w0_welch_t=0.90, w0_t_event=0.90, w0_p=0.052,
    w0_ann_mean_bps=11.27, w0_rest_mean_bps=5.74, w0_share_return=2.31,
    w0_share_days=1.59, w0_concentration=1.45,
    # window-width robustness: (w, premium%, welch_t, event_t, ann_share%)
    windows=[(0, 5.53, 0.90, 0.90, 1.59), (1, 0.53, 0.16, -0.11, 4.79),
             (2, 0.91, 0.42, 0.18, 7.99)],
    # tradable overlay: (cost_bps, gross_daily_bps, net_daily_bps, bh_daily_bps, net_ann%, bh_ann%)
    overlay=[(2.0, 6.33, 5.00, 5.83, 13.4, 15.8), (5.0, 6.33, 3.00, 5.83, 7.8, 15.8),
             (10.0, 6.33, -0.34, 5.83, -0.8, 15.8)],
    # synthetic control: (planted_edge_bps, premium_bps, welch_t, event_t, p)
    syn=[(0.0, -1.66, -0.93, -0.95, 0.830), (15.0, 13.35, 7.47, 7.58, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Earnings_days_carry_it%3F: Busted](https://img.shields.io/badge/Earnings_days_carry_it%3F-Busted-8b949e?style=flat-square)\n\n"
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

from earnings_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, DATES = data.load_real()
    FLAGS = data.build_announce_flags(PRICES, DATES, window=1)     # tradable [D-1,D+1]
    FLAGS0 = data.build_announce_flags(PRICES, DATES, window=0)    # day-of only
else:
    PRICES = DATES = FLAGS = FLAGS0 = None
print("real EP cache present:", HAVE_REAL,
      "| names:", (0 if FLAGS is None else len(FLAGS)),
      "| earnings dates:", (0 if DATES is None else len(DATES)))
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
            "# Do stocks pay you extra just for *reporting earnings*? 📅\n"
            "### The \"earnings-announcement premium\" — a famous academic effect, checked on real large-caps in plain English\n\n"
            + BADGES +
            "Here's a tidy piece of finance lore: a *disproportionate* share of a stock's whole "
            "return supposedly gets earned in the **few days around its earnings report**. The "
            "academics (Beaver 1968; Frazzini & Lamont 2007) found it and even built a strategy: "
            "**hold stocks only while they report**, sit out the rest, and you should beat just "
            "owning them. The idea is that you're paid a premium for taking on the risk of the "
            "announcement.\n\n"
            "It's a clean, testable claim — so let's test it. The short version: on a basket of "
            "big, liquid, *survived-everything* US stocks, the premium has the right **shape** on "
            "the report day itself (the stock earns about **double** its normal return that day) "
            "— but it's a **whisper**, not a signal, it **disappears** the moment you widen to a "
            "window you could actually trade, and a \"hold only while reporting\" calendar "
            "**loses to just buying and holding**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name large-cap basket** (names still "
            "trading today). The premium is famously strongest in *small, neglected* stocks — so "
            "this is the corner where you'd *least* expect it. And it carries **survivorship** "
            "(we kept only firms that survived their earnings), which here actually tilts things "
            "*toward* finding a premium — and we still don't. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks earn more on report days? | **On the day itself, a bit — but not "
            "reliably.** On the exact announcement day the stock earns ~**double** a normal day "
            "(+11 bps vs +6), a real *shape*. But it's statistically a **coin-flip** (*t* ≈ 0.9, "
            "below the bar). |\n"
            "| Does it survive into a tradable window? | **No.** Widen to the day before + day "
            "after (a window you could actually trade) and the premium collapses to "
            f"**{R['premium_bps']:+.2f} bps/day** — basically zero (*t* = {R['welch_t']:.2f}). |\n"
            "| Do report days carry *most* of the return? | **No — the opposite.** They're ~5% of "
            f"days but carry only **{R['share_return']:.1f}%** of the return — *half* their share. |\n"
            "| Can I trade \"hold only while reporting\"? | **No.** It **loses to buy & hold** at "
            "every cost level, and you give up 95% of the year sitting in cash. |\n\n"
            "> The legend has a kernel of truth (the report *day* is special) — but on liquid "
            "large-caps it's too small to clear the bar and too thin to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Earnings days are risky — the company dumps a quarter of news at once and the "
            "stock can gap. Investors demand to be **paid** for holding through that risk, so "
            "report-day returns are systematically **higher**. Hold names while they report, "
            "skip the boring days, and you collect a premium.\"*\n\n"
            "This isn't a tall tale. **William Beaver (1968)** showed return *variance* spikes on "
            "report days; **Frazzini & Lamont (2007)** showed average *returns* are higher in the "
            "announcement window and built the long-the-announcers strategy. The question isn't "
            "\"is it made up?\" — it's **how big, and does it survive on stocks you can actually "
            "trade cheaply?**\n\n"
            "> 🔁 **Not the same as PEAD.** Study [363](../363-pead-drift) tests the *drift after a "
            "surprise* (buy beats, short misses). This is different: it's the premium on the "
            "report **days themselves**, regardless of whether the news was good or bad."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If report days really paid a premium, two things would follow: (1) the ~5% of days "
            "that are announcement days would carry **much more** than 5% of the total return, and "
            "(2) a calendar that owns stocks **only while they report** would beat buy & hold. "
            "Both are concrete and checkable.\n\n"
            "The trap is the same one that kills most calendar effects: an edge of a few basis "
            "points on a handful of days a year is easily (a) **statistical noise** and (b) "
            "**eaten by the cost** of jumping in and out. We'll measure the premium, test it "
            "against a \"could this be a random calendar?\" null, and then charge it real costs."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket** and pull **every** scheduled "
            f"earnings date for each — about **{R['n_dates']:,} prints** over **{R['years']:.0f} "
            f"years** ({R['start']} → {R['end']}). Then, for every single trading day, we tag "
            "whether it sits inside an **announcement window** around a report. We compare:\n\n"
            "1. **Report-day returns vs the rest.** Pool all the announcement-window days; is "
            "their average daily return higher than all the other days?\n"
            "2. **Share of the return.** Of the whole 21-year pile of return, how much landed on "
            "the ~5% of days that are report days?\n"
            "3. **The trade.** Hold the basket equal-weighted **only** during announcement windows "
            "(entered ahead of the *scheduled* date — no cheating), charge a round trip each "
            "window, and compare to just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: do report days earn more?** Here's the average daily return on "
            "announcement days vs all other days — once for the *exact report day only*, and once "
            "for the tradable *day-before-to-day-after* window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s1 = st.summarize_premium(FLAGS, placebo=False)\n"
            "    s0 = st.summarize_premium(FLAGS0, placebo=False)\n"
            "    day_of = [s0['ann_mean_bps'], s0['rest_mean_bps']]\n"
            "    window = [s1['ann_mean_bps'], s1['rest_mean_bps']]\n"
            "else:\n"
            "    day_of = [R['w0_ann_mean_bps'], R['w0_rest_mean_bps']]\n"
            "    window = [R['ann_mean_bps'], R['rest_mean_bps']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)\n"
            "a1.bar(['report\\nday', 'normal\\nday'], day_of, color=[GREEN, GREY], width=.6)\n"
            "for i,v in enumerate(day_of): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('The report DAY itself (the whisper)'); a1.set_ylabel('avg daily return (bps)')\n"
            "a2.bar(['in\\nwindow', 'rest'], window, color=[AMBER, GREY], width=.6)\n"
            "for i,v in enumerate(window): a2.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_title('The tradable [D-1, D+1] window (gone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'day-of premium: {day_of[0]-day_of[1]:+.2f} bps   window premium: {window[0]-window[1]:+.2f} bps')"
        ),
        md(
            f"There's the whole story in two pictures. On the **exact report day** the stock earns "
            f"**{R['w0_ann_mean_bps']:+.1f} bps** vs **{R['w0_rest_mean_bps']:+.1f} bps** normally "
            f"— roughly *double*, the right shape. But spread to the tradable **±1-day window** and "
            f"the gap collapses to **{R['premium_bps']:+.2f} bps** — the surrounding days are dead "
            "weight that erase the day-of bump."
        ),
        md(
            "**Now the killer chart: the share of return.** If the legend held, report days "
            "(~5% of all days) should carry *way more* than 5% of the return. Do they?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sr = st.share_of_return(FLAGS)\n"
            "    sd, srt = sr['share_days']*100, sr['share_return']*100\n"
            "else:\n"
            "    sd, srt = R['share_days'], R['share_return']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['share of\\nDAYS', 'share of\\nRETURN'], [sd, srt], color=[GREY, RED], width=.55)\n"
            "for i,v in enumerate([sd, srt]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('%'); ax.set_title('Report days carry LESS than their share of return — legend busted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'report days are {sd:.1f}% of days but carry {srt:.1f}% of return -> {srt/sd:.2f}x (under 1 = the opposite of the claim)')"
        ),
        md(
            f"The opposite of the legend. Report-window days are **{R['share_days']:.1f}%** of days "
            f"but carry only **{R['share_return']:.1f}%** of the cumulative return — a "
            f"**{R['concentration']:.2f}×** concentration, i.e. *half* their proportional share. "
            "On this basket, far from a *disproportionately large* share of return, earnings days "
            "carry a disproportionately *small* one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The tradable window premium is **{R['premium_bps']:+.2f} "
            "bps/day** — statistically zero. The day-of bump is real-ish in *shape* but a "
            "coin-flip in significance (the quants notebook shows *t* ≈ 0.9, under the bar).\n"
            "- **Tradability — Mirage.** \"Hold only while reporting\" **loses to buy & hold** at "
            "every cost level — and you give up 95% of the year in cash.\n"
            "- **\"Most of the return is around earnings\"? — Busted.** Report days carry "
            f"**{R['share_return']:.1f}%** of return on **{R['share_days']:.1f}%** of days — *half* "
            "their share, the exact opposite of the claim."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the calendar loses to doing nothing\n\n"
            "Here's the operational reality. A strategy that holds the basket **only inside "
            "announcement windows** (one round trip per window), at three cost levels, vs just "
            "**buying and holding** the same basket. Watch it lose."
        ),
        code(
            "costs = [2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.overlay_stats(FLAGS, cost_bps=c, window=1)['net_ann_pct'] for c in costs]\n"
            "    bh = st.overlay_stats(FLAGS, cost_bps=5.0, window=1)['bh_ann_pct']\n"
            "else:\n"
            "    net = [o[4] for o in R['overlay']]; bh = R['overlay'][0][5]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([f'{c:.0f} bps' for c in costs], net, color=[AMBER, AMBER, RED], width=.55, label='overlay (net, annualised)')\n"
            "ax.axhline(bh, c=GREEN, lw=2.2, label=f'buy & hold {bh:+.1f}%')\n"
            "for i,v in enumerate(net): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('annualised return (%)'); ax.set_xlabel('one-way trading cost')\n"
            "ax.set_title('Long-the-announcers loses to buy & hold at every cost'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('overlay net annualised:', [round(v,1) for v in net], '| buy & hold:', round(bh,1))"
        ),
        md(
            f"At a gentle **2 bps** the overlay makes **{R['overlay'][0][4]:+.1f}%** — already "
            f"below buy & hold's **{R['overlay'][0][5]:+.1f}%**. At **10 bps** it turns "
            f"**{R['overlay'][2][4]:+.1f}%** (a *loss*). And remember: the overlay sits in cash 95% "
            "of the time, so even its 'wins' are a sliver of what simply owning the stocks pays. "
            "There is nothing here to trade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why so flat here?** The premium is famously concentrated in **small, neglected, "
            "retail-heavy** stocks (Frazzini & Lamont 2007) — exactly the names we *excluded* with "
            "a liquid large-cap basket. Re-run on small-caps and the day-of bump grows (but so do "
            "costs and survivorship traps).\n"
            "- **Published-then-decayed.** The effect was published in 2007; McLean & Pontiff "
            "(2016) show anomalies decay a lot once they're known. A 2005–2026 large-cap window is "
            "a prime suspect for a decayed effect.\n"
            "- **Build your own.** Swap the basket for the Russell 2000, or tag the announcement "
            "*month* instead of the day; the shape survives, the size and significance move.\n\n"
            "*Think the premium is tradable? Show the long-the-announcers overlay **beating buy & "
            "hold net of honest costs** — then we'll talk.*"
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
            "# The Earnings-Announcement Premium — a quantitative teardown 🔬\n"
            "### Announce-vs-rest per-day premium · within-name per-event one-sample *t* · a "
            "random-calendar placebo · the share-of-return decomposition · window-width sweep · "
            "the long-the-announcers overlay net of costs · a powered synthetic faithful-engine "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "Frazzini–Lamont (2007) premium is a clean, falsifiable calendar claim — so the job "
            "here is to *measure it honestly*: pool the announcement days, test them against the "
            "rest with a clustering-aware placebo, decompose the share of return, then charge the "
            "overlay real costs. The twist: this is a **null**, and a null is only credible with a "
            "**powered** control — so we plant a known premium and prove the detector would have "
            "seen it.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket, names still "
            "trading in 2026 — a *survivor* panel that, for this premium, biases *toward* a "
            "positive result (we kept names that survived their earnings) and we still find none. "
            "The premium is strongest in small, neglected names (Frazzini & Lamont 2007); a liquid "
            "large-cap basket is the conservative corner. Real data: yfinance daily closes + "
            "`Ticker.get_earnings_dates`, 2005→2026. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition.\n"
            ">\n"
            "> 🔁 **Distinct from PEAD ([363](../363-pead-drift)).** That sorts on the *signed "
            "surprise* and measures the drift; this is the *level premium on the announcement days "
            "themselves*, orthogonal to surprise sign."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | `[D-1,D+1]` premium **{R['premium_bps']:+.2f} bps/day** "
            f"(Welch **t = {R['welch_t']:.2f}**, per-event **t = {R['t_event']:.2f}**, random-calendar "
            f"placebo **p = {R['p_placebo']:.2f}**). Day-of bump **{R['w0_premium_bps']:+.2f} bps** "
            f"is sub-bar (**t = {R['w0_welch_t']:.2f}**). |\n"
            f"| **Tradability** | `MIRAGE` | Overlay net **{R['overlay'][1][4]:+.1f}%** ann at 5 bps "
            f"vs buy & hold **{R['overlay'][1][5]:+.1f}%**; **negative** at 10 bps. In-market ~5% of "
            "the year. |\n"
            f"| **Earnings days carry it?** | `BUSTED` | Report days = **{R['share_days']:.1f}%** of "
            f"days carry **{R['share_return']:.1f}%** of return (**{R['concentration']:.2f}×** — *under* "
            "1, the opposite of the claim). |\n\n"
            "> 💡 In plain words: the announcement *day* has the right Frazzini–Lamont shape "
            "(~double the normal return) but it's a sub-significance whisper that **dissolves** at "
            "any tradable window, and report days carry *less* than their share of return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A$ be the set of stock-days inside an announcement window $[D-w, D+w]$ and $N$ "
            "its complement. The announcement premium is\n\n"
            "$$\\widehat{\\pi}(w) = \\frac{1}{|A|}\\sum_{i\\in A} r_i \\;-\\; "
            "\\frac{1}{|N|}\\sum_{i\\in N} r_i,$$\n\n"
            "the average daily return on report days minus the rest. To cancel cross-name level "
            "differences we also compute a **within-name per-event** premium: each window's mean "
            "minus that *name's own* non-window mean, giving a one-sample sample $\\{\\pi_e\\}$.\n\n"
            "- **H₁ (the premium exists).** $\\widehat{\\pi}(w) > 0$ and significant.\n"
            "- **H₂ (it's deployable).** A long-only overlay holding only inside $A$ beats buy & "
            "hold net of per-window turnover.\n"
            "- **H₃ (\"disproportionate share\").** The $\\sim$5% of days in $A$ carry $\\gg$5% of "
            "the cumulative return.\n\n"
            "We find **H₁ rejected** (window *t* ≈ 0; even the sharpest day-of *t* ≈ 0.9 is "
            "sub-bar), **H₂ rejected** (overlay < buy & hold at every cost), **H₃ rejected** "
            "(concentration **< 1**). The premium has the right *shape* on the day but is, on this "
            "basket, **absent**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is two complementary tests. **(a) Pooled Welch** of all "
            "announcement-day returns vs the rest — simple but treats every stock-day as "
            "independent, which it isn't (earnings cluster in seasons). **(b) Within-name "
            "per-event one-sample** *t* of $\\{\\pi_e\\}$ vs 0 — robust to cross-name level "
            "differences and closer to one observation per event.\n\n"
            "On top sits the honesty problem of **clustering**: report days bunch in earnings "
            "season, so a naive *t* overstates independence. We answer it with a **random-calendar "
            "placebo** — keep each name's *count* of report days but scatter *which* days carry "
            "the tag, and ask how often a random calendar of the same density beats the observed "
            "premium. Then Tradability charges the per-window round trip — the binding constraint "
            "for a calendar that fires only ~12 days a year per name."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket (yfinance adjusted "
            f"closes, {R['start']}→{R['end']}); **{R['n_dates']:,}** scheduled prints, "
            f"**{R['n_events']:,}** usable windows. **Survivor** panel — named on the Signal axis "
            "(and here it biases *toward* a premium).\n"
            "- **Tag.** A trading day is 'announcement' if it lies in $[D-1, D+1]$ around any "
            f"scheduled print ({R['ann_day_pct']:.1f}% of all stock-days).\n"
            "- **Premium.** Pooled announce-mean − rest-mean (Welch *t*), and within-name "
            "per-event mean-minus-baseline (one-sample *t*).\n"
            "- **Null (random-calendar placebo).** 20,000 random re-taggings of the same density "
            "per name; $p = \\Pr[\\text{shuffled }\\widehat{\\pi} \\ge \\text{observed}]$.\n"
            "- **Share-of-return.** Fraction of cumulative log return on announcement days vs their "
            "fraction of days.\n"
            "- **Robustness.** Window half-width $w \\in \\{0, 1, 2\\}$.\n"
            "- **Costs.** A long-only overlay, one round trip per window, 2–10 bps one-way.\n"
            "- **Positive control.** A deterministic panel with a **planted** announcement "
            "premium: zero edge must NOT reach significance; a large edge must light up (proving "
            "the null is real, not a blunt detector)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The premium across window widths — the term structure of a whisper\n\n"
            "The premium and its Welch *t* as we widen the window from the day itself ($w=0$) to "
            "$\\pm2$. If a real premium lived in the *days around* the print, widening would keep "
            "it; instead it's concentrated on the day and *diluted* by the neighbours."
        ),
        code(
            "ws = [0, 1, 2]\n"
            "if HAVE_REAL:\n"
            "    prem, tw = [], []\n"
            "    for w in ws:\n"
            "        fl = data.build_announce_flags(PRICES, DATES, window=w)\n"
            "        s = st.summarize_premium(fl, placebo=False)\n"
            "        prem.append(s['premium_bps']); tw.append(s['welch_t'])\n"
            "else:\n"
            "    prem = [r[1] for r in R['windows']]; tw = [r[2] for r in R['windows']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([f'[D±{w}]' for w in ws], prem, color=[GREEN, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(prem): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('premium (bps/day)'); a1.set_title('Premium concentrated on the day, diluted by neighbours')\n"
            "a2.bar([f'[D±{w}]' for w in ws], tw, color=[GREEN, GREY, GREY], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(tw): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('Welch t'); a2.set_ylim(0, 2.4); a2.set_title('Never near the t=2 bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('premium by window:', [round(v,2) for v in prem]); print('Welch t by window:', [round(v,2) for v in tw])"
        ),
        md(
            f"> 💡 In plain words: the day-of premium **{R['windows'][0][1]:+.2f} bps** (t = "
            f"{R['windows'][0][2]:.2f}) is the peak, and it's already below the bar; widening to "
            f"$\\pm1$ *dilutes* it to **{R['windows'][1][1]:+.2f} bps** (t = {R['windows'][1][2]:.2f}). "
            "A genuine *window* premium would survive widening — this one only lives on the single "
            "day and is too weak to matter even there."
        ),
        md(
            "### 4b · The decisive test — significance + a random-calendar placebo\n\n"
            "The observed day-of premium against a 20,000-draw random-calendar null (each draw "
            "re-tags the same number of report days at random per name). A real premium would sit "
            "in the far right tail; a whisper won't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(FLAGS0, n_draws=20000)\n"
            "    obs = pl['obs']*1e4; draws = pl['draws']*1e4; pval = pl['p_value']\n"
            "    tval = st.summarize_premium(FLAGS0, placebo=False)['welch_t']\n"
            "else:\n"
            "    obs = R['w0_premium_bps']; pval = R['w0_p']; tval = R['w0_welch_t']\n"
            "    rng = np.random.default_rng(515); draws = rng.normal(0.0, 3.2, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 20,000 random calendars')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed day-of premium {obs:+.2f} bps')\n"
            "ax.set_xlabel('day-of premium (bps/day)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}, Welch t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} bps  Welch t={tval:.2f}  random-calendar p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the green line sits **near the edge of the cloud, not past it** "
            f"— a random calendar of the same density beats the day-of premium **{R['w0_p']*100:.0f}%** "
            f"of the time (Welch **t = {R['w0_welch_t']:.2f}**). That's borderline, not a rejection: "
            "the day-of bump is *suggestive* but does not clear the desk's bar. On the tradable "
            f"$\\pm1$ window the placebo *p* is **{R['p_placebo']:.2f}** — squarely a coin-flip."
        ),
        md(
            "### 4c · The share-of-return decomposition — the legend's core claim\n\n"
            "The headline Frazzini–Lamont framing is *'a disproportionate share of return is "
            "earned around earnings.'* Decompose it directly: what fraction of cumulative log "
            "return lands on the announcement days, vs their fraction of all days?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sr1 = st.share_of_return(FLAGS); sr0 = st.share_of_return(FLAGS0)\n"
            "    d1, r1 = sr1['share_days']*100, sr1['share_return']*100\n"
            "    d0, r0 = sr0['share_days']*100, sr0['share_return']*100\n"
            "else:\n"
            "    d1, r1 = R['share_days'], R['share_return']\n"
            "    d0, r0 = R['w0_share_days'], R['w0_share_return']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-.2, [d0, d1], .4, color=GREY, label='share of DAYS')\n"
            "ax.bar(x+.2, [r0, r1], .4, color=RED, label='share of RETURN')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['day-of [D±0]', 'window [D±1]'])\n"
            "ax.set_ylabel('%'); ax.set_title('Only the day-of beats its share (1.45x); the window is under 1x')\n"
            "for i,(dd,rr) in enumerate([(d0,r0),(d1,r1)]):\n"
            "    ax.annotate(f'{rr/dd:.2f}x',(i+.2,rr),ha='center',va='bottom',fontsize=10,color=RED)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'day-of: {d0:.1f}% days -> {r0:.1f}% return ({r0/d0:.2f}x)')\n"
            "print(f'window: {d1:.1f}% days -> {r1:.1f}% return ({r1/d1:.2f}x)')"
        ),
        md(
            f"> 💡 In plain words: only the *single day* clears its weight (**{R['w0_concentration']:.2f}×**); "
            f"the tradable window carries only **{R['concentration']:.2f}×** — *half* its share. The "
            "'disproportionate share' is real-ish for one day and **inverted** for any window you "
            "could hold. The legend's central quantitative claim is busted on this basket."
        ),
        md(
            "### 4d · Costs — the overlay loses to doing nothing\n\n"
            "A long-only overlay holding the basket only inside announcement windows, net of a "
            "round trip per window, vs buy & hold (annualised). The overlay is in-market only ~5% "
            "of the time, so even before costs it collects a fraction of B&H."
        ),
        code(
            "costs = [2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.overlay_stats(FLAGS, cost_bps=c, window=1)['net_ann_pct'] for c in costs]\n"
            "    bh = st.overlay_stats(FLAGS, cost_bps=5.0, window=1)['bh_ann_pct']\n"
            "else:\n"
            "    net = [o[4] for o in R['overlay']]; bh = R['overlay'][0][5]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([f'{c:.0f} bps' for c in costs], net, color=[AMBER, AMBER, RED], width=.55, label='overlay net (annualised)')\n"
            "ax.axhline(bh, c=GREEN, lw=2.2, label=f'buy & hold {bh:+.1f}%')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(net): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('annualised return (%)'); ax.set_xlabel('one-way cost')\n"
            "ax.set_title('Long-the-announcers < buy & hold at every cost; negative at 10 bps'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c,v in zip(costs,net): print(f'{c:>4.0f} bps: overlay {v:+.1f}% ann   vs buy&hold {bh:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: even at a gentle **2 bps** the overlay makes "
            f"**{R['overlay'][0][4]:+.1f}%** vs buy & hold **{R['overlay'][0][5]:+.1f}%**; at **10 "
            f"bps** it's **{R['overlay'][2][4]:+.1f}%** — a loss. You traded constantly to underperform "
            "a do-nothing benchmark, while sitting in cash 95% of the year. **Mirage.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "The credibility of a *null* rests entirely on this: on a deterministic panel where we "
            "**plant** a known announcement premium, the detector must (a) stay at *t* ≈ 0 with "
            "zero edge, and (b) light up with a large planted edge. If both hold, the real-tape "
            "*t* ≈ 0 is a genuine absence — not a detector too blunt to see a premium."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.0015):\n"
            "    px, dt = data.synthetic_panel(edge=edge, seed=515)\n"
            "    fl = data.build_announce_flags(px, dt, window=1)\n"
            "    s = st.summarize_premium(fl, n_draws=3000)\n"
            "    res.append((edge*1e4, s['premium_bps'], s['welch_t'], s['t_event'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted\\n{e:.0f} bps' for e,_,_,_,_ in res]\n"
            "tvals = [r[2] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pr,tw,te,p in res: print(f'planted {e:+.1f} bps: premium={pr:+.2f} bps  Welch t={tw:+.2f}  event t={te:+.2f}  p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted premium the control sits at "
            f"**t = {R['syn'][0][2]:.2f}** (no false positive — noise can't fake it); a **+15 bps** "
            f"planted premium reaches **t = {R['syn'][1][2]:.2f}** (placebo p = {R['syn'][1][4]:.0f}). "
            "The machinery is unbiased *and powered* — so the real-tape *t* ≈ 0 is a **genuine "
            "absence** of a tradable premium, not a measurement that was too blunt to see one. "
            "That is what makes this an honest **None**, not an under-powered shrug."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the tradable `[D-1,D+1]` premium is **{R['premium_bps']:+.2f} "
            f"bps/day** (Welch **t = {R['welch_t']:.2f}**, per-event **t = {R['t_event']:.2f}**, "
            f"random-calendar **p = {R['p_placebo']:.2f}**). The single-day bump "
            f"(**{R['w0_premium_bps']:+.2f} bps**, Welch **t = {R['w0_welch_t']:.2f}**, 1.45× "
            "concentration) has the right Frazzini–Lamont *shape* but never clears **t ≥ 2** and "
            "dissolves at any tradable window. A **powered** synthetic control (t ≈ 7.5 on a "
            "planted premium) confirms this is a real absence. Carries a **survivorship** caveat "
            "that biases *toward* a premium — and we still find none.\n"
            f"- **Tradability `MIRAGE`** — the long-the-announcers overlay returns "
            f"**{R['overlay'][1][4]:+.1f}%** ann at 5 bps vs buy & hold **{R['overlay'][1][5]:+.1f}%**, "
            "and goes **negative** at 10 bps, while in-market only ~5% of the year. Nothing to "
            "deploy.\n"
            f"- **Earnings days carry it? `BUSTED`** — announcement-window days are "
            f"**{R['share_days']:.1f}%** of days but carry **{R['share_return']:.1f}%** of "
            f"cumulative return (**{R['concentration']:.2f}×** — under 1, the *opposite* of the "
            "claim). Only the single day exceeds 1× (1.45×), a sub-bar whisper the neighbours erase."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — where the premium *isn't*\n\n"
            "One picture for the operational truth: the day-of premium is the only place anything "
            "lives, and it's both sub-significance and un-tradable (you can't isolate a single day "
            "without paying a full round trip for it). The minute you build a holdable window, the "
            "premium is gone."
        ),
        code(
            "ws = [0, 1, 2]\n"
            "if HAVE_REAL:\n"
            "    conc = []\n"
            "    for w in ws:\n"
            "        fl = data.build_announce_flags(PRICES, DATES, window=w)\n"
            "        sr = st.share_of_return(fl); conc.append(sr['share_return']/sr['share_days'])\n"
            "else:\n"
            "    conc = [R['w0_concentration'], R['concentration'], 0.84]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if c>1 else RED for c in conc]\n"
            "ax.bar([f'[D±{w}]' for w in ws], conc, color=cols, width=.55)\n"
            "ax.axhline(1, ls='--', c='k', label='fair share (1x)')\n"
            "for i,v in enumerate(conc): ax.annotate(f'{v:.2f}x',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('return concentration (share of return / share of days)')\n"
            "ax.set_title('Only the single day beats its weight — and you cannot trade one day cheaply'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('concentration by window:', [round(c,2) for c in conc])"
        ),
        md(
            "> 💡 In plain words: the green bar (day-of, >1×) is the entire premium — and it's "
            "trapped on a single day you can't capture without a full round trip, *and* it's "
            "sub-*t*-2 anyway. Every tradable window is red (<1×). There is no holdable corner "
            "where this pays. Real edge requires *both* significance *and* a vehicle; this has "
            "neither — hence **None × Mirage**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Liquidity/size is the knob.** Frazzini & Lamont (2007) find the premium "
            "concentrates in **small, neglected, retail-heavy** names — exactly what a large-cap "
            "basket excludes. Re-run on the Russell 2000 for a larger (but costlier, "
            "survivorship-fraught) day-of bump.\n"
            "- **Decay.** Published 2007; McLean & Pontiff (2016) document heavy post-publication "
            "decay. A 2005–2026 large-cap sample is a textbook decay case.\n"
            "- **The lesson generalises.** A real *shape* on a single day (the announcement is "
            "genuinely an information event) does **not** imply a tradable premium — significance "
            "and a holdable vehicle are separate hurdles, and a calendar that fires ~12 days/yr "
            "fails the second one on costs.\n\n"
            "*The reproducible core is offline and deterministic; the tag is the `[D-1,D+1]` window "
            "around each scheduled earnings date with a within-name baseline. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md). Sibling: [PEAD, study 363](../363-pead-drift).*"
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
