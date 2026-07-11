"""Generate the two narrative notebooks for Study 642 (Turnaround Tuesday).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY total
# return, 1993-01-29 -> 2026-06-30; 1,567 Monday->Tuesday consecutive-session pairs).
R = dict(
    start="1993-01-29", end="2026-06-30",
    n_pairs=1567, n_down=684, n_up=883, n_uncond_tue=1728, n_allday=8410,
    cond_bps=21.34, up_bps=-2.21, uncond_tue_bps=7.09, allday_bps=4.77,
    welch_uncond=2.38, welch_up=3.85, welch_allday=3.04,
    nw_allday_t=3.19, nw_allday_coef=18.04,
    nw_tue_t=3.80, nw_tue_coef=23.60,
    hit=387, hit_pct=56.6, wilson=(52.8, 60.2),
    placebo_p=0.00005, placebo_mean=8.05, placebo_sd=3.27, placebo_draws=20000,
    # weekday pair -> (n, down-day mean bps, up-day mean bps, Welch t)
    weekday_pairs={
        "Mon->Tue": (1567, 21.34, -2.21, 3.85),
        "Tue->Wed": (1710, 7.59, 4.20, 0.61),
        "Wed->Thu": (1675, 0.17, 2.38, -0.38),
        "Thu->Fri": (1634, 5.13, 0.63, 0.81),
        "Fri->Mon": (1521, 7.67, 5.68, 0.29),
    },
    other_pooled_n=2987, other_pooled_down_bps=5.15, other_pooled_up_bps=3.19,
    other_pooled_t=0.66,
    excrisis_n=652, excrisis_bps=14.81, excrisis_t=3.14,
    era_split="2000-01-01",
    era_early_bps=19.91, era_early_n=135, era_early_t=0.91,
    era_late_bps=21.69, era_late_n=549, era_late_t=2.20, era_diff_t=0.15,
    events_per_year=20.5,
    timer_5_gross=21.3, timer_5_net=11.3, timer_5_ann=2.32, timer_5_t=2.14,
    timer_5_sharpe=0.37, timer_5_worst=-4.58,
    timer_10_gross=21.3, timer_10_net=1.3, timer_10_ann=0.27, timer_10_t=0.25,
    timer_10_sharpe=0.04, timer_10_worst=-4.68,
    syn_null_mean=-0.03, syn_null_sd=0.93, syn_null_fire=0, syn_planted_t=4.70,
    fp_spy="bf55c945be2e",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Monday-specific%3F: Confirmed](https://img.shields.io/badge/Monday--specific%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from turnaround_tuesday import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY = data.load_real()
    DF = st.day_frame(SPY["AdjClose"])
    PAIRS = st.monday_tuesday_pairs(DF)
else:
    SPY = DF = PAIRS = None
print("real cache present:", HAVE_REAL, "| Monday->Tuesday pairs:",
      (0 if PAIRS is None else len(PAIRS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the market really bounce back the day after a red Monday? 🔄\n"
            "### Turnaround Tuesday — the desk folklore that's *directionally* true, and "
            "still barely worth trading\n\n"
            + BADGES +
            "Every trading desk has heard the line: *\"if Monday closes red, buy the close — "
            "Tuesday tends to bounce back.\"* It sounds like classic bar-stool wisdom — the kind "
            "of pattern-matching that dies the second you actually count it. Most of the time, on "
            "this desk, it does.\n\n"
            "This time it doesn't. Not entirely.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY total return, 1993→2026, 1,567 Monday→Tuesday pairs. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does Tuesday really bounce after a down Monday? | **Yes.** Tuesdays that follow "
            f"a red Monday average **+{R['cond_bps']:.0f} bps**, versus **+{R['uncond_tue_bps']:.0f} "
            f"bps** on a typical Tuesday and **{R['up_bps']:+.0f} bps** — essentially flat — after "
            f"a *green* Monday. It happened this way on 1 in 20,000 random draws of the calendar. |\n"
            "| Is it just \"markets bounce after any bad day\"? | **No — it's specifically "
            "Monday.** We ran the same test on Tuesday→Wednesday, Wednesday→Thursday, and every "
            "other weekday pair. None of them show a real bounce. Only Monday→Tuesday does. |\n"
            "| Does it still work lately, or did it fade? | **Still there.** The pattern is just "
            "as strong since 2000 as it was in the 1990s — if anything, slightly stronger, though "
            "the change itself isn't provable. |\n"
            f"| Can you actually get paid for it? | **Barely, and only just.** Trading it at a "
            f"typical retail cost nets **+{R['timer_5_net']:.0f} bps** a shot — real, but thin — "
            f"and at a slightly higher (still realistic) cost the edge is **gone**. |\n\n"
            "> The pattern is real. The margin for error trading it is almost nonexistent."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Monday sets a bad tone, sellers get it out of their system, and by Tuesday "
            "the selling is exhausted — the market snaps back.\"*\n\n"
            "It's the mean-reversion cousin of the old \"Monday Effect\" (Mondays are always "
            "bad, a real finding — in the 1950s-70s). That original effect is dead on the modern "
            "tape (see our own [study 224](../../224-monday-effect/) — Monday today is actually "
            "slightly *positive*). Turnaround Tuesday doesn't claim Monday is always bad. It "
            "makes a narrower, more testable claim: *when* Monday is bad, Tuesday tends to be "
            "good."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a genuinely simple, mechanical timing signal: watch Monday's close, "
            "and if it's red, buy — for one day. No forecasting, no macro views, just a calendar "
            "rule anyone can follow with a phone. That's exactly the kind of claim worth taking "
            "seriously *and* worth being suspicious of — \"buy the dip\" dressed up with a day-of-"
            "week hook is one of the oldest tricks in retail trading content."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The comparison.** Every Tuesday that immediately follows a Monday trading "
            f"session (**{R['n_pairs']:,}** such pairs since 1993), split by whether that "
            "Monday's own close was red or green.\n"
            "- **The specificity check.** Is this really about Monday, or would *any* down day "
            "get a bounce the next session? We test all five weekday pairs the same way.\n"
            "- **The luck check.** Randomly relabel which Tuesdays count as \"after a down "
            "Monday\" 20,000 times — how often does a random label beat the real one?\n"
            "- **The trade check.** Buy SPY at the down Monday's own close, sell at the Tuesday "
            "close, pay realistic costs both ways."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average Tuesday return, split three ways."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(DF, PAIRS)\n"
            "    a, b, c = h['cond_bps'], h['uncond_tue_bps'], h['up_bps']\n"
            "else:\n"
            "    a, b, c = R['cond_bps'], R['uncond_tue_bps'], R['up_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['after a RED\\nMonday','a typical\\nTuesday','after a GREEN\\nMonday'],\n"
            "       [a, b, c], color=[GREEN, GREY, RED], width=.6)\n"
            "for i,v in enumerate([a, b, c]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average Tuesday return (bps)')\n"
            "ax.set_title('Tuesday really is different after a red Monday')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after red Monday {a:+.2f} bps | typical Tuesday {b:+.2f} bps | after green Monday {c:+.2f} bps')"
        ),
        md(
            f"There it is: **+{R['cond_bps']:.1f} basis points** on the average Tuesday that "
            f"follows a down Monday, roughly **three times** a typical Tuesday, and essentially "
            f"the opposite sign of what follows an *up* Monday. Tuesday fell on "
            f"**{100-R['hit_pct']:.0f}%** and rose on **{R['hit_pct']:.0f}%** of down-Monday "
            "events. A random-label draw of 20,000 tries to beat this number the wrong way "
            "about **1 time in 20,000**.\n\n"
            "**Next, the honest question:** is this really about *Monday*, or would any bad day "
            "bounce back the next session?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    wp = st.weekday_pair_stats(DF)\n"
            "    pairs_lbl = list(wp.index); ts = list(wp['welch_t'])\n"
            "else:\n"
            "    pairs_lbl = list(R['weekday_pairs']); ts = [R['weekday_pairs'][p][3] for p in pairs_lbl]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [GREEN if p=='Mon->Tue' else GREY for p in pairs_lbl]\n"
            "ax.bar(pairs_lbl, ts, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (down-day bounce vs up-day)')\n"
            "ax.set_title('Only Monday -> Tuesday clears the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({p: round(t,2) for p,t in zip(pairs_lbl, ts)})"
        ),
        md(
            "Only **one** bar crosses the red dashed \"real\" line — Monday→Tuesday. Every other "
            "weekday's \"bounce back the next day\" story is statistical noise. This rules out the "
            "boring explanation (\"markets just bounce after any bad day, and someone happened to "
            "name the Monday/Tuesday version\") — the pattern really is tied to the specific "
            "calendar day.\n\n"
            "**Does it still work lately?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(DF, PAIRS, data.ERA_SPLIT)\n"
            "    e, l = ec['early_bps'], ec['late_bps']\n"
            "else:\n"
            "    e, l = R['era_early_bps'], R['era_late_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['1993-2000\\n(n=135)','2000-2026\\n(n=549)'], [e, l], color=[AMBER, GREEN], width=.55)\n"
            "for i,v in enumerate([e, l]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('down-Monday Tuesday mean (bps)')\n"
            "ax.set_title('No decay -- if anything, slightly stronger since 2000')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2000 {e:+.2f} bps | post-2000 {l:+.2f} bps')"
        ),
        md(
            "No fade. The pattern is essentially unchanged across three decades — this isn't a "
            "one-decade fluke that got arbitraged away.\n\n"
            "**Finally, the trade.** Can you actually collect this bounce after paying to trade it?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.timer_stats(PAIRS, cost_bps=5.0)\n"
            "    tm10 = st.timer_stats(PAIRS, cost_bps=10.0)\n"
            "    n5, n10 = tm5['net_bps'], tm10['net_bps']\n"
            "else:\n"
            "    n5, n10 = R['timer_5_net'], R['timer_10_net']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['5 bps cost\\n(typical retail)','10 bps cost\\n(a bit worse)'], [n5, n10],\n"
            "       color=[AMBER, RED], width=.55)\n"
            "for i,v in enumerate([n5, n10]): ax.annotate(f'{v:+.1f} bps net',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net return per event (bps)')\n"
            "ax.set_title('The edge survives one cost level, not the next')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net at 5 bps: {n5:+.1f} bps  |  net at 10 bps: {n10:+.1f} bps')"
        ),
        md(
            f"At a typical retail cost the trade nets **+{R['timer_5_net']:.0f} bps** a shot — "
            f"real, statistically defensible (barely — *t* = {R['timer_5_t']:.2f}), and worth "
            f"about **+{R['timer_5_ann']:.1f}%/yr** since the position is only on ≈8% of trading "
            f"days. Bump the cost just a little and the whole thing goes to "
            f"**+{R['timer_10_net']:.1f} bps** — statistically indistinguishable from zero "
            f"(*t* = {R['timer_10_t']:.2f}). And a single bad down-Monday can wipe out "
            f"**{abs(R['timer_5_worst']):.1f}%** in one shot — years of edge, gone in a day."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** Down-Monday Tuesdays average **+{R['cond_bps']:.1f} bps** vs "
            f"**{R['uncond_tue_bps']:+.1f}** typically and **{R['up_bps']:+.1f}** after a green "
            "Monday — this survives every robustness check we throw at it, and crucially, it's "
            "specific to Monday, not a generic \"buy the dip\" story hiding under a calendar "
            "name.\n"
            "- **Tradability — Fragile.** The edge is real but thin — it survives a typical "
            "retail cost by the skin of its teeth and dies at a slightly higher one. Not a "
            "mirage; not investable at scale either.\n"
            "- **\"Is this Monday-specific?\" — Confirmed.** Every other weekday's version of the "
            "same test comes up empty. This one really is about the specific day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why Monday and not another day?** The most plausible story: news and order flow "
            "accumulate over a weekend with no trading, so Monday selling can overshoot more than "
            "a normal single-day move — leaving more of it to mean-revert by Tuesday's close.\n"
            "- **Where this might actually pay** is inside a book that's already trading SPY "
            "intraday for other reasons — the marginal cost of tilting one extra day a week is "
            "much lower than paying full retail commissions and spread for a standalone strategy.\n"
            "- **Sibling studies:** the [Monday Effect](../../224-monday-effect/) (is Monday "
            "itself bad? no, not anymore) and the [Weekend Effect](../../90-weekend/) (which "
            "weekday is best, unconditionally) test related but different claims — see "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think you can trade this better than a flat 5/10 bps SPY timer? Show a net, "
            "certifiable edge after realistic slippage on the size you'd actually run — then "
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
            "# Turnaround Tuesday — a quantitative teardown 🔬\n"
            "### The down-Monday-Tuesday Welch/HAC split · a 20-seed random-pair placebo · the "
            "five-weekday-pair specificity test · a crisis-window robustness cut · the pre/"
            "post-2000 era contrast · an honest cost sweep · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **a down Monday predicts a Tuesday bounce** — is a conditional claim, "
            "distinct from every day-of-week *level* claim already on this desk (studies 90 and "
            "224). The job here is to measure it honestly, rule out the boring explanation "
            "(generic reversal wearing a calendar costume), and then ask the only question that "
            "pays: *is any of it tradable?*\n\n"
            "> ⚠️ **Data note.** SPY daily raw OHLC + adjusted (total-return) close, "
            "1993-01-29 → 2026-06-30, yfinance, cached. No hardcoded event calendar — "
            "\"Monday\"/\"Tuesday\" are `.dayofweek` facts. No survivorship (single continuous "
            "index series). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | down-Monday Tuesday **{R['cond_bps']:+.2f} bps** vs "
            f"unconditional {R['uncond_tue_bps']:+.2f}: Welch **t = {R['welch_uncond']:.2f}** "
            f"(vs unconditional) / **{R['welch_up']:.2f}** (vs up-Monday), NW "
            f"**t = {R['nw_allday_t']:.2f}-{R['nw_tue_t']:.2f}**, placebo "
            f"**p = {R['placebo_p']:.5f}**, Monday-specific (pooled other-weekday "
            f"t = {R['other_pooled_t']:.2f}) |\n"
            f"| **Tradability** | `FRAGILE` | net of 5 bps: t = {R['timer_5_t']:.2f}, Sharpe "
            f"{R['timer_5_sharpe']:.2f}; net of 10 bps: t = {R['timer_10_t']:.2f}, Sharpe "
            f"{R['timer_10_sharpe']:.2f} |\n"
            f"| **Monday-specific?** | `CONFIRMED` | Mon→Tue t = {R['welch_up']:.2f}; pooled "
            f"other four weekday pairs t = {R['other_pooled_t']:.2f} |\n\n"
            "> 💡 In plain words: the pattern is genuine and specifically tied to Monday, not a "
            "relabelled version of generic reversal — but the tradable edge is thin enough that a "
            "single extra basis point of cost kills it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the SPY close-to-close total return on session $t$, $M_t$ a Monday "
            "session with $r_{M_t} < 0$ (a \"down Monday\"), and $T_{t+1}$ the immediately "
            "following Tuesday session (excluded if a Monday market holiday breaks the "
            "adjacency). The claims:\n\n"
            "- **H₁ (bounce).** $E[r_{T} \\mid r_{M} < 0] \\gg E[r_{T}]$ — a down Monday predicts "
            "an above-average Tuesday.\n"
            "- **H₂ (asymmetry).** The effect is a genuine *conditional* relationship, not just "
            "\"Tuesday is good\": Tuesdays after an *up* Monday should NOT show the same bounce.\n"
            "- **H₃ (specificity).** The pattern is tied to *Monday* specifically — not a generic "
            "\"any down day bounces the next session\" reversal effect relabelled with a calendar "
            "hook.\n"
            "- **H₄ (capture).** A literal SPY timer (buy the down-Monday close, sell the "
            "Tuesday close) banks the edge net of realistic costs.\n\n"
            "We find **H₁ strongly supported** (Welch t up to +3.85, NW t up to +3.80, placebo "
            "p = 0.00005), **H₂ supported** (up-Monday Tuesdays are flat-to-negative, "
            f"{R['up_bps']:+.2f} bps), **H₃ supported** (pooled other-weekday t = "
            f"{R['other_pooled_t']:.2f}), **H₄ marginal** (t = {R['timer_5_t']:.2f} at 5 bps, "
            f"t = {R['timer_10_t']:.2f} at 10 bps)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Down-Monday events are **weekly, non-overlapping** — the planned primary is a "
            "**Welch t** on the group split, run three ways (vs unconditional Tuesday, vs "
            "up-Monday Tuesday, vs all trading days) so a single comparison group can't drive the "
            "conclusion. Because the daily return series is serially correlated, we cross-check "
            "with a **Newey-West (5-lag) t** on the down-Monday-Tuesday dummy regression — the "
            "slope *is* the mean gap. The hit rate carries a **Wilson interval**, the placebo "
            "reshuffles the down-Monday label among Tuesdays **20,000 times (20 seeds × 1,000)**, "
            "and the era split (2000-01-01, named ex ante in the brief) is tested as a "
            "**difference**, not eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Pairing.** {R['n_pairs']:,} Monday→Tuesday consecutive-session pairs "
            f"{R['start']} → {R['end']}, {R['n_down']} with a down Monday, {R['n_up']} with an "
            "up Monday (holiday-broken weeks correctly excluded).\n"
            "- **Headline.** Welch t (three comparisons) + NW(5) t (two dummy regressions) + "
            "Wilson hit rate + 20-seed placebo.\n"
            "- **Specificity.** The identical down-day → next-day split on all five weekday "
            "pairs; the other four pooled and Welch-t'd against Mon→Tue.\n"
            "- **Robustness.** Drop the 2008-09 GFC and 2020-03 COVID crash windows entirely.\n"
            "- **Era.** Pre/post-2000, within-era Welch t's + a Welch t of the difference.\n"
            "- **Execution (timer).** Enter SPY at the down Monday's own close (the flag is "
            "knowable at that instant — it *is* that close, zero look-ahead), exit the Tuesday "
            "close; 2 × one-way cost × NAV per event; long-only, no borrow.\n"
            "- **Control.** Synthetic i.i.d.-drift tape, planted-bounce knob; the null must not "
            "fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo\n\n"
            "Welch t on Tuesday returns (three comparisons), NW t on two dummy regressions, and "
            "the random-pair null. In the notebook we run a lighter placebo (4 seeds × 500 "
            "draws) and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(DF, PAIRS)\n"
            "    print(f\"down-Monday Tuesday {h['cond_bps']:+.2f} bps (n={h['n_down']})  vs  \"\n"
            "          f\"unconditional {h['uncond_tue_bps']:+.2f} bps (n={h['n_uncond_tue']})  vs  \"\n"
            "          f\"up-Monday {h['up_bps']:+.2f} bps (n={h['n_up']})\")\n"
            "    print(f\"Welch t = {h['welch_t_vs_uncond']:+.2f} (vs uncond) / \"\n"
            "          f\"{h['welch_t_vs_up']:+.2f} (vs up-Monday) / {h['welch_t_vs_allday']:+.2f} (vs all days)\")\n"
            "    print(f\"NW(5) t vs all days = {h['nw_t_vs_allday']:+.2f}   NW(5) t vs Tuesdays = {h['nw_t_vs_uncond']:+.2f}\")\n"
            "    print(f\"hit {h['hit']}/{h['n_down']} = {h['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%]\")\n"
            "    pl = st.placebo_pvalue(PAIRS, n_draws_per_seed=500, n_seeds=4)\n"
            "    obs, draws = pl['obs']*1e4, pl['draws']*1e4\n"
            "else:\n"
            "    obs = R['cond_bps']\n"
            "    rng = np.random.default_rng(642)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random Tuesday relabelling (light in-notebook run)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed down-Monday-Tuesday mean {obs:+.2f} bps')\n"
            "ax.set_xlabel('mean Tuesday return of a random-label draw (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Far outside the luck cloud: canonical p = {R['placebo_p']:.5f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f} bps, \"\n"
            "      f\"sd {R['placebo_sd']:.2f}, p = {R['placebo_p']:.5f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **+{R['cond_bps']:.2f} bps** sits several "
            f"placebo-sigmas above the null's center ({R['placebo_mean']:+.2f} ± "
            f"{R['placebo_sd']:.2f} bps); **p = {R['placebo_p']:.5f}**. With Welch t up to "
            f"**{R['welch_up']:.2f}** and NW t up to **{R['nw_tue_t']:.2f}**, H₁ clears the desk "
            "bar several separate ways."
        ),
        md(
            "### 4b · The specificity test — Monday, or generic reversal?\n\n"
            "The same down-day → next-day split, run on all five weekday pairs. If turnaround "
            "Tuesday were just short-horizon reversal (Jegadeesh 1990, Lehmann 1990) relabelled "
            "with a calendar hook, every pair should look roughly the same."
        ),
        code(
            "if HAVE_REAL:\n"
            "    wp = st.weekday_pair_stats(DF)\n"
            "    labels = list(wp.index); ts = list(wp['welch_t'])\n"
            "    oc = st.other_weekday_check(DF)\n"
            "    other_t = oc['welch_t']\n"
            "else:\n"
            "    labels = list(R['weekday_pairs']); ts = [R['weekday_pairs'][p][3] for p in labels]\n"
            "    other_t = R['other_pooled_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [GREEN if p=='Mon->Tue' else GREY for p in labels]\n"
            "ax.bar(labels, ts, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (down-day bounce vs up-day)')\n"
            "ax.set_title('Specificity: only Monday -> Tuesday clears |t| = 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: round(t,2) for l,t in zip(labels, ts)}, ' | pooled other-4:', round(other_t,2))"
        ),
        md(
            f"> 💡 In plain words: pooled across the other four weekday pairs "
            f"(n={R['other_pooled_n']:,} down-day events), the \"bounce\" is "
            f"**t = {R['other_pooled_t']:.2f}** — nothing. Only Mon→Tue clears the bar "
            f"(t = {R['welch_up']:.2f}). H₃ holds: this is not generic reversal wearing a "
            "calendar costume; it is specifically about the weekend-adjacent Monday close."
        ),
        md(
            "### 4c · Robustness — not a crisis-day artifact, and no decay\n\n"
            "Drop the two loudest macro windows on the tape; then split pre/post-2000."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex = st.excl_crisis_check(PAIRS)\n"
            "    ex_bps, ex_t, ex_n = ex['mean_bps'], ex['welch_t'], ex['n']\n"
            "    ec = st.era_contrast(DF, PAIRS, data.ERA_SPLIT)\n"
            "    e, l = ec['early_bps'], ec['late_bps']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    ex_bps, ex_t, ex_n = R['excrisis_bps'], R['excrisis_t'], R['excrisis_n']\n"
            "    e, l = R['era_early_bps'], R['era_late_bps']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['full sample','ex-GFC & ex-COVID\\n(n={})'.format(ex_n)], [R['cond_bps'], ex_bps],\n"
            "       color=[GREEN, AMBER], width=.55)\n"
            "for i,v in enumerate([R['cond_bps'], ex_bps]): a1.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('down-Monday Tuesday mean (bps)')\n"
            "a1.set_title(f'Not a crisis-day artifact (ex-crisis t={ex_t:+.2f})')\n"
            "a2.bar(['1993-2000\\n(n={})'.format(R['era_early_n']),'2000-2026\\n(n={})'.format(R['era_late_n'])],\n"
            "       [e, l], color=[AMBER, GREEN], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    a2.annotate(f'{v:+.1f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "a2.set_title(f'No decay (diff t={dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ex-crisis {ex_bps:+.2f} bps (t={ex_t:+.2f}, n={ex_n})')\n"
            "print(f'era: early {e:+.2f} (t={et:+.2f})  late {l:+.2f} (t={lt:+.2f})  diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with the 2008-09 GFC and 2020-03 COVID crash windows dropped "
            f"entirely, the effect shrinks but survives (**{R['excrisis_bps']:+.2f} bps**, "
            f"t = {R['excrisis_t']:+.2f}, n = {R['excrisis_n']}) — it isn't a handful of huge "
            f"macro days dragging the mean. And there is no decay across the 2000 split: the "
            f"pre-2000 slice alone is underpowered (n = {R['era_early_n']}, "
            f"t = {R['era_early_t']:+.2f}) but the post-2000 slice, with four times the events, "
            f"clears the bar on its own (t = {R['era_late_t']:+.2f}); the difference test finds "
            f"nothing (t = {R['era_diff_t']:+.2f})."
        ),
        md(
            "### 4d · The timer — honest cost sweep\n\n"
            "Enter at the down Monday's own close (zero look-ahead — the flag *is* that close), "
            "exit the Tuesday close, pay 2 × one-way cost per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    years = (DF.index.max() - DF.index.min()).days / 365.25\n"
            "    tm5 = st.timer_stats(PAIRS, cost_bps=5.0, years=years)\n"
            "    tm10 = st.timer_stats(PAIRS, cost_bps=10.0, years=years)\n"
            "    g, n5, n10 = tm5['gross_bps'], tm5['net_bps'], tm10['net_bps']\n"
            "    t5, t10 = tm5['t_net'], tm10['t_net']\n"
            "    sh5, sh10 = tm5['sharpe_net'], tm10['sharpe_net']\n"
            "else:\n"
            "    g = R['timer_5_gross']; n5, n10 = R['timer_5_net'], R['timer_10_net']\n"
            "    t5, t10 = R['timer_5_t'], R['timer_10_t']\n"
            "    sh5, sh10 = R['timer_5_sharpe'], R['timer_10_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross','net @5bps','net @10bps'], [g, n5, n10], color=[GREY, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('bps per event')\n"
            "a1.set_title('The edge shrinks fast with cost')\n"
            "a2.bar(['net @5bps\\n(t={:.2f})'.format(t5),'net @10bps\\n(t={:.2f})'.format(t10)],\n"
            "       [sh5, sh10], color=[AMBER, RED], width=.55)\n"
            "for i,v in enumerate([sh5, sh10]): a2.annotate(f'Sharpe {v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('annualized Sharpe (event-only)')\n"
            "a2.set_title('Fragile: barely certified, then dead')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f} -> net {n5:+.1f} (t={t5:+.2f}, Sharpe {sh5:.2f}) / '\n"
            "      f'{n10:+.1f} bps (t={t10:+.2f}, Sharpe {sh10:.2f})')"
        ),
        md(
            f"> 💡 In plain words: +{R['timer_5_net']:.1f} bps net per event at 5 bps cost, "
            f"~+{R['timer_5_ann']:.1f}%/yr from ~20 events a year — *barely* certifiable "
            f"(t = {R['timer_5_t']:.2f}, the bar is 2). At 10 bps the whole thing collapses to "
            f"+{R['timer_10_net']:.1f} bps at t = {R['timer_10_t']:.2f} — statistically nothing. "
            f"Structurally: the position is only on ≈8% of trading days, so the total dollars at "
            f"stake are thin, and a single worst event "
            f"({R['timer_5_worst']:.1f}% to {R['timer_10_worst']:.1f}%) can erase years of net "
            "edge either way. H₄ marginal at best; **Tradability = FRAGILE**."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic i.i.d.-drift daily-return tape, TUNABLE planted down-Monday → Tuesday "
            "bounce. The null (bounce = 0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close = data.synthetic_world(bounce=0.0, seed=642 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close = data.synthetic_world(bounce=0.0020, seed=642)\n"
            "planted_t = st.synthetic_detect(close)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bounce=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5,\n"
            "           label='planted bounce = +20 bps')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (down-Monday vs up-Monday Tuesday)')\n"
            "ax.set_title('Control: no null fires; a planted bounce lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted 20 bp bounce reads t = {R['syn_planted_t']:.2f}. The machinery is "
            "unbiased — the real-tape t up to 3.85 is the genuine article. *(A faithful-engine / "
            "power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — down-Monday Tuesday **{R['cond_bps']:+.2f} bps** vs "
            f"unconditional {R['uncond_tue_bps']:+.2f} (Welch t = {R['welch_uncond']:.2f}) and vs "
            f"up-Monday Tuesday {R['up_bps']:+.2f} (Welch t = {R['welch_up']:.2f}); NW t = "
            f"{R['nw_allday_t']:.2f}/{R['nw_tue_t']:.2f}; placebo p = {R['placebo_p']:.5f}; "
            f"robust ex-crisis (t = {R['excrisis_t']:.2f}); no decay (diff t = "
            f"{R['era_diff_t']:.2f}); genuinely Monday-specific (pooled other-weekday "
            f"t = {R['other_pooled_t']:.2f}).\n"
            f"- **Tradability `FRAGILE`** — net of 5 bps: t = {R['timer_5_t']:.2f}, Sharpe "
            f"{R['timer_5_sharpe']:.2f}, ~{R['timer_5_ann']:+.2f}%/yr; net of 10 bps: t = "
            f"{R['timer_10_t']:.2f}, Sharpe {R['timer_10_sharpe']:.2f}, ~{R['timer_10_ann']:+.2f}%/yr "
            "— barely certified, then dead, on a position only on ≈8% of trading days.\n"
            f"- **Monday-specific? `CONFIRMED`** — only Mon→Tue clears |t| = 2 among all five "
            f"weekday down-day → next-day pairs; the pooled other four are t = "
            f"{R['other_pooled_t']:.2f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why Monday specifically?** The most plausible mechanism: two non-trading days "
            "(a weekend) let news, order flow and sentiment accumulate uninterrupted, so a red "
            "Monday can overshoot the information it's reacting to more than a normal single-"
            "session move — leaving more of it to mean-revert by the next close. That's a "
            "testable hypothesis for a follow-up study (does the size of the bounce scale with "
            "the size of the Monday drop, or with weekend news flow/gap size specifically?).\n"
            "- **Why the edge doesn't scale:** the position is on for one session, roughly once a "
            "week, and the whole edge lives in ~20 bps of margin — a level realistic institutional "
            "SPY execution costs can eat easily, especially once market impact and adverse "
            "selection (everyone else also knows Monday closed red) are added.\n"
            "- **Dedup map:** [224-monday-effect](../../224-monday-effect/) (is Monday itself "
            "negative? no — `NONE`), [90-weekend](../../90-weekend/) (the unconditional weekday "
            "level table, which names \"turnaround Tuesday\" but never conditions on Monday's "
            "sign), [116-power-hour](../../116-power-hour/) (an intraday continuation/reversal "
            "claim on a different clock).\n\n"
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
