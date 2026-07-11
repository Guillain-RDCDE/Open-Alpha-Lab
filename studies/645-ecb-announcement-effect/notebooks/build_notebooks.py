"""Generate the two narrative notebooks for Study 645 (ECB Announcement Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FEZ/EURUSD tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance FEZ/EURUSD=X
# 2005-01-03 -> 2026-06-30; 208 hardcoded scheduled ECB decisions, 207 on the FEZ tape).
R = dict(
    start="2005-01-03", end="2026-06-30", n_ecb=207, n_rest=5198, n_calendar=208,
    cal_lo="2005-01-13", cal_hi="2026-06-11",
    ecb_pct=-0.103, rest_pct=+0.015, gap_pct=-0.118,
    welch_t=-0.82, nw_t=-0.83,
    hit=114, hit_pct=55.1, wilson=(48.3, 61.7),
    placebo_p=0.345, placebo_mean=+0.0141, placebo_sd=0.1093, placebo_draws=20000,
    # realized range
    rng_ecb=1.492, rng_rest=1.257, rng_ratio=1.19, rng_welch_t=+2.66, rng_nw_t=+2.83,
    rng_placebo_p=0.0008,
    # EURUSD
    fx_ecb=0.455, fx_rest=0.436, fx_t=+0.65,
    # event window: offset -> (mean pct, welch t)
    event={-5: (+0.105, +0.85), -4: (-0.099, -1.10), -3: (-0.203, -1.58),
           -2: (+0.171, +1.23), -1: (+0.146, +1.11), 0: (-0.103, -0.84),
           1: (+0.017, -0.01), 2: (-0.112, -0.91), 3: (+0.032, +0.12)},
    runup_pct=+0.121, runup_t=+0.54,
    # era contrast — return
    era_early=-0.137, era_early_n=115, era_early_t=-0.66,
    era_late=-0.061, era_late_n=92, era_late_t=-0.46, era_diff_t=+0.27,
    # era contrast — range
    rng_era_early=1.722, rng_era_early_t=+1.70,
    rng_era_late=1.206, rng_era_late_t=+1.65, rng_era_diff_t=-3.18,
    # third axis — timer
    tm_gross=-10.3, tm_net5=-20.3, tm_net10=-30.3, tm_net20=-50.3, tm_rest=+1.5, tm_t=-0.82,
    tm_ann5=-1.62, tm_ann10=-2.42, tm_ann20=-4.02,
    tm_n=207, tm_hit=55.1, tm_worst=-13.3,
    # synthetic control
    syn_null_mean=-0.07, syn_null_sd=0.73, syn_null_fire=0, syn_planted_t=+7.48,
    fp_fez="d1fc0178e274", fp_eurusd="c85271b5a68f",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Louder_not_more_directional%3F: Confirmed](https://img.shields.io/badge/Louder_not_more_directional%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from ecb_announcement_effect import data, strategy as st

ECB = data.ecb_calendar()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    FEZ, EURUSD = data.load_real()
    DF = st.day_frame(FEZ, EURUSD, ECB)
else:
    FEZ = EURUSD = DF = None
print("real cache present:", HAVE_REAL, "| scheduled ECB decisions:", len(ECB),
      "| tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the euro area flinch when the ECB speaks? 🇪🇺🔔\n"
            "### The ECB announcement effect — a market that gets **louder**, not more "
            "**predictable**\n\n"
            + BADGES +
            "Eight times a year (monthly, before 2015), on a date published **months in "
            "advance**, the European Central Bank's Governing Council announces what it will "
            "do with interest rates. Traders on both sides of the Atlantic swear these days "
            "are special: the euro whips around, European stocks get choppy, and — the "
            "folklore says — you can lean on the direction of the move if you know it's "
            "coming.\n\n"
            "We already know the Fed's version of this story is *real but unbankable* (see "
            "[637-fomc-vol-crush](../../637-fomc-vol-crush/) and "
            "[517-pre-fomc-drift](../../517-pre-fomc-drift/)). Does the ECB's own decision "
            "calendar do the same thing to euro-area equities?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 208 scheduled decision dates (2005→2026) hardcoded from the "
            "ECB's own year-ahead schedule press releases — *scheduled* meetings only. Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the euro-area stock market (FEZ) move a predictable *direction* on ECB "
            f"day? | **No.** {R['gap_pct']:+.2f} points versus a normal day — statistically "
            "nothing (a random 207-day calendar beats it about a third of the time). |\n"
            "| Is ECB day actually *quiet* then? | **No — it's louder.** FEZ's intraday "
            f"high-low swing runs about **{R['rng_ratio']:.0%}** bigger than normal — a real, "
            "measurable effect. The market moves *more*, just not in a knowable *direction*. |\n"
            "| Does the euro (EURUSD) react more than stocks? | **Not measurably.** Its "
            "average absolute daily move on decision day is statistically the same as any "
            "other day. |\n"
            "| Does the market ramp up *into* the decision, the way it does before a Fed "
            f"meeting? | **No sign of it.** The five days before an ECB decision add up to "
            f"**{R['runup_pct']:+.2f}%** — noise. |\n"
            "| Can you trade the day? | **No — it loses money before costs.** Holding FEZ "
            f"just for the decision day nets **{R['tm_gross']:+.1f} bps/event gross**, before "
            f"a single basis point of fee, with a **{R['tm_worst']:+.1f}%** worst day. |\n\n"
            "> Louder pulse, same coin flip."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The ECB is Europe's Fed. Its rate decisions carry real information — and "
            "just like Wall Street trades around FOMC day, European desks should see a "
            "systematic reaction: equities drift or jump on the news, the euro whips around, "
            "and the day is knowably different from any other Thursday.\"*\n\n"
            "It's a reasonable steelman: the Governing Council sets the price of money for "
            "~350 million people, on a calendar you can set your watch to, and the Fed's own "
            "version of this story turns out to be at least partly true (a real VIX crush, a "
            "real — if decayed — pre-meeting equity drift). Why wouldn't the ECB's decisions "
            "move markets the same way?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If ECB decision days really do move euro-area equities in a predictable "
            "direction, that's a calendar-timed edge available to anyone with the schedule — "
            "no forecasting skill required, just discipline about which days to be long or "
            "short. And if the day is genuinely *louder* (even without a direction), that's "
            "still useful: it tells options desks where the event premium belongs, and it "
            "tells anyone managing risk which Thursdays deserve a smaller position.\n\n"
            "So we ask three things: does FEZ move on ECB day, does it move *loudly*, and — "
            "the only question that actually pays — can you bank any of it after costs?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_calendar']}** scheduled ECB decision dates from "
            f"{R['cal_lo']} to {R['cal_hi']}, hardcoded from the ECB's own year-ahead schedule "
            "announcements — monthly through 2014, then the now-familiar 6-week cycle from "
            "2015.\n"
            "- **The comparison.** FEZ's daily return and intraday swing on those "
            f"**{R['n_ecb']}** decision days vs the other **{R['n_rest']:,}** trading days.\n"
            "- **The luck check.** Draw 207 random days instead, 20,000 times — how often "
            "does a random calendar produce a mean this large?\n"
            "- **The trade check.** Buy FEZ at the close *before* the decision (the date is "
            "public months ahead — no crystal ball needed), sell at the decision-day close, "
            "pay costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average FEZ return on decision days vs every other day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.decision_day_stats(DF)\n"
            "    ep, rp = s['ecb_pct'], s['rest_pct']\n"
            "else:\n"
            "    ep, rp = R['ecb_pct'], R['rest_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['ECB decision days\\n(n=207)','all other days\\n(n=5,198)'], [ep, rp],\n"
            "       color=[GREY, GREY], width=.6)\n"
            "for i,v in enumerate([ep, rp]): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average daily FEZ return (%)')\n"
            "ax.set_title('No directional edge: ECB day looks like any other day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'decision days {ep:+.3f}%   other days {rp:+.3f}%')"
        ),
        md(
            f"Flat. **{R['ecb_pct']:+.3f}%** on the average decision day versus "
            f"**{R['rest_pct']:+.3f}%** on a normal one — the gap is well inside noise, and a "
            "random calendar of 207 days matches or beats it about a third of the time. "
            "**But** — and this is the interesting part — the day isn't *quiet*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.range_stats(DF)\n"
            "    a, b = rg['ecb_range_pct'], rg['rest_range_pct']\n"
            "else:\n"
            "    a, b = R['rng_ecb'], R['rng_rest']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['ECB decision days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg FEZ high-low range (% of prev close)')\n"
            "ax.set_title('Louder: FEZ swings ~19% more on decision day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'range: ECB days {a:.3f}%  vs other {b:.3f}%')"
        ),
        md(
            f"FEZ's intraday high-low swing runs **{R['rng_ecb']:.2f}%** on decision days "
            f"versus **{R['rng_rest']:.2f}%** normally — about **{R['rng_ratio']:.0%}** "
            "bigger, and this one clears the statistical bar convincingly (the quants "
            "notebook has the full Welch/HAC/placebo breakdown). So the day genuinely *is* "
            "different — the market just can't tell you which way it'll go.\n\n"
            "**What about the euro itself?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fx = st.eurusd_stats(DF)\n"
            "    a, b = fx['ecb_abs_pct'], fx['rest_abs_pct']\n"
            "else:\n"
            "    a, b = R['fx_ecb'], R['fx_rest']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['ECB decision days','other days'], [a, b], color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg EURUSD |daily return| (%)')\n"
            "ax.set_title('EURUSD does not react more than usual either')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EURUSD |return|: ECB days {a:.3f}%  vs other {b:.3f}%')"
        ),
        md(
            "Surprising, but consistent: the currency doesn't whip around more than usual "
            "either. Whatever information the decision carries seems to already be priced in "
            "well before the press release — a point the quants notebook connects to how "
            "efficiently rate expectations trade in FX markets.\n\n"
            "**Finally, the trade.** Can you bank the \"louder day\" by just holding FEZ "
            "through it?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.timer_capture(DF, cost_bps=5.0)\n"
            "    g = tc['gross_bps']\n"
            "else:\n"
            "    g = R['tm_gross']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['ECB decision day\\n(gross, before costs)'], [g], color=RED, width=.45)\n"
            "ax.annotate(f'{g:+.1f} bps',(0,g),ha='center',va='top' if g<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg FEZ return per decision day (bps, gross)')\n"
            "ax.set_title('Negative before a single basis point of cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross return per decision day: {g:+.1f} bps')"
        ),
        md(
            f"**{R['tm_gross']:+.1f} bps per event, gross** — negative before any cost is "
            f"charged. Add the worst single day (**{R['tm_worst']:+.1f}%**) and the picture is "
            "clear: there's no drift to harvest, and \"the day is louder\" doesn't help you if "
            "you don't know which way it'll be loud."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Real on the vol (FEZ genuinely swings "
            f"**{R['rng_ratio']:.0%}** more on decision day), **none** on the drift (no "
            "directional edge, no pre-meeting ramp, no extra EURUSD reaction).\n"
            "- **Tradability — Mirage.** The only rule with a pulse loses money **before** "
            "costs.\n"
            "- **\"Louder, not more directional\"? — Confirmed.** The ECB resolves "
            "uncertainty on decision day; it just doesn't resolve it in a predictable "
            "direction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson travels.** Scheduled macro announcements load a vol "
            "premium into the surface that must expire once the news is out — the ECB is just "
            "one more instance, alongside the Fed ([637](../../637-fomc-vol-crush/)) and OPEC "
            "([606](../../606-opec-announcement-effect/)): louder day, coin-flip direction, "
            "no drift to bank in the underlying.\n"
            "- **Where the pros actually play it** is the *options* on decision day (the "
            "vol premium itself), not a directional bet on the ETF.\n"
            "- **Sibling studies:** the [Fed's own vol crush](../../637-fomc-vol-crush/), its "
            "[pre-meeting equity drift](../../517-pre-fomc-drift/) and "
            "[OPEC's version](../../606-opec-announcement-effect/) of exactly this same "
            "real-vol/no-drift finding — none of them is this study: euro-area equities, on "
            "the ECB's own calendar.\n\n"
            "*Think the euro-area reaction hides inside a shorter window than a daily bar can "
            "see? Show a net, certifiable intraday edge — after the market-maker's spread on "
            "ECB afternoons — then we'll talk.*"
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
            "# The ECB Announcement Effect — a quantitative teardown 🔬\n"
            "### Decision-day Welch/HAC splits · a two-sided random-calendar placebo · the "
            "[−5..+3] Lucca-Moench-style event anatomy · the EURUSD cross-check · the 2015 "
            "cadence-era contrast · an honest \"costs on a timer\" test · a 20-seed synthetic "
            "null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **euro-area equities react systematically around ECB Governing "
            "Council decisions** — is the ECB's own version of the Fed folklore this desk "
            "already tested twice ([637](../../637-fomc-vol-crush/), "
            "[517](../../517-pre-fomc-drift/)). The job here is to measure it honestly on the "
            "ECB's own calendar, then ask the only question that pays: *is any of it "
            "tradable?*\n\n"
            "> ⚠️ **Data note.** FEZ raw OHLC + EURUSD=X OHLC (2005→2026), yfinance, cached; "
            "**208 hardcoded scheduled ECB decision dates** from the ECB's own year-ahead "
            "schedule press releases (2005-2014, monthly) and statement archive (2015-2026, "
            "6-week cycle); 207 land on the FEZ tape (the scheduled 2013-07-04 decision fell "
            "on a US market holiday, named quirk). No survivorship (baskets/rates, not a "
            "panel). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_fez"] +
            "` / `" + R["fp_eurusd"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | drift: Welch **t = {R['welch_t']:.2f}**, NW "
            f"**t = {R['nw_t']:.2f}**, placebo **p = {R['placebo_p']:.3f}** — none. range: "
            f"**{R['rng_ratio']:.2f}×**, Welch **t = {R['rng_welch_t']:.2f}**, NW "
            f"**t = {R['rng_nw_t']:.2f}**, placebo **p = {R['rng_placebo_p']:.4f}** — real |\n"
            f"| **Tradability** | `MIRAGE` | timer rule gross **{R['tm_gross']:+.1f} bps/event**"
            f" (before costs), worst day **{R['tm_worst']:+.1f}%** |\n"
            "| **Louder, not more directional?** | `CONFIRMED` | range real "
            f"(*t* ≥ {R['rng_welch_t']:.2f}), direction (*t* = {R['welch_t']:.2f}) and EURUSD "
            f"reaction (*t* = {R['fx_t']:.2f}) both null |\n\n"
            "> 💡 In plain words: the ECB genuinely turns up the volume on decision day; it "
            "just never tells you which way the needle will swing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be FEZ's daily close-to-close log return and $D_t \\in \\{0,1\\}$ the "
            "scheduled-decision-day flag (known *ex ante* — the calendar is published months "
            "ahead). The decision lands at 13:45 CET, the press conference at 14:30/14:15 CET "
            "— both well before FEZ's US-market close — so the decision-day bar should "
            "contain the reaction. The claims:\n\n"
            "- **H₁ (drift).** $E[r_t \\mid D_t=1] \\ne E[r_t \\mid D_t=0]$ — a systematic "
            "directional move, large enough to clear noise.\n"
            "- **H₂ (loud, not calm).** Realized FEZ range on decision days exceeds the "
            "baseline — a genuine reaction is visible even without a signed direction.\n"
            "- **H₃ (FX cross-check).** EURUSD's absolute move is larger on decision days — "
            "the currency, pricing the policy surprise most directly, should react too.\n"
            "- **H₄ (anatomy).** A Lucca-Moench-style pre-meeting run-up and a post-day "
            "persistence pattern.\n"
            "- **H₅ (capture).** A calendar timing rule (FEZ, decision day only) banks the "
            "effect net of costs.\n\n"
            "We find **H₁ NOT supported** (drift indistinguishable from noise), **H₂ "
            "supported** (range *t* ≥ 2.66 three ways), **H₃ NOT supported** (EURUSD "
            "*t* = 0.65), **H₄ NOT supported** (every event-window offset |*t*| < 1.6), "
            "**H₅ NOT supported** (negative gross before costs)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Decision days are **single, non-overlapping events**, so the planned primary is a "
            "**Welch t** on the group split. Because realized range is strongly serially "
            "correlated (vol clustering), the **Newey-West (5-lag) t** on the dummy regression "
            "is the load-bearing robustness check for H₂, not a formality — a naive Welch "
            "split on an autocorrelated series can overstate significance, and we want the bar "
            "cleared on both. A two-sided **20,000-draw random-calendar placebo** (20 seeds × "
            "1,000 draws) checks both H₁ and H₂ against \"could a random 207-day calendar have "
            "looked this good\", and the 2015 cadence-era split (justified *ex ante* — the "
            "Governing Council's own July-2014 announcement) is tested as a **difference**, "
            "not eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_calendar']} scheduled decisions {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded (ECB schedule press releases + statement archive); "
            f"{R['n_ecb']} land on the FEZ tape (one US-holiday collision, named).\n"
            f"- **Tape.** FEZ raw OHLC + EURUSD=X OHLC {R['start']} → {R['end']}. As-of "
            "2026-06-30 (last complete month).\n"
            "- **Headline.** Welch t + NW(5) t + two-sided 20-seed placebo, on both the return "
            "(drift) and the range (loudness).\n"
            "- **Cross-check.** EURUSD |return| split on the same days.\n"
            "- **Anatomy.** Event window [−5..+3], per-offset Welch t vs far days; cumulative "
            "run-up per meeting, one-sample t.\n"
            "- **Execution (third axis).** Enter FEZ at the prior close (calendar public — "
            "zero look-ahead), exit the decision close; 2 × one-way cost × NAV per event; "
            "long-only, no borrow.\n"
            "- **Control.** Synthetic i.i.d.-return world, planted drift knob; the null must "
            "not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo — the drift\n\n"
            "Welch t on FEZ log return (decision vs rest), NW t on the dummy regression, and "
            "the random-calendar null. In the notebook we run a lighter placebo (4 seeds × 500 "
            "draws) and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.decision_day_stats(DF)\n"
            "    print(f\"ECB-day FEZ return {s['ecb_pct']:+.3f}%  vs  other {s['rest_pct']:+.3f}%\")\n"
            "    print(f\"Welch t = {s['welch_t']:+.2f}   NW(5) t = {s['nw_t']:+.2f}\")\n"
            "    print(f\"hit {s['hit_up']}/{s['n_ecb']} = {s['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    pl = st.placebo_pvalue(DF, column='fez_ret', n_draws_per_seed=500, n_seeds=4)\n"
            "    obs, draws = pl['obs'], pl['draws']\n"
            "else:\n"
            "    obs = R['ecb_pct'] / 100\n"
            "    rng = np.random.default_rng(645)\n"
            "    draws = rng.normal(R['placebo_mean']/100, R['placebo_sd']/100, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 207 days (light in-notebook run)')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed ECB-day mean {obs*100:+.3f}%')\n"
            "ax.set_xlabel('mean FEZ return of a random 207-day calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the luck cloud: canonical p = {R['placebo_p']:.3f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.4f}%, \"\n"
            "      f\"sd {R['placebo_sd']:.4f}%, p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['ecb_pct']:+.3f}%** sits well inside the "
            f"null's spread ({R['placebo_mean']:+.4f}% ± {R['placebo_sd']:.4f}%); "
            f"**p = {R['placebo_p']:.3f}** — about a third of random calendars do as well or "
            f"better. Welch t = **{R['welch_t']:.2f}** and NW t = **{R['nw_t']:.2f}** agree: "
            "H₁ (drift) is not supported."
        ),
        md(
            "### 4b · The realized-range cross-check — loudness IS real\n\n"
            "Same machinery, applied to (H−L)/prev-close instead of the signed return. Range "
            "is strongly autocorrelated (volatility clusters), so the NW cross-check is the "
            "one that actually decides this cell."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.range_stats(DF)\n"
            "    a, b, ratio, tw, tnw = (rg['ecb_range_pct'], rg['rest_range_pct'], rg['ratio'],\n"
            "                            rg['welch_t'], rg['nw_t'])\n"
            "    pr = st.placebo_pvalue(DF, column='fez_range', n_draws_per_seed=500, n_seeds=4)\n"
            "    draws_r = pr['draws']\n"
            "else:\n"
            "    a, b, ratio, tw, tnw = (R['rng_ecb'], R['rng_rest'], R['rng_ratio'],\n"
            "                            R['rng_welch_t'], R['rng_nw_t'])\n"
            "    rng2 = np.random.default_rng(645)\n"
            "    draws_r = rng2.normal(R['rng_rest']/100, 0.007, 2000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['ECB days','other days'], [a, b], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): a1.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('avg FEZ high-low range (%)')\n"
            "a1.set_title(f'{ratio:.2f}x louder (Welch t={tw:+.2f}, NW t={tnw:+.2f})')\n"
            "a2.hist(draws_r*100, bins=50, color=GREY, alpha=.85, label='null (light run)')\n"
            "a2.axvline(a, c=RED, lw=2.5, label=f'observed {a:.3f}%')\n"
            "a2.set_xlabel('mean range of a random 207-day calendar (%)')\n"
            "a2.set_title(f\"canonical placebo p = {R['rng_placebo_p']:.4f}\")\n"
            "a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'range: ECB {a:.3f}% vs other {b:.3f}% ({ratio:.2f}x)  '\n"
            "      f'Welch t={tw:+.2f}  NW t={tnw:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: FEZ's high-low swing runs **{R['rng_ratio']:.2f}×** normal "
            f"on decision day, and it clears *t* ≥ 2 **three** separate ways — raw Welch "
            f"(**{R['rng_welch_t']:.2f}**), the autocorrelation-robust NW cross-check "
            f"(**{R['rng_nw_t']:.2f}**), and a two-sided placebo (**p = "
            f"{R['rng_placebo_p']:.4f}**). H₂ is genuinely supported — this is the one real "
            "signal in the study."
        ),
        md(
            "### 4c · The EURUSD cross-check — the FX leg doesn't react more either\n\n"
            "If the decision were moving markets hard, the currency — which prices the policy "
            "surprise most directly — should be the first place to see it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fx = st.eurusd_stats(DF)\n"
            "    a, b, t = fx['ecb_abs_pct'], fx['rest_abs_pct'], fx['welch_t']\n"
            "else:\n"
            "    a, b, t = R['fx_ecb'], R['fx_rest'], R['fx_t']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['ECB days','other days'], [a, b], color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([a, b]): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg EURUSD |daily return| (%)')\n"
            "ax.set_title(f'No extra FX reaction either (Welch t = {t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'EURUSD |return|: ECB {a:.3f}%  vs other {b:.3f}%   Welch t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: EURUSD's absolute move on decision day ({R['fx_ecb']:.3f}%) "
            f"is statistically the same as any other day ({R['fx_rest']:.3f}%, "
            f"*t* = {R['fx_t']:.2f}). H₃ is not supported — whatever the FEZ range bump picks "
            "up, it isn't a proportionally larger currency shock."
        ),
        md(
            "### 4d · Anatomy — a Lucca-Moench-style event window\n\n"
            "Per-offset means with Welch t vs far-from-meeting days; the run-up is tested as a "
            "**cumulative per-meeting** quantity (one-sample t across 207 meetings)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(DF, ECB)\n"
            "    ks = list(ev.index); ms = list(ev['mean_pct']); ts = list(ev['welch_t'])\n"
            "    ru = st.runup_stats(DF, ECB)\n"
            "    ru_m, ru_t = ru['mean_runup_pct'], ru['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][1] for k in ks]; ru_m, ru_t = R['runup_pct'], R['runup_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k==0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean FEZ return (%)')\n"
            "a1.set_title('No clean anatomy: no ramp-in, no crush, no bounce')\n"
            "a2.bar([str(k) for k in ks], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from decision)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cumulative run-up [-5..-1]: {ru_m:+.3f}%/meeting (t = {ru_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: every single offset in the nine-day window sits below "
            f"|*t*| = 1.6. The pre-meeting run-up (**{R['runup_pct']:+.2f}%/meeting**, "
            f"*t* = {R['runup_t']:.2f}) shows no trace of a Lucca-Moench-style build-up, and "
            f"day 0 (**{R['event'][0][0]:+.3f}%**, *t* = {R['event'][0][1]:.2f}) isn't even the "
            "largest bar in the window — day −3 is bigger in magnitude. H₄ is not supported."
        ),
        md(
            "### 4e · The era contrast — justified split, tested as a difference\n\n"
            "Split at **2015-01-01** (the Governing Council's own move to the 6-week cycle, "
            "announced 2014-07; chosen ex ante, not snooped). Applied to both the return "
            "(already-null) and the range (the one real effect)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(DF, data.SIXWEEK_SPLIT, column='fez_ret')\n"
            "    ecr = st.era_contrast(DF, data.SIXWEEK_SPLIT, column='fez_range')\n"
            "    e, l = ec['early_pct'], ec['late_pct']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "    re, rl = ecr['early_pct'], ecr['late_pct']\n"
            "    ret_, rlt, rdt = ecr['welch_t_early'], ecr['welch_t_late'], ecr['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_early'], R['era_late']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "    re, rl = R['rng_era_early'], R['rng_era_late']\n"
            "    ret_, rlt, rdt = R['rng_era_early_t'], R['rng_era_late_t'], R['rng_era_diff_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "a1.bar(['2005-2014\\n(n=115)','2015-2026\\n(n=92)'], [e, l],\n"
            "       color=[GREY, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    a1.annotate(f'{v:+.3f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('ECB-day FEZ return (%)')\n"
            "a1.set_title('Drift: null in both eras')\n"
            "a2.bar(['2005-2014\\n(n=115)','2015-2026\\n(n=92)'], [re, rl], color=[AMBER, AMBER], width=.55)\n"
            "for i,(v,t_) in enumerate([(re,ret_),(rl,rlt)]):\n"
            "    a2.annotate(f'{v:.2f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('ECB-day FEZ range (%)')\n"
            "a2.set_title(f'Range: real level fell (diff t={rdt:+.2f}) but WITHIN-era t stays <2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return: early {e:+.3f} (t={et:+.2f})  late {l:+.3f} (t={lt:+.2f})  diff t={dt:+.2f}')\n"
            "print(f'range:  early {re:.3f} (t={ret_:+.2f})  late {rl:.3f} (t={rlt:+.2f})  diff t={rdt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the drift stays null in both eras "
            f"(*t* = {R['era_early_t']:.2f} then {R['era_late_t']:.2f}) — no story to tell "
            "there. The range level fell across the cadence switch "
            f"(diff *t* = {R['rng_era_diff_t']:.2f}), mostly because the crisis-heavy "
            "2005-2014 bucket carries a higher overall baseline volatility — **but** neither "
            f"era's *within-era* Welch t ({R['rng_era_early_t']:.2f} / "
            f"{R['rng_era_late_t']:.2f}) clears 2 on its own. Said plainly: the full-sample "
            "range significance leans on pooling all 207 events, not on a robust effect "
            "visible in either half alone — an honest caveat, not a reason to downgrade past "
            "`MIXED`."
        ),
        md(
            "### 4f · The third axis — the honest \"costs on a timer\" test\n\n"
            "Enter the prior close (zero look-ahead — the calendar is public months ahead), "
            "exit the decision close, pay 2 × one-way costs per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.timer_capture(DF, cost_bps=cb) for cb in (5.0, 10.0, 20.0)]\n"
            "    g = rows[0]['gross_bps']; n5, n10, n20 = (rows[0]['net_bps'], rows[1]['net_bps'],\n"
            "                                              rows[2]['net_bps'])\n"
            "    tv, worst = rows[0]['welch_t'], rows[0]['worst_day_pct']\n"
            "else:\n"
            "    g, n5, n10, n20 = R['tm_gross'], R['tm_net5'], R['tm_net10'], R['tm_net20']\n"
            "    tv, worst = R['tm_t'], R['tm_worst']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['gross','net 5 bps','net 10 bps','net 20 bps'], [g, n5, n10, n20],\n"
            "       color=[RED, RED, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10, n20]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bps per decision day')\n"
            "ax.set_title(f'Negative before costs (Welch t = {tv:+.2f}), worse after')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f} -> net {n5:+.1f} / {n10:+.1f} / {n20:+.1f} bps  '\n"
            "      f'(t={tv:+.2f});  worst day {worst:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: **{R['tm_gross']:+.1f} bps per event, gross** — already "
            f"negative before a single basis point of cost, at Welch *t* = {R['tm_t']:.2f} "
            f"(nowhere near significant either way), with a **{R['tm_worst']:+.1f}%** worst "
            "single day. H₅ is not supported. Structurally: the range bump is a genuine event "
            "premium, but it belongs in the **options** on decision day — exactly where it's "
            "already marked up — not in a directional bet on the underlying ETF."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic i.i.d.-return world, scheduled pseudo-decisions every 22nd business "
            "day (≈ the 6-week cadence), TUNABLE planted drift. The null (drift = 0) is "
            "checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, dec = data.synthetic_world(drift=0.0, seed=645 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, dec)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, dec = data.synthetic_world(drift=0.005, seed=645)\n"
            "planted_t = st.synthetic_detect(close, dec)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (drift=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted drift = +0.5%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (decision vs rest)')\n"
            "ax.set_title('Control: no null fires; a planted effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses "
            f"the bar; a planted 0.5% drift reads t = {R['syn_planted_t']:.2f}. The machinery "
            "is unbiased — the real-tape null (H₁) is a genuine absence, not a blind spot. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — real on the vol: FEZ range **{R['rng_ratio']:.2f}×** "
            f"baseline, Welch t = **{R['rng_welch_t']:.2f}**, NW t = **{R['rng_nw_t']:.2f}**, "
            f"placebo p = **{R['rng_placebo_p']:.4f}**. None on the drift: Welch t = "
            f"**{R['welch_t']:.2f}**, NW t = **{R['nw_t']:.2f}**, placebo p = "
            f"**{R['placebo_p']:.3f}**; EURUSD reaction null (*t* = {R['fx_t']:.2f}); no "
            f"pre-meeting run-up (*t* = {R['runup_t']:.2f}). Caveat named: neither era split "
            f"alone clears *t* = 2 on the range bump ({R['rng_era_early_t']:.2f} / "
            f"{R['rng_era_late_t']:.2f}).\n"
            f"- **Tradability `MIRAGE`** — the timer rule is negative **gross**, before costs "
            f"({R['tm_gross']:+.1f} bps/event, *t* = {R['tm_t']:.2f}), with a "
            f"{R['tm_worst']:+.1f}% worst day. There is no drift to harvest and the vol "
            "elevation is not a directional edge.\n"
            "- **\"Louder, not more directional\"? `CONFIRMED`** — the range bump is real and "
            "three-ways robust; direction (FEZ) and the FX cross-check (EURUSD) are both "
            "statistically indistinguishable from an ordinary Thursday."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is the announcement premium, again.** This is the third "
            "time this desk finds the same shape — real vol elevation, null signed drift — on "
            "a scheduled central-bank or cartel decision "
            "([637-fomc-vol-crush](../../637-fomc-vol-crush/), "
            "[606-opec-announcement-effect](../../606-opec-announcement-effect/), now the "
            "ECB). The professional expression lives in the **options** on decision day, not "
            "in a directional bet on the underlying.\n"
            "- **Why the FX cross-check came up empty is itself informative:** EURUSD is one "
            "of the most liquid, most efficiently-traded pairs on earth, with a deep options "
            "market pricing every scheduled central-bank date years ahead. If any market "
            "should show *zero* extra reaction to a fully-anticipated event, it's this one — "
            "measuring the analogous intraday (not daily) reaction is the natural sequel.\n"
            "- **Dedup map:** [637-fomc-vol-crush](../../637-fomc-vol-crush/) (the Fed's own "
            "vol crush), [517-pre-fomc-drift](../../517-pre-fomc-drift/) / "
            "[67-fed-drift](../../67-fed-drift/) (the Fed's pre-meeting equity drift and its "
            "decay), [135-fomc-cycle](../../135-fomc-cycle/) (Fed week-parity returns), "
            "[322-fomc-blackout](../../322-fomc-blackout/) (the Fed's pre-meeting window), "
            "[606-opec-announcement-effect](../../606-opec-announcement-effect/) (the closest "
            "sibling in *shape* — same Real-vol/None-drift verdict, different institution and "
            "asset class), [314-jackson-hole](../../314-jackson-hole/) (an unscheduled-content "
            "central-bank speech, not a rate decision).\n\n"
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
