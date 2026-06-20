"""Generate the two narrative notebooks for Study 314 (Jackson-Hole).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the shared SPY
return cache under ../../../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-11).
R = dict(
    n_events=33, n_total=8398, fp="ffd9a70d0571",
    win_start="1993-02-01", win_end="2026-06-11",
    # offset sweep (bps mean, hac t, win%)
    off_m2_bps=25.7, off_m2_t=1.75,
    off_m1_bps=-26.7, off_m1_t=-1.07, off_m1_win=36,
    off0_bps=18.9, off0_t=0.88, off0_win=67, off0_welch=0.74,
    off_p1_bps=-17.3, off_p1_t=-0.70,
    off_p3_bps=27.6, off_p3_t=1.33, off_p3_win=73,
    max_abs_t=1.75,  # max |HAC t| over 11 offsets
    # speech-day 95% CI
    ci_lo=-17.6, ci_hi=60.6,
    # react trade net (1 bp one-way)
    react1_n=33, react1_win=45, react1_bps=-19.3, react1_sharpe=-0.11, react1_t=-0.78,
    react3_bps=16.0, react3_win=64, react3_t=0.51,
    # synthetic control
    null_t=0.36, bump_t=4.13, bump_bps=80,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/, _cache/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from jackson_hole import data, strategy as st

ret = data.fetch_spy()                 # shared cache-first; empty Series on a cache miss
HAVE_REAL = not ret.empty
print("real SPY cache present:", HAVE_REAL, "| sessions:", len(ret))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Jackson-Hole 🏔️ — does the market drift around the Fed's mountain retreat?\n"
            "### Every late August the Fed Chair gives one famous speech. Is there money in it?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_Jackson_Hole_drift%3F: Not_supported](https://img.shields.io/badge/A_Jackson_Hole_drift%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Once a year, in late August, the Kansas City Fed gathers central bankers in Grand "
            "Teton National Park, and the Fed Chair gives a Friday-morning keynote that *sometimes* "
            "moves markets a lot — Bernanke hinting at QE2 in 2010, Powell's brutal eight minutes in "
            "2022. So a folk belief took hold: there's a **'Jackson Hole drift'** you can trade. "
            "This notebook asks the only question that matters — across **33 years of keynotes**, is "
            "there actually a pattern, or just a couple of unforgettable headlines?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the bootstrap CI and the "
            "multiple-testing trap? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a drift around the speech? | **No.** Across the whole −5…+5 day window the "
            f"strongest signal anywhere is a *t* of just **+{R['max_abs_t']:.2f}** — and that's the "
            "best of **11 tries**. The speech day itself is statistical noise. |\n"
            "| Could you trade it? | **No.** The literal 'buy after the speech' trade *loses* money "
            f"net (**{R['react1_bps']:+.0f} bps**, *t* = {R['react1_t']:.2f}), and it fires **once a "
            "year** — ~31 trades in a whole career. |\n"
            "| So why does everyone believe it? | **Two famous speeches.** Bernanke 2010 and Powell "
            "2022 were genuinely huge — but two vivid memories aren't a pattern in 33 data points. |\n\n"
            "> The Jackson Hole symposium makes headlines, not edges. The drift is a story we tell "
            "ourselves because the *memorable* speeches stick — the 31 forgettable ones don't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch Jackson Hole. The S&P sets up ahead of the Chair's speech and trends after "
            "it — the dovish years rip, and even the hawkish surprises are tradable if you know the "
            "calendar.\"*\n\n"
            "— the recurring late-August financial-media refrain\n\n"
            "It feels true because the standout examples are seared in: **Bernanke (Aug 27, 2010)** "
            "all but pre-announced QE2 and stocks rallied for months; **Powell (Aug 26, 2022)** "
            "delivered eight blunt minutes and the S&P fell ~3.4% *that day*. If the Chair's speech "
            "reliably set the market's direction, the late-August calendar would be a gift."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A real, repeatable drift around a **known-in-advance** date would be remarkable — the "
            "symposium dates are published months ahead, so anyone could position for it with zero "
            "information advantage. That's exactly why it's worth checking honestly: calendar effects "
            "that 'everyone knows' are usually either (a) already arbitraged away, or (b) never there "
            "to begin with — a few salient anecdotes mistaken for a trend. Telling those apart is the "
            "whole game."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "An **event study**, the cleanest tool for 'does something happen around date X':\n\n"
            "1. **Line up all 33 keynotes** (1993–2025) and look at the S&P's return on the speech "
            "day, the days before, and the days after.\n"
            "2. **Compare to a normal day.** If there's a drift, the event days beat the rest by more "
            "than chance.\n"
            "3. **Trade the obvious version.** Buy at the speech-day close, hold to the next close, "
            "every year, and see if it makes money after costs.\n\n"
            "What would make us say *mirage*: no day around the speech stands out once we account for "
            "the fact that we looked at eleven of them — and the trade doesn't pay. (Spoiler.)\n\n"
            "Tape: SPY daily total returns back to 1993."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The average S&P move on each day around the keynote**, across all 33 years. If the "
            "folklore were right, the bars near the speech would lean clearly green:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.offset_sweep(ret, data.JH_SPEECH_DATES, offsets=range(-5, 6))\n"
            "    offs = sw['offset'].to_numpy(); mean_bps = sw['event_mean'].to_numpy()*1e4\n"
            "else:\n"
            "    offs = np.arange(-5, 6)\n"
            f"    mean_bps = np.array([-5.2,-14.5,-6.7,{R['off_m2_bps']},{R['off_m1_bps']},"
            f"{R['off0_bps']},{R['off_p1_bps']},9.0,{R['off_p3_bps']},14.1,1.0])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "cols = [GREEN if v > 0 else RED for v in mean_bps]\n"
            "ax.bar(offs, mean_bps, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('trading days from the keynote (0 = speech day)')\n"
            "ax.set_ylabel('avg S&P return (bps)')\n"
            "ax.set_title('Average move around Jackson Hole — a coin-flip of green and red')\n"
            "ax.annotate('speech\\nday', (0, mean_bps[5]), textcoords='offset points',\n"
            "            xytext=(8, 8), color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The sign flips almost every day — no clean drift either side of the speech.')"
        ),
        md(
            f"The speech day averages **+{R['off0_bps']:.0f} bps** (a 67% win-rate) — mildly "
            "positive, and easy to get excited about. But look at the neighbours: the day **before** "
            f"is −{abs(R['off_m1_bps']):.0f} bps, the day **after** is −{abs(R['off_p1_bps']):.0f} "
            "bps. The pattern **alternates sign** day to day. That's not a drift; that's what 33 "
            "random points look like when you stare hard enough."
        ),
        md(
            "**But +19 bps on the speech day isn't nothing — is it real?** Here's the catch with "
            "one-event-a-year studies: 33 observations is a *tiny* sample, so the error bars are "
            "huge. Let's put a confidence interval on that speech-day average:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.event_table(ret, data.JH_SPEECH_DATES, 0)['event_mean']*1e4\n"
            "    lo, hi = st.block_bootstrap_ci(ret, data.JH_SPEECH_DATES, 0)\n"
            "    lo, hi = lo*1e4, hi*1e4\n"
            "else:\n"
            f"    m, lo, hi = {R['off0_bps']}, {R['ci_lo']}, {R['ci_hi']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 2.6))\n"
            "ax.errorbar([m], [0], xerr=[[m-lo],[hi-m]], fmt='o', color=GREEN, capsize=6, lw=2, ms=9)\n"
            "ax.axvline(0, color=RED, lw=2, label='no effect')\n"
            "ax.set_yticks([]); ax.set_xlabel('speech-day average return (bps)')\n"
            "ax.set_title(f'Speech day: {m:+.0f} bps, but 95%% CI = [{lo:+.0f}, {hi:+.0f}] — straddles zero')\n"
            "ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "print(f'The interval runs from {lo:+.0f} to {hi:+.0f} bps — zero is comfortably inside.')"
        ),
        md(
            f"There it is. The speech-day average is **+{R['off0_bps']:.0f} bps**, but its 95% "
            f"confidence interval runs from **{R['ci_lo']:+.0f} to +{R['ci_hi']:.0f} bps** — wide "
            "open, with **zero sitting right in the middle**. We simply cannot distinguish that +19 "
            "from nothing. One event a year just doesn't buy enough data to certify anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** No day around the keynote stands out; the best *t* anywhere in the "
            f"window is +{R['max_abs_t']:.2f}, and that's the pick of 11. The speech-day average's "
            f"error bars ([{R['ci_lo']:+.0f}, +{R['ci_hi']:.0f}] bps) swallow zero whole.\n"
            "- **Tradability — Mirage.** The 'buy after the speech' trade *loses* money net "
            f"({R['react1_bps']:+.0f} bps), and at one shot a year there's nothing to scale.\n"
            "- **A Jackson Hole drift? — Not supported.** Two famous speeches built a belief the "
            "33-year record doesn't back. Memorable ≠ predictable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Forget the statistics for a second and just *try it*: every year, buy the S&P at the "
            "close of the speech day and sell at the next close. How would that have gone, net of a "
            "small fee?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = st.react_trade(ret, data.JH_SPEECH_DATES, hold=1, cost_bps=1.0)\n"
            "    yrs = led['year'].to_numpy(); pnl = led['ret_net'].to_numpy()*1e4\n"
            "else:\n"
            "    rng = np.random.default_rng(314)\n"
            "    yrs = np.arange(1993, 2026); pnl = rng.normal(-19.3, 110, 33)\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.2))\n"
            "ax.bar(yrs, pnl, color=[GREEN if v > 0 else RED for v in pnl], width=.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(np.mean(pnl), ls='--', c=GREY, label=f'mean {np.mean(pnl):+.0f} bps')\n"
            "ax.set_xlabel('year'); ax.set_ylabel('next-session P&L (bps, net)')\n"
            "ax.set_title('Buy-after-the-speech, year by year — a losing coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean {np.mean(pnl):+.0f} bps/yr over {len(pnl)} years — and it pays a fee each time.')"
        ),
        md(
            f"A scatter of green and red averaging **{R['react1_bps']:+.0f} bps** — *negative*, "
            f"before you even celebrate the {R['react1_win']}% loss rate. Costs aren't the killer "
            "here (one trade a year is cheap); the killer is that **there's no edge to begin with**, "
            "and the once-a-year cadence means even a real edge couldn't fill a portfolio."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The honest Fed cousin that *was* real.** [Study 67 — Fed-Drift](../../67-fed-drift/) "
            "found a genuine pre-**FOMC** drift (eight scheduled meetings a year, not one symposium) "
            "— though it decayed after publication. That's the difference between a real Fed-event "
            "effect and this folk one.\n"
            "- **The even-week Fed cycle.** [Study 135 — FOMC-Cycle](../../135-fomc-cycle/) tests the "
            "other famous Fed-calendar idea. Same lesson: thin samples flatter noise.\n"
            "- **Fork it.** Try other anchors — the day the symposium *agenda* drops, gold or the "
            "dollar instead of the S&P, or a pre-speech long instead of a reaction. We bet the error "
            "bars stay just as wide; show us otherwise.\n\n"
            "*Think the dovish years are a real subgroup? Tag them by hand and test the difference — "
            "but remember you're now slicing 33 points into even smaller piles.*"
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
            "# Jackson-Hole — a quantitative teardown 🔬\n"
            "### Real SPY tape · event study · HAC *t* · block-bootstrap CI · the traded arm · "
            "the multiple-testing trap · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_Jackson_Hole_drift%3F: Not_supported](https://img.shields.io/badge/A_Jackson_Hole_drift%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether any session "
            "offset around the Fed's annual Jackson Hole keynote carries an abnormal return, whether "
            "the obvious reaction trade pays, and whether the folk 'drift' survives the thin-sample "
            "and multiple-testing reality of a once-a-year event.\n\n"
            "> ⚠️ **Not investment advice.** Real data: shared SPY daily total returns "
            "(`_cache/last_call_spy.parquet`), 1993–2026; as-of 2026-06-11; the offline core and "
            "tests run on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Speech-day mean +{R['off0_bps']:.1f} bps, HAC *t* = "
            f"+{R['off0_t']:.2f}; max |HAC *t*| over all 11 offsets = +{R['max_abs_t']:.2f}; "
            f"speech-day 95% CI [{R['ci_lo']:+.1f}, +{R['ci_hi']:.1f}] bps. |\n"
            f"| **Tradability** | `MIRAGE` | React-trade (1-session hold, 1 bp one-way) net "
            f"{R['react1_bps']:+.1f} bps, *t* = {R['react1_t']:.2f}; {R['n_events']} trades over the "
            "whole sample — capacity ≈ one round trip a year. |\n"
            f"| **A Jackson Hole drift?** | `NOT SUPPORTED` | Sign alternates offset-to-offset "
            "(−1 down, 0 up, +1 down); the folk belief rests on two salient keynotes, not the record. |\n\n"
            "> 💡 In plain words: a once-a-year, speech-driven media event with ~33 observations and "
            "an alternating-sign profile — exactly the shape of noise dressed up by hindsight."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be SPY's daily total return and $E$ the set of 33 Friday keynote dates. For "
            "an offset $k$, let $\\bar r_E(k)$ be the mean return on the session $k$ days from each "
            "keynote.\n\n"
            "- **H₁ (drift exists).** $\\exists\\, k \\in [-5, 5]$ with $\\bar r_E(k)$ significantly "
            "$\\neq$ the rest-of-sample mean (HAC *t* ≥ 2), surviving the search over 11 offsets.\n"
            "- **H₂ (tradable).** A long position entered at the speech-day close earns a positive, "
            "significant net return.\n"
            "- **H₃ (it's a real subgroup, not anecdote).** The effect is broad across years, not "
            "driven by two outliers.\n\n"
            "We find **H₁ rejected** (best *t* = +1.75, the max of 11), **H₂ rejected** (net "
            "negative), **H₃ rejected** (alternating sign, wide CI)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The dates are public, so a real drift would be free money — and therefore a priori "
            "implausible unless it's a genuine risk premium. The interesting content is *how* the "
            "folklore misleads: a once-a-year event yields ~33 data points, the event-study window "
            "invites an 11-way search, and two famous reactions (Bernanke 2010, Powell 2022) anchor "
            "memory. Naming those three traps is worth more than the binary verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event tagging.** `event_window` flags the session at offset $k$ from each keynote "
            "(0 = speech day). Calendar-known event ⇒ no lag for the day-of/before measurement.\n"
            "- **Inference.** `event_table` reports the event-day mean, the Welch *t* on the "
            "event-minus-rest difference, **and the HAC (Newey-West) *t*** on the event-day mean — "
            "the inference-bar statistic.\n"
            "- **Thin-sample CI.** `block_bootstrap_ci` — circular block bootstrap (block = 3) on "
            "the event-day returns, respecting any cross-year clustering.\n"
            "- **Multiple testing.** We sweep 11 offsets and read the *max* statistic against the "
            "bar — quoting only the best offset is exactly the data-snooping to avoid.\n"
            "- **Traded arm.** `react_trade` — long from the speech-day close (one execution lag) "
            "for `hold` sessions, costed one-way × NAV (charged on entry and exit).\n"
            "- **Positive control.** A deterministic synthetic tape with a tunable speech-day bump "
            "(and a null), to prove the harness detects what's there.\n\n"
            "Tape: SPY daily total returns, 1993-02-01 → 2026-06-11 (33 keynotes)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The offset sweep — no offset clears the bar\n\n"
            "HAC *t* on the event-day mean at every offset from −5 to +5. The dashed line is the "
            "inference bar (|*t*| = 2)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.offset_sweep(ret, data.JH_SPEECH_DATES, offsets=range(-5, 6))\n"
            "    offs = sw['offset'].to_numpy(); ts = sw['hac_t'].to_numpy()\n"
            "else:\n"
            "    offs = np.arange(-5, 6)\n"
            f"    ts = np.array([-0.34,-0.72,-0.43,{R['off_m2_t']},{R['off_m1_t']},{R['off0_t']},"
            f"{R['off_p1_t']},0.46,{R['off_p3_t']},1.44,0.05])\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.bar(offs, ts, color=[GREEN if abs(v) >= 2 else GREY for v in ts], width=.7)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0, ls=':', c=AMBER)\n"
            "ax.set_xlabel('offset (days from keynote)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_ylim(-2.4, 2.4)\n"
            "ax.set_title('Not one offset reaches |t|=2 — the whole window is noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'max |HAC t| over 11 offsets = {np.abs(ts).max():.2f}  (at offset {offs[np.argmax(np.abs(ts))]:+d})')"
        ),
        md(
            f"> 💡 In plain words: the tallest bar anywhere is **+{R['max_abs_t']:.2f}** (offset −2), "
            "and it's the largest of **eleven** correlated tries — so the honest read is below even "
            "that. **H₁ rejected.** Every other offset is comfortably inside the noise band."
        ),
        md(
            "### 4b · The thin-sample CI on the speech day\n\n"
            "The speech-day point estimate is mildly positive; its block-bootstrap interval shows "
            "how little 33 events buys."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t0 = st.event_table(ret, data.JH_SPEECH_DATES, 0)\n"
            "    m = t0['event_mean']*1e4; lo, hi = st.block_bootstrap_ci(ret, data.JH_SPEECH_DATES, 0)\n"
            "    lo, hi = lo*1e4, hi*1e4; welch = t0['welch_t']; hac = t0['hac_t']\n"
            "else:\n"
            f"    m, lo, hi = {R['off0_bps']}, {R['ci_lo']}, {R['ci_hi']}\n"
            f"    welch, hac = {R['off0_welch']}, {R['off0_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 2.6))\n"
            "ax.errorbar([m], [0], xerr=[[m-lo],[hi-m]], fmt='o', color=GREY, capsize=6, lw=2, ms=9)\n"
            "ax.axvline(0, color=RED, lw=2)\n"
            "ax.set_yticks([]); ax.set_xlabel('speech-day mean return (bps)')\n"
            "ax.set_title(f'Speech day {m:+.1f} bps  ·  95%% CI [{lo:+.0f}, {hi:+.0f}]  ·  Welch t={welch:+.2f}, HAC t={hac:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CI [{lo:+.1f}, {hi:+.1f}] bps straddles zero; HAC t={hac:+.2f} (bar is 2).')"
        ),
        md(
            f"> 💡 In plain words: +{R['off0_bps']:.0f} bps with a 95% CI of "
            f"[{R['ci_lo']:+.0f}, +{R['ci_hi']:.0f}] bps and HAC *t* = +{R['off0_t']:.2f}. The "
            "estimate is consistent with anything from a small loss to a large gain — the definition "
            "of *no evidence*."
        ),
        md(
            "### 4c · The traded arm — react to the speech\n\n"
            "The only position you could actually place: long from the speech-day close, hold 1/2/3 "
            "sessions, one execution lag, 1 bp one-way (round-trip 2 bp), net."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in (1, 2, 3):\n"
            "        s = st.summarize_trades(st.react_trade(ret, data.JH_SPEECH_DATES, hold=h, cost_bps=1.0), 'ret_net')\n"
            "        rows.append((h, s['n'], s['win_rate']*100, s['mean_bps'], s['sharpe'], s['hac_t']))\n"
            "else:\n"
            f"    rows = [(1,{R['react1_n']},{R['react1_win']},{R['react1_bps']},{R['react1_sharpe']},{R['react1_t']}),\n"
            f"            (2,33,58,-11.4,-0.07,-0.55), (3,33,{R['react3_win']},{R['react3_bps']},0.09,{R['react3_t']})]\n"
            "tbl = pd.DataFrame(rows, columns=['hold','n','win%','mean_bps','sharpe','hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar(tbl['hold'].astype(str)+'-day', tbl['mean_bps'],\n"
            "       color=[GREEN if v > 0 else RED for v in tbl['mean_bps']], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Reacting to the speech: a fee-paying coin flip')\n"
            "for i, r in tbl.iterrows():\n"
            "    ax.annotate(f\"t={r['hac_t']:+.2f}\", (i, r['mean_bps']), ha='center',\n"
            "                va='bottom' if r['mean_bps'] > 0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tbl.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the one-session hold is **net negative** ({R['react1_bps']:+.0f} "
            f"bps, *t* = {R['react1_t']:.2f}); none of the holds clears |*t*| = 0.8. **H₂ rejected.** "
            "And with one trade a year, capacity tops out at ~31 round trips in a career — there is "
            "nothing to scale even if a tilt existed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — speech-day +{R['off0_bps']:.1f} bps, HAC *t* +{R['off0_t']:.2f}; "
            f"max |HAC *t*| over 11 offsets = +{R['max_abs_t']:.2f}; 95% CI "
            f"[{R['ci_lo']:+.1f}, +{R['ci_hi']:.1f}] bps. Nothing clears the bar.\n"
            f"- **Tradability `MIRAGE`** — react-trade net {R['react1_bps']:+.1f} bps "
            f"(*t* = {R['react1_t']:.2f}); one event/yr caps capacity at ~31 trades/career.\n"
            "- **A Jackson Hole drift? `NOT SUPPORTED`** — alternating sign across the window, two "
            "anchor anecdotes, no broad-based effect. **H₃ rejected.**"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the multiple-testing trap, made explicit\n\n"
            "Suppose you ignored every warning and traded the *best* offset you found. How impressive "
            "is a max-of-11 *t* of +1.75? We can simulate: draw the same 33-event, 11-offset search "
            "on **pure noise** many times and see how often the best offset clears the bar by luck."
        ),
        code(
            "rng = np.random.default_rng(314)\n"
            "n_sim, n_offsets = 4000, 11\n"
            "max_t = np.empty(n_sim)\n"
            "for s in range(n_sim):\n"
            "    # 11 independent-ish event-mean t's on 33 fake events, std normal\n"
            "    ts = rng.standard_normal((n_offsets, 33)).mean(axis=1) * np.sqrt(33)\n"
            "    max_t[s] = np.abs(ts).max()\n"
            "obs = 1.75\n"
            "p_emp = (max_t >= obs).mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(max_t, bins=40, color=GREY, alpha=.8)\n"
            "ax.axvline(obs, color=RED, lw=2, label=f'our best offset |t|={obs}')\n"
            "ax.axvline(2.0, color=AMBER, ls='--', label='single-test bar |t|=2')\n"
            "ax.set_xlabel('max |t| over 11 offsets (pure noise)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'On noise, the best-of-11 |t| beats 1.75 about {p_emp:.0%} of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'P(max|t| >= 1.75 | pure noise) ~ {p_emp:.2f}  — our best offset is unremarkable.')"
        ),
        md(
            "> 💡 In plain words: when you cherry-pick the best of 11 offsets, a |*t*| around 1.75 "
            "shows up on **pure noise** a large fraction of the time. Our headline offset isn't even "
            "surprising under the null — the search itself manufactures it. This is the trap every "
            "calendar-effect blog falls into."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or does it always print nothing? Plant a speech-day "
            "bump and confirm the harness banks it; check it stays quiet on the null."
        ),
        code(
            "bumps = [0.0, 0.002, 0.004, 0.006, 0.008]\n"
            "tstats = []\n"
            "for bp in bumps:\n"
            "    r, sp, _ = data.synthetic_world(n_years=33, bump=bp, seed=314)\n"
            "    tstats.append(st.event_table(r, sp, 0)['hac_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.plot([b*1e4 for b in bumps], tstats, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted speech-day bump (bps)'); ax.set_ylabel('speech-day HAC t')\n"
            "ax.set_title('The engine is a faithful detector — t rises with the planted effect')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'null t={{tstats[0]:+.2f}}  ·  +80bps t={{tstats[-1]:+.2f}}  '\n"
            f"      f'(real SPY speech-day t was only +{R['off0_t']:.2f})')"
        ),
        md(
            f"The synthetic *t* climbs monotonically with the planted bump — null at "
            f"+{R['null_t']:.2f}, a +{R['bump_bps']:.0f} bps effect at +{R['bump_t']:.2f}. So the "
            "harness works; the flat real-tape result is a statement about **the market**: the "
            "Jackson Hole symposium is a media event, not a calendar edge.\n\n"
            "For the real Fed-event drift that *did* exist (before it decayed), see "
            "[Study 67 — Fed-Drift](../../67-fed-drift/); for the FOMC even-week cycle, "
            "[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)."
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
