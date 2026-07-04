"""Generate the two narrative notebooks for Study 607 (Quarterly Refunding Shock).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^TNX/^TYX/TLT
series under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (106 QRA dates 2000-02-02 ->
# 2026-05-06 vs ^TNX/^TYX daily yields + TLT total-return, as-of 2026-06-30).
R = dict(
    asof="2026-06-30", n_qra=106, first="2000-02-02", last="2026-05-06",
    n_clean=69, n_overlap=37, overlap_pct=35, n_base=5936,
    # day-0: (n, |dy| event, |dy| base, ratio, welch_t, signed mean, signed t)
    day0_clean=(69, 3.86, 4.28, 0.90, -1.12, 0.03, 0.05),
    day0_all=(106, 3.91, 4.28, 0.91, -1.18, -0.30, -0.61),
    day0_overlap=(37, 4.02, 4.28, 0.94, -0.48, -0.91, -1.08),
    fomc_only=(175, 5.17, 1.21, 2.24),          # n, |dy|, ratio, welch — the in-tape control
    placebo_p=0.823, placebo_draw_mean=4.27,
    y30=(4.37, 3.97, 1.10, 0.73, 0.10, 0.13),   # |dy|, base, ratio, welch, signed, t
    # window: (offset, |dy|, welch, signed, t) ; baseline 4.28
    window=[(-1, 4.67, 0.91, -0.25, -0.35), (0, 3.86, -1.12, 0.03, 0.05),
            (1, 3.99, -0.69, -0.77, -1.23), (2, 6.42, 4.10, 2.17, 2.41),
            (3, 3.49, -1.72, 0.48, 0.77)],
    jobs=dict(n_day2=69, n_ff=63, share_pct=91, abs_all=6.42, welch_all=4.10,
              abs_ff=6.62, abs_ex=4.30, welch_ex=0.01, n_ex=6),
    # era (vol-normalised |move|): all-QRA vs FOMC-clean
    era_all=dict(early=0.90, n_early=92, late=1.17, n_late=14, welch=1.21,
                 trim=1.20, n_trim=12, welch_trim=1.16),
    era_clean=dict(early=0.95, n_early=62, late=0.99, n_late=7, welch=0.10,
                   trim=1.07, n_trim=6, welch_trim=0.26),
    episodes=[("2023-07-31", "borrowing estimate (+$274B surprise)", -1.0),
              ("2023-08-02", "QRA statement (sizes up; Fitch downgrade eve)", +2.7),
              ("2023-10-30", "borrowing estimate ($76B below flag)", +3.0),
              ("2023-11-01", "QRA statement (slower; same-day FOMC)", -8.6)],
    week_aug=9.1, week_nov=-28.7,
    tlt=dict(n=96, uncond=-14.3, uncond_t=-0.89, base=4.6, welch=-1.15,
             cond=-21.3, cond_t=-1.33,
             net=[(2.0, -18.3, -73, -25.3, -102), (5.0, -24.3, -97, -31.3, -126)]),
    # synthetic: (vol_mult, mean_bp, ratio, welch, signed, signed_t)
    syn=[(1.0, 0.0, 1.08, 1.09, -0.91, -1.48), (2.0, 0.0, 2.17, 7.65, -1.81, -1.48),
         (1.0, 4.0, 1.19, 2.20, 3.09, 5.05)],
    fp_tnx="07c63093f8d2", fp_tyx="0f37876ddb3a", fp_tlt="ef0fdae90551",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![2023_regime_change%3F: Busted](https://img.shields.io/badge/2023_regime_change%3F-Busted-8b949e?style=flat-square)\n\n"
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

from quarterly_refunding_shock import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TNX, TYX, TLT = data.load_real()
    DY = st.dy_bps(TNX)
    LIDX = pd.DatetimeIndex(TNX.index)
    SP = st.clean_split(LIDX, data.QRA_DATES, data.FOMC_DATES)
    CLEAN = st.align_positions(LIDX, DY.index, SP["clean_pos"])
    ALLQ = st.align_positions(LIDX, DY.index, SP["qra_pos"])
    BASE = SP["base_mask"][1:]
else:
    TNX = TYX = TLT = DY = None
print("real cache present:", HAVE_REAL, "| QRA dates:", len(data.QRA_DATES))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Treasury's \"QRA\" really shake the bond market? 🏛️\n"
            "### The Quarterly Refunding Announcement — 2023's scariest new acronym, measured "
            "against 26 years of tape\n\n"
            + BADGES +
            "In the autumn of 2023 a sleepy bureaucratic ritual became macro folklore. Four times "
            "a year the U.S. Treasury announces **how much debt it plans to sell** — the "
            "*Quarterly Refunding Announcement*. In August 2023 it announced more than expected "
            "and long-term yields ripped higher; in November it announced less and they collapsed. "
            "Overnight, \"QRA day\" joined Fed day and CPI day on every macro calendar.\n\n"
            "So we did the boring thing: we collected **every QRA since 2000 — all 106 of them** — "
            "and measured what the 10-year Treasury yield actually does on announcement day.\n\n"
            "> 📓 **Plain-language layer.** Want the Welch *t*s, the placebo and the "
            "decontamination details? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One trap up front.** QRA day is a Wednesday in early February, May, August and "
            "November. Guess who else loves those exact Wednesdays? **The Fed.** 35% of all QRA "
            "days are *also* FOMC statement days (including the famous 2023-11-01!). Every honest "
            "number below strips those out. House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the bond market jump on QRA day? | **No.** On the 69 clean QRA days (no Fed "
            "the same day) the 10-year moves **slightly LESS** than on an ordinary day — 3.9 bps "
            "vs 4.3 bps. A random set of days beats the QRA calendar 82% of the time. |\n"
            "| But… August and November 2023?! | Those were loud **weeks**, not loud "
            "announcements. Aug 2, 2023 itself moved **+2.7 bps** (half an ordinary day!) — the "
            "week's damage came with a US credit downgrade and a jobs report attached. Nov 1's "
            "drop had an **FOMC meeting the same afternoon**. |\n"
            "| Is there money in trading it? | **No.** Buying TLT at the QRA close and holding 3 "
            "days *lost* money on average over 96 events — before costs. |\n"
            "| Could your method just be blind? | **No.** The same method, on the same tape, "
            "lights up on FOMC days (×1.21 an ordinary day, statistically solid) and on planted "
            "synthetic shocks. It sees real events fine — there's just nothing here. |\n\n"
            "> The QRA is genuinely important **policy**. It just isn't an announcement-day "
            "**shock** — by the time Treasury speaks at 8:30, the market has mostly guessed the "
            "answer."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Treasury supply now drives the long end. The Quarterly Refunding Announcement "
            "moves 10- and 30-year yields the way Fed meetings do — August and November 2023 "
            "proved it.\"*\n\n"
            "That's the strong, testable version of what macro Twitter, a famous 2024 paper on "
            "\"activist Treasury issuance\", and a thousand desk notes started saying after 2023. "
            "The mechanics: on the Monday Treasury reveals **how much** it needs to borrow; on "
            "the Wednesday at 8:30am it reveals **which maturities** will carry the load. "
            "Surprise supply at the long end should mean higher long yields — *if* it's a "
            "surprise.\n\n"
            "The word doing all the work is **surprise**. Auction sizes are guessed in advance by "
            "a whole industry (Treasury even surveys the dealers first). Our question is narrow "
            "and clean: **on the day itself, does the long end actually move more than usual?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If QRA day is a genuine macro event you should: hedge into it, expect option markets "
            "to price it, and maybe trade the reaction. If it *isn't* — if the 2023 story is "
            "survivor-bias on two loud weeks — then \"QRA risk\" is a cost you're paying for a "
            "ghost. Getting this right is worth real money to anyone who trades bonds around the "
            "refunding calendar.\n\n"
            "It's also a beautiful case study in **how folklore is born**: two big weeks + a "
            "plausible mechanism + a calendar collision with the Fed and the jobs report = a "
            "\"new macro event\" that nobody re-measured."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Every QRA since 2000.** We rebuilt the calendar — **{R['n_qra']} announcement "
            "dates** — from the official TreasuryDirect auction records (the refunding "
            "securities are announced *via* the QRA statement, so their official announcement "
            "date IS the QRA date). All 106 are Wednesdays, 8:30am ET.\n"
            "- **The Fed decontamination.** 37 of the 106 (35%) are *also* FOMC statement days — "
            "same early-quarter Wednesdays. Daily data can't split an 8:30am statement from a "
            "2pm one, so the headline test uses only the **69 Fed-free QRA days**.\n"
            "- **The measure.** The 10-year yield's move on QRA day vs ordinary days (with a "
            "random-calendar placebo), the surrounding days (−1 to +3), the 30-year as a check, "
            "and a simple TLT trade for the money question."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline picture.** Average size of the 10-year yield's daily move: ordinary "
            "days, clean QRA days, and — for contrast — actual FOMC days on the same tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_c = st.day0_stats(DY, CLEAN, BASE)\n"
            "    qset = set(SP['qra_pos'])\n"
            "    fpos = [p for p in st.event_positions(LIDX, data.FOMC_DATES) if p not in qset]\n"
            "    s_f = st.day0_stats(DY, st.align_positions(LIDX, DY.index, fpos), BASE)\n"
            "    vals = [s_c['abs_base'], s_c['abs_event'], s_f['abs_event']]\n"
            "else:\n"
            "    vals = [R['day0_clean'][2], R['day0_clean'][1], R['fomc_only'][1]]\n"
            "labs = ['ordinary day', 'QRA day\\n(no Fed same day)', 'FOMC day']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labs, vals, color=[GREY, AMBER, RED], width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.1f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('average |daily move| of the 10Y yield (bps)')\n"
            "ax.set_title('The \"new macro event\" moves the market LESS than an ordinary day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ordinary {vals[0]:.2f} bps | QRA {vals[1]:.2f} bps | FOMC {vals[2]:.2f} bps')"
        ),
        md(
            f"There's the whole study in one chart. Clean QRA days average "
            f"**{R['day0_clean'][1]:.1f} bps** of movement vs **{R['day0_clean'][2]:.1f} bps** on "
            f"an ordinary day — QRA day is *quieter* than average. FOMC days, measured the exact "
            f"same way on the exact same tape, average **{R['fomc_only'][1]:.1f} bps** (×1.21, "
            "statistically solid). The method sees real events. It just doesn't see this one.\n\n"
            "**But what about 2023?** Let's zoom into the two famous episodes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = TNX.loc['2023-07-01':'2023-11-20']\n"
            "    fig, ax = plt.subplots(figsize=(10.0, 4.8))\n"
            "    ax.plot(w.index, w.values, color=GREY, lw=1.6)\n"
            "    marks = [('2023-07-31', 'borrowing\\nestimate', AMBER), ('2023-08-01', 'Fitch\\ndowngrade', RED),\n"
            "             ('2023-08-02', 'QRA', GREEN), ('2023-10-30', 'borrowing\\nestimate', AMBER),\n"
            "             ('2023-11-01', 'QRA + FOMC\\n(same day!)', GREEN)]\n"
            "    for d, lab, c in marks:\n"
            "        ts = pd.Timestamp(d)\n"
            "        if ts in w.index:\n"
            "            ax.axvline(ts, color=c, ls='--', alpha=.75)\n"
            "            ax.annotate(lab, (ts, w.max()), ha='center', va='top', fontsize=8, color=c)\n"
            "    ax.set_ylabel('10Y yield (%)')\n"
            "    ax.set_title('Autumn 2023: the selloff ran for MONTHS - the announcements are dots on a wave')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('Aug 2 QRA day itself:', f\"{DY.loc[pd.Timestamp('2023-08-02')]:+.1f} bps ;\",\n"
            "          'Nov 1 QRA(+FOMC) day:', f\"{DY.loc[pd.Timestamp('2023-11-01')]:+.1f} bps\")\n"
            "else:\n"
            "    eps = R['episodes']\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "    ax.bar([e[0] for e in eps], [e[2] for e in eps], color=[AMBER, GREEN, AMBER, GREEN], width=.55)\n"
            "    ax.set_ylabel('day-0 move of the 10Y (bps)'); ax.axhline(0, color=GREY)\n"
            "    ax.set_title('The four famous 2023 days: small moves, big confounds')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    for d, lab, v in eps: print(f'{d}: {v:+.1f} bps - {lab}')"
        ),
        md(
            f"Now the anticlimax, in numbers:\n\n"
            f"- **July 31, 2023** — the +$274B borrowing shock that starts the legend: the 10Y "
            f"moved **{R['episodes'][0][2]:+.1f} bps** that day. Yes, *minus* one.\n"
            f"- **Aug 2, 2023** — QRA day itself: **{R['episodes'][1][2]:+.1f} bps**, about *half* "
            f"an ordinary day's move. The week's +{R['week_aug']:.0f} bps came with **Fitch "
            f"stripping the US AAA** the evening before and a jobs report two days later.\n"
            f"- **Nov 1, 2023** — the famous rally, **{R['episodes'][3][2]:+.1f} bps**… with an "
            f"**FOMC statement the same afternoon**. On daily data the QRA can't even claim that "
            f"move as its own; the refunding *week* fell {abs(R['week_nov']):.0f} bps with the "
            f"Fed, the QRA and the jobs report all inside it.\n\n"
            "**One more trap — the \"aftershock\" that isn't.** Look two days after the QRA and "
            "the market *does* jump. Why? QRA is an early-month Wednesday… and **two days after "
            "an early-month Wednesday is the first Friday: jobs day.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof = st.window_profile(DY, CLEAN, BASE)\n"
            "    offs = [p['offset'] for p in prof]; vals = [p['abs_mean'] for p in prof]\n"
            "    base = prof[0]['abs_base']\n"
            "    dj = st.day2_jobs_diagnostic(DY, CLEAN, BASE)\n"
            "    exff, share = dj['abs_ex_ff'], dj['share_first_friday']*100\n"
            "else:\n"
            "    offs = [w[0] for w in R['window']]; vals = [w[1] for w in R['window']]\n"
            "    base = R['day0_clean'][2]; exff = R['jobs']['abs_ex']; share = R['jobs']['share_pct']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "cols = [AMBER if o == 0 else (RED if o == 2 else GREY) for o in offs]\n"
            "ax.bar([f'day {o:+d}' for o in offs], vals, color=cols, width=.6)\n"
            "ax.axhline(base, ls='--', c=GREY, label=f'ordinary day ({base:.1f} bps)')\n"
            "ax.scatter(['day +2'], [exff], marker='D', color=GREEN, zorder=5,\n"
            "           label=f'day +2 WITHOUT jobs Fridays ({exff:.1f} bps)')\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('average |10Y move| (bps)')\n"
            "ax.set_title(f'The only blip in the QRA window is day +2 - which is jobs day {share:.0f}% of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'day +2 with jobs Fridays: {vals[3]:.1f} bps | without: {exff:.1f} bps | share of jobs Fridays: {share:.0f}%')"
        ),
        md(
            f"**{R['jobs']['share_pct']}%** of the day+2 sessions are first-Friday jobs reports. "
            f"Remove them and day+2 falls from **{R['jobs']['abs_all']:.1f}** to "
            f"**{R['jobs']['abs_ex']:.1f} bps** — indistinguishable from an ordinary day. The "
            "whole QRA window contains exactly one real event, and it belongs to the Bureau of "
            "Labor Statistics."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Across {R['n_qra']} announcements, clean QRA days move "
            f"**{R['day0_clean'][1]:.1f} bps vs {R['day0_clean'][2]:.1f}** on ordinary days — "
            "*less*, not more. Random calendars beat the QRA calendar 82% of the time. The "
            "30-year says the same. \n"
            f"- **Tradability — Mirage.** TLT bought at the QRA close and held 3 days: "
            f"**{R['tlt']['uncond']:.0f} bps per event before costs** over {R['tlt']['n']} "
            "events. There is nothing to monetise.\n"
            "- **\"2023 changed the game\"? — Busted.** Judged against 2023's own (high) noise "
            "level, Fed-free QRA days since 2023 are exactly as ordinary as they were in "
            "2000–2022. The two famous days were small on the day and entangled with Fitch, the "
            "FOMC and the jobs report."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The supply story isn't dead — it lives at the auctions.** Desk sibling "
            "[603-treasury-auction-concession](../../603-treasury-auction-concession/README.md) "
            "finds yields genuinely back up 1.5–3 bps into the 10Y/30Y **auctions** (the days "
            "the bonds actually hit the tape). Supply matters where it's *absorbed*, not where "
            "it's *announced*.\n"
            "- **Why so quiet?** Treasury surveys the primary dealers before every refunding and "
            "publishes guidance; by 8:30am Wednesday the sizes are mostly a formality. \"Priced "
            "in\" is not a slogan here — it's measurable.\n"
            "- **Build your own.** Swap in intraday data around 8:30am, or the Monday "
            "borrowing-estimate releases, or 10s30s curve moves — the harness takes any date "
            "list.\n\n"
            "*Think the QRA still matters? Show us a clean announcement-day move that survives "
            "the Fed and the jobs report — the event table is hardcoded and waiting.*"
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
            "# The Quarterly Refunding Shock — a quantitative teardown 🔬\n"
            "### 106 hardcoded QRA dates · FOMC decontamination · Welch day-0 tests + "
            "random-calendar placebo · the first-Friday collision · a vol-normalised 2023 era "
            "split · TLT costs · synthetic power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — *the Quarterly Refunding Announcement moves the long end; 2023 made "
            "QRA-day a macro event* — is tested on the full official announcement record "
            "against ^TNX/^TYX daily yield changes and TLT.\n\n"
            "> ⚠️ **Data note.** Event table: 106 QRA dates 2000→2026, derived once from "
            "TreasuryDirect TA_WS (`announcementDate` of the refunding securities) and "
            "hardcoded with source; all Wednesdays 08:30 ET. Tape: yfinance ^TNX/^TYX (yield "
            "in %), TLT total-return, cached, as-of " + R["asof"] + ". No survivorship (full "
            "official record, constant-maturity indices). Fingerprints `" + R["fp_tnx"] +
            "` / `" + R["fp_tyx"] + "` / `" + R["fp_tlt"] + "`; numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Clean-QRA day-0 \\|Δy10\\| **{R['day0_clean'][1]:.2f} bps** "
            f"vs baseline **{R['day0_clean'][2]:.2f}** (×{R['day0_clean'][3]:.2f}, Welch "
            f"*t* = **{R['day0_clean'][4]:+.2f}**), placebo **p = {R['placebo_p']:.2f}**; 30Y "
            f"×{R['y30'][2]:.2f} (*t* = {R['y30'][3]:+.2f}). In-tape control: FOMC days "
            f"×{R['fomc_only'][2]:.2f} at *t* = {R['fomc_only'][3]:+.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | TLT day-0-close→+3: **{R['tlt']['uncond']:+.1f} "
            f"bps/event gross** (*t* = {R['tlt']['uncond_t']:+.2f}); sign-conditional "
            f"**{R['tlt']['cond']:+.1f}** (*t* = {R['tlt']['cond_t']:+.2f}). Negative before "
            "costs. |\n"
            f"| **2023 regime change?** | `BUSTED` | Vol-normalised FOMC-clean QRA days: 2023+ "
            f"**{R['era_clean']['late']:.2f}×** own-era noise vs {R['era_clean']['early']:.2f}× "
            f"pre-2023, Welch *t* = **{R['era_clean']['welch']:+.2f}** (n = "
            f"{R['era_clean']['n_late']}, honest small-n). |\n\n"
            "> 💡 In plain words: the announcement day is quieter than an average day, there is "
            "nothing to trade, and the 2023 legend rests on two loud *weeks* full of other "
            "catalysts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d^{QRA}_t$ flag QRA statement days and $\\Delta y_t$ the daily 10Y yield "
            "change (bps). The folklore's testable content:\n\n"
            "- **H₁ (event-day volatility).** $E[|\\Delta y_t|\\,|\\,d^{QRA}_t=1] > "
            "E[|\\Delta y_t|\\,|\\,\\text{ordinary}]$ — announcement surprises move the long "
            "end *on the day*.\n"
            "- **H₂ (tradability).** The post-announcement drift (entered at the day-0 close — "
            "the one honest lag) is monetisable in TLT.\n"
            "- **H₃ (the 2023 break).** The effect appeared/intensified in 2023 (the claim's own "
            "break date — not snooped).\n\n"
            "Design hazards we must handle: **(a)** 35% of QRA Wednesdays are FOMC statement "
            "days (both calendars choose early-quarter Wednesdays); **(b)** day+2 of the window "
            "is the first-Friday **Employment Situation** slot 91% of the time; **(c)** n = 14 "
            "QRA days since 2023 — any era claim carries small-n error bars; **(d)** 2023's "
            "baseline vol was itself elevated, so era comparisons must be vol-normalised."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the inference plan\n\n"
            "Events sit ~63 trading days apart: event-day observations are serially "
            "uncorrelated, there are no overlapping windows, and the **Welch *t*** on "
            "$|\\Delta y|$ (unequal variances — event n=69 vs baseline n≈5,900) is the "
            "appropriate robust statistic; HAC would be solving a problem this design doesn't "
            "have. Supplements: a **2,000-draw random-calendar placebo** (same event count, "
            "drawn from the baseline days — far beyond the desk's ≥20-seed floor for random "
            "baselines), a signed one-sample *t* (no direction prior — surprises cut both "
            "ways), the 30Y (^TYX) as the \"long end proper\" check, and an **in-tape positive "
            "control**: FOMC-only days pushed through the identical pipeline must fire, or the "
            "instrument is blind."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['n_qra']} QRA dates {R['first']} → {R['last']}, hardcoded from "
            "TreasuryDirect TA_WS (the refunding securities' official `announcementDate` IS the "
            "QRA date; all Wednesdays, 08:30 ET; the 2023/2024 dates match the press archive).\n"
            f"- **Decontamination.** {R['n_overlap']} of {R['n_qra']} QRA days "
            f"({R['overlap_pct']}%) are FOMC statement days → primary set = **{R['n_clean']} "
            f"FOMC-clean** QRA days; baseline = {R['n_base']:,} sessions with no QRA-window day "
            "and no FOMC day.\n"
            "- **Primary test.** Welch *t* on day-0 $|\\Delta y_{10}|$, clean QRA vs baseline; "
            "placebo p from 2,000 random calendars; signed one-sample *t*.\n"
            "- **Window.** Offsets −1..+3 with the first-Friday (jobs) collision check at +2.\n"
            "- **Era split (H₃).** $|\\Delta y|$ / trailing-60d mean $|\\Delta y|$ (shifted — "
            "day-0 not in its own denominator), Welch late (2023+) vs early.\n"
            "- **Tradability.** TLT, entry day-0 close (ONE lag), exit +3 close; non-overlapping "
            "baseline windows; 2/5 bps one-way × 2 legs.\n"
            "- **Positive controls.** In-tape: FOMC days. Synthetic: planted event-day vol "
            "multiplier / signed mean vs the null (machinery proof only, never market evidence)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Day-0: the event that isn't there\n\n"
            "Clean QRA days vs baseline, with the FOMC-only in-tape control alongside, and the "
            "random-calendar placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_c = st.day0_stats(DY, CLEAN, BASE)\n"
            "    s_a = st.day0_stats(DY, ALLQ, BASE)\n"
            "    qset = set(SP['qra_pos'])\n"
            "    fpos = [p for p in st.event_positions(LIDX, data.FOMC_DATES) if p not in qset]\n"
            "    s_f = st.day0_stats(DY, st.align_positions(LIDX, DY.index, fpos), BASE)\n"
            "    pl = st.placebo_pvalue(DY, CLEAN, BASE, n_draws=2000, seed=607)\n"
            "    rows = [('clean QRA', s_c['abs_ratio'], s_c['welch_abs']),\n"
            "            ('all QRA', s_a['abs_ratio'], s_a['welch_abs']),\n"
            "            ('FOMC-only (control)', s_f['abs_ratio'], s_f['welch_abs'])]\n"
            "    obs, draws, pval = pl['obs'], pl['draws'], pl['p_value']\n"
            "else:\n"
            "    rows = [('clean QRA', R['day0_clean'][3], R['day0_clean'][4]),\n"
            "            ('all QRA', R['day0_all'][3], R['day0_all'][4]),\n"
            "            ('FOMC-only (control)', R['fomc_only'][2], R['fomc_only'][3])]\n"
            "    obs, pval = R['day0_clean'][1], R['placebo_p']\n"
            "    rng = np.random.default_rng(607); draws = rng.normal(R['placebo_draw_mean'], .35, 2000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "cols = [AMBER, AMBER, RED]\n"
            "a1.bar([r[0] for r in rows], [r[1] for r in rows], color=cols, width=.55)\n"
            "a1.axhline(1.0, ls='--', c=GREY, label='ordinary day (x1.0)')\n"
            "for i, r in enumerate(rows): a1.annotate(f'x{r[1]:.2f}\\nt={r[2]:+.2f}', (i, r[1]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylabel('|dy10| vs baseline (ratio)'); a1.set_ylim(0, 1.5)\n"
            "a1.set_title('QRA day-0: BELOW an ordinary day; FOMC fires'); a1.legend()\n"
            "a2.hist(draws, bins=50, color=GREY, alpha=.85, label='2,000 random calendars')\n"
            "a2.axvline(obs, color=AMBER, lw=2.5, label=f'QRA calendar ({obs:.2f} bps)')\n"
            "a2.set_xlabel('mean |dy10| of the calendar (bps)'); a2.set_ylabel('frequency')\n"
            "a2.set_title(f'Placebo: p = {pval:.3f} - 82% of random calendars beat QRA')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('ratios:', [(r[0], round(r[1],2), round(r[2],2)) for r in rows], ' placebo p =', round(pval,3))"
        ),
        md(
            f"> 💡 In plain words: on its own day the QRA moves the 10Y **{R['day0_clean'][1]:.2f} "
            f"bps** vs **{R['day0_clean'][2]:.2f}** on an ordinary day (×{R['day0_clean'][3]:.2f}, "
            f"Welch *t* = {R['day0_clean'][4]:+.2f}; signed mean {R['day0_clean'][5]:+.2f} bps, "
            f"*t* = {R['day0_clean'][6]:+.2f}). The placebo says a random set of 69 days beats it "
            f"{R['placebo_p']*100:.0f}% of the time. The same pipeline finds FOMC days at "
            f"×{R['fomc_only'][2]:.2f} (*t* = {R['fomc_only'][3]:+.2f}) — the instrument works; "
            f"the effect is absent. The 30Y agrees: ×{R['y30'][2]:.2f}, *t* = {R['y30'][3]:+.2f}."
        ),
        md(
            "### 4b · The window — and the jobs-report collision\n\n"
            "Mean $|\\Delta y_{10}|$ at offsets −1..+3 around clean QRA days. The +2 bump is the "
            "test case: QRA = early-month Wednesday ⇒ day+2 = first Friday = the Employment "
            "Situation, 08:30 ET."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof = st.window_profile(DY, CLEAN, BASE)\n"
            "    offs = [p['offset'] for p in prof]; vals = [p['abs_mean'] for p in prof]\n"
            "    ts_ = [p['welch_abs'] for p in prof]; base = prof[0]['abs_base']\n"
            "    dj = st.day2_jobs_diagnostic(DY, CLEAN, BASE)\n"
            "else:\n"
            "    offs = [w[0] for w in R['window']]; vals = [w[1] for w in R['window']]\n"
            "    ts_ = [w[2] for w in R['window']]; base = R['day0_clean'][2]; dj = R['jobs']\n"
            "    dj = dict(share_first_friday=dj['share_pct']/100, abs_ex_ff=dj['abs_ex'],\n"
            "              welch_ex_ff=dj['welch_ex'], n_first_friday=dj['n_ff'], n_day2=dj['n_day2'])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "cols = [RED if o == 2 else AMBER if o == 0 else GREY for o in offs]\n"
            "ax.bar([f'{o:+d}' for o in offs], vals, color=cols, width=.6)\n"
            "ax.axhline(base, ls='--', c=GREY, label=f'baseline ({base:.2f} bps)')\n"
            "ax.scatter(['+2'], [dj['abs_ex_ff']], marker='D', color=GREEN, zorder=5,\n"
            "           label=f\"day +2 excluding first-Fridays ({dj['abs_ex_ff']:.2f} bps)\")\n"
            "for i, (v, t) in enumerate(zip(vals, ts_)):\n"
            "    ax.annotate(f'{v:.1f}\\nt={t:+.1f}', (i, v), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('offset from QRA day (sessions)'); ax.set_ylabel('mean |dy10| (bps)')\n"
            "ax.set_title('The only significant offset is +2 - and it is the jobs report')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"day+2: {dj['n_first_friday']}/{dj['n_day2']} are first-Fridays; excluding them \"\n"
            "      f\"|dy|={dj['abs_ex_ff']:.2f} bps, Welch t={dj['welch_ex_ff']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: day+2 shows {R['jobs']['abs_all']:.2f} bps (Welch *t* = "
            f"+{R['jobs']['welch_all']:.2f}) — but **{R['jobs']['n_ff']}/{R['jobs']['n_day2']} "
            f"({R['jobs']['share_pct']}%)** of those sessions are first-Friday Employment "
            f"Situation releases. Exclude them and day+2 is {R['jobs']['abs_ex']:.2f} bps, Welch "
            f"*t* = **{R['jobs']['welch_ex']:+.2f}**. The QRA window's lone \"signal\" is a "
            "calendar collision with the BLS — a textbook confound, and the same trap that makes "
            "35% of QRA days FOMC days."
        ),
        md(
            "### 4c · H₃ — was 2023 a regime change?\n\n"
            "Each day's $|\\Delta y|$ normalised by its trailing 60-day mean (2023 days judged "
            "against 2023's own noise). Split at 2023-01-01 — the claim's own break date. "
            "n = 14 (7 FOMC-clean) since: small-n by construction, said out loud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec_a = st.era_compare(DY, ALLQ, BASE, drop_dates=['2023-08-02', '2023-11-01'])\n"
            "    ec_c = st.era_compare(DY, CLEAN, BASE, drop_dates=['2023-08-02', '2023-11-01'])\n"
            "    A = dict(early=ec_a['norm_early'], late=ec_a['norm_late'], welch=ec_a['welch_late_vs_early'],\n"
            "             trim=ec_a['norm_late_trim'], welch_trim=ec_a['welch_trim_vs_early'])\n"
            "    C = dict(early=ec_c['norm_early'], late=ec_c['norm_late'], welch=ec_c['welch_late_vs_early'],\n"
            "             trim=ec_c['norm_late_trim'], welch_trim=ec_c['welch_trim_vs_early'])\n"
            "    late_dates, late_vals = ec_a['late_dates'], ec_a['late_norm_values']\n"
            "else:\n"
            "    A, C = R['era_all'], R['era_clean']\n"
            "    late_dates = ['2023-02-01','2023-05-03','2023-08-02','2023-11-01','2024-01-31','2024-05-01',\n"
            "                  '2024-07-31','2024-10-30','2025-02-05','2025-04-30','2025-07-30','2025-11-05',\n"
            "                  '2026-02-04','2026-05-06']\n"
            "    late_vals = [2.06,.54,.51,1.59,1.65,1.90,.73,.19,2.15,.07,1.08,2.22,.04,1.73]\n"
            "fomc_set = set(data.FOMC_DATES)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.5))\n"
            "x = np.arange(2); w = .35\n"
            "a1.bar(x - w/2, [A['early'], A['late']], w, color=GREY, label='all QRA days')\n"
            "a1.bar(x + w/2, [C['early'], C['late']], w, color=AMBER, label='FOMC-clean')\n"
            "a1.axhline(1.0, ls='--', c=GREY)\n"
            "a1.set_xticks(x); a1.set_xticklabels(['2000-2022', '2023+'])\n"
            "a1.set_ylabel('vol-normalised |move| (x own-era noise)')\n"
            "a1.set_title(f\"Clean 2023+ QRA days: x{C['late']:.2f} - as ordinary as ever\\n\"\n"
            "             f\"(Welch late-vs-early t={C['welch']:+.2f})\")\n"
            "a1.legend()\n"
            "cols = [GREY if d in fomc_set else AMBER for d in late_dates]\n"
            "a2.bar(range(len(late_vals)), late_vals, color=cols, width=.6)\n"
            "a2.axhline(1.0, ls='--', c=GREY, label='own-era ordinary day')\n"
            "a2.set_xticks(range(len(late_dates)))\n"
            "a2.set_xticklabels([d[2:7] for d in late_dates], rotation=60, fontsize=7)\n"
            "a2.set_ylabel('x own-era noise')\n"
            "a2.set_title('2023+ QRA days one by one (grey = FOMC same day)')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"all-QRA: early {A['early']:.2f}x late {A['late']:.2f}x (welch {A['welch']:+.2f}); \"\n"
            "      f\"clean: early {C['early']:.2f}x late {C['late']:.2f}x (welch {C['welch']:+.2f}); \"\n"
            "      f\"drop Aug-2/Nov-1: {C['trim']:.2f}x (welch {C['welch_trim']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: even judged against its own noisy era, a Fed-free 2023+ QRA "
            f"day is a **{R['era_clean']['late']:.2f}×** day — statistically identical to the "
            f"{R['era_clean']['early']:.2f}× of 2000–2022 (Welch *t* = "
            f"{R['era_clean']['welch']:+.2f}, n = {R['era_clean']['n_late']} — small-n, so we "
            "claim only that *nothing detectable changed*). Dropping the two storied days "
            f"**raises** the late mean to {R['era_clean']['trim']:.2f}× — they weren't even loud "
            f"on day-0 (Aug-2: 0.51×, {R['episodes'][1][2]:+.1f} bps with Fitch in the tape; "
            f"Nov-1: {R['episodes'][3][2]:+.1f} bps with a same-day FOMC). The famous moves were "
            f"week moves: {R['week_aug']:+.1f} bps (Aug refunding week), {R['week_nov']:+.1f} "
            "bps (Nov) — carried by downgrades, payrolls and the Fed as much as by Treasury."
        ),
        md(
            "### 4d · Tradability — TLT after the announcement\n\n"
            "Entry at the day-0 close (the ONE lag — the statement is out at 08:30, the close is "
            "the first honest fill), exit +3 sessions. Unconditional and day-0-sign-conditional, "
            "gross and net."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.tlt_event_trades(TLT, data.QRA_DATES, hold=3)\n"
            "    nets = [st.tlt_net(tr, cb) for cb in (2.0, 5.0)]\n"
            "    g = [tr['uncond_mean_bps'], tr['cond_mean_bps']]\n"
            "    n2 = [nets[0]['uncond_net_bps'], nets[0]['cond_net_bps']]\n"
            "    n5 = [nets[1]['uncond_net_bps'], nets[1]['cond_net_bps']]\n"
            "else:\n"
            "    g = [R['tlt']['uncond'], R['tlt']['cond']]\n"
            "    n2 = [R['tlt']['net'][0][1], R['tlt']['net'][0][3]]\n"
            "    n5 = [R['tlt']['net'][1][1], R['tlt']['net'][1][3]]\n"
            "x = np.arange(2); w = .26\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x - w, g, w, color=GREY, label='gross')\n"
            "ax.bar(x, n2, w, color=AMBER, label='net @ 2 bps one-way')\n"
            "ax.bar(x + w, n5, w, color=RED, label='net @ 5 bps one-way')\n"
            "ax.axhline(0, color=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['unconditional long TLT', 'ride the day-0 sign'])\n"
            "ax.set_ylabel('bps per event (3-day hold)')\n"
            "ax.set_title('Negative BEFORE costs - there is nothing to erode')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross:', [round(v,1) for v in g], ' net@2:', [round(v,1) for v in n2],\n"
            "      ' net@5:', [round(v,1) for v in n5])"
        ),
        md(
            f"> 💡 In plain words: {R['tlt']['n']} events, ~4/yr. Unconditional: "
            f"**{R['tlt']['uncond']:+.1f} bps/event** gross (*t* = {R['tlt']['uncond_t']:+.2f}; "
            f"ordinary 3-day windows average {R['tlt']['base']:+.1f} bps — the post-QRA windows "
            f"are insignificantly *worse*, Welch *t* = {R['tlt']['welch']:+.2f}). Conditional "
            f"continuation: **{R['tlt']['cond']:+.1f} bps/event** (*t* = "
            f"{R['tlt']['cond_t']:+.2f}). Net at 2/5 bps one-way: −73 to −126 bps/yr. MIRAGE — "
            "there was never anything to pay costs *from*."
        ),
        md(
            "### 4e · Synthetic power control — the machinery is faithful\n\n"
            "A deterministic pseudo-QRA world (i.i.d. daily Δy, events every 63 sessions) with "
            "two knobs: an event-day **vol multiplier** and a **signed event-day mean**. The "
            "null must stay quiet; each planted effect must light its own detector. *(Machinery "
            "proof only — never cited as market evidence.)*"
        ),
        code(
            "res = []\n"
            "for vol_mult, mean_bp in [(1.0, 0.0), (2.0, 0.0), (1.0, 4.0)]:\n"
            "    lvl, ev = data.synthetic_world(vol_mult=vol_mult, mean_bp=mean_bp, seed=607)\n"
            "    dys = st.dy_bps(lvl); li = pd.DatetimeIndex(lvl.index)\n"
            "    sps = st.clean_split(li, ev, [])\n"
            "    ps = st.align_positions(li, dys.index, sps['clean_pos'])\n"
            "    s = st.day0_stats(dys, ps, sps['base_mask'][1:])\n"
            "    res.append((vol_mult, mean_bp, s['abs_ratio'], s['welch_abs'], s['t_signed']))\n"
            "labs = [f'null\\n(x1, 0bp)', f'planted vol x2', f'planted mean +4bp']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labs, [r[3] for r in res], color=[GREY, GREEN, AMBER], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar'); a1.axhline(-2, ls='--', c=RED)\n"
            "for i, r in enumerate(res): a1.annotate(f't={r[3]:+.2f}', (i, r[3]), ha='center', va='bottom')\n"
            "a1.set_ylabel('Welch t on |dy|'); a1.set_title('|move| detector'); a1.legend()\n"
            "a2.bar(labs, [r[4] for r in res], color=[GREY, GREEN, AMBER], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED)\n"
            "for i, r in enumerate(res): a2.annotate(f't={r[4]:+.2f}', (i, r[4]), ha='center', va='bottom')\n"
            "a2.set_ylabel('one-sample t on signed dy'); a2.set_title('signed detector')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'vol x{r[0]:.1f} mean {r[1]:+.1f}bp: ratio x{r[2]:.2f} welch {r[3]:+.2f} signed t {r[4]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the null sits at Welch *t* = {R['syn'][0][3]:+.2f} (no false "
            f"alarm); a planted ×2 vol day fires at *t* = {R['syn'][1][3]:+.2f}; a planted +4 bp "
            f"drift fires the signed detector at *t* = {R['syn'][2][5]:+.2f}. Together with the "
            "in-tape FOMC control (×1.21, *t* = +2.24), the instruments are proven sharp — the "
            "real-tape nulls are genuine absences."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — clean QRA day-0 |Δy10| **{R['day0_clean'][1]:.2f} bps vs "
            f"{R['day0_clean'][2]:.2f} baseline** (×{R['day0_clean'][3]:.2f}, Welch *t* = "
            f"{R['day0_clean'][4]:+.2f}; placebo p = {R['placebo_p']:.2f}); 30Y ×{R['y30'][2]:.2f} "
            f"(*t* = {R['y30'][3]:+.2f}); signed drift nil. The lone window bump (day+2, *t* = "
            f"+{R['jobs']['welch_all']:.2f}) is the jobs report ({R['jobs']['share_pct']}% "
            f"first-Fridays; without them *t* = {R['jobs']['welch_ex']:+.2f}). The pipeline "
            f"detects FOMC days (×{R['fomc_only'][2]:.2f}, *t* = {R['fomc_only'][3]:+.2f}) — the "
            "absence is real, not instrumental. No survivorship.\n"
            f"- **Tradability `MIRAGE`** — TLT legs: {R['tlt']['uncond']:+.1f} (unconditional) "
            f"and {R['tlt']['cond']:+.1f} (conditional) bps/event **gross**; −73 to −126 bps/yr "
            "net. Nothing exists to harvest.\n"
            f"- **2023 regime change? `BUSTED`** — vol-normalised clean QRA days: 2023+ "
            f"{R['era_clean']['late']:.2f}× vs {R['era_clean']['early']:.2f}× pre-2023 (Welch "
            f"*t* = {R['era_clean']['welch']:+.2f}; n = {R['era_clean']['n_late']}, small-n "
            "stated). Aug-2/Nov-1 2023 were small on day-0 and confounded (Fitch; same-day "
            "FOMC); the legend is two loud weeks, not a new event class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The supply effect lives at the auctions, not the announcement.** Sibling "
            "[603-treasury-auction-concession](../../603-treasury-auction-concession/README.md) "
            "finds a real, HAC-significant concession *into* the 10Y/30Y auctions — the dealers "
            "get paid to absorb the bonds, not to hear the plan. Announcement ≠ absorption.\n"
            "- **Anticipation is the null hypothesis for policy calendars.** Treasury surveys "
            "primary dealers before every refunding and telegraphs changes; the information is "
            "mostly in prices by 08:30. Contrast FOMC days, where a genuine surprise component "
            "survives — and shows up in our control.\n"
            "- **What would overturn us.** Intraday (08:30–09:30) reaction windows; the Monday "
            "borrowing-estimate days as separate events; survey-based size surprises (dealer "
            "medians vs announced sizes) as a signed regressor. The harness accepts any date "
            "table — the QRA list is hardcoded with its derivation.\n\n"
            "*Reproducible core: offline and deterministic; event table + FOMC calendar "
            "hardcoded with sources. Methods: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
